# src/trendops/api/routes/pipeline.py
"""
TrendOps Pipeline API 라우터

Week 6 Day 4: 파이프라인 실행 및 모니터링 API
"""
from datetime import datetime, timedelta
from typing import Literal
from uuid import UUID

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field
from sqlalchemy import Integer, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from trendops.database.connection import get_session
from trendops.database.models import DailyReport, PipelineMetric

router = APIRouter()


# =============================================================================
# Request/Response Schemas
# =============================================================================


class PipelineMetricCreate(BaseModel):
    """파이프라인 메트릭 생성 요청"""

    stage: str = Field(..., description="파이프라인 단계")
    keyword: str | None = Field(default=None, description="키워드")
    status: Literal["success", "failure", "partial"] = Field(..., description="상태")
    duration_ms: int | None = Field(default=None, ge=0, description="처리 시간(ms)")
    error_message: str | None = Field(default=None, description="에러 메시지")
    items_processed: int = Field(default=0, ge=0, description="처리된 항목 수")
    items_failed: int = Field(default=0, ge=0, description="실패한 항목 수")
    metadata: dict | None = Field(default=None, description="추가 메타데이터")


class PipelineMetricResponse(BaseModel):
    """파이프라인 메트릭 응답"""

    id: UUID
    stage: str
    keyword: str | None
    status: str
    duration_ms: int | None
    error_message: str | None
    items_processed: int
    items_failed: int
    recorded_at: datetime | None

    class Config:
        from_attributes = True


class PipelineMetricListResponse(BaseModel):
    """파이프라인 메트릭 목록 응답"""

    items: list[PipelineMetricResponse]
    total: int
    page: int
    page_size: int


class PipelineStageStats(BaseModel):
    """단계별 통계"""

    stage: str
    total_runs: int
    success_count: int
    failure_count: int
    success_rate: float
    avg_duration_ms: float | None
    total_items_processed: int


class PipelineSummary(BaseModel):
    """파이프라인 전체 요약"""

    total_runs: int
    success_rate: float
    avg_latency_ms: float | None
    stage_stats: list[PipelineStageStats]
    recent_errors: list[dict]


class DailyReportResponse(BaseModel):
    """일별 리포트 응답"""

    id: UUID
    report_date: datetime
    trends_detected: int
    articles_collected: int
    articles_analyzed: int
    images_generated: int
    posts_published: int
    posts_rejected: int
    errors_count: int
    success_rate: float | None
    avg_latency_ms: int | None
    created_at: datetime | None

    class Config:
        from_attributes = True


class TriggerPipelineRequest(BaseModel):
    """파이프라인 트리거 요청"""

    keywords: list[str] = Field(..., min_length=1, max_length=10, description="처리할 키워드")
    priority: int = Field(default=5, ge=0, le=10, description="우선순위")
    dry_run: bool = Field(default=False, description="테스트 실행 여부")


class TriggerPipelineResponse(BaseModel):
    """파이프라인 트리거 응답"""

    job_ids: list[str]
    message: str
    queued_count: int


# =============================================================================
# Helper Functions
# =============================================================================


def metric_to_response(metric: PipelineMetric) -> PipelineMetricResponse:
    """PipelineMetric 모델을 응답 스키마로 변환"""
    return PipelineMetricResponse(
        id=metric.id,
        stage=metric.stage,
        keyword=metric.keyword,
        status=metric.status,
        duration_ms=metric.duration_ms,
        error_message=metric.error_message,
        items_processed=metric.items_processed,
        items_failed=metric.items_failed,
        recorded_at=metric.recorded_at,
    )


# =============================================================================
# Endpoints
# =============================================================================


@router.get(
    "/metrics",
    response_model=PipelineMetricListResponse,
    summary="파이프라인 메트릭 조회",
    description="파이프라인 실행 메트릭을 조회합니다.",
)
async def list_pipeline_metrics(
    session: AsyncSession = Depends(get_session),
    stage: str | None = Query(None, description="단계 필터"),
    status: str | None = Query(None, description="상태 필터"),
    keyword: str | None = Query(None, description="키워드 필터"),
    start_date: datetime | None = Query(None, description="시작 날짜"),
    end_date: datetime | None = Query(None, description="종료 날짜"),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
) -> PipelineMetricListResponse:
    """파이프라인 메트릭 목록 조회"""
    query = select(PipelineMetric)
    count_query = select(func.count(PipelineMetric.id))

    if stage:
        query = query.where(PipelineMetric.stage == stage)
        count_query = count_query.where(PipelineMetric.stage == stage)

    if status:
        query = query.where(PipelineMetric.status == status)
        count_query = count_query.where(PipelineMetric.status == status)

    if keyword:
        query = query.where(PipelineMetric.keyword.ilike(f"%{keyword}%"))
        count_query = count_query.where(PipelineMetric.keyword.ilike(f"%{keyword}%"))

    if start_date:
        query = query.where(PipelineMetric.recorded_at >= start_date)
        count_query = count_query.where(PipelineMetric.recorded_at >= start_date)

    if end_date:
        query = query.where(PipelineMetric.recorded_at <= end_date)
        count_query = count_query.where(PipelineMetric.recorded_at <= end_date)

    total_result = await session.execute(count_query)
    total = total_result.scalar() or 0

    offset = (page - 1) * page_size
    query = query.order_by(PipelineMetric.recorded_at.desc()).offset(offset).limit(page_size)

    result = await session.execute(query)
    metrics = result.scalars().all()

    return PipelineMetricListResponse(
        items=[metric_to_response(m) for m in metrics],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.post(
    "/metrics",
    response_model=PipelineMetricResponse,
    status_code=201,
    summary="파이프라인 메트릭 기록",
    description="새로운 파이프라인 메트릭을 기록합니다.",
)
async def create_pipeline_metric(
    data: PipelineMetricCreate,
    session: AsyncSession = Depends(get_session),
) -> PipelineMetricResponse:
    """파이프라인 메트릭 기록"""
    metric = PipelineMetric(
        stage=data.stage,
        keyword=data.keyword,
        status=data.status,
        duration_ms=data.duration_ms,
        error_message=data.error_message,
        items_processed=data.items_processed,
        items_failed=data.items_failed,
        metadata_=data.metadata or {},
    )
    session.add(metric)
    await session.flush()
    await session.refresh(metric)

    return metric_to_response(metric)


@router.get(
    "/summary",
    response_model=PipelineSummary,
    summary="파이프라인 요약",
    description="파이프라인 실행 요약 정보를 조회합니다.",
)
async def get_pipeline_summary(
    session: AsyncSession = Depends(get_session),
    hours: int = Query(24, ge=1, le=168, description="조회 기간 (시간)"),
) -> PipelineSummary:
    """파이프라인 요약 조회"""
    since = datetime.now() - timedelta(hours=hours)

    # 전체 실행 수
    total_result = await session.execute(
        select(func.count(PipelineMetric.id)).where(PipelineMetric.recorded_at >= since)
    )
    total_runs = total_result.scalar() or 0

    # 성공률
    success_result = await session.execute(
        select(func.count(PipelineMetric.id))
        .where(PipelineMetric.recorded_at >= since)
        .where(PipelineMetric.status == "success")
    )
    success_count = success_result.scalar() or 0
    success_rate = (success_count / total_runs * 100) if total_runs > 0 else 0.0

    # 평균 지연시간
    avg_latency_result = await session.execute(
        select(func.avg(PipelineMetric.duration_ms))
        .where(PipelineMetric.recorded_at >= since)
        .where(PipelineMetric.duration_ms.isnot(None))
    )
    avg_latency = avg_latency_result.scalar()

    # 단계별 통계
    stage_result = await session.execute(
        select(
            PipelineMetric.stage,
            func.count(PipelineMetric.id),
            func.sum(func.cast(PipelineMetric.status == "success", Integer)),
            func.sum(func.cast(PipelineMetric.status == "failure", Integer)),
            func.avg(PipelineMetric.duration_ms),
            func.sum(PipelineMetric.items_processed),
        )
        .where(PipelineMetric.recorded_at >= since)
        .group_by(PipelineMetric.stage)
    )

    stage_stats = []
    for row in stage_result.all():
        stage, total, success, failure, avg_dur, items = row
        stage_stats.append(
            PipelineStageStats(
                stage=stage,
                total_runs=total or 0,
                success_count=int(success or 0),
                failure_count=int(failure or 0),
                success_rate=(int(success or 0) / total * 100) if total else 0.0,
                avg_duration_ms=round(float(avg_dur), 2) if avg_dur else None,
                total_items_processed=int(items or 0),
            )
        )

    # 최근 에러
    error_result = await session.execute(
        select(PipelineMetric)
        .where(PipelineMetric.recorded_at >= since)
        .where(PipelineMetric.status == "failure")
        .order_by(PipelineMetric.recorded_at.desc())
        .limit(10)
    )
    errors = error_result.scalars().all()
    recent_errors = [
        {
            "stage": e.stage,
            "keyword": e.keyword,
            "error": e.error_message,
            "recorded_at": e.recorded_at.isoformat() if e.recorded_at else None,
        }
        for e in errors
    ]

    return PipelineSummary(
        total_runs=total_runs,
        success_rate=round(success_rate, 2),
        avg_latency_ms=round(float(avg_latency), 2) if avg_latency else None,
        stage_stats=stage_stats,
        recent_errors=recent_errors,
    )


@router.get(
    "/reports/daily",
    response_model=list[DailyReportResponse],
    summary="일별 리포트 조회",
    description="일별 운영 리포트를 조회합니다.",
)
async def list_daily_reports(
    session: AsyncSession = Depends(get_session),
    days: int = Query(7, ge=1, le=30, description="조회 일수"),
) -> list[DailyReportResponse]:
    """일별 리포트 목록"""
    result = await session.execute(
        select(DailyReport).order_by(DailyReport.report_date.desc()).limit(days)
    )
    reports = result.scalars().all()

    return [
        DailyReportResponse(
            id=r.id,
            report_date=r.report_date,
            trends_detected=r.trends_detected,
            articles_collected=r.articles_collected,
            articles_analyzed=r.articles_analyzed,
            images_generated=r.images_generated,
            posts_published=r.posts_published,
            posts_rejected=r.posts_rejected,
            errors_count=r.errors_count,
            success_rate=r.success_rate,
            avg_latency_ms=r.avg_latency_ms,
            created_at=r.created_at,
        )
        for r in reports
    ]


@router.post(
    "/trigger",
    response_model=TriggerPipelineResponse,
    summary="파이프라인 트리거",
    description="새로운 파이프라인 실행을 트리거합니다.",
)
async def trigger_pipeline(
    request: TriggerPipelineRequest,
    session: AsyncSession = Depends(get_session),
) -> TriggerPipelineResponse:
    """파이프라인 수동 트리거"""
    from uuid import uuid4

    job_ids = []
    for keyword in request.keywords:
        job_id = str(uuid4())
        job_ids.append(job_id)

        # 메트릭 기록 (queued 상태)
        metric = PipelineMetric(
            stage="trigger",
            keyword=keyword,
            status="success",
            items_processed=1,
            metadata_={
                "job_id": job_id,
                "priority": request.priority,
                "dry_run": request.dry_run,
            },
        )
        session.add(metric)

    await session.flush()

    return TriggerPipelineResponse(
        job_ids=job_ids,
        message=f"Pipeline triggered for {len(request.keywords)} keywords",
        queued_count=len(request.keywords),
    )


@router.get(
    "/status",
    summary="파이프라인 상태",
    description="현재 파이프라인 상태를 조회합니다.",
)
async def get_pipeline_status(
    session: AsyncSession = Depends(get_session),
) -> dict:
    """파이프라인 현재 상태"""
    now = datetime.now()
    last_hour = now - timedelta(hours=1)

    # 최근 1시간 실행
    recent_result = await session.execute(
        select(func.count(PipelineMetric.id)).where(PipelineMetric.recorded_at >= last_hour)
    )
    recent_runs = recent_result.scalar() or 0

    # 최근 에러
    error_result = await session.execute(
        select(func.count(PipelineMetric.id))
        .where(PipelineMetric.recorded_at >= last_hour)
        .where(PipelineMetric.status == "failure")
    )
    recent_errors = error_result.scalar() or 0

    # 마지막 실행 시간
    last_run_result = await session.execute(
        select(PipelineMetric.recorded_at).order_by(PipelineMetric.recorded_at.desc()).limit(1)
    )
    last_run = last_run_result.scalar_one_or_none()

    return {
        "status": "running",
        "recent_runs_1h": recent_runs,
        "recent_errors_1h": recent_errors,
        "error_rate_1h": round(recent_errors / recent_runs * 100, 2) if recent_runs > 0 else 0.0,
        "last_run_at": last_run.isoformat() if last_run else None,
        "timestamp": now.isoformat(),
    }
