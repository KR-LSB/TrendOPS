# src/trendops/api/routes/analysis.py
"""
TrendOps Analysis API 라우터

Week 6 Day 4: LLM 분석 결과 API
"""
from datetime import datetime, timedelta
from typing import Literal
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from trendops.database.connection import get_session
from trendops.database.models import Analysis, Keyword

router = APIRouter()


# =============================================================================
# Request/Response Schemas
# =============================================================================


class AnalysisCreate(BaseModel):
    """분석 결과 생성 요청"""

    keyword_id: UUID = Field(..., description="키워드 ID")
    summary: str = Field(..., min_length=10, max_length=5000, description="분석 요약")
    key_points: list[str] = Field(default_factory=list, description="핵심 포인트")
    sentiment_ratio: dict[str, float] | None = Field(default=None, description="감성 비율")
    guardrail_passed: bool = Field(default=True, description="가드레일 통과 여부")
    guardrail_issues: list[dict] = Field(default_factory=list, description="가드레일 이슈")
    model_name: str | None = Field(default=None, description="사용 모델명")
    processing_time_ms: int | None = Field(default=None, description="처리 시간(ms)")


class AnalysisResponse(BaseModel):
    """분석 결과 응답"""

    id: UUID
    keyword_id: UUID
    keyword_text: str | None = None
    summary: str
    key_points: list[str]
    sentiment_ratio: dict[str, float] | None
    guardrail_passed: bool
    guardrail_issues: list[dict]
    revision_count: int
    model_name: str | None
    processing_time_ms: int | None
    created_at: datetime | None

    class Config:
        from_attributes = True


class AnalysisListResponse(BaseModel):
    """분석 결과 목록 응답"""

    items: list[AnalysisResponse]
    total: int
    page: int
    page_size: int
    has_next: bool


class AnalysisStats(BaseModel):
    """분석 통계"""

    total_analyses: int
    guardrail_passed_count: int
    guardrail_failed_count: int
    avg_processing_time_ms: float | None
    analyses_today: int
    analyses_this_week: int
    model_usage: dict[str, int]


class SentimentSummary(BaseModel):
    """감성 분석 요약"""

    keyword: str
    positive: float
    negative: float
    neutral: float
    sample_count: int


# =============================================================================
# Helper Functions
# =============================================================================


async def analysis_to_response(
    analysis: Analysis,
    session: AsyncSession,
) -> AnalysisResponse:
    """Analysis 모델을 응답 스키마로 변환"""
    # 키워드 텍스트 조회
    keyword_text = None
    if analysis.keyword_id:
        kw_result = await session.execute(
            select(Keyword.keyword).where(Keyword.id == analysis.keyword_id)
        )
        keyword_text = kw_result.scalar_one_or_none()

    return AnalysisResponse(
        id=analysis.id,
        keyword_id=analysis.keyword_id,
        keyword_text=keyword_text,
        summary=analysis.summary,
        key_points=analysis.key_points or [],
        sentiment_ratio=analysis.sentiment_ratio,
        guardrail_passed=analysis.guardrail_passed,
        guardrail_issues=analysis.guardrail_issues or [],
        revision_count=analysis.revision_count,
        model_name=analysis.model_name,
        processing_time_ms=analysis.processing_time_ms,
        created_at=analysis.created_at,
    )


# =============================================================================
# Endpoints
# =============================================================================


@router.get(
    "/",
    response_model=AnalysisListResponse,
    summary="분석 결과 목록 조회",
    description="LLM 분석 결과 목록을 조회합니다.",
)
async def list_analyses(
    session: AsyncSession = Depends(get_session),
    keyword_id: UUID | None = Query(None, description="키워드 ID 필터"),
    guardrail_passed: bool | None = Query(None, description="가드레일 통과 여부"),
    model_name: str | None = Query(None, description="모델명 필터"),
    start_date: datetime | None = Query(None, description="시작 날짜"),
    end_date: datetime | None = Query(None, description="종료 날짜"),
    page: int = Query(1, ge=1, description="페이지 번호"),
    page_size: int = Query(20, ge=1, le=100, description="페이지 크기"),
    sort_by: Literal["created_at", "processing_time_ms"] = Query(
        "created_at", description="정렬 기준"
    ),
    sort_order: Literal["asc", "desc"] = Query("desc", description="정렬 순서"),
) -> AnalysisListResponse:
    """분석 결과 목록 조회"""
    query = select(Analysis)
    count_query = select(func.count(Analysis.id))

    # 필터 적용
    if keyword_id:
        query = query.where(Analysis.keyword_id == keyword_id)
        count_query = count_query.where(Analysis.keyword_id == keyword_id)

    if guardrail_passed is not None:
        query = query.where(Analysis.guardrail_passed == guardrail_passed)
        count_query = count_query.where(Analysis.guardrail_passed == guardrail_passed)

    if model_name:
        query = query.where(Analysis.model_name == model_name)
        count_query = count_query.where(Analysis.model_name == model_name)

    if start_date:
        query = query.where(Analysis.created_at >= start_date)
        count_query = count_query.where(Analysis.created_at >= start_date)

    if end_date:
        query = query.where(Analysis.created_at <= end_date)
        count_query = count_query.where(Analysis.created_at <= end_date)

    # 전체 개수
    total_result = await session.execute(count_query)
    total = total_result.scalar() or 0

    # 정렬
    sort_column = getattr(Analysis, sort_by)
    if sort_order == "desc":
        query = query.order_by(sort_column.desc())
    else:
        query = query.order_by(sort_column.asc())

    # 페이지네이션
    offset = (page - 1) * page_size
    query = query.offset(offset).limit(page_size)

    # 실행
    result = await session.execute(query)
    analyses = result.scalars().all()

    items = [await analysis_to_response(a, session) for a in analyses]

    return AnalysisListResponse(
        items=items,
        total=total,
        page=page,
        page_size=page_size,
        has_next=(offset + page_size) < total,
    )


@router.post(
    "/",
    response_model=AnalysisResponse,
    status_code=201,
    summary="분석 결과 생성",
    description="새로운 분석 결과를 저장합니다.",
)
async def create_analysis(
    data: AnalysisCreate,
    session: AsyncSession = Depends(get_session),
) -> AnalysisResponse:
    """분석 결과 생성"""
    # 키워드 존재 확인
    kw_result = await session.execute(select(Keyword).where(Keyword.id == data.keyword_id))
    if not kw_result.scalar_one_or_none():
        raise HTTPException(status_code=404, detail="Keyword not found")

    analysis = Analysis(
        keyword_id=data.keyword_id,
        summary=data.summary,
        key_points=data.key_points,
        sentiment_ratio=data.sentiment_ratio,
        guardrail_passed=data.guardrail_passed,
        guardrail_issues=data.guardrail_issues,
        model_name=data.model_name,
        processing_time_ms=data.processing_time_ms,
    )
    session.add(analysis)
    await session.flush()
    await session.refresh(analysis)

    return await analysis_to_response(analysis, session)


@router.get(
    "/stats",
    response_model=AnalysisStats,
    summary="분석 통계",
    description="분석 관련 통계 정보를 조회합니다.",
)
async def get_analysis_stats(
    session: AsyncSession = Depends(get_session),
) -> AnalysisStats:
    """분석 통계 조회"""
    now = datetime.now()
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    week_start = today_start - timedelta(days=7)

    # 전체 개수
    total_result = await session.execute(select(func.count(Analysis.id)))
    total = total_result.scalar() or 0

    # 가드레일 통과/실패
    passed_result = await session.execute(
        select(func.count(Analysis.id)).where(Analysis.guardrail_passed == True)
    )
    passed = passed_result.scalar() or 0

    failed_result = await session.execute(
        select(func.count(Analysis.id)).where(Analysis.guardrail_passed == False)
    )
    failed = failed_result.scalar() or 0

    # 평균 처리 시간
    avg_time_result = await session.execute(
        select(func.avg(Analysis.processing_time_ms)).where(Analysis.processing_time_ms.isnot(None))
    )
    avg_time = avg_time_result.scalar()

    # 오늘 분석
    today_result = await session.execute(
        select(func.count(Analysis.id)).where(Analysis.created_at >= today_start)
    )
    today_count = today_result.scalar() or 0

    # 이번 주 분석
    week_result = await session.execute(
        select(func.count(Analysis.id)).where(Analysis.created_at >= week_start)
    )
    week_count = week_result.scalar() or 0

    # 모델별 사용량
    model_result = await session.execute(
        select(Analysis.model_name, func.count(Analysis.id))
        .where(Analysis.model_name.isnot(None))
        .group_by(Analysis.model_name)
    )
    model_usage = {row[0]: row[1] for row in model_result.all()}

    return AnalysisStats(
        total_analyses=total,
        guardrail_passed_count=passed,
        guardrail_failed_count=failed,
        avg_processing_time_ms=round(float(avg_time), 2) if avg_time else None,
        analyses_today=today_count,
        analyses_this_week=week_count,
        model_usage=model_usage,
    )


@router.get(
    "/sentiment",
    response_model=list[SentimentSummary],
    summary="감성 분석 요약",
    description="키워드별 감성 분석 결과를 요약합니다.",
)
async def get_sentiment_summary(
    session: AsyncSession = Depends(get_session),
    limit: int = Query(10, ge=1, le=50, description="최대 결과 수"),
) -> list[SentimentSummary]:
    """키워드별 감성 분석 요약"""
    # 최근 분석 결과에서 감성 비율이 있는 것만 조회
    result = await session.execute(
        select(Analysis)
        .where(Analysis.sentiment_ratio.isnot(None))
        .order_by(Analysis.created_at.desc())
        .limit(limit * 2)  # 키워드 중복 고려
    )
    analyses = result.scalars().all()

    # 키워드별 집계
    keyword_sentiments: dict[UUID, list[dict]] = {}
    for analysis in analyses:
        if analysis.keyword_id not in keyword_sentiments:
            keyword_sentiments[analysis.keyword_id] = []
        if analysis.sentiment_ratio:
            keyword_sentiments[analysis.keyword_id].append(analysis.sentiment_ratio)

    summaries = []
    for kw_id, sentiments in list(keyword_sentiments.items())[:limit]:
        # 키워드 텍스트 조회
        kw_result = await session.execute(select(Keyword.keyword).where(Keyword.id == kw_id))
        keyword_text = kw_result.scalar_one_or_none() or "Unknown"

        # 평균 계산
        pos = sum(s.get("positive", 0) for s in sentiments) / len(sentiments)
        neg = sum(s.get("negative", 0) for s in sentiments) / len(sentiments)
        neu = sum(s.get("neutral", 0) for s in sentiments) / len(sentiments)

        summaries.append(
            SentimentSummary(
                keyword=keyword_text,
                positive=round(pos, 2),
                negative=round(neg, 2),
                neutral=round(neu, 2),
                sample_count=len(sentiments),
            )
        )

    return summaries


@router.get(
    "/{analysis_id}",
    response_model=AnalysisResponse,
    summary="분석 결과 상세 조회",
    description="특정 분석 결과의 상세 정보를 조회합니다.",
)
async def get_analysis(
    analysis_id: UUID,
    session: AsyncSession = Depends(get_session),
) -> AnalysisResponse:
    """분석 결과 상세 조회"""
    result = await session.execute(select(Analysis).where(Analysis.id == analysis_id))
    analysis = result.scalar_one_or_none()

    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")

    return await analysis_to_response(analysis, session)


@router.delete(
    "/{analysis_id}",
    status_code=204,
    summary="분석 결과 삭제",
    description="분석 결과를 삭제합니다.",
)
async def delete_analysis(
    analysis_id: UUID,
    session: AsyncSession = Depends(get_session),
) -> None:
    """분석 결과 삭제"""
    result = await session.execute(select(Analysis).where(Analysis.id == analysis_id))
    analysis = result.scalar_one_or_none()

    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")

    await session.delete(analysis)


@router.get(
    "/keyword/{keyword_id}",
    response_model=list[AnalysisResponse],
    summary="키워드별 분석 조회",
    description="특정 키워드의 모든 분석 결과를 조회합니다.",
)
async def get_analyses_by_keyword(
    keyword_id: UUID,
    session: AsyncSession = Depends(get_session),
    limit: int = Query(10, ge=1, le=50, description="최대 결과 수"),
) -> list[AnalysisResponse]:
    """키워드별 분석 결과 조회"""
    result = await session.execute(
        select(Analysis)
        .where(Analysis.keyword_id == keyword_id)
        .order_by(Analysis.created_at.desc())
        .limit(limit)
    )
    analyses = result.scalars().all()

    return [await analysis_to_response(a, session) for a in analyses]
