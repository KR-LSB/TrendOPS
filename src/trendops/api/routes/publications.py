# src/trendops/api/routes/publications.py
"""
TrendOps Publications API 라우터

Week 6 Day 4: SNS 발행 관리 API
"""
from datetime import datetime, timedelta
from typing import Literal
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from trendops.database.connection import get_session
from trendops.database.models import Analysis, Keyword, Publication

router = APIRouter()


# =============================================================================
# Request/Response Schemas
# =============================================================================


class PublicationCreate(BaseModel):
    """발행 생성 요청"""

    keyword_id: UUID = Field(..., description="키워드 ID")
    analysis_id: UUID = Field(..., description="분석 ID")
    platform: Literal["instagram", "threads"] = Field(..., description="플랫폼")
    caption: str = Field(..., min_length=1, max_length=2200, description="캡션")
    image_path: str | None = Field(default=None, description="이미지 경로")
    metadata: dict | None = Field(default=None, description="추가 메타데이터")


class PublicationUpdate(BaseModel):
    """발행 수정 요청"""

    status: Literal["pending", "approved", "published", "failed", "rejected"] | None = None
    review_status: Literal["pending", "approved", "rejected", "modified"] | None = None
    reviewer_note: str | None = None
    post_id: str | None = None
    post_url: str | None = None


class PublicationResponse(BaseModel):
    """발행 응답"""

    id: UUID
    keyword_id: UUID
    keyword_text: str | None = None
    analysis_id: UUID
    platform: str
    post_id: str | None
    post_url: str | None
    image_path: str | None
    caption: str | None
    status: str
    review_status: str | None
    reviewer_note: str | None
    published_at: datetime | None
    created_at: datetime | None

    class Config:
        from_attributes = True


class PublicationListResponse(BaseModel):
    """발행 목록 응답"""

    items: list[PublicationResponse]
    total: int
    page: int
    page_size: int
    has_next: bool


class PublicationStats(BaseModel):
    """발행 통계"""

    total_publications: int
    published_count: int
    pending_count: int
    failed_count: int
    rejected_count: int
    publications_today: int
    publications_this_week: int
    platform_breakdown: dict[str, int]


class ReviewAction(BaseModel):
    """리뷰 액션 요청"""

    action: Literal["approve", "reject", "modify"]
    note: str | None = Field(default=None, description="리뷰어 노트")
    modified_caption: str | None = Field(default=None, description="수정된 캡션")


class PublishAction(BaseModel):
    """발행 실행 결과"""

    post_id: str = Field(..., description="발행된 포스트 ID")
    post_url: str | None = Field(default=None, description="발행된 포스트 URL")


# =============================================================================
# Helper Functions
# =============================================================================


async def publication_to_response(
    pub: Publication,
    session: AsyncSession,
) -> PublicationResponse:
    """Publication 모델을 응답 스키마로 변환"""
    # 키워드 텍스트 조회
    keyword_text = None
    if pub.keyword_id:
        kw_result = await session.execute(
            select(Keyword.keyword).where(Keyword.id == pub.keyword_id)
        )
        keyword_text = kw_result.scalar_one_or_none()

    return PublicationResponse(
        id=pub.id,
        keyword_id=pub.keyword_id,
        keyword_text=keyword_text,
        analysis_id=pub.analysis_id,
        platform=pub.platform,
        post_id=pub.post_id,
        post_url=pub.post_url,
        image_path=pub.image_path,
        caption=pub.caption,
        status=pub.status,
        review_status=pub.review_status,
        reviewer_note=pub.reviewer_note,
        published_at=pub.published_at,
        created_at=pub.created_at,
    )


# =============================================================================
# Endpoints
# =============================================================================


@router.get(
    "/",
    response_model=PublicationListResponse,
    summary="발행 목록 조회",
    description="SNS 발행 기록을 조회합니다.",
)
async def list_publications(
    session: AsyncSession = Depends(get_session),
    platform: Literal["instagram", "threads"] | None = Query(None, description="플랫폼"),
    status: str | None = Query(None, description="상태 필터"),
    review_status: str | None = Query(None, description="리뷰 상태"),
    keyword_id: UUID | None = Query(None, description="키워드 ID"),
    start_date: datetime | None = Query(None, description="시작 날짜"),
    end_date: datetime | None = Query(None, description="종료 날짜"),
    page: int = Query(1, ge=1, description="페이지 번호"),
    page_size: int = Query(20, ge=1, le=100, description="페이지 크기"),
    sort_order: Literal["asc", "desc"] = Query("desc", description="정렬 순서"),
) -> PublicationListResponse:
    """발행 목록 조회"""
    query = select(Publication)
    count_query = select(func.count(Publication.id))

    # 필터 적용
    if platform:
        query = query.where(Publication.platform == platform)
        count_query = count_query.where(Publication.platform == platform)

    if status:
        query = query.where(Publication.status == status)
        count_query = count_query.where(Publication.status == status)

    if review_status:
        query = query.where(Publication.review_status == review_status)
        count_query = count_query.where(Publication.review_status == review_status)

    if keyword_id:
        query = query.where(Publication.keyword_id == keyword_id)
        count_query = count_query.where(Publication.keyword_id == keyword_id)

    if start_date:
        query = query.where(Publication.created_at >= start_date)
        count_query = count_query.where(Publication.created_at >= start_date)

    if end_date:
        query = query.where(Publication.created_at <= end_date)
        count_query = count_query.where(Publication.created_at <= end_date)

    # 전체 개수
    total_result = await session.execute(count_query)
    total = total_result.scalar() or 0

    # 정렬
    if sort_order == "desc":
        query = query.order_by(Publication.created_at.desc())
    else:
        query = query.order_by(Publication.created_at.asc())

    # 페이지네이션
    offset = (page - 1) * page_size
    query = query.offset(offset).limit(page_size)

    # 실행
    result = await session.execute(query)
    publications = result.scalars().all()

    items = [await publication_to_response(p, session) for p in publications]

    return PublicationListResponse(
        items=items,
        total=total,
        page=page,
        page_size=page_size,
        has_next=(offset + page_size) < total,
    )


@router.post(
    "/",
    response_model=PublicationResponse,
    status_code=201,
    summary="발행 생성",
    description="새로운 발행 레코드를 생성합니다.",
)
async def create_publication(
    data: PublicationCreate,
    session: AsyncSession = Depends(get_session),
) -> PublicationResponse:
    """발행 생성"""
    # 키워드 확인
    kw_result = await session.execute(select(Keyword).where(Keyword.id == data.keyword_id))
    if not kw_result.scalar_one_or_none():
        raise HTTPException(status_code=404, detail="Keyword not found")

    # 분석 확인
    analysis_result = await session.execute(select(Analysis).where(Analysis.id == data.analysis_id))
    if not analysis_result.scalar_one_or_none():
        raise HTTPException(status_code=404, detail="Analysis not found")

    publication = Publication(
        keyword_id=data.keyword_id,
        analysis_id=data.analysis_id,
        platform=data.platform,
        caption=data.caption,
        image_path=data.image_path,
        status="pending",
        review_status="pending",
        metadata_=data.metadata or {},
    )
    session.add(publication)
    await session.flush()
    await session.refresh(publication)

    return await publication_to_response(publication, session)


@router.get(
    "/stats",
    response_model=PublicationStats,
    summary="발행 통계",
    description="발행 관련 통계 정보를 조회합니다.",
)
async def get_publication_stats(
    session: AsyncSession = Depends(get_session),
) -> PublicationStats:
    """발행 통계 조회"""
    now = datetime.now()
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    week_start = today_start - timedelta(days=7)

    # 전체 개수
    total_result = await session.execute(select(func.count(Publication.id)))
    total = total_result.scalar() or 0

    # 상태별 개수
    published_result = await session.execute(
        select(func.count(Publication.id)).where(Publication.status == "published")
    )
    published = published_result.scalar() or 0

    pending_result = await session.execute(
        select(func.count(Publication.id)).where(Publication.status == "pending")
    )
    pending = pending_result.scalar() or 0

    failed_result = await session.execute(
        select(func.count(Publication.id)).where(Publication.status == "failed")
    )
    failed = failed_result.scalar() or 0

    rejected_result = await session.execute(
        select(func.count(Publication.id)).where(Publication.status == "rejected")
    )
    rejected = rejected_result.scalar() or 0

    # 오늘 발행
    today_result = await session.execute(
        select(func.count(Publication.id)).where(Publication.created_at >= today_start)
    )
    today_count = today_result.scalar() or 0

    # 이번 주 발행
    week_result = await session.execute(
        select(func.count(Publication.id)).where(Publication.created_at >= week_start)
    )
    week_count = week_result.scalar() or 0

    # 플랫폼별 개수
    platform_result = await session.execute(
        select(Publication.platform, func.count(Publication.id)).group_by(Publication.platform)
    )
    platform_breakdown = {row[0]: row[1] for row in platform_result.all()}

    return PublicationStats(
        total_publications=total,
        published_count=published,
        pending_count=pending,
        failed_count=failed,
        rejected_count=rejected,
        publications_today=today_count,
        publications_this_week=week_count,
        platform_breakdown=platform_breakdown,
    )


@router.get(
    "/pending",
    response_model=list[PublicationResponse],
    summary="승인 대기 목록",
    description="승인 대기 중인 발행 목록을 조회합니다.",
)
async def list_pending_publications(
    session: AsyncSession = Depends(get_session),
    platform: Literal["instagram", "threads"] | None = Query(None),
    limit: int = Query(20, ge=1, le=50),
) -> list[PublicationResponse]:
    """승인 대기 발행 목록"""
    query = (
        select(Publication)
        .where(Publication.review_status == "pending")
        .order_by(Publication.created_at.asc())
        .limit(limit)
    )

    if platform:
        query = query.where(Publication.platform == platform)

    result = await session.execute(query)
    publications = result.scalars().all()

    return [await publication_to_response(p, session) for p in publications]


@router.get(
    "/{publication_id}",
    response_model=PublicationResponse,
    summary="발행 상세 조회",
    description="특정 발행 기록의 상세 정보를 조회합니다.",
)
async def get_publication(
    publication_id: UUID,
    session: AsyncSession = Depends(get_session),
) -> PublicationResponse:
    """발행 상세 조회"""
    result = await session.execute(select(Publication).where(Publication.id == publication_id))
    publication = result.scalar_one_or_none()

    if not publication:
        raise HTTPException(status_code=404, detail="Publication not found")

    return await publication_to_response(publication, session)


@router.patch(
    "/{publication_id}",
    response_model=PublicationResponse,
    summary="발행 수정",
    description="발행 정보를 수정합니다.",
)
async def update_publication(
    publication_id: UUID,
    data: PublicationUpdate,
    session: AsyncSession = Depends(get_session),
) -> PublicationResponse:
    """발행 수정"""
    result = await session.execute(select(Publication).where(Publication.id == publication_id))
    publication = result.scalar_one_or_none()

    if not publication:
        raise HTTPException(status_code=404, detail="Publication not found")

    update_data = data.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(publication, field, value)

    await session.flush()
    await session.refresh(publication)

    return await publication_to_response(publication, session)


@router.post(
    "/{publication_id}/review",
    response_model=PublicationResponse,
    summary="발행 리뷰",
    description="발행을 승인, 거부, 또는 수정합니다.",
)
async def review_publication(
    publication_id: UUID,
    action: ReviewAction,
    session: AsyncSession = Depends(get_session),
) -> PublicationResponse:
    """발행 리뷰 처리"""
    result = await session.execute(select(Publication).where(Publication.id == publication_id))
    publication = result.scalar_one_or_none()

    if not publication:
        raise HTTPException(status_code=404, detail="Publication not found")

    if publication.review_status != "pending":
        raise HTTPException(
            status_code=400, detail=f"Publication already reviewed: {publication.review_status}"
        )

    if action.action == "approve":
        publication.review_status = "approved"
        publication.status = "approved"
    elif action.action == "reject":
        publication.review_status = "rejected"
        publication.status = "rejected"
    elif action.action == "modify":
        publication.review_status = "modified"
        if action.modified_caption:
            publication.caption = action.modified_caption

    publication.reviewer_note = action.note

    await session.flush()
    await session.refresh(publication)

    return await publication_to_response(publication, session)


@router.post(
    "/{publication_id}/publish",
    response_model=PublicationResponse,
    summary="발행 실행",
    description="승인된 발행을 실제로 발행 처리합니다.",
)
async def execute_publish(
    publication_id: UUID,
    publish_result: PublishAction,
    session: AsyncSession = Depends(get_session),
) -> PublicationResponse:
    """발행 실행 (실제 SNS 발행 후 결과 기록)"""
    result = await session.execute(select(Publication).where(Publication.id == publication_id))
    publication = result.scalar_one_or_none()

    if not publication:
        raise HTTPException(status_code=404, detail="Publication not found")

    if publication.review_status not in ("approved", "modified"):
        raise HTTPException(
            status_code=400, detail="Publication must be approved before publishing"
        )

    publication.post_id = publish_result.post_id
    publication.post_url = publish_result.post_url
    publication.status = "published"
    publication.published_at = datetime.now()

    await session.flush()
    await session.refresh(publication)

    return await publication_to_response(publication, session)


@router.post(
    "/{publication_id}/fail",
    response_model=PublicationResponse,
    summary="발행 실패 기록",
    description="발행 실패를 기록합니다.",
)
async def mark_publication_failed(
    publication_id: UUID,
    error_note: str = Query(..., description="실패 사유"),
    session: AsyncSession = Depends(get_session),
) -> PublicationResponse:
    """발행 실패 기록"""
    result = await session.execute(select(Publication).where(Publication.id == publication_id))
    publication = result.scalar_one_or_none()

    if not publication:
        raise HTTPException(status_code=404, detail="Publication not found")

    publication.status = "failed"
    publication.reviewer_note = f"Failed: {error_note}"

    await session.flush()
    await session.refresh(publication)

    return await publication_to_response(publication, session)


@router.delete(
    "/{publication_id}",
    status_code=204,
    summary="발행 삭제",
    description="발행 기록을 삭제합니다.",
)
async def delete_publication(
    publication_id: UUID,
    session: AsyncSession = Depends(get_session),
) -> None:
    """발행 삭제"""
    result = await session.execute(select(Publication).where(Publication.id == publication_id))
    publication = result.scalar_one_or_none()

    if not publication:
        raise HTTPException(status_code=404, detail="Publication not found")

    if publication.status == "published":
        raise HTTPException(status_code=400, detail="Cannot delete published content")

    await session.delete(publication)
