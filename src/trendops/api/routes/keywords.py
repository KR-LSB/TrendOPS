# src/trendops/api/routes/keywords.py
"""
TrendOps Keywords API 라우터

Week 6 Day 4: 키워드 관리 CRUD API
"""
from datetime import datetime
from typing import Literal
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from trendops.database.connection import get_session
from trendops.database.models import Keyword


router = APIRouter()


# =============================================================================
# Request/Response Schemas
# =============================================================================

class KeywordCreate(BaseModel):
    """키워드 생성 요청"""
    keyword: str = Field(..., min_length=1, max_length=255, description="키워드 텍스트")
    source: str = Field(default="manual", max_length=50, description="소스")
    trend_score: float = Field(default=5.0, ge=0.0, le=10.0, description="트렌드 점수")
    metadata: dict | None = Field(default=None, description="추가 메타데이터")


class KeywordUpdate(BaseModel):
    """키워드 수정 요청"""
    trend_score: float | None = Field(default=None, ge=0.0, le=10.0, description="트렌드 점수")
    is_active: bool | None = Field(default=None, description="활성화 상태")
    metadata: dict | None = Field(default=None, description="추가 메타데이터")


class KeywordResponse(BaseModel):
    """키워드 응답"""
    id: UUID
    keyword: str
    source: str
    trend_score: float
    is_active: bool
    first_seen_at: datetime | None
    last_seen_at: datetime | None
    article_count: int = 0
    analysis_count: int = 0
    
    class Config:
        from_attributes = True


class KeywordListResponse(BaseModel):
    """키워드 목록 응답"""
    items: list[KeywordResponse]
    total: int
    page: int
    page_size: int
    has_next: bool


class KeywordStats(BaseModel):
    """키워드 통계"""
    total_keywords: int
    active_keywords: int
    avg_trend_score: float
    top_sources: dict[str, int]


# =============================================================================
# Helper Functions
# =============================================================================

def keyword_to_response(kw: Keyword) -> KeywordResponse:
    """Keyword 모델을 응답 스키마로 변환"""
    return KeywordResponse(
        id=kw.id,
        keyword=kw.keyword,
        source=kw.source,
        trend_score=kw.trend_score,
        is_active=kw.is_active,
        first_seen_at=kw.first_seen_at,
        last_seen_at=kw.last_seen_at,
        article_count=len(kw.articles) if kw.articles else 0,
        analysis_count=len(kw.analyses) if kw.analyses else 0,
    )


# =============================================================================
# Endpoints
# =============================================================================

@router.get(
    "/",
    response_model=KeywordListResponse,
    summary="키워드 목록 조회",
    description="등록된 키워드 목록을 조회합니다. 페이지네이션과 필터링을 지원합니다.",
)
async def list_keywords(
    session: AsyncSession = Depends(get_session),
    active_only: bool = Query(True, description="활성 키워드만 조회"),
    source: str | None = Query(None, description="소스 필터"),
    min_score: float | None = Query(None, ge=0.0, le=10.0, description="최소 트렌드 점수"),
    search: str | None = Query(None, description="키워드 검색"),
    page: int = Query(1, ge=1, description="페이지 번호"),
    page_size: int = Query(20, ge=1, le=100, description="페이지 크기"),
    sort_by: Literal["trend_score", "first_seen_at", "keyword"] = Query(
        "trend_score", description="정렬 기준"
    ),
    sort_order: Literal["asc", "desc"] = Query("desc", description="정렬 순서"),
) -> KeywordListResponse:
    """키워드 목록 조회 (페이지네이션)"""
    # 기본 쿼리
    query = select(Keyword)
    count_query = select(func.count(Keyword.id))
    
    # 필터 적용
    if active_only:
        query = query.where(Keyword.is_active == True)
        count_query = count_query.where(Keyword.is_active == True)
    
    if source:
        query = query.where(Keyword.source == source)
        count_query = count_query.where(Keyword.source == source)
    
    if min_score is not None:
        query = query.where(Keyword.trend_score >= min_score)
        count_query = count_query.where(Keyword.trend_score >= min_score)
    
    if search:
        query = query.where(Keyword.keyword.ilike(f"%{search}%"))
        count_query = count_query.where(Keyword.keyword.ilike(f"%{search}%"))
    
    # 전체 개수 조회
    total_result = await session.execute(count_query)
    total = total_result.scalar() or 0
    
    # 정렬
    sort_column = getattr(Keyword, sort_by)
    if sort_order == "desc":
        query = query.order_by(sort_column.desc())
    else:
        query = query.order_by(sort_column.asc())
    
    # 페이지네이션
    offset = (page - 1) * page_size
    query = query.offset(offset).limit(page_size)
    
    # 실행
    result = await session.execute(query)
    keywords = result.scalars().all()
    
    return KeywordListResponse(
        items=[keyword_to_response(kw) for kw in keywords],
        total=total,
        page=page,
        page_size=page_size,
        has_next=(offset + page_size) < total,
    )


@router.post(
    "/",
    response_model=KeywordResponse,
    status_code=201,
    summary="키워드 생성",
    description="새로운 키워드를 등록합니다.",
)
async def create_keyword(
    data: KeywordCreate,
    session: AsyncSession = Depends(get_session),
) -> KeywordResponse:
    """새 키워드 등록"""
    # 중복 체크
    existing = await session.execute(
        select(Keyword).where(
            Keyword.keyword == data.keyword,
            Keyword.source == data.source,
        )
    )
    if existing.scalar_one_or_none():
        raise HTTPException(
            status_code=409,
            detail=f"Keyword '{data.keyword}' from source '{data.source}' already exists"
        )
    
    # 생성
    keyword = Keyword(
        keyword=data.keyword,
        source=data.source,
        trend_score=data.trend_score,
        metadata_=data.metadata or {},
    )
    session.add(keyword)
    await session.flush()
    await session.refresh(keyword)
    
    return keyword_to_response(keyword)


@router.get(
    "/stats",
    response_model=KeywordStats,
    summary="키워드 통계",
    description="키워드 관련 통계 정보를 조회합니다.",
)
async def get_keyword_stats(
    session: AsyncSession = Depends(get_session),
) -> KeywordStats:
    """키워드 통계 조회"""
    # 전체 개수
    total_result = await session.execute(select(func.count(Keyword.id)))
    total = total_result.scalar() or 0
    
    # 활성 개수
    active_result = await session.execute(
        select(func.count(Keyword.id)).where(Keyword.is_active == True)
    )
    active = active_result.scalar() or 0
    
    # 평균 점수
    avg_result = await session.execute(
        select(func.avg(Keyword.trend_score)).where(Keyword.is_active == True)
    )
    avg_score = avg_result.scalar() or 0.0
    
    # 소스별 개수
    source_result = await session.execute(
        select(Keyword.source, func.count(Keyword.id))
        .group_by(Keyword.source)
    )
    top_sources = {row[0]: row[1] for row in source_result.all()}
    
    return KeywordStats(
        total_keywords=total,
        active_keywords=active,
        avg_trend_score=round(float(avg_score), 2),
        top_sources=top_sources,
    )


@router.get(
    "/{keyword_id}",
    response_model=KeywordResponse,
    summary="키워드 상세 조회",
    description="특정 키워드의 상세 정보를 조회합니다.",
)
async def get_keyword(
    keyword_id: UUID,
    session: AsyncSession = Depends(get_session),
) -> KeywordResponse:
    """키워드 상세 조회"""
    result = await session.execute(
        select(Keyword).where(Keyword.id == keyword_id)
    )
    keyword = result.scalar_one_or_none()
    
    if not keyword:
        raise HTTPException(status_code=404, detail="Keyword not found")
    
    return keyword_to_response(keyword)


@router.patch(
    "/{keyword_id}",
    response_model=KeywordResponse,
    summary="키워드 수정",
    description="키워드 정보를 부분 수정합니다.",
)
async def update_keyword(
    keyword_id: UUID,
    data: KeywordUpdate,
    session: AsyncSession = Depends(get_session),
) -> KeywordResponse:
    """키워드 수정"""
    result = await session.execute(
        select(Keyword).where(Keyword.id == keyword_id)
    )
    keyword = result.scalar_one_or_none()
    
    if not keyword:
        raise HTTPException(status_code=404, detail="Keyword not found")
    
    # 필드 업데이트
    update_data = data.model_dump(exclude_unset=True)
    if "metadata" in update_data:
        update_data["metadata_"] = update_data.pop("metadata")
    
    for field, value in update_data.items():
        setattr(keyword, field, value)
    
    keyword.last_seen_at = datetime.now()
    await session.flush()
    await session.refresh(keyword)
    
    return keyword_to_response(keyword)


@router.delete(
    "/{keyword_id}",
    status_code=204,
    summary="키워드 비활성화",
    description="키워드를 비활성화합니다 (soft delete).",
)
async def delete_keyword(
    keyword_id: UUID,
    session: AsyncSession = Depends(get_session),
) -> None:
    """키워드 비활성화 (soft delete)"""
    result = await session.execute(
        select(Keyword).where(Keyword.id == keyword_id)
    )
    keyword = result.scalar_one_or_none()
    
    if not keyword:
        raise HTTPException(status_code=404, detail="Keyword not found")
    
    keyword.is_active = False
    keyword.last_seen_at = datetime.now()


@router.post(
    "/{keyword_id}/activate",
    response_model=KeywordResponse,
    summary="키워드 활성화",
    description="비활성화된 키워드를 다시 활성화합니다.",
)
async def activate_keyword(
    keyword_id: UUID,
    session: AsyncSession = Depends(get_session),
) -> KeywordResponse:
    """키워드 재활성화"""
    result = await session.execute(
        select(Keyword).where(Keyword.id == keyword_id)
    )
    keyword = result.scalar_one_or_none()
    
    if not keyword:
        raise HTTPException(status_code=404, detail="Keyword not found")
    
    keyword.is_active = True
    keyword.last_seen_at = datetime.now()
    await session.flush()
    await session.refresh(keyword)
    
    return keyword_to_response(keyword)


@router.post(
    "/bulk",
    response_model=list[KeywordResponse],
    status_code=201,
    summary="키워드 일괄 등록",
    description="여러 키워드를 한 번에 등록합니다.",
)
async def bulk_create_keywords(
    keywords: list[KeywordCreate],
    session: AsyncSession = Depends(get_session),
) -> list[KeywordResponse]:
    """키워드 일괄 등록"""
    if len(keywords) > 100:
        raise HTTPException(
            status_code=400,
            detail="Maximum 100 keywords can be created at once"
        )
    
    created = []
    for data in keywords:
        # 중복 체크
        existing = await session.execute(
            select(Keyword).where(
                Keyword.keyword == data.keyword,
                Keyword.source == data.source,
            )
        )
        if existing.scalar_one_or_none():
            continue  # 중복은 스킵
        
        keyword = Keyword(
            keyword=data.keyword,
            source=data.source,
            trend_score=data.trend_score,
            metadata_=data.metadata or {},
        )
        session.add(keyword)
        created.append(keyword)
    
    await session.flush()
    for kw in created:
        await session.refresh(kw)
    
    return [keyword_to_response(kw) for kw in created]