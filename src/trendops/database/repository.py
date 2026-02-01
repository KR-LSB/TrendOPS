# src/trendops/database/repository.py
"""
TrendOps Database Repository
Week 6 Day 2: PostgreSQL Schema + SQLAlchemy

Repository pattern for database operations.
Provides clean interface for CRUD operations on each model.
"""
from datetime import datetime, timedelta
from typing import TypeVar, Generic
from uuid import UUID

from sqlalchemy import select, update, delete, func
from sqlalchemy.ext.asyncio import AsyncSession

from trendops.database.models import (
    Base,
    Keyword,
    Article,
    Analysis,
    Publication,
    PipelineMetric,
    DailyReport,
)


T = TypeVar("T", bound=Base)


class BaseRepository(Generic[T]):
    """Base repository with common CRUD operations"""
    
    def __init__(self, session: AsyncSession, model: type[T]):
        self.session = session
        self.model = model
    
    async def create(self, **kwargs) -> T:
        """Create a new record"""
        obj = self.model(**kwargs)
        self.session.add(obj)
        await self.session.flush()
        return obj
    
    async def get_by_id(self, id: UUID | str) -> T | None:
        """Get record by ID"""
        result = await self.session.execute(
            select(self.model).where(self.model.id == id)
        )
        return result.scalar_one_or_none()
    
    async def get_all(self, limit: int = 100, offset: int = 0) -> list[T]:
        """Get all records with pagination"""
        result = await self.session.execute(
            select(self.model).limit(limit).offset(offset)
        )
        return list(result.scalars().all())
    
    async def update(self, id: UUID | str, **kwargs) -> T | None:
        """Update record by ID"""
        await self.session.execute(
            update(self.model).where(self.model.id == id).values(**kwargs)
        )
        return await self.get_by_id(id)
    
    async def delete(self, id: UUID | str) -> bool:
        """Delete record by ID"""
        result = await self.session.execute(
            delete(self.model).where(self.model.id == id)
        )
        return result.rowcount > 0
    
    async def count(self) -> int:
        """Count total records"""
        result = await self.session.execute(
            select(func.count()).select_from(self.model)
        )
        return result.scalar_one()


class KeywordRepository(BaseRepository[Keyword]):
    """Repository for Keyword model"""
    
    def __init__(self, session: AsyncSession):
        super().__init__(session, Keyword)
    
    async def get_by_keyword(self, keyword: str, source: str | None = None) -> Keyword | None:
        """Get keyword by name and optional source"""
        query = select(Keyword).where(Keyword.keyword == keyword)
        if source:
            query = query.where(Keyword.source == source)
        result = await self.session.execute(query)
        return result.scalar_one_or_none()
    
    async def get_active_keywords(self, min_score: float = 0.0) -> list[Keyword]:
        """Get active keywords with minimum score"""
        result = await self.session.execute(
            select(Keyword)
            .where(Keyword.is_active == True)
            .where(Keyword.trend_score >= min_score)
            .order_by(Keyword.trend_score.desc())
        )
        return list(result.scalars().all())
    
    async def upsert(self, keyword: str, source: str, trend_score: float) -> Keyword:
        """Insert or update keyword"""
        existing = await self.get_by_keyword(keyword, source)
        if existing:
            existing.trend_score = trend_score
            existing.last_seen_at = datetime.utcnow()
            existing.is_active = True
            return existing
        return await self.create(
            keyword=keyword,
            source=source,
            trend_score=trend_score,
        )
    
    async def deactivate_old(self, hours: int = 24) -> int:
        """Deactivate keywords not seen in given hours"""
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        result = await self.session.execute(
            update(Keyword)
            .where(Keyword.last_seen_at < cutoff)
            .where(Keyword.is_active == True)
            .values(is_active=False)
        )
        return result.rowcount


class ArticleRepository(BaseRepository[Article]):
    """Repository for Article model"""
    
    def __init__(self, session: AsyncSession):
        super().__init__(session, Article)
    
    async def get_by_url(self, url: str) -> Article | None:
        """Get article by URL"""
        result = await self.session.execute(
            select(Article).where(Article.url == url)
        )
        return result.scalar_one_or_none()
    
    async def get_by_hash(self, content_hash: str) -> Article | None:
        """Get article by content hash"""
        result = await self.session.execute(
            select(Article).where(Article.content_hash == content_hash)
        )
        return result.scalar_one_or_none()
    
    async def get_by_keyword(
        self, keyword_id: UUID | str, limit: int = 100
    ) -> list[Article]:
        """Get articles for a keyword"""
        result = await self.session.execute(
            select(Article)
            .where(Article.keyword_id == keyword_id)
            .order_by(Article.collected_at.desc())
            .limit(limit)
        )
        return list(result.scalars().all())
    
    async def get_recent(self, hours: int = 24, limit: int = 100) -> list[Article]:
        """Get recently collected articles"""
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        result = await self.session.execute(
            select(Article)
            .where(Article.collected_at >= cutoff)
            .order_by(Article.collected_at.desc())
            .limit(limit)
        )
        return list(result.scalars().all())
    
    async def exists_by_url(self, url: str) -> bool:
        """Check if article URL exists"""
        result = await self.session.execute(
            select(func.count()).select_from(Article).where(Article.url == url)
        )
        return result.scalar_one() > 0


class AnalysisRepository(BaseRepository[Analysis]):
    """Repository for Analysis model"""
    
    def __init__(self, session: AsyncSession):
        super().__init__(session, Analysis)
    
    async def get_by_keyword(
        self, keyword_id: UUID | str, limit: int = 10
    ) -> list[Analysis]:
        """Get analyses for a keyword"""
        result = await self.session.execute(
            select(Analysis)
            .where(Analysis.keyword_id == keyword_id)
            .order_by(Analysis.created_at.desc())
            .limit(limit)
        )
        return list(result.scalars().all())
    
    async def get_latest_for_keyword(self, keyword_id: UUID | str) -> Analysis | None:
        """Get most recent analysis for a keyword"""
        result = await self.session.execute(
            select(Analysis)
            .where(Analysis.keyword_id == keyword_id)
            .order_by(Analysis.created_at.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()
    
    async def get_passed_guardrail(self, limit: int = 100) -> list[Analysis]:
        """Get analyses that passed guardrail"""
        result = await self.session.execute(
            select(Analysis)
            .where(Analysis.guardrail_passed == True)
            .order_by(Analysis.created_at.desc())
            .limit(limit)
        )
        return list(result.scalars().all())


class PublicationRepository(BaseRepository[Publication]):
    """Repository for Publication model"""
    
    def __init__(self, session: AsyncSession):
        super().__init__(session, Publication)
    
    async def get_by_platform(
        self, platform: str, limit: int = 100
    ) -> list[Publication]:
        """Get publications by platform"""
        result = await self.session.execute(
            select(Publication)
            .where(Publication.platform == platform)
            .order_by(Publication.created_at.desc())
            .limit(limit)
        )
        return list(result.scalars().all())
    
    async def get_by_status(
        self, status: str, limit: int = 100
    ) -> list[Publication]:
        """Get publications by status"""
        result = await self.session.execute(
            select(Publication)
            .where(Publication.status == status)
            .order_by(Publication.created_at.desc())
            .limit(limit)
        )
        return list(result.scalars().all())
    
    async def get_pending(self) -> list[Publication]:
        """Get pending publications"""
        return await self.get_by_status("pending")
    
    async def update_status(
        self,
        id: UUID | str,
        status: str,
        review_status: str | None = None,
        reviewer_note: str | None = None,
    ) -> Publication | None:
        """Update publication status"""
        values = {"status": status}
        if review_status:
            values["review_status"] = review_status
        if reviewer_note:
            values["reviewer_note"] = reviewer_note
        if status == "published":
            values["published_at"] = datetime.utcnow()
        return await self.update(id, **values)


class PipelineMetricRepository(BaseRepository[PipelineMetric]):
    """Repository for PipelineMetric model"""
    
    def __init__(self, session: AsyncSession):
        super().__init__(session, PipelineMetric)
    
    async def record_metric(
        self,
        stage: str,
        status: str,
        duration_ms: int | None = None,
        keyword: str | None = None,
        items_processed: int = 0,
        items_failed: int = 0,
        error_message: str | None = None,
    ) -> PipelineMetric:
        """Record a pipeline metric"""
        return await self.create(
            stage=stage,
            status=status,
            duration_ms=duration_ms,
            keyword=keyword,
            items_processed=items_processed,
            items_failed=items_failed,
            error_message=error_message,
        )
    
    async def get_by_stage(
        self, stage: str, hours: int = 24, limit: int = 100
    ) -> list[PipelineMetric]:
        """Get metrics by stage in time range"""
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        result = await self.session.execute(
            select(PipelineMetric)
            .where(PipelineMetric.stage == stage)
            .where(PipelineMetric.recorded_at >= cutoff)
            .order_by(PipelineMetric.recorded_at.desc())
            .limit(limit)
        )
        return list(result.scalars().all())
    
    async def get_success_rate(self, stage: str, hours: int = 24) -> float:
        """Calculate success rate for a stage"""
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        
        total = await self.session.execute(
            select(func.count())
            .select_from(PipelineMetric)
            .where(PipelineMetric.stage == stage)
            .where(PipelineMetric.recorded_at >= cutoff)
        )
        total_count = total.scalar_one()
        
        if total_count == 0:
            return 0.0
        
        success = await self.session.execute(
            select(func.count())
            .select_from(PipelineMetric)
            .where(PipelineMetric.stage == stage)
            .where(PipelineMetric.status == "success")
            .where(PipelineMetric.recorded_at >= cutoff)
        )
        success_count = success.scalar_one()
        
        return success_count / total_count


class DailyReportRepository(BaseRepository[DailyReport]):
    """Repository for DailyReport model"""
    
    def __init__(self, session: AsyncSession):
        super().__init__(session, DailyReport)
    
    async def get_by_date(self, report_date: datetime) -> DailyReport | None:
        """Get report by date"""
        # Normalize to start of day
        date_start = report_date.replace(hour=0, minute=0, second=0, microsecond=0)
        date_end = date_start + timedelta(days=1)
        
        result = await self.session.execute(
            select(DailyReport)
            .where(DailyReport.report_date >= date_start)
            .where(DailyReport.report_date < date_end)
        )
        return result.scalar_one_or_none()
    
    async def get_recent(self, days: int = 7) -> list[DailyReport]:
        """Get recent reports"""
        cutoff = datetime.utcnow() - timedelta(days=days)
        result = await self.session.execute(
            select(DailyReport)
            .where(DailyReport.report_date >= cutoff)
            .order_by(DailyReport.report_date.desc())
        )
        return list(result.scalars().all())
    
    async def upsert_today(self, **kwargs) -> DailyReport:
        """Insert or update today's report"""
        today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        existing = await self.get_by_date(today)
        
        if existing:
            for key, value in kwargs.items():
                setattr(existing, key, value)
            return existing
        
        return await self.create(report_date=today, **kwargs)


__all__ = [
    "BaseRepository",
    "KeywordRepository",
    "ArticleRepository",
    "AnalysisRepository",
    "PublicationRepository",
    "PipelineMetricRepository",
    "DailyReportRepository",
]