# src/trendops/database/models.py
"""
TrendOps SQLAlchemy ORM 모델

Week 6 Day 2: PostgreSQL 스키마 구현
"""
from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """SQLAlchemy 베이스 클래스"""

    pass


class Keyword(Base):
    """트렌드 키워드 테이블"""

    __tablename__ = "keywords"

    id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )
    keyword: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    source: Mapped[str] = mapped_column(String(50), nullable=False)
    trend_score: Mapped[float] = mapped_column(Float, default=0.0)
    first_seen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )
    last_seen_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    metadata_: Mapped[dict[str, Any]] = mapped_column(
        "metadata",
        JSONB,
        default=dict,
    )

    # Relationships
    articles: Mapped[list[Article]] = relationship(
        "Article",
        back_populates="keyword",
        lazy="selectin",
    )
    analyses: Mapped[list[Analysis]] = relationship(
        "Analysis",
        back_populates="keyword",
        lazy="selectin",
    )
    publications: Mapped[list[Publication]] = relationship(
        "Publication",
        back_populates="keyword",
        lazy="selectin",
    )

    def to_dict(self) -> dict[str, Any]:
        """딕셔너리 변환"""
        return {
            "id": str(self.id),
            "keyword": self.keyword,
            "source": self.source,
            "trend_score": self.trend_score,
            "is_active": self.is_active,
            "first_seen_at": self.first_seen_at.isoformat() if self.first_seen_at else None,
            "last_seen_at": self.last_seen_at.isoformat() if self.last_seen_at else None,
        }


class Article(Base):
    """수집된 기사 테이블"""

    __tablename__ = "articles"

    id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )
    keyword_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("keywords.id"),
        nullable=False,
        index=True,
    )
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    url: Mapped[str] = mapped_column(String(2000), nullable=False, unique=True)
    source: Mapped[str | None] = mapped_column(String(100))
    content: Mapped[str | None] = mapped_column(Text)
    content_hash: Mapped[str | None] = mapped_column(String(64), index=True)
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    collected_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )
    embedding_id: Mapped[str | None] = mapped_column(String(100))
    metadata_: Mapped[dict[str, Any]] = mapped_column(
        "metadata",
        JSONB,
        default=dict,
    )

    # Relationships
    keyword: Mapped[Keyword] = relationship("Keyword", back_populates="articles")

    def to_dict(self) -> dict[str, Any]:
        """딕셔너리 변환"""
        return {
            "id": str(self.id),
            "keyword_id": str(self.keyword_id),
            "title": self.title,
            "url": self.url,
            "source": self.source,
            "published_at": self.published_at.isoformat() if self.published_at else None,
            "collected_at": self.collected_at.isoformat() if self.collected_at else None,
        }


class Analysis(Base):
    """LLM 분석 결과 테이블"""

    __tablename__ = "analyses"

    id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )
    keyword_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("keywords.id"),
        nullable=False,
        index=True,
    )
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    key_points: Mapped[list[str]] = mapped_column(JSONB, default=list)
    sentiment_ratio: Mapped[dict[str, float] | None] = mapped_column(JSONB)
    guardrail_passed: Mapped[bool] = mapped_column(Boolean, default=True)
    guardrail_issues: Mapped[list[dict]] = mapped_column(JSONB, default=list)
    revision_count: Mapped[int] = mapped_column(Integer, default=0)
    model_name: Mapped[str | None] = mapped_column(String(100))
    processing_time_ms: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )

    # Relationships
    keyword: Mapped[Keyword] = relationship("Keyword", back_populates="analyses")
    publications: Mapped[list[Publication]] = relationship(
        "Publication",
        back_populates="analysis",
        lazy="selectin",
    )

    def to_dict(self) -> dict[str, Any]:
        """딕셔너리 변환"""
        return {
            "id": str(self.id),
            "keyword_id": str(self.keyword_id),
            "summary": self.summary,
            "key_points": self.key_points,
            "sentiment_ratio": self.sentiment_ratio,
            "guardrail_passed": self.guardrail_passed,
            "revision_count": self.revision_count,
            "model_name": self.model_name,
            "processing_time_ms": self.processing_time_ms,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class Publication(Base):
    """SNS 발행 기록 테이블"""

    __tablename__ = "publications"

    id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )
    keyword_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("keywords.id"),
        nullable=False,
    )
    analysis_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("analyses.id"),
        nullable=False,
    )
    platform: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    post_id: Mapped[str | None] = mapped_column(String(100))
    post_url: Mapped[str | None] = mapped_column(String(500))
    image_path: Mapped[str | None] = mapped_column(String(500))
    caption: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(50), default="pending", index=True)
    review_status: Mapped[str | None] = mapped_column(String(50))
    reviewer_note: Mapped[str | None] = mapped_column(Text)
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )
    metadata_: Mapped[dict[str, Any]] = mapped_column(
        "metadata",
        JSONB,
        default=dict,
    )

    # Relationships
    keyword: Mapped[Keyword] = relationship("Keyword", back_populates="publications")
    analysis: Mapped[Analysis] = relationship("Analysis", back_populates="publications")

    def to_dict(self) -> dict[str, Any]:
        """딕셔너리 변환"""
        return {
            "id": str(self.id),
            "keyword_id": str(self.keyword_id),
            "analysis_id": str(self.analysis_id),
            "platform": self.platform,
            "post_id": self.post_id,
            "post_url": self.post_url,
            "status": self.status,
            "review_status": self.review_status,
            "published_at": self.published_at.isoformat() if self.published_at else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class PipelineMetric(Base):
    """파이프라인 메트릭 테이블"""

    __tablename__ = "pipeline_metrics"

    id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )
    stage: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    keyword: Mapped[str | None] = mapped_column(String(255))
    status: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    duration_ms: Mapped[int | None] = mapped_column(Integer)
    error_message: Mapped[str | None] = mapped_column(Text)
    items_processed: Mapped[int] = mapped_column(Integer, default=0)
    items_failed: Mapped[int] = mapped_column(Integer, default=0)
    recorded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        index=True,
    )
    metadata_: Mapped[dict[str, Any]] = mapped_column(
        "metadata",
        JSONB,
        default=dict,
    )

    def to_dict(self) -> dict[str, Any]:
        """딕셔너리 변환"""
        return {
            "id": str(self.id),
            "stage": self.stage,
            "keyword": self.keyword,
            "status": self.status,
            "duration_ms": self.duration_ms,
            "items_processed": self.items_processed,
            "items_failed": self.items_failed,
            "recorded_at": self.recorded_at.isoformat() if self.recorded_at else None,
        }


class DailyReport(Base):
    """일별 리포트 테이블"""

    __tablename__ = "daily_reports"

    id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )
    report_date: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        unique=True,
        index=True,
    )
    trends_detected: Mapped[int] = mapped_column(Integer, default=0)
    articles_collected: Mapped[int] = mapped_column(Integer, default=0)
    articles_analyzed: Mapped[int] = mapped_column(Integer, default=0)
    images_generated: Mapped[int] = mapped_column(Integer, default=0)
    posts_published: Mapped[int] = mapped_column(Integer, default=0)
    posts_rejected: Mapped[int] = mapped_column(Integer, default=0)
    errors_count: Mapped[int] = mapped_column(Integer, default=0)
    success_rate: Mapped[float | None] = mapped_column(Float)
    avg_latency_ms: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )

    def to_dict(self) -> dict[str, Any]:
        """딕셔너리 변환"""
        return {
            "id": str(self.id),
            "report_date": self.report_date.isoformat() if self.report_date else None,
            "trends_detected": self.trends_detected,
            "articles_collected": self.articles_collected,
            "articles_analyzed": self.articles_analyzed,
            "images_generated": self.images_generated,
            "posts_published": self.posts_published,
            "posts_rejected": self.posts_rejected,
            "errors_count": self.errors_count,
            "success_rate": self.success_rate,
            "avg_latency_ms": self.avg_latency_ms,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
