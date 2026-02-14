# src/trendops/database/__init__.py
"""
TrendOps Database Module
Week 6 Day 2: PostgreSQL Schema + SQLAlchemy

Exports:
- Models: Base, Keyword, Article, Analysis, Publication, PipelineMetric, DailyReport
- Connection: DatabaseManager, get_database
- Repository: All repository classes
"""
from trendops.database.connection import (
    DatabaseManager,
    get_database,
    reset_database_manager,
)
from trendops.database.models import (
    Analysis,
    Article,
    Base,
    DailyReport,
    Keyword,
    PipelineMetric,
    Publication,
)
from trendops.database.repository import (
    AnalysisRepository,
    ArticleRepository,
    BaseRepository,
    DailyReportRepository,
    KeywordRepository,
    PipelineMetricRepository,
    PublicationRepository,
)

__all__ = [
    # Models
    "Base",
    "Keyword",
    "Article",
    "Analysis",
    "Publication",
    "PipelineMetric",
    "DailyReport",
    # Connection
    "DatabaseManager",
    "get_database",
    "reset_database_manager",
    # Repositories
    "BaseRepository",
    "KeywordRepository",
    "ArticleRepository",
    "AnalysisRepository",
    "PublicationRepository",
    "PipelineMetricRepository",
    "DailyReportRepository",
]
