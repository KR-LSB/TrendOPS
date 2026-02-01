# tests/test_week6_day2_database.py
"""
TrendOps Week 6 Day 2: Database Tests
PostgreSQL Schema + SQLAlchemy

Test coverage:
- Model definitions and relationships
- Connection manager
- Repository CRUD operations
- Migration script validation
"""
import json
from datetime import datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pytest


def read_file(path: Path) -> str:
    """UTF-8 file reader (Windows compatible)"""
    return path.read_text(encoding="utf-8")


@pytest.fixture
def project_root() -> Path:
    """Project root directory"""
    return Path(__file__).parent.parent


# =========================================================================
# FILE EXISTENCE TESTS
# =========================================================================

class TestFileExistence:
    """Test required files exist"""
    
    def test_models_file_exists(self, project_root: Path):
        assert (project_root / "src" / "trendops" / "database" / "models.py").exists()
    
    def test_connection_file_exists(self, project_root: Path):
        assert (project_root / "src" / "trendops" / "database" / "connection.py").exists()
    
    def test_repository_file_exists(self, project_root: Path):
        assert (project_root / "src" / "trendops" / "database" / "repository.py").exists()
    
    def test_init_file_exists(self, project_root: Path):
        assert (project_root / "src" / "trendops" / "database" / "__init__.py").exists()
    
    def test_alembic_ini_exists(self, project_root: Path):
        assert (project_root / "alembic.ini").exists()
    
    def test_alembic_env_exists(self, project_root: Path):
        assert (project_root / "alembic" / "env.py").exists()
    
    def test_migration_script_exists(self, project_root: Path):
        assert (project_root / "alembic" / "versions" / "001_initial_schema.py").exists()


# =========================================================================
# MODEL DEFINITION TESTS
# =========================================================================

class TestModelDefinitions:
    """Test SQLAlchemy model definitions"""
    
    def test_models_import(self):
        """Test models can be imported"""
        from trendops.database.models import (
            Base, Keyword, Article, Analysis, 
            Publication, PipelineMetric, DailyReport
        )
        assert Base is not None
        assert Keyword is not None
        assert Article is not None
        assert Analysis is not None
        assert Publication is not None
        assert PipelineMetric is not None
        assert DailyReport is not None
    
    def test_keyword_model_columns(self):
        """Test Keyword model has required columns"""
        from trendops.database.models import Keyword
        
        columns = {c.name for c in Keyword.__table__.columns}
        required = {"id", "keyword", "source", "trend_score", "is_active", "metadata"}
        assert required.issubset(columns)
    
    def test_article_model_columns(self):
        """Test Article model has required columns"""
        from trendops.database.models import Article
        
        columns = {c.name for c in Article.__table__.columns}
        required = {"id", "keyword_id", "title", "url", "content", "content_hash"}
        assert required.issubset(columns)
    
    def test_analysis_model_columns(self):
        """Test Analysis model has required columns"""
        from trendops.database.models import Analysis
        
        columns = {c.name for c in Analysis.__table__.columns}
        required = {"id", "keyword_id", "summary", "key_points", "guardrail_passed"}
        assert required.issubset(columns)
    
    def test_publication_model_columns(self):
        """Test Publication model has required columns"""
        from trendops.database.models import Publication
        
        columns = {c.name for c in Publication.__table__.columns}
        required = {"id", "keyword_id", "analysis_id", "platform", "status"}
        assert required.issubset(columns)
    
    def test_pipeline_metric_model_columns(self):
        """Test PipelineMetric model has required columns"""
        from trendops.database.models import PipelineMetric
        
        columns = {c.name for c in PipelineMetric.__table__.columns}
        required = {"id", "stage", "status", "duration_ms", "items_processed"}
        assert required.issubset(columns)
    
    def test_daily_report_model_columns(self):
        """Test DailyReport model has required columns"""
        from trendops.database.models import DailyReport
        
        columns = {c.name for c in DailyReport.__table__.columns}
        required = {"id", "report_date", "trends_detected", "posts_published", "success_rate"}
        assert required.issubset(columns)


# =========================================================================
# MODEL RELATIONSHIP TESTS
# =========================================================================

class TestModelRelationships:
    """Test model relationships"""
    
    def test_keyword_has_articles_relationship(self):
        """Test Keyword -> Article relationship"""
        from trendops.database.models import Keyword
        assert hasattr(Keyword, "articles")
    
    def test_keyword_has_analyses_relationship(self):
        """Test Keyword -> Analysis relationship"""
        from trendops.database.models import Keyword
        assert hasattr(Keyword, "analyses")
    
    def test_keyword_has_publications_relationship(self):
        """Test Keyword -> Publication relationship"""
        from trendops.database.models import Keyword
        assert hasattr(Keyword, "publications")
    
    def test_article_has_keyword_relationship(self):
        """Test Article -> Keyword relationship"""
        from trendops.database.models import Article
        assert hasattr(Article, "keyword")
    
    def test_analysis_has_publications_relationship(self):
        """Test Analysis -> Publication relationship"""
        from trendops.database.models import Analysis
        assert hasattr(Analysis, "publications")
    
    def test_publication_has_analysis_relationship(self):
        """Test Publication -> Analysis relationship"""
        from trendops.database.models import Publication
        assert hasattr(Publication, "analysis")


# =========================================================================
# MODEL INDEX TESTS
# =========================================================================

class TestModelIndexes:
    """Test model indexes are defined"""
    
    def test_keyword_has_indexes(self):
        """Test Keyword model has indexes"""
        from trendops.database.models import Keyword
        indexes = {idx.name for idx in Keyword.__table__.indexes}
        assert "idx_keyword_source" in indexes
        assert "idx_keyword_active_score" in indexes
    
    def test_article_has_indexes(self):
        """Test Article model has indexes"""
        from trendops.database.models import Article
        indexes = {idx.name for idx in Article.__table__.indexes}
        assert "idx_article_keyword_date" in indexes
        assert "idx_article_content_hash" in indexes
    
    def test_analysis_has_indexes(self):
        """Test Analysis model has indexes"""
        from trendops.database.models import Analysis
        indexes = {idx.name for idx in Analysis.__table__.indexes}
        assert "idx_analysis_keyword_date" in indexes
        assert "idx_analysis_guardrail" in indexes
    
    def test_publication_has_indexes(self):
        """Test Publication model has indexes"""
        from trendops.database.models import Publication
        indexes = {idx.name for idx in Publication.__table__.indexes}
        assert "idx_publication_status" in indexes


# =========================================================================
# CONNECTION MANAGER TESTS
# =========================================================================

class TestConnectionManager:
    """Test DatabaseManager"""
    
    def test_database_manager_import(self):
        """Test DatabaseManager can be imported"""
        from trendops.database.connection import DatabaseManager
        assert DatabaseManager is not None
    
    def test_database_manager_init(self):
        """Test DatabaseManager initialization"""
        from trendops.database.connection import DatabaseManager
        db = DatabaseManager("postgresql+asyncpg://test:test@localhost:5432/test")
        assert db.database_url == "postgresql+asyncpg://test:test@localhost:5432/test"
        assert db.engine is None
        assert db.session_factory is None
    
    def test_database_manager_url_conversion(self):
        """Test URL is converted to async driver"""
        from trendops.database.connection import DatabaseManager
        db = DatabaseManager("postgresql://test:test@localhost:5432/test")
        assert "asyncpg" in db.database_url
    
    def test_get_database_singleton(self):
        """Test get_database returns singleton"""
        from trendops.database.connection import get_database, reset_database_manager
        reset_database_manager()
        
        db1 = get_database()
        db2 = get_database()
        assert db1 is db2
        
        reset_database_manager()
    
    def test_is_connected_property(self):
        """Test is_connected property"""
        from trendops.database.connection import DatabaseManager
        db = DatabaseManager("postgresql+asyncpg://test:test@localhost:5432/test")
        assert db.is_connected is False


# =========================================================================
# REPOSITORY TESTS
# =========================================================================

class TestRepositoryDefinitions:
    """Test repository class definitions"""
    
    def test_repositories_import(self):
        """Test all repositories can be imported"""
        from trendops.database.repository import (
            BaseRepository,
            KeywordRepository,
            ArticleRepository,
            AnalysisRepository,
            PublicationRepository,
            PipelineMetricRepository,
            DailyReportRepository,
        )
        assert BaseRepository is not None
        assert KeywordRepository is not None
        assert ArticleRepository is not None
        assert AnalysisRepository is not None
        assert PublicationRepository is not None
        assert PipelineMetricRepository is not None
        assert DailyReportRepository is not None
    
    def test_keyword_repository_methods(self):
        """Test KeywordRepository has required methods"""
        from trendops.database.repository import KeywordRepository
        methods = dir(KeywordRepository)
        assert "get_by_keyword" in methods
        assert "get_active_keywords" in methods
        assert "upsert" in methods
        assert "deactivate_old" in methods
    
    def test_article_repository_methods(self):
        """Test ArticleRepository has required methods"""
        from trendops.database.repository import ArticleRepository
        methods = dir(ArticleRepository)
        assert "get_by_url" in methods
        assert "get_by_hash" in methods
        assert "get_by_keyword" in methods
        assert "exists_by_url" in methods
    
    def test_publication_repository_methods(self):
        """Test PublicationRepository has required methods"""
        from trendops.database.repository import PublicationRepository
        methods = dir(PublicationRepository)
        assert "get_by_platform" in methods
        assert "get_by_status" in methods
        assert "get_pending" in methods
        assert "update_status" in methods


# =========================================================================
# MIGRATION SCRIPT TESTS
# =========================================================================

class TestMigrationScript:
    """Test Alembic migration script"""
    
    def test_migration_has_upgrade(self, project_root: Path):
        """Test migration has upgrade function"""
        content = read_file(project_root / "alembic" / "versions" / "001_initial_schema.py")
        assert "def upgrade()" in content
    
    def test_migration_has_downgrade(self, project_root: Path):
        """Test migration has downgrade function"""
        content = read_file(project_root / "alembic" / "versions" / "001_initial_schema.py")
        assert "def downgrade()" in content
    
    def test_migration_creates_keywords_table(self, project_root: Path):
        """Test migration creates keywords table"""
        content = read_file(project_root / "alembic" / "versions" / "001_initial_schema.py")
        assert "'keywords'" in content
        assert "create_table" in content
    
    def test_migration_creates_articles_table(self, project_root: Path):
        """Test migration creates articles table"""
        content = read_file(project_root / "alembic" / "versions" / "001_initial_schema.py")
        assert "'articles'" in content
    
    def test_migration_creates_analyses_table(self, project_root: Path):
        """Test migration creates analyses table"""
        content = read_file(project_root / "alembic" / "versions" / "001_initial_schema.py")
        assert "'analyses'" in content
    
    def test_migration_creates_publications_table(self, project_root: Path):
        """Test migration creates publications table"""
        content = read_file(project_root / "alembic" / "versions" / "001_initial_schema.py")
        assert "'publications'" in content
    
    def test_migration_creates_indexes(self, project_root: Path):
        """Test migration creates indexes"""
        content = read_file(project_root / "alembic" / "versions" / "001_initial_schema.py")
        assert "create_index" in content
        assert "idx_keyword" in content
        assert "idx_article" in content


# =========================================================================
# ALEMBIC CONFIG TESTS
# =========================================================================

class TestAlembicConfig:
    """Test Alembic configuration"""
    
    def test_alembic_ini_has_script_location(self, project_root: Path):
        """Test alembic.ini has script_location"""
        content = read_file(project_root / "alembic.ini")
        assert "script_location" in content
    
    def test_alembic_env_imports_models(self, project_root: Path):
        """Test alembic/env.py imports models"""
        content = read_file(project_root / "alembic" / "env.py")
        assert "from trendops.database.models import Base" in content
    
    def test_alembic_env_has_target_metadata(self, project_root: Path):
        """Test alembic/env.py sets target_metadata"""
        content = read_file(project_root / "alembic" / "env.py")
        assert "target_metadata" in content


# =========================================================================
# SUMMARY TEST
# =========================================================================

class TestDay2Summary:
    """Day 2 summary test"""
    
    def test_all_day2_files_present(self, project_root: Path):
        """Test all Day 2 files are present"""
        required = [
            "src/trendops/database/models.py",
            "src/trendops/database/connection.py",
            "src/trendops/database/repository.py",
            "src/trendops/database/__init__.py",
            "alembic.ini",
            "alembic/env.py",
            "alembic/versions/001_initial_schema.py",
        ]
        missing = [f for f in required if not (project_root / f).exists()]
        assert len(missing) == 0, f"Missing: {missing}"
    
    def test_all_models_defined(self):
        """Test all 6 models are defined"""
        from trendops.database.models import (
            Keyword, Article, Analysis,
            Publication, PipelineMetric, DailyReport
        )
        models = [Keyword, Article, Analysis, Publication, PipelineMetric, DailyReport]
        assert len(models) == 6
        for model in models:
            assert hasattr(model, "__tablename__")