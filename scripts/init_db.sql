-- scripts/init_db.sql
-- TrendOps Database Initialization Script
-- Week 6: Polish & Portfolio
--
-- 이 스크립트는 Docker 컨테이너 최초 시작 시 자동 실행됩니다.
-- 테이블 생성은 Day 2에서 SQLAlchemy/Alembic으로 처리합니다.

-- =========================================================================
-- EXTENSIONS
-- =========================================================================

-- UUID 생성 함수
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- 텍스트 유사도 검색 (트라이그램)
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- =========================================================================
-- INITIAL SETUP
-- =========================================================================

-- 기본 스키마 확인
SELECT current_database(), current_user, version();

-- 접속 권한 확인
GRANT ALL PRIVILEGES ON DATABASE trendops TO trendops;

-- =========================================================================
-- HEALTH CHECK TABLE (선택적)
-- =========================================================================

-- 데이터베이스 연결 확인용 테이블
CREATE TABLE IF NOT EXISTS _health_check (
    id SERIAL PRIMARY KEY,
    checked_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    status VARCHAR(50) DEFAULT 'healthy'
);

-- 초기 레코드 삽입
INSERT INTO _health_check (status) VALUES ('initialized');

-- =========================================================================
-- SCHEMA NOTES
-- =========================================================================

-- 실제 테이블 스키마는 다음 파일에서 정의됩니다:
-- - src/trendops/database/models.py (SQLAlchemy ORM)
-- - alembic/versions/001_initial_schema.py (마이그레이션)
--
-- 마이그레이션 실행:
--   docker compose exec api alembic upgrade head

COMMENT ON DATABASE trendops IS 'TrendOps: Real-time trend analysis and SNS automation pipeline';