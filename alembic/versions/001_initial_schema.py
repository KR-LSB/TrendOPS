"""Initial schema

Revision ID: 001
Revises: 
Create Date: 2025-01-26

TrendOps Database Schema:
- keywords: Trend keywords from Google/Naver
- articles: Collected news articles
- analyses: LLM analysis results
- publications: SNS publication records
- pipeline_metrics: Performance metrics
- daily_reports: Daily statistics
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID, JSONB

# revision identifiers, used by Alembic.
revision: str = '001'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # === Keywords Table ===
    op.create_table(
        'keywords',
        sa.Column('id', UUID(as_uuid=True), primary_key=True),
        sa.Column('keyword', sa.String(255), nullable=False),
        sa.Column('source', sa.String(50), nullable=False),
        sa.Column('trend_score', sa.Float, server_default='0.0'),
        sa.Column('first_seen_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('last_seen_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('is_active', sa.Boolean, server_default='true'),
        sa.Column('metadata', JSONB, server_default='{}'),
    )
    op.create_index('idx_keyword_keyword', 'keywords', ['keyword'])
    op.create_index('idx_keyword_source', 'keywords', ['keyword', 'source'])
    op.create_index('idx_keyword_active_score', 'keywords', ['is_active', 'trend_score'])
    
    # === Articles Table ===
    op.create_table(
        'articles',
        sa.Column('id', UUID(as_uuid=True), primary_key=True),
        sa.Column('keyword_id', UUID(as_uuid=True), sa.ForeignKey('keywords.id', ondelete='CASCADE'), nullable=False),
        sa.Column('title', sa.String(500), nullable=False),
        sa.Column('url', sa.String(2000), nullable=False, unique=True),
        sa.Column('source', sa.String(100), nullable=True),
        sa.Column('content', sa.Text, nullable=True),
        sa.Column('content_hash', sa.String(64), nullable=True),
        sa.Column('published_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('collected_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('embedding_id', sa.String(100), nullable=True),
        sa.Column('metadata', JSONB, server_default='{}'),
    )
    op.create_index('idx_article_keyword_date', 'articles', ['keyword_id', 'collected_at'])
    op.create_index('idx_article_content_hash', 'articles', ['content_hash'])
    op.create_index('idx_article_url', 'articles', ['url'])
    
    # === Analyses Table ===
    op.create_table(
        'analyses',
        sa.Column('id', UUID(as_uuid=True), primary_key=True),
        sa.Column('keyword_id', UUID(as_uuid=True), sa.ForeignKey('keywords.id', ondelete='CASCADE'), nullable=False),
        sa.Column('summary', sa.Text, nullable=False),
        sa.Column('key_points', JSONB, server_default='[]'),
        sa.Column('sentiment_ratio', JSONB, nullable=True),
        sa.Column('guardrail_passed', sa.Boolean, server_default='true'),
        sa.Column('guardrail_issues', JSONB, server_default='[]'),
        sa.Column('revision_count', sa.Integer, server_default='0'),
        sa.Column('model_name', sa.String(100), nullable=True),
        sa.Column('processing_time_ms', sa.Integer, nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index('idx_analysis_keyword_date', 'analyses', ['keyword_id', 'created_at'])
    op.create_index('idx_analysis_guardrail', 'analyses', ['guardrail_passed'])
    
    # === Publications Table ===
    op.create_table(
        'publications',
        sa.Column('id', UUID(as_uuid=True), primary_key=True),
        sa.Column('keyword_id', UUID(as_uuid=True), sa.ForeignKey('keywords.id', ondelete='CASCADE'), nullable=False),
        sa.Column('analysis_id', UUID(as_uuid=True), sa.ForeignKey('analyses.id', ondelete='CASCADE'), nullable=False),
        sa.Column('platform', sa.String(50), nullable=False),
        sa.Column('post_id', sa.String(100), nullable=True),
        sa.Column('post_url', sa.String(500), nullable=True),
        sa.Column('image_path', sa.String(500), nullable=True),
        sa.Column('caption', sa.Text, nullable=True),
        sa.Column('status', sa.String(50), server_default='pending'),
        sa.Column('review_status', sa.String(50), nullable=True),
        sa.Column('reviewer_note', sa.Text, nullable=True),
        sa.Column('published_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('metadata', JSONB, server_default='{}'),
    )
    op.create_index('idx_publication_platform_date', 'publications', ['platform', 'created_at'])
    op.create_index('idx_publication_status', 'publications', ['status'])
    
    # === Pipeline Metrics Table ===
    op.create_table(
        'pipeline_metrics',
        sa.Column('id', UUID(as_uuid=True), primary_key=True),
        sa.Column('stage', sa.String(50), nullable=False),
        sa.Column('keyword', sa.String(255), nullable=True),
        sa.Column('status', sa.String(50), nullable=False),
        sa.Column('duration_ms', sa.Integer, nullable=True),
        sa.Column('error_message', sa.Text, nullable=True),
        sa.Column('items_processed', sa.Integer, server_default='0'),
        sa.Column('items_failed', sa.Integer, server_default='0'),
        sa.Column('recorded_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('metadata', JSONB, server_default='{}'),
    )
    op.create_index('idx_metric_stage_date', 'pipeline_metrics', ['stage', 'recorded_at'])
    op.create_index('idx_metric_status', 'pipeline_metrics', ['status'])
    
    # === Daily Reports Table ===
    op.create_table(
        'daily_reports',
        sa.Column('id', UUID(as_uuid=True), primary_key=True),
        sa.Column('report_date', sa.DateTime(timezone=True), nullable=False, unique=True),
        sa.Column('trends_detected', sa.Integer, server_default='0'),
        sa.Column('articles_collected', sa.Integer, server_default='0'),
        sa.Column('articles_analyzed', sa.Integer, server_default='0'),
        sa.Column('images_generated', sa.Integer, server_default='0'),
        sa.Column('posts_published', sa.Integer, server_default='0'),
        sa.Column('posts_rejected', sa.Integer, server_default='0'),
        sa.Column('errors_count', sa.Integer, server_default='0'),
        sa.Column('success_rate', sa.Float, nullable=True),
        sa.Column('avg_latency_ms', sa.Integer, nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index('idx_report_date', 'daily_reports', ['report_date'])


def downgrade() -> None:
    op.drop_table('daily_reports')
    op.drop_table('pipeline_metrics')
    op.drop_table('publications')
    op.drop_table('analyses')
    op.drop_table('articles')
    op.drop_table('keywords')