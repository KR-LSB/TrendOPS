# src/trendops/api/metrics.py
"""
TrendOps Prometheus 메트릭 정의

Week 6 Day 3-4: 모니터링 메트릭 + FastAPI 계측
"""
from prometheus_client import Counter, Gauge, Histogram, Info

# =============================================================================
# Pipeline Metrics
# =============================================================================

PIPELINE_RUNS_TOTAL = Counter(
    "trendops_pipeline_runs_total", "Total number of pipeline runs", ["stage", "status"]
)

PIPELINE_DURATION_SECONDS = Histogram(
    "trendops_pipeline_duration_seconds",
    "Pipeline stage duration in seconds",
    ["stage"],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0],
)

PIPELINE_ITEMS_PROCESSED = Counter(
    "trendops_items_processed_total", "Total items processed by pipeline", ["stage"]
)


# =============================================================================
# Collection Metrics
# =============================================================================

ARTICLES_COLLECTED = Counter(
    "trendops_articles_collected_total", "Total articles collected", ["source"]
)

ARTICLES_DEDUPLICATED = Counter(
    "trendops_articles_deduplicated_total", "Total articles removed by deduplication"
)


# =============================================================================
# Analysis Metrics
# =============================================================================

ANALYSIS_REQUESTS = Counter(
    "trendops_analysis_requests_total", "Total analysis requests", ["model", "status"]
)

GUARDRAIL_CHECKS = Counter(
    "trendops_guardrail_checks_total", "Total guardrail checks", ["check_type", "result"]
)

LLM_TOKENS_USED = Counter(
    "trendops_llm_tokens_total",
    "Total LLM tokens used",
    ["model", "type"],  # type: input, output
)


# =============================================================================
# Publication Metrics
# =============================================================================

PUBLICATIONS_TOTAL = Counter(
    "trendops_publications_total", "Total publication attempts", ["platform", "status"]
)

REVIEW_DECISIONS = Counter(
    "trendops_review_decisions_total",
    "Total review decisions",
    ["decision"],  # approved, rejected, modified
)


# =============================================================================
# System Metrics
# =============================================================================

ACTIVE_KEYWORDS = Gauge("trendops_active_keywords", "Number of active keywords being tracked")

QUEUE_SIZE = Gauge("trendops_queue_size", "Current size of job queue", ["queue"])

DB_CONNECTIONS = Gauge(
    "trendops_db_connections",
    "Number of active database connections",
    ["db_type"],  # postgres, redis, chromadb
)


# =============================================================================
# API Metrics
# =============================================================================

API_REQUESTS = Counter(
    "trendops_api_requests_total", "Total API requests", ["method", "endpoint", "status"]
)

API_LATENCY = Histogram(
    "trendops_api_latency_seconds",
    "API request latency in seconds",
    ["method", "endpoint"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)


# =============================================================================
# Info Metric
# =============================================================================

APP_INFO = Info("trendops_app", "Application information")


# =============================================================================
# Initialization
# =============================================================================


def init_metrics() -> None:
    """메트릭 초기화"""
    APP_INFO.info(
        {
            "version": "1.0.0",
            "environment": "production",
            "python_version": "3.10",
        }
    )


def record_pipeline_run(stage: str, status: str, duration: float) -> None:
    """파이프라인 실행 기록"""
    PIPELINE_RUNS_TOTAL.labels(stage=stage, status=status).inc()
    PIPELINE_DURATION_SECONDS.labels(stage=stage).observe(duration)


def record_collection(source: str, count: int) -> None:
    """수집 결과 기록"""
    ARTICLES_COLLECTED.labels(source=source).inc(count)


def record_analysis(model: str, status: str) -> None:
    """분석 요청 기록"""
    ANALYSIS_REQUESTS.labels(model=model, status=status).inc()


def record_publication(platform: str, status: str) -> None:
    """발행 결과 기록"""
    PUBLICATIONS_TOTAL.labels(platform=platform, status=status).inc()


def update_queue_size(queue_name: str, size: int) -> None:
    """큐 크기 업데이트"""
    QUEUE_SIZE.labels(queue=queue_name).set(size)


def update_active_keywords(count: int) -> None:
    """활성 키워드 수 업데이트"""
    ACTIVE_KEYWORDS.set(count)
