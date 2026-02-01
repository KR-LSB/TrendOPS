"""
TrendOps Prometheus Metrics Module
Week 6 Day 3: Grafana Dashboard Configuration

대시보드에서 사용하는 모든 Prometheus 메트릭 정의
블루프린트 4.3 섹션 기반
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, Generator, TypeVar

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    Info,
    generate_latest,
    CONTENT_TYPE_LATEST,
)

# Type variable for generic function decoration
F = TypeVar("F", bound=Callable[..., Any])

# =============================================================================
# Registry
# =============================================================================

# 기본 레지스트리 사용 (multiprocess 환경에서는 별도 설정 필요)
REGISTRY = CollectorRegistry(auto_describe=True)


# =============================================================================
# Application Info
# =============================================================================

APP_INFO = Info(
    name="trendops",
    documentation="TrendOps application information",
    registry=REGISTRY,
)
APP_INFO.info({
    "version": "1.0.0",
    "environment": "production",
    "component": "pipeline",
})


# =============================================================================
# Counter Metrics (누적 카운터)
# =============================================================================

# Keywords Detection
KEYWORDS_DETECTED = Counter(
    name="trendops_keywords_detected_total",
    documentation="Total number of keywords detected from trend sources",
    labelnames=["source"],  # google, naver
    registry=REGISTRY,
)

# Documents Collection
DOCUMENTS_COLLECTED = Counter(
    name="trendops_documents_collected_total",
    documentation="Total number of documents collected",
    labelnames=["source"],  # youtube, news_rss, community
    registry=REGISTRY,
)

# Posts Published
POSTS_PUBLISHED = Counter(
    name="trendops_posts_published_total",
    documentation="Total number of posts published to social media",
    labelnames=["platform"],  # instagram, threads
    registry=REGISTRY,
)

# Errors
ERRORS = Counter(
    name="trendops_errors_total",
    documentation="Total number of errors",
    labelnames=["stage", "error_type"],  # trigger, collect, analyze, publish
    registry=REGISTRY,
)

# Pipeline Stage Counters
TRIGGER_TOTAL = Counter(
    name="trendops_trigger_total",
    documentation="Total trigger operations",
    registry=REGISTRY,
)

TRIGGER_SUCCESS = Counter(
    name="trendops_trigger_success_total",
    documentation="Successful trigger operations",
    registry=REGISTRY,
)

COLLECT_TOTAL = Counter(
    name="trendops_collect_total",
    documentation="Total collect operations",
    registry=REGISTRY,
)

COLLECT_SUCCESS = Counter(
    name="trendops_collect_success_total",
    documentation="Successful collect operations",
    registry=REGISTRY,
)

ANALYZE_TOTAL = Counter(
    name="trendops_analyze_total",
    documentation="Total analyze operations",
    registry=REGISTRY,
)

ANALYZE_SUCCESS = Counter(
    name="trendops_analyze_success_total",
    documentation="Successful analyze operations",
    registry=REGISTRY,
)

PUBLISH_TOTAL = Counter(
    name="trendops_publish_total",
    documentation="Total publish operations",
    registry=REGISTRY,
)

PUBLISH_SUCCESS = Counter(
    name="trendops_publish_success_total",
    documentation="Successful publish operations",
    registry=REGISTRY,
)

# Guardrail Metrics
GUARDRAIL_PASSED = Counter(
    name="trendops_guardrail_passed_total",
    documentation="Content passed guardrail check",
    registry=REGISTRY,
)

GUARDRAIL_REJECTED = Counter(
    name="trendops_guardrail_rejected_total",
    documentation="Content rejected by guardrail",
    registry=REGISTRY,
)

GUARDRAIL_REVISED = Counter(
    name="trendops_guardrail_revised_total",
    documentation="Content auto-revised by guardrail",
    registry=REGISTRY,
)


# =============================================================================
# Histogram Metrics (지연 시간 분포)
# =============================================================================

# 기본 버킷 정의 (초 단위)
LATENCY_BUCKETS = (0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, float("inf"))
LLM_LATENCY_BUCKETS = (1.0, 2.5, 5.0, 10.0, 15.0, 20.0, 30.0, 45.0, 60.0, 90.0, 120.0, float("inf"))

# Pipeline Stage Durations
TRIGGER_DURATION = Histogram(
    name="trendops_trigger_duration_seconds",
    documentation="Duration of trigger operations",
    buckets=LATENCY_BUCKETS,
    registry=REGISTRY,
)

COLLECT_DURATION = Histogram(
    name="trendops_collect_duration_seconds",
    documentation="Duration of collect operations",
    buckets=LATENCY_BUCKETS,
    registry=REGISTRY,
)

ANALYZE_DURATION = Histogram(
    name="trendops_analyze_duration_seconds",
    documentation="Duration of analyze operations",
    buckets=LLM_LATENCY_BUCKETS,
    registry=REGISTRY,
)

PUBLISH_DURATION = Histogram(
    name="trendops_publish_duration_seconds",
    documentation="Duration of publish operations",
    buckets=LATENCY_BUCKETS,
    registry=REGISTRY,
)

# LLM Inference Duration
LLM_INFERENCE_DURATION = Histogram(
    name="trendops_llm_inference_duration_seconds",
    documentation="Duration of LLM inference calls",
    labelnames=["model", "operation"],  # qwen2.5:7b, summarize/sentiment/guardrail
    buckets=LLM_LATENCY_BUCKETS,
    registry=REGISTRY,
)

# Embedding Duration
EMBEDDING_DURATION = Histogram(
    name="trendops_embedding_duration_seconds",
    documentation="Duration of embedding generation",
    labelnames=["model"],  # bge-m3-ko
    buckets=LATENCY_BUCKETS,
    registry=REGISTRY,
)


# =============================================================================
# Gauge Metrics (현재 상태)
# =============================================================================

# GPU Memory Usage
VLLM_GPU_MEMORY = Gauge(
    name="trendops_vllm_gpu_memory_used_gb",
    documentation="vLLM GPU memory usage in GB",
    registry=REGISTRY,
)

EMBEDDING_MEMORY = Gauge(
    name="trendops_embedding_memory_used_gb",
    documentation="Embedding model memory usage in GB",
    registry=REGISTRY,
)

# Redis Queue Sizes
REDIS_QUEUE_SIZE = Gauge(
    name="trendops_redis_queue_size",
    documentation="Number of items in Redis queues",
    labelnames=["queue"],  # pending, processing, completed, failed
    registry=REGISTRY,
)

REDIS_MEMORY_USAGE = Gauge(
    name="trendops_redis_memory_usage_percent",
    documentation="Redis memory usage percentage",
    registry=REGISTRY,
)

# Deduplication Ratio
DEDUPLICATION_RATIO = Gauge(
    name="trendops_deduplication_ratio",
    documentation="Ratio of deduplicated documents (0-1)",
    registry=REGISTRY,
)

# Active Jobs
ACTIVE_JOBS = Gauge(
    name="trendops_active_jobs",
    documentation="Number of currently active jobs",
    labelnames=["stage"],
    registry=REGISTRY,
)

# ChromaDB Metrics
CHROMADB_COLLECTION_SIZE = Gauge(
    name="trendops_chromadb_collection_size",
    documentation="Number of documents in ChromaDB collection",
    labelnames=["collection"],
    registry=REGISTRY,
)


# =============================================================================
# Utility Functions & Decorators
# =============================================================================

@contextmanager
def track_duration(histogram: Histogram) -> Generator[None, None, None]:
    """
    컨텍스트 매니저로 실행 시간 측정
    
    Usage:
        with track_duration(TRIGGER_DURATION):
            await trigger_pipeline()
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        duration = time.perf_counter() - start
        histogram.observe(duration)


def track_operation(
    stage: str,
    duration_histogram: Histogram,
    total_counter: Counter,
    success_counter: Counter,
) -> Callable[[F], F]:
    """
    파이프라인 스테이지 데코레이터
    - 실행 시간 측정
    - 총 호출 수 카운트
    - 성공/실패 카운트
    
    Usage:
        @track_operation("trigger", TRIGGER_DURATION, TRIGGER_TOTAL, TRIGGER_SUCCESS)
        async def trigger_pipeline():
            ...
    """
    def decorator(func: F) -> F:
        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            total_counter.inc()
            ACTIVE_JOBS.labels(stage=stage).inc()
            start = time.perf_counter()
            try:
                result = await func(*args, **kwargs)
                success_counter.inc()
                return result
            except Exception as e:
                ERRORS.labels(stage=stage, error_type=type(e).__name__).inc()
                raise
            finally:
                duration = time.perf_counter() - start
                duration_histogram.observe(duration)
                ACTIVE_JOBS.labels(stage=stage).dec()
        
        @wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            total_counter.inc()
            ACTIVE_JOBS.labels(stage=stage).inc()
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                success_counter.inc()
                return result
            except Exception as e:
                ERRORS.labels(stage=stage, error_type=type(e).__name__).inc()
                raise
            finally:
                duration = time.perf_counter() - start
                duration_histogram.observe(duration)
                ACTIVE_JOBS.labels(stage=stage).dec()
        
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        return sync_wrapper  # type: ignore
    
    return decorator


def track_llm_inference(model: str, operation: str) -> Callable[[F], F]:
    """
    LLM 추론 시간 측정 데코레이터
    
    Usage:
        @track_llm_inference("qwen2.5:7b", "summarize")
        async def generate_summary(text: str) -> str:
            ...
    """
    def decorator(func: F) -> F:
        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            try:
                return await func(*args, **kwargs)
            finally:
                duration = time.perf_counter() - start
                LLM_INFERENCE_DURATION.labels(model=model, operation=operation).observe(duration)
        
        @wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                duration = time.perf_counter() - start
                LLM_INFERENCE_DURATION.labels(model=model, operation=operation).observe(duration)
        
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        return sync_wrapper  # type: ignore
    
    return decorator


# =============================================================================
# Metrics Collector Class
# =============================================================================

class TrendOpsMetrics:
    """
    TrendOps 메트릭 수집 통합 클래스
    
    Usage:
        metrics = TrendOpsMetrics()
        
        # 키워드 감지
        metrics.record_keyword_detected("google")
        
        # 문서 수집
        metrics.record_document_collected("youtube", count=10)
        
        # 파이프라인 지연 시간
        with metrics.track_stage("trigger"):
            await trigger_pipeline()
    """
    
    def record_keyword_detected(self, source: str, count: int = 1) -> None:
        """키워드 감지 기록"""
        KEYWORDS_DETECTED.labels(source=source).inc(count)
    
    def record_document_collected(self, source: str, count: int = 1) -> None:
        """문서 수집 기록"""
        DOCUMENTS_COLLECTED.labels(source=source).inc(count)
    
    def record_post_published(self, platform: str) -> None:
        """게시물 발행 기록"""
        POSTS_PUBLISHED.labels(platform=platform).inc()
    
    def record_error(self, stage: str, error_type: str) -> None:
        """에러 기록"""
        ERRORS.labels(stage=stage, error_type=error_type).inc()
    
    def record_guardrail_result(self, result: str) -> None:
        """Guardrail 결과 기록 (passed, rejected, revised)"""
        if result == "passed":
            GUARDRAIL_PASSED.inc()
        elif result == "rejected":
            GUARDRAIL_REJECTED.inc()
        elif result == "revised":
            GUARDRAIL_REVISED.inc()
    
    def update_queue_size(self, queue: str, size: int) -> None:
        """Redis 큐 크기 업데이트"""
        REDIS_QUEUE_SIZE.labels(queue=queue).set(size)
    
    def update_redis_memory(self, usage_percent: float) -> None:
        """Redis 메모리 사용량 업데이트"""
        REDIS_MEMORY_USAGE.set(usage_percent)
    
    def update_gpu_memory(self, vllm_gb: float, embedding_gb: float = 0.0) -> None:
        """GPU 메모리 사용량 업데이트"""
        VLLM_GPU_MEMORY.set(vllm_gb)
        EMBEDDING_MEMORY.set(embedding_gb)
    
    def update_deduplication_ratio(self, ratio: float) -> None:
        """중복 제거 비율 업데이트 (0-1)"""
        DEDUPLICATION_RATIO.set(ratio)
    
    def update_chromadb_size(self, collection: str, size: int) -> None:
        """ChromaDB 컬렉션 크기 업데이트"""
        CHROMADB_COLLECTION_SIZE.labels(collection=collection).set(size)
    
    @contextmanager
    def track_stage(self, stage: str) -> Generator[None, None, None]:
        """파이프라인 스테이지 추적"""
        histogram_map = {
            "trigger": TRIGGER_DURATION,
            "collect": COLLECT_DURATION,
            "analyze": ANALYZE_DURATION,
            "publish": PUBLISH_DURATION,
        }
        total_map = {
            "trigger": TRIGGER_TOTAL,
            "collect": COLLECT_TOTAL,
            "analyze": ANALYZE_TOTAL,
            "publish": PUBLISH_TOTAL,
        }
        success_map = {
            "trigger": TRIGGER_SUCCESS,
            "collect": COLLECT_SUCCESS,
            "analyze": ANALYZE_SUCCESS,
            "publish": PUBLISH_SUCCESS,
        }
        
        histogram = histogram_map.get(stage)
        total = total_map.get(stage)
        success = success_map.get(stage)
        
        if not histogram or not total or not success:
            yield
            return
        
        total.inc()
        ACTIVE_JOBS.labels(stage=stage).inc()
        start = time.perf_counter()
        try:
            yield
            success.inc()
        except Exception as e:
            ERRORS.labels(stage=stage, error_type=type(e).__name__).inc()
            raise
        finally:
            duration = time.perf_counter() - start
            histogram.observe(duration)
            ACTIVE_JOBS.labels(stage=stage).dec()
    
    @contextmanager
    def track_llm(self, model: str, operation: str) -> Generator[None, None, None]:
        """LLM 추론 추적"""
        start = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start
            LLM_INFERENCE_DURATION.labels(model=model, operation=operation).observe(duration)


# =============================================================================
# FastAPI Integration
# =============================================================================

def get_metrics_response() -> tuple[bytes, str]:
    """
    FastAPI 엔드포인트용 메트릭 응답 생성
    
    Usage:
        from fastapi import Response
        
        @app.get("/metrics")
        async def metrics():
            body, content_type = get_metrics_response()
            return Response(content=body, media_type=content_type)
    """
    return generate_latest(REGISTRY), CONTENT_TYPE_LATEST


def setup_fastapi_metrics(app: Any) -> None:
    """
    FastAPI 앱에 /metrics 엔드포인트 추가
    
    Usage:
        from fastapi import FastAPI
        app = FastAPI()
        setup_fastapi_metrics(app)
    """
    from fastapi import Response
    from fastapi.routing import APIRoute
    
    @app.get("/metrics", include_in_schema=False)
    async def metrics_endpoint() -> Response:
        body, content_type = get_metrics_response()
        return Response(content=body, media_type=content_type)


# =============================================================================
# Singleton Instance
# =============================================================================

# 전역 메트릭 인스턴스
metrics = TrendOpsMetrics()


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    import asyncio
    
    async def demo():
        print("TrendOps Metrics Demo")
        print("=" * 50)
        
        # 메트릭 기록 예시
        metrics.record_keyword_detected("google", count=5)
        metrics.record_keyword_detected("naver", count=3)
        
        metrics.record_document_collected("youtube", count=100)
        metrics.record_document_collected("news_rss", count=50)
        
        # 스테이지 추적 예시
        with metrics.track_stage("trigger"):
            await asyncio.sleep(0.1)  # 시뮬레이션
        
        with metrics.track_stage("collect"):
            await asyncio.sleep(0.5)
        
        with metrics.track_stage("analyze"):
            await asyncio.sleep(1.0)
        
        # LLM 추론 추적
        with metrics.track_llm("qwen2.5:7b", "summarize"):
            await asyncio.sleep(2.0)
        
        # 큐 상태 업데이트
        metrics.update_queue_size("pending", 10)
        metrics.update_queue_size("processing", 3)
        metrics.update_queue_size("completed", 47)
        
        # GPU 메모리 업데이트
        metrics.update_gpu_memory(vllm_gb=11.5, embedding_gb=0.0)
        
        # 메트릭 출력
        print("\nGenerated Metrics:")
        print("-" * 50)
        body, _ = get_metrics_response()
        # 주요 메트릭만 출력
        for line in body.decode().split('\n'):
            if line.startswith('trendops_') and not line.startswith('#'):
                print(line)
    
    asyncio.run(demo())