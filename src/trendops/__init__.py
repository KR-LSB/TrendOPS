# src/trendops/__init__.py
"""
TrendOps Week 4: Production-Level LLM Pipeline

모듈 구성:
- schemas: 통합 Pydantic 스키마 (Day 2)
- structured_analyzer: Outlines + Ollama 통합 (Day 1)
- guardrail: 콘텐츠 안전성 검증 (Day 3)
- safe_pipeline: Self-Correction Loop (Day 4)
- error_handler: Circuit Breaker + Retry (Day 5)

사용법:
    from trendops import SafeAnalysisPipeline, ContentGuardrail
    
    async with SafeAnalysisPipeline() as pipeline:
        result = await pipeline.analyze_safely(keyword, articles)
"""

from trendops.schemas import (
    # Enums
    TrendSource,
    JobStatus,
    SentimentType,
    ContentType,
    GuardrailAction,
    ReviewStatus,
    GenerationMethod,
    GuardrailIssueType,
    ErrorCategory,
    ErrorSeverity,
    PipelineStage,
    # Schemas
    TrendKeyword,
    TrendJob,
    NewsArticle,
    YouTubeComment,
    CollectionResult,
    SentimentRatio,
    AnalysisOutput,
    AnalysisResult,
    GuardrailIssue,
    GuardrailResult,
    GuardrailCheckRequest,
    ContentReview,
    PublishRequest,
    PublishResult,
    PipelineError,
    PipelineState,
)

from trendops.analyst.guardrail import (
    ContentGuardrail,
    GuardrailConfig,
    RuleBasedChecker,
    LLMBasedChecker,
    check_content_safety,
)

from trendops.analyst.safe_pipeline import (
    SafeAnalysisPipeline,
    SafePipelineResult,
    PipelineStatus,
    PipelineMetrics,
    analyze_keyword_safely,
)

from trendops.core.error_handler import (
    # Circuit Breaker
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerError,
    CircuitState,
    # Retry
    RetryConfig,
    RetryResult,
    RetryExhaustedError,
    retry_async,
    # Error Classification
    ErrorClassifier,
    # Decorators
    with_retry,
    with_circuit_breaker,
    with_error_handling,
    # Manager
    ErrorHandlerManager,
    LoggingErrorReporter,
    get_error_manager,
)

__version__ = "0.4.0"
__all__ = [
    # Enums
    "TrendSource",
    "JobStatus",
    "SentimentType",
    "ContentType",
    "GuardrailAction",
    "ReviewStatus",
    "GenerationMethod",
    "GuardrailIssueType",
    "ErrorCategory",
    "ErrorSeverity",
    "PipelineStage",
    # Schemas
    "TrendKeyword",
    "TrendJob",
    "NewsArticle",
    "YouTubeComment",
    "CollectionResult",
    "SentimentRatio",
    "AnalysisOutput",
    "AnalysisResult",
    "GuardrailIssue",
    "GuardrailResult",
    "GuardrailCheckRequest",
    "ContentReview",
    "PublishRequest",
    "PublishResult",
    "PipelineError",
    "PipelineState",
    # Guardrail
    "ContentGuardrail",
    "GuardrailConfig",
    "RuleBasedChecker",
    "LLMBasedChecker",
    "check_content_safety",
    # Pipeline
    "SafeAnalysisPipeline",
    "SafePipelineResult",
    "PipelineStatus",
    "PipelineMetrics",
    "analyze_keyword_safely",
    # Error Handler
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitBreakerError",
    "CircuitState",
    "RetryConfig",
    "RetryResult",
    "RetryExhaustedError",
    "retry_async",
    "ErrorClassifier",
    "with_retry",
    "with_circuit_breaker",
    "with_error_handling",
    "ErrorHandlerManager",
    "LoggingErrorReporter",
    "get_error_manager",
]