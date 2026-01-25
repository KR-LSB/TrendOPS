# tests/test_week4_day6_integration.py
"""
Week 4 Day 6: 전체 파이프라인 통합 테스트

테스트 범위:
1. 모듈 간 통합 검증
2. End-to-End 시나리오 테스트
3. 성능 측정 및 벤치마크
4. 에러 복구 시나리오
5. 배치 처리 테스트
6. Week 4 완료 검증

통합 모듈:
- Day 1: structured_analyzer.py (Outlines + Ollama)
- Day 2: schemas.py (Pydantic 스키마)
- Day 3: guardrail.py (콘텐츠 안전성)
- Day 4: safe_pipeline.py (Self-Correction Loop)
- Day 5: error_handler.py (Circuit Breaker + Retry)

실행:
    python test_week4_day6_integration.py
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

# =============================================================================
# Import All Week 4 Modules
# =============================================================================

try:
    # Day 2: Schemas
    from trendops.schemas import (
        # Enums
        TrendSource,
        JobStatus,
        SentimentType,
        GuardrailAction,
        GenerationMethod,
        ErrorCategory,
        ErrorSeverity,
        PipelineStage,
        GuardrailIssueType,
        # Schemas
        TrendKeyword,
        TrendJob,
        NewsArticle,
        CollectionResult,
        SentimentRatio,
        AnalysisOutput,
        AnalysisResult,
        GuardrailIssue,
        GuardrailResult,
        PipelineError,
        PipelineState,
    )
    
    # Day 3: Guardrail
    from trendops.analyst.guardrail import (
        ContentGuardrail,
        GuardrailConfig,
        RuleBasedChecker,
        check_content_safety,
    )
    
    # Day 4: Safe Pipeline
    from trendops.analyst.safe_pipeline import (
        SafeAnalysisPipeline,
        SafePipelineResult,
        PipelineStatus,
        PipelineMetrics,
        analyze_keyword_safely,
    )
    
    # Day 5: Error Handler
    from trendops.core.error_handler import (
        CircuitBreaker,
        CircuitBreakerConfig,
        CircuitBreakerError,
        CircuitState,
        RetryConfig,
        RetryResult,
        retry_async,
        ErrorClassifier,
        ErrorHandlerManager,
        with_retry,
        with_circuit_breaker,
        with_error_handling,
        get_error_manager,
    )
    
    IMPORTS_OK = True
    IMPORT_ERROR = None

except ImportError as e:
    # Fallback: 로컬 테스트용 (같은 디렉토리에 파일이 있는 경우)
    try:
        from schemas import (
            TrendSource, JobStatus, SentimentType, GuardrailAction,
            GenerationMethod, ErrorCategory, ErrorSeverity, PipelineStage,
            GuardrailIssueType, TrendKeyword, TrendJob, NewsArticle,
            CollectionResult, SentimentRatio, AnalysisOutput, AnalysisResult,
            GuardrailIssue, GuardrailResult, PipelineError, PipelineState,
        )
        from guardrail import (
            ContentGuardrail, GuardrailConfig, RuleBasedChecker, check_content_safety,
        )
        from safe_pipeline import (
            SafeAnalysisPipeline, SafePipelineResult, PipelineStatus,
            PipelineMetrics, analyze_keyword_safely,
        )
        from error_handler import (
            CircuitBreaker, CircuitBreakerConfig, CircuitBreakerError,
            CircuitState, RetryConfig, RetryResult, retry_async,
            ErrorClassifier, ErrorHandlerManager, with_retry,
            with_circuit_breaker, with_error_handling, get_error_manager,
        )
        IMPORTS_OK = True
        IMPORT_ERROR = None
    except ImportError as e2:
        IMPORTS_OK = False
        IMPORT_ERROR = str(e2)


# =============================================================================
# Test Data
# =============================================================================

SAMPLE_ARTICLES = [
    {
        "title": "트럼프, 중국산 제품 25% 관세 부과 발표",
        "summary": "미국 대통령이 무역 전쟁 격화 속에서 새로운 관세 정책을 발표했다.",
        "source": "경제일보",
    },
    {
        "title": "국내 수출기업들 '비상'…반도체·배터리 업종 타격 우려",
        "summary": "미국의 관세 정책 발표 이후 국내 수출 기업들이 비상 대응에 나섰다.",
        "source": "산업뉴스",
    },
    {
        "title": "전문가 분석: 국내 GDP 영향 전망",
        "summary": "경제 전문가들은 장기화될 경우 국내 경제에 상당한 영향이 있을 것으로 분석했다.",
        "source": "경제연구소",
    },
    {
        "title": "정부, 수출기업 지원 대책 발표 예정",
        "summary": "정부는 관세 영향을 최소화하기 위한 지원책을 마련 중이라고 밝혔다.",
        "source": "정책뉴스",
    },
    {
        "title": "증시 급락…코스피 2% 이상 하락",
        "summary": "관세 이슈로 인해 국내 증시가 급락하며 투자자들의 불안감이 커지고 있다.",
        "source": "금융뉴스",
    },
]

# 문제 있는 콘텐츠 샘플
PROBLEMATIC_CONTENTS = {
    "political_bias": "무능한 정부의 최악의 정책으로 국민이 고통받고 있다",
    "hate_speech": "그 사람들은 다 틀딱이야",
    "sensationalism": "충격!! 대박 사건!! 발칵 뒤집혔다!!!",
    "personal_info": "연락처: 010-1234-5678로 문의하세요",
    "safe_content": "트럼프 대통령이 관세 정책을 발표했습니다. 전문가들은 영향을 분석 중입니다.",
}


# =============================================================================
# Performance Metrics
# =============================================================================

@dataclass
class PerformanceReport:
    """성능 측정 리포트"""
    test_name: str
    total_time: float = 0.0
    iterations: int = 0
    success_count: int = 0
    failure_count: int = 0
    avg_time: float = 0.0
    min_time: float = float('inf')
    max_time: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)
    
    def record(self, elapsed: float, success: bool):
        self.iterations += 1
        self.total_time += elapsed
        if success:
            self.success_count += 1
        else:
            self.failure_count += 1
        self.avg_time = self.total_time / self.iterations
        self.min_time = min(self.min_time, elapsed)
        self.max_time = max(self.max_time, elapsed)
    
    def to_dict(self) -> dict:
        return {
            "test_name": self.test_name,
            "iterations": self.iterations,
            "success_rate": f"{self.success_count / self.iterations:.1%}" if self.iterations > 0 else "N/A",
            "total_time": f"{self.total_time:.2f}s",
            "avg_time": f"{self.avg_time:.3f}s",
            "min_time": f"{self.min_time:.3f}s" if self.min_time != float('inf') else "N/A",
            "max_time": f"{self.max_time:.3f}s",
        }


# =============================================================================
# Test Runner
# =============================================================================

class IntegrationTestRunner:
    """통합 테스트 러너"""
    
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.performance_reports: list[PerformanceReport] = []
    
    def log_result(self, name: str, passed: bool, message: str = ""):
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}: {name}")
        if message:
            print(f"         {message}")
        if passed:
            self.passed += 1
        else:
            self.failed += 1
    
    def log_section(self, title: str):
        print(f"\n{'─' * 60}")
        print(f"  {title}")
        print(f"{'─' * 60}")
    
    # ─────────────────────────────────────────────────────────────
    # Test 1: Module Import Verification
    # ─────────────────────────────────────────────────────────────
    
    async def test_module_imports(self):
        """모듈 임포트 검증"""
        print("\n📦 Test 1: Module Imports")
        print("-" * 50)
        
        self.log_result(
            "All Week 4 modules imported",
            IMPORTS_OK,
            IMPORT_ERROR if not IMPORTS_OK else "schemas, guardrail, safe_pipeline, error_handler"
        )
        
        if not IMPORTS_OK:
            return False
        
        # 스키마 클래스 존재 확인
        schema_classes = [
            SentimentRatio, AnalysisOutput, AnalysisResult,
            GuardrailIssue, GuardrailResult, PipelineError,
        ]
        all_schemas_exist = all(cls is not None for cls in schema_classes)
        self.log_result(
            "Day 2 schemas available",
            all_schemas_exist,
            f"{len(schema_classes)} schema classes"
        )
        
        # Guardrail 클래스 확인
        self.log_result(
            "Day 3 guardrail available",
            ContentGuardrail is not None,
        )
        
        # Pipeline 클래스 확인
        self.log_result(
            "Day 4 safe_pipeline available",
            SafeAnalysisPipeline is not None,
        )
        
        # Error Handler 클래스 확인
        self.log_result(
            "Day 5 error_handler available",
            CircuitBreaker is not None and ErrorClassifier is not None,
        )
        
        return True
    
    # ─────────────────────────────────────────────────────────────
    # Test 2: Schema Integration
    # ─────────────────────────────────────────────────────────────
    
    async def test_schema_integration(self):
        """스키마 통합 테스트"""
        print("\n📋 Test 2: Schema Integration")
        print("-" * 50)
        
        # SentimentRatio 정규화
        ratio = SentimentRatio(positive=0.6, negative=0.8, neutral=0.4)
        total = ratio.positive + ratio.negative + ratio.neutral
        self.log_result(
            "SentimentRatio auto-normalization",
            abs(total - 1.0) < 0.05,
            f"Total: {total:.2f}"
        )
        
        # AnalysisOutput 생성 및 검증
        analysis_output = AnalysisOutput(
            main_cause="트럼프 대통령의 관세 정책 발표로 인한 경제적 영향 우려",
            sentiment_ratio=SentimentRatio(positive=0.2, negative=0.5, neutral=0.3),
            key_opinions=[
                "수출 기업들의 피해 우려 확산",
                "반도체 업종 타격 예상",
                "정부 대응책 마련 촉구"
            ],
            summary="트럼프 대통령이 중국산 제품에 관세를 부과한다고 발표했습니다.\n국내 수출 기업들이 대응에 나섰으며 경제적 영향이 우려됩니다.\n정부는 지원책 마련에 나서고 있습니다."
        )
        self.log_result(
            "AnalysisOutput validation",
            len(analysis_output.main_cause) >= 10,
        )
        
        # AnalysisResult 생성
        result = AnalysisResult(
            keyword="트럼프 관세",
            analysis=analysis_output,
            source_count=5,
            model_version="qwen2.5:7b-instruct",
            inference_time_seconds=2.5,
            generation_method=GenerationMethod.MOCK,
        )
        self.log_result(
            "AnalysisResult creation",
            result.is_valid(),
            f"Quality score: {result.quality_score}"
        )
        
        # GuardrailResult 생성
        guardrail_result = GuardrailResult(
            content_id="test-123",
            action=GuardrailAction.PASS,
            is_safe=True,
            confidence=0.95,
            issues=[],
            original_content="테스트 콘텐츠",
        )
        self.log_result(
            "GuardrailResult creation",
            guardrail_result.is_safe,
        )
        
        # JSON 직렬화/역직렬화
        json_str = result.model_dump_json()
        restored = AnalysisResult.model_validate_json(json_str)
        self.log_result(
            "JSON serialization round-trip",
            restored.keyword == result.keyword,
            f"{len(json_str)} bytes"
        )
    
    # ─────────────────────────────────────────────────────────────
    # Test 3: Guardrail Integration
    # ─────────────────────────────────────────────────────────────
    
    async def test_guardrail_integration(self):
        """Guardrail 통합 테스트"""
        print("\n🛡️ Test 3: Guardrail Integration")
        print("-" * 50)
        
        guardrail = ContentGuardrail(use_mock=True)
        
        # 안전한 콘텐츠
        result = await guardrail.check(
            PROBLEMATIC_CONTENTS["safe_content"],
            keyword="트럼프 관세"
        )
        self.log_result(
            "Safe content passes",
            result.action == GuardrailAction.PASS,
            f"Action: {result.action.value}, Confidence: {result.confidence:.2f}"
        )
        
        # 정치적 편향
        result = await guardrail.check(PROBLEMATIC_CONTENTS["political_bias"])
        self.log_result(
            "Detects political bias",
            result.action != GuardrailAction.PASS,
            f"Action: {result.action.value}, Issues: {len(result.issues)}"
        )
        
        # 혐오 발언
        result = await guardrail.check(PROBLEMATIC_CONTENTS["hate_speech"])
        self.log_result(
            "Detects hate speech",
            result.action == GuardrailAction.REJECT,
            f"Action: {result.action.value}"
        )
        
        # 선정적 표현
        result = await guardrail.check(PROBLEMATIC_CONTENTS["sensationalism"])
        has_issues = len(result.issues) > 0
        self.log_result(
            "Detects sensationalism",
            has_issues,
            f"Issues: {len(result.issues)}"
        )
        
        # 개인정보
        result = await guardrail.check(PROBLEMATIC_CONTENTS["personal_info"])
        has_personal_info = any(
            i.issue_type == GuardrailIssueType.PERSONAL_INFO 
            for i in result.issues
        )
        self.log_result(
            "Detects personal info",
            has_personal_info,
        )
    
    # ─────────────────────────────────────────────────────────────
    # Test 4: Safe Pipeline End-to-End
    # ─────────────────────────────────────────────────────────────
    
    async def test_safe_pipeline_e2e(self):
        """Safe Pipeline End-to-End 테스트"""
        print("\n🔄 Test 4: Safe Pipeline E2E")
        print("-" * 50)
        
        report = PerformanceReport(test_name="safe_pipeline_e2e")
        
        async with SafeAnalysisPipeline(use_mock=True) as pipeline:
            # 정상 분석 흐름
            start = time.time()
            result = await pipeline.analyze_safely(
                keyword="트럼프 관세",
                articles=SAMPLE_ARTICLES,
            )
            elapsed = time.time() - start
            report.record(elapsed, result.success)
            
            self.log_result(
                "Normal analysis flow",
                result.success,
                f"Status: {result.status.value}, Time: {elapsed:.2f}s"
            )
            
            # 분석 결과 유효성
            if result.analysis:
                self.log_result(
                    "Analysis result valid",
                    result.analysis.is_valid(),
                    f"Quality: {result.analysis.quality_score}"
                )
            
            # Guardrail 결과 존재
            self.log_result(
                "Guardrail result present",
                result.guardrail_result is not None,
            )
            
            # 메트릭스 수집
            metrics = pipeline.get_metrics()
            self.log_result(
                "Metrics collected",
                metrics["total_attempts"] >= 1,
                f"Attempts: {metrics['total_attempts']}, Pass rate: {metrics['guardrail_pass_rate']:.1%}"
            )
        
        self.performance_reports.append(report)
    
    # ─────────────────────────────────────────────────────────────
    # Test 5: Error Handler Integration
    # ─────────────────────────────────────────────────────────────
    
    async def test_error_handler_integration(self):
        """에러 핸들러 통합 테스트"""
        print("\n⚠️ Test 5: Error Handler Integration")
        print("-" * 50)
        
        # Circuit Breaker + Retry 조합
        breaker = CircuitBreaker(
            name="integration_test",
            config=CircuitBreakerConfig(
                failure_threshold=3,
                timeout_seconds=1.0,
            ),
        )
        
        call_count = 0
        
        @with_error_handling(
            stage="integration",
            retry_config=RetryConfig(max_attempts=2, backoff_base=0.01),
            circuit_breaker=breaker,
        )
        async def flaky_operation(succeed: bool):
            nonlocal call_count
            call_count += 1
            if not succeed:
                raise ConnectionError("Simulated failure")
            return "success"
        
        # 성공 케이스
        result = await flaky_operation(True)
        self.log_result(
            "Error handling: success case",
            result == "success",
        )
        
        # 실패 케이스 (재시도)
        call_count = 0
        try:
            await flaky_operation(False)
            self.log_result("Error handling: retry on failure", False)
        except ConnectionError:
            self.log_result(
                "Error handling: retry exhausted",
                call_count >= 2,
                f"Attempts: {call_count}"
            )
        
        # Circuit Breaker 동작
        self.log_result(
            "Circuit breaker records failures",
            breaker.failure_count >= 1,
            f"Failures: {breaker.failure_count}"
        )
        
        # Error Manager
        manager = get_error_manager()
        error = PipelineError(
            category=ErrorCategory.NETWORK,
            message="Integration test error",
            stage="integration",
        )
        await manager.report_error(error)
        
        summary = manager.get_error_summary()
        self.log_result(
            "Error manager integration",
            summary["total"] >= 1,
        )
    
    # ─────────────────────────────────────────────────────────────
    # Test 6: Batch Processing
    # ─────────────────────────────────────────────────────────────
    
    async def test_batch_processing(self):
        """배치 처리 테스트"""
        print("\n📦 Test 6: Batch Processing")
        print("-" * 50)
        
        report = PerformanceReport(test_name="batch_processing")
        
        keywords = ["트럼프 관세", "AI 기술", "반도체 시장", "경제 전망", "주식 시장"]
        batch_items = [(kw, SAMPLE_ARTICLES[:3]) for kw in keywords]
        
        async with SafeAnalysisPipeline(use_mock=True) as pipeline:
            start = time.time()
            results = await pipeline.analyze_batch(batch_items, concurrency=3)
            total_elapsed = time.time() - start
            
            success_count = sum(1 for r in results if r.success)
            
            for r in results:
                report.record(r.total_time_seconds, r.success)
            
            self.log_result(
                "Batch processing completes",
                len(results) == len(keywords),
                f"Processed: {len(results)}/{len(keywords)}"
            )
            
            self.log_result(
                "Batch success rate",
                success_count == len(keywords),
                f"Success: {success_count}/{len(keywords)}"
            )
            
            avg_time = sum(r.total_time_seconds for r in results) / len(results)
            self.log_result(
                "Batch performance",
                total_elapsed < len(keywords) * 2,  # 병렬 처리로 개별 합보다 빨라야 함
                f"Total: {total_elapsed:.2f}s, Avg: {avg_time:.2f}s"
            )
        
        self.performance_reports.append(report)
    
    # ─────────────────────────────────────────────────────────────
    # Test 7: Pipeline State Tracking
    # ─────────────────────────────────────────────────────────────
    
    async def test_pipeline_state_tracking(self):
        """파이프라인 상태 추적 테스트"""
        print("\n📊 Test 7: Pipeline State Tracking")
        print("-" * 50)
        
        async with SafeAnalysisPipeline(use_mock=True) as pipeline:
            # 여러 번 실행
            for i in range(5):
                await pipeline.analyze_safely(f"키워드{i}", SAMPLE_ARTICLES[:2])
            
            metrics = pipeline.get_metrics()
            
            self.log_result(
                "Tracks total attempts",
                metrics["total_attempts"] == 5,
                f"Total: {metrics['total_attempts']}"
            )
            
            self.log_result(
                "Calculates pass rate",
                0 <= metrics["guardrail_pass_rate"] <= 1,
                f"Rate: {metrics['guardrail_pass_rate']:.1%}"
            )
            
            self.log_result(
                "Records average time",
                metrics["avg_time_seconds"] > 0,
                f"Avg: {metrics['avg_time_seconds']:.2f}s"
            )
            
            # 리셋 테스트
            pipeline.reset_metrics()
            reset_metrics = pipeline.get_metrics()
            self.log_result(
                "Metrics reset works",
                reset_metrics["total_attempts"] == 0,
            )
    
    # ─────────────────────────────────────────────────────────────
    # Test 8: Full Integration Scenario
    # ─────────────────────────────────────────────────────────────
    
    async def test_full_integration_scenario(self):
        """전체 통합 시나리오 테스트"""
        print("\n🎯 Test 8: Full Integration Scenario")
        print("-" * 50)
        
        # 시나리오: 실제 파이프라인 흐름 시뮬레이션
        
        # 1. TrendKeyword 생성
        trend = TrendKeyword(
            keyword="트럼프 관세",
            source=TrendSource.GOOGLE,
            trend_score=8.5,
        )
        self.log_result(
            "Step 1: TrendKeyword created",
            trend.keyword == "트럼프 관세",
            f"Score: {trend.trend_score}"
        )
        
        # 2. CollectionResult 생성
        articles = [
            NewsArticle(
                title=a["title"],
                link=f"https://example.com/{i}",
                summary=a["summary"],
                source=a["source"],
            )
            for i, a in enumerate(SAMPLE_ARTICLES)
        ]
        collection = CollectionResult(
            keyword=trend.keyword,
            articles=articles,
            source=TrendSource.GOOGLE,
        )
        self.log_result(
            "Step 2: CollectionResult created",
            collection.total_count == len(SAMPLE_ARTICLES),
            f"Articles: {collection.total_count}"
        )
        
        # 3. SafeAnalysisPipeline 실행
        async with SafeAnalysisPipeline(use_mock=True) as pipeline:
            article_dicts = [
                {"title": a.title, "summary": a.summary or "", "source": a.source}
                for a in collection.articles
            ]
            
            result = await pipeline.analyze_safely(
                keyword=trend.keyword,
                articles=article_dicts,
            )
            
            self.log_result(
                "Step 3: Analysis completed",
                result.success,
                f"Status: {result.status.value}"
            )
            
            # 4. 결과 검증
            if result.analysis:
                self.log_result(
                    "Step 4: Analysis valid",
                    result.analysis.is_valid(),
                    f"Main cause: {result.analysis.analysis.main_cause[:40]}..."
                )
            
            # 5. Guardrail 통과 확인
            if result.guardrail_result:
                self.log_result(
                    "Step 5: Guardrail passed",
                    result.guardrail_result.is_safe,
                    f"Action: {result.guardrail_result.action.value}"
                )
            
            # 6. 최종 요약 추출
            final_summary = result.get_final_summary()
            self.log_result(
                "Step 6: Final summary available",
                final_summary is not None and len(final_summary) > 0,
                f"Length: {len(final_summary) if final_summary else 0}"
            )
    
    # ─────────────────────────────────────────────────────────────
    # Test 9: Error Recovery Scenario
    # ─────────────────────────────────────────────────────────────
    
    async def test_error_recovery_scenario(self):
        """에러 복구 시나리오 테스트"""
        print("\n🔧 Test 9: Error Recovery Scenario")
        print("-" * 50)
        
        manager = ErrorHandlerManager()
        
        # 서비스별 Circuit Breaker 등록
        ollama_breaker = manager.register_breaker(
            "ollama",
            config=CircuitBreakerConfig(failure_threshold=3, timeout_seconds=1.0)
        )
        redis_breaker = manager.register_breaker(
            "redis",
            config=CircuitBreakerConfig(failure_threshold=5, timeout_seconds=2.0)
        )
        
        # 실패 시뮬레이션
        for i in range(3):
            ollama_breaker.record_failure(TimeoutError(f"Timeout {i}"))
        
        self.log_result(
            "Circuit breaker opens on failures",
            ollama_breaker.state == CircuitState.OPEN,
            f"State: {ollama_breaker.state.value}"
        )
        
        # 타임아웃 후 복구
        await asyncio.sleep(1.1)
        
        self.log_result(
            "Circuit breaker transitions to HALF_OPEN",
            ollama_breaker.state == CircuitState.HALF_OPEN,
            f"State: {ollama_breaker.state.value}"
        )
        
        # 성공으로 복구
        ollama_breaker.record_success()
        ollama_breaker.record_success()
        
        self.log_result(
            "Circuit breaker recovers to CLOSED",
            ollama_breaker.state == CircuitState.CLOSED,
            f"State: {ollama_breaker.state.value}"
        )
        
        # 전체 상태 조회
        all_stats = manager.get_all_breaker_stats()
        self.log_result(
            "Manager tracks all breakers",
            len(all_stats) == 2,
            f"Breakers: {list(all_stats.keys())}"
        )
    
    # ─────────────────────────────────────────────────────────────
    # Test 10: Performance Benchmark
    # ─────────────────────────────────────────────────────────────
    
    async def test_performance_benchmark(self):
        """성능 벤치마크 테스트"""
        print("\n⚡ Test 10: Performance Benchmark")
        print("-" * 50)
        
        report = PerformanceReport(test_name="performance_benchmark")
        iterations = 10
        
        async with SafeAnalysisPipeline(use_mock=True) as pipeline:
            for i in range(iterations):
                start = time.time()
                result = await pipeline.analyze_safely(
                    keyword=f"벤치마크_{i}",
                    articles=SAMPLE_ARTICLES[:3],
                )
                elapsed = time.time() - start
                report.record(elapsed, result.success)
        
        self.performance_reports.append(report)
        
        self.log_result(
            f"Completed {iterations} iterations",
            report.iterations == iterations,
        )
        
        self.log_result(
            "All iterations successful",
            report.success_count == iterations,
            f"Success: {report.success_count}/{iterations}"
        )
        
        self.log_result(
            "Average time reasonable",
            report.avg_time < 2.0,  # Mock 모드에서 2초 미만 예상
            f"Avg: {report.avg_time:.3f}s"
        )
        
        self.log_result(
            "Time variance acceptable",
            (report.max_time - report.min_time) < 1.0,
            f"Range: {report.min_time:.3f}s - {report.max_time:.3f}s"
        )
    
    # ─────────────────────────────────────────────────────────────
    # Generate Report
    # ─────────────────────────────────────────────────────────────
    
    def generate_report(self) -> dict:
        """최종 리포트 생성"""
        return {
            "week": 4,
            "date": datetime.now().isoformat(),
            "tests": {
                "passed": self.passed,
                "failed": self.failed,
                "total": self.passed + self.failed,
                "success_rate": f"{self.passed / (self.passed + self.failed):.1%}" if (self.passed + self.failed) > 0 else "N/A",
            },
            "performance": [r.to_dict() for r in self.performance_reports],
            "modules": {
                "day1": "structured_analyzer.py",
                "day2": "schemas.py",
                "day3": "guardrail.py",
                "day4": "safe_pipeline.py",
                "day5": "error_handler.py",
            },
        }
    
    # ─────────────────────────────────────────────────────────────
    # Run All Tests
    # ─────────────────────────────────────────────────────────────
    
    async def run_all_tests(self):
        """모든 테스트 실행"""
        print("\n" + "=" * 70)
        print("  Week 4 Day 6: Integration Tests")
        print("  TrendOps LLM Pipeline - Full Integration")
        print("=" * 70)
        
        # 임포트 확인
        imports_ok = await self.test_module_imports()
        if not imports_ok:
            print("\n❌ Module imports failed. Cannot continue.")
            return False
        
        # 모든 테스트 실행
        await self.test_schema_integration()
        await self.test_guardrail_integration()
        await self.test_safe_pipeline_e2e()
        await self.test_error_handler_integration()
        await self.test_batch_processing()
        await self.test_pipeline_state_tracking()
        await self.test_full_integration_scenario()
        await self.test_error_recovery_scenario()
        await self.test_performance_benchmark()
        
        # 최종 리포트
        report = self.generate_report()
        
        print("\n" + "=" * 70)
        print("  📊 Week 4 Integration Test Summary")
        print("=" * 70)
        print(f"  ✅ Passed: {self.passed}")
        print(f"  ❌ Failed: {self.failed}")
        print(f"  📈 Success Rate: {report['tests']['success_rate']}")
        
        print("\n  ⚡ Performance Summary:")
        for perf in self.performance_reports:
            p = perf.to_dict()
            print(f"     - {p['test_name']}: {p['avg_time']} avg, {p['success_rate']} success")
        
        print("\n  📦 Integrated Modules:")
        for day, module in report['modules'].items():
            print(f"     - {day}: {module}")
        
        print("\n" + "=" * 70)
        
        if self.failed == 0:
            print("  🎉 Week 4 Integration Tests PASSED!")
            print("     All modules working together correctly.")
        else:
            print("  ⚠️ Some tests failed. Review the results above.")
        
        print("=" * 70)
        
        return self.failed == 0


# =============================================================================
# Main
# =============================================================================

async def main():
    runner = IntegrationTestRunner()
    success = await runner.run_all_tests()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())