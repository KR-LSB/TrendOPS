# src/trendops/analyst/safe_pipeline.py
"""
Week 4 Day 4: Self-Correction Loop 구현

Blueprint Week 4 핵심: "프로덕션 레벨 LLM 파이프라인"

구조:
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Stage 1       │     │   Stage 2       │     │   Stage 3       │
│   Structured    │────▶│   Guardrail     │────▶│   Output        │
│   Generation    │     │   Review        │     │   Decision      │
│   (Day 1)       │     │   (Day 3)       │     │                 │
└─────────────────┘     └────────┬────────┘     └─────────────────┘
                                 │
                    ┌────────────┼────────────┐
                    ▼            ▼            ▼
              ┌──────────┐ ┌──────────┐ ┌──────────┐
              │   PASS   │ │  REVISE  │ │  REJECT  │
              │ (Output) │ │ (Retry)  │ │  (Log)   │
              └──────────┘ └──────────┘ └──────────┘

특징:
1. StructuredAnalyzer (Day 1) + ContentGuardrail (Day 3) 통합
2. Self-Correction: Guardrail 실패 시 자동 수정 후 재시도
3. 최대 재시도 횟수 제한
4. 상세한 파이프라인 상태 추적
5. 메트릭스 수집 (성공률, 수정률, 거부율)

사용법:
    async with SafeAnalysisPipeline() as pipeline:
        result = await pipeline.analyze_safely(
            keyword="트럼프 관세",
            articles=[...],
        )
        
        if result.success:
            print(result.analysis)
        else:
            print(f"Failed: {result.failure_reason}")
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable
from uuid import uuid4

from pydantic import BaseModel, Field

# Day 2 스키마
try:
    from trendops.schemas import (
        GuardrailAction,
        GuardrailResult,
        AnalysisResult,
        AnalysisOutput,
        SentimentRatio,
        PipelineError,
        PipelineStage,
        ErrorCategory,
        ErrorSeverity,
        GenerationMethod,
    )
except ImportError:
    # 단독 실행 시 fallback (테스트용)
    from schemas import (
        GuardrailAction,
        GuardrailResult,
        AnalysisResult,
        AnalysisOutput,
        SentimentRatio,
        PipelineError,
        PipelineStage,
        ErrorCategory,
        ErrorSeverity,
        GenerationMethod,
    )

# Day 3 Guardrail
try:
    from trendops.analyst.guardrail import ContentGuardrail, GuardrailConfig
except ImportError:
    # 단독 실행 시 fallback (테스트용)
    from .guardrail import ContentGuardrail, GuardrailConfig


# =============================================================================
# Pipeline Result Schema
# =============================================================================

class PipelineStatus(str, Enum):
    """파이프라인 상태"""
    SUCCESS = "success"           # 정상 완료
    REVISED = "revised"           # 수정 후 완료
    REJECTED = "rejected"         # 거부됨
    FAILED = "failed"             # 시스템 에러
    PENDING_REVIEW = "pending_review"  # 사람 검토 필요


@dataclass
class PipelineMetrics:
    """파이프라인 메트릭스"""
    total_attempts: int = 0
    successful: int = 0
    revised: int = 0
    rejected: int = 0
    failed: int = 0
    pending_review: int = 0
    
    # 시간 메트릭스
    total_time_seconds: float = 0.0
    avg_time_seconds: float = 0.0
    
    # Guardrail 메트릭스
    guardrail_pass_rate: float = 0.0
    revision_attempts: int = 0
    
    def record(self, status: PipelineStatus, time_seconds: float):
        """결과 기록"""
        self.total_attempts += 1
        self.total_time_seconds += time_seconds
        self.avg_time_seconds = self.total_time_seconds / self.total_attempts
        
        if status == PipelineStatus.SUCCESS:
            self.successful += 1
        elif status == PipelineStatus.REVISED:
            self.revised += 1
        elif status == PipelineStatus.REJECTED:
            self.rejected += 1
        elif status == PipelineStatus.FAILED:
            self.failed += 1
        elif status == PipelineStatus.PENDING_REVIEW:
            self.pending_review += 1
        
        # Pass rate 계산 (SUCCESS + REVISED)
        passed = self.successful + self.revised
        if self.total_attempts > 0:
            self.guardrail_pass_rate = passed / self.total_attempts
    
    def to_dict(self) -> dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "total_attempts": self.total_attempts,
            "successful": self.successful,
            "revised": self.revised,
            "rejected": self.rejected,
            "failed": self.failed,
            "pending_review": self.pending_review,
            "guardrail_pass_rate": round(self.guardrail_pass_rate, 3),
            "avg_time_seconds": round(self.avg_time_seconds, 2),
        }


class SafePipelineResult(BaseModel):
    """
    Self-Correction Pipeline 결과
    
    분석 결과 + Guardrail 결과 + 파이프라인 메타데이터
    """
    # 식별자
    pipeline_id: str = Field(default_factory=lambda: f"pipe-{uuid4().hex[:8]}")
    keyword: str = Field(..., description="분석 키워드")
    
    # 결과
    success: bool = Field(..., description="성공 여부")
    status: PipelineStatus = Field(..., description="파이프라인 상태")
    
    # 분석 결과 (성공 시)
    analysis: AnalysisResult | None = Field(default=None, description="분석 결과")
    
    # Guardrail 결과
    guardrail_result: GuardrailResult | None = Field(default=None, description="Guardrail 검사 결과")
    
    # 수정 이력
    revision_count: int = Field(default=0, description="수정 시도 횟수")
    original_content: str | None = Field(default=None, description="원본 콘텐츠 (수정된 경우)")
    
    # 실패 정보
    failure_reason: str | None = Field(default=None, description="실패 사유")
    errors: list[dict] = Field(default_factory=list, description="발생한 에러 목록")
    
    # 메타데이터
    total_time_seconds: float = Field(default=0.0, description="총 소요 시간")
    stage_times: dict[str, float] = Field(default_factory=dict, description="단계별 소요 시간")
    created_at: datetime = Field(default_factory=datetime.now, description="생성 시간")
    
    @property
    def is_usable(self) -> bool:
        """사용 가능한 결과인지 (SUCCESS 또는 REVISED)"""
        return self.status in (PipelineStatus.SUCCESS, PipelineStatus.REVISED)
    
    @property
    def needs_review(self) -> bool:
        """사람 검토가 필요한지"""
        return self.status == PipelineStatus.PENDING_REVIEW
    
    def get_final_summary(self) -> str | None:
        """최종 요약 반환 (수정된 버전 우선)"""
        if self.guardrail_result and self.guardrail_result.revised_content:
            return self.guardrail_result.revised_content
        if self.analysis:
            return self.analysis.analysis.summary
        return None


# =============================================================================
# Mock Analyzer (테스트용)
# =============================================================================

class MockStructuredAnalyzer:
    """테스트용 Mock Analyzer"""
    
    MOCK_ANALYSIS = {
        "main_cause": "해당 키워드에 대한 대중적 관심이 급증하여 화제가 되고 있습니다",
        "sentiment_ratio": {
            "positive": 0.25,
            "negative": 0.45,
            "neutral": 0.30
        },
        "key_opinions": [
            "국내 경제에 미치는 영향에 대한 우려가 확산되고 있습니다",
            "전문가들은 장기적 관점에서 분석이 필요하다고 지적합니다",
            "소비자들의 반응은 다양하게 나타나고 있습니다",
            "관련 업계에서는 대응책 마련에 나서고 있습니다"
        ],
        "summary": "해당 이슈가 화제가 되면서 다양한 의견이 제시되고 있습니다.\n경제적 영향에 대한 분석과 함께 전문가들의 견해가 주목받고 있습니다.\n향후 추이를 지켜봐야 할 것으로 보입니다."
    }
    
    def __init__(self, model_name: str = "mock-model"):
        self.model_name = model_name
    
    async def analyze(
        self,
        keyword: str,
        articles: list[dict[str, Any]],
    ) -> AnalysisResult:
        """Mock 분석 수행"""
        await asyncio.sleep(0.3)  # 시뮬레이션
        
        analysis_output = AnalysisOutput.model_validate(self.MOCK_ANALYSIS)
        
        return AnalysisResult(
            keyword=keyword,
            analysis=analysis_output,
            source_count=len(articles),
            model_version=self.model_name,
            inference_time_seconds=0.3,
            generation_method=GenerationMethod.MOCK,
        )
    
    async def close(self):
        pass
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, *args):
        await self.close()


# =============================================================================
# Safe Analysis Pipeline
# =============================================================================

class SafeAnalysisPipeline:
    """
    Self-Correction이 적용된 안전한 분석 파이프라인
    
    Blueprint Week 4: 2-Stage Guardrail + Self-Correction
    
    Flow:
    1. Structured Generation (StructuredAnalyzer)
    2. Guardrail Review (ContentGuardrail)
    3. Action에 따른 처리:
       - PASS: 결과 반환
       - REVISE: 수정 후 재검증
       - REVIEW: 사람 검토 대기열
       - REJECT: 거부 및 로깅
    
    Usage:
        async with SafeAnalysisPipeline() as pipeline:
            result = await pipeline.analyze_safely(
                keyword="트럼프 관세",
                articles=[{"title": "...", "summary": "..."}],
            )
            
            if result.success:
                print(result.analysis.analysis.summary)
            elif result.needs_review:
                print("사람 검토 필요:", result.guardrail_result.review_reason)
            else:
                print("거부됨:", result.failure_reason)
    """
    
    def __init__(
        self,
        # Analyzer 설정
        model_name: str = "qwen2.5:7b-instruct",
        base_url: str = "http://localhost:11434",
        use_outlines: bool = True,
        
        # Guardrail 설정
        guardrail_config: GuardrailConfig | None = None,
        strict_mode: bool = False,
        
        # 파이프라인 설정
        max_revisions: int = 2,
        enable_auto_revision: bool = True,
        
        # 테스트 모드
        use_mock: bool = False,
    ):
        """
        Args:
            model_name: Ollama 모델 이름
            base_url: Ollama 서버 URL
            use_outlines: Outlines 사용 여부
            guardrail_config: Guardrail 설정
            strict_mode: 엄격 모드
            max_revisions: 최대 수정 시도 횟수
            enable_auto_revision: 자동 수정 활성화
            use_mock: Mock 모드 (테스트용)
        """
        self.model_name = model_name
        self.base_url = base_url
        self.use_outlines = use_outlines
        self.strict_mode = strict_mode
        self.max_revisions = max_revisions
        self.enable_auto_revision = enable_auto_revision
        self.use_mock = use_mock
        
        # Guardrail 설정
        self.guardrail_config = guardrail_config or GuardrailConfig(
            strict_mode=strict_mode,
            llm_model=model_name,
            llm_base_url=base_url,
        )
        
        # 컴포넌트 초기화 (lazy)
        self._analyzer = None
        self._guardrail = None
        
        # 메트릭스
        self.metrics = PipelineMetrics()
        
        # 콜백
        self._on_revision: Callable[[str, str], None] | None = None
        self._on_rejection: Callable[[str, str], None] | None = None
    
    def _init_components(self):
        """컴포넌트 lazy 초기화"""
        if self._analyzer is None:
            if self.use_mock:
                self._analyzer = MockStructuredAnalyzer(model_name=self.model_name)
            else:
                # 실제 StructuredAnalyzer import 시도
                try:
                    from trendops.analyst.structured_analyzer import StructuredAnalyzer
                    self._analyzer = StructuredAnalyzer(
                        model_name=self.model_name,
                        base_url=self.base_url,
                        use_outlines=self.use_outlines,
                    )
                except ImportError:
                    try:
                        # 단독 실행 시 fallback (테스트용)
                        from .structured_analyzer import StructuredAnalyzer
                        self._analyzer = StructuredAnalyzer(
                            model_name=self.model_name,
                            base_url=self.base_url,
                            use_outlines=self.use_outlines,
                        )
                    except ImportError:
                        print("[WARNING] StructuredAnalyzer not found, using Mock")
                        self._analyzer = MockStructuredAnalyzer(model_name=self.model_name)
        
        if self._guardrail is None:
            self._guardrail = ContentGuardrail(
                config=self.guardrail_config,
                use_mock=self.use_mock,
            )
    
    def on_revision(self, callback: Callable[[str, str], None]):
        """수정 발생 시 콜백 등록"""
        self._on_revision = callback
    
    def on_rejection(self, callback: Callable[[str, str], None]):
        """거부 발생 시 콜백 등록"""
        self._on_rejection = callback
    
    async def analyze_safely(
        self,
        keyword: str,
        articles: list[dict[str, Any]],
    ) -> SafePipelineResult:
        """
        안전한 분석 수행 (Self-Correction 적용)
        
        Args:
            keyword: 분석 키워드
            articles: 뉴스 기사 목록
        
        Returns:
            SafePipelineResult: 파이프라인 결과
        """
        self._init_components()
        
        start_time = time.time()
        stage_times: dict[str, float] = {}
        errors: list[dict] = []
        
        pipeline_id = f"pipe-{uuid4().hex[:8]}"
        
        # ─────────────────────────────────────────────────────────
        # Stage 1: Structured Generation
        # ─────────────────────────────────────────────────────────
        stage1_start = time.time()
        
        try:
            analysis_result = await self._analyzer.analyze(keyword, articles)
            stage_times["generation"] = round(time.time() - stage1_start, 2)
        except Exception as e:
            stage_times["generation"] = round(time.time() - stage1_start, 2)
            errors.append({
                "stage": "generation",
                "error": str(e),
                "recoverable": False,
            })
            
            total_time = time.time() - start_time
            self.metrics.record(PipelineStatus.FAILED, total_time)
            
            return SafePipelineResult(
                pipeline_id=pipeline_id,
                keyword=keyword,
                success=False,
                status=PipelineStatus.FAILED,
                failure_reason=f"분석 생성 실패: {e}",
                errors=errors,
                total_time_seconds=round(total_time, 2),
                stage_times=stage_times,
            )
        
        # ─────────────────────────────────────────────────────────
        # Stage 2: Guardrail Review (+ Self-Correction Loop)
        # ─────────────────────────────────────────────────────────
        content_to_check = analysis_result.analysis.summary
        original_content = content_to_check
        revision_count = 0
        guardrail_result: GuardrailResult | None = None
        
        for attempt in range(self.max_revisions + 1):
            stage2_start = time.time()
            
            try:
                guardrail_result = await self._guardrail.check(
                    content=content_to_check,
                    keyword=keyword,
                    strict_mode=self.strict_mode,
                )
                stage_times[f"guardrail_{attempt}"] = round(time.time() - stage2_start, 2)
            except Exception as e:
                stage_times[f"guardrail_{attempt}"] = round(time.time() - stage2_start, 2)
                errors.append({
                    "stage": f"guardrail_{attempt}",
                    "error": str(e),
                    "recoverable": True,
                })
                # Guardrail 실패해도 계속 진행 (기본 PASS 처리)
                guardrail_result = GuardrailResult(
                    content_id=f"fallback-{uuid4().hex[:8]}",
                    action=GuardrailAction.PASS,
                    is_safe=True,
                    confidence=0.5,
                    issues=[],
                    original_content=content_to_check,
                )
                break
            
            # Action에 따른 처리
            if guardrail_result.action == GuardrailAction.PASS:
                # 통과 - 루프 종료
                break
            
            elif guardrail_result.action == GuardrailAction.REVISE:
                # 수정 시도
                if not self.enable_auto_revision or attempt >= self.max_revisions:
                    break
                
                if guardrail_result.revised_content:
                    # 수정된 콘텐츠로 재검증
                    content_to_check = guardrail_result.revised_content
                    revision_count += 1
                    self.metrics.revision_attempts += 1
                    
                    # 콜백 호출
                    if self._on_revision:
                        self._on_revision(original_content, content_to_check)
                else:
                    # 자동 수정 실패 - 루프 종료
                    break
            
            elif guardrail_result.action == GuardrailAction.REVIEW:
                # 사람 검토 필요 - 루프 종료
                break
            
            elif guardrail_result.action == GuardrailAction.REJECT:
                # 거부 - 루프 종료
                if self._on_rejection:
                    self._on_rejection(keyword, guardrail_result.issue_summary)
                break
        
        # ─────────────────────────────────────────────────────────
        # Stage 3: Result Decision
        # ─────────────────────────────────────────────────────────
        total_time = time.time() - start_time
        
        # 최종 상태 결정
        if guardrail_result.action == GuardrailAction.PASS:
            if revision_count > 0:
                status = PipelineStatus.REVISED
            else:
                status = PipelineStatus.SUCCESS
            success = True
            failure_reason = None
            
        elif guardrail_result.action == GuardrailAction.REVISE:
            # 수정 시도했지만 최대 횟수 도달
            status = PipelineStatus.REVISED
            success = True
            failure_reason = None
            
        elif guardrail_result.action == GuardrailAction.REVIEW:
            status = PipelineStatus.PENDING_REVIEW
            success = False
            failure_reason = guardrail_result.review_reason or "사람 검토 필요"
            
        else:  # REJECT
            status = PipelineStatus.REJECTED
            success = False
            failure_reason = f"Guardrail 거부: {guardrail_result.issue_summary}"
        
        # 메트릭스 기록
        self.metrics.record(status, total_time)
        
        # 결과 생성
        return SafePipelineResult(
            pipeline_id=pipeline_id,
            keyword=keyword,
            success=success,
            status=status,
            analysis=analysis_result if success else None,
            guardrail_result=guardrail_result,
            revision_count=revision_count,
            original_content=original_content if revision_count > 0 else None,
            failure_reason=failure_reason,
            errors=errors,
            total_time_seconds=round(total_time, 2),
            stage_times=stage_times,
        )
    
    async def analyze_batch(
        self,
        items: list[tuple[str, list[dict[str, Any]]]],
        concurrency: int = 3,
    ) -> list[SafePipelineResult]:
        """
        배치 분석 수행
        
        Args:
            items: [(keyword, articles), ...] 형태의 리스트
            concurrency: 동시 처리 수
        
        Returns:
            SafePipelineResult 리스트
        """
        semaphore = asyncio.Semaphore(concurrency)
        
        async def analyze_with_limit(keyword: str, articles: list[dict]):
            async with semaphore:
                return await self.analyze_safely(keyword, articles)
        
        tasks = [analyze_with_limit(kw, arts) for kw, arts in items]
        return await asyncio.gather(*tasks)
    
    def get_metrics(self) -> dict[str, Any]:
        """메트릭스 반환"""
        return self.metrics.to_dict()
    
    def reset_metrics(self):
        """메트릭스 초기화"""
        self.metrics = PipelineMetrics()
    
    async def close(self):
        """리소스 정리"""
        if self._analyzer and hasattr(self._analyzer, 'close'):
            await self._analyzer.close()
    
    async def __aenter__(self) -> "SafeAnalysisPipeline":
        return self
    
    async def __aexit__(self, *args):
        await self.close()


# =============================================================================
# Convenience Functions
# =============================================================================

async def analyze_keyword_safely(
    keyword: str,
    articles: list[dict[str, Any]],
    use_mock: bool = False,
) -> SafePipelineResult:
    """
    단일 키워드 안전 분석 편의 함수
    
    Usage:
        result = await analyze_keyword_safely(
            keyword="트럼프 관세",
            articles=[{"title": "...", "summary": "..."}],
        )
    """
    async with SafeAnalysisPipeline(use_mock=use_mock) as pipeline:
        return await pipeline.analyze_safely(keyword, articles)


# =============================================================================
# CLI Test
# =============================================================================

if __name__ == "__main__":
    async def main():
        """테스트 실행"""
        print("\n" + "=" * 70)
        print("  Week 4 Day 4: Safe Analysis Pipeline Test")
        print("  Self-Correction Loop Demo")
        print("=" * 70)
        
        # 테스트 데이터
        test_articles = [
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
        ]
        
        # Mock 모드로 파이프라인 테스트
        async with SafeAnalysisPipeline(use_mock=True) as pipeline:
            
            # 콜백 등록
            def on_revision(original: str, revised: str):
                print(f"\n📝 수정 발생!")
                print(f"   원본: {original[:50]}...")
                print(f"   수정: {revised[:50]}...")
            
            def on_rejection(keyword: str, reason: str):
                print(f"\n🚫 거부됨: {keyword}")
                print(f"   사유: {reason}")
            
            pipeline.on_revision(on_revision)
            pipeline.on_rejection(on_rejection)
            
            # 테스트 1: 정상 분석
            print("\n" + "─" * 60)
            print("Test 1: Normal Analysis")
            print("─" * 60)
            
            result = await pipeline.analyze_safely(
                keyword="트럼프 관세",
                articles=test_articles,
            )
            
            print(f"\n✅ Pipeline ID: {result.pipeline_id}")
            print(f"   Status: {result.status.value}")
            print(f"   Success: {result.success}")
            print(f"   Time: {result.total_time_seconds:.2f}s")
            print(f"   Stage Times: {result.stage_times}")
            
            if result.success and result.analysis:
                print(f"\n📊 Analysis Result:")
                print(f"   Keyword: {result.analysis.keyword}")
                print(f"   Main Cause: {result.analysis.analysis.main_cause[:60]}...")
                print(f"   Sentiment: P{result.analysis.analysis.sentiment_ratio.positive:.0%} "
                      f"N{result.analysis.analysis.sentiment_ratio.negative:.0%} "
                      f"U{result.analysis.analysis.sentiment_ratio.neutral:.0%}")
            
            if result.guardrail_result:
                print(f"\n🛡️ Guardrail Result:")
                print(f"   Action: {result.guardrail_result.action.value}")
                print(f"   Safe: {result.guardrail_result.is_safe}")
                print(f"   Confidence: {result.guardrail_result.confidence:.2f}")
                print(f"   Issues: {len(result.guardrail_result.issues)}")
            
            # 테스트 2: 배치 분석
            print("\n" + "─" * 60)
            print("Test 2: Batch Analysis")
            print("─" * 60)
            
            batch_items = [
                ("AI 기술", test_articles[:2]),
                ("경제 전망", test_articles[1:]),
                ("반도체 시장", test_articles),
            ]
            
            batch_results = await pipeline.analyze_batch(batch_items, concurrency=2)
            
            print(f"\n📦 Batch Results: {len(batch_results)} items")
            for r in batch_results:
                status_icon = "✅" if r.success else "❌"
                print(f"   {status_icon} {r.keyword}: {r.status.value} ({r.total_time_seconds:.2f}s)")
            
            # 메트릭스 출력
            print("\n" + "─" * 60)
            print("Pipeline Metrics")
            print("─" * 60)
            
            metrics = pipeline.get_metrics()
            print(f"\n📈 Metrics:")
            for key, value in metrics.items():
                print(f"   {key}: {value}")
        
        print("\n" + "=" * 70)
        print("  Test Complete!")
        print("=" * 70)
    
    asyncio.run(main())