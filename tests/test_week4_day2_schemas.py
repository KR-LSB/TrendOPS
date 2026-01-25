# tests/test_week4_day2_schemas.py
"""
Week 4 Day 2: 통합 스키마 테스트

테스트 범위:
1. 기본 Enum 테스트
2. Analysis 스키마 (SentimentRatio 정규화, AnalysisOutput 검증)
3. Guardrail 스키마 (Week 4 핵심)
4. Pipeline 스키마
5. Error 스키마
6. 직렬화/역직렬화 테스트

실행:
    python test_week4_day2_schemas.py
"""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import datetime
from uuid import uuid4

# Pydantic import
from pydantic import ValidationError

# 테스트 대상 스키마 import (같은 디렉토리에서)
try:
    from schemas import (
        # Enums
        TrendSource,
        JobStatus,
        SentimentType,
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
        CollectionResult,
        SentimentRatio,
        AnalysisOutput,
        AnalysisResult,
        GuardrailIssue,
        GuardrailResult,
        GuardrailCheckRequest,
        ContentReview,
        PipelineError,
        PipelineState,
    )
except ImportError:
    print("❌ schemas.py를 찾을 수 없습니다. 같은 디렉토리에 있는지 확인하세요.")
    sys.exit(1)


class TestRunner:
    """테스트 러너"""
    
    def __init__(self):
        self.passed = 0
        self.failed = 0
    
    def log_result(self, name: str, passed: bool, message: str = ""):
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}: {name}")
        if message:
            print(f"         {message}")
        if passed:
            self.passed += 1
        else:
            self.failed += 1
    
    def test_enums(self):
        """Enum 테스트"""
        print("\n📋 Test 1: Enums")
        print("-" * 50)
        
        # TrendSource
        try:
            assert TrendSource.GOOGLE.value == "google"
            assert TrendSource.NAVER.value == "naver"
            self.log_result("TrendSource enum", True)
        except AssertionError as e:
            self.log_result("TrendSource enum", False, str(e))
        
        # JobStatus
        try:
            assert JobStatus.PENDING.value == "pending"
            assert JobStatus.COMPLETED.value == "completed"
            self.log_result("JobStatus enum", True)
        except AssertionError as e:
            self.log_result("JobStatus enum", False, str(e))
        
        # GuardrailAction
        try:
            assert GuardrailAction.PASS.value == "pass"
            assert GuardrailAction.REJECT.value == "reject"
            assert GuardrailAction.REVISE.value == "revise"
            self.log_result("GuardrailAction enum", True)
        except AssertionError as e:
            self.log_result("GuardrailAction enum", False, str(e))
        
        # GuardrailIssueType
        try:
            assert GuardrailIssueType.POLITICAL_BIAS.value == "political_bias"
            assert GuardrailIssueType.PROFANITY.value == "profanity"
            self.log_result("GuardrailIssueType enum", True)
        except AssertionError as e:
            self.log_result("GuardrailIssueType enum", False, str(e))
    
    def test_sentiment_ratio(self):
        """SentimentRatio 정규화 테스트"""
        print("\n📊 Test 2: SentimentRatio Normalization")
        print("-" * 50)
        
        # 정상 비율
        try:
            ratio = SentimentRatio(positive=0.3, negative=0.5, neutral=0.2)
            total = ratio.positive + ratio.negative + ratio.neutral
            assert abs(total - 1.0) < 0.05
            self.log_result("Normal ratio (sum=1.0)", True, f"Total: {total}")
        except (ValidationError, AssertionError) as e:
            self.log_result("Normal ratio (sum=1.0)", False, str(e))
        
        # 비정규화 비율 -> 자동 정규화
        try:
            ratio = SentimentRatio(positive=0.6, negative=0.8, neutral=0.4)
            total = ratio.positive + ratio.negative + ratio.neutral
            assert abs(total - 1.0) < 0.05, f"Expected ~1.0, got {total}"
            self.log_result("Auto-normalization", True, f"0.6+0.8+0.4 → {total}")
        except (ValidationError, AssertionError) as e:
            self.log_result("Auto-normalization", False, str(e))
        
        # dominant_sentiment 프로퍼티
        try:
            ratio = SentimentRatio(positive=0.1, negative=0.7, neutral=0.2)
            assert ratio.dominant_sentiment == SentimentType.NEGATIVE
            self.log_result("dominant_sentiment property", True)
        except (ValidationError, AssertionError) as e:
            self.log_result("dominant_sentiment property", False, str(e))
        
        # to_display_dict
        try:
            ratio = SentimentRatio(positive=0.3, negative=0.5, neutral=0.2)
            display = ratio.to_display_dict()
            assert "긍정" in display and "부정" in display
            self.log_result("to_display_dict", True, str(display))
        except (ValidationError, AssertionError) as e:
            self.log_result("to_display_dict", False, str(e))
    
    def test_analysis_output(self):
        """AnalysisOutput 검증 테스트"""
        print("\n🧠 Test 3: AnalysisOutput Validation")
        print("-" * 50)
        
        valid_data = {
            "main_cause": "트럼프 대통령의 중국산 제품 25% 관세 부과 발표로 인한 관심 급증",
            "sentiment_ratio": {
                "positive": 0.15,
                "negative": 0.55,
                "neutral": 0.30
            },
            "key_opinions": [
                "국내 수출 기업들의 피해 우려 확산",
                "반도체·배터리 업종 주가 하락",
                "소비자 물가 상승 전망에 대한 불안감"
            ],
            "summary": "트럼프 대통령이 중국산 제품에 25% 관세를 부과한다고 발표했습니다.\n이에 따라 국내 수출 기업들이 비상 대응에 나섰습니다.\n정부는 지원 대책 마련에 나섰습니다."
        }
        
        # 유효한 데이터 파싱
        try:
            output = AnalysisOutput.model_validate(valid_data)
            self.log_result("Valid data parsing", True)
        except ValidationError as e:
            self.log_result("Valid data parsing", False, str(e))
        
        # main_cause 너무 짧음
        try:
            invalid_data = valid_data.copy()
            invalid_data["main_cause"] = "짧음"
            output = AnalysisOutput.model_validate(invalid_data)
            self.log_result("Reject short main_cause", False, "Should have rejected")
        except ValidationError:
            self.log_result("Reject short main_cause", True)
        
        # key_opinions 부족
        try:
            invalid_data = valid_data.copy()
            invalid_data["key_opinions"] = ["의견1", "의견2"]
            output = AnalysisOutput.model_validate(invalid_data)
            self.log_result("Reject insufficient opinions", False, "Should have rejected")
        except ValidationError:
            self.log_result("Reject insufficient opinions", True)
        
        # summary 너무 짧음
        try:
            invalid_data = valid_data.copy()
            invalid_data["summary"] = "짧은 요약"
            output = AnalysisOutput.model_validate(invalid_data)
            self.log_result("Reject short summary", False, "Should have rejected")
        except ValidationError:
            self.log_result("Reject short summary", True)
    
    def test_analysis_result(self):
        """AnalysisResult 테스트"""
        print("\n📈 Test 4: AnalysisResult")
        print("-" * 50)
        
        analysis_output = AnalysisOutput(
            main_cause="트럼프 대통령의 중국산 제품 25% 관세 부과 발표로 인한 관심 급증",
            sentiment_ratio=SentimentRatio(positive=0.15, negative=0.55, neutral=0.30),
            key_opinions=[
                "국내 수출 기업들의 피해 우려 확산",
                "반도체·배터리 업종 주가 하락",
                "소비자 물가 상승 전망에 대한 불안감"
            ],
            summary="트럼프 대통령이 중국산 제품에 25% 관세를 부과한다고 발표했습니다.\n이에 따라 국내 수출 기업들이 비상 대응에 나섰습니다.\n정부는 지원 대책 마련에 나섰습니다."
        )
        
        try:
            result = AnalysisResult(
                keyword="트럼프 관세",
                analysis=analysis_output,
                source_count=10,
                model_version="qwen2.5:7b-instruct",
                inference_time_seconds=5.2,
                generation_method=GenerationMethod.OUTLINES,
            )
            self.log_result("AnalysisResult creation", True)
        except ValidationError as e:
            self.log_result("AnalysisResult creation", False, str(e))
            return
        
        # is_valid 메서드
        try:
            assert result.is_valid() == True
            self.log_result("is_valid() method", True)
        except AssertionError as e:
            self.log_result("is_valid() method", False, str(e))
        
        # quality_score 프로퍼티
        try:
            score = result.quality_score
            assert 0 <= score <= 1
            self.log_result("quality_score property", True, f"Score: {score}")
        except AssertionError as e:
            self.log_result("quality_score property", False, str(e))
    
    def test_guardrail_schemas(self):
        """Guardrail 스키마 테스트 (Week 4 핵심)"""
        print("\n🛡️ Test 5: Guardrail Schemas")
        print("-" * 50)
        
        # GuardrailIssue
        try:
            issue = GuardrailIssue(
                issue_type=GuardrailIssueType.POLITICAL_BIAS,
                severity="high",
                description="특정 정당에 대한 편향적 표현이 포함되어 있습니다.",
                location="두 번째 문단",
                suggestion="중립적 표현으로 수정하세요."
            )
            assert issue.is_blocking == True
            self.log_result("GuardrailIssue (high severity)", True, "is_blocking=True")
        except (ValidationError, AssertionError) as e:
            self.log_result("GuardrailIssue (high severity)", False, str(e))
        
        # GuardrailIssue (low severity)
        try:
            issue = GuardrailIssue(
                issue_type=GuardrailIssueType.SENSATIONALISM,
                severity="low",
                description="약간 자극적인 표현이 있습니다.",
            )
            assert issue.is_blocking == False
            self.log_result("GuardrailIssue (low severity)", True, "is_blocking=False")
        except (ValidationError, AssertionError) as e:
            self.log_result("GuardrailIssue (low severity)", False, str(e))
        
        # GuardrailResult - PASS
        try:
            result = GuardrailResult(
                content_id="analysis-12345",
                action=GuardrailAction.PASS,
                is_safe=True,
                confidence=0.95,
                issues=[],
                original_content="트럼프 대통령이 관세 정책을 발표했습니다.",
            )
            assert result.has_blocking_issues == False
            self.log_result("GuardrailResult (PASS)", True)
        except (ValidationError, AssertionError) as e:
            self.log_result("GuardrailResult (PASS)", False, str(e))
        
        # GuardrailResult - REJECT with issues
        try:
            issue = GuardrailIssue(
                issue_type=GuardrailIssueType.HATE_SPEECH,
                severity="critical",
                description="혐오 발언이 포함되어 있습니다."
            )
            result = GuardrailResult(
                content_id="analysis-67890",
                action=GuardrailAction.REJECT,
                is_safe=False,
                confidence=0.99,
                issues=[issue],
                original_content="문제가 있는 콘텐츠",
            )
            assert result.has_blocking_issues == True
            assert result.issue_summary.startswith("1 issues")
            self.log_result("GuardrailResult (REJECT)", True, result.issue_summary)
        except (ValidationError, AssertionError) as e:
            self.log_result("GuardrailResult (REJECT)", False, str(e))
        
        # GuardrailCheckRequest
        try:
            request = GuardrailCheckRequest(
                content="검사할 콘텐츠입니다.",
                content_type="summary",
                keyword="트럼프 관세",
                strict_mode=True,
            )
            self.log_result("GuardrailCheckRequest", True)
        except ValidationError as e:
            self.log_result("GuardrailCheckRequest", False, str(e))
    
    def test_pipeline_error(self):
        """PipelineError 테스트"""
        print("\n⚠️ Test 6: PipelineError")
        print("-" * 50)
        
        # 직접 생성
        try:
            error = PipelineError(
                category=ErrorCategory.LLM,
                severity=ErrorSeverity.ERROR,
                message="LLM 응답 파싱 실패",
                stage="analyst",
                keyword="트럼프 관세",
                recoverable=True,
                retry_after_seconds=30,
            )
            self.log_result("Direct creation", True)
        except ValidationError as e:
            self.log_result("Direct creation", False, str(e))
        
        # from_exception 클래스메서드
        try:
            exc = ValueError("테스트 예외")
            error = PipelineError.from_exception(
                exception=exc,
                stage="collector",
                category=ErrorCategory.PARSING,
                keyword="테스트",
            )
            assert error.message == "테스트 예외"
            assert error.stack_trace is not None
            self.log_result("from_exception classmethod", True)
        except (ValidationError, AssertionError) as e:
            self.log_result("from_exception classmethod", False, str(e))
        
        # to_log_dict
        try:
            error = PipelineError(
                category=ErrorCategory.NETWORK,
                message="Connection timeout",
                stage="trigger",
            )
            log_dict = error.to_log_dict()
            assert "error_id" in log_dict
            assert log_dict["category"] == "network"
            self.log_result("to_log_dict method", True)
        except (ValidationError, AssertionError) as e:
            self.log_result("to_log_dict method", False, str(e))
    
    def test_pipeline_state(self):
        """PipelineState 테스트"""
        print("\n🔄 Test 7: PipelineState")
        print("-" * 50)
        
        try:
            state = PipelineState(
                job_id=uuid4(),
                keyword="트럼프 관세",
                current_stage=PipelineStage.TRIGGER,
            )
            self.log_result("PipelineState creation", True)
        except ValidationError as e:
            self.log_result("PipelineState creation", False, str(e))
            return
        
        # advance_stage
        try:
            state.advance_stage(PipelineStage.COLLECT)
            assert state.current_stage == PipelineStage.COLLECT
            assert PipelineStage.TRIGGER in state.stages_completed
            self.log_result("advance_stage method", True)
        except AssertionError as e:
            self.log_result("advance_stage method", False, str(e))
        
        # progress_percent
        try:
            progress = state.progress_percent
            assert 0 <= progress <= 100
            self.log_result("progress_percent property", True, f"{progress}%")
        except AssertionError as e:
            self.log_result("progress_percent property", False, str(e))
        
        # add_error
        try:
            error = PipelineError(
                category=ErrorCategory.NETWORK,
                message="Test error",
                stage="collect",
                recoverable=True,
            )
            state.add_error(error)
            assert len(state.errors) == 1
            assert state.is_failed == False  # recoverable이므로 실패 아님
            self.log_result("add_error (recoverable)", True)
        except AssertionError as e:
            self.log_result("add_error (recoverable)", False, str(e))
    
    def test_serialization(self):
        """직렬화/역직렬화 테스트"""
        print("\n📦 Test 8: Serialization")
        print("-" * 50)
        
        # AnalysisResult JSON 직렬화
        try:
            result = AnalysisResult(
                keyword="트럼프 관세",
                analysis=AnalysisOutput(
                    main_cause="트럼프 대통령의 관세 정책 발표로 인한 관심 급증",
                    sentiment_ratio=SentimentRatio(positive=0.2, negative=0.5, neutral=0.3),
                    key_opinions=["의견1", "의견2", "의견3"],
                    summary="트럼프 대통령이 관세 정책을 발표했습니다. 국내 기업들이 비상 대응에 나섰습니다. 정부는 지원책을 마련 중이며 시장은 불안한 모습을 보이고 있습니다."
                ),
                source_count=5,
                model_version="test",
                inference_time_seconds=1.0,
            )
            
            json_str = result.model_dump_json(indent=2)
            parsed = json.loads(json_str)
            
            # 역직렬화
            restored = AnalysisResult.model_validate(parsed)
            assert restored.keyword == result.keyword
            self.log_result("AnalysisResult JSON round-trip", True, f"{len(json_str)} bytes")
        except Exception as e:
            self.log_result("AnalysisResult JSON round-trip", False, str(e))
        
        # GuardrailResult JSON 직렬화
        try:
            result = GuardrailResult(
                content_id="test-123",
                action=GuardrailAction.PASS,
                is_safe=True,
                confidence=0.95,
                issues=[],
                original_content="테스트 콘텐츠",
            )
            
            json_str = result.model_dump_json()
            parsed = json.loads(json_str)
            restored = GuardrailResult.model_validate(parsed)
            
            assert restored.action == result.action
            self.log_result("GuardrailResult JSON round-trip", True)
        except Exception as e:
            self.log_result("GuardrailResult JSON round-trip", False, str(e))
    
    def run_all_tests(self):
        """모든 테스트 실행"""
        print("\n" + "=" * 70)
        print("  Week 4 Day 2: Schema Tests")
        print("=" * 70)
        
        self.test_enums()
        self.test_sentiment_ratio()
        self.test_analysis_output()
        self.test_analysis_result()
        self.test_guardrail_schemas()
        self.test_pipeline_error()
        self.test_pipeline_state()
        self.test_serialization()
        
        # 요약
        print("\n" + "=" * 70)
        print("  📊 Test Summary")
        print("=" * 70)
        print(f"  ✅ Passed: {self.passed}")
        print(f"  ❌ Failed: {self.failed}")
        total = self.passed + self.failed
        rate = self.passed / total * 100 if total > 0 else 0
        print(f"  📈 Success Rate: {rate:.1f}%")
        print("=" * 70)
        
        return self.failed == 0


def main():
    runner = TestRunner()
    success = runner.run_all_tests()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()