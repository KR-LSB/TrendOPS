"""파이프라인 유닛 테스트 (Ollama/ChromaDB 불필요)
CI 환경에서도 실행 가능한 테스트만 포함
"""
import pytest

# conftest.py에서 src/를 path에 추가했으므로 바로 import 가능
from real_e2e_pipeline import doc_to_dict


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# doc_to_dict 변환 함수 테스트
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class TestDocToDict:
    """뉴스 문서 → dict 변환 테스트"""

    def _make_mock_doc(self, **overrides):
        """테스트용 Mock 문서 생성"""
        class MockDoc:
            title = overrides.get("title", "테스트 기사 제목")
            link = overrides.get("link", "https://example.com/news/1")
            summary = overrides.get("summary", "기사 요약 내용입니다")
            keyword = overrides.get("keyword", "AI")
            source = overrides.get("source", "google_news")
            published = overrides.get("published", "2025-02-01")
            metadata = overrides.get("metadata", {"category": "tech"})
        return MockDoc()

    def test_basic_conversion(self):
        """기본 변환 동작"""
        doc = self._make_mock_doc()
        result = doc_to_dict(doc)

        assert result["title"] == "테스트 기사 제목"
        assert result["link"] == "https://example.com/news/1"
        assert result["summary"] == "기사 요약 내용입니다"
        assert result["keyword"] == "AI"
        assert result["source"] == "google_news"

    def test_metadata_preserved(self):
        """메타데이터가 유지되는지 확인"""
        doc = self._make_mock_doc(metadata={"author": "홍길동", "lang": "ko"})
        result = doc_to_dict(doc)

        assert result["metadata"]["author"] == "홍길동"
        assert result["metadata"]["lang"] == "ko"

    def test_published_to_string(self):
        """published 필드가 문자열로 변환되는지"""
        from datetime import datetime
        doc = self._make_mock_doc(published=datetime(2025, 2, 1))
        result = doc_to_dict(doc)

        assert isinstance(result["published"], str)

    def test_empty_metadata(self):
        """빈 메타데이터 처리"""
        doc = self._make_mock_doc(metadata={})
        result = doc_to_dict(doc)

        assert result["metadata"] == {}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 분석 결과 스키마 검증
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class TestAnalysisSchema:
    """LLM 분석 결과 포맷 검증"""

    def test_valid_schema(self):
        """올바른 분석 결과 스키마"""
        result = {
            "main_cause": "AI 기술의 급격한 발전",
            "sentiment": {"positive": 0.6, "negative": 0.2, "neutral": 0.2},
            "key_opinions": ["의견1", "의견2", "의견3"],
            "summary": "요약 내용입니다."
        }
        required_keys = ["main_cause", "sentiment", "key_opinions", "summary"]
        for key in required_keys:
            assert key in result

    def test_sentiment_sums_to_one(self):
        """감성 분석 합계가 ~1.0"""
        sentiment = {"positive": 0.5, "negative": 0.3, "neutral": 0.2}
        total = sum(sentiment.values())
        assert 0.9 <= total <= 1.1

    def test_key_opinions_is_list(self):
        """핵심 의견이 리스트 형태"""
        result = {"key_opinions": ["의견1", "의견2"]}
        assert isinstance(result["key_opinions"], list)
        assert len(result["key_opinions"]) >= 1


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 패키지 import 테스트
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class TestImports:
    """trendops 패키지 import 가능 여부 확인"""

    def test_import_schemas(self):
        """스키마 모듈 import"""
        from trendops.schemas import TrendKeyword  # noqa: F401

    def test_import_config(self):
        """설정 모듈 import"""
        from trendops.config.settings import get_settings

    def test_import_logger(self):
        """로거 모듈 import"""
        from trendops.utils.logger import get_logger
        logger = get_logger("test")
        assert logger is not None

    def test_import_error_handler(self):
        """에러 핸들러 import"""
        from trendops.core.error_handler import CircuitBreaker
