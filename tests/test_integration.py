"""통합 테스트 (Ollama + ChromaDB 필요)
CI에서는 자동으로 skip됨
"""
import pytest
import os

# CI 환경 감지 — GitHub Actions에서는 자동 skip
CI = os.getenv("CI", "false").lower() == "true"
skip_in_ci = pytest.mark.skipif(CI, reason="외부 서비스 필요 (CI에서 skip)")


@skip_in_ci
class TestTrigger:
    """Google Trends 트리거 통합 테스트"""

    @pytest.mark.asyncio
    async def test_fetch_trends(self):
        from trendops.trigger.trigger_google import GoogleTrendTrigger
        trigger = GoogleTrendTrigger()
        trends = await trigger.fetch_trends()
        assert isinstance(trends, list)
        assert len(trends) > 0
        assert hasattr(trends[0], "keyword")


@skip_in_ci
class TestCollector:
    """RSS 수집기 통합 테스트"""

    @pytest.mark.asyncio
    async def test_rss_fetch(self):
        from trendops.collector.collector_rss import RSSCollector
        async with RSSCollector(max_results=3) as rss:
            docs = await rss.fetch("AI")
            assert isinstance(docs, list)
            if docs:
                assert hasattr(docs[0], "title")


@skip_in_ci
class TestSearch:
    """하이브리드 검색 통합 테스트"""

    @pytest.mark.asyncio
    async def test_hybrid_search(self):
        from trendops.search.hybrid_search import get_hybrid_search, SearchMode
        search = get_hybrid_search()
        response = await search.search(
            query="AI 기술",
            n_results=3,
            mode=SearchMode.HYBRID
        )
        assert response is not None


@skip_in_ci
class TestPipeline:
    """E2E 파이프라인 통합 테스트"""

    @pytest.mark.asyncio
    async def test_pipeline_runs(self):
        from real_e2e_pipeline import run_real_pipeline
        result = await run_real_pipeline(
            max_keywords=2,
            max_articles=3,
            model="exaone3.5"
        )
        assert isinstance(result, dict)
        assert "success" in result