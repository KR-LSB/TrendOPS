# src/trendops/collector/collector_rss_google.py
"""Google News RSS Collector - 차단 위험 Zero의 안정적인 뉴스 소스"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any
from urllib.parse import quote

import aiohttp
import feedparser
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


class Article(BaseModel):
    """수집된 뉴스 기사 모델"""

    title: str = Field(..., description="기사 제목")
    link: str = Field(..., description="기사 원본 URL")
    published: datetime | None = Field(None, description="발행 시간")
    source: str = Field(default="google_news_rss", description="수집 소스")
    keyword: str = Field(default="", description="검색 키워드")
    summary: str = Field(default="", description="기사 요약")

    @field_validator("published", mode="before")
    @classmethod
    def parse_published(cls, v: Any) -> datetime | None:
        if v is None:
            return None
        if isinstance(v, datetime):
            return v
        if isinstance(v, str):
            try:
                from email.utils import parsedate_to_datetime

                return parsedate_to_datetime(v)
            except (ValueError, TypeError):
                return None
        return None


class CollectionResult(BaseModel):
    """수집 결과 모델"""

    keyword: str
    articles: list[Article]
    source: str = "google_news_rss"
    collected_at: datetime = Field(default_factory=datetime.now)
    success: bool = True
    error_message: str | None = None

    @property
    def count(self) -> int:
        """수집된 기사 수 반환"""
        return len(self.articles)

    def is_valid(self) -> bool:
        return self.success and len(self.articles) > 0

    def with_source(self, source: str) -> CollectionResult:
        self.source = source
        return self


class RetryConfig(BaseModel):
    """재시도 설정"""

    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0


class GoogleNewsRSSCollector:
    """
    Google News RSS Collector

    - 키워드 기반 뉴스 검색
    - 차단 위험 Zero (공식 RSS)
    - 비동기 + Retry 지원
    """

    RSS_URL = "https://news.google.com/rss/search?q={keyword}&hl=ko&gl=KR&ceid=KR:ko"

    def __init__(
        self,
        retry_config: RetryConfig | None = None,
        timeout_seconds: int = 30,
    ):
        self.retry_config = retry_config or RetryConfig()
        self.timeout = aiohttp.ClientTimeout(total=timeout_seconds)
        self._session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=self.timeout,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                },
            )
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    async def _fetch_rss_content(self, url: str) -> str:
        """RSS URL에서 XML 콘텐츠 가져오기"""
        session = await self._get_session()
        async with session.get(url) as response:
            response.raise_for_status()
            return await response.text()

    async def _fetch_with_retry(self, url: str) -> str:
        """지수 백오프를 적용한 재시도"""
        last_exception: Exception | None = None
        config = self.retry_config

        for attempt in range(config.max_attempts):
            try:
                return await self._fetch_rss_content(url)
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                last_exception = e
                if attempt < config.max_attempts - 1:
                    delay = min(
                        config.base_delay * (config.exponential_base**attempt),
                        config.max_delay,
                    )
                    logger.warning(
                        f"RSS fetch attempt {attempt + 1} failed: {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    await asyncio.sleep(delay)

        raise last_exception or Exception("Unknown error during fetch")

    def _is_relevant(self, article: Article, keyword: str) -> bool:
        """
        [Garbage Filter] 기사가 검색 키워드와 관련이 있는지 검증

        Google News는 검색 결과가 없을 때 인기 뉴스(Top Stories)를 반환하는 경우가 있음.
        이를 방지하기 위해 제목이나 요약에 키워드 단어가 포함되어 있는지 확인.
        """
        # 1. 키워드 전처리 (해시태그, 따옴표 제거)
        clean_kw = keyword.replace("#", "").replace('"', "").replace("'", "").strip()
        if not clean_kw:
            return True  # 키워드가 없으면 검증 패스

        # 2. 검색 대상 텍스트 (제목 + 요약)
        target_text = (article.title + " " + article.summary).lower()

        # 3. 키워드 토큰화 (띄어쓰기 기준 분리)
        # 예: "Ajit Pawar" -> ["ajit", "pawar"]
        tokens = [t.lower() for t in clean_kw.split() if len(t) > 1]

        if not tokens:  # 토큰이 없거나 한 글자 단어뿐이면 검증 완화
            return clean_kw.lower() in target_text

        # 4. 토큰 중 하나라도 포함되어 있으면 통과 (OR 조건)
        # 너무 엄격하면(AND) 검색 유연성이 떨어지므로, 하나라도 매칭되면 관련 기사로 인정
        return any(token in target_text for token in tokens)

    def _parse_feed(self, content: str, keyword: str, max_results: int) -> list[Article]:
        """feedparser로 RSS 파싱"""
        feed = feedparser.parse(content)
        articles: list[Article] = []

        for entry in feed.entries[:max_results]:
            try:
                article = Article(
                    title=entry.get("title", ""),
                    link=entry.get("link", ""),
                    published=entry.get("published"),
                    source="google_news_rss",
                    keyword=keyword,
                    summary=entry.get("summary", ""),
                )

                # [추가] 관련성 검증
                if self._is_relevant(article, keyword):
                    articles.append(article)

            except Exception as e:
                logger.warning(f"Failed to parse article: {e}")
                continue

        return articles

    async def fetch(self, keyword: str, max_results: int = 20) -> CollectionResult:
        """
        키워드로 Google News RSS 검색 (관련성 필터 적용)
        """
        url = self.RSS_URL.format(keyword=quote(keyword))
        logger.info(f"Fetching Google News RSS for keyword: {keyword}")

        try:
            content = await self._fetch_with_retry(url)

            # 파싱 과정에서 _is_relevant 필터링 수행
            articles = self._parse_feed(content, keyword, max_results)

            # 필터링 후 개수 확인
            if not articles:
                logger.warning(
                    f"No relevant articles found for '{keyword}' (filtered unrelated results)"
                )

            logger.info(f"Successfully fetched {len(articles)} relevant articles for '{keyword}'")

            return CollectionResult(
                keyword=keyword,
                articles=articles,
                source="google_news_rss",
                success=True,
            )

        except Exception as e:
            logger.error(f"Failed to fetch RSS for '{keyword}': {e}")
            return CollectionResult(
                keyword=keyword,
                articles=[],
                source="google_news_rss",
                success=False,
                error_message=str(e),
            )

    async def fetch_multiple(
        self,
        keywords: list[str],
        max_results_per_keyword: int = 20,
        concurrency: int = 3,
    ) -> list[CollectionResult]:
        """여러 키워드 동시 수집"""
        semaphore = asyncio.Semaphore(concurrency)

        async def fetch_with_semaphore(kw: str) -> CollectionResult:
            async with semaphore:
                result = await self.fetch(kw, max_results_per_keyword)
                await asyncio.sleep(0.5)  # Rate limit 보호
                return result

        tasks = [fetch_with_semaphore(kw) for kw in keywords]
        return await asyncio.gather(*tasks)

    async def __aenter__(self) -> GoogleNewsRSSCollector:
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()


# CLI 테스트용
if __name__ == "__main__":

    async def main() -> None:
        async with GoogleNewsRSSCollector() as collector:
            # 테스트: 관련 없는 결과가 잘 걸러지는지 확인
            print("--- Testing #ajitpawar (Expect 0 relevant or only relevant) ---")
            result = await collector.fetch("#ajitpawar", max_results=5)
            print(f"Success: {result.success}")
            print(f"Relevant Articles found: {result.count}")
            for article in result.articles:
                print(f"  - {article.title}")

            print("\n--- Testing 트럼프 (Expect relevant results) ---")
            result = await collector.fetch("트럼프", max_results=5)
            print(f"Articles found: {result.count}")
            for article in result.articles:
                print(f"  - {article.title}")

    asyncio.run(main())
