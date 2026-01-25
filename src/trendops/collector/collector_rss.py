# src/trendops/collector/collector_rss.py
"""
Week 2 MVP용 RSS 수집기

Google News RSS를 사용한 키워드 기반 뉴스 수집
- 차단 위험 Zero (공식 RSS)
- 비동기 동작 (aiohttp + feedparser)
- 간단한 재시도 로직
"""
from __future__ import annotations

import asyncio
from datetime import datetime
from email.utils import parsedate_to_datetime
from typing import Any
from urllib.parse import quote

import aiohttp
import feedparser
from pydantic import BaseModel, Field, field_validator

import logging

logger = logging.getLogger(__name__)


# =============================================================================
# SCHEMA
# =============================================================================

class TrendDocument(BaseModel):
    """
    수집된 뉴스 문서 스키마
    
    VectorStore의 Document.create()와 호환되는 구조
    """
    title: str = Field(..., description="기사 제목")
    link: str = Field(..., description="기사 URL")
    published: datetime | None = Field(None, description="발행 시간")
    summary: str = Field(default="", description="기사 요약/설명")
    
    # 추가 메타데이터 (VectorStore 연동용)
    keyword: str = Field(default="", description="검색 키워드")
    source: str = Field(default="google_news_rss", description="수집 소스")
    
    @field_validator("published", mode="before")
    @classmethod
    def parse_published(cls, v: Any) -> datetime | None:
        """다양한 날짜 포맷 파싱"""
        if v is None:
            return None
        if isinstance(v, datetime):
            return v
        if isinstance(v, str):
            try:
                # RFC 2822 포맷 (RSS 표준)
                return parsedate_to_datetime(v)
            except (ValueError, TypeError):
                pass
            try:
                # ISO 포맷 시도
                return datetime.fromisoformat(v.replace("Z", "+00:00"))
            except (ValueError, TypeError):
                pass
        return None
    
    @property
    def text(self) -> str:
        """VectorStore 저장용 텍스트 (제목 + 요약)"""
        parts = [self.title]
        if self.summary:
            parts.append(self.summary)
        return " ".join(parts)
    
    class Config:
        json_schema_extra = {
            "example": {
                "title": "트럼프, 중국산 제품 25% 관세 부과 발표",
                "link": "https://news.google.com/...",
                "published": "2025-02-15T09:00:00",
                "summary": "미국 대통령이 새로운 관세 정책을 발표했다.",
                "keyword": "트럼프 관세",
                "source": "google_news_rss",
            }
        }


class CollectionResult(BaseModel):
    """수집 결과"""
    keyword: str
    documents: list[TrendDocument]
    collected_at: datetime = Field(default_factory=datetime.now)
    success: bool = True
    error_message: str | None = None
    
    @property
    def count(self) -> int:
        return len(self.documents)


# =============================================================================
# RSS COLLECTOR
# =============================================================================

class RSSCollector:
    """
    Google News RSS 수집기
    
    Usage:
        collector = RSSCollector()
        
        # 단일 키워드 수집
        result = await collector.fetch("트럼프 관세")
        for doc in result.documents:
            print(doc.title)
        
        # 여러 키워드 동시 수집
        results = await collector.fetch_multiple(["AI 반도체", "비트코인"])
    """
    
    # Google News RSS URL (한국어, 한국 지역)
    RSS_URL = "https://news.google.com/rss/search?q={keyword}&hl=ko&gl=KR&ceid=KR:ko"
    
    # 재시도 설정
    MAX_RETRIES = 3
    BASE_DELAY = 1.0
    
    def __init__(
        self,
        timeout_seconds: int = 30,
        max_results: int = 20,
    ):
        """
        Args:
            timeout_seconds: HTTP 요청 타임아웃
            max_results: 키워드당 최대 결과 수
        """
        self._timeout = aiohttp.ClientTimeout(total=timeout_seconds)
        self._max_results = max_results
        self._session: aiohttp.ClientSession | None = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """HTTP 세션 (재사용)"""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=self._timeout,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/120.0.0.0 Safari/537.36"
                    ),
                    "Accept": "application/rss+xml, application/xml, text/xml",
                },
            )
        return self._session
    
    async def close(self) -> None:
        """세션 종료"""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
    
    async def fetch(
        self,
        keyword: str,
        max_results: int | None = None,
    ) -> CollectionResult:
        """
        키워드로 Google News RSS 수집
        
        Args:
            keyword: 검색 키워드
            max_results: 최대 결과 수 (None이면 기본값)
            
        Returns:
            CollectionResult: 수집 결과
        """
        max_results = max_results or self._max_results
        url = self.RSS_URL.format(keyword=quote(keyword))
        
        logger.info(f"Fetching RSS for: {keyword}")
        
        try:
            # 재시도 로직 적용
            content = await self._fetch_with_retry(url)
            
            # feedparser로 파싱
            documents = self._parse_feed(content, keyword, max_results)
            
            logger.info(
                f"Fetched {len(documents)} documents",
                extra={"keyword": keyword, "count": len(documents)},
            )
            
            return CollectionResult(
                keyword=keyword,
                documents=documents,
                success=True,
            )
            
        except Exception as e:
            logger.error(f"RSS fetch failed: {e}", extra={"keyword": keyword})
            return CollectionResult(
                keyword=keyword,
                documents=[],
                success=False,
                error_message=str(e),
            )
    
    async def fetch_multiple(
        self,
        keywords: list[str],
        max_results_per_keyword: int | None = None,
        concurrency: int = 3,
    ) -> list[CollectionResult]:
        """
        여러 키워드 동시 수집 (동시성 제한)
        
        Args:
            keywords: 키워드 목록
            max_results_per_keyword: 키워드당 최대 결과 수
            concurrency: 동시 요청 수 제한
            
        Returns:
            list[CollectionResult]: 수집 결과 목록
        """
        semaphore = asyncio.Semaphore(concurrency)
        
        async def fetch_with_limit(kw: str) -> CollectionResult:
            async with semaphore:
                result = await self.fetch(kw, max_results_per_keyword)
                # Rate limit 보호 (요청 간 간격)
                await asyncio.sleep(0.5)
                return result
        
        tasks = [fetch_with_limit(kw) for kw in keywords]
        return await asyncio.gather(*tasks)
    
    async def _fetch_with_retry(self, url: str) -> str:
        """지수 백오프를 적용한 HTTP 요청"""
        last_error: Exception | None = None
        
        for attempt in range(self.MAX_RETRIES):
            try:
                session = await self._get_session()
                async with session.get(url) as response:
                    response.raise_for_status()
                    return await response.text()
                    
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                last_error = e
                
                if attempt < self.MAX_RETRIES - 1:
                    delay = self.BASE_DELAY * (2 ** attempt)
                    logger.warning(
                        f"Retry {attempt + 1}/{self.MAX_RETRIES} in {delay:.1f}s: {e}"
                    )
                    await asyncio.sleep(delay)
        
        raise last_error or Exception("Unknown fetch error")
    
    def _parse_feed(
        self,
        content: str,
        keyword: str,
        max_results: int,
    ) -> list[TrendDocument]:
        """feedparser로 RSS 파싱"""
        feed = feedparser.parse(content)
        documents: list[TrendDocument] = []
        
        for entry in feed.entries[:max_results]:
            try:
                doc = TrendDocument(
                    title=entry.get("title", "").strip(),
                    link=entry.get("link", ""),
                    published=entry.get("published"),
                    summary=self._clean_summary(entry.get("summary", "")),
                    keyword=keyword,
                    source="google_news_rss",
                )
                
                # 제목이 비어있으면 스킵
                if doc.title:
                    documents.append(doc)
                    
            except Exception as e:
                logger.warning(f"Failed to parse entry: {e}")
                continue
        
        return documents
    
    @staticmethod
    def _clean_summary(summary: str) -> str:
        """HTML 태그 제거 및 정리"""
        import re
        
        if not summary:
            return ""
        
        # HTML 태그 제거
        clean = re.sub(r"<[^>]+>", "", summary)
        # 연속 공백 정리
        clean = re.sub(r"\s+", " ", clean)
        return clean.strip()
    
    # Context Manager 지원
    async def __aenter__(self) -> RSSCollector:
        return self
    
    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

async def fetch_news(
    keyword: str,
    max_results: int = 20,
) -> list[TrendDocument]:
    """
    간편 뉴스 수집 함수
    
    Args:
        keyword: 검색 키워드
        max_results: 최대 결과 수
        
    Returns:
        list[TrendDocument]: 수집된 문서 목록
        
    Usage:
        from trendops.collector.collector_rss import fetch_news
        
        docs = await fetch_news("트럼프 관세", max_results=10)
        for doc in docs:
            print(f"[{doc.published}] {doc.title}")
    """
    async with RSSCollector(max_results=max_results) as collector:
        result = await collector.fetch(keyword)
        return result.documents


async def fetch_news_multiple(
    keywords: list[str],
    max_results_per_keyword: int = 20,
) -> dict[str, list[TrendDocument]]:
    """
    여러 키워드 뉴스 수집
    
    Args:
        keywords: 키워드 목록
        max_results_per_keyword: 키워드당 최대 결과 수
        
    Returns:
        dict[str, list[TrendDocument]]: 키워드별 문서 목록
    """
    async with RSSCollector(max_results=max_results_per_keyword) as collector:
        results = await collector.fetch_multiple(keywords)
        return {r.keyword: r.documents for r in results}


# =============================================================================
# CLI TEST
# =============================================================================

if __name__ == "__main__":
    async def main() -> None:
        print("=" * 60)
        print("RSS Collector Test (Google News)")
        print("=" * 60)
        
        async with RSSCollector() as collector:
            # 단일 키워드 테스트
            print("\n[1] Single keyword fetch...")
            result = await collector.fetch("AI 반도체", max_results=5)
            
            print(f"    Keyword: {result.keyword}")
            print(f"    Success: {result.success}")
            print(f"    Count: {result.count}")
            
            if result.documents:
                print("\n    Documents:")
                for i, doc in enumerate(result.documents, 1):
                    pub = doc.published.strftime("%Y-%m-%d %H:%M") if doc.published else "N/A"
                    print(f"    {i}. [{pub}] {doc.title[:50]}...")
                    print(f"       Link: {doc.link[:60]}...")
            
            # 여러 키워드 테스트
            print("\n[2] Multiple keywords fetch...")
            keywords = ["트럼프", "비트코인"]
            results = await collector.fetch_multiple(keywords, max_results_per_keyword=3)
            
            for r in results:
                print(f"\n    [{r.keyword}] {r.count} documents")
                for doc in r.documents[:2]:
                    print(f"      - {doc.title[:40]}...")
            
            # TrendDocument 속성 테스트
            print("\n[3] TrendDocument properties...")
            if result.documents:
                doc = result.documents[0]
                print(f"    title: {doc.title[:50]}...")
                print(f"    text (for VectorStore): {doc.text[:80]}...")
                print(f"    keyword: {doc.keyword}")
                print(f"    source: {doc.source}")
        
        print("\n" + "=" * 60)
        print("✅ RSS Collector test completed!")
        print("=" * 60)
    
    asyncio.run(main())