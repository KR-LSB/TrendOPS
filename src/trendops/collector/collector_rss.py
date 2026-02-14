# src/trendops/collector/collector_rss.py
"""
Google News RSS Collector (Standardized)

Week 3 Refactoring:
- Inherits from BaseCollector
- Uses standardized TrendDocument
"""
from __future__ import annotations

import logging
from datetime import datetime
from email.utils import parsedate_to_datetime
from typing import Any
from urllib.parse import quote

import aiohttp
import feedparser

# [변경] 공통 규격 임포트
from trendops.collector.collector_base import BaseCollector, TrendDocument

logger = logging.getLogger(__name__)


class RSSCollector(BaseCollector):
    """
    Google News RSS 수집기 (BaseCollector 구현체)
    """

    RSS_URL = "https://news.google.com/rss/search?q={keyword}&hl=ko&gl=KR&ceid=KR:ko"
    MAX_RETRIES = 3
    BASE_DELAY = 1.0

    def __init__(self, max_results: int = 20, timeout_seconds: int = 30):
        super().__init__(max_results=max_results, timeout_seconds=timeout_seconds)
        self._max_results = max_results
        self._timeout = aiohttp.ClientTimeout(total=timeout_seconds)
        self._session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=self._timeout,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                },
            )
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    async def fetch(self, keyword: str, **kwargs) -> list[TrendDocument]:
        """
        RSS 데이터를 가져와 표준 TrendDocument 리스트로 반환
        """
        max_results = kwargs.get("max_results", self._max_results)
        url = self.RSS_URL.format(keyword=quote(keyword))

        logger.info(f"Fetching RSS for: {keyword}")

        try:
            content = await self._fetch_with_retry(url)
            documents = self._parse_feed(content, keyword, max_results)

            if not documents:
                logger.warning(f"No relevant documents found for '{keyword}'")
            else:
                logger.info(f"Fetched {len(documents)} documents")

            return documents

        except Exception as e:
            logger.error(f"RSS fetch failed: {e}")
            return []

    async def _fetch_with_retry(self, url: str) -> str:
        import asyncio

        last_error = None
        for attempt in range(self.MAX_RETRIES):
            try:
                session = await self._get_session()
                async with session.get(url) as response:
                    response.raise_for_status()
                    return await response.text()
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                last_error = e
                if attempt < self.MAX_RETRIES - 1:
                    await asyncio.sleep(self.BASE_DELAY * (2**attempt))
        raise last_error or Exception("Unknown fetch error")

    def _parse_feed(self, content: str, keyword: str, max_results: int) -> list[TrendDocument]:
        feed = feedparser.parse(content)
        documents: list[TrendDocument] = []

        for entry in feed.entries[:max_results]:
            try:
                title = entry.get("title", "").strip()
                summary = self._clean_summary(entry.get("summary", ""))

                # 관련성 필터
                if not self._is_relevant(title, summary, keyword):
                    continue

                # [변경] 표준 TrendDocument 생성
                doc = TrendDocument(
                    title=title,
                    link=entry.get("link", ""),
                    published=self._parse_published(entry.get("published")),
                    summary=summary,
                    keyword=keyword,
                    source="google_news_rss",
                    metadata={
                        "guid": entry.get("id", ""),
                        "author": entry.get("author", "unknown"),
                    },
                )
                documents.append(doc)

            except Exception:
                continue

        return documents

    def _is_relevant(self, title: str, summary: str, keyword: str) -> bool:
        """키워드 관련성 검증"""
        clean_kw = keyword.replace("#", "").strip()
        if not clean_kw:
            return True
        target_text = (title + " " + summary).lower()
        tokens = [t.lower() for t in clean_kw.split() if len(t) > 1]
        if not tokens:
            return clean_kw.lower() in target_text
        return any(token in target_text for token in tokens)

    @staticmethod
    def _clean_summary(summary: str) -> str:
        import re

        if not summary:
            return ""
        clean = re.sub(r"<[^>]+>", "", summary)
        clean = re.sub(r"\s+", " ", clean)
        return clean.strip()

    @staticmethod
    def _parse_published(v: Any) -> datetime | None:
        if not v:
            return None
        if isinstance(v, datetime):
            return v
        if isinstance(v, str):
            try:
                return parsedate_to_datetime(v)
            except:
                pass
            try:
                return datetime.fromisoformat(v.replace("Z", "+00:00"))
            except:
                pass
        return None
