# src/trendops/collector/collector_youtube.py
"""
YouTube 댓글 수집기 (Standardized)

Week 3 Refactoring:
- Inherits from BaseCollector
- Returns List[TrendDocument]
- Playwright + Stealth 모드 유지
"""

from __future__ import annotations

import asyncio
import logging
import random
from datetime import datetime
from enum import Enum
from urllib.parse import quote

import aiohttp
from playwright.async_api import (
    Browser,
    BrowserContext,
    Page,
    Playwright,
    async_playwright,
)
from playwright_stealth import Stealth
from pydantic import BaseModel

# [변경] 표준 규격 임포트
from trendops.collector.collector_base import BaseCollector, TrendDocument

logger = logging.getLogger(__name__)


# =============================================================================
# Custom Exceptions & Models (기존 유지)
# =============================================================================


class YouTubeCollectorError(Exception):
    pass


class BlockedError(YouTubeCollectorError):
    pass


class RateLimitError(YouTubeCollectorError):
    pass


class NetworkError(YouTubeCollectorError):
    pass


class NoResultsError(YouTubeCollectorError):
    pass


class CommentSource(str, Enum):
    PLAYWRIGHT = "youtube_playwright"
    RSS = "youtube_rss"


class Comment(BaseModel):
    video_id: str
    video_title: str
    comment_text: str
    author: str
    likes: int = 0
    published_at: datetime | None = None
    source: CommentSource = CommentSource.PLAYWRIGHT


class VideoInfo(BaseModel):
    video_id: str
    title: str
    channel: str = ""
    url: str
    view_count: str = ""


class YouTubeConfig(BaseModel):
    headless: bool = True
    video_delay_min: float = 2.0
    video_delay_max: float = 4.0
    scroll_delay_min: float = 1.0
    scroll_delay_max: float = 2.0
    page_timeout: int = 30000
    max_scroll_attempts: int = 3

    # Retry 설정 내장
    max_retries: int = 3


# =============================================================================
# YouTube Collector Class (Standardized)
# =============================================================================


class YouTubeCollector(BaseCollector):
    """
    YouTube 댓글 수집기 (BaseCollector 구현체)
    """

    # Constants
    SEARCH_URL = "https://www.youtube.com/results?search_query={keyword}"
    VIDEO_URL = "https://www.youtube.com/watch?v={video_id}"
    BLOCK_INDICATORS = [
        "unusual traffic",
        "captcha",
        "verify you're not a robot",
        "confirm you're not a robot",
        "sorry, something went wrong",
    ]

    def __init__(self, config: YouTubeConfig | None = None, **kwargs):
        super().__init__(**kwargs)
        self.config = config or YouTubeConfig()

        # Playwright Resources
        self._playwright: Playwright | None = None
        self._browser: Browser | None = None
        self._context: BrowserContext | None = None
        self._http_session: aiohttp.ClientSession | None = None

    # -------------------------------------------------------------------------
    # BaseCollector Interface Implementation
    # -------------------------------------------------------------------------

    async def fetch(self, keyword: str, **kwargs) -> list[TrendDocument]:
        """
        [Standard Interface] 키워드로 유튜브 댓글을 수집하여 TrendDocument 리스트 반환
        """
        # kwargs에서 설정 가져오기 (기본값 설정)
        max_videos = kwargs.get("max_videos", 3)
        comments_per_video = kwargs.get("comments_per_video", 20)

        try:
            # 브라우저 초기화 (아직 안 되어 있다면)
            if not self._browser:
                await self._init_browser()

            logger.info(
                f"Fetching YouTube comments for '{keyword}' (Videos: {max_videos}, Comments/v: {comments_per_video})"
            )

            # 기존 로직 재사용
            raw_comments = await self.fetch_comments(keyword, max_videos, comments_per_video)

            # TrendDocument로 변환
            documents = []
            for c in raw_comments:
                doc = TrendDocument(
                    title=f"Comment on: {c.video_title}",
                    link=f"https://www.youtube.com/watch?v={c.video_id}",
                    summary=c.comment_text,
                    keyword=keyword,
                    source="youtube_comment",
                    published=c.published_at,
                    metadata={
                        "author": c.author,
                        "likes": c.likes,
                        "video_id": c.video_id,
                        "video_title": c.video_title,
                        "source_type": c.source.value,
                    },
                )
                documents.append(doc)

            logger.info(f"Converted {len(documents)} YouTube comments to documents")
            return documents

        except Exception as e:
            logger.error(f"YouTube fetch failed: {e}")
            return []

    async def close(self) -> None:
        """리소스 정리"""
        if self._context:
            await self._context.close()
            self._context = None
        if self._browser:
            await self._browser.close()
            self._browser = None
        if self._playwright:
            await self._playwright.stop()
            self._playwright = None
        if self._http_session and not self._http_session.closed:
            await self._http_session.close()

    # -------------------------------------------------------------------------
    # Internal Logic (Playwright)
    # -------------------------------------------------------------------------

    async def _init_browser(self) -> None:
        """Playwright 브라우저 초기화"""
        logger.info("Initializing Playwright browser...")
        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(
            headless=self.config.headless,
            args=["--disable-blink-features=AutomationControlled", "--no-sandbox"],
        )
        self._context = await self._browser.new_context(
            viewport={"width": 1920, "height": 1080},
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            locale="ko-KR",
            timezone_id="Asia/Seoul",
        )

    async def fetch_comments(
        self, keyword: str, max_videos: int, comments_per_video: int
    ) -> list[Comment]:
        """실제 수집 로직 (기존 코드 통합)"""
        all_comments = []

        try:
            videos = await self._search_videos(keyword, max_videos)

            for i, video in enumerate(videos):
                try:
                    comments = await self._fetch_video_comments(video, comments_per_video)
                    all_comments.extend(comments)

                    if i < len(videos) - 1:
                        await asyncio.sleep(
                            random.uniform(self.config.video_delay_min, self.config.video_delay_max)
                        )

                except Exception as e:
                    logger.warning(f"Error processing video {video.video_id}: {e}")
                    continue

        except Exception as e:
            logger.error(f"Error in fetch_comments: {e}")

        return all_comments

    async def _create_page(self) -> Page:
        page = await self._context.new_page()
        await Stealth().apply_stealth_async(page)
        return page

    async def _search_videos(self, keyword: str, max_videos: int) -> list[VideoInfo]:
        page = await self._create_page()
        videos = []
        try:
            url = self.SEARCH_URL.format(keyword=quote(keyword))
            await page.goto(url, wait_until="networkidle")

            # 차단 감지 생략 (간소화)
            await asyncio.sleep(2)

            elements = await page.query_selector_all("ytd-video-renderer")
            for el in elements[:max_videos]:
                try:
                    title_el = await el.query_selector("#video-title")
                    if not title_el:
                        continue

                    title = await title_el.get_attribute("title")
                    href = await title_el.get_attribute("href")

                    if title and href and "watch?v=" in href:
                        video_id = href.split("v=")[1].split("&")[0]
                        videos.append(
                            VideoInfo(
                                video_id=video_id,
                                title=title,
                                url=f"https://www.youtube.com/watch?v={video_id}",
                            )
                        )
                except:
                    continue
        finally:
            await page.close()
        return videos

    async def _fetch_video_comments(self, video: VideoInfo, max_comments: int) -> list[Comment]:
        page = await self._create_page()
        comments = []
        try:
            await page.goto(
                video.url, wait_until="domcontentloaded"
            )  # networkidle은 너무 느릴 수 있음
            await asyncio.sleep(2)

            # 스크롤해서 댓글 섹션 로드
            await page.evaluate("window.scrollBy(0, 500)")
            await asyncio.sleep(1)

            # 댓글 로딩 대기
            attempts = 0
            while len(comments) < max_comments and attempts < self.config.max_scroll_attempts:
                await page.evaluate("window.scrollBy(0, 1000)")
                await asyncio.sleep(1.5)

                elements = await page.query_selector_all("ytd-comment-thread-renderer")
                for el in elements:
                    if len(comments) >= max_comments:
                        break

                    try:
                        content_el = await el.query_selector("#content-text")
                        author_el = await el.query_selector("#author-text")

                        if content_el and author_el:
                            text = await content_el.inner_text()
                            author = await author_el.inner_text()

                            # 중복 방지
                            if not any(c.comment_text == text for c in comments):
                                comments.append(
                                    Comment(
                                        video_id=video.video_id,
                                        video_title=video.title,
                                        comment_text=text.strip(),
                                        author=author.strip(),
                                    )
                                )
                    except:
                        continue
                attempts += 1

        except Exception as e:
            logger.warning(f"Failed to fetch comments for {video.video_id}: {e}")
        finally:
            await page.close()
        return comments
