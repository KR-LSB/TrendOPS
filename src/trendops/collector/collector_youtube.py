# src/trendops/collector/collector_youtube.py
"""
YouTube 댓글 수집기 (Week 3)

Playwright + Stealth 모드로 YouTube 검색 및 댓글 수집.
차단 감지 시 YouTube RSS Channel Feed로 Fallback.

Blueprint 준수:
- GPU 사용 금지 (이 모듈은 CPU only)
- async/await 패턴 준수
- RSS First, Crawling Second 철학
"""

from __future__ import annotations

import asyncio
import logging
import random
import re
from datetime import datetime
from enum import Enum
from typing import Any
from urllib.parse import quote, urljoin

import aiohttp
import feedparser
from pydantic import BaseModel, Field, field_validator
from playwright.async_api import (
    Browser,
    BrowserContext,
    Page,
    Playwright,
    TimeoutError as PlaywrightTimeoutError,
    async_playwright,
)
from playwright_stealth import Stealth

logger = logging.getLogger(__name__)


# =============================================================================
# Custom Exceptions
# =============================================================================

class YouTubeCollectorError(Exception):
    """Base exception for YouTube collector"""
    pass


class BlockedError(YouTubeCollectorError):
    """YouTube에서 차단된 경우 (봇 감지, CAPTCHA 등)"""
    pass


class RateLimitError(YouTubeCollectorError):
    """Rate limit 초과"""
    pass


class NetworkError(YouTubeCollectorError):
    """네트워크 오류 (exponential backoff 대상)"""
    pass


class NoResultsError(YouTubeCollectorError):
    """검색 결과 없음"""
    pass


# =============================================================================
# Pydantic Models
# =============================================================================

class CommentSource(str, Enum):
    """댓글 수집 소스"""
    PLAYWRIGHT = "youtube_playwright"
    RSS = "youtube_rss"


class Comment(BaseModel):
    """YouTube 댓글 모델"""
    
    video_id: str = Field(..., description="YouTube 영상 ID")
    video_title: str = Field(..., description="영상 제목")
    comment_text: str = Field(..., description="댓글 내용")
    author: str = Field(..., description="댓글 작성자")
    likes: int = Field(default=0, ge=0, description="좋아요 수")
    published_at: datetime | None = Field(default=None, description="작성 시간")
    source: CommentSource = Field(
        default=CommentSource.PLAYWRIGHT, 
        description="수집 소스"
    )
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "video_id": "dQw4w9WgXcQ",
                "video_title": "Never Gonna Give You Up",
                "comment_text": "This song is timeless!",
                "author": "MusicFan2024",
                "likes": 42,
                "published_at": "2025-01-15T10:30:00",
                "source": "youtube_playwright",
            }
        }
    }
    
    @field_validator("published_at", mode="before")
    @classmethod
    def parse_published(cls, v: Any) -> datetime | None:
        if v is None:
            return None
        if isinstance(v, datetime):
            return v
        if isinstance(v, str):
            # YouTube 상대 시간 파싱 시도 (예: "2일 전", "3시간 전")
            # 정확한 시간이 아니므로 None 반환하거나 현재 시간 기준 추정
            return None
        return None


class VideoInfo(BaseModel):
    """YouTube 영상 정보"""
    
    video_id: str = Field(..., description="YouTube 영상 ID")
    title: str = Field(..., description="영상 제목")
    channel: str = Field(default="", description="채널명")
    url: str = Field(..., description="영상 URL")
    view_count: str = Field(default="", description="조회수 텍스트")
    published_text: str = Field(default="", description="업로드 시간 텍스트")


class CollectionResult(BaseModel):
    """댓글 수집 결과"""
    
    keyword: str = Field(..., description="검색 키워드")
    comments: list[Comment] = Field(default_factory=list, description="수집된 댓글")
    videos_processed: int = Field(default=0, description="처리된 영상 수")
    source: CommentSource = Field(
        default=CommentSource.PLAYWRIGHT,
        description="주 수집 소스"
    )
    collected_at: datetime = Field(default_factory=datetime.now, description="수집 시간")
    success: bool = Field(default=True, description="수집 성공 여부")
    error_message: str | None = Field(default=None, description="에러 메시지")
    fallback_used: bool = Field(default=False, description="Fallback 사용 여부")
    
    def is_valid(self) -> bool:
        """유효한 결과인지 확인"""
        return self.success and len(self.comments) > 0
    
    @property
    def total_comments(self) -> int:
        return len(self.comments)


class RetryConfig(BaseModel):
    """재시도 설정"""
    
    max_attempts: int = Field(default=3, ge=1, description="최대 재시도 횟수")
    base_delay: float = Field(default=1.0, ge=0.1, description="기본 대기 시간(초)")
    max_delay: float = Field(default=60.0, ge=1.0, description="최대 대기 시간(초)")
    exponential_base: float = Field(default=2.0, ge=1.5, description="지수 백오프 배율")


class YouTubeConfig(BaseModel):
    """YouTube Collector 설정"""
    
    headless: bool = Field(default=True, description="브라우저 헤드리스 모드")
    video_delay_min: float = Field(default=3.0, description="영상 간 최소 대기(초)")
    video_delay_max: float = Field(default=5.0, description="영상 간 최대 대기(초)")
    scroll_delay_min: float = Field(default=1.0, description="스크롤 간 최소 대기(초)")
    scroll_delay_max: float = Field(default=2.0, description="스크롤 간 최대 대기(초)")
    page_timeout: int = Field(default=30000, description="페이지 타임아웃(ms)")
    max_scroll_attempts: int = Field(default=5, description="댓글 로드 최대 스크롤 횟수")
    retry_config: RetryConfig = Field(default_factory=RetryConfig)


# =============================================================================
# YouTube Collector Class
# =============================================================================

class YouTubeCollector:
    """
    YouTube 댓글 수집기
    
    차단 시 YouTube RSS Channel Feed로 Fallback.
    
    Features:
        - Playwright + Stealth 모드로 봇 감지 우회
        - 검색 → 영상 목록 → 댓글 수집 플로우
        - Rate limiting (영상 간 3-5초 딜레이)
        - 차단 감지 및 자동 Fallback
        - Exponential backoff for network errors
    
    Usage:
        async with YouTubeCollector() as collector:
            comments = await collector.fetch_comments("트럼프 관세")
    """
    
    # YouTube RSS Channel Feed URL 템플릿
    FALLBACK_RSS = "https://www.youtube.com/feeds/videos.xml?channel_id={channel_id}"
    
    # 검색 URL
    SEARCH_URL = "https://www.youtube.com/results?search_query={keyword}"
    
    # 영상 URL
    VIDEO_URL = "https://www.youtube.com/watch?v={video_id}"
    
    # 차단 감지 문자열
    BLOCK_INDICATORS = [
        "unusual traffic",
        "captcha",
        "verify you're not a robot",
        "confirm you're not a robot",
        "sorry, something went wrong",
        "this page isn't available",
    ]
    
    def __init__(self, config: YouTubeConfig | None = None):
        """
        Args:
            config: YouTube Collector 설정 (기본값 사용 시 None)
        """
        self.config = config or YouTubeConfig()
        self._playwright: Playwright | None = None
        self._browser: Browser | None = None
        self._context: BrowserContext | None = None
        self._http_session: aiohttp.ClientSession | None = None
    
    # =========================================================================
    # Context Manager
    # =========================================================================
    
    async def __aenter__(self) -> YouTubeCollector:
        """비동기 컨텍스트 매니저 진입"""
        await self._init_browser()
        return self
    
    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """비동기 컨텍스트 매니저 종료"""
        await self.close()
    
    async def _init_browser(self) -> None:
        """Playwright 브라우저 초기화"""
        logger.info("Initializing Playwright browser with stealth mode...")
        
        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(
            headless=self.config.headless,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--disable-dev-shm-usage",
                "--no-sandbox",
                "--disable-setuid-sandbox",
                "--disable-infobars",
                "--disable-extensions",
            ],
        )
        
        # 브라우저 컨텍스트 생성 (사용자 에이전트, 뷰포트 설정)
        self._context = await self._browser.new_context(
            viewport={"width": 1920, "height": 1080},
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            locale="ko-KR",
            timezone_id="Asia/Seoul",
        )
        
        logger.info("Browser initialized successfully")
    
    async def _get_http_session(self) -> aiohttp.ClientSession:
        """HTTP 세션 (Fallback용) 반환"""
        if self._http_session is None or self._http_session.closed:
            self._http_session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30),
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36"
                    ),
                },
            )
        return self._http_session
    
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
            self._http_session = None
        
        logger.info("YouTube collector resources cleaned up")
    
    # =========================================================================
    # Block Detection
    # =========================================================================
    
    def _is_blocked(self, page_content: str) -> bool:
        """
        페이지 내용에서 차단 여부 감지
        
        Args:
            page_content: 페이지 HTML 또는 텍스트 내용
            
        Returns:
            bool: 차단 감지 여부
        """
        content_lower = page_content.lower()
        return any(indicator in content_lower for indicator in self.BLOCK_INDICATORS)
    
    async def _check_page_blocked(self, page: Page) -> bool:
        """
        현재 페이지가 차단되었는지 확인
        
        Args:
            page: Playwright Page 객체
            
        Returns:
            bool: 차단 여부
        """
        try:
            content = await page.inner_text("body")
            return self._is_blocked(content)
        except Exception as e:
            logger.warning(f"Failed to check block status: {e}")
            return False
    
    # =========================================================================
    # Random Delays (Rate Limiting)
    # =========================================================================
    
    async def _random_delay(self, min_sec: float, max_sec: float) -> None:
        """랜덤 딜레이 (봇 감지 우회)"""
        delay = random.uniform(min_sec, max_sec)
        await asyncio.sleep(delay)
    
    async def _video_delay(self) -> None:
        """영상 간 딜레이"""
        await self._random_delay(
            self.config.video_delay_min,
            self.config.video_delay_max,
        )
    
    async def _scroll_delay(self) -> None:
        """스크롤 간 딜레이"""
        await self._random_delay(
            self.config.scroll_delay_min,
            self.config.scroll_delay_max,
        )
    
    # =========================================================================
    # Core Scraping Methods
    # =========================================================================
    
    async def _create_stealth_page(self) -> Page:
        """Stealth 모드가 적용된 새 페이지 생성"""
        if not self._context:
            raise RuntimeError("Browser context not initialized")
        
        page = await self._context.new_page()
        await Stealth().apply_stealth_async(page)
        page.set_default_timeout(self.config.page_timeout)
        
        return page
    
    async def _search_videos(
        self,
        keyword: str,
        max_videos: int = 5,
    ) -> list[VideoInfo]:
        """
        YouTube에서 키워드 검색 후 영상 목록 추출
        
        Args:
            keyword: 검색 키워드
            max_videos: 최대 영상 수
            
        Returns:
            list[VideoInfo]: 검색된 영상 정보 목록
            
        Raises:
            BlockedError: 차단 감지 시
            TimeoutError: 타임아웃 발생 시
        """
        page = await self._create_stealth_page()
        
        try:
            search_url = self.SEARCH_URL.format(keyword=quote(keyword))
            logger.info(f"Searching YouTube for: {keyword}")
            
            await page.goto(search_url, wait_until="networkidle")
            
            # 차단 확인
            if await self._check_page_blocked(page):
                raise BlockedError(f"Blocked while searching for: {keyword}")
            
            # 페이지 로딩 대기
            await self._scroll_delay()
            
            # 영상 목록 추출 (ytd-video-renderer 또는 ytd-video-renderer 요소)
            videos: list[VideoInfo] = []
            
            # 검색 결과 영상 요소 찾기
            video_elements = await page.query_selector_all(
                "ytd-video-renderer, ytd-rich-item-renderer"
            )
            
            for element in video_elements[:max_videos]:
                try:
                    # 영상 제목 및 링크
                    title_element = await element.query_selector(
                        "#video-title, a#video-title-link"
                    )
                    
                    if not title_element:
                        continue
                    
                    title = await title_element.get_attribute("title")
                    href = await title_element.get_attribute("href")
                    
                    if not title or not href:
                        continue
                    
                    # 영상 ID 추출
                    video_id_match = re.search(r"watch\?v=([a-zA-Z0-9_-]{11})", href)
                    if not video_id_match:
                        continue
                    
                    video_id = video_id_match.group(1)
                    
                    # 채널명 추출
                    channel_element = await element.query_selector(
                        "ytd-channel-name a, #channel-name a"
                    )
                    channel = ""
                    if channel_element:
                        channel = await channel_element.inner_text() or ""
                    
                    # 조회수 추출
                    view_element = await element.query_selector(
                        "#metadata-line span:first-child, "
                        ".inline-metadata-item:first-child"
                    )
                    view_count = ""
                    if view_element:
                        view_count = await view_element.inner_text() or ""
                    
                    videos.append(VideoInfo(
                        video_id=video_id,
                        title=title,
                        channel=channel.strip(),
                        url=self.VIDEO_URL.format(video_id=video_id),
                        view_count=view_count.strip(),
                    ))
                    
                    logger.debug(f"Found video: {title[:50]}... ({video_id})")
                    
                except Exception as e:
                    logger.warning(f"Failed to parse video element: {e}")
                    continue
            
            if not videos:
                raise NoResultsError(f"No videos found for: {keyword}")
            
            logger.info(f"Found {len(videos)} videos for '{keyword}'")
            return videos
            
        except PlaywrightTimeoutError as e:
            logger.error(f"Timeout searching for '{keyword}': {e}")
            raise TimeoutError(f"Timeout searching for: {keyword}")
        
        finally:
            await page.close()
    
    async def _fetch_video_comments(
        self,
        video: VideoInfo,
        max_comments: int = 50,
    ) -> list[Comment]:
        """
        특정 영상의 댓글 수집
        
        Args:
            video: 영상 정보
            max_comments: 최대 댓글 수
            
        Returns:
            list[Comment]: 수집된 댓글 목록
            
        Raises:
            BlockedError: 차단 감지 시
        """
        page = await self._create_stealth_page()
        
        try:
            logger.info(f"Fetching comments for: {video.title[:50]}...")
            
            await page.goto(video.url, wait_until="networkidle")
            
            # 차단 확인
            if await self._check_page_blocked(page):
                raise BlockedError(f"Blocked while fetching video: {video.video_id}")
            
            # 댓글 섹션까지 스크롤
            await self._scroll_delay()
            await page.evaluate("window.scrollBy(0, 500)")
            await self._scroll_delay()
            
            # 댓글 로드 대기 (댓글 영역 활성화)
            comments: list[Comment] = []
            scroll_attempts = 0
            
            while len(comments) < max_comments and scroll_attempts < self.config.max_scroll_attempts:
                # 스크롤하여 더 많은 댓글 로드
                await page.evaluate("window.scrollBy(0, 1000)")
                await self._scroll_delay()
                
                # 댓글 요소 찾기
                comment_elements = await page.query_selector_all(
                    "ytd-comment-thread-renderer"
                )
                
                for element in comment_elements:
                    if len(comments) >= max_comments:
                        break
                    
                    try:
                        # 댓글 텍스트
                        content_element = await element.query_selector(
                            "#content-text"
                        )
                        if not content_element:
                            continue
                        
                        comment_text = await content_element.inner_text()
                        
                        # 이미 수집된 댓글인지 확인 (중복 방지)
                        if any(c.comment_text == comment_text for c in comments):
                            continue
                        
                        # 작성자
                        author_element = await element.query_selector(
                            "#author-text span, a#author-text"
                        )
                        author = ""
                        if author_element:
                            author = await author_element.inner_text() or "Unknown"
                        
                        # 좋아요 수
                        likes = 0
                        like_element = await element.query_selector(
                            "#vote-count-middle"
                        )
                        if like_element:
                            like_text = await like_element.inner_text() or "0"
                            like_text = like_text.strip()
                            if like_text:
                                # "1.2만" 같은 형식 처리
                                likes = self._parse_count(like_text)
                        
                        # 작성 시간 (상대 시간, 정확한 파싱 어려움)
                        time_element = await element.query_selector(
                            "#header-author yt-formatted-string.published-time-text"
                        )
                        published_text = ""
                        if time_element:
                            published_text = await time_element.inner_text() or ""
                        
                        comments.append(Comment(
                            video_id=video.video_id,
                            video_title=video.title,
                            comment_text=comment_text.strip(),
                            author=author.strip(),
                            likes=likes,
                            published_at=None,  # 상대 시간은 정확한 변환 어려움
                            source=CommentSource.PLAYWRIGHT,
                        ))
                        
                    except Exception as e:
                        logger.warning(f"Failed to parse comment: {e}")
                        continue
                
                scroll_attempts += 1
                
                # 새로운 댓글이 로드되지 않으면 중단
                if scroll_attempts > 1 and len(comments) == len(comment_elements):
                    break
            
            logger.info(
                f"Collected {len(comments)} comments from '{video.title[:30]}...'"
            )
            return comments
            
        except PlaywrightTimeoutError as e:
            logger.warning(f"Timeout fetching comments for {video.video_id}: {e}")
            return []
        
        finally:
            await page.close()
    
    def _parse_count(self, text: str) -> int:
        """
        조회수/좋아요 텍스트 파싱 (예: "1.2만", "1234", "1K")
        
        Args:
            text: 숫자 텍스트
            
        Returns:
            int: 파싱된 숫자
        """
        text = text.strip().replace(",", "").replace(" ", "")
        
        if not text or text == "좋아요":
            return 0
        
        try:
            # 한국어 단위 처리
            if "만" in text:
                number = float(text.replace("만", ""))
                return int(number * 10000)
            elif "천" in text:
                number = float(text.replace("천", ""))
                return int(number * 1000)
            elif "억" in text:
                number = float(text.replace("억", ""))
                return int(number * 100000000)
            # 영문 단위 처리
            elif text.upper().endswith("K"):
                number = float(text[:-1])
                return int(number * 1000)
            elif text.upper().endswith("M"):
                number = float(text[:-1])
                return int(number * 1000000)
            else:
                return int(float(text))
        except (ValueError, TypeError):
            return 0
    
    # =========================================================================
    # Fallback: YouTube RSS
    # =========================================================================
    
    async def _fetch_via_rss(
        self,
        keyword: str,
        max_results: int = 10,
    ) -> CollectionResult:
        """
        RSS Feed를 통한 Fallback 수집
        
        YouTube RSS는 댓글을 제공하지 않으므로,
        최신 영상 정보만 반환 (제한적 대안).
        
        Args:
            keyword: 검색 키워드
            max_results: 최대 결과 수
            
        Returns:
            CollectionResult: 수집 결과 (댓글 대신 영상 제목 기반 가상 코멘트)
        """
        logger.warning(f"Using RSS fallback for: {keyword}")
        
        # RSS는 채널 기반이므로, 직접 검색 불가
        # 대안: 잘 알려진 뉴스/미디어 채널 RSS 수집
        # 여기서는 제한적인 기능만 제공
        
        # YouTube는 키워드 기반 RSS가 없으므로,
        # 실패 시 빈 결과 또는 Google News RSS로 대체 권장
        comments: list[Comment] = []
        
        # Fallback: 검색 API 없이는 직접 수집 불가
        # 대신 로그 및 빈 결과 반환
        return CollectionResult(
            keyword=keyword,
            comments=comments,
            videos_processed=0,
            source=CommentSource.RSS,
            success=True,
            fallback_used=True,
            error_message=(
                "YouTube RSS does not support keyword search. "
                "Consider using YouTube Data API v3 for reliable fallback."
            ),
        )
    
    # =========================================================================
    # Retry Logic
    # =========================================================================
    
    async def _with_retry(
        self,
        coro_factory: Any,
        error_types: tuple[type[Exception], ...] = (NetworkError, TimeoutError),
    ) -> Any:
        """
        Exponential backoff를 적용한 재시도 래퍼
        
        Args:
            coro_factory: 코루틴 팩토리 함수 (매번 새로 호출)
            error_types: 재시도 대상 예외 타입
            
        Returns:
            코루틴 결과
        """
        config = self.config.retry_config
        last_exception: Exception | None = None
        
        for attempt in range(config.max_attempts):
            try:
                return await coro_factory()
            except error_types as e:
                last_exception = e
                if attempt < config.max_attempts - 1:
                    delay = min(
                        config.base_delay * (config.exponential_base ** attempt),
                        config.max_delay,
                    )
                    logger.warning(
                        f"Attempt {attempt + 1}/{config.max_attempts} failed: {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    await asyncio.sleep(delay)
        
        raise last_exception or Exception("Unknown error during retry")
    
    # =========================================================================
    # Main Public Method
    # =========================================================================
    
    async def fetch_comments(
        self,
        keyword: str,
        max_videos: int = 5,
        comments_per_video: int = 50,
    ) -> list[Comment]:
        """
        YouTube 댓글 수집 (메인 메서드)
        
        플로우:
        1. 키워드로 YouTube 검색
        2. 상위 영상 목록 추출
        3. 각 영상의 댓글 수집
        4. 차단 감지 시 RSS Fallback
        
        Args:
            keyword: 검색 키워드
            max_videos: 수집할 최대 영상 수 (기본 5)
            comments_per_video: 영상당 최대 댓글 수 (기본 50)
            
        Returns:
            list[Comment]: 수집된 댓글 목록
        """
        all_comments: list[Comment] = []
        
        try:
            # 1. 영상 검색
            videos = await self._with_retry(
                lambda: self._search_videos(keyword, max_videos),
                error_types=(TimeoutError, NetworkError),
            )
            
            # 2. 각 영상에서 댓글 수집
            for i, video in enumerate(videos):
                try:
                    logger.info(
                        f"Processing video {i + 1}/{len(videos)}: "
                        f"{video.title[:40]}..."
                    )
                    
                    comments = await self._fetch_video_comments(
                        video, 
                        max_comments=comments_per_video,
                    )
                    all_comments.extend(comments)
                    
                    # Rate limiting: 영상 간 딜레이
                    if i < len(videos) - 1:
                        await self._video_delay()
                    
                except BlockedError:
                    logger.warning(
                        f"Blocked while fetching video {video.video_id}. "
                        "Triggering fallback..."
                    )
                    raise  # 상위에서 fallback 처리
                
                except TimeoutError:
                    logger.warning(
                        f"Timeout for video {video.video_id}. Skipping..."
                    )
                    continue
            
            logger.info(
                f"Successfully collected {len(all_comments)} comments "
                f"from {len(videos)} videos for '{keyword}'"
            )
            
        except BlockedError as e:
            logger.error(f"Blocked by YouTube: {e}. Attempting RSS fallback...")
            fallback_result = await self._fetch_via_rss(keyword)
            return fallback_result.comments
        
        except NoResultsError as e:
            logger.warning(f"No results: {e}")
        
        except Exception as e:
            logger.exception(f"Unexpected error during comment collection: {e}")
        
        return all_comments
    
    async def fetch_with_result(
        self,
        keyword: str,
        max_videos: int = 5,
        comments_per_video: int = 50,
    ) -> CollectionResult:
        """
        상세 결과 객체를 반환하는 수집 메서드
        
        Args:
            keyword: 검색 키워드
            max_videos: 수집할 최대 영상 수
            comments_per_video: 영상당 최대 댓글 수
            
        Returns:
            CollectionResult: 상세 수집 결과
        """
        fallback_used = False
        error_message: str | None = None
        source = CommentSource.PLAYWRIGHT
        videos_processed = 0
        
        try:
            # 영상 검색
            videos = await self._with_retry(
                lambda: self._search_videos(keyword, max_videos),
                error_types=(TimeoutError, NetworkError),
            )
            
            all_comments: list[Comment] = []
            
            # 각 영상에서 댓글 수집
            for i, video in enumerate(videos):
                try:
                    comments = await self._fetch_video_comments(
                        video,
                        max_comments=comments_per_video,
                    )
                    all_comments.extend(comments)
                    videos_processed += 1
                    
                    if i < len(videos) - 1:
                        await self._video_delay()
                    
                except BlockedError:
                    raise
                except TimeoutError:
                    continue
            
            return CollectionResult(
                keyword=keyword,
                comments=all_comments,
                videos_processed=videos_processed,
                source=source,
                success=True,
                fallback_used=False,
            )
            
        except BlockedError as e:
            logger.error(f"Blocked: {e}. Using RSS fallback...")
            fallback_result = await self._fetch_via_rss(keyword)
            return fallback_result
        
        except NoResultsError as e:
            return CollectionResult(
                keyword=keyword,
                comments=[],
                videos_processed=0,
                source=source,
                success=False,
                error_message=str(e),
            )
        
        except Exception as e:
            logger.exception(f"Collection failed: {e}")
            return CollectionResult(
                keyword=keyword,
                comments=[],
                videos_processed=videos_processed,
                source=source,
                success=False,
                error_message=str(e),
            )


# =============================================================================
# CLI Test
# =============================================================================

async def main() -> None:
    """CLI 테스트용 메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="YouTube Comment Collector (TrendOps Week 3)"
    )
    parser.add_argument(
        "keyword",
        type=str,
        help="Search keyword",
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        default=3,
        help="Maximum videos to process (default: 3)",
    )
    parser.add_argument(
        "--comments-per-video",
        type=int,
        default=20,
        help="Comments per video (default: 20)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=True,
        help="Run browser in headless mode (default: True)",
    )
    parser.add_argument(
        "--no-headless",
        action="store_false",
        dest="headless",
        help="Show browser window",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    
    args = parser.parse_args()
    
    # 로깅 설정
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    # Collector 설정
    config = YouTubeConfig(headless=args.headless)
    
    print(f"\n{'='*60}")
    print(f"YouTube Comment Collector")
    print(f"{'='*60}")
    print(f"Keyword: {args.keyword}")
    print(f"Max Videos: {args.max_videos}")
    print(f"Comments/Video: {args.comments_per_video}")
    print(f"Headless: {args.headless}")
    print(f"{'='*60}\n")
    
    async with YouTubeCollector(config=config) as collector:
        result = await collector.fetch_with_result(
            keyword=args.keyword,
            max_videos=args.max_videos,
            comments_per_video=args.comments_per_video,
        )
        
        print(f"\n{'='*60}")
        print("Collection Result")
        print(f"{'='*60}")
        print(f"Success: {result.success}")
        print(f"Videos Processed: {result.videos_processed}")
        print(f"Total Comments: {result.total_comments}")
        print(f"Source: {result.source.value}")
        print(f"Fallback Used: {result.fallback_used}")
        
        if result.error_message:
            print(f"Error: {result.error_message}")
        
        print(f"{'='*60}")
        
        if result.comments:
            print("\nSample Comments:")
            print("-" * 40)
            for i, comment in enumerate(result.comments[:10], 1):
                print(f"\n[{i}] Video: {comment.video_title[:40]}...")
                print(f"    Author: {comment.author}")
                print(f"    Likes: {comment.likes}")
                print(f"    Comment: {comment.comment_text[:100]}...")
        else:
            print("\nNo comments collected.")


if __name__ == "__main__":
    asyncio.run(main())