# src/trendops/publisher/threads_publisher.py
"""
TrendOps Threads Publishing API Publisher

Week 5 Day 3: Threads 연동

Features:
- Threads Publishing API v1.0 기반
- 텍스트 전용 포스트 지원
- 이미지 포함 포스트 지원
- 답글(Reply) 기능
- Rate Limit 관리
- Instagram Publisher와 자격증명 공유 가능

Flow:
1. 컨테이너 생성 (TEXT 또는 IMAGE)
2. 컨테이너 상태 확인 (FINISHED 대기)
3. 컨테이너 발행
4. Permalink 조회

Limitations:
- 이미지 1장만 가능 (캐러셀 미지원)
- 텍스트 최대 500자
- 동영상 최대 5분

Usage:
    from trendops.publisher.threads_publisher import ThreadsPublisher
    
    publisher = ThreadsPublisher(
        access_token="your_access_token",
        user_id="your_threads_user_id",
    )
    
    # 텍스트 전용
    result = await publisher.publish(text="Hello Threads!")
    
    # 이미지 포함
    result = await publisher.publish(
        text="Check this out!",
        image_url="https://example.com/image.png",
    )

Author: TrendOps Team
Created: Week 5 Day 3
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, date
from enum import Enum
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import httpx

# 로깅 설정
logger = logging.getLogger(__name__)


# =============================================================================
# Exceptions
# =============================================================================

class ThreadsAPIError(Exception):
    """Threads API 에러"""
    
    def __init__(
        self,
        code: int,
        message: str,
        subcode: int | None = None,
        error_type: str | None = None,
    ):
        self.code = code
        self.message = message
        self.subcode = subcode
        self.error_type = error_type
        super().__init__(f"[{code}] {message}")
    
    @classmethod
    def from_response(cls, error_data: dict) -> "ThreadsAPIError":
        """API 응답에서 에러 객체 생성"""
        return cls(
            code=error_data.get("code", 0),
            message=error_data.get("message", "Unknown error"),
            subcode=error_data.get("error_subcode"),
            error_type=error_data.get("type"),
        )
    
    @property
    def is_rate_limit(self) -> bool:
        """Rate Limit 에러 여부"""
        return self.code in (4, 17, 32, 613)
    
    @property
    def is_auth_error(self) -> bool:
        """인증 에러 여부"""
        return self.code in (190, 102, 104)
    
    @property
    def is_retryable(self) -> bool:
        """재시도 가능 여부"""
        return self.code in (1, 2, 4, 17, 32, 613)


class ThreadsRateLimitError(ThreadsAPIError):
    """Rate Limit 초과 에러"""
    
    def __init__(self, message: str = "Rate limit exceeded", retry_after: int | None = None):
        super().__init__(code=4, message=message)
        self.retry_after = retry_after


class ThreadsContainerError(ThreadsAPIError):
    """Container 관련 에러"""
    
    def __init__(self, container_id: str, status: str, message: str):
        super().__init__(code=36003, message=message)
        self.container_id = container_id
        self.status = status


# =============================================================================
# Enums & Data Classes
# =============================================================================

class ThreadsMediaType(str, Enum):
    """Threads 미디어 타입"""
    TEXT = "TEXT"
    IMAGE = "IMAGE"
    VIDEO = "VIDEO"
    CAROUSEL = "CAROUSEL"  # 미지원 (예약)


class ContainerStatus(str, Enum):
    """Container 상태"""
    IN_PROGRESS = "IN_PROGRESS"
    FINISHED = "FINISHED"
    ERROR = "ERROR"
    EXPIRED = "EXPIRED"
    PUBLISHED = "PUBLISHED"


@dataclass
class ThreadsPublishResult:
    """Threads 발행 결과"""
    success: bool
    content_id: str = ""
    platform: str = "threads"
    post_id: str | None = None
    post_url: str | None = None
    error_message: str | None = None
    error_code: int | None = None
    published_at: datetime = field(default_factory=datetime.now)
    
    # 메트릭
    container_creation_time_ms: float = 0.0
    container_wait_time_ms: float = 0.0
    publish_time_ms: float = 0.0
    total_time_ms: float = 0.0
    
    # 미디어 정보
    media_type: str = "TEXT"
    text_length: int = 0
    is_reply: bool = False
    
    @property
    def metrics(self) -> dict[str, float]:
        return {
            "container_creation_time_ms": self.container_creation_time_ms,
            "container_wait_time_ms": self.container_wait_time_ms,
            "publish_time_ms": self.publish_time_ms,
            "total_time_ms": self.total_time_ms,
        }


@dataclass 
class ThreadsRateLimitState:
    """Rate Limit 상태 추적"""
    daily_count: int = 0
    daily_limit: int = 25  # Threads 예상 한도 (공식 미공개)
    hourly_count: int = 0
    hourly_limit: int = 100  # API 호출 한도
    last_daily_reset: date = field(default_factory=date.today)
    last_hourly_reset: datetime = field(default_factory=datetime.now)
    
    def check_and_reset(self) -> None:
        """리셋 시간 확인 및 카운터 리셋"""
        from datetime import timedelta
        
        now = datetime.now()
        today = now.date()
        
        # 일일 리셋
        if today > self.last_daily_reset:
            self.daily_count = 0
            self.last_daily_reset = today
            logger.info("Threads daily rate limit counter reset")
        
        # 시간별 리셋
        hour_diff = (now - self.last_hourly_reset).total_seconds() / 3600
        if hour_diff >= 1:
            self.hourly_count = 0
            self.last_hourly_reset = now
    
    def can_post(self) -> bool:
        """발행 가능 여부"""
        self.check_and_reset()
        return self.daily_count < self.daily_limit
    
    def can_call_api(self) -> bool:
        """API 호출 가능 여부"""
        self.check_and_reset()
        return self.hourly_count < self.hourly_limit
    
    def increment_post(self) -> None:
        self.daily_count += 1
    
    def increment_api_call(self) -> None:
        self.hourly_count += 1


# =============================================================================
# Image Hosting Service Interface (Day 2에서 재사용)
# =============================================================================

class ImageHostingService:
    """이미지 호스팅 서비스 인터페이스"""
    
    async def upload(self, image_path: Path) -> str:
        """이미지 업로드 후 공개 URL 반환"""
        raise NotImplementedError
    
    async def delete(self, url: str) -> bool:
        """업로드된 이미지 삭제"""
        raise NotImplementedError


class MockImageHostingService(ImageHostingService):
    """Mock 이미지 호스팅 (테스트용)"""
    
    def __init__(self, base_url: str = "https://example.com/images"):
        self.base_url = base_url
        self._uploaded: dict[str, Path] = {}
    
    async def upload(self, image_path: Path) -> str:
        file_id = uuid4().hex[:12]
        url = f"{self.base_url}/{file_id}.png"
        self._uploaded[url] = image_path
        return url
    
    async def delete(self, url: str) -> bool:
        if url in self._uploaded:
            del self._uploaded[url]
            return True
        return False


# =============================================================================
# Threads Publisher
# =============================================================================

class ThreadsPublisher:
    """
    Threads Publishing API 발행기
    
    Week 5 Day 3: Threads 연동
    
    Note: Threads API는 Instagram Graph API의 확장
    - 같은 access_token 사용 가능
    - 별도의 Threads User ID 필요
    
    Supported Media Types:
    - TEXT: 텍스트 전용 포스트 (이미지 없음)
    - IMAGE: 이미지 포함 포스트
    
    Limitations:
    - 이미지 1장만 가능 (캐러셀 미지원)
    - 텍스트 최대 500자
    - 동영상 최대 5분 (VIDEO 타입)
    
    Flow:
    1. 컨테이너 생성 (POST /{user-id}/threads)
    2. 상태 확인 (GET /{container-id}?fields=status)
    3. 발행 (POST /{user-id}/threads_publish)
    
    Usage:
        publisher = ThreadsPublisher(
            access_token="...",
            user_id="...",
        )
        
        # 텍스트 전용
        result = await publisher.publish(text="Hello Threads!")
        
        # 이미지 포함
        result = await publisher.publish(
            text="Check this out!",
            image_url="https://example.com/image.png",
        )
        
        # 답글
        result = await publisher.publish(
            text="Great post!",
            reply_to="17841400000000001",
        )
    """
    
    BASE_URL = "https://graph.threads.net/v1.0"
    DEFAULT_TIMEOUT = 30.0
    CONTAINER_POLL_INTERVAL = 2.0
    CONTAINER_TIMEOUT = 60
    MAX_TEXT_LENGTH = 500
    
    def __init__(
        self,
        access_token: str,
        user_id: str,
        daily_limit: int = 25,
        hourly_limit: int = 100,
        image_hosting: ImageHostingService | None = None,
        timeout: float = DEFAULT_TIMEOUT,
    ):
        """
        Threads Publisher 초기화
        
        Args:
            access_token: Meta Graph API Access Token
            user_id: Threads User ID
            daily_limit: 일일 발행 한도
            hourly_limit: 시간당 API 호출 한도
            image_hosting: 이미지 호스팅 서비스
            timeout: HTTP 타임아웃
        """
        self.access_token = access_token
        self.user_id = user_id
        self.image_hosting = image_hosting or MockImageHostingService()
        self.timeout = timeout
        
        # Rate Limit
        self.rate_limit = ThreadsRateLimitState(
            daily_limit=daily_limit,
            hourly_limit=hourly_limit,
        )
        
        # HTTP 클라이언트
        self._client: httpx.AsyncClient | None = None
    
    async def _get_client(self) -> httpx.AsyncClient:
        """HTTP 클라이언트 (Lazy initialization)"""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout),
                headers={"User-Agent": "TrendOps/1.0"},
            )
        return self._client
    
    async def close(self) -> None:
        """HTTP 클라이언트 종료"""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
    
    async def __aenter__(self) -> "ThreadsPublisher":
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()
    
    def _build_url(self, endpoint: str) -> str:
        """API URL 생성"""
        return f"{self.BASE_URL}/{endpoint}"
    
    async def _request(
        self,
        method: str,
        endpoint: str,
        params: dict[str, Any] | None = None,
        data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        API 요청 수행
        """
        if not self.rate_limit.can_call_api():
            raise ThreadsRateLimitError("Hourly API call limit exceeded")
        
        client = await self._get_client()
        url = self._build_url(endpoint)
        
        # access_token 추가
        params = params or {}
        params["access_token"] = self.access_token
        
        self.rate_limit.increment_api_call()
        
        try:
            if method.upper() == "GET":
                response = await client.get(url, params=params)
            elif method.upper() == "POST":
                response = await client.post(url, params=params, data=data)
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            response_data = response.json()
            
            # 에러 체크
            if "error" in response_data:
                error = ThreadsAPIError.from_response(response_data["error"])
                logger.error(f"Threads API error: {error}")
                raise error
            
            return response_data
            
        except httpx.TimeoutException as e:
            logger.error(f"Request timeout: {endpoint}")
            raise ThreadsAPIError(code=0, message=f"Request timeout: {e}")
        except httpx.RequestError as e:
            logger.error(f"Request error: {e}")
            raise ThreadsAPIError(code=0, message=f"Request error: {e}")
    
    def _validate_text(self, text: str) -> str:
        """텍스트 유효성 검사 및 정리"""
        if len(text) > self.MAX_TEXT_LENGTH:
            logger.warning(
                f"Text truncated from {len(text)} to {self.MAX_TEXT_LENGTH} chars"
            )
            text = text[:self.MAX_TEXT_LENGTH - 3] + "..."
        return text
    
    async def publish(
        self,
        text: str,
        image_url: str | None = None,
        reply_to: str | None = None,
        content_id: str | None = None,
    ) -> ThreadsPublishResult:
        """
        Threads 포스트 발행
        
        Args:
            text: 포스트 텍스트 (max 500 chars)
            image_url: 이미지 URL (optional)
            reply_to: 답글 대상 media_id (optional)
            content_id: TrendOps 내부 콘텐츠 ID
        
        Returns:
            ThreadsPublishResult: 발행 결과
        """
        import time
        total_start = time.time()
        
        content_id = content_id or uuid4().hex[:8]
        text = self._validate_text(text)
        media_type = ThreadsMediaType.IMAGE if image_url else ThreadsMediaType.TEXT
        
        # Rate Limit 체크
        if not self.rate_limit.can_post():
            return ThreadsPublishResult(
                success=False,
                content_id=content_id,
                error_message=f"Daily limit exceeded: {self.rate_limit.daily_count}/{self.rate_limit.daily_limit}",
                error_code=99999,
            )
        
        try:
            # Step 1: 컨테이너 생성
            step_start = time.time()
            container_id = await self._create_container(
                text=text,
                image_url=image_url,
                media_type=media_type,
                reply_to=reply_to,
            )
            container_creation_time = (time.time() - step_start) * 1000
            logger.info(f"Threads container created: {container_id}")
            
            # Step 2: 컨테이너 상태 대기
            step_start = time.time()
            await self._wait_for_container(container_id)
            container_wait_time = (time.time() - step_start) * 1000
            logger.info(f"Threads container ready: {container_id}")
            
            # Step 3: 발행
            step_start = time.time()
            media_id = await self._publish_container(container_id)
            publish_time = (time.time() - step_start) * 1000
            logger.info(f"Threads media published: {media_id}")
            
            # Step 4: Permalink 조회
            permalink = await self._get_permalink(media_id)
            
            # Rate Limit 카운터 증가
            self.rate_limit.increment_post()
            
            total_time = (time.time() - total_start) * 1000
            
            return ThreadsPublishResult(
                success=True,
                content_id=content_id,
                post_id=media_id,
                post_url=permalink,
                container_creation_time_ms=container_creation_time,
                container_wait_time_ms=container_wait_time,
                publish_time_ms=publish_time,
                total_time_ms=total_time,
                media_type=media_type.value,
                text_length=len(text),
                is_reply=reply_to is not None,
            )
            
        except ThreadsAPIError as e:
            logger.error(f"Threads publish failed: {e}")
            return ThreadsPublishResult(
                success=False,
                content_id=content_id,
                error_message=e.message,
                error_code=e.code,
                total_time_ms=(time.time() - total_start) * 1000,
            )
        except Exception as e:
            logger.exception(f"Unexpected error: {e}")
            return ThreadsPublishResult(
                success=False,
                content_id=content_id,
                error_message=str(e),
                total_time_ms=(time.time() - total_start) * 1000,
            )
    
    async def publish_local(
        self,
        text: str,
        image_path: Path | str | None = None,
        reply_to: str | None = None,
        content_id: str | None = None,
        cleanup: bool = True,
    ) -> ThreadsPublishResult:
        """
        로컬 이미지로 발행
        
        Args:
            text: 포스트 텍스트
            image_path: 로컬 이미지 경로 (optional)
            reply_to: 답글 대상 media_id
            content_id: TrendOps 내부 콘텐츠 ID
            cleanup: 발행 후 호스팅 이미지 삭제 여부
        
        Returns:
            ThreadsPublishResult
        """
        image_url = None
        
        # 이미지 호스팅
        if image_path:
            image_path = Path(image_path)
            
            if not image_path.exists():
                return ThreadsPublishResult(
                    success=False,
                    content_id=content_id or "",
                    error_message=f"Image not found: {image_path}",
                )
            
            try:
                image_url = await self.image_hosting.upload(image_path)
                logger.info(f"Image uploaded for Threads: {image_url}")
            except Exception as e:
                return ThreadsPublishResult(
                    success=False,
                    content_id=content_id or "",
                    error_message=f"Image hosting failed: {e}",
                )
        
        # 발행
        result = await self.publish(
            text=text,
            image_url=image_url,
            reply_to=reply_to,
            content_id=content_id,
        )
        
        # 정리
        if cleanup and image_url:
            try:
                await self.image_hosting.delete(image_url)
            except Exception as e:
                logger.warning(f"Failed to delete hosted image: {e}")
        
        return result
    
    async def _create_container(
        self,
        text: str,
        image_url: str | None = None,
        media_type: ThreadsMediaType = ThreadsMediaType.TEXT,
        reply_to: str | None = None,
    ) -> str:
        """
        Threads 컨테이너 생성
        
        POST /{user-id}/threads
        """
        params: dict[str, Any] = {
            "media_type": media_type.value,
            "text": text,
        }
        
        if image_url and media_type == ThreadsMediaType.IMAGE:
            params["image_url"] = image_url
        
        if reply_to:
            params["reply_to_id"] = reply_to
        
        response = await self._request(
            method="POST",
            endpoint=f"{self.user_id}/threads",
            params=params,
        )
        
        container_id = response.get("id")
        if not container_id:
            raise ThreadsAPIError(
                code=36003,
                message="Failed to create Threads container: No ID returned",
            )
        
        return container_id
    
    async def _wait_for_container(
        self,
        container_id: str,
        timeout: int | None = None,
    ) -> bool:
        """
        컨테이너 상태 확인 (FINISHED 대기)
        
        GET /{container-id}?fields=status
        """
        timeout = timeout or self.CONTAINER_TIMEOUT
        start_time = asyncio.get_event_loop().time()
        
        while True:
            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed > timeout:
                raise ThreadsContainerError(
                    container_id=container_id,
                    status="TIMEOUT",
                    message=f"Container did not finish within {timeout}s",
                )
            
            response = await self._request(
                method="GET",
                endpoint=container_id,
                params={"fields": "status,error_message"},
            )
            
            status_str = response.get("status", "")
            
            try:
                status = ContainerStatus(status_str)
            except ValueError:
                status = None
            
            if status == ContainerStatus.FINISHED:
                return True
            elif status == ContainerStatus.ERROR:
                error_msg = response.get("error_message", "Unknown container error")
                raise ThreadsContainerError(
                    container_id=container_id,
                    status="ERROR",
                    message=error_msg,
                )
            elif status == ContainerStatus.EXPIRED:
                raise ThreadsContainerError(
                    container_id=container_id,
                    status="EXPIRED",
                    message="Container expired",
                )
            
            await asyncio.sleep(self.CONTAINER_POLL_INTERVAL)
    
    async def _publish_container(self, container_id: str) -> str:
        """
        컨테이너 발행
        
        POST /{user-id}/threads_publish
        """
        response = await self._request(
            method="POST",
            endpoint=f"{self.user_id}/threads_publish",
            params={"creation_id": container_id},
        )
        
        media_id = response.get("id")
        if not media_id:
            raise ThreadsAPIError(
                code=36003,
                message="Failed to publish Threads: No ID returned",
            )
        
        return media_id
    
    async def _get_permalink(self, media_id: str) -> str | None:
        """
        미디어 Permalink 조회
        
        GET /{media-id}?fields=permalink
        """
        try:
            response = await self._request(
                method="GET",
                endpoint=media_id,
                params={"fields": "permalink"},
            )
            return response.get("permalink")
        except ThreadsAPIError:
            logger.warning(f"Failed to get permalink for {media_id}")
            return None
    
    async def get_user_profile(self) -> dict[str, Any]:
        """
        사용자 프로필 조회
        
        GET /{user-id}?fields=id,username,threads_profile_picture_url,threads_biography
        """
        response = await self._request(
            method="GET",
            endpoint=self.user_id,
            params={
                "fields": "id,username,threads_profile_picture_url,threads_biography",
            },
        )
        return response
    
    async def verify_token(self) -> bool:
        """Access Token 유효성 검증"""
        try:
            await self.get_user_profile()
            return True
        except ThreadsAPIError as e:
            if e.is_auth_error:
                return False
            raise
    
    def get_rate_limit_status(self) -> dict[str, Any]:
        """Rate Limit 상태 반환"""
        self.rate_limit.check_and_reset()
        return {
            "daily_count": self.rate_limit.daily_count,
            "daily_limit": self.rate_limit.daily_limit,
            "daily_remaining": self.rate_limit.daily_limit - self.rate_limit.daily_count,
            "hourly_count": self.rate_limit.hourly_count,
            "hourly_limit": self.rate_limit.hourly_limit,
            "hourly_remaining": self.rate_limit.hourly_limit - self.rate_limit.hourly_count,
            "can_post": self.rate_limit.can_post(),
        }


# =============================================================================
# Factory Functions
# =============================================================================

def create_threads_publisher(
    access_token: str | None = None,
    user_id: str | None = None,
    **kwargs,
) -> ThreadsPublisher:
    """
    Threads Publisher 팩토리 함수
    
    환경 변수에서 자격증명 로드 가능
    """
    import os
    
    access_token = access_token or os.getenv("META_ACCESS_TOKEN", "")
    user_id = user_id or os.getenv("THREADS_USER_ID", "")
    
    if not access_token or not user_id:
        logger.warning("Threads credentials not configured")
    
    return ThreadsPublisher(
        access_token=access_token,
        user_id=user_id,
        **kwargs,
    )


# =============================================================================
# CLI Interface
# =============================================================================

async def main():
    """CLI 테스트용"""
    import argparse
    
    parser = argparse.ArgumentParser(description="TrendOps Threads Publisher")
    parser.add_argument("--token", required=True, help="Access Token")
    parser.add_argument("--user-id", required=True, help="Threads User ID")
    parser.add_argument("--text", required=True, help="Post text")
    parser.add_argument("--image-url", help="Image URL (optional)")
    parser.add_argument("--reply-to", help="Reply to media ID")
    parser.add_argument("--verify-only", action="store_true", help="Only verify token")
    
    args = parser.parse_args()
    
    async with ThreadsPublisher(
        access_token=args.token,
        user_id=args.user_id,
    ) as publisher:
        if args.verify_only:
            valid = await publisher.verify_token()
            print(f"Token valid: {valid}")
            return
        
        result = await publisher.publish(
            text=args.text,
            image_url=args.image_url,
            reply_to=args.reply_to,
        )
        
        print("=" * 60)
        print("🧵 Threads 발행 결과")
        print("=" * 60)
        print(f"성공: {result.success}")
        
        if result.success:
            print(f"Post ID: {result.post_id}")
            print(f"URL: {result.post_url}")
            print(f"미디어 타입: {result.media_type}")
            print(f"텍스트 길이: {result.text_length}자")
            print(f"총 소요시간: {result.total_time_ms:.1f}ms")
        else:
            print(f"에러: [{result.error_code}] {result.error_message}")


if __name__ == "__main__":
    asyncio.run(main())