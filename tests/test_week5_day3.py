# tests/test_week5_day3_threads.py
"""
TrendOps Week 5 Day 3: Threads Publisher 테스트

테스트 범위:
1. 텍스트 전용 포스트
2. 이미지 포함 포스트
3. 텍스트 길이 검증 (500자)
4. 답글 포스트
5. Rate Limit 처리
6. 에러 처리

실행:
    pytest tests/test_week5_day3_threads.py -v
    pytest tests/test_week5_day3_threads.py -v -k "test_publish"
"""

from __future__ import annotations

import asyncio
import tempfile
from datetime import date, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# 테스트 대상 임포트
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "publisher"))

from trendops.publisher.threads_publisher import (
    ThreadsPublisher,
    ThreadsPublishResult,
    ThreadsAPIError,
    ThreadsRateLimitError,
    ThreadsContainerError,
    ThreadsMediaType,
    ContainerStatus,
    ThreadsRateLimitState,
    MockImageHostingService,
    create_threads_publisher,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mock_credentials():
    """Mock 자격증명"""
    return {
        "access_token": "mock_threads_token_12345",
        "user_id": "threads_user_123456789",
    }


@pytest.fixture
def publisher(mock_credentials):
    """기본 ThreadsPublisher 인스턴스"""
    return ThreadsPublisher(
        access_token=mock_credentials["access_token"],
        user_id=mock_credentials["user_id"],
    )


@pytest.fixture
def sample_text():
    """샘플 텍스트"""
    return "AI 규제 법안이 유럽에서 통과되었습니다. 이것이 기술 업계에 미치는 영향을 분석해봤습니다. #AI #규제 #TrendOps"


@pytest.fixture
def sample_image_url():
    """샘플 이미지 URL"""
    return "https://example.com/threads_image.png"


@pytest.fixture
def temp_image(tmp_path):
    """임시 이미지 파일"""
    from PIL import Image
    
    img = Image.new("RGB", (100, 100), color="blue")
    path = tmp_path / "threads_test.png"
    img.save(path)
    return path


# =============================================================================
# Unit Tests - Exceptions
# =============================================================================

class TestExceptions:
    """예외 클래스 테스트"""
    
    def test_threads_api_error_basic(self):
        """기본 API 에러"""
        error = ThreadsAPIError(code=190, message="Invalid access token")
        
        assert error.code == 190
        assert error.message == "Invalid access token"
        assert str(error) == "[190] Invalid access token"
    
    def test_threads_api_error_from_response(self):
        """API 응답에서 에러 생성"""
        response = {
            "code": 4,
            "message": "Rate limit reached",
            "error_subcode": 123,
            "type": "OAuthException",
        }
        
        error = ThreadsAPIError.from_response(response)
        
        assert error.code == 4
        assert error.subcode == 123
        assert error.error_type == "OAuthException"
    
    def test_is_rate_limit(self):
        """Rate Limit 에러 판별"""
        rate_error = ThreadsAPIError(code=4, message="Limit")
        assert rate_error.is_rate_limit is True
        
        auth_error = ThreadsAPIError(code=190, message="Token")
        assert auth_error.is_rate_limit is False
    
    def test_is_auth_error(self):
        """인증 에러 판별"""
        auth_error = ThreadsAPIError(code=190, message="Token")
        assert auth_error.is_auth_error is True
        
        rate_error = ThreadsAPIError(code=4, message="Limit")
        assert rate_error.is_auth_error is False
    
    def test_is_retryable(self):
        """재시도 가능 에러 판별"""
        retryable = ThreadsAPIError(code=1, message="Unknown")
        assert retryable.is_retryable is True
        
        not_retryable = ThreadsAPIError(code=190, message="Token")
        assert not_retryable.is_retryable is False
    
    def test_threads_rate_limit_error(self):
        """Rate Limit 에러"""
        error = ThreadsRateLimitError(retry_after=1800)
        
        assert error.code == 4
        assert error.retry_after == 1800
    
    def test_threads_container_error(self):
        """Container 에러"""
        error = ThreadsContainerError(
            container_id="container_123",
            status="ERROR",
            message="Upload failed",
        )
        
        assert error.container_id == "container_123"
        assert error.status == "ERROR"


# =============================================================================
# Unit Tests - ThreadsRateLimitState
# =============================================================================

class TestThreadsRateLimitState:
    """Rate Limit 상태 테스트"""
    
    def test_initial_state(self):
        """초기 상태"""
        state = ThreadsRateLimitState()
        
        assert state.daily_count == 0
        assert state.hourly_count == 0
        assert state.can_post() is True
        assert state.can_call_api() is True
    
    def test_increment_post(self):
        """발행 카운터 증가"""
        state = ThreadsRateLimitState(daily_limit=25)
        
        for _ in range(5):
            state.increment_post()
        
        assert state.daily_count == 5
    
    def test_daily_limit_reached(self):
        """일일 한도 도달"""
        state = ThreadsRateLimitState(daily_limit=3)
        
        for _ in range(3):
            state.increment_post()
        
        assert state.can_post() is False
    
    def test_hourly_limit_reached(self):
        """시간당 한도 도달"""
        state = ThreadsRateLimitState(hourly_limit=5)
        
        for _ in range(5):
            state.increment_api_call()
        
        assert state.can_call_api() is False
    
    def test_daily_reset(self):
        """일일 리셋"""
        state = ThreadsRateLimitState()
        state.daily_count = 20
        state.last_daily_reset = date.today() - timedelta(days=1)
        
        state.check_and_reset()
        
        assert state.daily_count == 0
    
    def test_hourly_reset(self):
        """시간별 리셋"""
        state = ThreadsRateLimitState()
        state.hourly_count = 80
        state.last_hourly_reset = datetime.now() - timedelta(hours=2)
        
        state.check_and_reset()
        
        assert state.hourly_count == 0


# =============================================================================
# Unit Tests - ThreadsPublishResult
# =============================================================================

class TestThreadsPublishResult:
    """발행 결과 테스트"""
    
    def test_success_result_text(self):
        """텍스트 전용 성공 결과"""
        result = ThreadsPublishResult(
            success=True,
            content_id="content_123",
            post_id="threads_post_456",
            post_url="https://threads.net/@user/post/ABC",
            media_type="TEXT",
            text_length=150,
            is_reply=False,
        )
        
        assert result.success is True
        assert result.platform == "threads"
        assert result.media_type == "TEXT"
        assert result.is_reply is False
    
    def test_success_result_image(self):
        """이미지 포함 성공 결과"""
        result = ThreadsPublishResult(
            success=True,
            content_id="content_456",
            post_id="threads_post_789",
            media_type="IMAGE",
            text_length=100,
        )
        
        assert result.media_type == "IMAGE"
    
    def test_failure_result(self):
        """실패 결과"""
        result = ThreadsPublishResult(
            success=False,
            content_id="content_123",
            error_message="Rate limit exceeded",
            error_code=4,
        )
        
        assert result.success is False
        assert result.post_id is None
    
    def test_reply_result(self):
        """답글 결과"""
        result = ThreadsPublishResult(
            success=True,
            content_id="reply_123",
            post_id="threads_reply_789",
            is_reply=True,
        )
        
        assert result.is_reply is True
    
    def test_metrics_property(self):
        """메트릭 속성"""
        result = ThreadsPublishResult(
            success=True,
            content_id="test",
            container_creation_time_ms=50.0,
            container_wait_time_ms=1000.0,
            publish_time_ms=100.0,
            total_time_ms=1150.0,
        )
        
        metrics = result.metrics
        
        assert metrics["total_time_ms"] == 1150.0


# =============================================================================
# Unit Tests - ThreadsMediaType
# =============================================================================

class TestThreadsMediaType:
    """미디어 타입 테스트"""
    
    def test_text_type(self):
        """TEXT 타입"""
        assert ThreadsMediaType.TEXT.value == "TEXT"
    
    def test_image_type(self):
        """IMAGE 타입"""
        assert ThreadsMediaType.IMAGE.value == "IMAGE"
    
    def test_video_type(self):
        """VIDEO 타입"""
        assert ThreadsMediaType.VIDEO.value == "VIDEO"


# =============================================================================
# Integration Tests - ThreadsPublisher
# =============================================================================

class TestThreadsPublisher:
    """Threads Publisher 통합 테스트"""
    
    def test_initialization(self, mock_credentials):
        """초기화"""
        publisher = ThreadsPublisher(
            access_token=mock_credentials["access_token"],
            user_id=mock_credentials["user_id"],
        )
        
        assert publisher.access_token == mock_credentials["access_token"]
        assert publisher.user_id == mock_credentials["user_id"]
        assert publisher.rate_limit.daily_limit == 25
    
    def test_validate_text_normal(self, publisher, sample_text):
        """정상 텍스트"""
        result = publisher._validate_text(sample_text)
        
        assert result == sample_text
    
    def test_validate_text_truncation(self, publisher):
        """텍스트 길이 초과 시 자르기 (500자)"""
        long_text = "A" * 600
        
        result = publisher._validate_text(long_text)
        
        assert len(result) <= 500
        assert result.endswith("...")
    
    def test_validate_text_exactly_500(self, publisher):
        """정확히 500자"""
        text_500 = "A" * 500
        
        result = publisher._validate_text(text_500)
        
        assert len(result) == 500
        assert not result.endswith("...")
    
    @pytest.mark.asyncio
    async def test_context_manager(self, mock_credentials):
        """컨텍스트 매니저"""
        async with ThreadsPublisher(
            access_token=mock_credentials["access_token"],
            user_id=mock_credentials["user_id"],
        ) as publisher:
            assert publisher is not None
    
    def test_get_rate_limit_status(self, publisher):
        """Rate Limit 상태 조회"""
        status = publisher.get_rate_limit_status()
        
        assert "daily_count" in status
        assert "daily_limit" in status
        assert "can_post" in status
        assert status["can_post"] is True


# =============================================================================
# Mock API Tests
# =============================================================================

class TestThreadsPublisherWithMock:
    """Mock API 테스트"""
    
    @pytest.mark.asyncio
    async def test_text_only_post(self, publisher, sample_text):
        """텍스트 전용 포스트"""
        with patch.object(publisher, '_create_container', new_callable=AsyncMock) as m_create:
            with patch.object(publisher, '_wait_for_container', new_callable=AsyncMock) as m_wait:
                with patch.object(publisher, '_publish_container', new_callable=AsyncMock) as m_publish:
                    with patch.object(publisher, '_get_permalink', new_callable=AsyncMock) as m_permalink:
                        m_create.return_value = "container_text_123"
                        m_wait.return_value = True
                        m_publish.return_value = "media_text_456"
                        m_permalink.return_value = "https://threads.net/@user/post/ABC"
                        
                        result = await publisher.publish(text=sample_text)
                        
                        assert result.success is True
                        assert result.media_type == "TEXT"
                        assert result.post_id == "media_text_456"
                        
                        # TEXT 타입으로 컨테이너 생성 확인
                        m_create.assert_called_once()
                        call_kwargs = m_create.call_args
                        assert call_kwargs[1]["media_type"] == ThreadsMediaType.TEXT
    
    @pytest.mark.asyncio
    async def test_image_post(self, publisher, sample_text, sample_image_url):
        """이미지 포함 포스트"""
        with patch.object(publisher, '_create_container', new_callable=AsyncMock) as m_create:
            with patch.object(publisher, '_wait_for_container', new_callable=AsyncMock) as m_wait:
                with patch.object(publisher, '_publish_container', new_callable=AsyncMock) as m_publish:
                    with patch.object(publisher, '_get_permalink', new_callable=AsyncMock) as m_permalink:
                        m_create.return_value = "container_img_123"
                        m_wait.return_value = True
                        m_publish.return_value = "media_img_456"
                        m_permalink.return_value = "https://threads.net/@user/post/DEF"
                        
                        result = await publisher.publish(
                            text=sample_text,
                            image_url=sample_image_url,
                        )
                        
                        assert result.success is True
                        assert result.media_type == "IMAGE"
    
    @pytest.mark.asyncio
    async def test_reply_to_post(self, publisher, sample_text):
        """답글 포스트"""
        with patch.object(publisher, '_create_container', new_callable=AsyncMock) as m_create:
            with patch.object(publisher, '_wait_for_container', new_callable=AsyncMock) as m_wait:
                with patch.object(publisher, '_publish_container', new_callable=AsyncMock) as m_publish:
                    with patch.object(publisher, '_get_permalink', new_callable=AsyncMock) as m_permalink:
                        m_create.return_value = "container_reply_123"
                        m_wait.return_value = True
                        m_publish.return_value = "media_reply_456"
                        m_permalink.return_value = None
                        
                        result = await publisher.publish(
                            text="Great analysis!",
                            reply_to="original_post_789",
                        )
                        
                        assert result.success is True
                        assert result.is_reply is True
    
    @pytest.mark.asyncio
    async def test_create_container_text(self, publisher, sample_text):
        """컨테이너 생성 - TEXT"""
        with patch.object(publisher, '_request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = {"id": "container_123"}
            
            container_id = await publisher._create_container(
                text=sample_text,
                media_type=ThreadsMediaType.TEXT,
            )
            
            assert container_id == "container_123"
            
            # 호출 파라미터 확인
            call_args = mock_request.call_args
            params = call_args[1]["params"]
            assert params["media_type"] == "TEXT"
            assert params["text"] == sample_text
    
    @pytest.mark.asyncio
    async def test_create_container_image(self, publisher, sample_text, sample_image_url):
        """컨테이너 생성 - IMAGE"""
        with patch.object(publisher, '_request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = {"id": "container_img_123"}
            
            container_id = await publisher._create_container(
                text=sample_text,
                image_url=sample_image_url,
                media_type=ThreadsMediaType.IMAGE,
            )
            
            assert container_id == "container_img_123"
            
            call_args = mock_request.call_args
            params = call_args[1]["params"]
            assert params["media_type"] == "IMAGE"
            assert params["image_url"] == sample_image_url
    
    @pytest.mark.asyncio
    async def test_create_container_with_reply(self, publisher):
        """컨테이너 생성 - 답글"""
        with patch.object(publisher, '_request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = {"id": "container_reply_123"}
            
            await publisher._create_container(
                text="Reply text",
                media_type=ThreadsMediaType.TEXT,
                reply_to="parent_post_456",
            )
            
            call_args = mock_request.call_args
            params = call_args[1]["params"]
            assert params["reply_to_id"] == "parent_post_456"
    
    @pytest.mark.asyncio
    async def test_wait_for_container_success(self, publisher):
        """컨테이너 대기 - 성공"""
        with patch.object(publisher, '_request', new_callable=AsyncMock) as mock_request:
            mock_request.side_effect = [
                {"status": "IN_PROGRESS"},
                {"status": "FINISHED"},
            ]
            
            with patch('asyncio.sleep', new_callable=AsyncMock):
                result = await publisher._wait_for_container("container_123", timeout=10)
            
            assert result is True
    
    @pytest.mark.asyncio
    async def test_wait_for_container_error(self, publisher):
        """컨테이너 대기 - 에러"""
        with patch.object(publisher, '_request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = {
                "status": "ERROR",
                "error_message": "Upload failed",
            }
            
            with pytest.raises(ThreadsContainerError) as exc_info:
                await publisher._wait_for_container("container_123")
            
            assert exc_info.value.status == "ERROR"
    
    @pytest.mark.asyncio
    async def test_wait_for_container_expired(self, publisher):
        """컨테이너 대기 - 만료"""
        with patch.object(publisher, '_request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = {"status": "EXPIRED"}
            
            with pytest.raises(ThreadsContainerError) as exc_info:
                await publisher._wait_for_container("container_123")
            
            assert exc_info.value.status == "EXPIRED"
    
    @pytest.mark.asyncio
    async def test_publish_container(self, publisher):
        """컨테이너 발행"""
        with patch.object(publisher, '_request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = {"id": "media_123"}
            
            media_id = await publisher._publish_container("container_123")
            
            assert media_id == "media_123"
    
    @pytest.mark.asyncio
    async def test_get_permalink(self, publisher):
        """Permalink 조회"""
        with patch.object(publisher, '_request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = {
                "permalink": "https://threads.net/@user/post/XYZ",
            }
            
            permalink = await publisher._get_permalink("media_123")
            
            assert permalink == "https://threads.net/@user/post/XYZ"
    
    @pytest.mark.asyncio
    async def test_get_permalink_failure(self, publisher):
        """Permalink 조회 실패"""
        with patch.object(publisher, '_request', new_callable=AsyncMock) as mock_request:
            mock_request.side_effect = ThreadsAPIError(code=100, message="Invalid")
            
            permalink = await publisher._get_permalink("media_123")
            
            assert permalink is None
    
    @pytest.mark.asyncio
    async def test_daily_limit_check(self, publisher, sample_text):
        """일일 한도 체크"""
        publisher.rate_limit.daily_count = 25
        
        result = await publisher.publish(text=sample_text)
        
        assert result.success is False
        assert "Daily limit exceeded" in result.error_message
    
    @pytest.mark.asyncio
    async def test_api_error_handling(self, publisher, sample_text):
        """API 에러 처리"""
        with patch.object(publisher, '_create_container', new_callable=AsyncMock) as m_create:
            m_create.side_effect = ThreadsAPIError(code=190, message="Invalid token")
            
            result = await publisher.publish(text=sample_text)
            
            assert result.success is False
            assert result.error_code == 190
    
    @pytest.mark.asyncio
    async def test_verify_token_valid(self, publisher):
        """토큰 검증 - 유효"""
        with patch.object(publisher, 'get_user_profile', new_callable=AsyncMock) as mock_profile:
            mock_profile.return_value = {"id": "123", "username": "test_user"}
            
            result = await publisher.verify_token()
            
            assert result is True
    
    @pytest.mark.asyncio
    async def test_verify_token_invalid(self, publisher):
        """토큰 검증 - 무효"""
        with patch.object(publisher, 'get_user_profile', new_callable=AsyncMock) as mock_profile:
            mock_profile.side_effect = ThreadsAPIError(code=190, message="Invalid token")
            
            result = await publisher.verify_token()
            
            assert result is False
    
    @pytest.mark.asyncio
    async def test_publish_local_image(self, publisher, sample_text, temp_image):
        """로컬 이미지 발행"""
        with patch.object(publisher, 'publish', new_callable=AsyncMock) as mock_publish:
            mock_publish.return_value = ThreadsPublishResult(
                success=True,
                content_id="test",
                post_id="media_123",
                media_type="IMAGE",
            )
            
            result = await publisher.publish_local(
                text=sample_text,
                image_path=temp_image,
            )
            
            assert result.success is True
            mock_publish.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_publish_local_text_only(self, publisher, sample_text):
        """로컬 발행 - 텍스트만"""
        with patch.object(publisher, 'publish', new_callable=AsyncMock) as mock_publish:
            mock_publish.return_value = ThreadsPublishResult(
                success=True,
                content_id="test",
                post_id="media_123",
                media_type="TEXT",
            )
            
            result = await publisher.publish_local(text=sample_text)
            
            assert result.success is True
            assert mock_publish.call_args[1]["image_url"] is None
    
    @pytest.mark.asyncio
    async def test_publish_local_image_not_found(self, publisher, sample_text):
        """로컬 이미지 없음"""
        result = await publisher.publish_local(
            text=sample_text,
            image_path=Path("/nonexistent/image.png"),
        )
        
        assert result.success is False
        assert "not found" in result.error_message.lower()


# =============================================================================
# Factory Function Tests
# =============================================================================

class TestFactoryFunctions:
    """팩토리 함수 테스트"""
    
    def test_create_threads_publisher_with_args(self):
        """인자로 생성"""
        publisher = create_threads_publisher(
            access_token="test_token",
            user_id="test_user",
        )
        
        assert publisher.access_token == "test_token"
        assert publisher.user_id == "test_user"
    
    def test_create_threads_publisher_from_env(self, monkeypatch):
        """환경 변수에서 생성"""
        monkeypatch.setenv("META_ACCESS_TOKEN", "env_token")
        monkeypatch.setenv("THREADS_USER_ID", "env_user")
        
        publisher = create_threads_publisher()
        
        assert publisher.access_token == "env_token"
        assert publisher.user_id == "env_user"


# =============================================================================
# Text Length Edge Cases
# =============================================================================

class TestTextLengthValidation:
    """텍스트 길이 검증 상세 테스트"""
    
    def test_text_499_chars(self, publisher):
        """499자 (한도 이내)"""
        text = "A" * 499
        result = publisher._validate_text(text)
        
        assert len(result) == 499
        assert not result.endswith("...")
    
    def test_text_500_chars(self, publisher):
        """500자 (정확히 한도)"""
        text = "A" * 500
        result = publisher._validate_text(text)
        
        assert len(result) == 500
        assert not result.endswith("...")
    
    def test_text_501_chars(self, publisher):
        """501자 (한도 초과)"""
        text = "A" * 501
        result = publisher._validate_text(text)
        
        assert len(result) == 500
        assert result.endswith("...")
    
    def test_korean_text_length(self, publisher):
        """한글 텍스트 길이 검증"""
        korean_text = "한글테스트" * 110  # 550자
        result = publisher._validate_text(korean_text)
        
        assert len(result) <= 500
        assert result.endswith("...")


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])