# tests/test_week5_day2_instagram.py
"""
TrendOps Week 5 Day 2: Instagram Publisher 테스트

테스트 범위:
1. 미디어 컨테이너 생성
2. 전체 발행 플로우 (Mock)
3. 일일 한도 체크
4. Rate Limit 에러 처리
5. 토큰 만료 에러 처리
6. 캡션 길이 검증

실행:
    pytest tests/test_week5_day2_instagram.py -v
    pytest tests/test_week5_day2_instagram.py -v -k "test_publish"
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

from trendops.publisher.instagram_publisher import (
    InstagramPublisher,
    InstagramAPIError,
    RateLimitExceededError,
    DailyLimitExceededError,
    ContainerTimeoutError,
    ContainerStatus,
    PublishResult,
    RateLimitState,
    MockImageHostingService,
    LocalServerHostingService,
    create_instagram_publisher,
    ERROR_CODES,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mock_credentials():
    """Mock 자격증명"""
    return {
        "access_token": "mock_access_token_12345",
        "account_id": "17841400000000000",
    }


@pytest.fixture
def publisher(mock_credentials):
    """기본 InstagramPublisher 인스턴스"""
    return InstagramPublisher(
        access_token=mock_credentials["access_token"],
        account_id=mock_credentials["account_id"],
    )


@pytest.fixture
def mock_http_client():
    """Mock HTTP 클라이언트"""
    mock_client = AsyncMock()
    mock_client.is_closed = False
    mock_client.aclose = AsyncMock()
    return mock_client


@pytest.fixture
def sample_image_url():
    """샘플 이미지 URL"""
    return "https://example.com/test_image.png"


@pytest.fixture
def sample_caption():
    """샘플 캡션"""
    return "트렌드 분석 카드뉴스입니다. 오늘의 핫이슈를 확인하세요!"


@pytest.fixture
def sample_hashtags():
    """샘플 해시태그"""
    return ["TrendOps", "트렌드", "AI분석"]


@pytest.fixture
def temp_image(tmp_path):
    """임시 이미지 파일"""
    from PIL import Image
    
    img = Image.new("RGB", (100, 100), color="red")
    path = tmp_path / "test_image.png"
    img.save(path)
    return path


# =============================================================================
# Unit Tests - Exceptions
# =============================================================================

class TestExceptions:
    """예외 클래스 테스트"""
    
    def test_instagram_api_error_basic(self):
        """기본 API 에러"""
        error = InstagramAPIError(code=190, message="Invalid access token")
        
        assert error.code == 190
        assert error.message == "Invalid access token"
        assert str(error) == "[190] Invalid access token"
    
    def test_instagram_api_error_from_response(self):
        """API 응답에서 에러 생성"""
        response = {
            "code": 4,
            "message": "Application request limit reached",
            "error_subcode": 2207050,
            "fbtrace_id": "ABC123",
        }
        
        error = InstagramAPIError.from_response(response)
        
        assert error.code == 4
        assert error.subcode == 2207050
        assert error.fb_trace_id == "ABC123"
    
    def test_is_rate_limit(self):
        """Rate Limit 에러 판별"""
        rate_limit_error = InstagramAPIError(code=4, message="Limit reached")
        assert rate_limit_error.is_rate_limit is True
        
        auth_error = InstagramAPIError(code=190, message="Invalid token")
        assert auth_error.is_rate_limit is False
    
    def test_is_auth_error(self):
        """인증 에러 판별"""
        auth_error = InstagramAPIError(code=190, message="Invalid token")
        assert auth_error.is_auth_error is True
        
        rate_limit_error = InstagramAPIError(code=4, message="Limit reached")
        assert rate_limit_error.is_auth_error is False
    
    def test_is_retryable(self):
        """재시도 가능 에러 판별"""
        retryable = InstagramAPIError(code=1, message="Unknown error")
        assert retryable.is_retryable is True
        
        not_retryable = InstagramAPIError(code=190, message="Invalid token")
        assert not_retryable.is_retryable is False
    
    def test_rate_limit_exceeded_error(self):
        """Rate Limit 초과 에러"""
        error = RateLimitExceededError(retry_after=3600)
        
        assert error.code == 4
        assert error.retry_after == 3600
    
    def test_daily_limit_exceeded_error(self):
        """일일 한도 초과 에러"""
        error = DailyLimitExceededError(current_count=25, limit=25)
        
        assert error.current_count == 25
        assert error.limit == 25
    
    def test_container_timeout_error(self):
        """컨테이너 타임아웃 에러"""
        error = ContainerTimeoutError(container_id="12345", timeout=60)
        
        assert error.container_id == "12345"
        assert error.timeout == 60


# =============================================================================
# Unit Tests - RateLimitState
# =============================================================================

class TestRateLimitState:
    """Rate Limit 상태 추적 테스트"""
    
    def test_initial_state(self):
        """초기 상태"""
        state = RateLimitState()
        
        assert state.daily_count == 0
        assert state.hourly_count == 0
        assert state.can_post() is True
        assert state.can_call_api() is True
    
    def test_increment_post(self):
        """발행 카운터 증가"""
        state = RateLimitState(daily_limit=25)
        
        for i in range(5):
            state.increment_post()
        
        assert state.daily_count == 5
        assert state.can_post() is True
    
    def test_daily_limit_reached(self):
        """일일 한도 도달"""
        state = RateLimitState(daily_limit=3)
        
        for _ in range(3):
            state.increment_post()
        
        assert state.daily_count == 3
        assert state.can_post() is False
    
    def test_hourly_limit_reached(self):
        """시간당 한도 도달"""
        state = RateLimitState(hourly_limit=5)
        
        for _ in range(5):
            state.increment_api_call()
        
        assert state.hourly_count == 5
        assert state.can_call_api() is False
    
    def test_daily_reset(self):
        """일일 리셋"""
        state = RateLimitState(daily_limit=25)
        state.daily_count = 20
        state.last_daily_reset = date.today() - timedelta(days=1)
        
        # check_and_reset 호출
        state.check_and_reset()
        
        assert state.daily_count == 0
        assert state.last_daily_reset == date.today()
    
    def test_hourly_reset(self):
        """시간별 리셋"""
        state = RateLimitState(hourly_limit=200)
        state.hourly_count = 150
        state.last_hourly_reset = datetime.now() - timedelta(hours=2)
        
        state.check_and_reset()
        
        assert state.hourly_count == 0


# =============================================================================
# Unit Tests - PublishResult
# =============================================================================

class TestPublishResult:
    """발행 결과 테스트"""
    
    def test_success_result(self):
        """성공 결과"""
        result = PublishResult(
            success=True,
            content_id="content_123",
            post_id="17841400000000001",
            post_url="https://instagram.com/p/ABC123",
            container_creation_time_ms=100.0,
            container_wait_time_ms=2000.0,
            publish_time_ms=150.0,
            total_time_ms=2250.0,
        )
        
        assert result.success is True
        assert result.platform == "instagram"
        assert result.post_id is not None
    
    def test_failure_result(self):
        """실패 결과"""
        result = PublishResult(
            success=False,
            content_id="content_123",
            error_message="Rate limit exceeded",
            error_code=4,
        )
        
        assert result.success is False
        assert result.post_id is None
        assert result.error_code == 4
    
    def test_metrics_property(self):
        """메트릭 속성"""
        result = PublishResult(
            success=True,
            content_id="test",
            container_creation_time_ms=100.0,
            container_wait_time_ms=2000.0,
            publish_time_ms=150.0,
            total_time_ms=2250.0,
        )
        
        metrics = result.metrics
        
        assert "container_creation_time_ms" in metrics
        assert metrics["total_time_ms"] == 2250.0


# =============================================================================
# Unit Tests - ImageHostingService
# =============================================================================

class TestMockImageHostingService:
    """Mock 이미지 호스팅 서비스 테스트"""
    
    @pytest.mark.asyncio
    async def test_upload(self, temp_image):
        """업로드"""
        service = MockImageHostingService()
        
        url = await service.upload(temp_image)
        
        assert url.startswith("https://example.com/images/")
        assert url.endswith(".png")
    
    @pytest.mark.asyncio
    async def test_delete(self, temp_image):
        """삭제"""
        service = MockImageHostingService()
        
        url = await service.upload(temp_image)
        result = await service.delete(url)
        
        assert result is True
    
    @pytest.mark.asyncio
    async def test_delete_not_found(self):
        """존재하지 않는 URL 삭제"""
        service = MockImageHostingService()
        
        result = await service.delete("https://example.com/nonexistent.png")
        
        assert result is False


class TestLocalServerHostingService:
    """로컬 서버 호스팅 서비스 테스트"""
    
    @pytest.mark.asyncio
    async def test_upload(self, temp_image, tmp_path):
        """업로드"""
        upload_dir = tmp_path / "uploads"
        service = LocalServerHostingService(
            server_url="http://localhost:8080",
            upload_dir=upload_dir,
        )
        
        url = await service.upload(temp_image)
        
        assert url.startswith("http://localhost:8080/")
        assert len(list(upload_dir.iterdir())) == 1
    
    @pytest.mark.asyncio
    async def test_delete(self, temp_image, tmp_path):
        """삭제"""
        upload_dir = tmp_path / "uploads"
        service = LocalServerHostingService(
            server_url="http://localhost:8080",
            upload_dir=upload_dir,
        )
        
        url = await service.upload(temp_image)
        result = await service.delete(url)
        
        assert result is True
        assert len(list(upload_dir.iterdir())) == 0


# =============================================================================
# Integration Tests - InstagramPublisher
# =============================================================================

class TestInstagramPublisher:
    """Instagram Publisher 통합 테스트"""
    
    def test_initialization(self, mock_credentials):
        """초기화"""
        publisher = InstagramPublisher(
            access_token=mock_credentials["access_token"],
            account_id=mock_credentials["account_id"],
        )
        
        assert publisher.access_token == mock_credentials["access_token"]
        assert publisher.account_id == mock_credentials["account_id"]
        assert publisher.rate_limit.daily_limit == 25
    
    def test_build_caption_with_hashtags(self, publisher, sample_caption, sample_hashtags):
        """캡션 + 해시태그 조합"""
        result = publisher._build_caption(sample_caption, sample_hashtags)
        
        assert sample_caption in result
        assert "#TrendOps" in result
        assert "#트렌드" in result
        assert "#AI분석" in result
    
    def test_validate_caption_truncation(self, publisher):
        """캡션 길이 초과 시 자르기"""
        long_caption = "A" * 2500
        
        result = publisher._validate_caption(long_caption)
        
        assert len(result) <= 2200
        assert result.endswith("...")
    
    def test_validate_caption_normal(self, publisher, sample_caption):
        """정상 캡션"""
        result = publisher._validate_caption(sample_caption)
        
        assert result == sample_caption
    
    @pytest.mark.asyncio
    async def test_context_manager(self, mock_credentials):
        """컨텍스트 매니저"""
        async with InstagramPublisher(
            access_token=mock_credentials["access_token"],
            account_id=mock_credentials["account_id"],
        ) as publisher:
            assert publisher is not None
    
    def test_get_rate_limit_status(self, publisher):
        """Rate Limit 상태 조회"""
        status = publisher.get_rate_limit_status()
        
        assert "daily_count" in status
        assert "daily_limit" in status
        assert "daily_remaining" in status
        assert "can_post" in status
        assert status["can_post"] is True


# =============================================================================
# Mock API Tests
# =============================================================================

class TestInstagramPublisherWithMock:
    """Mock API를 사용한 테스트"""
    
    @pytest.mark.asyncio
    async def test_create_container(self, publisher, sample_image_url, sample_caption):
        """미디어 컨테이너 생성"""
        with patch.object(publisher, '_request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = {"id": "container_123456"}
            
            container_id = await publisher._create_media_container(
                image_url=sample_image_url,
                caption=sample_caption,
            )
            
            assert container_id == "container_123456"
            mock_request.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_wait_for_container_success(self, publisher):
        """컨테이너 상태 대기 - 성공"""
        with patch.object(publisher, '_request', new_callable=AsyncMock) as mock_request:
            # 첫 번째는 IN_PROGRESS, 두 번째는 FINISHED
            mock_request.side_effect = [
                {"status_code": "IN_PROGRESS"},
                {"status_code": "FINISHED"},
            ]
            
            with patch('asyncio.sleep', new_callable=AsyncMock):
                result = await publisher._wait_for_container("container_123", timeout=10)
            
            assert result is True
            assert mock_request.call_count == 2
    
    @pytest.mark.asyncio
    async def test_wait_for_container_error(self, publisher):
        """컨테이너 상태 대기 - 에러"""
        with patch.object(publisher, '_request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = {
                "status_code": "ERROR",
                "status": "Media upload failed",
            }
            
            with pytest.raises(InstagramAPIError) as exc_info:
                await publisher._wait_for_container("container_123")
            
            assert exc_info.value.code == 36003
    
    @pytest.mark.asyncio
    async def test_publish_media(self, publisher):
        """미디어 발행"""
        with patch.object(publisher, '_request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = {"id": "media_789"}
            
            media_id = await publisher._publish_media("container_123")
            
            assert media_id == "media_789"
    
    @pytest.mark.asyncio
    async def test_get_permalink(self, publisher):
        """Permalink 조회"""
        with patch.object(publisher, '_request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = {
                "permalink": "https://www.instagram.com/p/ABC123/",
            }
            
            permalink = await publisher._get_permalink("media_789")
            
            assert permalink == "https://www.instagram.com/p/ABC123/"
    
    @pytest.mark.asyncio
    async def test_publish_flow_success(
        self,
        publisher,
        sample_image_url,
        sample_caption,
        sample_hashtags,
    ):
        """전체 발행 플로우 - 성공"""
        with patch.object(publisher, '_create_media_container', new_callable=AsyncMock) as mock_create:
            with patch.object(publisher, '_wait_for_container', new_callable=AsyncMock) as mock_wait:
                with patch.object(publisher, '_publish_media', new_callable=AsyncMock) as mock_publish:
                    with patch.object(publisher, '_get_permalink', new_callable=AsyncMock) as mock_permalink:
                        mock_create.return_value = "container_123"
                        mock_wait.return_value = True
                        mock_publish.return_value = "media_456"
                        mock_permalink.return_value = "https://instagram.com/p/ABC123/"
                        
                        result = await publisher.publish(
                            image_url=sample_image_url,
                            caption=sample_caption,
                            hashtags=sample_hashtags,
                        )
                        
                        assert result.success is True
                        assert result.post_id == "media_456"
                        assert result.post_url == "https://instagram.com/p/ABC123/"
                        assert result.total_time_ms > 0
    
    @pytest.mark.asyncio
    async def test_publish_flow_api_error(self, publisher, sample_image_url, sample_caption):
        """전체 발행 플로우 - API 에러"""
        with patch.object(publisher, '_create_media_container', new_callable=AsyncMock) as mock_create:
            mock_create.side_effect = InstagramAPIError(
                code=190,
                message="Invalid access token",
            )
            
            result = await publisher.publish(
                image_url=sample_image_url,
                caption=sample_caption,
            )
            
            assert result.success is False
            assert result.error_code == 190
            assert "Invalid access token" in result.error_message
    
    @pytest.mark.asyncio
    async def test_daily_limit_check(self, publisher, sample_image_url, sample_caption):
        """일일 한도 체크"""
        # 한도 채우기
        publisher.rate_limit.daily_count = 25
        
        result = await publisher.publish(
            image_url=sample_image_url,
            caption=sample_caption,
        )
        
        assert result.success is False
        assert "Daily limit exceeded" in result.error_message
    
    @pytest.mark.asyncio
    async def test_rate_limit_handling(self, publisher):
        """Rate Limit 에러 처리"""
        # 시간당 한도 채우기
        publisher.rate_limit.hourly_count = 200
        
        with pytest.raises(RateLimitExceededError):
            await publisher._request("GET", "test")
    
    @pytest.mark.asyncio
    async def test_verify_token_valid(self, publisher):
        """토큰 검증 - 유효"""
        with patch.object(publisher, 'get_account_info', new_callable=AsyncMock) as mock_info:
            mock_info.return_value = {"username": "test_user"}
            
            result = await publisher.verify_token()
            
            assert result is True
    
    @pytest.mark.asyncio
    async def test_verify_token_invalid(self, publisher):
        """토큰 검증 - 무효"""
        with patch.object(publisher, 'get_account_info', new_callable=AsyncMock) as mock_info:
            mock_info.side_effect = InstagramAPIError(code=190, message="Invalid token")
            
            result = await publisher.verify_token()
            
            assert result is False
    
    @pytest.mark.asyncio
    async def test_publish_local_image(self, publisher, temp_image, sample_caption):
        """로컬 이미지 발행"""
        with patch.object(publisher, 'publish', new_callable=AsyncMock) as mock_publish:
            mock_publish.return_value = PublishResult(
                success=True,
                content_id="test",
                post_id="media_123",
            )
            
            result = await publisher.publish_local(
                image_path=temp_image,
                caption=sample_caption,
            )
            
            assert result.success is True
            mock_publish.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_publish_local_image_not_found(self, publisher, sample_caption):
        """로컬 이미지 없음"""
        result = await publisher.publish_local(
            image_path=Path("/nonexistent/image.png"),
            caption=sample_caption,
        )
        
        assert result.success is False
        assert "not found" in result.error_message.lower()


# =============================================================================
# Factory Function Tests
# =============================================================================

class TestFactoryFunctions:
    """팩토리 함수 테스트"""
    
    def test_create_instagram_publisher_with_args(self):
        """인자로 생성"""
        publisher = create_instagram_publisher(
            access_token="test_token",
            account_id="test_account",
        )
        
        assert publisher.access_token == "test_token"
        assert publisher.account_id == "test_account"
    
    def test_create_instagram_publisher_from_env(self, monkeypatch):
        """환경 변수에서 생성"""
        monkeypatch.setenv("META_ACCESS_TOKEN", "env_token")
        monkeypatch.setenv("INSTAGRAM_BUSINESS_ACCOUNT_ID", "env_account")
        
        publisher = create_instagram_publisher()
        
        assert publisher.access_token == "env_token"
        assert publisher.account_id == "env_account"


# =============================================================================
# Error Codes Tests
# =============================================================================

class TestErrorCodes:
    """에러 코드 매핑 테스트"""
    
    def test_error_codes_defined(self):
        """에러 코드 정의 확인"""
        assert 190 in ERROR_CODES
        assert 4 in ERROR_CODES
        assert 17 in ERROR_CODES
        assert 36003 in ERROR_CODES
    
    def test_error_code_messages(self):
        """에러 코드 메시지"""
        assert ERROR_CODES[190] == "Invalid access token"
        assert ERROR_CODES[4] == "Application request limit reached"


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])