# tests/test_week5_day4_review_scheduler.py
"""
TrendOps Week 5 Day 4: Review Gate + Scheduler 테스트

테스트 범위:
1. Review Gate
   - 검토 요청 제출
   - 승인/거절/수정 처리
   - Auto-Approve 모드
   - 타임아웃 처리
   - Slack 메시지 빌더

2. Scheduler
   - Job 등록 및 실행
   - Job 상태 관리
   - 일시 정지/재개
   - 커스텀 Job 추가

실행:
    pytest tests/test_week5_day4_review_scheduler.py -v
    pytest tests/test_week5_day4_review_scheduler.py -v -k "review"
    pytest tests/test_week5_day4_review_scheduler.py -v -k "scheduler"
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# 테스트 대상 임포트
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "publisher"))

from trendops.publisher.review_gate import (
    HumanReviewGate,
    ReviewRequest,
    ReviewResult,
    ReviewAction,
    ReviewPriority,
    ReviewGateError,
    ReviewTimeoutError,
    ReviewNotFoundError,
    InMemoryReviewStorage,
    RedisReviewStorage,
    SlackMessageBuilder,
    create_review_gate,
)

from trendops.publisher.scheduler import (
    TrendOpsScheduler,
    JobConfig,
    JobResult,
    JobInfo,
    JobStatus,
    TriggerType,
    PipelineCallbacks,
    create_scheduler,
)


# =============================================================================
# Review Gate Fixtures
# =============================================================================

@pytest.fixture
def mock_webhook_url():
    """Mock Slack Webhook URL"""
    return "https://hooks.slack.com/services/TEST/WEBHOOK/URL"


@pytest.fixture
def review_gate(mock_webhook_url):
    """Review Gate 인스턴스"""
    return HumanReviewGate(
        slack_webhook_url=mock_webhook_url,
        auto_approve=False,
    )


@pytest.fixture
def sample_review_data():
    """샘플 검토 데이터"""
    return {
        "content_id": "content_123",
        "keyword": "AI 규제",
        "summary": "유럽연합이 AI 규제 법안을 통과시켰습니다.",
        "caption": "AI 규제 분석 카드뉴스 #AI #규제",
        "image_url": "https://example.com/card.png",
    }


# =============================================================================
# Review Action Tests
# =============================================================================

class TestReviewAction:
    """ReviewAction Enum 테스트"""
    
    def test_review_actions(self):
        """액션 값 확인"""
        assert ReviewAction.PENDING.value == "pending"
        assert ReviewAction.APPROVED.value == "approved"
        assert ReviewAction.REJECTED.value == "rejected"
        assert ReviewAction.MODIFIED.value == "modified"
        assert ReviewAction.TIMEOUT.value == "timeout"
    
    def test_review_priority(self):
        """우선순위 값 확인"""
        assert ReviewPriority.LOW.value == "low"
        assert ReviewPriority.NORMAL.value == "normal"
        assert ReviewPriority.HIGH.value == "high"
        assert ReviewPriority.URGENT.value == "urgent"


# =============================================================================
# Review Request Tests
# =============================================================================

class TestReviewRequest:
    """ReviewRequest 테스트"""
    
    def test_create_review_request(self):
        """검토 요청 생성"""
        review = ReviewRequest(
            review_id="review_123",
            content_id="content_456",
            keyword="테스트",
            summary="테스트 요약",
            caption="테스트 캡션",
        )
        
        assert review.review_id == "review_123"
        assert review.status == ReviewAction.PENDING
        assert review.priority == ReviewPriority.NORMAL
    
    def test_to_dict(self):
        """딕셔너리 변환"""
        review = ReviewRequest(
            review_id="review_123",
            content_id="content_456",
            keyword="테스트",
            summary="테스트 요약",
            caption="테스트 캡션",
            priority=ReviewPriority.HIGH,
        )
        
        data = review.to_dict()
        
        assert data["review_id"] == "review_123"
        assert data["status"] == "pending"
        assert data["priority"] == "high"
    
    def test_from_dict(self):
        """딕셔너리에서 생성"""
        data = {
            "review_id": "review_123",
            "content_id": "content_456",
            "keyword": "테스트",
            "summary": "테스트 요약",
            "caption": "테스트 캡션",
            "image_url": None,
            "image_path": None,
            "created_at": "2025-01-25T12:00:00",
            "updated_at": "2025-01-25T12:00:00",
            "status": "approved",
            "priority": "high",
            "reviewer_id": "user_123",
            "reviewer_note": "좋습니다",
            "modified_caption": None,
        }
        
        review = ReviewRequest.from_dict(data)
        
        assert review.review_id == "review_123"
        assert review.status == ReviewAction.APPROVED
        assert review.priority == ReviewPriority.HIGH


# =============================================================================
# Review Result Tests
# =============================================================================

class TestReviewResult:
    """ReviewResult 테스트"""
    
    def test_approved_result(self):
        """승인 결과"""
        result = ReviewResult(
            review_id="review_123",
            status=ReviewAction.APPROVED,
        )
        
        assert result.is_approved is True
        assert result.is_rejected is False
        assert result.is_modified is False
    
    def test_rejected_result(self):
        """거절 결과"""
        result = ReviewResult(
            review_id="review_123",
            status=ReviewAction.REJECTED,
        )
        
        assert result.is_rejected is True
        assert result.is_approved is False
    
    def test_modified_result(self):
        """수정 결과"""
        result = ReviewResult(
            review_id="review_123",
            status=ReviewAction.MODIFIED,
            modified_caption="수정된 캡션",
        )
        
        assert result.is_modified is True
        assert result.final_caption == "수정된 캡션"


# =============================================================================
# InMemory Storage Tests
# =============================================================================

class TestInMemoryReviewStorage:
    """인메모리 저장소 테스트"""
    
    @pytest.mark.asyncio
    async def test_save_and_get(self):
        """저장 및 조회"""
        storage = InMemoryReviewStorage()
        
        review = ReviewRequest(
            review_id="review_123",
            content_id="content_456",
            keyword="테스트",
            summary="요약",
            caption="캡션",
        )
        
        await storage.save(review)
        retrieved = await storage.get("review_123")
        
        assert retrieved is not None
        assert retrieved.review_id == "review_123"
    
    @pytest.mark.asyncio
    async def test_update(self):
        """업데이트"""
        storage = InMemoryReviewStorage()
        
        review = ReviewRequest(
            review_id="review_123",
            content_id="content_456",
            keyword="테스트",
            summary="요약",
            caption="캡션",
        )
        
        await storage.save(review)
        updated = await storage.update(
            "review_123",
            status=ReviewAction.APPROVED,
            reviewer_id="user_001",
        )
        
        assert updated is not None
        assert updated.status == ReviewAction.APPROVED
        assert updated.reviewer_id == "user_001"
    
    @pytest.mark.asyncio
    async def test_delete(self):
        """삭제"""
        storage = InMemoryReviewStorage()
        
        review = ReviewRequest(
            review_id="review_123",
            content_id="content_456",
            keyword="테스트",
            summary="요약",
            caption="캡션",
        )
        
        await storage.save(review)
        result = await storage.delete("review_123")
        
        assert result is True
        assert await storage.get("review_123") is None
    
    @pytest.mark.asyncio
    async def test_list_pending(self):
        """대기 중인 항목 목록"""
        storage = InMemoryReviewStorage()
        
        for i in range(3):
            review = ReviewRequest(
                review_id=f"review_{i}",
                content_id=f"content_{i}",
                keyword="테스트",
                summary="요약",
                caption="캡션",
            )
            await storage.save(review)
        
        # 하나 승인
        await storage.update("review_1", status=ReviewAction.APPROVED)
        
        pending = await storage.list_pending()
        
        assert len(pending) == 2


# =============================================================================
# Slack Message Builder Tests
# =============================================================================

class TestSlackMessageBuilder:
    """Slack 메시지 빌더 테스트"""
    
    def test_build_review_request(self):
        """검토 요청 메시지"""
        review = ReviewRequest(
            review_id="review_123",
            content_id="content_456",
            keyword="AI 규제",
            summary="AI 규제 법안이 통과되었습니다.",
            caption="분석 캡션",
            image_url="https://example.com/image.png",
            priority=ReviewPriority.HIGH,
        )
        
        message = SlackMessageBuilder.build_review_request(review)
        
        assert "blocks" in message
        assert len(message["blocks"]) >= 4
        
        # 헤더 확인
        assert message["blocks"][0]["type"] == "header"
        
        # 액션 버튼 확인
        actions_block = message["blocks"][-1]
        assert actions_block["type"] == "actions"
        assert len(actions_block["elements"]) == 3
    
    def test_build_action_result(self):
        """액션 결과 메시지"""
        message = SlackMessageBuilder.build_action_result(
            review_id="review_123",
            action=ReviewAction.APPROVED,
            reviewer="user_001",
            note="좋은 콘텐츠입니다",
        )
        
        assert "blocks" in message
        assert "승인됨" in message["blocks"][0]["text"]["text"]
    
    def test_build_auto_approve_alert(self):
        """자동 승인 알림"""
        message = SlackMessageBuilder.build_auto_approve_alert(100)
        
        assert "blocks" in message
        assert "100" in message["blocks"][1]["text"]["text"]


# =============================================================================
# Human Review Gate Tests
# =============================================================================

class TestHumanReviewGate:
    """Human Review Gate 테스트"""
    
    def test_initialization(self, mock_webhook_url):
        """초기화"""
        gate = HumanReviewGate(
            slack_webhook_url=mock_webhook_url,
            auto_approve=False,
        )
        
        assert gate.slack_webhook_url == mock_webhook_url
        assert gate.auto_approve is False
    
    @pytest.mark.asyncio
    async def test_context_manager(self, mock_webhook_url):
        """컨텍스트 매니저"""
        async with HumanReviewGate(
            slack_webhook_url=mock_webhook_url,
        ) as gate:
            assert gate is not None
    
    @pytest.mark.asyncio
    async def test_submit_for_review(self, review_gate, sample_review_data):
        """검토 요청 제출"""
        with patch.object(review_gate, '_send_slack_review_request', new_callable=AsyncMock) as mock_send:
            mock_send.return_value = True
            
            review_id = await review_gate.submit_for_review(**sample_review_data)
            
            assert review_id is not None
            assert len(review_id) == 12
            mock_send.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_submit_with_auto_approve(self, mock_webhook_url, sample_review_data):
        """자동 승인 모드"""
        gate = HumanReviewGate(
            slack_webhook_url=mock_webhook_url,
            auto_approve=True,
        )
        
        review_id = await gate.submit_for_review(**sample_review_data)
        
        review = await gate.get_review(review_id)
        assert review is not None
        assert review.status == ReviewAction.APPROVED
    
    @pytest.mark.asyncio
    async def test_handle_approve_action(self, review_gate, sample_review_data):
        """승인 액션 처리"""
        with patch.object(review_gate, '_send_slack_review_request', new_callable=AsyncMock):
            review_id = await review_gate.submit_for_review(**sample_review_data)
        
        result = await review_gate.handle_slack_action(
            action_id="approve_content",
            review_id=review_id,
            reviewer_id="user_001",
            note="승인합니다",
        )
        
        assert result.status == ReviewAction.APPROVED
        assert result.reviewer_id == "user_001"
    
    @pytest.mark.asyncio
    async def test_handle_reject_action(self, review_gate, sample_review_data):
        """거절 액션 처리"""
        with patch.object(review_gate, '_send_slack_review_request', new_callable=AsyncMock):
            review_id = await review_gate.submit_for_review(**sample_review_data)
        
        result = await review_gate.handle_slack_action(
            action_id="reject_content",
            review_id=review_id,
            reviewer_id="user_001",
            note="품질이 낮습니다",
        )
        
        assert result.status == ReviewAction.REJECTED
    
    @pytest.mark.asyncio
    async def test_handle_modify_action(self, review_gate, sample_review_data):
        """수정 액션 처리"""
        with patch.object(review_gate, '_send_slack_review_request', new_callable=AsyncMock):
            review_id = await review_gate.submit_for_review(**sample_review_data)
        
        result = await review_gate.handle_slack_action(
            action_id="modify_content",
            review_id=review_id,
            reviewer_id="user_001",
            modified_caption="수정된 캡션입니다",
        )
        
        assert result.status == ReviewAction.MODIFIED
        assert result.modified_caption == "수정된 캡션입니다"
    
    @pytest.mark.asyncio
    async def test_wait_for_approval_success(self, review_gate, sample_review_data):
        """승인 대기 - 성공"""
        with patch.object(review_gate, '_send_slack_review_request', new_callable=AsyncMock):
            review_id = await review_gate.submit_for_review(**sample_review_data)
        
        # 백그라운드에서 승인 처리
        async def approve_later():
            await asyncio.sleep(0.1)
            await review_gate.handle_slack_action(
                action_id="approve_content",
                review_id=review_id,
            )
        
        asyncio.create_task(approve_later())
        
        result = await review_gate.wait_for_approval(
            review_id,
            timeout=5,
            poll_interval=0.05,
        )
        
        assert result.status == ReviewAction.APPROVED
    
    @pytest.mark.asyncio
    async def test_wait_for_approval_timeout(self, review_gate, sample_review_data):
        """승인 대기 - 타임아웃"""
        with patch.object(review_gate, '_send_slack_review_request', new_callable=AsyncMock):
            review_id = await review_gate.submit_for_review(**sample_review_data)
        
        result = await review_gate.wait_for_approval(
            review_id,
            timeout=0.1,
            poll_interval=0.05,
        )
        
        assert result.status == ReviewAction.TIMEOUT
    
    @pytest.mark.asyncio
    async def test_consecutive_approvals_counter(self, review_gate, sample_review_data):
        """연속 승인 카운터"""
        with patch.object(review_gate, '_send_slack_review_request', new_callable=AsyncMock):
            with patch.object(review_gate, '_send_slack_message', new_callable=AsyncMock):
                for i in range(5):
                    data = sample_review_data.copy()
                    data["content_id"] = f"content_{i}"
                    
                    review_id = await review_gate.submit_for_review(**data)
                    
                    await review_gate.handle_slack_action(
                        action_id="approve_content",
                        review_id=review_id,
                    )
        
        assert review_gate.get_consecutive_approvals() == 5
    
    @pytest.mark.asyncio
    async def test_consecutive_approvals_reset_on_reject(self, review_gate, sample_review_data):
        """거절 시 연속 승인 카운터 리셋"""
        with patch.object(review_gate, '_send_slack_review_request', new_callable=AsyncMock):
            # 3번 승인
            for i in range(3):
                data = sample_review_data.copy()
                data["content_id"] = f"content_{i}"
                
                review_id = await review_gate.submit_for_review(**data)
                await review_gate.handle_slack_action(
                    action_id="approve_content",
                    review_id=review_id,
                )
            
            assert review_gate.get_consecutive_approvals() == 3
            
            # 1번 거절
            data = sample_review_data.copy()
            data["content_id"] = "content_rejected"
            review_id = await review_gate.submit_for_review(**data)
            await review_gate.handle_slack_action(
                action_id="reject_content",
                review_id=review_id,
            )
        
        assert review_gate.get_consecutive_approvals() == 0
    
    @pytest.mark.asyncio
    async def test_get_pending_reviews(self, review_gate, sample_review_data):
        """대기 중인 검토 목록"""
        with patch.object(review_gate, '_send_slack_review_request', new_callable=AsyncMock):
            for i in range(3):
                data = sample_review_data.copy()
                data["content_id"] = f"content_{i}"
                await review_gate.submit_for_review(**data)
        
        pending = await review_gate.get_pending_reviews()
        
        assert len(pending) == 3
    
    @pytest.mark.asyncio
    async def test_cancel_review(self, review_gate, sample_review_data):
        """검토 취소"""
        with patch.object(review_gate, '_send_slack_review_request', new_callable=AsyncMock):
            review_id = await review_gate.submit_for_review(**sample_review_data)
        
        result = await review_gate.cancel_review(review_id)
        
        assert result is True
        assert await review_gate.get_review(review_id) is None


# =============================================================================
# Scheduler Fixtures
# =============================================================================

@pytest.fixture
def scheduler():
    """기본 스케줄러"""
    return TrendOpsScheduler()


@pytest.fixture
def mock_callbacks():
    """Mock 콜백"""
    callbacks = MagicMock(spec=PipelineCallbacks)
    callbacks.trigger_pipeline = AsyncMock(return_value={"status": "ok"})
    callbacks.analysis_pipeline = AsyncMock(return_value={"status": "ok"})
    callbacks.publish_pipeline = AsyncMock(return_value={"status": "ok"})
    callbacks.daily_report = AsyncMock(return_value={"status": "ok"})
    return callbacks


# =============================================================================
# Job Status Tests
# =============================================================================

class TestJobStatus:
    """JobStatus Enum 테스트"""
    
    def test_job_statuses(self):
        """상태 값 확인"""
        assert JobStatus.PENDING.value == "pending"
        assert JobStatus.RUNNING.value == "running"
        assert JobStatus.COMPLETED.value == "completed"
        assert JobStatus.FAILED.value == "failed"
        assert JobStatus.PAUSED.value == "paused"
    
    def test_trigger_types(self):
        """트리거 타입 확인"""
        assert TriggerType.CRON.value == "cron"
        assert TriggerType.INTERVAL.value == "interval"
        assert TriggerType.DATE.value == "date"


# =============================================================================
# Job Config Tests
# =============================================================================

class TestJobConfig:
    """JobConfig 테스트"""
    
    def test_create_job_config(self):
        """Job 설정 생성"""
        async def dummy_func():
            pass
        
        config = JobConfig(
            job_id="test_job",
            name="테스트 Job",
            trigger_type=TriggerType.CRON,
            trigger_args={"minute": 0},
            func=dummy_func,
        )
        
        assert config.job_id == "test_job"
        assert config.max_instances == 1
        assert config.enabled is True
    
    def test_to_dict(self):
        """딕셔너리 변환"""
        async def dummy_func():
            pass
        
        config = JobConfig(
            job_id="test_job",
            name="테스트 Job",
            trigger_type=TriggerType.INTERVAL,
            trigger_args={"seconds": 60},
            func=dummy_func,
        )
        
        data = config.to_dict()
        
        assert data["job_id"] == "test_job"
        assert data["trigger_type"] == "interval"


# =============================================================================
# Job Result Tests
# =============================================================================

class TestJobResult:
    """JobResult 테스트"""
    
    def test_success_result(self):
        """성공 결과"""
        result = JobResult(
            job_id="test_job",
            status=JobStatus.COMPLETED,
            started_at=datetime(2025, 1, 25, 12, 0, 0),
            completed_at=datetime(2025, 1, 25, 12, 0, 5),
            result={"items": 10},
        )
        
        assert result.status == JobStatus.COMPLETED
        assert result.duration_seconds == 5.0
    
    def test_failed_result(self):
        """실패 결과"""
        result = JobResult(
            job_id="test_job",
            status=JobStatus.FAILED,
            started_at=datetime(2025, 1, 25, 12, 0, 0),
            completed_at=datetime(2025, 1, 25, 12, 0, 1),
            error_message="Connection error",
        )
        
        assert result.status == JobStatus.FAILED
        assert result.error_message == "Connection error"


# =============================================================================
# TrendOps Scheduler Tests
# =============================================================================

class TestTrendOpsScheduler:
    """TrendOps Scheduler 테스트"""
    
    def test_initialization(self):
        """초기화"""
        scheduler = TrendOpsScheduler()
        
        assert scheduler.job_count == 4  # 기본 4개 Job
        assert scheduler.is_running is False
    
    def test_default_jobs(self, scheduler):
        """기본 Job 등록 확인"""
        jobs = scheduler.get_jobs_status()
        job_ids = [j["job_id"] for j in jobs]
        
        assert "trend_detection" in job_ids
        assert "data_analysis" in job_ids
        assert "content_publish" in job_ids
        assert "daily_report" in job_ids
    
    @pytest.mark.asyncio
    async def test_run_job_now(self, mock_callbacks):
        """Job 즉시 실행"""
        scheduler = TrendOpsScheduler(callbacks=mock_callbacks)
        
        result = await scheduler.run_job_now("trend_detection")
        
        assert result.status == JobStatus.COMPLETED
        mock_callbacks.trigger_pipeline.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_run_job_with_error(self, mock_callbacks):
        """Job 실행 실패"""
        mock_callbacks.analysis_pipeline.side_effect = Exception("Test error")
        scheduler = TrendOpsScheduler(callbacks=mock_callbacks)
        
        result = await scheduler.run_job_now("data_analysis")
        
        assert result.status == JobStatus.FAILED
        assert "Test error" in result.error_message
    
    @pytest.mark.asyncio
    async def test_add_custom_job(self, scheduler):
        """커스텀 Job 추가"""
        async def custom_func():
            return {"custom": True}
        
        success = scheduler.add_custom_job(
            func=custom_func,
            trigger="cron",
            job_id="custom_job",
            name="커스텀 Job",
            hour=12,
            minute=30,
        )
        
        assert success is True
        assert scheduler.job_count == 5
    
    @pytest.mark.asyncio
    async def test_run_custom_job(self, scheduler):
        """커스텀 Job 실행"""
        async def custom_func():
            return {"value": 42}
        
        scheduler.add_custom_job(
            func=custom_func,
            trigger="interval",
            job_id="custom_job",
            seconds=60,
        )
        
        result = await scheduler.run_job_now("custom_job")
        
        assert result.status == JobStatus.COMPLETED
        assert result.result == {"value": 42}
    
    def test_remove_job(self, scheduler):
        """Job 제거"""
        result = scheduler.remove_job("daily_report")
        
        assert result is True
        assert scheduler.job_count == 3
    
    def test_get_job_history(self, scheduler):
        """Job 히스토리 조회"""
        history = scheduler.get_job_history("trend_detection")
        
        assert isinstance(history, list)
    
    @pytest.mark.asyncio
    async def test_job_history_recording(self, mock_callbacks):
        """Job 실행 히스토리 기록"""
        scheduler = TrendOpsScheduler(callbacks=mock_callbacks)
        
        # 3번 실행
        for _ in range(3):
            await scheduler.run_job_now("trend_detection")
        
        history = scheduler.get_job_history("trend_detection", limit=10)
        
        assert len(history) == 3
        assert all(h["status"] == "completed" for h in history)
    
    @pytest.mark.asyncio
    async def test_start_and_shutdown_with_apscheduler(self, scheduler):
        """APScheduler와 함께 시작/종료 (async context)"""
        # async context에서 시작
        await scheduler.start_async()
        assert scheduler.is_running is True
        
        scheduler.shutdown()
        assert scheduler.is_running is False
    
    def test_start_without_event_loop(self, scheduler):
        """Event loop 없이 시작 시도 - graceful 처리"""
        # APScheduler가 있어도 event loop 없으면 에러 없이 처리됨
        scheduler.start()
        assert scheduler.is_running is True
        
        scheduler.shutdown()
        assert scheduler.is_running is False


# =============================================================================
# Factory Function Tests
# =============================================================================

class TestFactoryFunctions:
    """팩토리 함수 테스트"""
    
    def test_create_review_gate(self):
        """Review Gate 생성"""
        gate = create_review_gate(
            slack_webhook_url="https://hooks.slack.com/test",
        )
        
        assert gate is not None
        assert isinstance(gate.storage, InMemoryReviewStorage)
    
    def test_create_scheduler(self):
        """Scheduler 생성"""
        scheduler = create_scheduler()
        
        assert scheduler is not None
        assert scheduler.job_count == 4


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """통합 테스트"""
    
    @pytest.mark.asyncio
    async def test_review_then_publish_flow(self, mock_webhook_url, mock_callbacks):
        """검토 후 발행 플로우"""
        # 1. Review Gate 설정
        gate = HumanReviewGate(
            slack_webhook_url=mock_webhook_url,
        )
        
        # 2. 검토 요청
        with patch.object(gate, '_send_slack_review_request', new_callable=AsyncMock):
            review_id = await gate.submit_for_review(
                content_id="content_001",
                keyword="테스트",
                summary="테스트 요약",
                caption="테스트 캡션",
            )
        
        # 3. 승인
        await gate.handle_slack_action(
            action_id="approve_content",
            review_id=review_id,
            reviewer_id="admin",
        )
        
        # 4. 검토 결과 확인
        review = await gate.get_review(review_id)
        assert review.status == ReviewAction.APPROVED
        
        # 5. Scheduler로 발행 실행
        scheduler = TrendOpsScheduler(callbacks=mock_callbacks)
        result = await scheduler.run_job_now("content_publish")
        
        assert result.status == JobStatus.COMPLETED


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])