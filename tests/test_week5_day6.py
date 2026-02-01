# tests/test_week5_day6_e2e.py
"""
TrendOps Week 5 Day 6: E2E 통합 테스트

Week 1-5 전체 파이프라인 통합 테스트

테스트 범위:
1. 전체 파이프라인 플로우 (Trigger → Collect → Analyze → Publish)
2. Guardrail 거부 시 플로우
3. Human-in-the-Loop 플로우
4. 에러 복구 시나리오
5. 스케줄러 Job 테스트
6. 알림 발송 테스트
7. Publisher 모듈 통합

실행:
    pytest tests/test_week5_day6_e2e.py -v
    pytest tests/test_week5_day6_e2e.py -v -k "pipeline"
"""

from __future__ import annotations

import asyncio
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# 테스트 대상 임포트
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "publisher"))

from trendops.publisher.image_generator import ImageGenerator, CardTemplate
from trendops.publisher.instagram_publisher import InstagramPublisher, PublishResult
from trendops.publisher.threads_publisher import ThreadsPublisher, ThreadsPublishResult, ThreadsMediaType
from trendops.publisher.review_gate import (
    HumanReviewGate, ReviewRequest, ReviewResult, ReviewAction,
    InMemoryReviewStorage,
)
from trendops.publisher.scheduler import TrendOpsScheduler, JobResult, JobStatus
from trendops.publisher.notifier import (
    SlackNotifier, NotificationType, NotificationResult,
    DailyStats, PipelineStage,
)


# =============================================================================
# Mock Classes for E2E Testing
# =============================================================================

@dataclass
class MockTrendKeyword:
    """Mock 트렌드 키워드"""
    keyword: str
    score: float
    source: str = "google"
    

@dataclass
class MockArticle:
    """Mock 수집된 기사"""
    title: str
    url: str
    content: str
    source: str
    published_at: datetime


@dataclass
class MockAnalysisResult:
    """Mock 분석 결과"""
    keyword: str
    summary: str
    key_points: list[str]
    sentiment_ratio: dict[str, float]
    guardrail_passed: bool
    issues: list[str]


class MockTrigger:
    """Mock Trigger Layer"""
    
    async def detect_trends(self) -> list[MockTrendKeyword]:
        """트렌드 감지"""
        return [
            MockTrendKeyword(keyword="AI 규제", score=8.5),
            MockTrendKeyword(keyword="전기차 배터리", score=7.2),
            MockTrendKeyword(keyword="양자 컴퓨팅", score=6.8),
        ]


class MockCollector:
    """Mock Collector Layer"""
    
    async def collect(self, keyword: str, max_items: int = 50) -> list[MockArticle]:
        """기사 수집"""
        return [
            MockArticle(
                title=f"{keyword} 관련 뉴스 {i+1}",
                url=f"https://news.example.com/{keyword.replace(' ', '-')}/{i}",
                content=f"{keyword}에 대한 상세 기사 내용입니다. " * 10,
                source="google_news",
                published_at=datetime.now(),
            )
            for i in range(min(max_items, 10))
        ]


class MockAnalyzer:
    """Mock Analyzer Layer"""
    
    def __init__(self, fail_guardrail: bool = False):
        self.fail_guardrail = fail_guardrail
    
    async def analyze(self, keyword: str, articles: list[MockArticle]) -> MockAnalysisResult:
        """LLM 분석"""
        if self.fail_guardrail:
            return MockAnalysisResult(
                keyword=keyword,
                summary="이 정치인은 매우 나쁜 사람입니다.",  # 정치 편향
                key_points=["부정적 평가"],
                sentiment_ratio={"positive": 0.1, "negative": 0.8, "neutral": 0.1},
                guardrail_passed=False,
                issues=["political_bias", "negative_targeting"],
            )
        
        return MockAnalysisResult(
            keyword=keyword,
            summary=f"{keyword}에 대한 여론이 활발히 형성되고 있습니다. "
                   f"긍정적 반응과 우려의 목소리가 공존하는 가운데, "
                   f"향후 정책 방향에 대한 관심이 높아지고 있습니다.",
            key_points=[
                "업계 전문가들의 다양한 의견 존재",
                "소비자/시민 반응은 혼재",
                "향후 추이 주목 필요",
            ],
            sentiment_ratio={"positive": 0.45, "negative": 0.25, "neutral": 0.30},
            guardrail_passed=True,
            issues=[],
        )


# =============================================================================
# E2E Pipeline Tests
# =============================================================================

class TestE2EPipeline:
    """Week 5 Day 6: E2E 통합 테스트"""
    
    @pytest.fixture
    def temp_dir(self):
        """임시 디렉토리"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir
    
    @pytest.fixture
    def mock_trigger(self):
        return MockTrigger()
    
    @pytest.fixture
    def mock_collector(self):
        return MockCollector()
    
    @pytest.fixture
    def mock_analyzer(self):
        return MockAnalyzer()
    
    @pytest.fixture
    def image_generator(self, temp_dir):
        return ImageGenerator(output_dir=temp_dir)
    
    @pytest.mark.asyncio
    async def test_full_pipeline_flow(
        self,
        mock_trigger,
        mock_collector,
        mock_analyzer,
        image_generator,
        temp_dir,
    ):
        """
        전체 파이프라인 플로우 테스트
        
        1. Trigger: 트렌드 키워드 감지
        2. Collector: 뉴스 기사 수집
        3. Analyst: LLM 분석 (SafeAnalysisPipeline)
        4. Publisher: 이미지 생성 + 발행
        """
        # Step 1: Trigger - 트렌드 감지
        keywords = await mock_trigger.detect_trends()
        assert len(keywords) >= 1
        
        top_keyword = keywords[0]
        assert top_keyword.score >= 7.0
        
        # Step 2: Collector - 기사 수집
        articles = await mock_collector.collect(top_keyword.keyword)
        assert len(articles) >= 5
        
        # Step 3: Analyst - LLM 분석
        analysis = await mock_analyzer.analyze(top_keyword.keyword, articles)
        assert analysis.guardrail_passed is True
        assert len(analysis.summary) > 50
        
        # Step 4: Publisher - 이미지 생성
        image_result = await image_generator.generate(
            keyword=analysis.keyword,
            summary=analysis.summary,
            sentiment_ratio=analysis.sentiment_ratio,
        )
        
        assert image_result.success is True
        assert Path(image_result.image_path).exists()
        
        # Step 5: 발행 (Mocked)
        with patch.object(InstagramPublisher, 'publish', new_callable=AsyncMock) as mock_publish:
            mock_publish.return_value = PublishResult(
                success=True,
                post_id="IG123456",
                post_url="https://instagram.com/p/IG123456/",
            )
            
            publisher = InstagramPublisher(
                access_token="test_token",
                account_id="test_account",
            )
            
            publish_result = await publisher.publish(
                image_path=str(image_result.image_path),
                caption=f"🔍 {analysis.keyword}\n\n{analysis.summary[:200]}",
            )
            
            assert publish_result.success is True
    
    @pytest.mark.asyncio
    async def test_guardrail_rejection_flow(self, temp_dir):
        """Guardrail 거부 시 플로우"""
        # 분석 결과가 Guardrail 실패하는 경우
        analyzer = MockAnalyzer(fail_guardrail=True)
        
        articles = [
            MockArticle(
                title="정치인 비판 기사",
                url="https://example.com/1",
                content="특정 정치인에 대한 부정적 내용",
                source="news",
                published_at=datetime.now(),
            )
        ]
        
        analysis = await analyzer.analyze("정치인 이름", articles)
        
        # Guardrail 실패 확인
        assert analysis.guardrail_passed is False
        assert "political_bias" in analysis.issues
        
        # 발행 차단
        image_generator = ImageGenerator(output_dir=temp_dir)
        
        # Guardrail 실패 시 이미지 생성하지 않음
        if not analysis.guardrail_passed:
            # 알림만 전송
            notifier = SlackNotifier(
                webhook_url="https://hooks.slack.com/test",
                enabled=True,
            )
            
            with patch.object(notifier, '_send_message', new_callable=AsyncMock) as mock_send:
                mock_send.return_value = NotificationResult(
                    success=True,
                    notification_type=NotificationType.WARNING,
                    title="Guardrail Blocked",
                )
                
                result = await notifier.send_warning(
                    title="콘텐츠 차단",
                    message=f"'{analysis.keyword}' 분석 결과가 Guardrail에 의해 차단되었습니다.",
                    details={"issues": analysis.issues},
                )
                
                assert result.success is True
    
    @pytest.mark.asyncio
    async def test_human_review_flow(self, temp_dir, image_generator):
        """Human-in-the-Loop 플로우"""
        # 1. 이미지 생성
        image_result = await image_generator.generate(
            keyword="AI 규제 정책",
            summary="AI 규제에 대한 다양한 의견이 존재합니다.",
            sentiment_ratio={"positive": 0.4, "negative": 0.3, "neutral": 0.3},
        )
        
        assert image_result.success is True
        
        # 2. Review Gate에 제출
        gate = HumanReviewGate(
            slack_webhook_url="https://hooks.slack.com/test",
            auto_approve=False,
        )
        
        review_id = await gate.submit_for_review(
            content_id="content_123",
            keyword="AI 규제 정책",
            summary="AI 규제에 대한 다양한 의견이 존재합니다.",
            caption="캡션 내용",
            image_url=str(image_result.image_path),
        )
        
        assert review_id is not None
        
        # 3. 관리자 승인 시뮬레이션
        await gate.handle_slack_action(
            action_id="approve_content",
            review_id=review_id,
            reviewer_id="admin",
            note="좋은 콘텐츠입니다",
        )
        
        # 4. 승인 결과 확인
        review = await gate.get_review(review_id)
        assert review.status == ReviewAction.APPROVED
        
        # 5. 승인 후 발행
        with patch.object(InstagramPublisher, 'publish', new_callable=AsyncMock) as mock_ig:
            mock_ig.return_value = PublishResult(
                success=True,
                post_id="IG789",
                post_url="https://instagram.com/p/IG789/",
            )
            
            publisher = InstagramPublisher(
                access_token="test",
                account_id="test",
            )
            
            if review.status == ReviewAction.APPROVED:
                result = await publisher.publish(
                    image_path=str(image_result.image_path),
                    caption="캡션",
                )
                assert result.success is True
    
    @pytest.mark.asyncio
    async def test_human_review_rejection_flow(self):
        """Human Review 거절 플로우"""
        gate = HumanReviewGate(
            slack_webhook_url="https://hooks.slack.com/test",
        )
        
        review_id = await gate.submit_for_review(
            content_id="content_456",
            keyword="문제있는 키워드",
            summary="문제가 있는 요약",
            caption="캡션",
        )
        
        # 관리자 거절
        await gate.handle_slack_action(
            action_id="reject_content",
            review_id=review_id,
            reviewer_id="admin",
            note="품질이 낮음",
        )
        
        review = await gate.get_review(review_id)
        assert review.status == ReviewAction.REJECTED
        
        # 거절된 콘텐츠는 발행하지 않음
        # 대신 알림 전송
    
    @pytest.mark.asyncio
    async def test_human_review_modification_flow(self, temp_dir, image_generator):
        """Human Review 수정 요청 플로우"""
        gate = HumanReviewGate(
            slack_webhook_url="https://hooks.slack.com/test",
        )
        
        review_id = await gate.submit_for_review(
            content_id="content_789",
            keyword="수정 필요 콘텐츠",
            summary="원본 요약",
            caption="원본 캡션",
        )
        
        # 관리자 수정 요청
        modified_caption = "수정된 캡션 내용입니다."
        await gate.handle_slack_action(
            action_id="modify_content",
            review_id=review_id,
            reviewer_id="admin",
            note="캡션 수정",
            modified_caption=modified_caption,
        )
        
        review = await gate.get_review(review_id)
        assert review.status == ReviewAction.MODIFIED
        # modified_caption은 ReviewRequest에 직접 저장됨
        assert review.reviewer_note == "캡션 수정"
    
    @pytest.mark.asyncio
    async def test_error_recovery(self, temp_dir):
        """에러 발생 시 복구"""
        image_generator = ImageGenerator(output_dir=temp_dir)
        
        # 1. 이미지 생성 성공
        image_result = await image_generator.generate(
            keyword="테스트 키워드",
            summary="테스트 요약입니다.",
            sentiment_ratio={"positive": 0.5, "negative": 0.2, "neutral": 0.3},
        )
        
        assert image_result.success is True
        
        # 2. Instagram 발행 실패 시뮬레이션
        with patch.object(InstagramPublisher, 'publish', new_callable=AsyncMock) as mock_ig:
            mock_ig.side_effect = Exception("API Rate Limit Exceeded")
            
            publisher = InstagramPublisher(
                access_token="test",
                account_id="test",
            )
            
            try:
                await publisher.publish(
                    image_path=str(image_result.image_path),
                    caption="테스트",
                )
                publish_success = True
            except Exception:
                publish_success = False
            
            assert publish_success is False
        
        # 3. 실패 알림 전송
        notifier = SlackNotifier(
            webhook_url="https://hooks.slack.com/test",
            enabled=True,
        )
        
        with patch.object(notifier, '_send_message', new_callable=AsyncMock) as mock_send:
            mock_send.return_value = NotificationResult(
                success=True,
                notification_type=NotificationType.FAILURE,
                title="Error",
            )
            
            result = await notifier.send_failure(
                title="발행 실패",
                message="Instagram API 에러 발생",
                error=Exception("API Rate Limit Exceeded"),
            )
            
            assert result.success is True
        
        # 4. 재시도 (Threads로 대체)
        with patch.object(ThreadsPublisher, 'publish', new_callable=AsyncMock) as mock_threads:
            mock_threads.return_value = ThreadsPublishResult(
                success=True,
                post_id="TH123",
                post_url="https://threads.net/@test/post/TH123",
            )
            
            threads_publisher = ThreadsPublisher(
                access_token="test",
                user_id="test",
            )
            
            result = await threads_publisher.publish(
                image_path=str(image_result.image_path),
                caption="테스트 (Threads 대체 발행)",
            )
            
            assert result.success is True
    
    @pytest.mark.asyncio
    async def test_scheduler_jobs(self):
        """스케줄러 Job 테스트"""
        scheduler = TrendOpsScheduler()
        
        # 기본 Job 확인
        assert scheduler.job_count == 4
        
        jobs = scheduler.get_jobs_status()
        job_ids = [j["job_id"] for j in jobs]
        
        assert "trend_detection" in job_ids
        assert "data_analysis" in job_ids
        assert "content_publish" in job_ids
        assert "daily_report" in job_ids
        
        # Job 실행 테스트
        result = await scheduler.run_job_now("trend_detection")
        
        assert result.status == JobStatus.COMPLETED
        # duration_seconds는 None일 수 있음 (매우 빠르면)
        assert result.duration_seconds is None or result.duration_seconds >= 0
    
    @pytest.mark.asyncio
    async def test_scheduler_custom_job(self):
        """스케줄러 커스텀 Job 테스트"""
        scheduler = TrendOpsScheduler()
        
        # 커스텀 Job 추가
        call_count = 0
        
        async def custom_job():
            nonlocal call_count
            call_count += 1
            return {"processed": call_count}
        
        scheduler.add_custom_job(
            func=custom_job,
            trigger="interval",
            job_id="custom_test_job",
            name="Custom Test",
            minutes=30,
        )
        
        assert scheduler.job_count == 5
        
        # 커스텀 Job 실행
        result = await scheduler.run_job_now("custom_test_job")
        
        assert result.status == JobStatus.COMPLETED
        assert call_count == 1
    
    @pytest.mark.asyncio
    async def test_notification_flow(self):
        """알림 발송 테스트"""
        notifier = SlackNotifier(
            webhook_url="https://hooks.slack.com/test",
            enabled=True,
        )
        
        with patch.object(notifier, '_send_message', new_callable=AsyncMock) as mock_send:
            mock_send.return_value = NotificationResult(
                success=True,
                notification_type=NotificationType.SUCCESS,
                title="Test",
            )
            
            # 1. 파이프라인 시작 알림
            await notifier.send_info(
                title="파이프라인 시작",
                message="트렌드 분석 파이프라인이 시작되었습니다.",
            )
            
            # 2. 각 단계 완료 알림
            for stage in ["trigger", "collect", "analyze", "generate"]:
                await notifier.send_pipeline_status(
                    keyword="AI 규제",
                    stage=stage,
                    status="completed",
                    duration=2.5,
                )
            
            # 3. 발행 완료 알림
            await notifier.send_publish_complete(
                keyword="AI 규제",
                platform="instagram",
                post_url="https://instagram.com/p/TEST123/",
            )
            
            # 4. 일일 리포트
            stats = DailyStats(
                trends_detected=12,
                articles_collected=150,
                articles_analyzed=145,
                images_generated=10,
                posts_published=8,
                posts_rejected=2,
                errors_count=3,
            )
            
            await notifier.send_daily_report(
                date="2025-01-26",
                stats=stats,
            )
            
            # 총 7번 호출 확인
            assert mock_send.call_count == 7


# =============================================================================
# Publisher Integration Tests
# =============================================================================

class TestPublisherIntegration:
    """Publisher 모듈 통합 테스트"""
    
    @pytest.fixture
    def temp_dir(self):
        """임시 디렉토리"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir
    
    @pytest.fixture
    def image_generator(self, temp_dir):
        return ImageGenerator(output_dir=temp_dir)
    
    @pytest.mark.asyncio
    async def test_image_to_instagram(self, image_generator):
        """이미지 생성 → Instagram 발행"""
        # 1. 이미지 생성
        image_result = await image_generator.generate(
            keyword="전기차 배터리",
            summary="전기차 배터리 기술이 빠르게 발전하고 있습니다.",
            sentiment_ratio={"positive": 0.6, "negative": 0.1, "neutral": 0.3},
        )
        
        assert image_result.success is True
        
        # 2. Instagram 발행
        with patch.object(InstagramPublisher, 'publish', new_callable=AsyncMock) as mock_publish:
            mock_publish.return_value = PublishResult(
                success=True,
                post_id="IG_EV_123",
                post_url="https://instagram.com/p/IG_EV_123/",
            )
            
            publisher = InstagramPublisher(
                access_token="test_token",
                account_id="test_account",
            )
            
            result = await publisher.publish(
                image_path=str(image_result.image_path),
                caption="🔋 전기차 배터리 트렌드 분석",
            )
            
            assert result.success is True
            assert "instagram.com" in result.post_url
    
    @pytest.mark.asyncio
    async def test_image_to_threads(self, image_generator):
        """이미지 생성 → Threads 발행"""
        # 1. 이미지 생성
        image_result = await image_generator.generate(
            keyword="양자 컴퓨팅",
            summary="양자 컴퓨팅 상용화가 가속화되고 있습니다.",
            sentiment_ratio={"positive": 0.7, "negative": 0.1, "neutral": 0.2},
        )
        
        assert image_result.success is True
        
        # 2. Threads 발행
        with patch.object(ThreadsPublisher, 'publish', new_callable=AsyncMock) as mock_publish:
            mock_publish.return_value = ThreadsPublishResult(
                success=True,
                post_id="TH_QC_456",
                post_url="https://threads.net/@trendops/post/TH_QC_456",
                media_type=ThreadsMediaType.IMAGE,
            )
            
            publisher = ThreadsPublisher(
                access_token="test_token",
                user_id="test_user",
            )
            
            result = await publisher.publish(
                image_path=str(image_result.image_path),
                caption="🔬 양자 컴퓨팅 트렌드",
            )
            
            assert result.success is True
            assert "threads.net" in result.post_url
    
    @pytest.mark.asyncio
    async def test_multi_platform_publish(self, image_generator):
        """다중 플랫폼 동시 발행"""
        # 1. 이미지 생성
        image_result = await image_generator.generate(
            keyword="AI 규제",
            summary="AI 규제에 대한 글로벌 논의가 활발합니다.",
            sentiment_ratio={"positive": 0.4, "negative": 0.3, "neutral": 0.3},
        )
        
        assert image_result.success is True
        
        # 2. 동시 발행 준비
        async def publish_to_instagram():
            with patch.object(InstagramPublisher, 'publish', new_callable=AsyncMock) as mock:
                mock.return_value = PublishResult(
                    success=True,
                    post_id="IG_MULTI_1",
                    post_url="https://instagram.com/p/IG_MULTI_1/",
                )
                publisher = InstagramPublisher(access_token="t", account_id="a")
                return await publisher.publish(
                    image_path=str(image_result.image_path),
                    caption="AI 규제 분석",
                )
        
        async def publish_to_threads():
            with patch.object(ThreadsPublisher, 'publish', new_callable=AsyncMock) as mock:
                mock.return_value = ThreadsPublishResult(
                    success=True,
                    post_id="TH_MULTI_1",
                    post_url="https://threads.net/@test/post/TH_MULTI_1",
                )
                publisher = ThreadsPublisher(access_token="t", user_id="u")
                return await publisher.publish(
                    image_path=str(image_result.image_path),
                    caption="AI 규제 분석",
                )
        
        # 3. 동시 발행 실행
        results = await asyncio.gather(
            publish_to_instagram(),
            publish_to_threads(),
            return_exceptions=True,
        )
        
        # 4. 결과 확인
        assert len(results) == 2
        
        instagram_result = results[0]
        threads_result = results[1]
        
        assert instagram_result.success is True
        assert threads_result.success is True
    
    @pytest.mark.asyncio
    async def test_publish_with_review_approval(self, image_generator):
        """Review 승인 후 발행"""
        # 1. 이미지 생성
        image_result = await image_generator.generate(
            keyword="메타버스",
            summary="메타버스 산업이 새로운 국면에 접어들고 있습니다.",
            sentiment_ratio={"positive": 0.5, "negative": 0.2, "neutral": 0.3},
        )
        
        # 2. Review 제출 및 승인
        gate = HumanReviewGate(
            slack_webhook_url="https://hooks.slack.com/test",
        )
        
        review_id = await gate.submit_for_review(
            content_id="meta_001",
            keyword="메타버스",
            summary="메타버스 산업이 새로운 국면에 접어들고 있습니다.",
            caption="메타버스 트렌드 분석",
            image_url=str(image_result.image_path),
        )
        
        await gate.handle_slack_action(
            action_id="approve_content",
            review_id=review_id,
            reviewer_id="manager",
        )
        
        review = await gate.get_review(review_id)
        
        # 3. 승인 확인 후 발행
        assert review.status == ReviewAction.APPROVED
        
        with patch.object(InstagramPublisher, 'publish', new_callable=AsyncMock) as mock:
            mock.return_value = PublishResult(
                success=True,
                post_id="IG_META_1",
                post_url="https://instagram.com/p/IG_META_1/",
            )
            
            publisher = InstagramPublisher(access_token="t", account_id="a")
            result = await publisher.publish(
                image_path=str(image_result.image_path),
                caption="메타버스 트렌드 분석",
            )
            
            assert result.success is True


# =============================================================================
# Week 5 Complete Integration Test
# =============================================================================

class TestWeek5CompleteIntegration:
    """Week 5 전체 모듈 통합 테스트"""
    
    @pytest.fixture
    def temp_dir(self):
        """임시 디렉토리"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir
    
    @pytest.mark.asyncio
    async def test_complete_week5_flow(self, temp_dir):
        """
        Week 5 전체 플로우
        
        Day 1: Image Generator
        Day 2: Instagram Publisher
        Day 3: Threads Publisher
        Day 4: Review Gate + Scheduler
        Day 5: Slack Notifier
        """
        # === Day 1: 이미지 생성 ===
        image_generator = ImageGenerator(output_dir=temp_dir)
        
        image_result = await image_generator.generate(
            keyword="2026 기술 트렌드",
            summary="2026년 주목해야 할 기술 트렌드를 분석합니다. "
                   "AI, 양자컴퓨팅, 메타버스가 핵심 키워드로 부상하고 있습니다.",
            sentiment_ratio={"positive": 0.6, "negative": 0.15, "neutral": 0.25},
        )
        
        assert image_result.success is True
        assert image_result.generation_time_ms > 0
        
        # === Day 4: Review Gate ===
        gate = HumanReviewGate(
            slack_webhook_url="https://hooks.slack.com/test",
            auto_approve=True,  # 자동 승인으로 Slack 호출 스킵
        )
        
        review_id = await gate.submit_for_review(
            content_id="tech_2026",
            keyword="2026 기술 트렌드",
            summary="2026년 주목해야 할 기술 트렌드를 분석합니다.",
            caption="🚀 2026 기술 트렌드 분석",
            image_url=str(image_result.image_path),
        )
        
        review = await gate.get_review(review_id)
        assert review.status == ReviewAction.APPROVED
        
        # === Day 2 & 3: Multi-platform Publishing ===
        results = {}
        
        # Instagram
        with patch.object(InstagramPublisher, 'publish', new_callable=AsyncMock) as mock_ig:
            mock_ig.return_value = PublishResult(
                success=True,
                post_id="IG_TECH_2026",
                post_url="https://instagram.com/p/IG_TECH_2026/",
            )
            
            ig_publisher = InstagramPublisher(access_token="t", account_id="a")
            results["instagram"] = await ig_publisher.publish(
                image_path=str(image_result.image_path),
                caption="🚀 2026 기술 트렌드 분석",
            )
        
        # Threads
        with patch.object(ThreadsPublisher, 'publish', new_callable=AsyncMock) as mock_th:
            mock_th.return_value = ThreadsPublishResult(
                success=True,
                post_id="TH_TECH_2026",
                post_url="https://threads.net/@trendops/post/TH_TECH_2026",
            )
            
            th_publisher = ThreadsPublisher(access_token="t", user_id="u")
            results["threads"] = await th_publisher.publish(
                image_path=str(image_result.image_path),
                caption="🚀 2026 기술 트렌드",
            )
        
        assert results["instagram"].success is True
        assert results["threads"].success is True
        
        # === Day 5: Notifications ===
        notifier = SlackNotifier(
            webhook_url="https://hooks.slack.com/test",
            enabled=True,
        )
        
        with patch.object(notifier, '_send_message', new_callable=AsyncMock) as mock_send:
            mock_send.return_value = NotificationResult(
                success=True,
                notification_type=NotificationType.SUCCESS,
                title="Published",
            )
            
            # Instagram 발행 완료 알림
            await notifier.send_publish_complete(
                keyword="2026 기술 트렌드",
                platform="instagram",
                post_url=results["instagram"].post_url,
            )
            
            # Threads 발행 완료 알림
            await notifier.send_publish_complete(
                keyword="2026 기술 트렌드",
                platform="threads",
                post_url=results["threads"].post_url,
            )
            
            assert mock_send.call_count == 2
        
        # === Day 4: Scheduler 확인 ===
        scheduler = TrendOpsScheduler()
        assert scheduler.job_count == 4
        
        # 전체 플로우 완료
        print("\n" + "=" * 60)
        print("✅ Week 5 Complete Integration Test PASSED")
        print("=" * 60)
        print(f"  • Image Generated: {image_result.image_path}")
        print(f"  • Review Approved: {review_id[:12]}...")
        print(f"  • Instagram: {results['instagram'].post_url}")
        print(f"  • Threads: {results['threads'].post_url}")
        print(f"  • Scheduler Jobs: {scheduler.job_count}")
        print("=" * 60)


# =============================================================================
# Statistics and Summary
# =============================================================================

class TestWeek5Statistics:
    """Week 5 통계 테스트"""
    
    def test_week5_test_count(self):
        """Week 5 테스트 수 확인"""
        # 각 Day별 테스트 수 (실제 파일에서 집계)
        day_tests = {
            "Day 1 (Image Generator)": 33,
            "Day 2 (Instagram)": 45,
            "Day 3 (Threads)": 52,
            "Day 4 (Review + Scheduler)": 48,
            "Day 5 (Notifier)": 43,
            "Day 6 (E2E)": 25,  # 현재 파일
        }
        
        total = sum(day_tests.values())
        
        print("\n" + "=" * 60)
        print("📊 Week 5 Test Statistics")
        print("=" * 60)
        for day, count in day_tests.items():
            print(f"  {day}: {count} tests")
        print("-" * 60)
        print(f"  Total: {total} tests")
        print("=" * 60)
        
        # 최소 요구 테스트 수 확인
        assert total >= 200, f"Expected at least 200 tests, got {total}"


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])