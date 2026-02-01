# src/trendops/publisher/review_gate.py
"""
TrendOps Human-in-the-Loop 승인 게이트

Week 5 Day 4: Slack 기반 승인 워크플로우

Features:
- Slack Webhook 기반 검토 요청
- Interactive Button을 통한 승인/거절/수정
- Redis 기반 상태 저장 (옵션)
- Auto-Approval 모드 지원
- 타임아웃 처리

Flow:
1. 콘텐츠 생성 완료 → submit_for_review()
2. Slack으로 검토 요청 알림 (버튼 포함)
3. 관리자가 [승인]/[거절]/[수정] 선택
4. wait_for_approval()이 결과 반환
5. 결과에 따라 발행 또는 스킵

Auto-Approval:
- 연속 100건 무수정 승인 시 자동 승인 모드 전환 알림

Usage:
    from trendops.publisher.review_gate import HumanReviewGate, ReviewAction
    
    gate = HumanReviewGate(
        slack_webhook_url="https://hooks.slack.com/...",
    )
    
    # 검토 요청 제출
    review_id = await gate.submit_for_review(
        content_id="content_123",
        keyword="트럼프 관세",
        summary="요약 내용...",
        image_url="https://...",
        caption="캡션 내용",
    )
    
    # 승인 대기
    result = await gate.wait_for_approval(review_id, timeout=3600)
    
    if result.status == ReviewAction.APPROVED:
        await publisher.publish(content)

Author: TrendOps Team
Created: Week 5 Day 4
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Awaitable
from uuid import uuid4

import httpx

# 로깅 설정
logger = logging.getLogger(__name__)


# =============================================================================
# Enums
# =============================================================================

class ReviewAction(str, Enum):
    """검토 액션"""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    MODIFIED = "modified"
    TIMEOUT = "timeout"
    ERROR = "error"


class ReviewPriority(str, Enum):
    """검토 우선순위"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ReviewRequest:
    """검토 요청"""
    review_id: str
    content_id: str
    keyword: str
    summary: str
    caption: str
    image_url: str | None = None
    image_path: str | None = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    status: ReviewAction = ReviewAction.PENDING
    priority: ReviewPriority = ReviewPriority.NORMAL
    reviewer_id: str | None = None
    reviewer_note: str | None = None
    modified_caption: str | None = None
    
    def to_dict(self) -> dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "review_id": self.review_id,
            "content_id": self.content_id,
            "keyword": self.keyword,
            "summary": self.summary,
            "caption": self.caption,
            "image_url": self.image_url,
            "image_path": self.image_path,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "status": self.status.value,
            "priority": self.priority.value,
            "reviewer_id": self.reviewer_id,
            "reviewer_note": self.reviewer_note,
            "modified_caption": self.modified_caption,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ReviewRequest":
        """딕셔너리에서 생성"""
        return cls(
            review_id=data["review_id"],
            content_id=data["content_id"],
            keyword=data["keyword"],
            summary=data["summary"],
            caption=data["caption"],
            image_url=data.get("image_url"),
            image_path=data.get("image_path"),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            status=ReviewAction(data["status"]),
            priority=ReviewPriority(data.get("priority", "normal")),
            reviewer_id=data.get("reviewer_id"),
            reviewer_note=data.get("reviewer_note"),
            modified_caption=data.get("modified_caption"),
        )


@dataclass
class ReviewResult:
    """검토 결과"""
    review_id: str
    status: ReviewAction
    reviewer_id: str | None = None
    reviewer_note: str | None = None
    modified_caption: str | None = None
    elapsed_time_seconds: float = 0.0
    
    @property
    def is_approved(self) -> bool:
        return self.status == ReviewAction.APPROVED
    
    @property
    def is_rejected(self) -> bool:
        return self.status == ReviewAction.REJECTED
    
    @property
    def is_modified(self) -> bool:
        return self.status == ReviewAction.MODIFIED
    
    @property
    def final_caption(self) -> str | None:
        """최종 캡션 (수정된 경우 수정본 반환)"""
        return self.modified_caption


# =============================================================================
# Exceptions
# =============================================================================

class ReviewGateError(Exception):
    """Review Gate 에러"""
    pass


class ReviewTimeoutError(ReviewGateError):
    """검토 타임아웃"""
    def __init__(self, review_id: str, timeout: int):
        self.review_id = review_id
        self.timeout = timeout
        super().__init__(f"Review {review_id} timed out after {timeout}s")


class ReviewNotFoundError(ReviewGateError):
    """검토 요청 없음"""
    def __init__(self, review_id: str):
        self.review_id = review_id
        super().__init__(f"Review {review_id} not found")


# =============================================================================
# Storage Interface
# =============================================================================

class ReviewStorage:
    """검토 상태 저장소 인터페이스"""
    
    async def save(self, review: ReviewRequest) -> None:
        """검토 요청 저장"""
        raise NotImplementedError
    
    async def get(self, review_id: str) -> ReviewRequest | None:
        """검토 요청 조회"""
        raise NotImplementedError
    
    async def update(self, review_id: str, **updates) -> ReviewRequest | None:
        """검토 요청 업데이트"""
        raise NotImplementedError
    
    async def delete(self, review_id: str) -> bool:
        """검토 요청 삭제"""
        raise NotImplementedError
    
    async def list_pending(self) -> list[ReviewRequest]:
        """대기 중인 검토 목록"""
        raise NotImplementedError


class InMemoryReviewStorage(ReviewStorage):
    """인메모리 저장소 (테스트/개발용)"""
    
    def __init__(self):
        self._storage: dict[str, ReviewRequest] = {}
    
    async def save(self, review: ReviewRequest) -> None:
        self._storage[review.review_id] = review
    
    async def get(self, review_id: str) -> ReviewRequest | None:
        return self._storage.get(review_id)
    
    async def update(self, review_id: str, **updates) -> ReviewRequest | None:
        review = self._storage.get(review_id)
        if not review:
            return None
        
        for key, value in updates.items():
            if hasattr(review, key):
                setattr(review, key, value)
        
        review.updated_at = datetime.now()
        return review
    
    async def delete(self, review_id: str) -> bool:
        if review_id in self._storage:
            del self._storage[review_id]
            return True
        return False
    
    async def list_pending(self) -> list[ReviewRequest]:
        return [
            r for r in self._storage.values()
            if r.status == ReviewAction.PENDING
        ]


class RedisReviewStorage(ReviewStorage):
    """Redis 기반 저장소"""
    
    def __init__(self, redis_client, prefix: str = "trendops:review:"):
        self.redis = redis_client
        self.prefix = prefix
        self.ttl = 86400 * 7  # 7일
    
    def _key(self, review_id: str) -> str:
        return f"{self.prefix}{review_id}"
    
    async def save(self, review: ReviewRequest) -> None:
        key = self._key(review.review_id)
        data = json.dumps(review.to_dict())
        await self.redis.setex(key, self.ttl, data)
    
    async def get(self, review_id: str) -> ReviewRequest | None:
        key = self._key(review_id)
        data = await self.redis.get(key)
        if data:
            return ReviewRequest.from_dict(json.loads(data))
        return None
    
    async def update(self, review_id: str, **updates) -> ReviewRequest | None:
        review = await self.get(review_id)
        if not review:
            return None
        
        for key, value in updates.items():
            if hasattr(review, key):
                setattr(review, key, value)
        
        review.updated_at = datetime.now()
        await self.save(review)
        return review
    
    async def delete(self, review_id: str) -> bool:
        key = self._key(review_id)
        result = await self.redis.delete(key)
        return result > 0
    
    async def list_pending(self) -> list[ReviewRequest]:
        # 실제로는 별도 인덱스 필요
        pattern = f"{self.prefix}*"
        keys = await self.redis.keys(pattern)
        
        pending = []
        for key in keys:
            data = await self.redis.get(key)
            if data:
                review = ReviewRequest.from_dict(json.loads(data))
                if review.status == ReviewAction.PENDING:
                    pending.append(review)
        
        return pending


# =============================================================================
# Slack Message Builder
# =============================================================================

class SlackMessageBuilder:
    """Slack 메시지 빌더"""
    
    @staticmethod
    def build_review_request(review: ReviewRequest) -> dict[str, Any]:
        """검토 요청 메시지 생성"""
        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": "📋 콘텐츠 검토 요청",
                    "emoji": True,
                }
            },
            {
                "type": "section",
                "fields": [
                    {
                        "type": "mrkdwn",
                        "text": f"*Review ID:*\n`{review.review_id}`",
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*키워드:*\n{review.keyword}",
                    },
                ]
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*요약:*\n{review.summary[:500]}{'...' if len(review.summary) > 500 else ''}",
                }
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*캡션:*\n```{review.caption}```",
                }
            },
        ]
        
        # 이미지 URL이 있으면 추가
        if review.image_url:
            blocks.append({
                "type": "image",
                "image_url": review.image_url,
                "alt_text": f"카드뉴스: {review.keyword}",
            })
        
        # 우선순위 표시
        priority_emoji = {
            ReviewPriority.LOW: "🟢",
            ReviewPriority.NORMAL: "🟡",
            ReviewPriority.HIGH: "🟠",
            ReviewPriority.URGENT: "🔴",
        }
        
        blocks.append({
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": f"{priority_emoji.get(review.priority, '🟡')} 우선순위: {review.priority.value.upper()} | 생성: {review.created_at.strftime('%Y-%m-%d %H:%M:%S')}",
                }
            ]
        })
        
        # 액션 버튼
        blocks.append({
            "type": "actions",
            "block_id": f"review_actions_{review.review_id}",
            "elements": [
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "✅ 승인", "emoji": True},
                    "style": "primary",
                    "action_id": "approve_content",
                    "value": review.review_id,
                },
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "❌ 거절", "emoji": True},
                    "style": "danger",
                    "action_id": "reject_content",
                    "value": review.review_id,
                },
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "✏️ 수정", "emoji": True},
                    "action_id": "modify_content",
                    "value": review.review_id,
                },
            ]
        })
        
        return {"blocks": blocks}
    
    @staticmethod
    def build_action_result(
        review_id: str,
        action: ReviewAction,
        reviewer: str | None = None,
        note: str | None = None,
    ) -> dict[str, Any]:
        """액션 결과 메시지"""
        action_text = {
            ReviewAction.APPROVED: "✅ 승인됨",
            ReviewAction.REJECTED: "❌ 거절됨",
            ReviewAction.MODIFIED: "✏️ 수정됨",
        }
        
        text = f"*{action_text.get(action, action.value)}*\nReview ID: `{review_id}`"
        
        if reviewer:
            text += f"\n검토자: <@{reviewer}>"
        if note:
            text += f"\n메모: {note}"
        
        return {
            "blocks": [
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": text},
                }
            ]
        }
    
    @staticmethod
    def build_auto_approve_alert(consecutive_count: int) -> dict[str, Any]:
        """자동 승인 전환 알림"""
        return {
            "blocks": [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": "🤖 자동 승인 모드 전환 검토 필요",
                        "emoji": True,
                    }
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"연속 *{consecutive_count}*건이 무수정 승인되었습니다.\n자동 승인 모드로 전환을 검토해주세요.",
                    }
                },
                {
                    "type": "actions",
                    "elements": [
                        {
                            "type": "button",
                            "text": {"type": "plain_text", "text": "🤖 자동 승인 활성화"},
                            "style": "primary",
                            "action_id": "enable_auto_approve",
                        },
                        {
                            "type": "button",
                            "text": {"type": "plain_text", "text": "계속 수동 검토"},
                            "action_id": "keep_manual_review",
                        },
                    ]
                }
            ]
        }


# =============================================================================
# Human Review Gate
# =============================================================================

class HumanReviewGate:
    """
    Human-in-the-Loop 승인 게이트
    
    Week 5 Day 4: Slack 기반 승인 워크플로우
    
    Flow:
    1. 콘텐츠 생성 완료 → submit_for_review()
    2. Slack으로 검토 요청 알림
    3. 관리자가 [승인]/[거절]/[수정] 선택
    4. wait_for_approval()이 결과 반환
    
    Auto-Approval:
    - 연속 100건 무수정 승인 시 자동화 전환 검토 알림
    """
    
    AUTO_APPROVE_THRESHOLD = 100
    DEFAULT_TIMEOUT = 3600  # 1시간
    DEFAULT_POLL_INTERVAL = 5  # 5초
    
    def __init__(
        self,
        slack_webhook_url: str,
        storage: ReviewStorage | None = None,
        auto_approve: bool = False,
        timeout: float = 30.0,
    ):
        """
        Human Review Gate 초기화
        
        Args:
            slack_webhook_url: Slack Incoming Webhook URL
            storage: 검토 상태 저장소 (기본: InMemory)
            auto_approve: 자동 승인 모드
            timeout: HTTP 요청 타임아웃
        """
        self.slack_webhook_url = slack_webhook_url
        self.storage = storage or InMemoryReviewStorage()
        self.auto_approve = auto_approve
        self.timeout = timeout
        
        self._consecutive_approvals = 0
        self._client: httpx.AsyncClient | None = None
        
        # 콜백 함수
        self._on_approved: Callable[[ReviewResult], Awaitable[None]] | None = None
        self._on_rejected: Callable[[ReviewResult], Awaitable[None]] | None = None
        self._on_modified: Callable[[ReviewResult], Awaitable[None]] | None = None
    
    async def _get_client(self) -> httpx.AsyncClient:
        """HTTP 클라이언트"""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout),
            )
        return self._client
    
    async def close(self) -> None:
        """리소스 정리"""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
    
    async def __aenter__(self) -> "HumanReviewGate":
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()
    
    def on_approved(self, callback: Callable[[ReviewResult], Awaitable[None]]) -> None:
        """승인 콜백 등록"""
        self._on_approved = callback
    
    def on_rejected(self, callback: Callable[[ReviewResult], Awaitable[None]]) -> None:
        """거절 콜백 등록"""
        self._on_rejected = callback
    
    def on_modified(self, callback: Callable[[ReviewResult], Awaitable[None]]) -> None:
        """수정 콜백 등록"""
        self._on_modified = callback
    
    async def submit_for_review(
        self,
        content_id: str,
        keyword: str,
        summary: str,
        caption: str,
        image_url: str | None = None,
        image_path: str | None = None,
        priority: ReviewPriority = ReviewPriority.NORMAL,
    ) -> str:
        """
        검토 요청 제출
        
        Args:
            content_id: 콘텐츠 ID
            keyword: 트렌드 키워드
            summary: 분석 요약
            caption: 발행 캡션
            image_url: 이미지 URL (Slack 표시용)
            image_path: 로컬 이미지 경로
            priority: 우선순위
        
        Returns:
            review_id: 검토 요청 ID
        """
        # Auto-approve 모드
        if self.auto_approve:
            logger.info(f"Auto-approve mode: content {content_id} automatically approved")
            review_id = uuid4().hex[:12]
            
            review = ReviewRequest(
                review_id=review_id,
                content_id=content_id,
                keyword=keyword,
                summary=summary,
                caption=caption,
                image_url=image_url,
                image_path=image_path,
                priority=priority,
                status=ReviewAction.APPROVED,
            )
            
            await self.storage.save(review)
            return review_id
        
        # 검토 요청 생성
        review_id = uuid4().hex[:12]
        
        review = ReviewRequest(
            review_id=review_id,
            content_id=content_id,
            keyword=keyword,
            summary=summary,
            caption=caption,
            image_url=image_url,
            image_path=image_path,
            priority=priority,
        )
        
        # 저장
        await self.storage.save(review)
        
        # Slack 알림 전송
        await self._send_slack_review_request(review)
        
        logger.info(f"Review submitted: {review_id} for content {content_id}")
        
        return review_id
    
    async def wait_for_approval(
        self,
        review_id: str,
        timeout: int | None = None,
        poll_interval: int | None = None,
    ) -> ReviewResult:
        """
        승인 대기 (polling)
        
        Args:
            review_id: 검토 요청 ID
            timeout: 타임아웃 (초)
            poll_interval: 폴링 간격 (초)
        
        Returns:
            ReviewResult: 검토 결과
        """
        timeout = timeout or self.DEFAULT_TIMEOUT
        poll_interval = poll_interval or self.DEFAULT_POLL_INTERVAL
        
        start_time = asyncio.get_event_loop().time()
        
        while True:
            elapsed = asyncio.get_event_loop().time() - start_time
            
            if elapsed > timeout:
                # 타임아웃 처리
                await self.storage.update(
                    review_id,
                    status=ReviewAction.TIMEOUT,
                )
                
                return ReviewResult(
                    review_id=review_id,
                    status=ReviewAction.TIMEOUT,
                    elapsed_time_seconds=elapsed,
                )
            
            # 상태 확인
            review = await self.storage.get(review_id)
            
            if not review:
                raise ReviewNotFoundError(review_id)
            
            if review.status != ReviewAction.PENDING:
                # 검토 완료
                result = ReviewResult(
                    review_id=review_id,
                    status=review.status,
                    reviewer_id=review.reviewer_id,
                    reviewer_note=review.reviewer_note,
                    modified_caption=review.modified_caption,
                    elapsed_time_seconds=elapsed,
                )
                
                # 콜백 실행
                await self._execute_callback(result)
                
                return result
            
            # 대기
            await asyncio.sleep(poll_interval)
    
    async def handle_slack_action(
        self,
        action_id: str,
        review_id: str,
        reviewer_id: str | None = None,
        note: str | None = None,
        modified_caption: str | None = None,
    ) -> ReviewResult:
        """
        Slack 액션 처리
        
        Args:
            action_id: 액션 ID (approve_content, reject_content, modify_content)
            review_id: 검토 요청 ID
            reviewer_id: 검토자 ID
            note: 검토 메모
            modified_caption: 수정된 캡션
        
        Returns:
            ReviewResult: 검토 결과
        """
        action_map = {
            "approve_content": ReviewAction.APPROVED,
            "reject_content": ReviewAction.REJECTED,
            "modify_content": ReviewAction.MODIFIED,
        }
        
        status = action_map.get(action_id)
        if not status:
            raise ValueError(f"Unknown action: {action_id}")
        
        # 상태 업데이트
        review = await self.storage.update(
            review_id,
            status=status,
            reviewer_id=reviewer_id,
            reviewer_note=note,
            modified_caption=modified_caption,
        )
        
        if not review:
            raise ReviewNotFoundError(review_id)
        
        # 연속 승인 카운터 업데이트
        if status == ReviewAction.APPROVED and not modified_caption:
            self._consecutive_approvals += 1
            await self._check_auto_approve_threshold()
        else:
            self._consecutive_approvals = 0
        
        logger.info(
            f"Review {review_id} {status.value} by {reviewer_id}"
        )
        
        return ReviewResult(
            review_id=review_id,
            status=status,
            reviewer_id=reviewer_id,
            reviewer_note=note,
            modified_caption=modified_caption,
        )
    
    async def get_review(self, review_id: str) -> ReviewRequest | None:
        """검토 요청 조회"""
        return await self.storage.get(review_id)
    
    async def get_pending_reviews(self) -> list[ReviewRequest]:
        """대기 중인 검토 목록"""
        return await self.storage.list_pending()
    
    async def cancel_review(self, review_id: str) -> bool:
        """검토 요청 취소"""
        return await self.storage.delete(review_id)
    
    async def _send_slack_review_request(self, review: ReviewRequest) -> bool:
        """Slack 검토 요청 메시지 전송"""
        try:
            client = await self._get_client()
            message = SlackMessageBuilder.build_review_request(review)
            
            response = await client.post(
                self.slack_webhook_url,
                json=message,
            )
            
            if response.status_code == 200:
                logger.info(f"Slack review request sent: {review.review_id}")
                return True
            else:
                logger.error(
                    f"Slack request failed: {response.status_code} - {response.text}"
                )
                return False
                
        except Exception as e:
            logger.exception(f"Failed to send Slack message: {e}")
            return False
    
    async def _send_slack_message(self, message: dict[str, Any]) -> bool:
        """일반 Slack 메시지 전송"""
        try:
            client = await self._get_client()
            
            response = await client.post(
                self.slack_webhook_url,
                json=message,
            )
            
            return response.status_code == 200
            
        except Exception as e:
            logger.exception(f"Failed to send Slack message: {e}")
            return False
    
    async def _check_auto_approve_threshold(self) -> None:
        """자동 승인 전환 체크"""
        if self._consecutive_approvals >= self.AUTO_APPROVE_THRESHOLD:
            logger.info(
                f"Auto-approve threshold reached: {self._consecutive_approvals} consecutive approvals"
            )
            
            # 알림 전송
            message = SlackMessageBuilder.build_auto_approve_alert(
                self._consecutive_approvals
            )
            await self._send_slack_message(message)
    
    async def _execute_callback(self, result: ReviewResult) -> None:
        """결과 콜백 실행"""
        try:
            if result.is_approved and self._on_approved:
                await self._on_approved(result)
            elif result.is_rejected and self._on_rejected:
                await self._on_rejected(result)
            elif result.is_modified and self._on_modified:
                await self._on_modified(result)
        except Exception as e:
            logger.exception(f"Callback error for review {result.review_id}: {e}")
    
    def set_auto_approve(self, enabled: bool) -> None:
        """자동 승인 모드 설정"""
        self.auto_approve = enabled
        logger.info(f"Auto-approve mode: {'enabled' if enabled else 'disabled'}")
    
    def get_consecutive_approvals(self) -> int:
        """연속 승인 횟수 조회"""
        return self._consecutive_approvals
    
    def reset_consecutive_approvals(self) -> None:
        """연속 승인 카운터 리셋"""
        self._consecutive_approvals = 0


# =============================================================================
# Factory Functions
# =============================================================================

def create_review_gate(
    slack_webhook_url: str | None = None,
    redis_client=None,
    auto_approve: bool = False,
) -> HumanReviewGate:
    """
    Review Gate 팩토리 함수
    """
    import os
    
    slack_webhook_url = slack_webhook_url or os.getenv("SLACK_WEBHOOK_URL", "")
    
    if not slack_webhook_url:
        logger.warning("Slack webhook URL not configured")
    
    # 저장소 선택
    storage: ReviewStorage
    if redis_client:
        storage = RedisReviewStorage(redis_client)
    else:
        storage = InMemoryReviewStorage()
    
    return HumanReviewGate(
        slack_webhook_url=slack_webhook_url,
        storage=storage,
        auto_approve=auto_approve,
    )


# =============================================================================
# CLI Interface
# =============================================================================

async def main():
    """CLI 테스트용"""
    import argparse
    
    parser = argparse.ArgumentParser(description="TrendOps Review Gate")
    parser.add_argument("--webhook", required=True, help="Slack Webhook URL")
    parser.add_argument("--keyword", default="테스트", help="Keyword")
    parser.add_argument("--summary", default="테스트 요약입니다.", help="Summary")
    parser.add_argument("--caption", default="테스트 캡션", help="Caption")
    parser.add_argument("--timeout", type=int, default=60, help="Timeout (seconds)")
    
    args = parser.parse_args()
    
    async with HumanReviewGate(slack_webhook_url=args.webhook) as gate:
        print("=" * 60)
        print("📋 검토 요청 제출")
        print("=" * 60)
        
        review_id = await gate.submit_for_review(
            content_id="test_content_001",
            keyword=args.keyword,
            summary=args.summary,
            caption=args.caption,
        )
        
        print(f"Review ID: {review_id}")
        print(f"\n승인 대기 중... (timeout: {args.timeout}s)")
        
        result = await gate.wait_for_approval(
            review_id,
            timeout=args.timeout,
            poll_interval=3,
        )
        
        print("\n" + "=" * 60)
        print("📋 검토 결과")
        print("=" * 60)
        print(f"상태: {result.status.value}")
        print(f"검토자: {result.reviewer_id}")
        print(f"메모: {result.reviewer_note}")
        print(f"소요시간: {result.elapsed_time_seconds:.1f}초")


if __name__ == "__main__":
    asyncio.run(main())