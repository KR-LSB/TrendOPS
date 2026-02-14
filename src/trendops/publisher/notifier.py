# src/trendops/publisher/notifier.py
"""
TrendOps Slack Notifier

Week 5 Day 5: Webhook 기반 알림 시스템

Features:
- 성공/실패/경고/정보 알림
- 파이프라인 상태 알림
- 발행 완료 알림
- 에러 알림 (스택트레이스 포함)
- 일일 리포트 전송
- Rich 메시지 포맷 (Block Kit)

Usage:
    from trendops.publisher.notifier import SlackNotifier, NotificationType

    notifier = SlackNotifier(webhook_url="https://hooks.slack.com/...")

    # 성공 알림
    await notifier.send_success(
        title="발행 완료",
        message="트럼프 관세 콘텐츠가 Instagram에 발행되었습니다.",
        details={"post_url": "https://..."},
    )

    # 실패 알림
    await notifier.send_failure(
        title="발행 실패",
        message="Instagram API 에러",
        error=exception,
    )

    # 일일 리포트
    await notifier.send_daily_report(
        date="2025-01-25",
        stats={"trends": 10, "posts": 5, "errors": 1},
    )

Author: TrendOps Team
Created: Week 5 Day 5
"""

from __future__ import annotations

import asyncio
import logging
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import httpx

# 로깅 설정
logger = logging.getLogger(__name__)


# =============================================================================
# Enums
# =============================================================================


class NotificationType(str, Enum):
    """알림 타입"""

    SUCCESS = "success"
    FAILURE = "failure"
    WARNING = "warning"
    INFO = "info"


class PipelineStage(str, Enum):
    """파이프라인 단계"""

    TRIGGER = "trigger"
    COLLECT = "collect"
    ANALYZE = "analyze"
    GENERATE = "generate"
    REVIEW = "review"
    PUBLISH = "publish"


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class NotificationResult:
    """알림 전송 결과"""

    success: bool
    notification_type: NotificationType
    title: str
    timestamp: datetime = field(default_factory=datetime.now)
    error_message: str | None = None
    response_code: int | None = None

    @property
    def is_success(self) -> bool:
        return self.success


@dataclass
class DailyStats:
    """일일 통계"""

    trends_detected: int = 0
    articles_collected: int = 0
    articles_analyzed: int = 0
    images_generated: int = 0
    posts_published: int = 0
    posts_rejected: int = 0
    errors_count: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "trends_detected": self.trends_detected,
            "articles_collected": self.articles_collected,
            "articles_analyzed": self.articles_analyzed,
            "images_generated": self.images_generated,
            "posts_published": self.posts_published,
            "posts_rejected": self.posts_rejected,
            "errors_count": self.errors_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, int]) -> DailyStats:
        return cls(
            **{
                k: data.get(k, 0)
                for k in [
                    "trends_detected",
                    "articles_collected",
                    "articles_analyzed",
                    "images_generated",
                    "posts_published",
                    "posts_rejected",
                    "errors_count",
                ]
            }
        )


# =============================================================================
# Slack Message Builder
# =============================================================================


class SlackBlockBuilder:
    """Slack Block Kit 메시지 빌더"""

    # 색상 코드
    COLORS = {
        NotificationType.SUCCESS: "#36a64f",
        NotificationType.FAILURE: "#ff0000",
        NotificationType.WARNING: "#ffcc00",
        NotificationType.INFO: "#0066ff",
    }

    # 이모지
    EMOJIS = {
        NotificationType.SUCCESS: "✅",
        NotificationType.FAILURE: "❌",
        NotificationType.WARNING: "⚠️",
        NotificationType.INFO: "ℹ️",
    }

    @classmethod
    def build_notification(
        cls,
        notification_type: NotificationType,
        title: str,
        message: str,
        details: dict[str, Any] | None = None,
        timestamp: datetime | None = None,
    ) -> dict[str, Any]:
        """기본 알림 메시지 빌드"""
        emoji = cls.EMOJIS.get(notification_type, "📢")
        color = cls.COLORS.get(notification_type, "#808080")
        timestamp = timestamp or datetime.now()

        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"{emoji} {title}",
                    "emoji": True,
                },
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": message,
                },
            },
        ]

        # 상세 정보 추가
        if details:
            fields = []
            for key, value in details.items():
                fields.append(
                    {
                        "type": "mrkdwn",
                        "text": f"*{key}:*\n{value}",
                    }
                )

            # 최대 10개 필드 (Slack 제한)
            for i in range(0, len(fields), 2):
                block_fields = fields[i : i + 2]
                blocks.append(
                    {
                        "type": "section",
                        "fields": block_fields,
                    }
                )

        # 타임스탬프
        blocks.append(
            {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": f"📅 {timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
                    }
                ],
            }
        )

        return {
            "attachments": [
                {
                    "color": color,
                    "blocks": blocks,
                }
            ]
        }

    @classmethod
    def build_error_notification(
        cls,
        title: str,
        message: str,
        error: Exception | None = None,
        stack_trace: bool = True,
    ) -> dict[str, Any]:
        """에러 알림 메시지 빌드"""
        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"❌ {title}",
                    "emoji": True,
                },
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": message,
                },
            },
        ]

        if error:
            error_type = type(error).__name__
            error_msg = str(error)

            blocks.append(
                {
                    "type": "section",
                    "fields": [
                        {
                            "type": "mrkdwn",
                            "text": f"*Error Type:*\n`{error_type}`",
                        },
                        {
                            "type": "mrkdwn",
                            "text": f"*Error Message:*\n`{error_msg[:200]}`",
                        },
                    ],
                }
            )

            if stack_trace:
                tb = traceback.format_exception(type(error), error, error.__traceback__)
                tb_text = "".join(tb)[-1500:]  # 마지막 1500자

                blocks.append(
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": f"*Stack Trace:*\n```{tb_text}```",
                        },
                    }
                )

        blocks.append(
            {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                    }
                ],
            }
        )

        return {
            "attachments": [
                {
                    "color": cls.COLORS[NotificationType.FAILURE],
                    "blocks": blocks,
                }
            ]
        }

    @classmethod
    def build_daily_report(
        cls,
        date: str,
        stats: dict[str, int] | DailyStats,
    ) -> dict[str, Any]:
        """일일 리포트 메시지 빌드"""
        if isinstance(stats, DailyStats):
            stats_dict = stats.to_dict()
        else:
            stats_dict = stats

        # 성공률 계산
        total_processed = stats_dict.get("articles_analyzed", 0)
        posts_published = stats_dict.get("posts_published", 0)
        errors = stats_dict.get("errors_count", 0)

        success_rate = 0
        if total_processed > 0:
            success_rate = ((total_processed - errors) / total_processed) * 100

        # 상태 이모지
        if success_rate >= 90:
            status_emoji = "🟢"
            status_text = "Excellent"
        elif success_rate >= 70:
            status_emoji = "🟡"
            status_text = "Good"
        else:
            status_emoji = "🔴"
            status_text = "Needs Attention"

        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"📊 TrendOps 일일 리포트 - {date}",
                    "emoji": True,
                },
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"{status_emoji} *Overall Status:* {status_text} ({success_rate:.1f}%)",
                },
            },
            {"type": "divider"},
            {
                "type": "section",
                "fields": [
                    {
                        "type": "mrkdwn",
                        "text": f"*🔍 트렌드 감지:*\n{stats_dict.get('trends_detected', 0)}건",
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*📰 기사 수집:*\n{stats_dict.get('articles_collected', 0)}건",
                    },
                ],
            },
            {
                "type": "section",
                "fields": [
                    {
                        "type": "mrkdwn",
                        "text": f"*🤖 기사 분석:*\n{stats_dict.get('articles_analyzed', 0)}건",
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*🖼️ 이미지 생성:*\n{stats_dict.get('images_generated', 0)}건",
                    },
                ],
            },
            {
                "type": "section",
                "fields": [
                    {
                        "type": "mrkdwn",
                        "text": f"*📱 발행 완료:*\n{stats_dict.get('posts_published', 0)}건",
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*🚫 발행 거절:*\n{stats_dict.get('posts_rejected', 0)}건",
                    },
                ],
            },
            {"type": "divider"},
            {
                "type": "section",
                "fields": [
                    {
                        "type": "mrkdwn",
                        "text": f"*❌ 에러:*\n{stats_dict.get('errors_count', 0)}건",
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*✅ 성공률:*\n{success_rate:.1f}%",
                    },
                ],
            },
            {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": f"Generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                    }
                ],
            },
        ]

        # 성공률에 따른 색상
        if success_rate >= 90:
            color = cls.COLORS[NotificationType.SUCCESS]
        elif success_rate >= 70:
            color = cls.COLORS[NotificationType.WARNING]
        else:
            color = cls.COLORS[NotificationType.FAILURE]

        return {
            "attachments": [
                {
                    "color": color,
                    "blocks": blocks,
                }
            ]
        }

    @classmethod
    def build_pipeline_status(
        cls,
        keyword: str,
        stage: str | PipelineStage,
        status: str,
        duration: float,
        details: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """파이프라인 상태 메시지 빌드"""
        if isinstance(stage, PipelineStage):
            stage = stage.value

        # 상태에 따른 이모지와 색상
        status_lower = status.lower()
        if status_lower in ("success", "completed", "done"):
            emoji = "✅"
            color = cls.COLORS[NotificationType.SUCCESS]
        elif status_lower in ("failed", "error"):
            emoji = "❌"
            color = cls.COLORS[NotificationType.FAILURE]
        elif status_lower in ("warning", "partial"):
            emoji = "⚠️"
            color = cls.COLORS[NotificationType.WARNING]
        else:
            emoji = "🔄"
            color = cls.COLORS[NotificationType.INFO]

        # 단계 이모지
        stage_emojis = {
            "trigger": "🎯",
            "collect": "📰",
            "analyze": "🤖",
            "generate": "🖼️",
            "review": "📋",
            "publish": "📱",
        }
        stage_emoji = stage_emojis.get(stage.lower(), "⚙️")

        blocks = [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"{emoji} *파이프라인 상태*\n*키워드:* `{keyword}`",
                },
            },
            {
                "type": "section",
                "fields": [
                    {
                        "type": "mrkdwn",
                        "text": f"*{stage_emoji} 단계:*\n{stage.upper()}",
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*상태:*\n{status}",
                    },
                ],
            },
            {
                "type": "section",
                "fields": [
                    {
                        "type": "mrkdwn",
                        "text": f"*⏱️ 소요시간:*\n{duration:.2f}s",
                    },
                ],
            },
        ]

        if details:
            detail_fields = []
            for key, value in list(details.items())[:4]:  # 최대 4개
                detail_fields.append(
                    {
                        "type": "mrkdwn",
                        "text": f"*{key}:*\n{value}",
                    }
                )

            if detail_fields:
                blocks.append(
                    {
                        "type": "section",
                        "fields": detail_fields,
                    }
                )

        blocks.append(
            {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                    }
                ],
            }
        )

        return {
            "attachments": [
                {
                    "color": color,
                    "blocks": blocks,
                }
            ]
        }

    @classmethod
    def build_publish_complete(
        cls,
        keyword: str,
        platform: str,
        post_url: str | None = None,
        image_path: str | None = None,
        metrics: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """발행 완료 메시지 빌드"""
        platform_emojis = {
            "instagram": "📸",
            "threads": "🧵",
            "twitter": "🐦",
            "facebook": "📘",
        }
        platform_emoji = platform_emojis.get(platform.lower(), "📱")

        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": "✅ 발행 완료",
                    "emoji": True,
                },
            },
            {
                "type": "section",
                "fields": [
                    {
                        "type": "mrkdwn",
                        "text": f"*키워드:*\n`{keyword}`",
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*{platform_emoji} 플랫폼:*\n{platform.capitalize()}",
                    },
                ],
            },
        ]

        if post_url:
            blocks.append(
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*🔗 게시물 URL:*\n<{post_url}>",
                    },
                }
            )

        if metrics:
            metric_fields = []
            for key, value in list(metrics.items())[:4]:
                metric_fields.append(
                    {
                        "type": "mrkdwn",
                        "text": f"*{key}:*\n{value}",
                    }
                )

            if metric_fields:
                blocks.append(
                    {
                        "type": "section",
                        "fields": metric_fields,
                    }
                )

        blocks.append(
            {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                    }
                ],
            }
        )

        return {
            "attachments": [
                {
                    "color": cls.COLORS[NotificationType.SUCCESS],
                    "blocks": blocks,
                }
            ]
        }


# =============================================================================
# Slack Notifier
# =============================================================================


class SlackNotifier:
    """
    Slack 알림 발송기

    Week 5 Day 5: Webhook 기반 알림

    Notifications:
    - 파이프라인 성공/실패
    - 발행 완료
    - 에러 알림
    - 일일 리포트

    Usage:
        notifier = SlackNotifier(webhook_url="...")

        await notifier.send_success(
            title="발행 완료",
            message="트럼프 관세 콘텐츠가 Instagram에 발행되었습니다.",
            details={"post_url": "https://..."},
        )
    """

    DEFAULT_TIMEOUT = 30.0

    def __init__(
        self,
        webhook_url: str,
        timeout: float = DEFAULT_TIMEOUT,
        enabled: bool = True,
    ):
        """
        Slack Notifier 초기화

        Args:
            webhook_url: Slack Incoming Webhook URL
            timeout: HTTP 요청 타임아웃
            enabled: 알림 활성화 여부
        """
        self.webhook_url = webhook_url
        self.timeout = timeout
        self.enabled = enabled

        self._client: httpx.AsyncClient | None = None
        self._sent_count = 0
        self._error_count = 0

    async def _get_client(self) -> httpx.AsyncClient:
        """HTTP 클라이언트 (Lazy initialization)"""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout),
                headers={"Content-Type": "application/json"},
            )
        return self._client

    async def close(self) -> None:
        """리소스 정리"""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> SlackNotifier:
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()

    async def _send_message(self, message: dict[str, Any]) -> NotificationResult:
        """Slack 메시지 전송"""
        if not self.enabled:
            logger.debug("Notifications disabled, skipping")
            return NotificationResult(
                success=True,
                notification_type=NotificationType.INFO,
                title="Skipped (disabled)",
            )

        try:
            client = await self._get_client()

            response = await client.post(
                self.webhook_url,
                json=message,
            )

            if response.status_code == 200:
                self._sent_count += 1
                logger.debug("Slack message sent successfully")
                return NotificationResult(
                    success=True,
                    notification_type=NotificationType.SUCCESS,
                    title="Message sent",
                    response_code=response.status_code,
                )
            else:
                self._error_count += 1
                logger.error(f"Slack API error: {response.status_code} - {response.text}")
                return NotificationResult(
                    success=False,
                    notification_type=NotificationType.FAILURE,
                    title="API Error",
                    error_message=f"HTTP {response.status_code}: {response.text}",
                    response_code=response.status_code,
                )

        except httpx.TimeoutException as e:
            self._error_count += 1
            logger.error(f"Slack request timeout: {e}")
            return NotificationResult(
                success=False,
                notification_type=NotificationType.FAILURE,
                title="Timeout",
                error_message=str(e),
            )
        except Exception as e:
            self._error_count += 1
            logger.exception(f"Slack notification error: {e}")
            return NotificationResult(
                success=False,
                notification_type=NotificationType.FAILURE,
                title="Error",
                error_message=str(e),
            )

    async def send(
        self,
        notification_type: NotificationType,
        title: str,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> NotificationResult:
        """
        알림 전송

        Args:
            notification_type: 알림 타입
            title: 제목
            message: 메시지
            details: 상세 정보

        Returns:
            NotificationResult
        """
        slack_message = SlackBlockBuilder.build_notification(
            notification_type=notification_type,
            title=title,
            message=message,
            details=details,
        )

        result = await self._send_message(slack_message)
        result.notification_type = notification_type
        result.title = title

        return result

    async def send_success(
        self,
        title: str,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> NotificationResult:
        """성공 알림"""
        return await self.send(
            notification_type=NotificationType.SUCCESS,
            title=title,
            message=message,
            details=details,
        )

    async def send_failure(
        self,
        title: str,
        message: str,
        error: Exception | None = None,
        include_stack_trace: bool = True,
        details: dict[str, Any] | None = None,
    ) -> NotificationResult:
        """
        실패 알림

        Args:
            title: 제목
            message: 메시지
            error: 예외 객체 (선택)
            include_stack_trace: 스택트레이스 포함 여부
            details: 추가 상세 정보
        """
        if error:
            slack_message = SlackBlockBuilder.build_error_notification(
                title=title,
                message=message,
                error=error,
                stack_trace=include_stack_trace,
            )
        else:
            slack_message = SlackBlockBuilder.build_notification(
                notification_type=NotificationType.FAILURE,
                title=title,
                message=message,
                details=details,
            )

        result = await self._send_message(slack_message)
        result.notification_type = NotificationType.FAILURE
        result.title = title

        return result

    async def send_warning(
        self,
        title: str,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> NotificationResult:
        """경고 알림"""
        return await self.send(
            notification_type=NotificationType.WARNING,
            title=title,
            message=message,
            details=details,
        )

    async def send_info(
        self,
        title: str,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> NotificationResult:
        """정보 알림"""
        return await self.send(
            notification_type=NotificationType.INFO,
            title=title,
            message=message,
            details=details,
        )

    async def send_daily_report(
        self,
        date: str,
        stats: dict[str, int] | DailyStats,
    ) -> NotificationResult:
        """
        일일 리포트 전송

        Args:
            date: 날짜 (YYYY-MM-DD)
            stats: 통계 데이터
        """
        slack_message = SlackBlockBuilder.build_daily_report(
            date=date,
            stats=stats,
        )

        result = await self._send_message(slack_message)
        result.notification_type = NotificationType.INFO
        result.title = f"Daily Report - {date}"

        return result

    async def send_pipeline_status(
        self,
        keyword: str,
        stage: str | PipelineStage,
        status: str,
        duration: float,
        details: dict[str, Any] | None = None,
    ) -> NotificationResult:
        """
        파이프라인 상태 알림

        Args:
            keyword: 트렌드 키워드
            stage: 파이프라인 단계
            status: 상태
            duration: 소요 시간 (초)
            details: 추가 상세 정보
        """
        slack_message = SlackBlockBuilder.build_pipeline_status(
            keyword=keyword,
            stage=stage,
            status=status,
            duration=duration,
            details=details,
        )

        result = await self._send_message(slack_message)
        result.notification_type = NotificationType.INFO
        result.title = f"Pipeline: {keyword} - {stage}"

        return result

    async def send_publish_complete(
        self,
        keyword: str,
        platform: str,
        post_url: str | None = None,
        image_path: str | None = None,
        metrics: dict[str, Any] | None = None,
    ) -> NotificationResult:
        """
        발행 완료 알림

        Args:
            keyword: 트렌드 키워드
            platform: 플랫폼 (instagram, threads 등)
            post_url: 게시물 URL
            image_path: 이미지 경로
            metrics: 성능 메트릭
        """
        slack_message = SlackBlockBuilder.build_publish_complete(
            keyword=keyword,
            platform=platform,
            post_url=post_url,
            image_path=image_path,
            metrics=metrics,
        )

        result = await self._send_message(slack_message)
        result.notification_type = NotificationType.SUCCESS
        result.title = f"Published: {keyword} to {platform}"

        return result

    def get_stats(self) -> dict[str, int]:
        """전송 통계"""
        return {
            "sent_count": self._sent_count,
            "error_count": self._error_count,
        }

    def reset_stats(self) -> None:
        """통계 리셋"""
        self._sent_count = 0
        self._error_count = 0


# =============================================================================
# Factory Functions
# =============================================================================


def create_notifier(
    webhook_url: str | None = None,
    enabled: bool = True,
) -> SlackNotifier:
    """
    Notifier 팩토리 함수

    환경 변수에서 webhook URL 로드 가능
    """
    import os

    webhook_url = webhook_url or os.getenv("SLACK_WEBHOOK_URL", "")

    if not webhook_url:
        logger.warning("Slack webhook URL not configured")
        enabled = False

    return SlackNotifier(
        webhook_url=webhook_url,
        enabled=enabled,
    )


# =============================================================================
# CLI Interface
# =============================================================================


async def main():
    """CLI 테스트용"""
    import argparse

    parser = argparse.ArgumentParser(description="TrendOps Slack Notifier")
    parser.add_argument("--webhook", required=True, help="Slack Webhook URL")
    parser.add_argument(
        "--type",
        choices=["success", "failure", "warning", "info"],
        default="info",
        help="Notification type",
    )
    parser.add_argument("--title", default="Test Notification", help="Title")
    parser.add_argument("--message", default="This is a test message.", help="Message")
    parser.add_argument("--daily-report", action="store_true", help="Send daily report")

    args = parser.parse_args()

    async with SlackNotifier(webhook_url=args.webhook) as notifier:
        if args.daily_report:
            result = await notifier.send_daily_report(
                date=datetime.now().strftime("%Y-%m-%d"),
                stats={
                    "trends_detected": 15,
                    "articles_collected": 120,
                    "articles_analyzed": 115,
                    "images_generated": 10,
                    "posts_published": 8,
                    "posts_rejected": 2,
                    "errors_count": 3,
                },
            )
        else:
            notification_type = NotificationType(args.type)
            result = await notifier.send(
                notification_type=notification_type,
                title=args.title,
                message=args.message,
            )

        print(f"Result: {'Success' if result.success else 'Failed'}")
        if result.error_message:
            print(f"Error: {result.error_message}")


if __name__ == "__main__":
    asyncio.run(main())
