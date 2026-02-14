# src/trendops/publisher/scheduler.py
"""
TrendOps Pipeline Scheduler

Week 5 Day 4: APScheduler 기반 자동 실행
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

# 로깅 설정
logger = logging.getLogger(__name__)


# =============================================================================
# Job 상태 및 설정
# =============================================================================


class JobStatus(str, Enum):
    """Job 상태"""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"


class TriggerType(str, Enum):
    """트리거 타입"""

    CRON = "cron"
    INTERVAL = "interval"
    DATE = "date"


@dataclass
class JobConfig:
    """Job 설정"""

    job_id: str
    name: str
    trigger_type: TriggerType
    trigger_args: dict[str, Any]
    func: Callable[..., Awaitable[Any]]
    max_instances: int = 1
    misfire_grace_time: int = 300
    enabled: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "name": self.name,
            "trigger_type": self.trigger_type.value,
            "trigger_args": self.trigger_args,
            "max_instances": self.max_instances,
            "misfire_grace_time": self.misfire_grace_time,
            "enabled": self.enabled,
        }


@dataclass
class JobResult:
    """Job 실행 결과"""

    job_id: str
    status: JobStatus
    started_at: datetime
    completed_at: datetime | None = None
    error_message: str | None = None
    result: Any = None

    @property
    def duration_seconds(self) -> float | None:
        if self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None


@dataclass
class JobInfo:
    """Job 정보"""

    job_id: str
    name: str
    trigger_type: str
    next_run_time: datetime | None
    last_run_time: datetime | None = None
    last_status: JobStatus = JobStatus.PENDING
    run_count: int = 0
    error_count: int = 0
    is_paused: bool = False


# =============================================================================
# Pipeline Callbacks
# =============================================================================


class PipelineCallbacks:
    """파이프라인 콜백 인터페이스"""

    async def trigger_pipeline(self) -> dict[str, Any]:
        logger.info("Executing trigger pipeline")
        return {"status": "completed", "trends_detected": 0}

    async def analysis_pipeline(self) -> dict[str, Any]:
        logger.info("Executing analysis pipeline")
        return {"status": "completed", "items_analyzed": 0}

    async def publish_pipeline(self) -> dict[str, Any]:
        logger.info("Executing publish pipeline")
        return {"status": "completed", "posts_published": 0}

    async def daily_report(self) -> dict[str, Any]:
        logger.info("Executing daily report")
        return {"status": "completed", "report_generated": True}


# =============================================================================
# TrendOps Scheduler
# =============================================================================


class TrendOpsScheduler:
    """
    TrendOps 파이프라인 스케줄러
    Week 5 Day 4: APScheduler 통합
    """

    def __init__(
        self,
        callbacks: PipelineCallbacks | None = None,
        timezone: str = "Asia/Seoul",
    ):
        self.callbacks = callbacks or PipelineCallbacks()
        self.timezone = timezone

        self._scheduler = None
        self._jobs: dict[str, JobConfig] = {}
        self._job_history: dict[str, list[JobResult]] = {}
        self._job_info: dict[str, JobInfo] = {}
        self._running = False

        self._setup_default_jobs()

    def _setup_default_jobs(self) -> None:
        """기본 Job 등록"""
        default_jobs = [
            JobConfig(
                job_id="trend_detection",
                name="트렌드 감지",
                trigger_type=TriggerType.CRON,
                trigger_args={"minute": 0},
                func=self.callbacks.trigger_pipeline,
            ),
            JobConfig(
                job_id="data_analysis",
                name="데이터 수집/분석",
                trigger_type=TriggerType.CRON,
                trigger_args={"minute": 10},
                func=self.callbacks.analysis_pipeline,
            ),
            JobConfig(
                job_id="content_publish",
                name="SNS 발행",
                trigger_type=TriggerType.CRON,
                trigger_args={"hour": "9,14,20", "minute": 0},
                func=self.callbacks.publish_pipeline,
                misfire_grace_time=600,
            ),
            JobConfig(
                job_id="daily_report",
                name="일일 리포트",
                trigger_type=TriggerType.CRON,
                trigger_args={"hour": 23, "minute": 0},
                func=self.callbacks.daily_report,
            ),
        ]

        for job_config in default_jobs:
            self._jobs[job_config.job_id] = job_config
            self._job_info[job_config.job_id] = JobInfo(
                job_id=job_config.job_id,
                name=job_config.name,
                trigger_type=job_config.trigger_type.value,
                next_run_time=None,
            )
            self._job_history[job_config.job_id] = []

    def _initialize_scheduler(self) -> None:
        """APScheduler 초기화"""
        try:
            from apscheduler.schedulers.asyncio import AsyncIOScheduler
            from apscheduler.triggers.cron import CronTrigger
            from apscheduler.triggers.date import DateTrigger
            from apscheduler.triggers.interval import IntervalTrigger

            self._scheduler = AsyncIOScheduler(timezone=self.timezone)
            self._CronTrigger = CronTrigger
            self._IntervalTrigger = IntervalTrigger
            self._DateTrigger = DateTrigger

            for job_id, config in self._jobs.items():
                if config.enabled:
                    self._add_job_to_scheduler(config)

            logger.info("APScheduler initialized")

        except ImportError:
            logger.warning("APScheduler not installed. Using fallback scheduler.")
            self._scheduler = None

    def _add_job_to_scheduler(self, config: JobConfig) -> None:
        """APScheduler에 Job 추가"""
        if not self._scheduler:
            return

        if config.trigger_type == TriggerType.CRON:
            trigger = self._CronTrigger(**config.trigger_args)
        elif config.trigger_type == TriggerType.INTERVAL:
            trigger = self._IntervalTrigger(**config.trigger_args)
        elif config.trigger_type == TriggerType.DATE:
            trigger = self._DateTrigger(**config.trigger_args)
        else:
            raise ValueError(f"Unknown trigger type: {config.trigger_type}")

        async def job_wrapper():
            await self._execute_job(config.job_id)

        self._scheduler.add_job(
            job_wrapper,
            trigger,
            id=config.job_id,
            name=config.name,
            max_instances=config.max_instances,
            misfire_grace_time=config.misfire_grace_time,
            replace_existing=True,
        )

        logger.info(f"Job registered: {config.job_id} ({config.name})")

    async def _execute_job(self, job_id: str) -> JobResult:
        """Job 실행"""
        config = self._jobs.get(job_id)
        if not config:
            logger.error(f"Job not found: {job_id}")
            return JobResult(
                job_id=job_id,
                status=JobStatus.FAILED,
                started_at=datetime.now(),
                completed_at=datetime.now(),
                error_message="Job not found",
            )

        started_at = datetime.now()
        info = self._job_info[job_id]
        info.last_run_time = started_at
        info.last_status = JobStatus.RUNNING

        logger.info(f"Job started: {job_id} ({config.name})")

        try:
            result = await config.func()

            completed_at = datetime.now()
            job_result = JobResult(
                job_id=job_id,
                status=JobStatus.COMPLETED,
                started_at=started_at,
                completed_at=completed_at,
                result=result,
            )

            info.last_status = JobStatus.COMPLETED
            info.run_count += 1

            logger.info(
                f"Job completed: {job_id} ({config.name}) in {job_result.duration_seconds:.2f}s"
            )

        except Exception as e:
            completed_at = datetime.now()
            job_result = JobResult(
                job_id=job_id,
                status=JobStatus.FAILED,
                started_at=started_at,
                completed_at=completed_at,
                error_message=str(e),
            )

            info.last_status = JobStatus.FAILED
            info.run_count += 1
            info.error_count += 1
            logger.error(f"Job failed: {job_id} ({config.name}) - {e}")

        history = self._job_history[job_id]
        history.append(job_result)
        if len(history) > 100:
            self._job_history[job_id] = history[-100:]

        return job_result

    def start(self) -> None:
        """스케줄러 시작"""
        if self._running:
            logger.warning("Scheduler already running")
            return

        self._initialize_scheduler()

        if self._scheduler:
            try:
                self._scheduler.start()
                self._running = True
                logger.info("TrendOps Scheduler started")
                self._update_next_run_times()
            except RuntimeError as e:
                if "no running event loop" in str(e):
                    logger.warning(
                        "Cannot start AsyncIOScheduler outside async context. "
                        "Use 'await scheduler.start_async()' or run within event loop."
                    )
                    # 내부 스케줄러는 시작 안 됐지만, 상태는 running으로 관리 (수동 실행 등을 위해)
                    self._running = True
                else:
                    raise
        else:
            logger.warning("Running without APScheduler")
            self._running = True

    async def start_async(self) -> None:
        """스케줄러 시작 (비동기)"""
        if self._running:
            logger.warning("Scheduler already running")
            return

        self._initialize_scheduler()

        if self._scheduler:
            self._scheduler.start()
            self._running = True
            logger.info("TrendOps Scheduler started (async)")
            self._update_next_run_times()
        else:
            logger.warning("Running without APScheduler")
            self._running = True

    def shutdown(self, wait: bool = True) -> None:
        """스케줄러 종료 (수정됨: 안전한 종료 처리)"""
        if not self._running:
            return

        if self._scheduler:
            # 수정된 부분: 스케줄러가 실제로 실행 중일 때만 종료 시도
            try:
                if self._scheduler.running:
                    self._scheduler.shutdown(wait=wait)
            except Exception as e:
                # 이미 종료되었거나 실행되지 않은 상태라면 에러를 무시
                logger.debug(f"Scheduler shutdown ignored: {e}")

            self._scheduler = None

        self._running = False
        logger.info("TrendOps Scheduler stopped")

    def _update_next_run_times(self) -> None:
        if not self._scheduler:
            return

        try:
            for job in self._scheduler.get_jobs():
                info = self._job_info.get(job.id)
                if info:
                    info.next_run_time = job.next_run_time
        except Exception:
            pass

    def pause_job(self, job_id: str) -> bool:
        if not self._scheduler:
            return False

        try:
            self._scheduler.pause_job(job_id)
            info = self._job_info.get(job_id)
            if info:
                info.is_paused = True
                info.next_run_time = None
            logger.info(f"Job paused: {job_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to pause job {job_id}: {e}")
            return False

    def resume_job(self, job_id: str) -> bool:
        if not self._scheduler:
            return False

        try:
            self._scheduler.resume_job(job_id)
            info = self._job_info.get(job_id)
            if info:
                info.is_paused = False
            self._update_next_run_times()
            logger.info(f"Job resumed: {job_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to resume job {job_id}: {e}")
            return False

    def remove_job(self, job_id: str) -> bool:
        if self._scheduler:
            try:
                self._scheduler.remove_job(job_id)
            except Exception:
                pass

        if job_id in self._jobs:
            del self._jobs[job_id]
            del self._job_info[job_id]
            del self._job_history[job_id]
            logger.info(f"Job removed: {job_id}")
            return True

        return False

    def add_custom_job(
        self,
        func: Callable[..., Awaitable[Any]],
        trigger: str,
        job_id: str,
        name: str | None = None,
        max_instances: int = 1,
        misfire_grace_time: int = 300,
        **trigger_args,
    ) -> bool:
        try:
            trigger_type = TriggerType(trigger)
        except ValueError:
            logger.error(f"Invalid trigger type: {trigger}")
            return False

        config = JobConfig(
            job_id=job_id,
            name=name or job_id,
            trigger_type=trigger_type,
            trigger_args=trigger_args,
            func=func,
            max_instances=max_instances,
            misfire_grace_time=misfire_grace_time,
        )

        self._jobs[job_id] = config
        self._job_info[job_id] = JobInfo(
            job_id=job_id,
            name=config.name,
            trigger_type=trigger_type.value,
            next_run_time=None,
        )
        self._job_history[job_id] = []

        if self._scheduler and self._running:
            self._add_job_to_scheduler(config)
            self._update_next_run_times()

        logger.info(f"Custom job added: {job_id}")
        return True

    async def run_job_now(self, job_id: str) -> JobResult:
        return await self._execute_job(job_id)

    def get_jobs_status(self) -> list[dict[str, Any]]:
        self._update_next_run_times()

        return [
            {
                "job_id": info.job_id,
                "name": info.name,
                "trigger_type": info.trigger_type,
                "next_run_time": info.next_run_time.isoformat() if info.next_run_time else None,
                "last_run_time": info.last_run_time.isoformat() if info.last_run_time else None,
                "last_status": info.last_status.value,
                "run_count": info.run_count,
                "error_count": info.error_count,
                "is_paused": info.is_paused,
            }
            for info in self._job_info.values()
        ]

    def get_job_history(self, job_id: str, limit: int = 10) -> list[dict[str, Any]]:
        history = self._job_history.get(job_id, [])
        return [
            {
                "job_id": r.job_id,
                "status": r.status.value,
                "started_at": r.started_at.isoformat(),
                "completed_at": r.completed_at.isoformat() if r.completed_at else None,
                "duration_seconds": r.duration_seconds,
                "error_message": r.error_message,
            }
            for r in history[-limit:]
        ]

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def job_count(self) -> int:
        return len(self._jobs)


# =============================================================================
# Factory Functions
# =============================================================================


def create_scheduler(
    callbacks: PipelineCallbacks | None = None,
    timezone: str = "Asia/Seoul",
) -> TrendOpsScheduler:
    return TrendOpsScheduler(
        callbacks=callbacks,
        timezone=timezone,
    )


# =============================================================================
# CLI Interface
# =============================================================================


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="TrendOps Scheduler")
    parser.add_argument("--list", action="store_true", help="List jobs")
    parser.add_argument("--run", help="Run job immediately")
    parser.add_argument("--start", action="store_true", help="Start scheduler")
    parser.add_argument("--duration", type=int, default=60, help="Run duration (seconds)")

    args = parser.parse_args()
    scheduler = TrendOpsScheduler()

    if args.list:
        print("=" * 60)
        print("📋 등록된 Jobs")
        print("=" * 60)
        for job in scheduler.get_jobs_status():
            print(f"\n[{job['job_id']}] {job['name']}")
            print(f"  트리거: {job['trigger_type']}")
            print(f"  다음 실행: {job['next_run_time'] or 'N/A'}")
        return

    if args.run:
        print(f"Running job: {args.run}")
        result = await scheduler.run_job_now(args.run)
        print(f"Status: {result.status.value}")
        return

    if args.start:
        print("Starting scheduler...")
        scheduler.start()
        try:
            await asyncio.sleep(args.duration)
        finally:
            scheduler.shutdown()
            print("Scheduler stopped")


if __name__ == "__main__":
    asyncio.run(main())
