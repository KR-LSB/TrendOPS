# src/trendops/queue/queue_redis.py
import json
from datetime import datetime
from typing import AsyncGenerator
from uuid import UUID

import redis.asyncio as aioredis
from redis.asyncio import Redis
from redis.exceptions import ConnectionError, RedisError

from trendops.config.settings import get_settings
from trendops.queue.queue_models import JobStatus, TrendJob
from trendops.utils.logger import get_logger

logger = get_logger(__name__)


class RedisKeys:
    """Redis Key 패턴 정의 (Blueprint Section 3.3)"""
    JOBS_PENDING = "jobs:pending"
    JOBS_PROCESSING = "jobs:processing"
    JOBS_COMPLETED = "jobs:completed"
    
    @staticmethod
    def job_detail(job_id: UUID | str) -> str:
        return f"job:{job_id}"


class RedisQueue:
    """
    Redis 기반 Job Queue 관리자
    
    Blueprint Section 3.3 Redis Key Design 준수:
    - jobs:pending: LIST (대기 중인 Job IDs)
    - jobs:processing: SET (처리 중인 Job IDs)  
    - jobs:completed: SORTED SET (완료된 Jobs, score=timestamp)
    - job:{job_id}: HASH (Job 상세 정보)
    """
    
    def __init__(self, redis_client: Redis | None = None):
        self._client = redis_client
        self._settings = get_settings()
    
    async def connect(self) -> None:
        """Redis 연결 초기화"""
        if self._client is None:
            try:
                self._client = await aioredis.from_url(
                    self._settings.redis_url,
                    encoding="utf-8",
                    decode_responses=True,
                    max_connections=10,
                )
                await self._client.ping()
                logger.info("Redis connected", extra={"url": self._settings.redis_url})
            except ConnectionError as e:
                logger.error("Redis connection failed", extra={"error": str(e)})
                raise
    
    async def disconnect(self) -> None:
        """Redis 연결 종료"""
        if self._client:
            await self._client.close()
            self._client = None
            logger.info("Redis disconnected")
    
    @property
    def client(self) -> Redis:
        """Redis 클라이언트 반환"""
        if self._client is None:
            raise RuntimeError("Redis not connected. Call connect() first.")
        return self._client
    
    async def push_job(self, job: TrendJob) -> str:
        """
        Job을 pending 큐에 추가
        
        Args:
            job: TrendJob 인스턴스
            
        Returns:
            job_id 문자열
        """
        job_id = str(job.job_id)
        job_key = RedisKeys.job_detail(job_id)
        
        try:
            pipe = self.client.pipeline()
            
            # Job 상세 정보 저장 (HASH)
            pipe.hset(job_key, mapping={
                "job_id": job_id,
                "keyword": job.keyword_info.keyword,
                "source": job.keyword_info.source.value,
                "trend_score": str(job.keyword_info.trend_score),
                "status": job.status.value,
                "created_at": job.created_at.isoformat(),
                "updated_at": job.updated_at.isoformat(),
                "data": job.model_dump_json(),
            })
            
            # TTL 설정 (7일)
            pipe.expire(job_key, 60 * 60 * 24 * 7)
            
            # pending 리스트에 job_id 추가 (RPUSH - FIFO)
            pipe.rpush(RedisKeys.JOBS_PENDING, job_id)
            
            await pipe.execute()
            
            logger.info(
                "Job pushed to pending queue",
                extra={"job_id": job_id, "keyword": job.keyword_info.keyword}
            )
            return job_id
            
        except RedisError as e:
            logger.error("Failed to push job", extra={"job_id": job_id, "error": str(e)})
            raise
    
    async def pop_job(self, timeout: int = 0) -> TrendJob | None:
        """
        Pending 큐에서 Job을 블로킹 팝
        
        Args:
            timeout: 블로킹 타임아웃 (초), 0이면 무한 대기
            
        Returns:
            TrendJob 또는 None (타임아웃 시)
        """
        try:
            # BLPOP: 블로킹 팝 (FIFO)
            result = await self.client.blpop(
                RedisKeys.JOBS_PENDING,
                timeout=timeout
            )
            
            if result is None:
                return None
            
            _, job_id = result
            
            # Job 상세 정보 조회
            job_key = RedisKeys.job_detail(job_id)
            job_data = await self.client.hget(job_key, "data")
            
            if job_data is None:
                logger.warning("Job data not found", extra={"job_id": job_id})
                return None
            
            job = TrendJob.model_validate_json(job_data)
            
            logger.info(
                "Job popped from pending queue",
                extra={"job_id": job_id, "keyword": job.keyword_info.keyword}
            )
            return job
            
        except RedisError as e:
            logger.error("Failed to pop job", extra={"error": str(e)})
            raise
    
    async def mark_processing(self, job_id: UUID | str) -> bool:
        """
        Job 상태를 processing으로 변경
        
        Args:
            job_id: Job ID
            
        Returns:
            성공 여부
        """
        job_id_str = str(job_id)
        job_key = RedisKeys.job_detail(job_id_str)
        
        try:
            pipe = self.client.pipeline()
            
            # processing SET에 추가
            pipe.sadd(RedisKeys.JOBS_PROCESSING, job_id_str)
            
            # Job 상태 업데이트
            now = datetime.now().isoformat()
            pipe.hset(job_key, mapping={
                "status": JobStatus.PROCESSING.value,
                "updated_at": now,
            })
            
            # 전체 Job 데이터 업데이트
            job_data = await self.client.hget(job_key, "data")
            if job_data:
                job = TrendJob.model_validate_json(job_data)
                job.mark_processing()
                pipe.hset(job_key, "data", job.model_dump_json())
            
            await pipe.execute()
            
            logger.info("Job marked as processing", extra={"job_id": job_id_str})
            return True
            
        except RedisError as e:
            logger.error(
                "Failed to mark job as processing",
                extra={"job_id": job_id_str, "error": str(e)}
            )
            return False
    
    async def mark_completed(self, job_id: UUID | str) -> bool:
        """
        Job 상태를 completed로 변경
        
        Args:
            job_id: Job ID
            
        Returns:
            성공 여부
        """
        job_id_str = str(job_id)
        job_key = RedisKeys.job_detail(job_id_str)
        
        try:
            pipe = self.client.pipeline()
            
            # processing SET에서 제거
            pipe.srem(RedisKeys.JOBS_PROCESSING, job_id_str)
            
            # completed SORTED SET에 추가 (score = timestamp)
            timestamp = datetime.now().timestamp()
            pipe.zadd(RedisKeys.JOBS_COMPLETED, {job_id_str: timestamp})
            
            # Job 상태 업데이트
            now = datetime.now().isoformat()
            pipe.hset(job_key, mapping={
                "status": JobStatus.COMPLETED.value,
                "updated_at": now,
            })
            
            # 전체 Job 데이터 업데이트
            job_data = await self.client.hget(job_key, "data")
            if job_data:
                job = TrendJob.model_validate_json(job_data)
                job.mark_completed()
                pipe.hset(job_key, "data", job.model_dump_json())
            
            await pipe.execute()
            
            logger.info("Job marked as completed", extra={"job_id": job_id_str})
            return True
            
        except RedisError as e:
            logger.error(
                "Failed to mark job as completed",
                extra={"job_id": job_id_str, "error": str(e)}
            )
            return False
    
    async def mark_failed(self, job_id: UUID | str, error_message: str) -> bool:
        """
        Job 상태를 failed로 변경
        
        Args:
            job_id: Job ID
            error_message: 에러 메시지
            
        Returns:
            성공 여부
        """
        job_id_str = str(job_id)
        job_key = RedisKeys.job_detail(job_id_str)
        
        try:
            pipe = self.client.pipeline()
            
            # processing SET에서 제거
            pipe.srem(RedisKeys.JOBS_PROCESSING, job_id_str)
            
            # Job 상태 업데이트
            now = datetime.now().isoformat()
            pipe.hset(job_key, mapping={
                "status": JobStatus.FAILED.value,
                "updated_at": now,
                "error_message": error_message,
            })
            
            # 전체 Job 데이터 업데이트
            job_data = await self.client.hget(job_key, "data")
            if job_data:
                job = TrendJob.model_validate_json(job_data)
                job.mark_failed(error_message)
                pipe.hset(job_key, "data", job.model_dump_json())
            
            await pipe.execute()
            
            logger.warning(
                "Job marked as failed",
                extra={"job_id": job_id_str, "error": error_message}
            )
            return True
            
        except RedisError as e:
            logger.error(
                "Failed to mark job as failed",
                extra={"job_id": job_id_str, "error": str(e)}
            )
            return False
    
    async def get_job(self, job_id: UUID | str) -> TrendJob | None:
        """
        Job 상세 정보 조회
        
        Args:
            job_id: Job ID
            
        Returns:
            TrendJob 또는 None
        """
        job_key = RedisKeys.job_detail(str(job_id))
        
        try:
            job_data = await self.client.hget(job_key, "data")
            if job_data is None:
                return None
            return TrendJob.model_validate_json(job_data)
        except RedisError as e:
            logger.error("Failed to get job", extra={"job_id": str(job_id), "error": str(e)})
            return None
    
    async def get_queue_stats(self) -> dict[str, int]:
        """
        큐 통계 조회
        
        Returns:
            각 상태별 Job 수
        """
        try:
            pipe = self.client.pipeline()
            pipe.llen(RedisKeys.JOBS_PENDING)
            pipe.scard(RedisKeys.JOBS_PROCESSING)
            pipe.zcard(RedisKeys.JOBS_COMPLETED)
            
            results = await pipe.execute()
            
            return {
                "pending": results[0],
                "processing": results[1],
                "completed": results[2],
            }
        except RedisError as e:
            logger.error("Failed to get queue stats", extra={"error": str(e)})
            return {"pending": 0, "processing": 0, "completed": 0}
    
    async def requeue_stale_jobs(self, stale_threshold_seconds: int = 300) -> int:
        """
        오래된 processing Job을 pending으로 재큐잉
        
        Args:
            stale_threshold_seconds: 오래됨 판단 기준 (초)
            
        Returns:
            재큐잉된 Job 수
        """
        try:
            processing_jobs = await self.client.smembers(RedisKeys.JOBS_PROCESSING)
            requeued_count = 0
            
            for job_id in processing_jobs:
                job_key = RedisKeys.job_detail(job_id)
                updated_at_str = await self.client.hget(job_key, "updated_at")
                
                if updated_at_str:
                    updated_at = datetime.fromisoformat(updated_at_str)
                    age_seconds = (datetime.now() - updated_at).total_seconds()
                    
                    if age_seconds > stale_threshold_seconds:
                        pipe = self.client.pipeline()
                        pipe.srem(RedisKeys.JOBS_PROCESSING, job_id)
                        pipe.lpush(RedisKeys.JOBS_PENDING, job_id)
                        pipe.hset(job_key, "status", JobStatus.PENDING.value)
                        await pipe.execute()
                        
                        requeued_count += 1
                        logger.info(
                            "Stale job requeued",
                            extra={"job_id": job_id, "age_seconds": age_seconds}
                        )
            
            return requeued_count
            
        except RedisError as e:
            logger.error("Failed to requeue stale jobs", extra={"error": str(e)})
            return 0


async def get_redis_queue() -> AsyncGenerator[RedisQueue, None]:
    """FastAPI Dependency용 Redis Queue 생성기"""
    queue = RedisQueue()
    await queue.connect()
    try:
        yield queue
    finally:
        await queue.disconnect()