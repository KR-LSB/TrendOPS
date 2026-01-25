# src/trendops/pipeline/data_pipeline.py
"""
병렬 처리 데이터 파이프라인

Blueprint 준수:
- Phase 1-2: multiprocessing으로 충분
- Phase 3: 데이터 100배 증가 시 Ray로 마이그레이션

포트폴리오 스토리:
"초기에는 표준 라이브러리로 구현 후,
 확장성 필요 시 Ray로 마이그레이션 가능한 설계"

⚠️ CRITICAL HARDWARE CONSTRAINT:
- CPU 바운드 작업: ProcessPoolExecutor (8 workers)
- I/O 바운드 작업: asyncio.gather + Semaphore
- GPU는 vLLM 전용 (이 모듈에서 사용 금지)
"""
from __future__ import annotations

import asyncio
import functools
import logging
import time
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Generic, TypeVar

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Type Variables
T = TypeVar("T")
R = TypeVar("R")


# =============================================================================
# DATA MODELS
# =============================================================================


class TaskStatus(str, Enum):
    """작업 상태"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskResult(BaseModel, Generic[T]):
    """
    작업 결과 모델
    
    성공/실패 여부와 결과/에러를 함께 담는 컨테이너
    """
    
    success: bool = Field(..., description="성공 여부")
    result: T | None = Field(default=None, description="작업 결과")
    error: str | None = Field(default=None, description="에러 메시지")
    task_id: str | None = Field(default=None, description="작업 ID")
    elapsed_ms: float = Field(default=0.0, description="소요 시간 (ms)")
    
    class Config:
        arbitrary_types_allowed = True


class BatchResult(BaseModel, Generic[T]):
    """
    배치 처리 결과 모델
    
    전체 배치의 성공/실패 통계 포함
    """
    
    total: int = Field(..., description="전체 작업 수")
    succeeded: int = Field(default=0, description="성공한 작업 수")
    failed: int = Field(default=0, description="실패한 작업 수")
    results: list[TaskResult[T]] = Field(default_factory=list, description="개별 결과")
    elapsed_ms: float = Field(default=0.0, description="총 소요 시간 (ms)")
    
    @property
    def success_rate(self) -> float:
        """성공률"""
        if self.total == 0:
            return 0.0
        return self.succeeded / self.total
    
    @property
    def successful_results(self) -> list[T]:
        """성공한 결과만 반환"""
        return [r.result for r in self.results if r.success and r.result is not None]
    
    class Config:
        arbitrary_types_allowed = True


class PipelineStats(BaseModel):
    """파이프라인 실행 통계"""
    
    pipeline_name: str = Field(..., description="파이프라인 이름")
    started_at: datetime = Field(default_factory=datetime.now, description="시작 시간")
    ended_at: datetime | None = Field(default=None, description="종료 시간")
    
    # 처리 통계
    total_items: int = Field(default=0, description="전체 아이템 수")
    processed_items: int = Field(default=0, description="처리된 아이템 수")
    failed_items: int = Field(default=0, description="실패한 아이템 수")
    
    # 성능 통계
    cpu_tasks: int = Field(default=0, description="CPU 작업 수")
    io_tasks: int = Field(default=0, description="I/O 작업 수")
    avg_task_ms: float = Field(default=0.0, description="평균 작업 시간 (ms)")
    
    @property
    def elapsed_seconds(self) -> float:
        """총 소요 시간 (초)"""
        end = self.ended_at or datetime.now()
        return (end - self.started_at).total_seconds()
    
    @property
    def throughput(self) -> float:
        """처리량 (items/sec)"""
        elapsed = self.elapsed_seconds
        if elapsed == 0:
            return 0.0
        return self.processed_items / elapsed
    
    @property
    def success_rate(self) -> float:
        """성공률"""
        if self.total_items == 0:
            return 0.0
        return self.processed_items / self.total_items


# =============================================================================
# DATA PIPELINE
# =============================================================================


class DataPipeline:
    """
    병렬 처리 데이터 파이프라인
    
    Phase 1-2: multiprocessing으로 충분
    Phase 3: 데이터 100배 증가 시 Ray로 마이그레이션
    
    포트폴리오 스토리:
    "초기에는 표준 라이브러리로 구현 후,
     확장성 필요 시 Ray로 마이그레이션"
    
    사용 예시:
        pipeline = DataPipeline(num_workers=8)
        
        # CPU 바운드 배치 처리
        results = pipeline.process_batch_cpu(
            documents=["doc1", "doc2", ...],
            processor_func=tokenize_document,
        )
        
        # I/O 바운드 배치 처리
        results = await pipeline.process_batch_io(
            items=urls,
            async_func=fetch_url,
            max_concurrent=10,
        )
    """
    
    # 기본 워커 수 (CPU 코어 기반)
    DEFAULT_CPU_WORKERS: int = 8
    DEFAULT_IO_WORKERS: int = 16
    DEFAULT_MAX_CONCURRENT: int = 10
    
    def __init__(
        self,
        num_workers: int = DEFAULT_CPU_WORKERS,
        max_concurrent_io: int = DEFAULT_MAX_CONCURRENT,
        name: str = "DataPipeline",
        use_threads: bool = False,
    ):
        """
        Args:
            num_workers: CPU 작업용 프로세스 수
            max_concurrent_io: I/O 작업 동시 실행 수
            name: 파이프라인 이름 (로깅용)
            use_threads: True면 ProcessPool 대신 ThreadPool 사용
                         (로컬 함수/람다 지원, 하지만 GIL 제약)
        """
        self.num_workers = num_workers
        self.max_concurrent_io = max_concurrent_io
        self.name = name
        self.use_threads = use_threads
        
        # 통계
        self._stats = PipelineStats(pipeline_name=name)
        
        logger.info(
            f"DataPipeline '{name}' initialized",
            extra={
                "num_workers": num_workers,
                "max_concurrent_io": max_concurrent_io,
            }
        )
    
    # =========================================================================
    # CPU BOUND PROCESSING
    # =========================================================================
    
    def process_batch_cpu(
        self,
        items: list[T],
        processor_func: Callable[[T], R],
        timeout: float | None = None,
        ordered: bool = False,
    ) -> BatchResult[R]:
        """
        CPU 바운드 배치 처리 (ProcessPoolExecutor)
        
        ⚠️ processor_func는 반드시 picklable해야 함
           (모듈 레벨 함수 또는 staticmethod 권장)
        
        Args:
            items: 처리할 아이템 리스트
            processor_func: 처리 함수 (picklable)
            timeout: 개별 작업 타임아웃 (초)
            ordered: 결과 순서 유지 여부
            
        Returns:
            BatchResult[R]: 배치 처리 결과
        """
        if not items:
            return BatchResult(total=0, succeeded=0, failed=0)
        
        start_time = time.time()
        results: list[TaskResult[R]] = []
        succeeded = 0
        failed = 0
        
        logger.debug(f"Processing {len(items)} items with {self.num_workers} CPU workers")
        
        # ThreadPool vs ProcessPool 선택
        # - ProcessPool: CPU 바운드 작업에 적합 (GIL 회피), 하지만 pickle 필요
        # - ThreadPool: 로컬 함수/람다 지원, GIL 제약 있지만 I/O 대기 시 효율적
        executor_class = ThreadPoolExecutor if self.use_threads else ProcessPoolExecutor
        
        with executor_class(max_workers=self.num_workers) as executor:
            if ordered:
                # 순서 유지: map 사용
                futures_map = {
                    executor.submit(processor_func, item): i
                    for i, item in enumerate(items)
                }
                ordered_results: dict[int, TaskResult[R]] = {}
                
                for future in as_completed(futures_map):
                    idx = futures_map[future]
                    task_start = time.time()
                    
                    try:
                        result = future.result(timeout=timeout)
                        task_result = TaskResult(
                            success=True,
                            result=result,
                            task_id=f"cpu_task_{idx}",
                            elapsed_ms=(time.time() - task_start) * 1000,
                        )
                        succeeded += 1
                    except Exception as e:
                        task_result = TaskResult(
                            success=False,
                            error=str(e),
                            task_id=f"cpu_task_{idx}",
                            elapsed_ms=(time.time() - task_start) * 1000,
                        )
                        failed += 1
                        logger.warning(f"CPU task {idx} failed: {e}")
                    
                    ordered_results[idx] = task_result
                
                # 순서대로 결과 정렬
                results = [ordered_results[i] for i in range(len(items))]
            else:
                # 순서 무관: as_completed 사용
                futures = [
                    executor.submit(processor_func, item)
                    for item in items
                ]
                
                for i, future in enumerate(as_completed(futures)):
                    task_start = time.time()
                    
                    try:
                        result = future.result(timeout=timeout)
                        task_result = TaskResult(
                            success=True,
                            result=result,
                            task_id=f"cpu_task_{i}",
                            elapsed_ms=(time.time() - task_start) * 1000,
                        )
                        succeeded += 1
                    except Exception as e:
                        task_result = TaskResult(
                            success=False,
                            error=str(e),
                            task_id=f"cpu_task_{i}",
                            elapsed_ms=(time.time() - task_start) * 1000,
                        )
                        failed += 1
                        logger.warning(f"CPU task {i} failed: {e}")
                    
                    results.append(task_result)
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        # 통계 업데이트
        self._stats.cpu_tasks += len(items)
        self._stats.total_items += len(items)
        self._stats.processed_items += succeeded
        self._stats.failed_items += failed
        
        logger.info(
            f"CPU batch completed: {succeeded}/{len(items)} succeeded in {elapsed_ms:.1f}ms"
        )
        
        return BatchResult(
            total=len(items),
            succeeded=succeeded,
            failed=failed,
            results=results,
            elapsed_ms=elapsed_ms,
        )
    
    def process_batch_cpu_chunked(
        self,
        items: list[T],
        processor_func: Callable[[T], R],
        chunk_size: int = 100,
        timeout: float | None = None,
    ) -> BatchResult[R]:
        """
        대용량 데이터를 청크 단위로 CPU 처리
        
        메모리 효율을 위해 대량 데이터를 분할 처리
        
        Args:
            items: 처리할 아이템 리스트
            processor_func: 처리 함수
            chunk_size: 청크 크기
            timeout: 작업 타임아웃
            
        Returns:
            BatchResult[R]: 통합된 배치 결과
        """
        if not items:
            return BatchResult(total=0, succeeded=0, failed=0)
        
        start_time = time.time()
        all_results: list[TaskResult[R]] = []
        total_succeeded = 0
        total_failed = 0
        
        num_chunks = (len(items) + chunk_size - 1) // chunk_size
        logger.info(f"Processing {len(items)} items in {num_chunks} chunks")
        
        for i in range(0, len(items), chunk_size):
            chunk = items[i:i + chunk_size]
            chunk_result = self.process_batch_cpu(
                items=chunk,
                processor_func=processor_func,
                timeout=timeout,
                ordered=True,
            )
            
            all_results.extend(chunk_result.results)
            total_succeeded += chunk_result.succeeded
            total_failed += chunk_result.failed
            
            logger.debug(
                f"Chunk {i // chunk_size + 1}/{num_chunks}: "
                f"{chunk_result.succeeded}/{len(chunk)} succeeded"
            )
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        return BatchResult(
            total=len(items),
            succeeded=total_succeeded,
            failed=total_failed,
            results=all_results,
            elapsed_ms=elapsed_ms,
        )
    
    # =========================================================================
    # I/O BOUND PROCESSING
    # =========================================================================
    
    async def process_batch_io(
        self,
        items: list[T],
        async_func: Callable[[T], Any],
        max_concurrent: int | None = None,
        timeout: float | None = None,
    ) -> BatchResult[R]:
        """
        I/O 바운드 배치 처리 (asyncio.gather + Semaphore)
        
        네트워크 요청, 파일 I/O 등 비동기 작업에 적합
        
        Args:
            items: 처리할 아이템 리스트
            async_func: 비동기 처리 함수
            max_concurrent: 동시 실행 수 (None이면 기본값 사용)
            timeout: 개별 작업 타임아웃 (초)
            
        Returns:
            BatchResult[R]: 배치 처리 결과
        """
        if not items:
            return BatchResult(total=0, succeeded=0, failed=0)
        
        start_time = time.time()
        max_concurrent = max_concurrent or self.max_concurrent_io
        semaphore = asyncio.Semaphore(max_concurrent)
        
        logger.debug(f"Processing {len(items)} items with max_concurrent={max_concurrent}")
        
        async def bounded_call(item: T, idx: int) -> TaskResult[R]:
            """세마포어로 제한된 비동기 호출"""
            async with semaphore:
                task_start = time.time()
                try:
                    if timeout:
                        result = await asyncio.wait_for(
                            async_func(item),
                            timeout=timeout,
                        )
                    else:
                        result = await async_func(item)
                    
                    return TaskResult(
                        success=True,
                        result=result,
                        task_id=f"io_task_{idx}",
                        elapsed_ms=(time.time() - task_start) * 1000,
                    )
                except asyncio.TimeoutError:
                    return TaskResult(
                        success=False,
                        error="Timeout",
                        task_id=f"io_task_{idx}",
                        elapsed_ms=(time.time() - task_start) * 1000,
                    )
                except Exception as e:
                    logger.warning(f"I/O task {idx} failed: {e}")
                    return TaskResult(
                        success=False,
                        error=str(e),
                        task_id=f"io_task_{idx}",
                        elapsed_ms=(time.time() - task_start) * 1000,
                    )
        
        # 모든 작업 병렬 실행
        tasks = [bounded_call(item, i) for i, item in enumerate(items)]
        results = await asyncio.gather(*tasks)
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        succeeded = sum(1 for r in results if r.success)
        failed = len(results) - succeeded
        
        # 통계 업데이트
        self._stats.io_tasks += len(items)
        self._stats.total_items += len(items)
        self._stats.processed_items += succeeded
        self._stats.failed_items += failed
        
        logger.info(
            f"I/O batch completed: {succeeded}/{len(items)} succeeded in {elapsed_ms:.1f}ms"
        )
        
        return BatchResult(
            total=len(items),
            succeeded=succeeded,
            failed=failed,
            results=list(results),
            elapsed_ms=elapsed_ms,
        )
    
    async def process_batch_io_streaming(
        self,
        items: list[T],
        async_func: Callable[[T], Any],
        max_concurrent: int | None = None,
        on_result: Callable[[TaskResult], None] | None = None,
    ) -> BatchResult[R]:
        """
        스트리밍 방식 I/O 배치 처리
        
        결과가 나오는 대로 콜백 호출 (실시간 진행률 표시에 유용)
        
        Args:
            items: 처리할 아이템 리스트
            async_func: 비동기 처리 함수
            max_concurrent: 동시 실행 수
            on_result: 결과 도착 시 호출할 콜백
            
        Returns:
            BatchResult[R]: 배치 처리 결과
        """
        if not items:
            return BatchResult(total=0, succeeded=0, failed=0)
        
        start_time = time.time()
        max_concurrent = max_concurrent or self.max_concurrent_io
        semaphore = asyncio.Semaphore(max_concurrent)
        results: list[TaskResult[R]] = []
        succeeded = 0
        failed = 0
        
        async def bounded_call(item: T, idx: int) -> TaskResult[R]:
            async with semaphore:
                task_start = time.time()
                try:
                    result = await async_func(item)
                    task_result = TaskResult(
                        success=True,
                        result=result,
                        task_id=f"io_stream_{idx}",
                        elapsed_ms=(time.time() - task_start) * 1000,
                    )
                except Exception as e:
                    task_result = TaskResult(
                        success=False,
                        error=str(e),
                        task_id=f"io_stream_{idx}",
                        elapsed_ms=(time.time() - task_start) * 1000,
                    )
                
                # 콜백 호출
                if on_result:
                    on_result(task_result)
                
                return task_result
        
        # 작업 생성
        tasks = [asyncio.create_task(bounded_call(item, i)) for i, item in enumerate(items)]
        
        # 완료되는 대로 수집
        for coro in asyncio.as_completed(tasks):
            result = await coro
            results.append(result)
            if result.success:
                succeeded += 1
            else:
                failed += 1
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        return BatchResult(
            total=len(items),
            succeeded=succeeded,
            failed=failed,
            results=results,
            elapsed_ms=elapsed_ms,
        )
    
    # =========================================================================
    # MIXED PROCESSING (CPU + I/O)
    # =========================================================================
    
    async def process_mixed_pipeline(
        self,
        items: list[T],
        cpu_func: Callable[[T], Any],
        io_func: Callable[[Any], Any],
        cpu_first: bool = True,
    ) -> BatchResult:
        """
        CPU + I/O 혼합 파이프라인
        
        예: 텍스트 토큰화 (CPU) → API 호출 (I/O)
        또는: 데이터 수집 (I/O) → 전처리 (CPU)
        
        Args:
            items: 처리할 아이템
            cpu_func: CPU 바운드 함수
            io_func: I/O 바운드 함수
            cpu_first: CPU 먼저 실행 여부
            
        Returns:
            BatchResult: 최종 결과
        """
        start_time = time.time()
        
        if cpu_first:
            # Phase 1: CPU 처리
            logger.info("Phase 1: CPU processing")
            cpu_result = self.process_batch_cpu(items, cpu_func, ordered=True)
            
            if cpu_result.failed > 0:
                logger.warning(f"CPU phase: {cpu_result.failed} failures")
            
            # Phase 2: I/O 처리 (성공한 결과만)
            logger.info("Phase 2: I/O processing")
            io_items = cpu_result.successful_results
            
            if not io_items:
                return cpu_result
            
            io_result = await self.process_batch_io(io_items, io_func)
            
        else:
            # Phase 1: I/O 처리
            logger.info("Phase 1: I/O processing")
            io_result = await self.process_batch_io(items, io_func)
            
            if io_result.failed > 0:
                logger.warning(f"I/O phase: {io_result.failed} failures")
            
            # Phase 2: CPU 처리 (성공한 결과만)
            logger.info("Phase 2: CPU processing")
            cpu_items = io_result.successful_results
            
            if not cpu_items:
                return io_result
            
            cpu_result = self.process_batch_cpu(cpu_items, cpu_func, ordered=True)
            io_result = cpu_result  # 최종 결과
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        # 최종 결과의 elapsed_ms 업데이트
        final_result = io_result if cpu_first else cpu_result
        final_result.elapsed_ms = elapsed_ms
        
        return final_result
    
    # =========================================================================
    # STATISTICS
    # =========================================================================
    
    def get_stats(self) -> PipelineStats:
        """파이프라인 통계 반환"""
        # 평균 작업 시간 계산
        total_tasks = self._stats.cpu_tasks + self._stats.io_tasks
        if total_tasks > 0 and self._stats.elapsed_seconds > 0:
            self._stats.avg_task_ms = (
                self._stats.elapsed_seconds * 1000 / total_tasks
            )
        
        return self._stats.model_copy()
    
    def reset_stats(self) -> None:
        """통계 초기화"""
        self._stats = PipelineStats(pipeline_name=self.name)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def run_cpu_parallel(
    items: list[T],
    func: Callable[[T], R],
    num_workers: int = 8,
) -> list[R]:
    """
    간단한 CPU 병렬 처리 헬퍼
    
    Args:
        items: 처리할 아이템
        func: 처리 함수
        num_workers: 워커 수
        
    Returns:
        list[R]: 처리 결과 (성공한 것만)
    """
    pipeline = DataPipeline(num_workers=num_workers)
    result = pipeline.process_batch_cpu(items, func)
    return result.successful_results


async def run_io_parallel(
    items: list[T],
    func: Callable[[T], Any],
    max_concurrent: int = 10,
) -> list[Any]:
    """
    간단한 I/O 병렬 처리 헬퍼
    
    Args:
        items: 처리할 아이템
        func: 비동기 처리 함수
        max_concurrent: 동시 실행 수
        
    Returns:
        list: 처리 결과 (성공한 것만)
    """
    pipeline = DataPipeline(max_concurrent_io=max_concurrent)
    result = await pipeline.process_batch_io(items, func)
    return result.successful_results


# =============================================================================
# RAY MIGRATION STUB
# =============================================================================


class RayPipelineStub:
    """
    Ray 마이그레이션을 위한 스텁 클래스
    
    Phase 3에서 데이터가 100배 증가할 때 사용
    현재는 DataPipeline으로 충분
    
    마이그레이션 경로:
    1. DataPipeline → RayPipeline 교체
    2. process_batch_cpu → ray.remote 데코레이터
    3. process_batch_io → ray.wait + ray.get
    
    Usage (Phase 3):
        # pip install ray
        # from trendops.pipeline.ray_pipeline import RayPipeline
        # pipeline = RayPipeline(num_workers=16)
    """
    
    def __init__(self, num_workers: int = 16):
        raise NotImplementedError(
            "Ray pipeline is not yet implemented. "
            "Use DataPipeline for Phase 1-2. "
            "Implement RayPipeline in Phase 3 when data scales 100x."
        )


# =============================================================================
# CLI TEST
# =============================================================================


if __name__ == "__main__":
    import random
    
    print("=" * 70)
    print("  DataPipeline Test")
    print("=" * 70)
    
    # 테스트 함수들
    def cpu_task(x: int) -> int:
        """CPU 바운드 테스트 작업"""
        # 간단한 계산
        result = sum(i * i for i in range(x * 1000))
        return result
    
    async def io_task(x: int) -> dict:
        """I/O 바운드 테스트 작업"""
        # 네트워크 지연 시뮬레이션
        await asyncio.sleep(random.uniform(0.01, 0.05))
        return {"input": x, "result": x * 2}
    
    async def main():
        pipeline = DataPipeline(num_workers=4, max_concurrent_io=10, name="TestPipeline")
        
        # 1. CPU 배치 테스트
        print("\n[1] CPU Batch Processing...")
        items = list(range(1, 21))  # 1-20
        cpu_result = pipeline.process_batch_cpu(items, cpu_task)
        print(f"    ✓ Total: {cpu_result.total}")
        print(f"    ✓ Succeeded: {cpu_result.succeeded}")
        print(f"    ✓ Failed: {cpu_result.failed}")
        print(f"    ✓ Elapsed: {cpu_result.elapsed_ms:.1f}ms")
        print(f"    ✓ Success rate: {cpu_result.success_rate:.1%}")
        
        # 2. I/O 배치 테스트
        print("\n[2] I/O Batch Processing...")
        io_result = await pipeline.process_batch_io(items, io_task)
        print(f"    ✓ Total: {io_result.total}")
        print(f"    ✓ Succeeded: {io_result.succeeded}")
        print(f"    ✓ Failed: {io_result.failed}")
        print(f"    ✓ Elapsed: {io_result.elapsed_ms:.1f}ms")
        
        # 3. 스트리밍 테스트
        print("\n[3] Streaming I/O Processing...")
        completed = 0
        
        def on_result(result: TaskResult):
            nonlocal completed
            completed += 1
            print(f"    Progress: {completed}/{len(items)}", end="\r")
        
        stream_result = await pipeline.process_batch_io_streaming(
            items[:10],
            io_task,
            on_result=on_result,
        )
        print(f"\n    ✓ Streaming completed: {stream_result.succeeded}/{stream_result.total}")
        
        # 4. 통계 확인
        print("\n[4] Pipeline Stats...")
        stats = pipeline.get_stats()
        print(f"    ✓ CPU tasks: {stats.cpu_tasks}")
        print(f"    ✓ I/O tasks: {stats.io_tasks}")
        print(f"    ✓ Total items: {stats.total_items}")
        print(f"    ✓ Throughput: {stats.throughput:.1f} items/sec")
        
        print("\n" + "=" * 70)
        print("✅ DataPipeline tests completed!")
        print("=" * 70)
    
    asyncio.run(main())