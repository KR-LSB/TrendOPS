#!/usr/bin/env python3
"""
TrendOps Performance Benchmark Script

Week 6 Day 6: 성능 벤치마크

각 파이프라인 단계의 성능을 측정하고 리포트를 생성합니다.
- Trigger: 트렌드 감지 속도
- Collector: RSS 수집 처리량
- Embedding: 임베딩 생성 속도
- Deduplication: 중복 제거 효율
- LLM Analysis: 분석 레이턴시
- Image Generation: 이미지 생성 속도

Usage:
    python scripts/benchmark.py [--full] [--export json|csv]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, TypeVar

# Rich for beautiful output
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
    from rich.table import Table
    from rich.text import Text
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class BenchmarkResult:
    """단일 벤치마크 결과"""
    stage: str
    duration_ms: float
    items_processed: int
    success_rate: float
    throughput: float = 0.0  # items/sec
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    memory_mb: float = 0.0
    errors: list[str] = field(default_factory=list)
    
    def __post_init__(self):
        if self.duration_ms > 0 and self.items_processed > 0:
            self.throughput = (self.items_processed / self.duration_ms) * 1000


@dataclass
class BenchmarkSuite:
    """전체 벤치마크 결과"""
    name: str
    timestamp: datetime
    results: list[BenchmarkResult]
    total_duration_ms: float = 0.0
    hardware_info: dict = field(default_factory=dict)
    
    def __post_init__(self):
        self.total_duration_ms = sum(r.duration_ms for r in self.results)
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "timestamp": self.timestamp.isoformat(),
            "total_duration_ms": self.total_duration_ms,
            "hardware_info": self.hardware_info,
            "results": [asdict(r) for r in self.results],
        }


# =============================================================================
# Benchmark Utilities
# =============================================================================

T = TypeVar("T")

async def measure_async(
    func: Callable[[], T],
    iterations: int = 5,
    warmup: int = 1,
) -> tuple[T, list[float]]:
    """비동기 함수 실행 시간 측정"""
    # Warmup
    for _ in range(warmup):
        try:
            if asyncio.iscoroutinefunction(func):
                await func()
            else:
                func()
        except Exception:
            pass
    
    # Actual measurements
    durations = []
    result = None
    
    for _ in range(iterations):
        start = time.perf_counter()
        try:
            if asyncio.iscoroutinefunction(func):
                result = await func()
            else:
                result = func()
        except Exception:
            pass
        end = time.perf_counter()
        durations.append((end - start) * 1000)  # ms
    
    return result, durations


def calculate_percentiles(durations: list[float]) -> tuple[float, float, float]:
    """p50, p95, p99 계산"""
    if not durations:
        return 0.0, 0.0, 0.0
    
    sorted_d = sorted(durations)
    n = len(sorted_d)
    
    p50 = sorted_d[int(n * 0.5)] if n > 0 else 0
    p95 = sorted_d[int(n * 0.95)] if n >= 20 else sorted_d[-1]
    p99 = sorted_d[int(n * 0.99)] if n >= 100 else sorted_d[-1]
    
    return p50, p95, p99


def get_memory_usage() -> float:
    """현재 메모리 사용량 (MB)"""
    try:
        import resource
        usage = resource.getrusage(resource.RUSAGE_SELF)
        return usage.ru_maxrss / 1024  # KB to MB
    except ImportError:
        return 0.0


def get_hardware_info() -> dict:
    """하드웨어 정보 수집"""
    import platform
    import os
    
    info = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cpu_count": os.cpu_count(),
    }
    
    # GPU 정보 (선택적)
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            info["gpu"] = result.stdout.strip()
    except Exception:
        info["gpu"] = "N/A"
    
    return info


# =============================================================================
# Individual Benchmarks
# =============================================================================

async def benchmark_trigger(iterations: int = 5) -> BenchmarkResult:
    """Trigger 단계 벤치마크"""
    durations = []
    items_processed = 0
    errors = []
    
    # Mock trigger function (실제 환경에서는 실제 모듈 import)
    async def mock_trigger():
        await asyncio.sleep(0.1)  # Simulate API call
        return [
            {"keyword": f"keyword_{i}", "score": 7.0 + i * 0.5}
            for i in range(10)
        ]
    
    for _ in range(iterations):
        start = time.perf_counter()
        try:
            keywords = await mock_trigger()
            items_processed = len(keywords)
        except Exception as e:
            errors.append(str(e))
        end = time.perf_counter()
        durations.append((end - start) * 1000)
    
    p50, p95, p99 = calculate_percentiles(durations)
    avg_duration = statistics.mean(durations) if durations else 0
    
    return BenchmarkResult(
        stage="Trigger",
        duration_ms=avg_duration,
        items_processed=items_processed,
        success_rate=100.0 * (1 - len(errors) / max(iterations, 1)),
        p50_ms=p50,
        p95_ms=p95,
        p99_ms=p99,
        memory_mb=get_memory_usage(),
        errors=errors,
    )


async def benchmark_collector(iterations: int = 5) -> BenchmarkResult:
    """Collector 단계 벤치마크"""
    durations = []
    items_processed = 0
    errors = []
    
    # Mock collector function
    async def mock_collect():
        await asyncio.sleep(0.2)  # Simulate RSS fetch
        return [
            {"title": f"Article {i}", "link": f"https://example.com/{i}"}
            for i in range(20)
        ]
    
    for _ in range(iterations):
        start = time.perf_counter()
        try:
            articles = await mock_collect()
            items_processed = len(articles)
        except Exception as e:
            errors.append(str(e))
        end = time.perf_counter()
        durations.append((end - start) * 1000)
    
    p50, p95, p99 = calculate_percentiles(durations)
    avg_duration = statistics.mean(durations) if durations else 0
    
    return BenchmarkResult(
        stage="Collector",
        duration_ms=avg_duration,
        items_processed=items_processed,
        success_rate=100.0 * (1 - len(errors) / max(iterations, 1)),
        p50_ms=p50,
        p95_ms=p95,
        p99_ms=p99,
        memory_mb=get_memory_usage(),
        errors=errors,
    )


async def benchmark_embedding(iterations: int = 5) -> BenchmarkResult:
    """Embedding 단계 벤치마크"""
    durations = []
    items_processed = 64  # batch size
    errors = []
    
    # Mock embedding function (CPU-based simulation)
    def mock_embed():
        import random
        time.sleep(0.05)  # Simulate CPU embedding
        return [[random.random() for _ in range(1024)] for _ in range(items_processed)]
    
    for _ in range(iterations):
        start = time.perf_counter()
        try:
            embeddings = mock_embed()
            items_processed = len(embeddings)
        except Exception as e:
            errors.append(str(e))
        end = time.perf_counter()
        durations.append((end - start) * 1000)
    
    p50, p95, p99 = calculate_percentiles(durations)
    avg_duration = statistics.mean(durations) if durations else 0
    
    return BenchmarkResult(
        stage="Embedding",
        duration_ms=avg_duration,
        items_processed=items_processed,
        success_rate=100.0 * (1 - len(errors) / max(iterations, 1)),
        p50_ms=p50,
        p95_ms=p95,
        p99_ms=p99,
        memory_mb=get_memory_usage(),
        errors=errors,
    )


async def benchmark_deduplication(iterations: int = 5) -> BenchmarkResult:
    """Deduplication 단계 벤치마크"""
    durations = []
    items_processed = 100
    unique_items = 40  # 60% dedup rate
    errors = []
    
    # Mock deduplication
    def mock_dedup():
        time.sleep(0.03)  # Simulate hash + semantic dedup
        return unique_items
    
    for _ in range(iterations):
        start = time.perf_counter()
        try:
            unique = mock_dedup()
        except Exception as e:
            errors.append(str(e))
        end = time.perf_counter()
        durations.append((end - start) * 1000)
    
    p50, p95, p99 = calculate_percentiles(durations)
    avg_duration = statistics.mean(durations) if durations else 0
    dedup_rate = 100.0 * (1 - unique_items / items_processed)
    
    return BenchmarkResult(
        stage="Deduplication",
        duration_ms=avg_duration,
        items_processed=items_processed,
        success_rate=dedup_rate,  # Using success_rate for dedup efficiency
        p50_ms=p50,
        p95_ms=p95,
        p99_ms=p99,
        memory_mb=get_memory_usage(),
        errors=errors,
    )


async def benchmark_hybrid_search(iterations: int = 5) -> BenchmarkResult:
    """Hybrid Search 단계 벤치마크"""
    durations = []
    items_processed = 10  # top-k results
    errors = []
    
    # Mock hybrid search (BM25 + Vector)
    def mock_search():
        time.sleep(0.02)  # Simulate search
        return [
            {"doc_id": i, "score": 0.9 - i * 0.05}
            for i in range(items_processed)
        ]
    
    for _ in range(iterations):
        start = time.perf_counter()
        try:
            results = mock_search()
            items_processed = len(results)
        except Exception as e:
            errors.append(str(e))
        end = time.perf_counter()
        durations.append((end - start) * 1000)
    
    p50, p95, p99 = calculate_percentiles(durations)
    avg_duration = statistics.mean(durations) if durations else 0
    
    return BenchmarkResult(
        stage="HybridSearch",
        duration_ms=avg_duration,
        items_processed=items_processed,
        success_rate=100.0 * (1 - len(errors) / max(iterations, 1)),
        p50_ms=p50,
        p95_ms=p95,
        p99_ms=p99,
        memory_mb=get_memory_usage(),
        errors=errors,
    )


async def benchmark_llm_analysis(iterations: int = 3) -> BenchmarkResult:
    """LLM Analysis 단계 벤치마크"""
    durations = []
    items_processed = 1  # analysis per request
    errors = []
    
    # Mock LLM call (simulates vLLM/Ollama)
    async def mock_llm():
        await asyncio.sleep(0.5)  # Simulate LLM inference
        return {
            "summary": "This is a summary...",
            "sentiment": {"positive": 0.4, "negative": 0.3, "neutral": 0.3},
            "key_points": ["Point 1", "Point 2", "Point 3"],
        }
    
    for _ in range(iterations):
        start = time.perf_counter()
        try:
            result = await mock_llm()
        except Exception as e:
            errors.append(str(e))
        end = time.perf_counter()
        durations.append((end - start) * 1000)
    
    p50, p95, p99 = calculate_percentiles(durations)
    avg_duration = statistics.mean(durations) if durations else 0
    
    return BenchmarkResult(
        stage="LLM Analysis",
        duration_ms=avg_duration,
        items_processed=items_processed,
        success_rate=100.0 * (1 - len(errors) / max(iterations, 1)),
        p50_ms=p50,
        p95_ms=p95,
        p99_ms=p99,
        memory_mb=get_memory_usage(),
        errors=errors,
    )


async def benchmark_guardrail(iterations: int = 5) -> BenchmarkResult:
    """Guardrail 단계 벤치마크"""
    durations = []
    items_processed = 1
    pass_count = 0
    errors = []
    
    # Mock guardrail check
    def mock_guardrail():
        time.sleep(0.01)  # Simulate rule checks
        return {"action": "pass", "issues": []}
    
    for _ in range(iterations):
        start = time.perf_counter()
        try:
            result = mock_guardrail()
            if result["action"] == "pass":
                pass_count += 1
        except Exception as e:
            errors.append(str(e))
        end = time.perf_counter()
        durations.append((end - start) * 1000)
    
    p50, p95, p99 = calculate_percentiles(durations)
    avg_duration = statistics.mean(durations) if durations else 0
    
    return BenchmarkResult(
        stage="Guardrail",
        duration_ms=avg_duration,
        items_processed=items_processed,
        success_rate=100.0 * pass_count / max(iterations, 1),
        p50_ms=p50,
        p95_ms=p95,
        p99_ms=p99,
        memory_mb=get_memory_usage(),
        errors=errors,
    )


async def benchmark_image_generation(iterations: int = 3) -> BenchmarkResult:
    """Image Generation 단계 벤치마크"""
    durations = []
    items_processed = 1
    errors = []
    
    # Mock image generation
    def mock_image_gen():
        time.sleep(0.15)  # Simulate Pillow image creation
        return {"path": "/tmp/card_news.png", "size": (1080, 1080)}
    
    for _ in range(iterations):
        start = time.perf_counter()
        try:
            result = mock_image_gen()
        except Exception as e:
            errors.append(str(e))
        end = time.perf_counter()
        durations.append((end - start) * 1000)
    
    p50, p95, p99 = calculate_percentiles(durations)
    avg_duration = statistics.mean(durations) if durations else 0
    
    return BenchmarkResult(
        stage="ImageGen",
        duration_ms=avg_duration,
        items_processed=items_processed,
        success_rate=100.0 * (1 - len(errors) / max(iterations, 1)),
        p50_ms=p50,
        p95_ms=p95,
        p99_ms=p99,
        memory_mb=get_memory_usage(),
        errors=errors,
    )


# =============================================================================
# Main Benchmark Runner
# =============================================================================

async def run_full_benchmark(
    full: bool = False,
    export_format: str | None = None,
) -> BenchmarkSuite:
    """전체 벤치마크 실행"""
    
    if RICH_AVAILABLE:
        console = Console()
        console.print(Panel.fit(
            "[bold blue]TrendOps Performance Benchmark[/]\n"
            "파이프라인 각 단계의 성능을 측정합니다.",
            title="🚀 Benchmark Suite",
        ))
    else:
        print("=" * 60)
        print("TrendOps Performance Benchmark")
        print("=" * 60)
    
    results: list[BenchmarkResult] = []
    iterations = 10 if full else 5
    
    # Benchmark stages
    stages = [
        ("Trigger", benchmark_trigger),
        ("Collector", benchmark_collector),
        ("Embedding", benchmark_embedding),
        ("Deduplication", benchmark_deduplication),
        ("HybridSearch", benchmark_hybrid_search),
        ("LLM Analysis", benchmark_llm_analysis),
        ("Guardrail", benchmark_guardrail),
        ("ImageGen", benchmark_image_generation),
    ]
    
    if RICH_AVAILABLE:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console,
        ) as progress:
            task = progress.add_task("Running benchmarks...", total=len(stages))
            
            for name, func in stages:
                progress.update(task, description=f"Benchmarking {name}...")
                result = await func(iterations)
                results.append(result)
                progress.advance(task)
    else:
        for name, func in stages:
            print(f"Benchmarking {name}...")
            result = await func(iterations)
            results.append(result)
    
    # Create suite
    suite = BenchmarkSuite(
        name="TrendOps Full Benchmark",
        timestamp=datetime.now(),
        results=results,
        hardware_info=get_hardware_info(),
    )
    
    # Display results
    if RICH_AVAILABLE:
        display_results_rich(console, suite)
    else:
        display_results_plain(suite)
    
    # Export if requested
    if export_format:
        export_results(suite, export_format)
    
    return suite


def display_results_rich(console: Console, suite: BenchmarkSuite):
    """Rich를 사용한 결과 출력"""
    console.print("\n")
    
    # Results table
    table = Table(title="📊 Benchmark Results", show_lines=True)
    table.add_column("Stage", style="cyan", width=15)
    table.add_column("Avg (ms)", justify="right", style="green")
    table.add_column("P50 (ms)", justify="right")
    table.add_column("P95 (ms)", justify="right")
    table.add_column("Throughput", justify="right", style="yellow")
    table.add_column("Success", justify="right")
    
    for r in suite.results:
        success_style = "green" if r.success_rate >= 95 else "yellow" if r.success_rate >= 80 else "red"
        table.add_row(
            r.stage,
            f"{r.duration_ms:.1f}",
            f"{r.p50_ms:.1f}",
            f"{r.p95_ms:.1f}",
            f"{r.throughput:.1f}/s" if r.throughput > 0 else "N/A",
            Text(f"{r.success_rate:.1f}%", style=success_style),
        )
    
    console.print(table)
    
    # Summary
    console.print(Panel(
        f"[bold]Total Duration:[/] {suite.total_duration_ms:.1f} ms\n"
        f"[bold]Platform:[/] {suite.hardware_info.get('platform', 'N/A')}\n"
        f"[bold]Python:[/] {suite.hardware_info.get('python_version', 'N/A')}\n"
        f"[bold]CPU Cores:[/] {suite.hardware_info.get('cpu_count', 'N/A')}\n"
        f"[bold]GPU:[/] {suite.hardware_info.get('gpu', 'N/A')}",
        title="📋 Summary",
        border_style="blue",
    ))


def display_results_plain(suite: BenchmarkSuite):
    """플레인 텍스트 결과 출력"""
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)
    print(f"{'Stage':<15} {'Avg(ms)':<10} {'P50(ms)':<10} {'P95(ms)':<10} {'Success':<10}")
    print("-" * 80)
    
    for r in suite.results:
        print(f"{r.stage:<15} {r.duration_ms:<10.1f} {r.p50_ms:<10.1f} {r.p95_ms:<10.1f} {r.success_rate:<10.1f}%")
    
    print("-" * 80)
    print(f"Total Duration: {suite.total_duration_ms:.1f} ms")
    print(f"Platform: {suite.hardware_info.get('platform', 'N/A')}")


def export_results(suite: BenchmarkSuite, format: str):
    """결과 내보내기"""
    output_dir = Path("./benchmark_results")
    output_dir.mkdir(exist_ok=True)
    
    timestamp = suite.timestamp.strftime("%Y%m%d_%H%M%S")
    
    if format == "json":
        output_path = output_dir / f"benchmark_{timestamp}.json"
        with open(output_path, "w") as f:
            json.dump(suite.to_dict(), f, indent=2)
        print(f"\n📁 Results exported to: {output_path}")
    
    elif format == "csv":
        output_path = output_dir / f"benchmark_{timestamp}.csv"
        with open(output_path, "w") as f:
            f.write("stage,duration_ms,p50_ms,p95_ms,p99_ms,throughput,success_rate,memory_mb\n")
            for r in suite.results:
                f.write(f"{r.stage},{r.duration_ms:.2f},{r.p50_ms:.2f},{r.p95_ms:.2f},"
                       f"{r.p99_ms:.2f},{r.throughput:.2f},{r.success_rate:.2f},{r.memory_mb:.2f}\n")
        print(f"\n📁 Results exported to: {output_path}")


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    """CLI 진입점"""
    parser = argparse.ArgumentParser(
        description="TrendOps Performance Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/benchmark.py              # 기본 벤치마크 (5회 반복)
    python scripts/benchmark.py --full       # 전체 벤치마크 (10회 반복)
    python scripts/benchmark.py --export json  # JSON 내보내기
    python scripts/benchmark.py --export csv   # CSV 내보내기
        """,
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="전체 벤치마크 실행 (더 많은 반복)",
    )
    parser.add_argument(
        "--export",
        choices=["json", "csv"],
        help="결과 내보내기 형식",
    )
    
    args = parser.parse_args()
    
    try:
        asyncio.run(run_full_benchmark(full=args.full, export_format=args.export))
    except KeyboardInterrupt:
        print("\n⚠️ Benchmark interrupted by user")
        sys.exit(1)


if __name__ == "__main__":
    main()