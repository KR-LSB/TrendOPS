# tests/test_week6_day6_benchmark.py
"""
TrendOps Week 6 Day 6: Benchmark & E2E Tests

벤치마크 스크립트 검증 및 E2E 파이프라인 테스트:
1. BenchmarkResult 데이터 클래스 검증
2. BenchmarkSuite 생성 및 직렬화 테스트
3. 개별 벤치마크 함수 테스트
4. 데모 스크립트 구조 검증
5. E2E 파이프라인 통합 테스트
"""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# =============================================================================
# Import benchmark module for testing
# =============================================================================

# Add scripts to path for import
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

try:
    from scripts.benchmark import (
        BenchmarkResult,
        BenchmarkSuite,
        benchmark_trigger,
        benchmark_collector,
        benchmark_embedding,
        benchmark_deduplication,
        benchmark_hybrid_search,
        benchmark_llm_analysis,
        benchmark_guardrail,
        benchmark_image_generation,
        calculate_percentiles,
        get_memory_usage,
        run_full_benchmark,
    )
    BENCHMARK_AVAILABLE = True
except ImportError:
    BENCHMARK_AVAILABLE = False

try:
    from scripts.demo import (
        DemoRunner,
        DemoKeyword,
        DemoArticle,
        DemoAnalysis,
        DemoContent,
        SAMPLE_KEYWORDS,
        SAMPLE_ARTICLES,
        SAMPLE_ANALYSIS,
    )
    DEMO_AVAILABLE = True
except ImportError:
    DEMO_AVAILABLE = False


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def sample_benchmark_result() -> BenchmarkResult:
    """샘플 BenchmarkResult"""
    if not BENCHMARK_AVAILABLE:
        pytest.skip("benchmark module not available")
    return BenchmarkResult(
        stage="Test",
        duration_ms=100.0,
        items_processed=50,
        success_rate=98.0,
        p50_ms=95.0,
        p95_ms=150.0,
        p99_ms=200.0,
        memory_mb=512.0,
        errors=[],
    )


@pytest.fixture
def sample_benchmark_suite(sample_benchmark_result: BenchmarkResult) -> BenchmarkSuite:
    """샘플 BenchmarkSuite"""
    if not BENCHMARK_AVAILABLE:
        pytest.skip("benchmark module not available")
    return BenchmarkSuite(
        name="Test Suite",
        timestamp=datetime(2025, 2, 15, 10, 30, 0),
        results=[sample_benchmark_result],
        hardware_info={"platform": "Linux", "cpu_count": 8},
    )


# =============================================================================
# Test 1: BenchmarkResult 데이터 클래스 검증
# =============================================================================

class TestBenchmarkResult:
    """BenchmarkResult 테스트"""
    
    @pytest.mark.skipif(not BENCHMARK_AVAILABLE, reason="benchmark module not available")
    def test_benchmark_result_creation(self):
        """BenchmarkResult 생성 테스트"""
        result = BenchmarkResult(
            stage="trigger",
            duration_ms=150.0,
            items_processed=30,
            success_rate=100.0,
        )
        
        assert result.stage == "trigger"
        assert result.duration_ms == 150.0
        assert result.items_processed == 30
        assert result.success_rate == 100.0
    
    @pytest.mark.skipif(not BENCHMARK_AVAILABLE, reason="benchmark module not available")
    def test_benchmark_result_throughput_calculation(self):
        """처리량 자동 계산 테스트"""
        result = BenchmarkResult(
            stage="collector",
            duration_ms=1000.0,  # 1초
            items_processed=100,
            success_rate=100.0,
        )
        
        # throughput = items / seconds = 100 / 1 = 100 items/sec
        assert result.throughput == pytest.approx(100.0, rel=0.01)
    
    @pytest.mark.skipif(not BENCHMARK_AVAILABLE, reason="benchmark module not available")
    def test_benchmark_result_with_errors(self):
        """에러 포함 결과 테스트"""
        result = BenchmarkResult(
            stage="llm",
            duration_ms=5000.0,
            items_processed=1,
            success_rate=80.0,
            errors=["Timeout", "Connection error"],
        )
        
        assert len(result.errors) == 2
        assert result.success_rate == 80.0


# =============================================================================
# Test 2: BenchmarkSuite 생성 및 직렬화 테스트
# =============================================================================

class TestBenchmarkSuite:
    """BenchmarkSuite 테스트"""
    
    @pytest.mark.skipif(not BENCHMARK_AVAILABLE, reason="benchmark module not available")
    def test_benchmark_suite_creation(self, sample_benchmark_result: BenchmarkResult):
        """BenchmarkSuite 생성 테스트"""
        suite = BenchmarkSuite(
            name="Test Suite",
            timestamp=datetime.now(),
            results=[sample_benchmark_result],
        )
        
        assert suite.name == "Test Suite"
        assert len(suite.results) == 1
        assert suite.total_duration_ms == sample_benchmark_result.duration_ms
    
    @pytest.mark.skipif(not BENCHMARK_AVAILABLE, reason="benchmark module not available")
    def test_benchmark_suite_total_duration(self):
        """전체 소요 시간 자동 계산 테스트"""
        results = [
            BenchmarkResult("stage1", 100.0, 10, 100.0),
            BenchmarkResult("stage2", 200.0, 20, 100.0),
            BenchmarkResult("stage3", 300.0, 30, 100.0),
        ]
        
        suite = BenchmarkSuite(
            name="Multi-stage",
            timestamp=datetime.now(),
            results=results,
        )
        
        assert suite.total_duration_ms == 600.0
    
    @pytest.mark.skipif(not BENCHMARK_AVAILABLE, reason="benchmark module not available")
    def test_benchmark_suite_to_dict(self, sample_benchmark_suite: BenchmarkSuite):
        """JSON 직렬화 테스트"""
        data = sample_benchmark_suite.to_dict()
        
        assert "name" in data
        assert "timestamp" in data
        assert "results" in data
        assert "total_duration_ms" in data
        assert "hardware_info" in data
        
        # JSON 직렬화 가능 여부 확인
        json_str = json.dumps(data)
        assert len(json_str) > 0
    
    @pytest.mark.skipif(not BENCHMARK_AVAILABLE, reason="benchmark module not available")
    def test_benchmark_suite_hardware_info(self, sample_benchmark_suite: BenchmarkSuite):
        """하드웨어 정보 테스트"""
        assert "platform" in sample_benchmark_suite.hardware_info
        assert sample_benchmark_suite.hardware_info["cpu_count"] == 8


# =============================================================================
# Test 3: 개별 벤치마크 함수 테스트
# =============================================================================

class TestIndividualBenchmarks:
    """개별 벤치마크 함수 테스트"""
    
    @pytest.mark.skipif(not BENCHMARK_AVAILABLE, reason="benchmark module not available")
    @pytest.mark.asyncio
    async def test_benchmark_trigger(self):
        """Trigger 벤치마크 테스트"""
        result = await benchmark_trigger(iterations=2)
        
        assert result.stage == "Trigger"
        assert result.duration_ms > 0
        assert result.items_processed > 0
        assert 0 <= result.success_rate <= 100
    
    @pytest.mark.skipif(not BENCHMARK_AVAILABLE, reason="benchmark module not available")
    @pytest.mark.asyncio
    async def test_benchmark_collector(self):
        """Collector 벤치마크 테스트"""
        result = await benchmark_collector(iterations=2)
        
        assert result.stage == "Collector"
        assert result.duration_ms > 0
        assert result.items_processed > 0
    
    @pytest.mark.skipif(not BENCHMARK_AVAILABLE, reason="benchmark module not available")
    @pytest.mark.asyncio
    async def test_benchmark_embedding(self):
        """Embedding 벤치마크 테스트"""
        result = await benchmark_embedding(iterations=2)
        
        assert result.stage == "Embedding"
        assert result.duration_ms > 0
        # Embedding은 배치 처리이므로 items_processed가 배치 크기여야 함
        assert result.items_processed >= 1
    
    @pytest.mark.skipif(not BENCHMARK_AVAILABLE, reason="benchmark module not available")
    @pytest.mark.asyncio
    async def test_benchmark_deduplication(self):
        """Deduplication 벤치마크 테스트"""
        result = await benchmark_deduplication(iterations=2)
        
        assert result.stage == "Deduplication"
        assert result.duration_ms > 0
        # success_rate를 dedup 효율로 사용
        assert 0 <= result.success_rate <= 100


# =============================================================================
# Test 4: 데모 스크립트 구조 검증
# =============================================================================

class TestDemoStructure:
    """데모 스크립트 구조 테스트"""
    
    @pytest.mark.skipif(not DEMO_AVAILABLE, reason="demo module not available")
    def test_demo_keyword_creation(self):
        """DemoKeyword 생성 테스트"""
        keyword = DemoKeyword(
            keyword="AI 규제",
            score=9.2,
            source="google",
        )
        
        assert keyword.keyword == "AI 규제"
        assert keyword.score == 9.2
        assert keyword.source == "google"
    
    @pytest.mark.skipif(not DEMO_AVAILABLE, reason="demo module not available")
    def test_demo_article_creation(self):
        """DemoArticle 생성 테스트"""
        article = DemoArticle(
            title="Test Article",
            source="연합뉴스",
            published="5분 전",
        )
        
        assert article.title == "Test Article"
        assert article.source == "연합뉴스"
    
    @pytest.mark.skipif(not DEMO_AVAILABLE, reason="demo module not available")
    def test_demo_analysis_creation(self):
        """DemoAnalysis 생성 테스트"""
        analysis = DemoAnalysis(
            keyword="테스트",
            summary="테스트 요약",
            sentiment={"positive": 0.5, "negative": 0.3, "neutral": 0.2},
            key_points=["Point 1", "Point 2"],
        )
        
        assert analysis.keyword == "테스트"
        assert len(analysis.key_points) == 2
        assert sum(analysis.sentiment.values()) == pytest.approx(1.0)
    
    @pytest.mark.skipif(not DEMO_AVAILABLE, reason="demo module not available")
    def test_sample_data_exists(self):
        """샘플 데이터 존재 확인"""
        assert len(SAMPLE_KEYWORDS) >= 5
        assert len(SAMPLE_ARTICLES) >= 3
        assert SAMPLE_ANALYSIS is not None
        assert SAMPLE_ANALYSIS.keyword == "AI 규제"


# =============================================================================
# Test 5: E2E 파이프라인 통합 테스트
# =============================================================================

class TestE2EPipeline:
    """E2E 파이프라인 통합 테스트"""
    
    @pytest.mark.skipif(not BENCHMARK_AVAILABLE, reason="benchmark module not available")
    def test_calculate_percentiles_empty(self):
        """빈 리스트 퍼센타일 계산"""
        p50, p95, p99 = calculate_percentiles([])
        
        assert p50 == 0.0
        assert p95 == 0.0
        assert p99 == 0.0
    
    @pytest.mark.skipif(not BENCHMARK_AVAILABLE, reason="benchmark module not available")
    def test_calculate_percentiles_single(self):
        """단일 값 퍼센타일 계산"""
        p50, p95, p99 = calculate_percentiles([100.0])
        
        assert p50 == 100.0
        assert p95 == 100.0
        assert p99 == 100.0
    
    @pytest.mark.skipif(not BENCHMARK_AVAILABLE, reason="benchmark module not available")
    def test_calculate_percentiles_multiple(self):
        """여러 값 퍼센타일 계산"""
        durations = list(range(1, 101))  # 1 to 100
        p50, p95, p99 = calculate_percentiles(durations)
        
        assert p50 == pytest.approx(50, rel=0.1)
        assert p95 == pytest.approx(95, rel=0.1)
        assert p99 == pytest.approx(99, rel=0.1)
    
    @pytest.mark.skipif(not BENCHMARK_AVAILABLE, reason="benchmark module not available")
    def test_get_memory_usage(self):
        """메모리 사용량 조회"""
        memory = get_memory_usage()
        
        # 0 이상이어야 함 (0은 지원 안 되는 플랫폼)
        assert memory >= 0
    
    @pytest.mark.skipif(not BENCHMARK_AVAILABLE, reason="benchmark module not available")
    @pytest.mark.asyncio
    async def test_full_benchmark_suite_runs(self):
        """전체 벤치마크 스위트 실행 테스트"""
        # 실제 실행 대신 구조만 검증
        suite = await run_full_benchmark(full=False, export_format=None)
        
        assert suite is not None
        assert suite.name == "TrendOps Full Benchmark"
        assert len(suite.results) == 8  # 8개 단계
        assert suite.total_duration_ms > 0
    
    @pytest.mark.skipif(not DEMO_AVAILABLE, reason="demo module not available")
    @pytest.mark.asyncio
    async def test_demo_runner_fast_mode(self):
        """데모 러너 빠른 모드 테스트"""
        runner = DemoRunner(fast=True, interactive=False)
        
        assert runner.fast is True
        assert runner.delay_factor == 0.2
        
        # 지연 시간 테스트
        import time
        start = time.time()
        await runner.delay(1.0)  # 1초 * 0.2 = 0.2초
        elapsed = time.time() - start
        
        assert elapsed < 0.5  # 0.5초 미만


# =============================================================================
# Utility Tests
# =============================================================================

class TestUtilities:
    """유틸리티 함수 테스트"""
    
    @pytest.mark.skipif(not BENCHMARK_AVAILABLE, reason="benchmark module not available")
    def test_benchmark_result_default_values(self):
        """BenchmarkResult 기본값 테스트"""
        result = BenchmarkResult(
            stage="test",
            duration_ms=0,
            items_processed=0,
            success_rate=0,
        )
        
        assert result.throughput == 0.0
        assert result.p50_ms == 0.0
        assert result.p95_ms == 0.0
        assert result.p99_ms == 0.0
        assert result.memory_mb == 0.0
        assert result.errors == []
    
    @pytest.mark.skipif(not BENCHMARK_AVAILABLE, reason="benchmark module not available")
    def test_benchmark_suite_empty_results(self):
        """빈 결과 스위트 테스트"""
        suite = BenchmarkSuite(
            name="Empty",
            timestamp=datetime.now(),
            results=[],
        )
        
        assert suite.total_duration_ms == 0.0
        assert len(suite.results) == 0


# =============================================================================
# Script Execution Tests
# =============================================================================

class TestScriptExecution:
    """스크립트 실행 테스트"""
    
    def test_benchmark_script_exists(self):
        """benchmark.py 스크립트 존재 확인"""
        script_path = Path(__file__).parent.parent / "scripts" / "benchmark.py"
        assert script_path.exists(), "benchmark.py must exist in scripts/"
    
    def test_demo_script_exists(self):
        """demo.py 스크립트 존재 확인"""
        script_path = Path(__file__).parent.parent / "scripts" / "demo.py"
        assert script_path.exists(), "demo.py must exist in scripts/"
    
    def test_benchmark_script_has_main(self):
        """benchmark.py에 main 함수 존재"""
        script_path = Path(__file__).parent.parent / "scripts" / "benchmark.py"
        content = script_path.read_text(encoding="utf-8")
        
        assert "def main():" in content
        assert 'if __name__ == "__main__":' in content
    
    def test_demo_script_has_main(self):
        """demo.py에 main 함수 존재"""
        script_path = Path(__file__).parent.parent / "scripts" / "demo.py"
        content = script_path.read_text(encoding="utf-8")
        
        assert "def main():" in content
        assert 'if __name__ == "__main__":' in content


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])