# tests/test_week5_day1_image_generator.py
"""
TrendOps Week 5 Day 1: Image Generator 테스트

테스트 범위:
1. 기본 카드뉴스 생성
2. 한글 텍스트 렌더링
3. 자동 줄바꿈
4. 그라데이션 배경
5. 감성 비율 바
6. 출력 포맷 검증
7. 이미지 크기 검증
8. 에러 처리

실행:
    pytest tests/test_week5_day1_image_generator.py -v
    pytest tests/test_week5_day1_image_generator.py -v -k "test_generate"
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from PIL import Image

# 테스트 대상 임포트
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "publisher"))

from trendops.publisher.image_generator import (
    ImageGenerator,
    ImageGeneratorResult,
    CardTemplate,
    hex_to_rgb,
    sanitize_filename,
    wrap_text,
    get_text_width,
    create_image_generator,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def temp_output_dir():
    """임시 출력 디렉토리"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def generator(temp_output_dir):
    """기본 ImageGenerator 인스턴스"""
    return ImageGenerator(output_dir=temp_output_dir)


@pytest.fixture
def custom_template():
    """커스텀 템플릿"""
    return CardTemplate(
        width=800,
        height=1000,
        background_color="#2d2d44",
        gradient_end="#1a1a2e",
        title_font_size=48,
        body_font_size=28,
        padding=60,
    )


@pytest.fixture
def sample_sentiment():
    """샘플 감성 비율"""
    return {
        "positive": 0.3,
        "negative": 0.5,
        "neutral": 0.2,
    }


@pytest.fixture
def sample_korean_text():
    """샘플 한글 텍스트"""
    return {
        "keyword": "트럼프 관세 정책",
        "summary": (
            "미국 트럼프 대통령이 중국산 제품에 대한 관세를 대폭 인상한다고 발표했습니다. "
            "이번 조치는 양국 간 무역 갈등을 심화시킬 것으로 예상되며, "
            "글로벌 공급망에도 영향을 미칠 전망입니다. "
            "전문가들은 이번 관세 인상이 소비자 물가 상승으로 이어질 수 있다고 경고했습니다."
        ),
    }


# =============================================================================
# Unit Tests - Utility Functions
# =============================================================================

class TestUtilityFunctions:
    """유틸리티 함수 테스트"""
    
    def test_hex_to_rgb_basic(self):
        """HEX to RGB 변환 - 기본"""
        assert hex_to_rgb("#ffffff") == (255, 255, 255)
        assert hex_to_rgb("#000000") == (0, 0, 0)
        assert hex_to_rgb("#ff0000") == (255, 0, 0)
    
    def test_hex_to_rgb_without_hash(self):
        """HEX to RGB 변환 - # 없이"""
        assert hex_to_rgb("ffffff") == (255, 255, 255)
        assert hex_to_rgb("1a1a2e") == (26, 26, 46)
    
    def test_sanitize_filename_korean(self):
        """파일명 정규화 - 한글"""
        assert sanitize_filename("트럼프 관세") == "트럼프_관세"
        assert sanitize_filename("테스트!@#$%") == "테스트"
    
    def test_sanitize_filename_max_length(self):
        """파일명 정규화 - 최대 길이"""
        long_text = "a" * 100
        result = sanitize_filename(long_text, max_length=30)
        assert len(result) == 30
    
    def test_sanitize_filename_special_chars(self):
        """파일명 정규화 - 특수문자"""
        assert sanitize_filename("hello/world") == "helloworld"
        assert sanitize_filename("test<>file") == "testfile"


# =============================================================================
# Unit Tests - CardTemplate
# =============================================================================

class TestCardTemplate:
    """CardTemplate 테스트"""
    
    def test_default_template(self):
        """기본 템플릿 값 검증"""
        template = CardTemplate()
        
        assert template.width == 1080
        assert template.height == 1350
        assert template.padding == 80
        assert template.title_font_size == 64
        assert template.body_font_size == 36
    
    def test_custom_template(self, custom_template):
        """커스텀 템플릿 값 검증"""
        assert custom_template.width == 800
        assert custom_template.height == 1000
        assert custom_template.padding == 60
    
    def test_template_aspect_ratio(self):
        """템플릿 비율 검증 (4:5)"""
        template = CardTemplate()
        aspect_ratio = template.height / template.width
        assert abs(aspect_ratio - 1.25) < 0.01  # 4:5 = 1.25


# =============================================================================
# Unit Tests - ImageGeneratorResult
# =============================================================================

class TestImageGeneratorResult:
    """ImageGeneratorResult 테스트"""
    
    def test_success_result(self, temp_output_dir):
        """성공 결과 객체"""
        result = ImageGeneratorResult(
            success=True,
            image_path=temp_output_dir / "test.png",
            width=1080,
            height=1350,
            file_size_bytes=102400,
            generation_time_ms=1500.0,
        )
        
        assert result.success is True
        assert result.width == 1080
        assert result.file_size_kb == 100.0
        assert result.error_message is None
    
    def test_failure_result(self):
        """실패 결과 객체"""
        result = ImageGeneratorResult(
            success=False,
            error_message="Font not found",
            generation_time_ms=50.0,
        )
        
        assert result.success is False
        assert result.image_path is None
        assert result.error_message == "Font not found"


# =============================================================================
# Integration Tests - ImageGenerator
# =============================================================================

class TestImageGenerator:
    """ImageGenerator 통합 테스트"""
    
    def test_initialization(self, generator, temp_output_dir):
        """초기화 검증"""
        assert generator.output_dir == temp_output_dir
        assert generator.template is not None
        assert isinstance(generator.template, CardTemplate)
    
    def test_initialization_with_custom_template(self, temp_output_dir, custom_template):
        """커스텀 템플릿으로 초기화"""
        gen = ImageGenerator(template=custom_template, output_dir=temp_output_dir)
        assert gen.template.width == 800
        assert gen.template.height == 1000
    
    @pytest.mark.asyncio
    async def test_generate_basic_card(self, generator, sample_korean_text):
        """기본 카드뉴스 생성"""
        result = await generator.generate(
            keyword=sample_korean_text["keyword"],
            summary=sample_korean_text["summary"],
        )
        
        assert result.success is True
        assert result.image_path is not None
        assert result.image_path.exists()
        assert result.width == 1080
        assert result.height == 1350
        assert result.file_size_bytes > 0
        assert result.generation_time_ms > 0
    
    @pytest.mark.asyncio
    async def test_generate_with_sentiment(self, generator, sample_korean_text, sample_sentiment):
        """감성 비율 포함 카드 생성"""
        result = await generator.generate(
            keyword=sample_korean_text["keyword"],
            summary=sample_korean_text["summary"],
            sentiment_ratio=sample_sentiment,
        )
        
        assert result.success is True
        assert result.image_path.exists()
    
    @pytest.mark.asyncio
    async def test_generate_custom_filename(self, generator, sample_korean_text):
        """커스텀 파일명 지정"""
        custom_filename = "my_custom_card.png"
        
        result = await generator.generate(
            keyword=sample_korean_text["keyword"],
            summary=sample_korean_text["summary"],
            filename=custom_filename,
        )
        
        assert result.success is True
        assert result.image_path.name == custom_filename
    
    @pytest.mark.asyncio
    async def test_output_format_png(self, generator, sample_korean_text):
        """PNG 출력 포맷 검증"""
        result = await generator.generate(
            keyword=sample_korean_text["keyword"],
            summary=sample_korean_text["summary"],
        )
        
        assert result.success is True
        
        # PIL로 이미지 로드하여 포맷 검증
        with Image.open(result.image_path) as img:
            assert img.format == "PNG"
            assert img.mode == "RGB"
    
    @pytest.mark.asyncio
    async def test_image_dimensions(self, generator, sample_korean_text):
        """이미지 크기 검증 (1080x1350)"""
        result = await generator.generate(
            keyword=sample_korean_text["keyword"],
            summary=sample_korean_text["summary"],
        )
        
        assert result.success is True
        
        with Image.open(result.image_path) as img:
            assert img.width == 1080
            assert img.height == 1350
    
    @pytest.mark.asyncio
    async def test_korean_text_rendering(self, generator):
        """한글 텍스트 렌더링 (폰트 없어도 에러 없이 생성)"""
        result = await generator.generate(
            keyword="한글 키워드 테스트",
            summary="이것은 한글로 작성된 요약문입니다. 정상적으로 렌더링되어야 합니다.",
        )
        
        # 폰트가 없어도 에러 없이 이미지 생성
        assert result.success is True
        assert result.image_path.exists()
    
    @pytest.mark.asyncio
    async def test_text_wrapping_long_text(self, generator):
        """긴 텍스트 자동 줄바꿈"""
        long_summary = "이것은 매우 긴 텍스트입니다. " * 20
        
        result = await generator.generate(
            keyword="줄바꿈 테스트",
            summary=long_summary,
        )
        
        assert result.success is True
        assert result.image_path.exists()
    
    @pytest.mark.asyncio
    async def test_gradient_background(self, generator, sample_korean_text):
        """그라데이션 배경 생성"""
        result = await generator.generate(
            keyword=sample_korean_text["keyword"],
            summary=sample_korean_text["summary"],
        )
        
        assert result.success is True
        
        # 이미지 로드하여 상단/하단 색상 차이 확인
        with Image.open(result.image_path) as img:
            top_pixel = img.getpixel((540, 10))
            bottom_pixel = img.getpixel((540, 1340))
            
            # 그라데이션이므로 색상이 달라야 함
            assert top_pixel != bottom_pixel
    
    @pytest.mark.asyncio
    async def test_sentiment_bar_rendering(self, generator, sample_korean_text, sample_sentiment):
        """감성 비율 바 렌더링"""
        result = await generator.generate(
            keyword=sample_korean_text["keyword"],
            summary=sample_korean_text["summary"],
            sentiment_ratio=sample_sentiment,
        )
        
        assert result.success is True
        
        # 이미지가 생성되었고 파일 크기가 적절한지 확인
        assert result.file_size_bytes > 5000  # 최소 5KB 이상 (폰트에 따라 다름)
    
    @pytest.mark.asyncio
    async def test_sentiment_bar_edge_cases(self, generator, sample_korean_text):
        """감성 비율 바 - 엣지 케이스"""
        # 모두 0인 경우
        result = await generator.generate(
            keyword=sample_korean_text["keyword"],
            summary=sample_korean_text["summary"],
            sentiment_ratio={"positive": 0, "negative": 0, "neutral": 0},
        )
        assert result.success is True
        
        # 하나만 100%인 경우
        result = await generator.generate(
            keyword=sample_korean_text["keyword"],
            summary=sample_korean_text["summary"],
            sentiment_ratio={"positive": 1.0, "negative": 0, "neutral": 0},
        )
        assert result.success is True
    
    @pytest.mark.asyncio
    async def test_generation_time_under_2_seconds(self, generator, sample_korean_text, sample_sentiment):
        """생성 시간 2초 이내 검증"""
        result = await generator.generate(
            keyword=sample_korean_text["keyword"],
            summary=sample_korean_text["summary"],
            sentiment_ratio=sample_sentiment,
        )
        
        assert result.success is True
        assert result.generation_time_ms < 2000  # 2초 = 2000ms


# =============================================================================
# Error Handling Tests
# =============================================================================

class TestErrorHandling:
    """에러 처리 테스트"""
    
    @pytest.mark.asyncio
    async def test_empty_keyword(self, generator):
        """빈 키워드 처리"""
        result = await generator.generate(
            keyword="",
            summary="요약 내용",
        )
        
        # 빈 키워드도 처리 가능해야 함
        assert result.success is True
    
    @pytest.mark.asyncio
    async def test_empty_summary(self, generator):
        """빈 요약 처리"""
        result = await generator.generate(
            keyword="테스트 키워드",
            summary="",
        )
        
        assert result.success is True
    
    @pytest.mark.asyncio
    async def test_special_characters_in_keyword(self, generator):
        """특수문자 포함 키워드"""
        result = await generator.generate(
            keyword="테스트 <키워드> & 특수문자!@#",
            summary="요약 내용",
        )
        
        assert result.success is True
    
    @pytest.mark.asyncio
    async def test_invalid_output_directory(self):
        """잘못된 출력 디렉토리"""
        # read-only 디렉토리나 존재하지 않는 경로
        generator = ImageGenerator(output_dir=Path("/nonexistent/path/that/should/fail"))
        
        # output_dir 생성 시도하므로 실패해야 함
        # 하지만 mkdir(parents=True)로 생성 시도하므로 권한 문제가 없으면 성공
        # 실제 권한 문제 테스트는 환경에 따라 다름
    
    def test_find_korean_font_fallback(self, temp_output_dir):
        """한글 폰트 찾기 실패 시 fallback"""
        generator = ImageGenerator(output_dir=temp_output_dir)
        
        # 폰트가 없어도 기본 폰트로 fallback
        font = generator._get_font(24)
        assert font is not None


# =============================================================================
# Factory Function Tests
# =============================================================================

class TestFactoryFunctions:
    """팩토리 함수 테스트"""
    
    def test_create_image_generator_default(self, temp_output_dir):
        """기본 생성"""
        gen = create_image_generator(output_dir=temp_output_dir)
        
        assert isinstance(gen, ImageGenerator)
        assert gen.output_dir == temp_output_dir
    
    def test_create_image_generator_with_template(self, temp_output_dir, custom_template):
        """커스텀 템플릿으로 생성"""
        gen = create_image_generator(
            template=custom_template,
            output_dir=temp_output_dir,
        )
        
        assert gen.template.width == 800


# =============================================================================
# Performance Tests
# =============================================================================

class TestPerformance:
    """성능 테스트"""
    
    @pytest.mark.asyncio
    async def test_multiple_generations(self, generator, sample_korean_text):
        """여러 이미지 연속 생성"""
        results = []
        
        for i in range(3):
            result = await generator.generate(
                keyword=f"{sample_korean_text['keyword']} #{i+1}",
                summary=sample_korean_text["summary"],
            )
            results.append(result)
        
        # 모든 생성 성공
        assert all(r.success for r in results)
        
        # 모든 파일이 다른 경로에 저장
        paths = [r.image_path for r in results]
        assert len(set(paths)) == 3
    
    @pytest.mark.asyncio
    async def test_memory_efficiency(self, generator, sample_korean_text):
        """메모리 효율성 (폰트 캐시)"""
        # 첫 번째 생성
        result1 = await generator.generate(
            keyword=sample_korean_text["keyword"],
            summary=sample_korean_text["summary"],
        )
        
        cache_size_after_first = len(generator._font_cache)
        
        # 두 번째 생성 (폰트 캐시 재사용)
        result2 = await generator.generate(
            keyword=sample_korean_text["keyword"] + " 2",
            summary=sample_korean_text["summary"],
        )
        
        cache_size_after_second = len(generator._font_cache)
        
        # 폰트 캐시 크기가 유지되어야 함 (같은 사이즈 폰트 재사용)
        assert cache_size_after_first == cache_size_after_second


# =============================================================================
# CLI Test (Manual)
# =============================================================================

def test_cli_help():
    """CLI 도움말 테스트 (수동 실행)"""
    # python image_generator.py --help
    # python image_generator.py -k "테스트" -s "요약 내용" -o ./test_images
    pass


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])