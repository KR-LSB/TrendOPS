# src/trendops/analyst/structured_analyzer.py
"""
Week 4 Day 1: Outlines + Ollama 통합 Structured Analyzer

Blueprint Week 4 Goal:
- JSON 출력 100% 보장 (Outlines guided decoding)
- 7B 모델의 출력 불안정 해결
- Retry 불필요한 확정적 JSON 생성

핵심 설계:
1. Primary: Outlines + Ollama (문법적 JSON 강제)
2. Fallback: Ollama JSON 모드 + Pydantic validation
3. 기존 analyzer_llm.py 스키마 완전 호환
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, TypeVar

from pydantic import BaseModel, Field, ValidationError, field_validator

# =============================================================================
# Pydantic Schemas (analyzer_llm.py와 호환)
# =============================================================================


class SentimentRatio(BaseModel):
    """감성 비율 스키마"""

    positive: float = Field(..., ge=0.0, le=1.0, description="긍정 비율")
    negative: float = Field(..., ge=0.0, le=1.0, description="부정 비율")
    neutral: float = Field(..., ge=0.0, le=1.0, description="중립 비율")

    @field_validator("positive", "negative", "neutral", mode="after")
    @classmethod
    def round_ratio(cls, v: float) -> float:
        return round(v, 2)

    def model_post_init(self, __context: Any) -> None:
        """비율 합이 1.0이 되도록 정규화"""
        total = self.positive + self.negative + self.neutral
        if total > 0 and abs(total - 1.0) > 0.01:
            self.positive = round(self.positive / total, 2)
            self.negative = round(self.negative / total, 2)
            self.neutral = round(1.0 - self.positive - self.negative, 2)


class AnalysisOutput(BaseModel):
    """
    LLM 분석 출력 스키마 (Outlines용)

    Week 4: Outlines가 이 스키마를 기반으로 JSON 문법 강제
    """

    main_cause: str = Field(
        ..., min_length=10, max_length=200, description="이 키워드가 뜬 핵심 원인 (1문장)"
    )
    sentiment_ratio: SentimentRatio = Field(..., description="여론 감성 비율")
    key_opinions: list[str] = Field(..., min_length=3, max_length=5, description="핵심 의견 3-5개")
    summary: str = Field(..., min_length=50, max_length=300, description="3줄 요약")


class AnalysisResult(BaseModel):
    """분석 결과 전체 스키마"""

    keyword: str = Field(..., description="분석 대상 키워드")
    analysis: AnalysisOutput = Field(..., description="LLM 분석 결과")
    source_count: int = Field(..., ge=0, description="분석에 사용된 소스 수")
    model_version: str = Field(..., description="사용된 모델 버전")
    inference_time_seconds: float = Field(..., ge=0, description="추론 소요 시간")
    generation_method: str = Field(default="outlines", description="생성 방식")
    created_at: datetime = Field(default_factory=datetime.now, description="생성 시간")

    def is_valid(self) -> bool:
        """분석 결과 유효성 검사"""
        return (
            len(self.analysis.main_cause) >= 10
            and len(self.analysis.key_opinions) >= 3
            and len(self.analysis.summary) >= 50
        )


# =============================================================================
# Prompts (Blueprint Section 1.2.3 & 1.3.3)
# =============================================================================

SYSTEM_PROMPT = """당신은 중립적이고 객관적인 뉴스 분석 전문가입니다.

## 필수 준수 사항:
1. 특정 정치인/정당을 "좋다/나쁘다"로 평가하지 마세요.
2. 원문 기사를 그대로 인용하지 말고, 통계적 요약만 제공하세요.
3. "~한 것으로 알려졌다", "~라는 의견이 있다" 형태의 객관적 서술만 사용하세요.
4. 검증되지 않은 사실을 단정적으로 서술하지 마세요.
5. 개인을 특정할 수 있는 정보(이름, 연락처 등)는 절대 포함하지 마세요.

## 콘텐츠 성격:
- 이 콘텐츠는 "정보 요약형"입니다.
- 우리의 역할은 "현상 설명"이지, "의견 제시"가 아닙니다.
- 독자가 스스로 판단할 수 있도록 균형 잡힌 정보를 제공합니다.

## 당신의 역할:
1. 제공된 뉴스/여론 데이터를 분석하여 핵심 내용을 요약합니다.
2. 감정적이거나 편향된 표현을 배제하고 사실에 기반한 분석을 제공합니다.
3. 다양한 관점을 균형 있게 반영합니다.
"""


def build_user_prompt(keyword: str, context: str) -> str:
    """사용자 프롬프트 생성"""
    return f"""## 분석 대상 키워드: {keyword}

## 수집된 뉴스/여론 데이터:
{context}

## 분석 지침:
1. main_cause: 이 키워드가 화제가 된 핵심 원인을 1문장으로 설명
2. sentiment_ratio: 긍정/부정/중립 비율 (합이 1.0)
3. key_opinions: 주요 여론/의견 3-5개
4. summary: 전체 상황을 3줄로 요약 (줄바꿈은 \\n 사용)

JSON 형식으로만 응답하세요."""


# =============================================================================
# Generation Backend Interface
# =============================================================================

T = TypeVar("T", bound=BaseModel)


class GenerationBackend(ABC):
    """생성 백엔드 추상 클래스"""

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        schema: type[T],
        system_prompt: str | None = None,
    ) -> T:
        """스키마에 맞는 구조화된 출력 생성"""
        pass

    @abstractmethod
    def get_name(self) -> str:
        """백엔드 이름 반환"""
        pass


class OutlinesOllamaBackend(GenerationBackend):
    """
    Outlines + Ollama 백엔드

    JSON 문법을 강제하여 100% 유효한 JSON 출력 보장
    """

    def __init__(
        self,
        model_name: str = "exaone3.5",
        base_url: str = "http://localhost:11434",
    ):
        self.model_name = model_name
        self.base_url = base_url
        self._model = None
        self._generator_cache: dict[type, Any] = {}

    def _get_model(self):
        """Outlines 모델 lazy loading (호환성 개선 패치)"""
        if self._model is None:
            try:
                from outlines import models

                # 1. models.ollama가 존재하는지 확인 (최신 버전 outlines)
                if hasattr(models, "ollama"):
                    self._model = models.ollama(
                        self.model_name,
                        base_url=self.base_url,
                    )
                # 2. 없다면 OpenAI 호환 모드로 연결 (구버전 outlines 대응)
                # Ollama는 http://localhost:11434/v1 에서 OpenAI API와 호환됩니다.
                else:
                    # URL 끝에 /v1이 없으면 추가
                    base_url = self.base_url.rstrip("/")
                    if not base_url.endswith("/v1"):
                        base_url += "/v1"

                    self._model = models.openai(
                        self.model_name,
                        base_url=base_url,
                        api_key="ollama",  # 더미 키 (Ollama는 키 검사 안함)
                    )

            except ImportError:
                raise ImportError("outlines 라이브러리가 필요합니다: pip install outlines")
            except Exception as e:
                # 상세 에러 로깅
                print(f"[DEBUG] Outlines Init Error: {e}")
                raise RuntimeError(f"Ollama 모델 로드 실패: {e}")

        return self._model

    def _get_generator(self, schema: type[T]):
        """스키마별 generator 캐싱"""
        if schema not in self._generator_cache:
            from outlines import generate

            model = self._get_model()
            self._generator_cache[schema] = generate.json(model, schema)
        return self._generator_cache[schema]

    async def generate(
        self,
        prompt: str,
        schema: type[T],
        system_prompt: str | None = None,
    ) -> T:
        """Outlines를 사용한 구조화된 JSON 생성"""
        generator = self._get_generator(schema)

        # 시스템 프롬프트와 사용자 프롬프트 결합
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"

        # Outlines는 동기 함수이므로 executor에서 실행
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            generator,
            full_prompt,
        )

        return result

    def get_name(self) -> str:
        return f"outlines-ollama:{self.model_name}"


class OllamaJsonModeBackend(GenerationBackend):
    """
    Ollama JSON 모드 백엔드 (Fallback)

    Ollama의 native JSON 모드 + Pydantic validation
    Outlines가 실패할 경우 사용
    """

    def __init__(
        self,
        model_name: str = "exaone3.5",
        base_url: str = "http://localhost:11434",
        max_retries: int = 3,
    ):
        self.model_name = model_name
        self.base_url = base_url
        self.max_retries = max_retries
        self._client = None

    def _get_client(self):
        """Ollama 클라이언트 lazy loading"""
        if self._client is None:
            try:
                from ollama import AsyncClient

                self._client = AsyncClient(host=self.base_url)
            except ImportError:
                raise ImportError("ollama 라이브러리가 필요합니다: pip install ollama")
        return self._client

    async def generate(
        self,
        prompt: str,
        schema: type[T],
        system_prompt: str | None = None,
    ) -> T:
        """Ollama JSON 모드를 사용한 생성 + Pydantic 검증"""
        client = self._get_client()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        last_error: Exception | None = None

        for attempt in range(self.max_retries):
            try:
                response = await client.chat(
                    model=self.model_name,
                    messages=messages,
                    format="json",  # Ollama JSON 모드
                    options={
                        "temperature": 0.3,
                        "num_predict": 2048,
                    },
                )

                content = response["message"]["content"]

                # JSON 파싱 시도
                try:
                    data = json.loads(content)
                except json.JSONDecodeError:
                    # JSON 블록 추출 시도
                    json_match = re.search(r"```json\s*(.*?)\s*```", content, re.DOTALL)
                    if json_match:
                        data = json.loads(json_match.group(1))
                    else:
                        # 중괄호로 시작하는 부분 추출
                        json_start = content.find("{")
                        json_end = content.rfind("}") + 1
                        if json_start >= 0 and json_end > json_start:
                            data = json.loads(content[json_start:json_end])
                        else:
                            raise ValueError("JSON을 찾을 수 없습니다")

                # Pydantic 검증
                return schema.model_validate(data)

            except (json.JSONDecodeError, ValidationError) as e:
                last_error = e
                # 에러 피드백과 함께 재시도
                if attempt < self.max_retries - 1:
                    feedback_msg = f"이전 응답이 유효하지 않습니다. 오류: {str(e)}\n\n올바른 JSON 형식으로 다시 응답해주세요."
                    messages.append({"role": "assistant", "content": content})
                    messages.append({"role": "user", "content": feedback_msg})
                continue
            except Exception as e:
                last_error = e
                break

        raise RuntimeError(f"JSON 생성 실패 (시도 {self.max_retries}회): {last_error}")

    def get_name(self) -> str:
        return f"ollama-json:{self.model_name}"


# =============================================================================
# Structured Analyzer (Main Class)
# =============================================================================


class StructuredAnalyzer:
    """
    Week 4 핵심: 구조화된 출력 보장 분석기

    특징:
    1. Outlines를 사용한 JSON 문법 강제 (Primary)
    2. Ollama JSON 모드 fallback (Secondary)
    3. 기존 analyzer_llm.py와 완전 호환

    Usage:
        async with StructuredAnalyzer() as analyzer:
            result = await analyzer.analyze(keyword, articles)
    """

    def __init__(
        self,
        model_name: str = "exaone3.5",
        base_url: str = "http://localhost:11434",
        use_outlines: bool = True,
    ):
        """
        Args:
            model_name: Ollama 모델 이름
            base_url: Ollama 서버 URL
            use_outlines: Outlines 사용 여부 (False면 JSON 모드만 사용)
        """
        self.model_name = model_name
        self.base_url = base_url
        self.use_outlines = use_outlines

        # 백엔드 초기화
        self._primary_backend: GenerationBackend | None = None
        self._fallback_backend: GenerationBackend | None = None
        self._backend_initialized = False

    def _init_backends(self) -> None:
        """백엔드 lazy 초기화"""
        if self._backend_initialized:
            return

        # Primary: Outlines + Ollama
        if self.use_outlines:
            try:
                self._primary_backend = OutlinesOllamaBackend(
                    model_name=self.model_name,
                    base_url=self.base_url,
                )
            except ImportError:
                print("[WARNING] Outlines 사용 불가, JSON 모드로 fallback")
                self._primary_backend = None

        # Fallback: Ollama JSON 모드
        self._fallback_backend = OllamaJsonModeBackend(
            model_name=self.model_name,
            base_url=self.base_url,
        )

        self._backend_initialized = True

    def _build_context(self, articles: list[dict[str, Any]]) -> str:
        """뉴스 기사 목록을 컨텍스트 문자열로 변환"""
        context_parts = []

        for i, article in enumerate(articles[:15], 1):  # 최대 15개
            title = article.get("title", "제목 없음")
            summary = article.get("summary") or article.get("description", "")
            source = article.get("source", "알 수 없음")
            published = article.get("published") or article.get("published_at", "")

            # 요약이 너무 길면 자르기
            if len(summary) > 300:
                summary = summary[:300] + "..."

            context_parts.append(
                f"[{i}] {title}\n"
                f"    출처: {source} | 발행: {published}\n"
                f"    요약: {summary}"
            )

        return "\n\n".join(context_parts)

    async def analyze(
        self,
        keyword: str,
        articles: list[dict[str, Any]],
    ) -> AnalysisResult:
        """
        뉴스 기사들을 분석하여 구조화된 결과 반환

        Args:
            keyword: 분석 대상 키워드
            articles: 뉴스 기사 목록

        Returns:
            AnalysisResult: 100% 유효한 구조화된 분석 결과
        """
        if not articles:
            raise ValueError("분석할 기사가 없습니다")

        if not keyword or not keyword.strip():
            raise ValueError("키워드가 비어있습니다")

        self._init_backends()

        # 컨텍스트 구성
        context = self._build_context(articles)
        prompt = build_user_prompt(keyword, context)

        start_time = time.time()
        analysis: AnalysisOutput | None = None
        backend_used = ""

        # Primary 백엔드 시도
        if self._primary_backend is not None:
            try:
                analysis = await self._primary_backend.generate(
                    prompt=prompt,
                    schema=AnalysisOutput,
                    system_prompt=SYSTEM_PROMPT,
                )
                backend_used = self._primary_backend.get_name()
            except Exception as e:
                print(f"[WARNING] Primary 백엔드 실패: {e}")

        # Fallback 백엔드 시도
        if analysis is None and self._fallback_backend is not None:
            try:
                analysis = await self._fallback_backend.generate(
                    prompt=prompt,
                    schema=AnalysisOutput,
                    system_prompt=SYSTEM_PROMPT,
                )
                backend_used = self._fallback_backend.get_name()
            except Exception as e:
                raise RuntimeError(f"모든 백엔드 실패: {e}")

        if analysis is None:
            raise RuntimeError("분석 생성 실패: 사용 가능한 백엔드가 없습니다")

        inference_time = time.time() - start_time

        return AnalysisResult(
            keyword=keyword,
            analysis=analysis,
            source_count=len(articles),
            model_version=self.model_name,
            inference_time_seconds=round(inference_time, 2),
            generation_method=backend_used,
        )

    async def analyze_from_collection_result(
        self,
        collection_result: Any,
    ) -> AnalysisResult:
        """
        CollectionResult 객체로부터 직접 분석 수행

        collector_rss_google.py의 CollectionResult와 연동
        """
        articles = [
            {
                "title": article.title,
                "summary": article.summary,
                "source": article.source,
                "published": article.published.isoformat() if article.published else None,
            }
            for article in collection_result.articles
        ]

        return await self.analyze(
            keyword=collection_result.keyword,
            articles=articles,
        )

    async def close(self) -> None:
        """리소스 정리"""
        pass  # 현재 특별한 정리 불필요

    async def __aenter__(self) -> StructuredAnalyzer:
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()


# =============================================================================
# Convenience Functions
# =============================================================================


async def analyze_keyword_structured(
    keyword: str,
    articles: list[dict[str, Any]],
    model_name: str = "exaone3.5",
) -> AnalysisResult:
    """
    단일 키워드 분석 편의 함수

    Usage:
        result = await analyze_keyword_structured(
            keyword="트럼프 관세",
            articles=[
                {"title": "...", "summary": "...", "source": "..."},
                ...
            ]
        )
    """
    async with StructuredAnalyzer(model_name=model_name) as analyzer:
        return await analyzer.analyze(keyword, articles)


# =============================================================================
# CLI Test
# =============================================================================

if __name__ == "__main__":

    async def main() -> None:
        """테스트 실행"""
        # 테스트용 더미 뉴스 데이터
        test_articles = [
            {
                "title": "트럼프, 중국산 제품 25% 관세 부과 발표",
                "summary": "미국 대통령이 무역 전쟁 격화 속에서 새로운 관세 정책을 발표했다. 이에 따라 중국산 제품에 25%의 추가 관세가 부과될 예정이다.",
                "source": "경제일보",
                "published": "2025-02-15T09:00:00",
            },
            {
                "title": "국내 수출기업들 '비상'…반도체·배터리 업종 타격 우려",
                "summary": "미국의 관세 정책 발표 이후 국내 수출 기업들이 비상 대응에 나섰다. 특히 반도체와 배터리 업종이 직접적인 타격을 받을 것으로 전망된다.",
                "source": "산업뉴스",
                "published": "2025-02-15T10:30:00",
            },
            {
                "title": '전문가 "무역전쟁 장기화 시 국내 GDP 0.5%p 하락 가능"',
                "summary": "경제 전문가들은 미중 무역전쟁이 장기화될 경우 국내 경제에 상당한 영향을 미칠 것으로 분석했다.",
                "source": "경제연구소",
                "published": "2025-02-15T11:00:00",
            },
            {
                "title": "증시 급락…코스피 2% 이상 하락 마감",
                "summary": "관세 정책 발표 여파로 국내 증시가 급락했다. 코스피는 2% 이상 하락하며 투자자들의 불안감이 커지고 있다.",
                "source": "증권타임스",
                "published": "2025-02-15T15:30:00",
            },
            {
                "title": '정부 "수출기업 지원 대책 마련 중"',
                "summary": "정부는 미국의 관세 정책에 대응하여 수출 기업 지원 대책을 마련 중이라고 밝혔다.",
                "source": "정책브리핑",
                "published": "2025-02-15T16:00:00",
            },
        ]

        print("\n" + "=" * 70)
        print("  Week 4 Day 1: Structured Analyzer Test")
        print("  Outlines + Ollama = JSON 100% 보장")
        print("=" * 70)

        try:
            # Outlines 모드로 테스트
            print("\n🧪 Testing with Outlines backend...")
            result = await analyze_keyword_structured(
                keyword="트럼프 관세",
                articles=test_articles,
            )

            print("\n✅ 분석 완료!")
            print("\n📊 분석 결과:")
            print(f"   키워드: {result.keyword}")
            print(f"   소스 수: {result.source_count}")
            print(f"   모델: {result.model_version}")
            print(f"   생성 방식: {result.generation_method}")
            print(f"   추론 시간: {result.inference_time_seconds:.2f}초")

            print("\n🔍 핵심 원인:")
            print(f"   {result.analysis.main_cause}")

            print("\n📈 감성 비율:")
            print(f"   긍정: {result.analysis.sentiment_ratio.positive:.0%}")
            print(f"   부정: {result.analysis.sentiment_ratio.negative:.0%}")
            print(f"   중립: {result.analysis.sentiment_ratio.neutral:.0%}")

            print("\n💬 핵심 의견:")
            for i, opinion in enumerate(result.analysis.key_opinions, 1):
                print(f"   {i}. {opinion}")

            print("\n📄 3줄 요약:")
            for line in result.analysis.summary.split("\n"):
                print(f"   {line}")

            print(f"\n   유효성 검사: {'✅ 통과' if result.is_valid() else '❌ 실패'}")

            # JSON 직렬화 테스트
            print("\n📦 JSON 직렬화 테스트:")
            json_output = result.model_dump_json(indent=2)
            print(f"   크기: {len(json_output)} bytes")
            print("   ✅ JSON 직렬화 성공")

        except Exception as e:
            print(f"\n❌ 테스트 실패: {e}")
            import traceback

            traceback.print_exc()

    asyncio.run(main())
