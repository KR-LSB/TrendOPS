# src/trendops/analyst/analyzer_llm.py
"""
vLLM 기반 뉴스 분석 클라이언트

Blueprint Week 2: LLM 연동
- AsyncOpenAI 클라이언트 사용 (vLLM OpenAI API 호환)
- 중립적 뉴스 분석가 페르소나
- JSON 구조화 출력 (Week 4에서 Outlines로 100% 보장 예정)
"""

from __future__ import annotations

import asyncio
import json
import re
from datetime import datetime
from typing import Any

from openai import AsyncOpenAI, APIError, APIConnectionError, RateLimitError
from pydantic import BaseModel, Field, field_validator

from trendops.config.settings import get_settings
from trendops.utils.logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# Pydantic Models (Blueprint Section 3: Analysis Results Schema)
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
    LLM 분석 출력 스키마
    
    Blueprint Section 6.1 참조:
    Week 4에서 Outlines/guided_decoding으로 100% JSON 보장 예정
    """
    main_cause: str = Field(
        ..., 
        min_length=10,
        max_length=200,
        description="이 키워드가 뜬 핵심 원인 (1문장)"
    )
    sentiment_ratio: SentimentRatio = Field(
        ...,
        description="여론 감성 비율"
    )
    key_opinions: list[str] = Field(
        ...,
        min_length=3,
        max_length=5,
        description="핵심 의견 3-5개"
    )
    summary: str = Field(
        ...,
        min_length=50,
        max_length=300,
        description="3줄 요약"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "main_cause": "트럼프 대통령의 중국산 제품 25% 관세 부과 발표로 인한 관심 급증",
                "sentiment_ratio": {
                    "positive": 0.15,
                    "negative": 0.55,
                    "neutral": 0.30
                },
                "key_opinions": [
                    "국내 수출 기업들의 피해 우려 확산",
                    "반도체·배터리 업종 주가 하락",
                    "소비자 물가 상승 전망에 대한 불안감"
                ],
                "summary": "트럼프 대통령이 중국산 제품에 25% 관세를 부과한다고 발표했습니다.\n이에 따라 국내 수출 기업들의 피해 우려가 확산되고 있습니다.\n특히 반도체와 배터리 업종의 주가가 하락하며 시장이 불안해하고 있습니다."
            }
        }


class AnalysisResult(BaseModel):
    """분석 결과 전체 스키마"""
    keyword: str = Field(..., description="분석 대상 키워드")
    analysis: AnalysisOutput = Field(..., description="LLM 분석 결과")
    source_count: int = Field(..., ge=0, description="분석에 사용된 소스 수")
    model_version: str = Field(..., description="사용된 모델 버전")
    inference_time_seconds: float = Field(..., ge=0, description="추론 소요 시간")
    created_at: datetime = Field(default_factory=datetime.now, description="생성 시간")
    
    def is_valid(self) -> bool:
        """분석 결과 유효성 검사"""
        return (
            len(self.analysis.main_cause) >= 10
            and len(self.analysis.key_opinions) >= 3
            and len(self.analysis.summary) >= 50
        )


# =============================================================================
# System Prompts (Blueprint Section 1.2.3 & 1.3.3)
# =============================================================================

# Blueprint Section 1.2.3: CONTENT_POLICY
CONTENT_POLICY = """
## 필수 준수 사항:
1. 특정 정치인/정당을 "좋다/나쁘다"로 평가하지 마세요.
2. 원문 기사를 그대로 인용하지 말고, 통계적 요약만 제공하세요.
3. "~한 것으로 알려졌다", "~라는 의견이 있다" 형태의 객관적 서술만 사용하세요.
4. 검증되지 않은 사실을 단정적으로 서술하지 마세요.
5. 개인을 특정할 수 있는 정보(이름, 연락처 등)는 절대 포함하지 마세요.

## 콘텐츠 성격 정의:
- 이 콘텐츠는 "정보 요약형"입니다.
- 우리의 역할은 "현상 설명"이지, "의견 제시"가 아닙니다.
- 독자가 스스로 판단할 수 있도록 균형 잡힌 정보를 제공합니다.
"""

# Blueprint Section 1.3.3: CONTENT_TONE_GUIDE
CONTENT_TONE_GUIDE = """
## 콘텐츠 성격
- 우리는 "정보 요약형 미디어"입니다.
- 독자에게 판단을 강요하지 않습니다.
- 팩트를 제공하고, 판단은 독자의 몫입니다.

## 어조 가이드
✅ 좋은 예:
- "트럼프 대통령의 관세 정책 발표 후 검색량이 3배 증가했습니다."
- "커뮤니티 반응은 긍정 45%, 부정 40%로 엇갈리는 모습입니다."
- "전문가들은 이 정책이 국내 산업에 영향을 줄 수 있다고 분석합니다."

❌ 나쁜 예:
- "또다시 충격적인 발표가 있었습니다!" (자극적)
- "네티즌들이 분노하고 있습니다" (선동적)
- "이 정책은 명백히 잘못되었습니다" (의견 강요)
- "모두가 이 소식에 환호하고 있습니다" (과장)

## 핵심 질문 (콘텐츠 작성 전 자문)
1. 이 문장이 특정 입장을 옹호하거나 비난하는가?
2. 검증되지 않은 사실을 단정적으로 서술하는가?
3. 독자가 스스로 판단할 여지를 남기는가?
"""

SYSTEM_PROMPT = f"""당신은 중립적이고 객관적인 뉴스 분석 전문가입니다.

{CONTENT_POLICY}

{CONTENT_TONE_GUIDE}

## 당신의 역할:
1. 제공된 뉴스/여론 데이터를 분석하여 핵심 내용을 요약합니다.
2. 감정적이거나 편향된 표현을 배제하고 사실에 기반한 분석을 제공합니다.
3. 다양한 관점을 균형 있게 반영합니다.
4. 반드시 지정된 JSON 형식으로만 응답합니다.
"""

USER_PROMPT_TEMPLATE = """## 분석 대상 키워드: {keyword}

## 수집된 뉴스/여론 데이터:
{context}

## 출력 형식:
반드시 아래 JSON 형식으로만 응답하세요. 다른 텍스트 없이 JSON만 출력하세요.

```json
{{
    "main_cause": "이 키워드가 화제가 된 핵심 원인을 1문장으로 설명",
    "sentiment_ratio": {{
        "positive": 0.0~1.0 사이 숫자 (긍정 비율),
        "negative": 0.0~1.0 사이 숫자 (부정 비율),
        "neutral": 0.0~1.0 사이 숫자 (중립 비율)
    }},
    "key_opinions": [
        "핵심 의견 1",
        "핵심 의견 2",
        "핵심 의견 3"
    ],
    "summary": "3줄 요약 (줄바꿈은 \\n 사용)"
}}
```

주의사항:
- 감성 비율의 합은 1.0이 되어야 합니다.
- 핵심 의견은 최소 3개, 최대 5개입니다.
- 요약은 80-120자 내외로 작성하세요.
- 특정 인물이나 정당에 대한 직접적인 평가를 피하세요.
"""


# =============================================================================
# Retry Configuration
# =============================================================================

class RetryConfig(BaseModel):
    """재시도 설정"""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 30.0
    exponential_base: float = 2.0


# =============================================================================
# LLM Analyzer
# =============================================================================

class LLMAnalyzer:
    """
    vLLM 기반 뉴스 분석기
    
    Blueprint Week 2 핵심 컴포넌트:
    - AsyncOpenAI 클라이언트로 vLLM 서버 연동
    - 중립적 분석가 페르소나
    - JSON 구조화 출력
    
    Week 4에서 Outlines 적용으로 JSON 100% 보장 예정
    """
    
    def __init__(
        self,
        retry_config: RetryConfig | None = None,
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ):
        self._settings = get_settings()
        self._retry_config = retry_config or RetryConfig()
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._client: AsyncOpenAI | None = None
    
    @property
    def client(self) -> AsyncOpenAI:
        """AsyncOpenAI 클라이언트 (lazy initialization)"""
        if self._client is None:
            # vLLM은 OpenAI API 호환 - base_url만 변경
            self._client = AsyncOpenAI(
                base_url=f"{self._settings.vllm_url}/v1",
                api_key="EMPTY",  # vLLM은 API key 불필요
            )
            logger.info(
                "LLM client initialized",
                extra={
                    "base_url": self._settings.vllm_url,
                    "model": self._settings.vllm_model,
                }
            )
        return self._client
    
    def _build_context(self, articles: list[dict[str, Any]]) -> str:
        """뉴스 기사들을 컨텍스트 문자열로 변환"""
        context_parts: list[str] = []
        
        for i, article in enumerate(articles, 1):
            title = article.get("title", "제목 없음")
            summary = article.get("summary", article.get("description", ""))
            source = article.get("source", "알 수 없음")
            published = article.get("published", article.get("published_at", ""))
            
            part = f"[기사 {i}]\n제목: {title}"
            if summary:
                # 요약이 너무 길면 자르기
                summary = summary[:500] + "..." if len(summary) > 500 else summary
                part += f"\n요약: {summary}"
            if source:
                part += f"\n출처: {source}"
            if published:
                part += f"\n발행: {published}"
            
            context_parts.append(part)
        
        return "\n\n".join(context_parts)
    
    def _parse_json_response(self, content: str) -> dict[str, Any]:
        """
        LLM 응답에서 JSON 추출 및 파싱
        
        Week 4에서 Outlines 적용 시 이 함수는 불필요해짐
        """
        # JSON 코드 블록 추출 시도
        json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # 코드 블록 없이 JSON만 있는 경우
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                json_str = json_match.group(0)
            else:
                raise ValueError("No JSON found in response")
        
        # JSON 파싱
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse failed: {e}", extra={"content": content[:200]})
            raise ValueError(f"Invalid JSON: {e}")
    
    async def _call_llm(
        self, 
        keyword: str, 
        context: str,
    ) -> tuple[str, float]:
        """
        vLLM 서버에 분석 요청
        
        Returns:
            (response_content, inference_time_seconds)
        """
        start_time = datetime.now()
        
        user_prompt = USER_PROMPT_TEMPLATE.format(
            keyword=keyword,
            context=context,
        )
        
        response = await self.client.chat.completions.create(
            model=self._settings.vllm_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )
        
        inference_time = (datetime.now() - start_time).total_seconds()
        content = response.choices[0].message.content or ""
        
        logger.debug(
            "LLM response received",
            extra={
                "inference_time": inference_time,
                "response_length": len(content),
                "finish_reason": response.choices[0].finish_reason,
            }
        )
        
        return content, inference_time
    
    async def _analyze_with_retry(
        self,
        keyword: str,
        context: str,
    ) -> tuple[AnalysisOutput, float]:
        """재시도 로직이 적용된 분석 수행"""
        config = self._retry_config
        last_exception: Exception | None = None
        total_inference_time = 0.0
        
        for attempt in range(config.max_attempts):
            try:
                # LLM 호출
                content, inference_time = await self._call_llm(keyword, context)
                total_inference_time += inference_time
                
                # JSON 파싱
                json_data = self._parse_json_response(content)
                
                # Pydantic 검증
                analysis = AnalysisOutput.model_validate(json_data)
                
                logger.info(
                    "Analysis completed successfully",
                    extra={
                        "keyword": keyword,
                        "attempt": attempt + 1,
                        "inference_time": total_inference_time,
                    }
                )
                
                return analysis, total_inference_time
                
            except (ValueError, json.JSONDecodeError) as e:
                # JSON 파싱 실패 - 재시도
                last_exception = e
                logger.warning(
                    f"JSON parsing failed, attempt {attempt + 1}/{config.max_attempts}",
                    extra={"error": str(e), "keyword": keyword}
                )
                
            except (APIError, APIConnectionError) as e:
                # API 오류 - 재시도
                last_exception = e
                logger.warning(
                    f"API error, attempt {attempt + 1}/{config.max_attempts}",
                    extra={"error": str(e), "keyword": keyword}
                )
                
            except RateLimitError as e:
                # Rate limit - 더 긴 대기
                last_exception = e
                logger.warning(
                    f"Rate limited, attempt {attempt + 1}/{config.max_attempts}",
                    extra={"error": str(e), "keyword": keyword}
                )
            
            # 재시도 대기
            if attempt < config.max_attempts - 1:
                delay = min(
                    config.base_delay * (config.exponential_base ** attempt),
                    config.max_delay
                )
                await asyncio.sleep(delay)
        
        # 모든 재시도 실패
        raise RuntimeError(
            f"Analysis failed after {config.max_attempts} attempts: {last_exception}"
        )
    
    async def analyze(
        self,
        keyword: str,
        articles: list[dict[str, Any]],
    ) -> AnalysisResult:
        """
        뉴스 기사들을 분석하여 요약 결과 반환
        
        Args:
            keyword: 분석 대상 키워드
            articles: 뉴스 기사 목록 (dict 형태)
                - title: 기사 제목
                - summary/description: 기사 요약
                - source: 출처
                - published/published_at: 발행일
        
        Returns:
            AnalysisResult: 분석 결과
            
        Raises:
            RuntimeError: 분석 실패 시
            ValueError: 입력 데이터 검증 실패 시
        """
        if not articles:
            raise ValueError("No articles provided for analysis")
        
        if not keyword or not keyword.strip():
            raise ValueError("Keyword cannot be empty")
        
        logger.info(
            "Starting analysis",
            extra={"keyword": keyword, "article_count": len(articles)}
        )
        
        # 컨텍스트 구성
        context = self._build_context(articles)
        
        # LLM 분석 수행
        analysis, inference_time = await self._analyze_with_retry(keyword, context)
        
        # 결과 구성
        result = AnalysisResult(
            keyword=keyword,
            analysis=analysis,
            source_count=len(articles),
            model_version=self._settings.vllm_model,
            inference_time_seconds=inference_time,
        )
        
        logger.info(
            "Analysis result created",
            extra={
                "keyword": keyword,
                "is_valid": result.is_valid(),
                "inference_time": inference_time,
            }
        )
        
        return result
    
    async def analyze_from_collection_result(
        self,
        collection_result: Any,  # CollectionResult 타입
    ) -> AnalysisResult:
        """
        CollectionResult 객체로부터 직접 분석 수행
        
        collector_rss_google.py의 CollectionResult와 연동
        """
        # CollectionResult의 articles를 dict로 변환
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
        """클라이언트 리소스 정리"""
        if self._client is not None:
            await self._client.close()
            self._client = None
            logger.info("LLM client closed")
    
    async def __aenter__(self) -> "LLMAnalyzer":
        return self
    
    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()


# =============================================================================
# Convenience Functions
# =============================================================================

async def analyze_keyword(
    keyword: str,
    articles: list[dict[str, Any]],
) -> AnalysisResult:
    """
    단일 키워드 분석 편의 함수
    
    Usage:
        result = await analyze_keyword(
            keyword="트럼프 관세",
            articles=[
                {"title": "...", "summary": "...", "source": "..."},
                ...
            ]
        )
    """
    async with LLMAnalyzer() as analyzer:
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
                "title": "전문가 \"무역전쟁 장기화 시 국내 GDP 0.5%p 하락 가능\"",
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
                "title": "정부 \"수출기업 지원 대책 마련 중\"",
                "summary": "정부는 미국의 관세 정책에 대응하여 수출 기업 지원 대책을 마련 중이라고 밝혔다.",
                "source": "정책브리핑",
                "published": "2025-02-15T16:00:00",
            },
        ]
        
        print("\n" + "=" * 60)
        print("  LLM Analyzer Test")
        print("=" * 60)
        
        try:
            result = await analyze_keyword(
                keyword="트럼프 관세",
                articles=test_articles,
            )
            
            print(f"\n✅ 분석 완료!")
            print(f"\n📊 분석 결과:")
            print(f"   키워드: {result.keyword}")
            print(f"   소스 수: {result.source_count}")
            print(f"   모델: {result.model_version}")
            print(f"   추론 시간: {result.inference_time_seconds:.2f}초")
            print(f"\n📝 핵심 원인:")
            print(f"   {result.analysis.main_cause}")
            print(f"\n📈 감성 비율:")
            print(f"   긍정: {result.analysis.sentiment_ratio.positive:.0%}")
            print(f"   부정: {result.analysis.sentiment_ratio.negative:.0%}")
            print(f"   중립: {result.analysis.sentiment_ratio.neutral:.0%}")
            print(f"\n💬 핵심 의견:")
            for i, opinion in enumerate(result.analysis.key_opinions, 1):
                print(f"   {i}. {opinion}")
            print(f"\n📄 3줄 요약:")
            for line in result.analysis.summary.split("\n"):
                print(f"   {line}")
            print(f"\n   유효성 검사: {'✅ 통과' if result.is_valid() else '❌ 실패'}")
            
        except Exception as e:
            print(f"\n❌ 분석 실패: {e}")
            raise
    
    asyncio.run(main())