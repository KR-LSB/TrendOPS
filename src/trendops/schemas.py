# src/trendops/schemas.py
"""
TrendOps 통합 Pydantic 스키마 모듈

Week 4 Day 2: 스키마 분리 및 확장

구조:
1. Base Enums & Common - 공통 열거형 및 기본 스키마
2. Trend & Queue - 트렌드 감지 및 Job 큐 스키마
3. Collection - 데이터 수집 스키마
4. Analysis - LLM 분석 스키마 (Week 2-4)
5. Guardrail - 콘텐츠 안전성 검증 스키마 (Week 4)
6. Publisher - SNS 발행 스키마 (Week 5 준비)
7. Error - 에러 핸들링 스키마

사용법:
    from trendops.schemas import (
        # Enums
        TrendSource, JobStatus, SentimentType, GuardrailAction,
        # Analysis
        SentimentRatio, AnalysisOutput, AnalysisResult,
        # Guardrail
        GuardrailIssue, GuardrailResult,
        # etc.
    )
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    field_validator,
    model_validator,
)

# =============================================================================
# 1. Base Enums & Common
# =============================================================================


class TrendSource(str, Enum):
    """트렌드 데이터 소스"""

    GOOGLE = "google"
    NAVER = "naver"
    YOUTUBE = "youtube"
    RSS = "rss"


class JobStatus(str, Enum):
    """Job 처리 상태"""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class SentimentType(str, Enum):
    """감성 유형"""

    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class ContentType(str, Enum):
    """콘텐츠 유형"""

    NEWS = "news"
    COMMENT = "comment"
    POST = "post"
    VIDEO = "video"


class GuardrailAction(str, Enum):
    """Guardrail 판정 액션"""

    PASS = "pass"  # 통과
    REVISE = "revise"  # 수정 필요
    REJECT = "reject"  # 거부
    REVIEW = "review"  # 사람 검토 필요


class ReviewStatus(str, Enum):
    """Human Review 상태 (Blueprint 1.2.5)"""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    MODIFIED = "modified"


class GenerationMethod(str, Enum):
    """LLM 생성 방식"""

    OUTLINES = "outlines"  # Outlines 문법 강제
    OLLAMA_JSON = "ollama_json"  # Ollama JSON 모드
    OPENAI = "openai"  # OpenAI API
    VLLM = "vllm"  # vLLM
    MOCK = "mock"  # 테스트용 Mock


# =============================================================================
# 2. Trend & Queue Schemas
# =============================================================================


class TrendKeyword(BaseModel):
    """트렌드 키워드 스키마"""

    keyword: str = Field(..., min_length=1, max_length=255, description="검색 키워드")
    source: TrendSource = Field(default=TrendSource.GOOGLE, description="데이터 소스")
    trend_score: float = Field(..., ge=0.0, le=10.0, description="트렌드 점수 (0-10)")
    search_volume: int | None = Field(default=None, ge=0, description="검색량")
    discovered_at: datetime = Field(default_factory=datetime.now, description="발견 시간")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "keyword": "트럼프 관세",
                "source": "google",
                "trend_score": 8.5,
                "search_volume": 50000,
                "discovered_at": "2025-02-15T10:30:00",
            }
        }
    )


class TrendJob(BaseModel):
    """트렌드 처리 Job 스키마"""

    job_id: UUID = Field(default_factory=uuid4, description="고유 Job ID")
    keyword_info: TrendKeyword = Field(..., description="키워드 정보")
    status: JobStatus = Field(default=JobStatus.PENDING, description="처리 상태")
    priority: int = Field(default=0, ge=0, le=10, description="우선순위 (0-10, 높을수록 우선)")
    created_at: datetime = Field(default_factory=datetime.now, description="생성 시간")
    updated_at: datetime = Field(default_factory=datetime.now, description="수정 시간")
    started_at: datetime | None = Field(default=None, description="처리 시작 시간")
    completed_at: datetime | None = Field(default=None, description="처리 완료 시간")
    error_message: str | None = Field(default=None, description="에러 메시지")
    retry_count: int = Field(default=0, ge=0, description="재시도 횟수")
    max_retries: int = Field(default=3, ge=0, description="최대 재시도 횟수")

    def mark_processing(self) -> TrendJob:
        """상태를 processing으로 변경"""
        self.status = JobStatus.PROCESSING
        self.started_at = datetime.now()
        self.updated_at = datetime.now()
        return self

    def mark_completed(self) -> TrendJob:
        """상태를 completed로 변경"""
        self.status = JobStatus.COMPLETED
        self.completed_at = datetime.now()
        self.updated_at = datetime.now()
        return self

    def mark_failed(self, error: str) -> TrendJob:
        """상태를 failed로 변경"""
        self.status = JobStatus.FAILED
        self.error_message = error
        self.updated_at = datetime.now()
        self.retry_count += 1
        return self

    def can_retry(self) -> bool:
        """재시도 가능 여부"""
        return self.retry_count < self.max_retries

    @property
    def processing_time_seconds(self) -> float | None:
        """처리 소요 시간 (초)"""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None


# =============================================================================
# 3. Collection Schemas
# =============================================================================


class NewsArticle(BaseModel):
    """뉴스 기사 스키마"""

    title: str = Field(..., min_length=1, max_length=500, description="기사 제목")
    link: HttpUrl = Field(..., description="기사 URL")
    summary: str | None = Field(default=None, max_length=2000, description="기사 요약")
    description: str | None = Field(
        default=None, max_length=2000, description="기사 설명 (summary alias)"
    )
    content: str | None = Field(default=None, description="본문 전체 (크롤링 시)")
    published_at: datetime | None = Field(default=None, description="발행일")
    source: str = Field(..., min_length=1, max_length=100, description="출처")
    author: str | None = Field(default=None, max_length=200, description="저자")
    image_url: HttpUrl | None = Field(default=None, description="대표 이미지 URL")
    content_type: ContentType = Field(default=ContentType.NEWS, description="콘텐츠 유형")

    @property
    def effective_summary(self) -> str:
        """summary 또는 description 반환"""
        return self.summary or self.description or ""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "title": "트럼프, 중국산 제품 25% 관세 부과 발표",
                "link": "https://news.example.com/article/12345",
                "summary": "미국 대통령이 새로운 관세 정책을 발표했다.",
                "published_at": "2025-02-15T09:00:00",
                "source": "google_news_rss",
            }
        }
    )


class YouTubeComment(BaseModel):
    """YouTube 댓글 스키마 (Week 3)"""

    comment_id: str = Field(..., description="댓글 ID")
    video_id: str = Field(..., description="비디오 ID")
    text: str = Field(..., min_length=1, max_length=10000, description="댓글 내용")
    author: str = Field(..., description="작성자")
    like_count: int = Field(default=0, ge=0, description="좋아요 수")
    published_at: datetime | None = Field(default=None, description="작성 시간")
    content_type: ContentType = Field(default=ContentType.COMMENT)


class CollectionResult(BaseModel):
    """수집 결과 통합 스키마"""

    keyword: str = Field(..., description="검색 키워드")
    articles: list[NewsArticle] = Field(default_factory=list, description="수집된 기사")
    comments: list[YouTubeComment] = Field(default_factory=list, description="수집된 댓글")
    collected_at: datetime = Field(default_factory=datetime.now, description="수집 시간")
    source: TrendSource = Field(..., description="수집 소스")

    @property
    def total_count(self) -> int:
        """총 수집 항목 수"""
        return len(self.articles) + len(self.comments)

    @property
    def has_data(self) -> bool:
        """데이터 존재 여부"""
        return self.total_count > 0


# =============================================================================
# 4. Analysis Schemas (Week 2-4)
# =============================================================================


class SentimentRatio(BaseModel):
    """
    감성 비율 스키마

    긍정/부정/중립 비율의 합이 1.0이 되도록 자동 정규화
    """

    positive: float = Field(..., ge=0.0, le=1.0, description="긍정 비율")
    negative: float = Field(..., ge=0.0, le=1.0, description="부정 비율")
    neutral: float = Field(..., ge=0.0, le=1.0, description="중립 비율")

    @field_validator("positive", "negative", "neutral", mode="after")
    @classmethod
    def round_ratio(cls, v: float) -> float:
        """소수점 2자리로 반올림"""
        return round(v, 2)

    @model_validator(mode="after")
    def normalize_ratios(self) -> SentimentRatio:
        """비율 합이 1.0이 되도록 정규화"""
        total = self.positive + self.negative + self.neutral
        if total > 0 and abs(total - 1.0) > 0.01:
            self.positive = round(self.positive / total, 2)
            self.negative = round(self.negative / total, 2)
            self.neutral = round(1.0 - self.positive - self.negative, 2)
        return self

    @property
    def dominant_sentiment(self) -> SentimentType:
        """가장 높은 감성 유형 반환"""
        max_val = max(self.positive, self.negative, self.neutral)
        if max_val == self.positive:
            return SentimentType.POSITIVE
        elif max_val == self.negative:
            return SentimentType.NEGATIVE
        return SentimentType.NEUTRAL

    def to_display_dict(self) -> dict[str, str]:
        """표시용 딕셔너리 반환"""
        return {
            "긍정": f"{self.positive:.0%}",
            "부정": f"{self.negative:.0%}",
            "중립": f"{self.neutral:.0%}",
        }


class AnalysisOutput(BaseModel):
    """
    LLM 분석 출력 스키마

    Outlines/guided_decoding으로 100% JSON 보장
    """

    main_cause: str = Field(
        ..., min_length=10, max_length=200, description="이 키워드가 뜬 핵심 원인 (1문장)"
    )
    sentiment_ratio: SentimentRatio = Field(..., description="여론 감성 비율")
    key_opinions: list[str] = Field(..., min_length=3, max_length=5, description="핵심 의견 3-5개")
    summary: str = Field(..., min_length=50, max_length=300, description="3줄 요약")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "main_cause": "트럼프 대통령의 중국산 제품 25% 관세 부과 발표로 인한 관심 급증",
                "sentiment_ratio": {"positive": 0.15, "negative": 0.55, "neutral": 0.30},
                "key_opinions": [
                    "국내 수출 기업들의 피해 우려 확산",
                    "반도체·배터리 업종 주가 하락",
                    "소비자 물가 상승 전망에 대한 불안감",
                ],
                "summary": "트럼프 대통령이 중국산 제품에 25% 관세를 부과한다고 발표했습니다.\n이에 따라 국내 수출 기업들이 비상 대응에 나섰으며, 특히 반도체와 배터리 업종의 타격이 우려됩니다.\n증시는 급락하고 정부는 지원 대책 마련에 나섰습니다.",
            }
        }
    )


class AnalysisResult(BaseModel):
    """분석 결과 전체 스키마"""

    keyword: str = Field(..., description="분석 대상 키워드")
    analysis: AnalysisOutput = Field(..., description="LLM 분석 결과")
    source_count: int = Field(..., ge=0, description="분석에 사용된 소스 수")
    model_version: str = Field(..., description="사용된 모델 버전")
    inference_time_seconds: float = Field(..., ge=0, description="추론 소요 시간")
    generation_method: GenerationMethod = Field(
        default=GenerationMethod.OUTLINES, description="생성 방식"
    )
    created_at: datetime = Field(default_factory=datetime.now, description="생성 시간")

    def is_valid(self) -> bool:
        """분석 결과 유효성 검사"""
        return (
            len(self.analysis.main_cause) >= 10
            and len(self.analysis.key_opinions) >= 3
            and len(self.analysis.summary) >= 50
        )

    @property
    def quality_score(self) -> float:
        """품질 점수 (0-1)"""
        score = 0.0

        # main_cause 길이 점수 (10-200자)
        cause_len = len(self.analysis.main_cause)
        if 30 <= cause_len <= 150:
            score += 0.25
        elif 10 <= cause_len <= 200:
            score += 0.15

        # key_opinions 수 점수
        opinions_count = len(self.analysis.key_opinions)
        if opinions_count >= 4:
            score += 0.25
        elif opinions_count >= 3:
            score += 0.20

        # summary 길이 점수
        summary_len = len(self.analysis.summary)
        if 100 <= summary_len <= 250:
            score += 0.25
        elif 50 <= summary_len <= 300:
            score += 0.15

        # source_count 점수
        if self.source_count >= 10:
            score += 0.25
        elif self.source_count >= 5:
            score += 0.15
        elif self.source_count >= 3:
            score += 0.10

        return round(score, 2)


# =============================================================================
# 5. Guardrail Schemas (Week 4)
# =============================================================================


class GuardrailIssueType(str, Enum):
    """Guardrail 이슈 유형"""

    POLITICAL_BIAS = "political_bias"  # 정치적 편향
    PROFANITY = "profanity"  # 욕설/비속어
    MISINFORMATION = "misinformation"  # 허위 정보
    PERSONAL_INFO = "personal_info"  # 개인정보
    HATE_SPEECH = "hate_speech"  # 혐오 발언
    SENSATIONALISM = "sensationalism"  # 선정적 표현
    UNVERIFIED_CLAIM = "unverified_claim"  # 검증되지 않은 주장
    COPYRIGHT = "copyright"  # 저작권 문제


class GuardrailIssue(BaseModel):
    """Guardrail 탐지 이슈"""

    issue_type: GuardrailIssueType = Field(..., description="이슈 유형")
    severity: Literal["low", "medium", "high", "critical"] = Field(..., description="심각도")
    description: str = Field(..., description="이슈 설명")
    location: str | None = Field(default=None, description="문제 위치 (텍스트 일부)")
    suggestion: str | None = Field(default=None, description="수정 제안")

    @property
    def is_blocking(self) -> bool:
        """차단 필요 여부"""
        return self.severity in ("high", "critical")


class GuardrailResult(BaseModel):
    """
    Guardrail 검사 결과 스키마

    Week 4: Self-Correction Loop의 핵심 스키마
    """

    content_id: str = Field(..., description="검사 대상 콘텐츠 ID")
    action: GuardrailAction = Field(..., description="권장 액션")
    is_safe: bool = Field(..., description="안전 여부")
    confidence: float = Field(..., ge=0.0, le=1.0, description="판정 신뢰도")
    issues: list[GuardrailIssue] = Field(default_factory=list, description="탐지된 이슈 목록")
    original_content: str = Field(..., description="원본 콘텐츠")
    revised_content: str | None = Field(default=None, description="수정된 콘텐츠 (revise 시)")
    review_reason: str | None = Field(default=None, description="검토 필요 사유")
    checked_at: datetime = Field(default_factory=datetime.now, description="검사 시간")
    model_version: str = Field(default="guardrail-v1", description="Guardrail 모델 버전")

    @property
    def has_blocking_issues(self) -> bool:
        """차단 필요한 이슈 존재 여부"""
        return any(issue.is_blocking for issue in self.issues)

    @property
    def issue_summary(self) -> str:
        """이슈 요약"""
        if not self.issues:
            return "No issues found"
        types = [issue.issue_type.value for issue in self.issues]
        return f"{len(self.issues)} issues: {', '.join(types)}"

    def to_log_dict(self) -> dict[str, Any]:
        """로깅용 딕셔너리"""
        return {
            "content_id": self.content_id,
            "action": self.action.value,
            "is_safe": self.is_safe,
            "confidence": self.confidence,
            "issue_count": len(self.issues),
            "has_blocking": self.has_blocking_issues,
        }

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "content_id": "analysis-12345",
                "action": "pass",
                "is_safe": True,
                "confidence": 0.95,
                "issues": [],
                "original_content": "트럼프 대통령이 관세 정책을 발표했습니다.",
                "revised_content": None,
                "review_reason": None,
            }
        }
    )


class GuardrailCheckRequest(BaseModel):
    """Guardrail 검사 요청"""

    content: str = Field(..., min_length=1, description="검사할 콘텐츠")
    content_type: Literal["summary", "opinion", "full"] = Field(
        default="summary", description="콘텐츠 유형"
    )
    keyword: str | None = Field(default=None, description="관련 키워드")
    strict_mode: bool = Field(default=False, description="엄격 모드")


# =============================================================================
# 6. Publisher Schemas (Week 5 준비)
# =============================================================================


class ContentReview(BaseModel):
    """
    발행 전 검토 대기 콘텐츠

    Blueprint Section 1.2.5: Human-in-the-Loop
    """

    content_id: str = Field(..., description="콘텐츠 ID")
    keyword: str = Field(..., description="관련 키워드")
    summary: str = Field(..., description="콘텐츠 요약")
    full_content: str = Field(..., description="전체 콘텐츠")
    image_path: str | None = Field(default=None, description="이미지 경로")
    status: ReviewStatus = Field(default=ReviewStatus.PENDING, description="검토 상태")
    generated_at: datetime = Field(default_factory=datetime.now, description="생성 시간")
    reviewed_at: datetime | None = Field(default=None, description="검토 시간")
    reviewer_note: str | None = Field(default=None, description="검토자 노트")
    guardrail_result: GuardrailResult | None = Field(
        default=None, description="Guardrail 검사 결과"
    )

    def approve(self, note: str | None = None) -> ContentReview:
        """승인 처리"""
        self.status = ReviewStatus.APPROVED
        self.reviewed_at = datetime.now()
        self.reviewer_note = note
        return self

    def reject(self, note: str) -> ContentReview:
        """거부 처리"""
        self.status = ReviewStatus.REJECTED
        self.reviewed_at = datetime.now()
        self.reviewer_note = note
        return self


class PublishRequest(BaseModel):
    """SNS 발행 요청"""

    content_id: str = Field(..., description="콘텐츠 ID")
    platform: Literal["instagram", "threads"] = Field(..., description="발행 플랫폼")
    caption: str = Field(..., max_length=2200, description="캡션")
    image_path: str = Field(..., description="이미지 경로")
    hashtags: list[str] = Field(default_factory=list, description="해시태그 목록")
    scheduled_at: datetime | None = Field(default=None, description="예약 발행 시간")


class PublishResult(BaseModel):
    """SNS 발행 결과"""

    content_id: str = Field(..., description="콘텐츠 ID")
    platform: str = Field(..., description="발행 플랫폼")
    post_id: str | None = Field(default=None, description="발행된 게시물 ID")
    post_url: HttpUrl | None = Field(default=None, description="게시물 URL")
    success: bool = Field(..., description="발행 성공 여부")
    error_message: str | None = Field(default=None, description="에러 메시지")
    published_at: datetime = Field(default_factory=datetime.now, description="발행 시간")


# =============================================================================
# 7. Error Schemas
# =============================================================================


class ErrorSeverity(str, Enum):
    """에러 심각도"""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ErrorCategory(str, Enum):
    """에러 카테고리"""

    NETWORK = "network"
    RATE_LIMIT = "rate_limit"
    PARSING = "parsing"
    VALIDATION = "validation"
    LLM = "llm"
    DATABASE = "database"
    AUTHENTICATION = "authentication"
    UNKNOWN = "unknown"


class PipelineError(BaseModel):
    """파이프라인 에러 스키마"""

    error_id: UUID = Field(default_factory=uuid4, description="에러 ID")
    category: ErrorCategory = Field(..., description="에러 카테고리")
    severity: ErrorSeverity = Field(default=ErrorSeverity.ERROR, description="심각도")
    message: str = Field(..., description="에러 메시지")
    details: dict[str, Any] = Field(default_factory=dict, description="상세 정보")
    stage: str = Field(..., description="발생 단계 (trigger/collector/analyst/publisher)")
    keyword: str | None = Field(default=None, description="관련 키워드")
    job_id: UUID | None = Field(default=None, description="관련 Job ID")
    recoverable: bool = Field(default=True, description="복구 가능 여부")
    retry_after_seconds: int | None = Field(default=None, description="재시도 대기 시간")
    occurred_at: datetime = Field(default_factory=datetime.now, description="발생 시간")
    stack_trace: str | None = Field(default=None, description="스택 트레이스")

    @classmethod
    def from_exception(
        cls,
        exception: Exception,
        stage: str,
        category: ErrorCategory = ErrorCategory.UNKNOWN,
        **kwargs,
    ) -> PipelineError:
        """예외로부터 PipelineError 생성"""
        import traceback

        return cls(
            category=category,
            message=str(exception),
            stage=stage,
            stack_trace=traceback.format_exc(),
            **kwargs,
        )

    def to_log_dict(self) -> dict[str, Any]:
        """로깅용 딕셔너리"""
        return {
            "error_id": str(self.error_id),
            "category": self.category.value,
            "severity": self.severity.value,
            "message": self.message,
            "stage": self.stage,
            "keyword": self.keyword,
            "recoverable": self.recoverable,
        }


# =============================================================================
# 8. Pipeline State Schemas
# =============================================================================


class PipelineStage(str, Enum):
    """파이프라인 단계"""

    TRIGGER = "trigger"
    COLLECT = "collect"
    DEDUPLICATE = "deduplicate"
    ANALYZE = "analyze"
    GUARDRAIL = "guardrail"
    REVIEW = "review"
    PUBLISH = "publish"


class PipelineState(BaseModel):
    """파이프라인 상태 스키마"""

    job_id: UUID = Field(..., description="Job ID")
    keyword: str = Field(..., description="처리 중인 키워드")
    current_stage: PipelineStage = Field(..., description="현재 단계")
    stages_completed: list[PipelineStage] = Field(default_factory=list, description="완료된 단계")
    started_at: datetime = Field(default_factory=datetime.now, description="시작 시간")
    updated_at: datetime = Field(default_factory=datetime.now, description="갱신 시간")
    errors: list[PipelineError] = Field(default_factory=list, description="발생한 에러")

    # 단계별 결과
    trigger_result: TrendKeyword | None = Field(default=None)
    collection_result: CollectionResult | None = Field(default=None)
    analysis_result: AnalysisResult | None = Field(default=None)
    guardrail_result: GuardrailResult | None = Field(default=None)
    publish_result: PublishResult | None = Field(default=None)

    def advance_stage(self, next_stage: PipelineStage) -> PipelineState:
        """다음 단계로 진행"""
        self.stages_completed.append(self.current_stage)
        self.current_stage = next_stage
        self.updated_at = datetime.now()
        return self

    def add_error(self, error: PipelineError) -> PipelineState:
        """에러 추가"""
        self.errors.append(error)
        self.updated_at = datetime.now()
        return self

    @property
    def is_failed(self) -> bool:
        """실패 여부"""
        return any(
            e.severity in (ErrorSeverity.ERROR, ErrorSeverity.CRITICAL) and not e.recoverable
            for e in self.errors
        )

    @property
    def progress_percent(self) -> float:
        """진행률 (%)"""
        total_stages = len(PipelineStage)
        completed = len(self.stages_completed)
        return round(completed / total_stages * 100, 1)


# =============================================================================
# Export All
# =============================================================================

__all__ = [
    # Enums
    "TrendSource",
    "JobStatus",
    "SentimentType",
    "ContentType",
    "GuardrailAction",
    "ReviewStatus",
    "GenerationMethod",
    "GuardrailIssueType",
    "ErrorSeverity",
    "ErrorCategory",
    "PipelineStage",
    # Trend & Queue
    "TrendKeyword",
    "TrendJob",
    # Collection
    "NewsArticle",
    "YouTubeComment",
    "CollectionResult",
    # Analysis
    "SentimentRatio",
    "AnalysisOutput",
    "AnalysisResult",
    # Guardrail
    "GuardrailIssue",
    "GuardrailResult",
    "GuardrailCheckRequest",
    # Publisher
    "ContentReview",
    "PublishRequest",
    "PublishResult",
    # Error
    "PipelineError",
    # Pipeline State
    "PipelineState",
]
