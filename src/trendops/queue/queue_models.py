# src/trendops/queue/queue_models.py
from datetime import datetime
from enum import Enum
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, HttpUrl


class TrendSource(str, Enum):
    """트렌드 데이터 소스"""

    GOOGLE = "google"
    NAVER = "naver"


class JobStatus(str, Enum):
    """Job 처리 상태"""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class TrendKeyword(BaseModel):
    """트렌드 키워드 스키마"""

    keyword: str = Field(..., min_length=1, max_length=255, description="검색 키워드")
    source: TrendSource = Field(default=TrendSource.GOOGLE, description="데이터 소스")
    trend_score: float = Field(..., ge=0.0, le=10.0, description="트렌드 점수 (0-10)")
    discovered_at: datetime = Field(default_factory=datetime.now, description="발견 시간")

    class Config:
        json_schema_extra = {
            "example": {
                "keyword": "트럼프 관세",
                "source": "google",
                "trend_score": 8.5,
                "discovered_at": "2025-02-15T10:30:00",
            }
        }


class NewsArticle(BaseModel):
    """뉴스 기사 스키마"""

    title: str = Field(..., min_length=1, max_length=500, description="기사 제목")
    link: HttpUrl = Field(..., description="기사 URL")
    description: str | None = Field(default=None, max_length=2000, description="기사 요약")
    published_at: datetime | None = Field(default=None, description="발행일")
    source: str = Field(..., min_length=1, max_length=100, description="출처 (예: google_news_rss)")

    class Config:
        json_schema_extra = {
            "example": {
                "title": "트럼프, 중국산 제품 25% 관세 부과 발표",
                "link": "https://news.example.com/article/12345",
                "description": "미국 대통령이 새로운 관세 정책을 발표했다.",
                "published_at": "2025-02-15T09:00:00",
                "source": "google_news_rss",
            }
        }


class TrendJob(BaseModel):
    """트렌드 처리 Job 스키마"""

    job_id: UUID = Field(default_factory=uuid4, description="고유 Job ID")
    keyword_info: TrendKeyword = Field(..., description="키워드 정보")
    status: JobStatus = Field(default=JobStatus.PENDING, description="처리 상태")
    created_at: datetime = Field(default_factory=datetime.now, description="생성 시간")
    updated_at: datetime = Field(default_factory=datetime.now, description="수정 시간")
    error_message: str | None = Field(default=None, description="에러 메시지 (실패 시)")
    retry_count: int = Field(default=0, ge=0, description="재시도 횟수")

    def mark_processing(self) -> "TrendJob":
        """상태를 processing으로 변경"""
        self.status = JobStatus.PROCESSING
        self.updated_at = datetime.now()
        return self

    def mark_completed(self) -> "TrendJob":
        """상태를 completed로 변경"""
        self.status = JobStatus.COMPLETED
        self.updated_at = datetime.now()
        return self

    def mark_failed(self, error: str) -> "TrendJob":
        """상태를 failed로 변경"""
        self.status = JobStatus.FAILED
        self.error_message = error
        self.updated_at = datetime.now()
        self.retry_count += 1
        return self

    class Config:
        json_schema_extra = {
            "example": {
                "job_id": "550e8400-e29b-41d4-a716-446655440000",
                "keyword_info": {
                    "keyword": "트럼프 관세",
                    "source": "google",
                    "trend_score": 8.5,
                    "discovered_at": "2025-02-15T10:30:00",
                },
                "status": "pending",
                "created_at": "2025-02-15T10:30:00",
                "updated_at": "2025-02-15T10:30:00",
                "error_message": None,
                "retry_count": 0,
            }
        }


class CollectedArticles(BaseModel):
    """수집된 기사 묶음 스키마"""

    keyword: str = Field(..., description="검색 키워드")
    articles: list[NewsArticle] = Field(default_factory=list, description="수집된 기사 목록")
    collected_at: datetime = Field(default_factory=datetime.now, description="수집 시간")
    total_count: int = Field(default=0, ge=0, description="수집된 기사 수")

    def __init__(self, **data):
        super().__init__(**data)
        if self.total_count == 0:
            self.total_count = len(self.articles)
