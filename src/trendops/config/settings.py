# src/trendops/config/settings.py
"""
TrendOps 설정 관리

Blueprint Section 7.3: 환경 변수 관리 (.env)
Pydantic Settings를 사용한 타입 안전한 설정 로드
"""
from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    TrendOps 전역 설정
    
    ⚠️ CRITICAL HARDWARE CONSTRAINTS (Blueprint 준수):
    - GPU (16GB): vLLM 전용 (Qwen2.5-7B-AWQ)
    - CPU: Embedding (bge-m3-korean), 전처리, 비즈니스 로직
    
    모든 설정은 .env 파일에서 로드되며, 환경 변수로 오버라이드 가능
    """
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )
    
    # =========================================================================
    # DATABASE SETTINGS
    # =========================================================================
    
    # Redis - Job Queue, Cache
    redis_url: str = Field(
        default="redis://localhost:6379",
        description="Redis 연결 URL (Job Queue, Rate Limit, Cache)",
    )
    
    # ChromaDB - Vector Store
    chromadb_path: str = Field(
        default="./data/chromadb",
        description="ChromaDB 영속화 디렉토리 경로",
    )
    chromadb_collection_name: str = Field(
        default="trend_documents",
        description="ChromaDB 컬렉션 이름",
    )
    
    # PostgreSQL (Week 3+에서 사용)
    postgres_url: str = Field(
        default="postgresql://trendops:trendops@localhost:5432/trendops",
        description="PostgreSQL 연결 URL (분석 결과 영속화)",
    )
    
    # =========================================================================
    # MODEL SETTINGS
    # =========================================================================
    
    # vLLM - LLM 서빙 (GPU 전용)
    vllm_url: str = Field(
        default="http://localhost:11434",
        description="vLLM OpenAI 호환 API 엔드포인트",
    )
    vllm_model: str = Field(
        default="exaone3.5:latest",  # Ollama 모델명
        description="vLLM에서 서빙할 모델 이름",
    )
    vllm_max_tokens: int = Field(
        default=1024,
        ge=128,
        le=4096,
        description="LLM 응답 최대 토큰 수",
    )
    vllm_temperature: float = Field(
        default=0.3,
        ge=0.0,
        le=2.0,
        description="LLM 샘플링 온도 (낮을수록 결정적)",
    )
    
    # Embedding Model (CPU 전용)
    embedding_model_name: str = Field(
        default="upskyy/bge-m3-korean",
        description="Sentence Transformers 임베딩 모델 (CPU에서 실행)",
    )
    embedding_batch_size: int = Field(
        default=64,
        ge=1,
        le=256,
        description="임베딩 배치 크기 (CPU 모드에서는 크게 설정)",
    )
    embedding_max_seq_length: int = Field(
        default=512,
        ge=128,
        le=8192,
        description="임베딩 최대 시퀀스 길이",
    )
    
    # =========================================================================
    # HARDWARE SETTINGS
    # =========================================================================
    
    # CPU Thread 설정 (Embedding, 전처리용)
    cpu_threads: int = Field(
        default=8,
        ge=1,
        le=32,
        description="CPU 작업용 스레드 수 (Embedding, 전처리)",
    )
    
    # GPU 설정 (vLLM 전용)
    gpu_memory_utilization: float = Field(
        default=0.90,
        ge=0.5,
        le=0.95,
        description="vLLM GPU 메모리 사용률 (단독 점유이므로 높게 설정)",
    )
    
    # =========================================================================
    # COLLECTOR SETTINGS
    # =========================================================================
    
    rss_timeout_seconds: int = Field(
        default=30,
        ge=5,
        le=120,
        description="RSS 요청 타임아웃 (초)",
    )
    rss_max_results: int = Field(
        default=20,
        ge=5,
        le=100,
        description="RSS 키워드당 최대 수집 기사 수",
    )
    rss_rate_limit_delay: float = Field(
        default=0.5,
        ge=0.0,
        le=5.0,
        description="RSS 요청 간 딜레이 (초)",
    )
    
    # =========================================================================
    # TRIGGER SETTINGS
    # =========================================================================
    
    trend_min_score: float = Field(
        default=7.0,
        ge=0.0,
        le=10.0,
        description="Job Queue에 추가할 최소 트렌드 점수",
    )
    trend_max_keywords: int = Field(
        default=10,
        ge=1,
        le=50,
        description="한 번에 처리할 최대 트렌드 키워드 수",
    )
    
    # =========================================================================
    # ANALYST SETTINGS (Week 2)
    # =========================================================================
    
    pipeline_top_k_retrieve: int = Field(
        default=5,
        ge=1,
        le=50,
        description="파이프라인 실행 시 검색할 상위 문서 개수 (Top-K)",
    )

    pipeline_min_similarity: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="파이프라인 검색 시 최소 유사도 임계값",
    )

    analyst_context_window: int = Field(
        default=10,
        ge=3,
        le=30,
        description="LLM 분석 시 사용할 컨텍스트 문서 수",
    )
    analyst_similarity_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="RAG 검색 최소 유사도 임계값",
    )
    
    # =========================================================================
    # DEDUPLICATION SETTINGS (Week 3)
    # =========================================================================
    
    dedup_similarity_threshold: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="의미 기반 중복 제거 유사도 임계값 (95% 이상 중복 판정)",
    )
    
    # =========================================================================
    # API KEYS (Week 5+에서 사용)
    # =========================================================================
    
    instagram_access_token: str = Field(
        default="",
        description="Instagram Graph API 액세스 토큰",
    )
    instagram_business_account_id: str = Field(
        default="",
        description="Instagram 비즈니스 계정 ID",
    )
    google_api_key: str = Field(
        default="",
        description="Google API 키 (선택적)",
    )
    
    # =========================================================================
    # MONITORING SETTINGS
    # =========================================================================
    
    slack_webhook_url: str = Field(
        default="",
        description="Slack 알림 웹훅 URL",
    )
    
    # =========================================================================
    # ENVIRONMENT SETTINGS
    # =========================================================================
    
    env: str = Field(
        default="development",
        description="실행 환경 (development | staging | production)",
    )
    log_level: str = Field(
        default="INFO",
        description="로깅 레벨 (DEBUG | INFO | WARNING | ERROR)",
    )
    log_dir: str = Field(
        default="./logs",
        description="로그 파일 디렉토리",
    )
    
    # =========================================================================
    # VALIDATORS
    # =========================================================================
    
    @field_validator("chromadb_path", "log_dir")
    @classmethod
    def ensure_directory_exists(cls, v: str) -> str:
        """디렉토리 경로 검증 및 생성"""
        path = Path(v)
        path.mkdir(parents=True, exist_ok=True)
        return str(path.resolve())
    
    @field_validator("env")
    @classmethod
    def validate_env(cls, v: str) -> str:
        """환경 값 검증"""
        allowed = {"development", "staging", "production"}
        if v.lower() not in allowed:
            raise ValueError(f"env must be one of {allowed}")
        return v.lower()
    
    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """로그 레벨 검증"""
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in allowed:
            raise ValueError(f"log_level must be one of {allowed}")
        return v.upper()
    
    # =========================================================================
    # METHODS
    # =========================================================================
    
    def get_chromadb_path(self) -> Path:
        """
        ChromaDB 저장 경로 반환 (디렉토리 자동 생성)
        
        Returns:
            Path: ChromaDB 영속화 디렉토리 경로
        """
        path = Path(self.chromadb_path)
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    # =========================================================================
    # COMPUTED PROPERTIES
    # =========================================================================
    
    @property
    def is_production(self) -> bool:
        """프로덕션 환경 여부"""
        return self.env == "production"
    
    @property
    def is_development(self) -> bool:
        """개발 환경 여부"""
        return self.env == "development"
    
    @property
    def vllm_api_base(self) -> str:
        """vLLM OpenAI 호환 API Base URL"""
        return f"{self.vllm_url}/v1"

    @property
    def chroma_persist_directory(self) -> Path:
        """
        [호환성 패치]
        기존 코드가 chroma_persist_directory를 참조할 경우를 위한 alias
        Path 객체 반환 (.mkdir() 호출 가능)
        """
        return Path(self.chromadb_path)

    @property
    def chroma_persist_path(self) -> Path:
        """
        [호환성 패치]
        파이프라인 코드가 .mkdir()을 호출할 수 있도록 
        Path 객체 반환
        """
        return Path(self.chromadb_path)

    @property
    def chroma_collection_name(self) -> str:
        """
        [호환성 패치] 
        코드(chroma_collection_name)와 설정(chromadb_collection_name) 이름 매핑
        """
        return self.chromadb_collection_name

    @property
    def chroma_distance_metric(self) -> str:
        """
        [호환성 패치]
        ChromaDB 거리 계산 방식 (기본값: cosine)
        """
        return "cosine"


@lru_cache()
def get_settings() -> Settings:
    """
    싱글톤 패턴으로 설정 로드
    
    최초 호출 시 .env 파일을 파싱하고 캐시
    이후 호출에서는 캐시된 인스턴스 반환
    
    Returns:
        Settings: 전역 설정 인스턴스
        
    Usage:
        from trendops.config.settings import get_settings
        settings = get_settings()
        print(settings.vllm_url)
    """
    return Settings()


def clear_settings_cache() -> None:
    """
    설정 캐시 초기화 (테스트용)
    
    단위 테스트에서 설정을 변경할 때 사용
    """
    get_settings.cache_clear()