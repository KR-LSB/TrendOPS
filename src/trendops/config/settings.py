# src/trendops/config/settings.py
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    TrendOps 설정 관리
    
    Pydantic Settings - .env 자동 로드
    타입 검증 + IDE 자동완성 지원
    """
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )
    
    # === API Keys ===
    instagram_access_token: str = ""
    instagram_business_account_id: str = ""
    google_api_key: str = ""
    
    # === Database ===
    redis_url: str = "redis://localhost:6379"
    postgres_url: str = "postgresql://trendops:trendops@localhost:5432/trendops"
    db_password: str = ""
    
    # === vLLM ===
    vllm_url: str = "http://localhost:8001"
    vllm_model: str = "Qwen/Qwen2.5-7B-Instruct-AWQ"
    
    # === Monitoring ===
    slack_webhook_url: str = ""
    
    # === Environment ===
    env: str = "development"
    log_level: str = "INFO"
    
    # === Collector Settings ===
    rss_timeout_seconds: int = 30
    rss_max_results: int = 20
    
    # === Trigger Settings ===
    trend_min_score: float = 7.0
    trend_max_keywords: int = 10


@lru_cache()
def get_settings() -> Settings:
    """싱글톤 패턴으로 설정 로드"""
    return Settings()