from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class TrendDocument(BaseModel):
    """모든 수집기가 반환해야 하는 표준 문서 포맷"""

    title: str
    link: str
    published: datetime | None = None
    summary: str = ""
    keyword: str = ""
    source: str = "unknown"
    content: str = ""  # 본문 전체 (옵션)
    metadata: dict[str, Any] = Field(default_factory=dict)  # 조회수, 좋아요 등 추가 정보

    @property
    def text_to_embed(self) -> str:
        """임베딩용 텍스트 반환 (제목 + 요약)"""
        return f"{self.title} {self.summary}"


class BaseCollector(ABC):
    """모든 수집기의 부모 클래스"""

    def __init__(self, **kwargs):
        self.config = kwargs

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    @abstractmethod
    async def fetch(self, keyword: str, **kwargs) -> list[TrendDocument]:
        """키워드로 데이터를 수집하여 표준 문서 리스트를 반환"""
        pass

    @abstractmethod
    async def close(self):
        """리소스 정리 (세션 종료 등)"""
        pass
