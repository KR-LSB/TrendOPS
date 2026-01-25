# src/trendops/service/deduplicator.py
"""
Semantic Deduplication Service

Blueprint Week 3 목표:
- 동일한 정보 반복 학습 방지로 스토리지 60% 절약
- 유사도 0.95 이상 문서 필터링
- EmbeddingService (CPU) 활용

⚠️ CRITICAL HARDWARE CONSTRAINT:
- Embedding은 CPU 전용 (GPU 절대 금지)
- GPU (16GB)는 vLLM 전용
"""
from __future__ import annotations

import threading
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, computed_field

from trendops.config.settings import get_settings
from trendops.service.embedding_service import EmbeddingService, get_embedding_service
from trendops.store.vector_store import VectorStore, get_vector_store, SearchResult
from trendops.utils.logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# DATA MODELS
# =============================================================================


class DeduplicationStats(BaseModel):
    """중복 제거 통계 모델"""
    
    total_processed: int = Field(default=0, description="총 처리 문서 수")
    unique_added: int = Field(default=0, description="유니크하게 추가된 문서 수")
    duplicates_filtered: int = Field(default=0, description="중복으로 필터링된 문서 수")
    started_at: datetime = Field(default_factory=datetime.now, description="시작 시간")
    
    @computed_field
    @property
    def dedup_ratio(self) -> float:
        """중복 제거 비율 (0.0 ~ 1.0)"""
        if self.total_processed == 0:
            return 0.0
        return self.duplicates_filtered / self.total_processed
    
    @computed_field
    @property
    def unique_ratio(self) -> float:
        """유니크 비율 (0.0 ~ 1.0)"""
        if self.total_processed == 0:
            return 0.0
        return self.unique_added / self.total_processed
    
    @computed_field
    @property
    def storage_saved_percent(self) -> float:
        """저장소 절약 비율 (%)"""
        return self.dedup_ratio * 100
    
    def update(self, is_duplicate: bool) -> None:
        """통계 업데이트"""
        self.total_processed += 1
        if is_duplicate:
            self.duplicates_filtered += 1
        else:
            self.unique_added += 1


class DeduplicationResult(BaseModel):
    """중복 제거 결과 모델"""
    
    is_added: bool = Field(..., description="추가 여부")
    reason: str = Field(..., description="결과 사유")
    doc_id: str | None = Field(default=None, description="문서 ID (추가된 경우)")
    similar_doc: SearchResult | None = Field(default=None, description="유사 문서 (중복인 경우)")
    similarity: float | None = Field(default=None, description="유사도 (중복인 경우)")


# =============================================================================
# SEMANTIC DEDUPLICATOR
# =============================================================================


class SemanticDeduplicator:
    """
    의미 기반 중복 제거 서비스
    
    기존 VectorStore의 유사도 검색을 활용하여 이미 존재하는 유사 문서 필터링
    
    Blueprint Week 3 요구사항:
    - 유사도 0.95 이상 문서 중복으로 판정
    - 같은 키워드 내에서만 중복 검사
    - EmbeddingService (CPU) 사용
    
    사용 예시:
        from trendops.service.deduplicator import get_deduplicator
        
        dedup = get_deduplicator()
        is_added, reason = await dedup.add_if_unique(
            text="트럼프가 관세 정책을 발표했다.",
            metadata={"title": "트럼프 관세", "keyword": "트럼프", "source": "google_news"}
        )
        
        # 통계 확인
        stats = dedup.get_stats()
        print(f"중복 제거율: {stats.dedup_ratio:.1%}")
    """
    
    # 기본 유사도 임계값 (95%)
    SIMILARITY_THRESHOLD: float = 0.95
    
    _instance: SemanticDeduplicator | None = None
    _lock: threading.Lock = threading.Lock()
    _initialized: bool = False
    
    def __new__(cls) -> SemanticDeduplicator:
        """Thread-safe Singleton 패턴"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self) -> None:
        """Deduplicator 초기화"""
        if SemanticDeduplicator._initialized:
            return
        
        with SemanticDeduplicator._lock:
            if SemanticDeduplicator._initialized:
                return
            
            self._settings = get_settings()
            self._embedding_service: EmbeddingService | None = None
            self._vector_store: VectorStore | None = None
            self._stats = DeduplicationStats()
            
            # 설정값에서 임계값 로드 (있으면)
            if hasattr(self._settings, 'dedup_similarity_threshold'):
                self.SIMILARITY_THRESHOLD = self._settings.dedup_similarity_threshold
            
            SemanticDeduplicator._initialized = True
            
            logger.info(
                "SemanticDeduplicator initialized",
                extra={"threshold": self.SIMILARITY_THRESHOLD}
            )
    
    @property
    def embedding_service(self) -> EmbeddingService:
        """Lazy loading for EmbeddingService"""
        if self._embedding_service is None:
            self._embedding_service = get_embedding_service()
        return self._embedding_service
    
    @property
    def vector_store(self) -> VectorStore:
        """Lazy loading for VectorStore"""
        if self._vector_store is None:
            self._vector_store = get_vector_store()
        return self._vector_store
    
    def _compute_embedding(self, text: str) -> list[float]:
        """
        텍스트 임베딩 계산 (CPU 전용)
        
        Args:
            text: 임베딩할 텍스트
            
        Returns:
            list[float]: 임베딩 벡터
        """
        embedding = self.embedding_service.embed(text)
        return embedding.tolist()
    
    def _find_similar(
        self,
        embedding: list[float],
        keyword: str,
    ) -> SearchResult | None:
        """
        키워드 내 유사 문서 검색
        
        Args:
            embedding: 쿼리 임베딩
            keyword: 검색 키워드 (동일 키워드 내에서만 검색)
            
        Returns:
            SearchResult | None: 임계값 이상 유사 문서 (없으면 None)
        """
        # 기존 VectorStore의 search_by_keyword 활용
        results = self.vector_store.search_by_keyword(
            query_embedding=embedding,
            keyword=keyword,
            top_k=1,
        )
        
        # 유사도 임계값 필터링
        if results and results[0].similarity >= self.SIMILARITY_THRESHOLD:
            return results[0]
        
        return None
    
    async def add_if_unique(
        self,
        text: str,
        metadata: dict[str, Any],
    ) -> tuple[bool, str]:
        """
        유니크한 경우에만 문서 추가
        
        이미 유사한 문서가 존재하면 추가하지 않음.
        같은 키워드 내에서만 중복 검사.
        
        Args:
            text: 문서 텍스트
            metadata: 메타데이터 (keyword, title 필수)
            
        Returns:
            tuple[bool, str]: (추가 여부, 사유)
            
        Raises:
            ValueError: keyword가 metadata에 없는 경우
        """
        # 키워드 확인
        keyword = metadata.get("keyword")
        if not keyword:
            raise ValueError("metadata must contain 'keyword' field")
        
        # 텍스트 검증
        if not text or not text.strip():
            self._stats.update(is_duplicate=False)
            return False, "Empty text"
        
        # 임베딩 계산 (CPU)
        embedding = self._compute_embedding(text)
        
        # 같은 키워드 내에서 유사 문서 검색
        similar_doc = self._find_similar(embedding, keyword)
        
        # 중복 발견
        if similar_doc:
            self._stats.update(is_duplicate=True)
            
            logger.debug(
                "Duplicate detected",
                extra={
                    "keyword": keyword,
                    "similarity": f"{similar_doc.similarity:.2%}",
                    "similar_title": similar_doc.title[:50] if similar_doc.title else "",
                }
            )
            
            return False, f"Duplicate (similarity: {similar_doc.similarity:.2%})"
        
        # 유니크한 문서 - VectorStore에 추가
        result = self.vector_store.add_documents(
            contents=[text],
            embeddings=[embedding],
            metadatas=[metadata],
            skip_duplicates=True,  # ID 기반 중복도 체크
        )
        
        self._stats.update(is_duplicate=False)
        
        doc_id = result.doc_ids[0] if result.doc_ids else "unknown"
        
        logger.debug(
            "Unique document added",
            extra={
                "doc_id": doc_id[:8] + "...",
                "keyword": keyword,
                "text_length": len(text),
            }
        )
        
        return True, "Added"
    
    async def add_if_unique_detailed(
        self,
        text: str,
        metadata: dict[str, Any],
    ) -> DeduplicationResult:
        """
        유니크한 경우에만 문서 추가 (상세 결과 반환)
        
        Args:
            text: 문서 텍스트
            metadata: 메타데이터 (keyword, title 필수)
            
        Returns:
            DeduplicationResult: 상세 결과
        """
        keyword = metadata.get("keyword")
        if not keyword:
            raise ValueError("metadata must contain 'keyword' field")
        
        if not text or not text.strip():
            self._stats.update(is_duplicate=False)
            return DeduplicationResult(
                is_added=False,
                reason="Empty text",
            )
        
        # 임베딩 계산
        embedding = self._compute_embedding(text)
        
        # 유사 문서 검색
        similar_doc = self._find_similar(embedding, keyword)
        
        # 중복 발견
        if similar_doc:
            self._stats.update(is_duplicate=True)
            
            return DeduplicationResult(
                is_added=False,
                reason=f"Duplicate (similarity: {similar_doc.similarity:.2%})",
                similar_doc=similar_doc,
                similarity=similar_doc.similarity,
            )
        
        # 유니크 - 추가
        result = self.vector_store.add_documents(
            contents=[text],
            embeddings=[embedding],
            metadatas=[metadata],
            skip_duplicates=True,
        )
        
        self._stats.update(is_duplicate=False)
        
        doc_id = result.doc_ids[0] if result.doc_ids else None
        
        return DeduplicationResult(
            is_added=True,
            reason="Added",
            doc_id=doc_id,
        )
    
    async def check_duplicate(
        self,
        text: str,
        keyword: str,
    ) -> tuple[bool, float | None]:
        """
        중복 여부만 확인 (추가하지 않음)
        
        Args:
            text: 확인할 텍스트
            keyword: 검색 키워드
            
        Returns:
            tuple[bool, float | None]: (중복 여부, 유사도)
        """
        if not text or not text.strip():
            return False, None
        
        embedding = self._compute_embedding(text)
        similar_doc = self._find_similar(embedding, keyword)
        
        if similar_doc:
            return True, similar_doc.similarity
        
        return False, None
    
    async def add_batch_unique(
        self,
        items: list[tuple[str, dict[str, Any]]],
    ) -> list[DeduplicationResult]:
        """
        배치 중복 제거 추가
        
        Args:
            items: (text, metadata) 튜플 리스트
            
        Returns:
            list[DeduplicationResult]: 각 항목의 결과
        """
        results: list[DeduplicationResult] = []
        
        for text, metadata in items:
            result = await self.add_if_unique_detailed(text, metadata)
            results.append(result)
        
        # 배치 결과 로깅
        added = sum(1 for r in results if r.is_added)
        duplicates = len(results) - added
        
        logger.info(
            "Batch deduplication complete",
            extra={
                "total": len(results),
                "added": added,
                "duplicates": duplicates,
                "dedup_ratio": f"{duplicates / len(results):.1%}" if results else "0%",
            }
        )
        
        return results
    
    def get_stats(self) -> DeduplicationStats:
        """
        중복 제거 통계 반환
        
        Returns:
            DeduplicationStats: 누적 통계
        """
        return self._stats.model_copy()
    
    def reset_stats(self) -> None:
        """통계 초기화"""
        self._stats = DeduplicationStats()
        logger.info("Deduplication stats reset")


# =============================================================================
# GLOBAL INSTANCE & FACTORY
# =============================================================================


_global_instance: SemanticDeduplicator | None = None
_global_lock = threading.Lock()


def get_deduplicator() -> SemanticDeduplicator:
    """
    전역 SemanticDeduplicator 인스턴스 반환
    
    Returns:
        SemanticDeduplicator: 싱글톤 인스턴스
        
    Usage:
        from trendops.service.deduplicator import get_deduplicator
        
        dedup = get_deduplicator()
        is_added, reason = await dedup.add_if_unique(
            text="뉴스 내용",
            metadata={"keyword": "트럼프", "title": "기사 제목"}
        )
    """
    global _global_instance
    
    if _global_instance is None:
        with _global_lock:
            if _global_instance is None:
                _global_instance = SemanticDeduplicator()
    
    return _global_instance


def reset_deduplicator() -> None:
    """
    전역 인스턴스 초기화 (테스트용)
    
    ⚠️ 주의: 프로덕션에서는 사용 금지
    """
    global _global_instance
    
    with _global_lock:
        _global_instance = None
        SemanticDeduplicator._instance = None
        SemanticDeduplicator._initialized = False
    
    logger.warning("SemanticDeduplicator global instance reset")


# =============================================================================
# CLI TEST
# =============================================================================


if __name__ == "__main__":
    import asyncio
    
    async def main():
        print("=" * 70)
        print("  SemanticDeduplicator Test")
        print("=" * 70)
        
        # 인스턴스 생성
        print("\n[1] Creating SemanticDeduplicator...")
        dedup = get_deduplicator()
        print(f"    ✓ Threshold: {dedup.SIMILARITY_THRESHOLD:.0%}")
        
        # 테스트 키워드
        keyword = "__test_dedup__"
        
        # 테스트 데이터
        test_articles = [
            ("트럼프 대통령이 중국산 제품에 25% 관세를 부과한다고 발표했다.", 
             {"keyword": keyword, "title": "트럼프 관세 발표", "source": "test"}),
            ("트럼프가 중국 제품에 25% 관세 부과를 선언했다.",  # 거의 동일
             {"keyword": keyword, "title": "트럼프 관세 선언", "source": "test"}),
            ("삼성전자 주가가 10% 급등했다.",  # 다른 주제
             {"keyword": keyword, "title": "삼성전자 주가", "source": "test"}),
            ("삼성전자 주식이 10% 상승세를 기록했다.",  # 거의 동일
             {"keyword": keyword, "title": "삼성전자 상승", "source": "test"}),
        ]
        
        # 순차 추가 테스트
        print("\n[2] Adding articles sequentially...")
        for i, (text, meta) in enumerate(test_articles, 1):
            is_added, reason = await dedup.add_if_unique(text=text, metadata=meta)
            status = "✓ Added" if is_added else "✗ Skip"
            print(f"    [{i}] {status}: {reason}")
            print(f"        Text: {text[:50]}...")
        
        # 통계 확인
        print("\n[3] Deduplication stats...")
        stats = dedup.get_stats()
        print(f"    ✓ Total processed: {stats.total_processed}")
        print(f"    ✓ Unique added: {stats.unique_added}")
        print(f"    ✓ Duplicates filtered: {stats.duplicates_filtered}")
        print(f"    ✓ Dedup ratio: {stats.dedup_ratio:.1%}")
        print(f"    ✓ Storage saved: {stats.storage_saved_percent:.1f}%")
        
        # 중복 체크 테스트 (추가 없이)
        print("\n[4] Check duplicate (without adding)...")
        test_text = "트럼프가 중국에 관세를 부과했다."
        is_dup, similarity = await dedup.check_duplicate(test_text, keyword)
        print(f"    Text: {test_text}")
        print(f"    Is duplicate: {is_dup}")
        if similarity:
            print(f"    Similarity: {similarity:.2%}")
        
        # 클린업
        print("\n[5] Cleanup...")
        store = dedup.vector_store
        deleted = store.delete_by_keyword(keyword)
        print(f"    ✓ Deleted {deleted} test documents")
        
        print("\n" + "=" * 70)
        print("✅ SemanticDeduplicator test completed!")
        print("=" * 70)
    
    asyncio.run(main())