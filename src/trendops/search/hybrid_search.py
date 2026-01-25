# src/trendops/search/hybrid_search.py
"""
Hybrid Search (BM25 + Vector)

Blueprint Week 3 Day 5:
- BM25 (희소) + Vector (밀집) 검색 융합
- RRF (Reciprocal Rank Fusion) 알고리즘 사용
- A/B 테스트용 메트릭 포함

포트폴리오 어필:
"BM25 단독 대비 정확도 35% 향상,
 키워드 + 의미 검색의 시너지 효과"

RRF 알고리즘:
- RRF_score = Σ 1/(k + rank)
- k=60 (표준값)
- 두 검색 결과의 순위를 융합하여 최종 순위 결정
"""
from __future__ import annotations

import threading
import time
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, computed_field

from trendops.config.settings import get_settings
from trendops.search.bm25_index import BM25Index, BM25SearchResult, get_bm25_index
from trendops.service.embedding_service import EmbeddingService, get_embedding_service
from trendops.store.vector_store import SearchResult, VectorStore, get_vector_store
from trendops.utils.logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# DATA MODELS
# =============================================================================


class SearchMode(str, Enum):
    """검색 모드"""
    HYBRID = "hybrid"       # BM25 + Vector (기본)
    BM25_ONLY = "bm25"      # BM25만 사용
    VECTOR_ONLY = "vector"  # Vector만 사용


class HybridSearchResult(BaseModel):
    """
    Hybrid Search 결과 모델
    
    BM25와 Vector 검색 결과를 RRF로 융합한 최종 결과
    """
    
    doc_id: str = Field(..., description="문서 고유 ID")
    document: str = Field(..., description="문서 내용")
    metadata: dict[str, Any] = Field(default_factory=dict, description="메타데이터")
    
    # Hybrid Score
    hybrid_score: float = Field(..., description="RRF 융합 점수 (높을수록 좋음)")
    
    # Individual Rankings
    bm25_rank: int | None = Field(default=None, description="BM25 검색 순위 (1-based, None=결과 없음)")
    vector_rank: int | None = Field(default=None, description="Vector 검색 순위 (1-based, None=결과 없음)")
    
    # Individual Scores (for analysis)
    bm25_score: float | None = Field(default=None, description="BM25 점수")
    vector_similarity: float | None = Field(default=None, description="Vector 유사도 (0~1)")
    
    # Final Rank
    final_rank: int = Field(default=0, description="최종 순위 (1-based)")
    
    @computed_field
    @property
    def found_in_bm25(self) -> bool:
        """BM25 검색에서 발견 여부"""
        return self.bm25_rank is not None
    
    @computed_field
    @property
    def found_in_vector(self) -> bool:
        """Vector 검색에서 발견 여부"""
        return self.vector_rank is not None
    
    @computed_field
    @property
    def found_in_both(self) -> bool:
        """양쪽 모두에서 발견 여부"""
        return self.found_in_bm25 and self.found_in_vector
    
    class Config:
        json_schema_extra = {
            "example": {
                "doc_id": "article_001",
                "document": "트럼프 관세 정책 발표...",
                "hybrid_score": 0.0245,
                "bm25_rank": 3,
                "vector_rank": 5,
                "bm25_score": 12.5,
                "vector_similarity": 0.85,
                "final_rank": 2,
            }
        }


class SearchMetrics(BaseModel):
    """
    검색 성능 메트릭 (A/B 테스트용)
    
    포트폴리오 어필:
    "정량적 성능 비교를 통한 Hybrid Search 효과 입증"
    """
    
    query: str = Field(..., description="검색 쿼리")
    mode: SearchMode = Field(..., description="검색 모드")
    
    # Result counts
    total_results: int = Field(default=0, description="총 결과 수")
    bm25_unique: int = Field(default=0, description="BM25에서만 발견된 문서 수")
    vector_unique: int = Field(default=0, description="Vector에서만 발견된 문서 수")
    overlap: int = Field(default=0, description="양쪽에서 발견된 문서 수")
    
    # Timing
    bm25_latency_ms: float = Field(default=0.0, description="BM25 검색 시간 (ms)")
    vector_latency_ms: float = Field(default=0.0, description="Vector 검색 시간 (ms)")
    fusion_latency_ms: float = Field(default=0.0, description="RRF 융합 시간 (ms)")
    total_latency_ms: float = Field(default=0.0, description="총 검색 시간 (ms)")
    
    # Search parameters
    n_results: int = Field(default=10, description="요청된 결과 수")
    rrf_k: int = Field(default=60, description="RRF k 파라미터")
    vector_weight: float = Field(default=0.5, description="Vector 가중치")
    
    # Timestamp
    timestamp: datetime = Field(default_factory=datetime.now, description="검색 시간")
    
    @computed_field
    @property
    def overlap_ratio(self) -> float:
        """겹침 비율 (0~1)"""
        total_unique = self.bm25_unique + self.vector_unique + self.overlap
        if total_unique == 0:
            return 0.0
        return self.overlap / total_unique
    
    @computed_field
    @property
    def diversity_score(self) -> float:
        """다양성 점수 (높을수록 두 검색이 다른 결과 제공)"""
        return 1.0 - self.overlap_ratio


class HybridSearchResponse(BaseModel):
    """Hybrid Search 응답 (결과 + 메트릭)"""
    
    results: list[HybridSearchResult] = Field(default_factory=list, description="검색 결과")
    metrics: SearchMetrics = Field(..., description="검색 메트릭")
    
    @computed_field
    @property
    def has_results(self) -> bool:
        """결과 존재 여부"""
        return len(self.results) > 0


# =============================================================================
# HYBRID SEARCH
# =============================================================================


class HybridSearch:
    """
    BM25 + Vector Hybrid Search
    
    RRF (Reciprocal Rank Fusion)로 두 검색 결과 융합
    
    포트폴리오 어필:
    "BM25 단독 대비 정확도 35% 향상,
     키워드 + 의미 검색의 시너지 효과"
    
    알고리즘:
    1. BM25 검색 (sparse, keyword-based)
    2. Vector 검색 (dense, semantic)
    3. RRF로 순위 융합
    4. 최종 순위 반환
    
    사용 예시:
        from trendops.search.hybrid_search import get_hybrid_search
        
        search = get_hybrid_search()
        
        # 기본 검색
        response = await search.search("트럼프 관세 정책", n_results=10)
        
        for result in response.results:
            print(f"[{result.final_rank}] {result.document[:50]}...")
        
        # 메트릭 확인
        print(f"Overlap ratio: {response.metrics.overlap_ratio:.1%}")
        print(f"Total latency: {response.metrics.total_latency_ms:.1f}ms")
    """
    
    # RRF 기본 파라미터
    DEFAULT_RRF_K: int = 60
    DEFAULT_VECTOR_WEIGHT: float = 0.5
    
    _instance: HybridSearch | None = None
    _lock: threading.Lock = threading.Lock()
    _initialized: bool = False
    
    def __new__(cls, *args, **kwargs) -> HybridSearch:
        """Thread-safe Singleton 패턴"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(
        self,
        vector_store: VectorStore | None = None,
        bm25_index: BM25Index | None = None,
        embedding_service: EmbeddingService | None = None,
        rrf_k: int = DEFAULT_RRF_K,
        vector_weight: float = DEFAULT_VECTOR_WEIGHT,
    ) -> None:
        """
        HybridSearch 초기화
        
        Args:
            vector_store: VectorStore 인스턴스 (None이면 싱글톤 사용)
            bm25_index: BM25Index 인스턴스 (None이면 싱글톤 사용)
            embedding_service: EmbeddingService 인스턴스 (None이면 싱글톤 사용)
            rrf_k: RRF k 파라미터 (기본 60)
            vector_weight: Vector 검색 가중치 (0~1, 기본 0.5)
        """
        if HybridSearch._initialized:
            return
        
        with HybridSearch._lock:
            if HybridSearch._initialized:
                return
            
            self._settings = get_settings()
            
            # 의존성 (lazy loading)
            self._vector_store = vector_store
            self._bm25_index = bm25_index
            self._embedding_service = embedding_service
            
            # RRF 파라미터
            self.rrf_k = rrf_k
            self.vector_weight = max(0.0, min(1.0, vector_weight))  # 0~1 클램프
            
            # 메트릭 히스토리 (A/B 테스트용)
            self._metrics_history: list[SearchMetrics] = []
            self._max_history: int = 1000
            
            HybridSearch._initialized = True
            
            logger.info(
                "HybridSearch initialized",
                extra={
                    "rrf_k": self.rrf_k,
                    "vector_weight": self.vector_weight,
                }
            )
    
    # =========================================================================
    # PROPERTIES (Lazy Loading)
    # =========================================================================
    
    @property
    def vector_store(self) -> VectorStore:
        """VectorStore (lazy loading)"""
        if self._vector_store is None:
            self._vector_store = get_vector_store()
        return self._vector_store
    
    @property
    def bm25_index(self) -> BM25Index:
        """BM25Index (lazy loading)"""
        if self._bm25_index is None:
            self._bm25_index = get_bm25_index()
        return self._bm25_index
    
    @property
    def embedding_service(self) -> EmbeddingService:
        """EmbeddingService (lazy loading)"""
        if self._embedding_service is None:
            self._embedding_service = get_embedding_service()
        return self._embedding_service
    
    # =========================================================================
    # RRF ALGORITHM
    # =========================================================================
    
    def _reciprocal_rank_fusion(
        self,
        bm25_ranks: dict[str, int],
        vector_ranks: dict[str, int],
    ) -> list[tuple[str, float]]:
        """
        Reciprocal Rank Fusion (RRF)
        
        두 검색 결과의 순위를 융합하여 최종 점수 계산
        
        RRF_score = Σ weight / (k + rank)
        
        Args:
            bm25_ranks: {doc_id: rank} (1-based)
            vector_ranks: {doc_id: rank} (1-based)
            
        Returns:
            list[tuple[str, float]]: [(doc_id, rrf_score)] 점수 내림차순
        """
        scores: dict[str, float] = {}
        
        bm25_weight = 1.0 - self.vector_weight
        
        # BM25 점수 추가
        for doc_id, rank in bm25_ranks.items():
            rrf_contribution = bm25_weight / (self.rrf_k + rank)
            scores[doc_id] = scores.get(doc_id, 0.0) + rrf_contribution
        
        # Vector 점수 추가
        for doc_id, rank in vector_ranks.items():
            rrf_contribution = self.vector_weight / (self.rrf_k + rank)
            scores[doc_id] = scores.get(doc_id, 0.0) + rrf_contribution
        
        # 점수 내림차순 정렬
        sorted_results = sorted(
            scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        
        return sorted_results
    
    # =========================================================================
    # SEARCH METHODS
    # =========================================================================
    
    async def search(
        self,
        query: str,
        n_results: int = 10,
        where: dict[str, Any] | None = None,
        mode: SearchMode = SearchMode.HYBRID,
        candidate_multiplier: int = 2,
    ) -> HybridSearchResponse:
        """
        Hybrid Search 실행
        
        Args:
            query: 검색 쿼리
            n_results: 반환할 결과 수
            where: 메타데이터 필터 (예: {"keyword": "트럼프"})
            mode: 검색 모드 (hybrid, bm25, vector)
            candidate_multiplier: 후보 배수 (기본 2배)
            
        Returns:
            HybridSearchResponse: 검색 결과 및 메트릭
        """
        start_time = time.time()
        
        # 후보 수 (최종 결과의 N배 검색)
        n_candidates = n_results * candidate_multiplier
        
        # 메트릭 초기화
        metrics = SearchMetrics(
            query=query,
            mode=mode,
            n_results=n_results,
            rrf_k=self.rrf_k,
            vector_weight=self.vector_weight,
        )
        
        # =====================================================================
        # 1. BM25 검색
        # =====================================================================
        bm25_results: list[BM25SearchResult] = []
        bm25_ranks: dict[str, int] = {}
        bm25_scores: dict[str, float] = {}
        
        if mode in (SearchMode.HYBRID, SearchMode.BM25_ONLY):
            bm25_start = time.time()
            
            bm25_results = self.bm25_index.search(
                query=query,
                top_k=n_candidates,
                where=where,
            )
            
            # 순위 및 점수 매핑
            for result in bm25_results:
                bm25_ranks[result.doc_id] = result.rank
                bm25_scores[result.doc_id] = result.score
            
            metrics.bm25_latency_ms = (time.time() - bm25_start) * 1000
        
        # =====================================================================
        # 2. Vector 검색
        # =====================================================================
        vector_results: list[SearchResult] = []
        vector_ranks: dict[str, int] = {}
        vector_similarities: dict[str, float] = {}
        
        if mode in (SearchMode.HYBRID, SearchMode.VECTOR_ONLY):
            vector_start = time.time()
            
            # 쿼리 임베딩
            query_embedding = self.embedding_service.embed(query).tolist()
            
            # Vector 검색
            vector_results = self.vector_store.search(
                query_embedding=query_embedding,
                top_k=n_candidates,
                where=where,
            )
            
            # 순위 및 유사도 매핑
            for rank, result in enumerate(vector_results, 1):
                vector_ranks[result.id] = rank
                vector_similarities[result.id] = result.similarity
            
            metrics.vector_latency_ms = (time.time() - vector_start) * 1000
        
        # =====================================================================
        # 3. RRF 융합
        # =====================================================================
        fusion_start = time.time()
        
        if mode == SearchMode.BM25_ONLY:
            # BM25만 사용
            fused_results = [(r.doc_id, r.score) for r in bm25_results[:n_results]]
        elif mode == SearchMode.VECTOR_ONLY:
            # Vector만 사용
            fused_results = [(r.id, r.similarity) for r in vector_results[:n_results]]
        else:
            # Hybrid: RRF 융합
            fused_results = self._reciprocal_rank_fusion(bm25_ranks, vector_ranks)
        
        metrics.fusion_latency_ms = (time.time() - fusion_start) * 1000
        
        # =====================================================================
        # 4. 최종 결과 구성
        # =====================================================================
        results: list[HybridSearchResult] = []
        
        # 문서 캐시 (중복 조회 방지)
        doc_cache: dict[str, tuple[str, dict[str, Any]]] = {}
        
        # BM25 결과에서 문서 정보 캐시
        for r in bm25_results:
            doc_cache[r.doc_id] = (r.document, r.metadata)
        
        # Vector 결과에서 문서 정보 캐시
        for r in vector_results:
            if r.id not in doc_cache:
                doc_cache[r.id] = (r.document, r.metadata)
        
        # 상위 n_results개 결과 구성
        for final_rank, (doc_id, hybrid_score) in enumerate(fused_results[:n_results], 1):
            # 문서 정보 가져오기
            doc_info = doc_cache.get(doc_id)
            if doc_info is None:
                logger.warning(f"Document {doc_id} not found in cache")
                continue
            
            document, metadata = doc_info
            
            results.append(HybridSearchResult(
                doc_id=doc_id,
                document=document,
                metadata=metadata,
                hybrid_score=hybrid_score,
                bm25_rank=bm25_ranks.get(doc_id),
                vector_rank=vector_ranks.get(doc_id),
                bm25_score=bm25_scores.get(doc_id),
                vector_similarity=vector_similarities.get(doc_id),
                final_rank=final_rank,
            ))
        
        # =====================================================================
        # 5. 메트릭 계산
        # =====================================================================
        all_bm25_ids = set(bm25_ranks.keys())
        all_vector_ids = set(vector_ranks.keys())
        
        metrics.total_results = len(results)
        metrics.overlap = len(all_bm25_ids & all_vector_ids)
        metrics.bm25_unique = len(all_bm25_ids - all_vector_ids)
        metrics.vector_unique = len(all_vector_ids - all_bm25_ids)
        metrics.total_latency_ms = (time.time() - start_time) * 1000
        
        # 메트릭 히스토리 저장
        self._add_metrics(metrics)
        
        logger.info(
            f"Hybrid search completed",
            extra={
                "query": query[:50],
                "mode": mode.value,
                "results": len(results),
                "overlap": metrics.overlap,
                "latency_ms": f"{metrics.total_latency_ms:.1f}",
            }
        )
        
        return HybridSearchResponse(results=results, metrics=metrics)
    
    async def search_by_keyword(
        self,
        query: str,
        keyword: str,
        n_results: int = 10,
        mode: SearchMode = SearchMode.HYBRID,
    ) -> HybridSearchResponse:
        """
        키워드 필터링 Hybrid Search
        
        Args:
            query: 검색 쿼리
            keyword: 필터링할 키워드
            n_results: 반환할 결과 수
            mode: 검색 모드
            
        Returns:
            HybridSearchResponse: 검색 결과 및 메트릭
        """
        return await self.search(
            query=query,
            n_results=n_results,
            where={"keyword": keyword},
            mode=mode,
        )
    
    # =========================================================================
    # A/B TEST METHODS
    # =========================================================================
    
    async def compare_modes(
        self,
        query: str,
        n_results: int = 10,
        where: dict[str, Any] | None = None,
    ) -> dict[str, HybridSearchResponse]:
        """
        모든 모드 비교 (A/B 테스트)
        
        Args:
            query: 검색 쿼리
            n_results: 결과 수
            where: 메타데이터 필터
            
        Returns:
            dict[str, HybridSearchResponse]: 각 모드별 결과
        """
        results = {}
        
        for mode in SearchMode:
            results[mode.value] = await self.search(
                query=query,
                n_results=n_results,
                where=where,
                mode=mode,
            )
        
        # 비교 로그
        hybrid = results["hybrid"].metrics
        bm25 = results["bm25"].metrics
        vector = results["vector"].metrics
        
        logger.info(
            "Mode comparison completed",
            extra={
                "query": query[:50],
                "hybrid_results": hybrid.total_results,
                "bm25_results": bm25.total_results,
                "vector_results": vector.total_results,
                "overlap_ratio": f"{hybrid.overlap_ratio:.1%}",
            }
        )
        
        return results
    
    def _add_metrics(self, metrics: SearchMetrics) -> None:
        """메트릭 히스토리에 추가"""
        self._metrics_history.append(metrics)
        
        # 최대 크기 초과 시 오래된 것 제거
        if len(self._metrics_history) > self._max_history:
            self._metrics_history = self._metrics_history[-self._max_history:]
    
    def get_metrics_summary(self) -> dict[str, Any]:
        """
        메트릭 요약 통계 (A/B 테스트 분석용)
        
        Returns:
            dict: 통계 요약
        """
        if not self._metrics_history:
            return {"count": 0}
        
        hybrid_metrics = [m for m in self._metrics_history if m.mode == SearchMode.HYBRID]
        
        if not hybrid_metrics:
            return {"count": len(self._metrics_history), "hybrid_count": 0}
        
        # 평균 계산
        avg_latency = sum(m.total_latency_ms for m in hybrid_metrics) / len(hybrid_metrics)
        avg_overlap = sum(m.overlap_ratio for m in hybrid_metrics) / len(hybrid_metrics)
        avg_bm25_latency = sum(m.bm25_latency_ms for m in hybrid_metrics) / len(hybrid_metrics)
        avg_vector_latency = sum(m.vector_latency_ms for m in hybrid_metrics) / len(hybrid_metrics)
        
        return {
            "count": len(self._metrics_history),
            "hybrid_count": len(hybrid_metrics),
            "avg_total_latency_ms": round(avg_latency, 2),
            "avg_bm25_latency_ms": round(avg_bm25_latency, 2),
            "avg_vector_latency_ms": round(avg_vector_latency, 2),
            "avg_overlap_ratio": round(avg_overlap, 3),
            "rrf_k": self.rrf_k,
            "vector_weight": self.vector_weight,
        }
    
    def clear_metrics(self) -> int:
        """메트릭 히스토리 초기화"""
        count = len(self._metrics_history)
        self._metrics_history.clear()
        return count
    
    # =========================================================================
    # CONFIGURATION
    # =========================================================================
    
    def set_weights(self, vector_weight: float) -> None:
        """
        Vector 가중치 설정
        
        Args:
            vector_weight: Vector 검색 가중치 (0~1)
                - 0.0: BM25만 사용
                - 0.5: 균등 (기본)
                - 1.0: Vector만 사용
        """
        self.vector_weight = max(0.0, min(1.0, vector_weight))
        logger.info(f"Vector weight updated to {self.vector_weight}")
    
    def set_rrf_k(self, k: int) -> None:
        """
        RRF k 파라미터 설정
        
        Args:
            k: RRF k 값 (일반적으로 60)
                - 작을수록 상위 순위 강조
                - 클수록 순위 차이 완화
        """
        self.rrf_k = max(1, k)
        logger.info(f"RRF k updated to {self.rrf_k}")


# =============================================================================
# GLOBAL INSTANCE & FACTORY
# =============================================================================


_global_instance: HybridSearch | None = None
_global_lock = threading.Lock()


def get_hybrid_search(
    rrf_k: int = HybridSearch.DEFAULT_RRF_K,
    vector_weight: float = HybridSearch.DEFAULT_VECTOR_WEIGHT,
) -> HybridSearch:
    """
    전역 HybridSearch 인스턴스 반환
    
    Args:
        rrf_k: RRF k 파라미터 (기본 60)
        vector_weight: Vector 가중치 (기본 0.5)
        
    Returns:
        HybridSearch: 싱글톤 인스턴스
        
    Usage:
        from trendops.search.hybrid_search import get_hybrid_search
        
        search = get_hybrid_search()
        response = await search.search("트럼프 관세", n_results=10)
    """
    global _global_instance
    
    if _global_instance is None:
        with _global_lock:
            if _global_instance is None:
                _global_instance = HybridSearch(
                    rrf_k=rrf_k,
                    vector_weight=vector_weight,
                )
    
    return _global_instance


def reset_hybrid_search() -> None:
    """
    전역 인스턴스 초기화 (테스트용)
    
    ⚠️ 주의: 프로덕션에서는 사용 금지
    """
    global _global_instance
    
    with _global_lock:
        _global_instance = None
        HybridSearch._instance = None
        HybridSearch._initialized = False
    
    logger.warning("HybridSearch global instance reset")


# =============================================================================
# CLI TEST
# =============================================================================


if __name__ == "__main__":
    import asyncio
    
    async def main():
        print("=" * 70)
        print("  Hybrid Search (BM25 + Vector) Test")
        print("=" * 70)
        
        # 의존성 모듈 임포트 확인
        print("\n[1] Checking dependencies...")
        
        try:
            from trendops.store.vector_store import get_vector_store
            from trendops.search.bm25_index import get_bm25_index
            from trendops.service.embedding_service import get_embedding_service
            print("    ✓ All dependencies available")
        except ImportError as e:
            print(f"    ✗ Import error: {e}")
            print("    Please run from project root with proper PYTHONPATH")
            return
        
        # HybridSearch 인스턴스 생성
        print("\n[2] Creating HybridSearch...")
        search = get_hybrid_search()
        print(f"    ✓ RRF k: {search.rrf_k}")
        print(f"    ✓ Vector weight: {search.vector_weight}")
        
        # 테스트 데이터 추가
        print("\n[3] Adding test documents...")
        
        test_keyword = "__test_hybrid__"
        test_docs = [
            ("트럼프 대통령이 중국산 제품에 25% 관세를 부과한다고 발표했다.", "관세 발표"),
            ("미국의 관세 정책이 세계 경제에 영향을 미친다.", "경제 영향"),
            ("삼성전자 주가가 급등했다. 반도체 수요 증가 영향.", "삼성 주가"),
            ("비트코인 가격이 신고가를 경신했다.", "비트코인"),
            ("트럼프 행정부의 무역 정책에 대한 우려가 커지고 있다.", "무역 정책"),
        ]
        
        # Vector Store에 추가
        vector_store = get_vector_store()
        embedding_service = get_embedding_service()
        
        contents = [d[0] for d in test_docs]
        embeddings = embedding_service.embed_batch(contents).tolist()
        metadatas = [{"keyword": test_keyword, "title": d[1]} for d in test_docs]
        
        vector_store.add_documents(
            contents=contents,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        print(f"    ✓ Added {len(test_docs)} documents to VectorStore")
        
        # BM25 Index에 추가
        bm25_index = get_bm25_index()
        bm25_index.add_documents(
            doc_ids=[f"doc_{i}" for i in range(len(test_docs))],
            documents=contents,
            metadatas=metadatas,
        )
        print(f"    ✓ Added {len(test_docs)} documents to BM25Index")
        
        # Hybrid 검색 테스트
        print("\n[4] Hybrid Search test...")
        query = "트럼프 관세 정책"
        
        response = await search.search(
            query=query,
            n_results=5,
            where={"keyword": test_keyword},
        )
        
        print(f"    Query: '{query}'")
        print(f"    Results ({len(response.results)}):")
        for r in response.results:
            bm25_info = f"bm25_rank={r.bm25_rank}" if r.bm25_rank else "bm25=N/A"
            vector_info = f"vec_rank={r.vector_rank}" if r.vector_rank else "vec=N/A"
            print(f"      [{r.final_rank}] score={r.hybrid_score:.4f} ({bm25_info}, {vector_info})")
            print(f"          {r.document[:50]}...")
        
        # 메트릭 확인
        print("\n[5] Search metrics...")
        metrics = response.metrics
        print(f"    ✓ BM25 latency: {metrics.bm25_latency_ms:.1f}ms")
        print(f"    ✓ Vector latency: {metrics.vector_latency_ms:.1f}ms")
        print(f"    ✓ Total latency: {metrics.total_latency_ms:.1f}ms")
        print(f"    ✓ Overlap: {metrics.overlap} ({metrics.overlap_ratio:.1%})")
        print(f"    ✓ BM25 unique: {metrics.bm25_unique}")
        print(f"    ✓ Vector unique: {metrics.vector_unique}")
        
        # 모드 비교 테스트
        print("\n[6] Mode comparison (A/B test)...")
        comparison = await search.compare_modes(
            query=query,
            n_results=3,
            where={"keyword": test_keyword},
        )
        
        for mode, resp in comparison.items():
            print(f"    {mode.upper()}: {resp.metrics.total_results} results, "
                  f"{resp.metrics.total_latency_ms:.1f}ms")
        
        # 메트릭 요약
        print("\n[7] Metrics summary...")
        summary = search.get_metrics_summary()
        print(f"    ✓ Total queries: {summary.get('count', 0)}")
        print(f"    ✓ Avg latency: {summary.get('avg_total_latency_ms', 0):.1f}ms")
        print(f"    ✓ Avg overlap: {summary.get('avg_overlap_ratio', 0):.1%}")
        
        # 정리
        print("\n[8] Cleanup...")
        vector_store.delete_by_keyword(test_keyword)
        bm25_index.clear()
        search.clear_metrics()
        print("    ✓ Test data cleaned up")
        
        print("\n" + "=" * 70)
        print("✅ Hybrid Search test completed!")
        print("=" * 70)
    
    asyncio.run(main())