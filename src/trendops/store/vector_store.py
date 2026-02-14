# src/trendops/store/vector_store.py
"""
Vector Store (ChromaDB)

Blueprint Section 1.6 Resource Allocation:
- ChromaDB: Vector DB for similarity search
- Embedding은 CPU에서 실행 (EmbeddingService 활용)

⚠️ CRITICAL: 이 모듈은 Embedding을 직접 수행하지 않습니다.
            EmbeddingService를 통해 CPU에서 임베딩을 생성한 후 저장합니다.
"""
from __future__ import annotations

import hashlib
import threading
from datetime import datetime
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings
from pydantic import BaseModel, Field, computed_field

from trendops.config.settings import get_settings
from trendops.utils.logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# DATA MODELS
# =============================================================================


class SearchResult(BaseModel):
    """벡터 검색 결과 모델"""

    id: str = Field(..., description="문서 고유 ID")
    document: str = Field(..., description="문서 내용")
    metadata: dict[str, Any] = Field(default_factory=dict, description="메타데이터")
    distance: float = Field(..., description="거리 (낮을수록 유사)")

    @computed_field
    @property
    def similarity(self) -> float:
        """
        코사인 유사도 (0~1)

        ChromaDB cosine distance = 1 - cosine_similarity
        따라서 similarity = 1 - distance
        """
        return max(0.0, 1.0 - self.distance)

    @computed_field
    @property
    def keyword(self) -> str | None:
        """메타데이터에서 키워드 추출"""
        return self.metadata.get("keyword")

    @computed_field
    @property
    def title(self) -> str | None:
        """메타데이터에서 제목 추출"""
        return self.metadata.get("title")


class AddDocumentsResult(BaseModel):
    """문서 추가 결과"""

    total: int = Field(..., description="총 입력 문서 수")
    added: int = Field(..., description="추가된 문서 수")
    skipped: int = Field(default=0, description="스킵된 문서 수 (중복 등)")
    doc_ids: list[str] = Field(default_factory=list, description="추가된 문서 ID 목록")


class CollectionStats(BaseModel):
    """컬렉션 통계"""

    name: str = Field(..., description="컬렉션 이름")
    count: int = Field(..., description="총 문서 수")
    metadata: dict[str, Any] = Field(default_factory=dict, description="컬렉션 메타데이터")


# =============================================================================
# VECTOR STORE
# =============================================================================


class VectorStore:
    """
    ChromaDB 기반 Vector Store

    Blueprint Week 3 요구사항:
    - 문서 저장 및 유사도 검색
    - 키워드 기반 필터링 지원
    - Embedding은 외부에서 전달 (EmbeddingService 사용)

    사용 예시:
        from trendops.store.vector_store import get_vector_store
        from trendops.service.embedding_service import get_embedding_service

        store = get_vector_store()
        embed_service = get_embedding_service()

        # 문서 추가
        embeddings = embed_service.embed_batch(["문서1", "문서2"])
        store.add_documents(
            contents=["문서1", "문서2"],
            embeddings=embeddings.tolist(),
            metadatas=[{"keyword": "트럼프"}, {"keyword": "트럼프"}]
        )

        # 검색
        query_emb = embed_service.embed("관세 정책")
        results = store.search(query_emb.tolist(), top_k=5)
    """

    _instance: VectorStore | None = None
    _lock: threading.Lock = threading.Lock()
    _initialized: bool = False

    def __new__(cls) -> VectorStore:
        """Thread-safe Singleton 패턴"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """VectorStore 초기화"""
        if VectorStore._initialized:
            return

        with VectorStore._lock:
            if VectorStore._initialized:
                return

            self._settings = get_settings()
            self._client: chromadb.ClientAPI | None = None
            self._collection: chromadb.Collection | None = None

            # 초기화
            self._init_client()
            self._init_collection()

            VectorStore._initialized = True

            logger.info(
                "VectorStore initialized",
                extra={
                    "persist_path": str(self._settings.chromadb_path),
                    "collection": self._settings.chromadb_collection_name,
                },
            )

    def _init_client(self) -> None:
        """ChromaDB 클라이언트 초기화"""
        persist_path = self._settings.get_chromadb_path()

        self._client = chromadb.PersistentClient(
            path=str(persist_path),
            settings=ChromaSettings(
                anonymized_telemetry=False,
                allow_reset=True,
            ),
        )

        logger.debug(f"ChromaDB client initialized at {persist_path}")

    def _init_collection(self) -> None:
        """컬렉션 초기화 (없으면 생성)"""
        collection_name = self._settings.chromadb_collection_name

        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={
                "hnsw:space": "cosine",  # 코사인 거리 사용
                "created_at": datetime.now().isoformat(),
            },
        )

        count = self._collection.count()
        logger.debug(f"Collection '{collection_name}' ready with {count} documents")

    @property
    def collection(self) -> chromadb.Collection:
        """현재 컬렉션"""
        if self._collection is None:
            raise RuntimeError("VectorStore not properly initialized")
        return self._collection

    @property
    def client(self) -> chromadb.ClientAPI:
        """ChromaDB 클라이언트"""
        if self._client is None:
            raise RuntimeError("VectorStore not properly initialized")
        return self._client

    # =========================================================================
    # DOCUMENT OPERATIONS
    # =========================================================================

    @staticmethod
    def _generate_doc_id(content: str, metadata: dict[str, Any] | None = None) -> str:
        """
        문서 ID 생성 (결정적 해시)

        동일 내용 + 키워드 → 동일 ID (자연 중복 방지)
        """
        hash_input = content
        if metadata and "keyword" in metadata:
            hash_input = f"{metadata['keyword']}:{content}"

        return hashlib.sha256(hash_input.encode()).hexdigest()[:16]

    def add_documents(
        self,
        contents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]] | None = None,
        ids: list[str] | None = None,
        skip_duplicates: bool = True,
    ) -> AddDocumentsResult:
        """
        문서 배치 추가

        Args:
            contents: 문서 내용 리스트
            embeddings: 임베딩 벡터 리스트 (EmbeddingService에서 생성)
            metadatas: 메타데이터 리스트
            ids: 문서 ID 리스트 (None이면 자동 생성)
            skip_duplicates: 중복 ID 스킵 여부

        Returns:
            AddDocumentsResult: 추가 결과
        """
        if len(contents) != len(embeddings):
            raise ValueError(
                f"contents ({len(contents)}) and embeddings ({len(embeddings)}) "
                "must have same length"
            )

        # 메타데이터 기본값
        if metadatas is None:
            metadatas = [{} for _ in contents]
        elif len(metadatas) != len(contents):
            raise ValueError("metadatas must have same length as contents")

        # 타임스탬프 추가
        for meta in metadatas:
            if "added_at" not in meta:
                meta["added_at"] = datetime.now().isoformat()

        # ID 생성 또는 검증
        if ids is None:
            ids = [
                self._generate_doc_id(content, meta) for content, meta in zip(contents, metadatas)
            ]
        elif len(ids) != len(contents):
            raise ValueError("ids must have same length as contents")

        # 중복 체크 및 필터링
        if skip_duplicates:
            existing_ids = set(self._get_existing_ids(ids))

            filtered_data = [
                (doc_id, content, emb, meta)
                for doc_id, content, emb, meta in zip(ids, contents, embeddings, metadatas)
                if doc_id not in existing_ids
            ]

            if not filtered_data:
                logger.debug(f"All {len(ids)} documents already exist, skipping")
                return AddDocumentsResult(
                    total=len(contents),
                    added=0,
                    skipped=len(contents),
                    doc_ids=[],
                )

            # 필터링된 데이터 언패킹
            ids, contents, embeddings, metadatas = zip(*filtered_data)
            ids = list(ids)
            contents = list(contents)
            embeddings = list(embeddings)
            metadatas = list(metadatas)

            skipped = len(ids) - len(filtered_data) if filtered_data else len(ids)
        else:
            skipped = 0

        # ChromaDB에 추가
        self.collection.add(
            ids=ids,
            documents=contents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        logger.info(f"Added {len(ids)} documents", extra={"added": len(ids), "skipped": skipped})

        return AddDocumentsResult(
            total=len(contents) + skipped,
            added=len(ids),
            skipped=skipped,
            doc_ids=ids,
        )

    def _get_existing_ids(self, ids: list[str]) -> list[str]:
        """기존에 존재하는 ID 목록 반환"""
        try:
            result = self.collection.get(ids=ids, include=[])
            return result.get("ids", [])
        except Exception:
            return []

    # =========================================================================
    # SEARCH OPERATIONS
    # =========================================================================

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        where: dict[str, Any] | None = None,
        where_document: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """
        벡터 유사도 검색

        Args:
            query_embedding: 쿼리 임베딩 벡터
            top_k: 반환할 최대 결과 수
            where: 메타데이터 필터 (예: {"keyword": "트럼프"})
            where_document: 문서 내용 필터

        Returns:
            list[SearchResult]: 검색 결과 (유사도 내림차순)
        """
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            where_document=where_document,
            include=["documents", "metadatas", "distances"],
        )

        return self._parse_query_results(results)

    def search_by_keyword(
        self,
        query_embedding: list[float],
        keyword: str,
        top_k: int = 10,
    ) -> list[SearchResult]:
        """
        키워드 필터링 벡터 검색

        특정 키워드 내에서만 유사 문서 검색
        (중복 제거에 사용)

        Args:
            query_embedding: 쿼리 임베딩 벡터
            keyword: 필터링할 키워드
            top_k: 반환할 최대 결과 수

        Returns:
            list[SearchResult]: 검색 결과
        """
        return self.search(
            query_embedding=query_embedding,
            top_k=top_k,
            where={"keyword": keyword},
        )

    def search_by_text(
        self,
        query_text: str,
        embedding_service: Any,  # EmbeddingService (순환 import 방지)
        top_k: int = 10,
        where: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """
        텍스트 쿼리로 검색 (편의 메서드)

        EmbeddingService를 사용하여 쿼리를 임베딩한 후 검색

        Args:
            query_text: 검색 쿼리 텍스트
            embedding_service: EmbeddingService 인스턴스
            top_k: 반환할 최대 결과 수
            where: 메타데이터 필터

        Returns:
            list[SearchResult]: 검색 결과
        """
        query_embedding = embedding_service.embed(query_text).tolist()
        return self.search(query_embedding, top_k=top_k, where=where)

    def _parse_query_results(self, results: dict) -> list[SearchResult]:
        """ChromaDB 쿼리 결과를 SearchResult 리스트로 변환"""
        search_results: list[SearchResult] = []

        if not results or not results.get("ids"):
            return search_results

        # ChromaDB는 배치 쿼리 형태로 반환 (첫 번째 결과만 사용)
        ids = results["ids"][0]
        documents = results["documents"][0] if results.get("documents") else [None] * len(ids)
        metadatas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(ids)
        distances = results["distances"][0] if results.get("distances") else [0.0] * len(ids)

        for doc_id, doc, meta, dist in zip(ids, documents, metadatas, distances):
            search_results.append(
                SearchResult(
                    id=doc_id,
                    document=doc or "",
                    metadata=meta or {},
                    distance=dist,
                )
            )

        return search_results

    # =========================================================================
    # RETRIEVAL FOR RAG
    # =========================================================================

    def retrieve_for_rag(
        self,
        query_embedding: list[float],
        keyword: str | None = None,
        top_k: int = 5,
        min_similarity: float = 0.5,
    ) -> list[SearchResult]:
        """
        RAG용 문서 검색

        최소 유사도 필터링 적용

        Args:
            query_embedding: 쿼리 임베딩
            keyword: 키워드 필터 (선택)
            top_k: 검색 결과 수
            min_similarity: 최소 유사도 임계값

        Returns:
            list[SearchResult]: 필터링된 검색 결과
        """
        where = {"keyword": keyword} if keyword else None
        results = self.search(query_embedding, top_k=top_k * 2, where=where)

        # 유사도 필터링
        filtered = [r for r in results if r.similarity >= min_similarity]

        return filtered[:top_k]

    # =========================================================================
    # MANAGEMENT OPERATIONS
    # =========================================================================

    def get_stats(self) -> CollectionStats:
        """컬렉션 통계 조회"""
        return CollectionStats(
            name=self.collection.name,
            count=self.collection.count(),
            metadata=self.collection.metadata or {},
        )

    def delete_by_keyword(self, keyword: str) -> int:
        """
        키워드로 문서 삭제

        Args:
            keyword: 삭제할 키워드

        Returns:
            int: 삭제된 문서 수
        """
        # 먼저 해당 키워드의 문서 ID 조회
        results = self.collection.get(
            where={"keyword": keyword},
            include=[],
        )

        ids_to_delete = results.get("ids", [])

        if ids_to_delete:
            self.collection.delete(ids=ids_to_delete)
            logger.info(f"Deleted {len(ids_to_delete)} documents with keyword '{keyword}'")

        return len(ids_to_delete)

    def delete_by_ids(self, ids: list[str]) -> int:
        """
        ID로 문서 삭제

        Args:
            ids: 삭제할 문서 ID 리스트

        Returns:
            int: 삭제된 문서 수
        """
        if not ids:
            return 0

        self.collection.delete(ids=ids)
        logger.info(f"Deleted {len(ids)} documents by IDs")

        return len(ids)

    def get_by_ids(self, ids: list[str]) -> list[SearchResult]:
        """
        ID로 문서 조회

        Args:
            ids: 조회할 문서 ID 리스트

        Returns:
            list[SearchResult]: 조회된 문서 목록
        """
        results = self.collection.get(
            ids=ids,
            include=["documents", "metadatas"],
        )

        search_results: list[SearchResult] = []

        for doc_id, doc, meta in zip(
            results.get("ids", []),
            results.get("documents", []),
            results.get("metadatas", []),
        ):
            search_results.append(
                SearchResult(
                    id=doc_id,
                    document=doc or "",
                    metadata=meta or {},
                    distance=0.0,  # ID 조회는 거리 없음
                )
            )

        return search_results

    def clear(self) -> int:
        """
        컬렉션 전체 삭제 (주의!)

        Returns:
            int: 삭제된 문서 수
        """
        count = self.collection.count()

        if count > 0:
            # 모든 ID 조회 후 삭제
            all_ids = self.collection.get(include=[])["ids"]
            self.collection.delete(ids=all_ids)
            logger.warning(f"Cleared {count} documents from collection")

        return count


# =============================================================================
# GLOBAL INSTANCE & FACTORY
# =============================================================================


_global_instance: VectorStore | None = None
_global_lock = threading.Lock()


def get_vector_store() -> VectorStore:
    """
    전역 VectorStore 인스턴스 반환

    Returns:
        VectorStore: 싱글톤 인스턴스

    Usage:
        from trendops.store.vector_store import get_vector_store

        store = get_vector_store()
        results = store.search(query_embedding, top_k=10)
    """
    global _global_instance

    if _global_instance is None:
        with _global_lock:
            if _global_instance is None:
                _global_instance = VectorStore()

    return _global_instance


def reset_vector_store() -> None:
    """
    전역 인스턴스 초기화 (테스트용)

    ⚠️ 주의: 프로덕션에서는 사용 금지
    """
    global _global_instance

    with _global_lock:
        _global_instance = None
        VectorStore._instance = None
        VectorStore._initialized = False

    logger.warning("VectorStore global instance reset")


# =============================================================================
# CLI TEST
# =============================================================================


if __name__ == "__main__":
    import numpy as np

    print("=" * 70)
    print("  VectorStore (ChromaDB) Test")
    print("=" * 70)

    # 인스턴스 생성
    print("\n[1] Creating VectorStore...")
    store = get_vector_store()
    stats = store.get_stats()
    print(f"    ✓ Collection: {stats.name}")
    print(f"    ✓ Documents: {stats.count}")

    # 테스트 데이터
    test_keyword = "__test_vector_store__"
    test_docs = [
        "트럼프가 중국에 25% 관세를 부과했다.",
        "미국의 관세 정책이 세계 경제에 영향을 미친다.",
        "삼성전자 주가가 급등했다.",
    ]

    # 가짜 임베딩 생성 (실제로는 EmbeddingService 사용)
    print("\n[2] Adding test documents...")
    dim = 1024  # bge-m3-korean 임베딩 차원
    fake_embeddings = [np.random.randn(dim).tolist() for _ in test_docs]

    result = store.add_documents(
        contents=test_docs,
        embeddings=fake_embeddings,
        metadatas=[{"keyword": test_keyword, "title": f"Test {i}"} for i in range(len(test_docs))],
    )
    print(f"    ✓ Added: {result.added}, Skipped: {result.skipped}")

    # 검색 테스트
    print("\n[3] Searching...")
    query_embedding = np.random.randn(dim).tolist()
    results = store.search_by_keyword(query_embedding, test_keyword, top_k=3)
    print(f"    ✓ Found {len(results)} results")
    for i, r in enumerate(results, 1):
        print(f"      [{i}] sim={r.similarity:.3f}: {r.document[:40]}...")

    # 통계
    print("\n[4] Stats...")
    stats = store.get_stats()
    print(f"    ✓ Total documents: {stats.count}")

    # 정리
    print("\n[5] Cleanup...")
    deleted = store.delete_by_keyword(test_keyword)
    print(f"    ✓ Deleted {deleted} test documents")

    print("\n" + "=" * 70)
    print("✅ VectorStore test completed!")
    print("=" * 70)
