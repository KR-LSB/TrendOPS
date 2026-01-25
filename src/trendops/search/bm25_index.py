# src/trendops/search/bm25_index.py
"""
BM25 Sparse Search Index

Blueprint Week 3 Day 3-4:
- BM25 (희소) 검색 인덱스 구축
- 한국어 토크나이저 적용
- Hybrid Search의 Sparse 검색 컴포넌트

기술 스택:
- rank-bm25: BM25 알고리즘 구현
- 한국어 토크나이저: 기본 공백 분리 + 불용어 제거

⚠️ KoNLPy 설치 이슈로 인해 기본 토크나이저 사용
   필요 시 KoNLPy (Okt, Komoran 등)로 업그레이드 가능
"""
from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from trendops.config.settings import get_settings
from trendops.utils.logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# DATA MODELS
# =============================================================================


class BM25SearchResult(BaseModel):
    """BM25 검색 결과"""
    
    doc_id: str = Field(..., description="문서 ID")
    document: str = Field(..., description="문서 내용")
    metadata: dict[str, Any] = Field(default_factory=dict, description="메타데이터")
    score: float = Field(..., description="BM25 점수 (높을수록 관련성 높음)")
    rank: int = Field(..., description="검색 결과 순위 (1-based)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "doc_id": "article_001",
                "document": "트럼프 관세 정책 발표...",
                "score": 12.456,
                "rank": 1,
            }
        }


class IndexStats(BaseModel):
    """인덱스 통계"""
    
    total_documents: int = Field(default=0, description="총 문서 수")
    total_tokens: int = Field(default=0, description="총 토큰 수")
    avg_doc_length: float = Field(default=0.0, description="평균 문서 길이")
    vocabulary_size: int = Field(default=0, description="어휘 크기")
    last_updated: datetime | None = Field(default=None, description="마지막 업데이트 시간")


@dataclass
class IndexedDocument:
    """인덱스에 저장된 문서"""
    doc_id: str
    document: str
    tokens: list[str]
    metadata: dict[str, Any]


# =============================================================================
# KOREAN TOKENIZER
# =============================================================================


class KoreanTokenizer:
    """
    한국어 토크나이저 (기본 버전)
    
    특징:
    - 공백 기반 분리
    - 한국어 불용어 제거
    - 특수문자 제거
    - 최소 길이 필터링
    
    ⚠️ Production에서는 KoNLPy (Okt, Komoran)로 업그레이드 권장
    """
    
    # 한국어 불용어 (기본)
    STOPWORDS: set[str] = {
        # 조사
        "이", "가", "을", "를", "은", "는", "의", "에", "에서", "로", "으로",
        "와", "과", "도", "만", "까지", "부터", "에게", "한테", "께",
        # 접속사/부사
        "그리고", "그러나", "하지만", "그래서", "따라서", "또한", "또",
        "및", "등", "등등", "즉", "곧",
        # 대명사
        "이것", "그것", "저것", "여기", "거기", "저기",
        "나", "너", "우리", "당신", "그", "그녀",
        # 일반 불용어
        "것", "수", "때", "중", "내", "위", "후", "전", "간",
        "말", "더", "덜", "매우", "너무", "아주", "정말", "진짜",
        "약", "각", "매", "별", "총", "전체", "일부",
        # 기타
        "있다", "없다", "하다", "되다", "이다", "아니다",
        "있는", "없는", "하는", "되는", "인", "한",
    }
    
    # 최소 토큰 길이
    MIN_TOKEN_LENGTH: int = 2
    
    def __init__(
        self,
        stopwords: set[str] | None = None,
        min_length: int = 2,
        use_konlpy: bool = False,
    ):
        """
        Args:
            stopwords: 커스텀 불용어 셋 (None이면 기본값 사용)
            min_length: 최소 토큰 길이
            use_konlpy: KoNLPy 사용 여부 (설치 필요)
        """
        self.stopwords = stopwords or self.STOPWORDS
        self.min_length = min_length
        self.use_konlpy = use_konlpy
        self._konlpy_tokenizer = None
        
        if use_konlpy:
            self._init_konlpy()
    
    def _init_konlpy(self) -> None:
        """KoNLPy 토크나이저 초기화"""
        try:
            from konlpy.tag import Okt
            self._konlpy_tokenizer = Okt()
            logger.info("KoNLPy (Okt) tokenizer initialized")
        except ImportError:
            logger.warning(
                "KoNLPy not installed, falling back to basic tokenizer. "
                "Install with: pip install konlpy"
            )
            self.use_konlpy = False
    
    def tokenize(self, text: str) -> list[str]:
        """
        텍스트를 토큰화
        
        Args:
            text: 입력 텍스트
            
        Returns:
            list[str]: 토큰 리스트 (정규화된)
        """
        if not text or not text.strip():
            return []
        
        # 소문자 변환 (영어)
        text = text.lower()
        
        # 특수문자 제거 (한글, 영문, 숫자만 유지)
        text = re.sub(r'[^\w\s가-힣]', ' ', text)
        
        # 토큰화
        if self.use_konlpy and self._konlpy_tokenizer:
            # KoNLPy 형태소 분석
            tokens = self._konlpy_tokenizer.nouns(text)  # 명사만 추출
        else:
            # 기본 공백 분리
            tokens = text.split()
        
        # 필터링
        tokens = [
            token.strip()
            for token in tokens
            if (
                token.strip() and
                len(token.strip()) >= self.min_length and
                token.strip() not in self.stopwords
            )
        ]
        
        return tokens
    
    def tokenize_batch(self, texts: list[str]) -> list[list[str]]:
        """
        배치 토큰화
        
        Args:
            texts: 텍스트 리스트
            
        Returns:
            list[list[str]]: 토큰 리스트의 리스트
        """
        return [self.tokenize(text) for text in texts]


# =============================================================================
# BM25 INDEX
# =============================================================================


class BM25Index:
    """
    BM25 Sparse Search Index
    
    Blueprint Week 3 요구사항:
    - BM25 (희소) 검색
    - 한국어 토크나이저 적용
    - Hybrid Search와 통합
    
    알고리즘: BM25 Okapi
    - k1=1.5 (term frequency saturation)
    - b=0.75 (document length normalization)
    
    사용 예시:
        from trendops.search.bm25_index import get_bm25_index
        
        index = get_bm25_index()
        
        # 문서 추가
        index.add_documents(
            doc_ids=["doc1", "doc2"],
            documents=["트럼프 관세", "삼성전자 주가"],
            metadatas=[{"keyword": "트럼프"}, {"keyword": "삼성"}]
        )
        
        # 검색
        results = index.search("관세 정책", top_k=5)
    """
    
    _instance: BM25Index | None = None
    _lock: threading.Lock = threading.Lock()
    _initialized: bool = False
    
    def __new__(cls) -> BM25Index:
        """Thread-safe Singleton 패턴"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self) -> None:
        """BM25 인덱스 초기화"""
        if BM25Index._initialized:
            return
        
        with BM25Index._lock:
            if BM25Index._initialized:
                return
            
            self._settings = get_settings()
            
            # 토크나이저
            self._tokenizer = KoreanTokenizer(
                use_konlpy=False,  # 기본: 기본 토크나이저 사용
            )
            
            # 문서 저장소
            self._documents: dict[str, IndexedDocument] = {}
            self._doc_id_to_idx: dict[str, int] = {}
            self._idx_to_doc_id: dict[int, str] = {}
            
            # BM25 인덱스 (lazy initialization)
            self._bm25: Any = None  # rank_bm25.BM25Okapi
            self._corpus: list[list[str]] = []
            self._needs_rebuild: bool = True
            
            # 통계
            self._stats = IndexStats()
            
            BM25Index._initialized = True
            
            logger.info("BM25Index initialized")
    
    @property
    def tokenizer(self) -> KoreanTokenizer:
        """토크나이저"""
        return self._tokenizer
    
    @property
    def document_count(self) -> int:
        """인덱싱된 문서 수"""
        return len(self._documents)
    
    # =========================================================================
    # DOCUMENT OPERATIONS
    # =========================================================================
    
    def add_documents(
        self,
        doc_ids: list[str],
        documents: list[str],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> int:
        """
        문서 배치 추가
        
        Args:
            doc_ids: 문서 ID 리스트
            documents: 문서 내용 리스트
            metadatas: 메타데이터 리스트
            
        Returns:
            int: 추가된 문서 수
        """
        if len(doc_ids) != len(documents):
            raise ValueError("doc_ids and documents must have same length")
        
        if metadatas is None:
            metadatas = [{} for _ in documents]
        elif len(metadatas) != len(documents):
            raise ValueError("metadatas must have same length as documents")
        
        added = 0
        
        for doc_id, doc, meta in zip(doc_ids, documents, metadatas):
            if doc_id in self._documents:
                logger.debug(f"Document {doc_id} already exists, skipping")
                continue
            
            # 토큰화
            tokens = self._tokenizer.tokenize(doc)
            
            if not tokens:
                logger.debug(f"Document {doc_id} has no tokens after filtering, skipping")
                continue
            
            # 저장
            idx = len(self._corpus)
            self._documents[doc_id] = IndexedDocument(
                doc_id=doc_id,
                document=doc,
                tokens=tokens,
                metadata=meta,
            )
            self._doc_id_to_idx[doc_id] = idx
            self._idx_to_doc_id[idx] = doc_id
            self._corpus.append(tokens)
            
            added += 1
        
        if added > 0:
            self._needs_rebuild = True
            self._update_stats()
            logger.info(f"Added {added} documents to BM25 index")
        
        return added
    
    def add_document(
        self,
        doc_id: str,
        document: str,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """
        단일 문서 추가
        
        Returns:
            bool: 추가 성공 여부
        """
        return self.add_documents(
            doc_ids=[doc_id],
            documents=[document],
            metadatas=[metadata or {}],
        ) > 0
    
    def remove_document(self, doc_id: str) -> bool:
        """
        문서 제거
        
        ⚠️ 주의: 제거 후 인덱스 재구축 필요
        
        Args:
            doc_id: 제거할 문서 ID
            
        Returns:
            bool: 제거 성공 여부
        """
        if doc_id not in self._documents:
            return False
        
        # 문서 삭제
        del self._documents[doc_id]
        
        # 인덱스 재구축 필요
        self._rebuild_index()
        
        logger.info(f"Removed document {doc_id} from BM25 index")
        return True
    
    def _rebuild_index(self) -> None:
        """인덱스 재구축"""
        self._corpus = []
        self._doc_id_to_idx = {}
        self._idx_to_doc_id = {}
        
        for idx, (doc_id, doc) in enumerate(self._documents.items()):
            self._corpus.append(doc.tokens)
            self._doc_id_to_idx[doc_id] = idx
            self._idx_to_doc_id[idx] = doc_id
        
        self._needs_rebuild = True
        self._update_stats()
    
    def _ensure_bm25(self) -> None:
        """BM25 인덱스 생성/갱신"""
        if self._needs_rebuild and self._corpus:
            try:
                from rank_bm25 import BM25Okapi
                
                self._bm25 = BM25Okapi(
                    self._corpus,
                    k1=1.5,  # term frequency saturation parameter
                    b=0.75,  # document length normalization
                )
                self._needs_rebuild = False
                
                logger.debug(f"BM25 index built with {len(self._corpus)} documents")
                
            except ImportError:
                raise ImportError(
                    "rank-bm25 not installed. Install with: pip install rank-bm25"
                )
    
    def _update_stats(self) -> None:
        """통계 업데이트"""
        total_tokens = sum(len(doc.tokens) for doc in self._documents.values())
        doc_count = len(self._documents)
        
        # 어휘 계산
        vocabulary = set()
        for doc in self._documents.values():
            vocabulary.update(doc.tokens)
        
        self._stats = IndexStats(
            total_documents=doc_count,
            total_tokens=total_tokens,
            avg_doc_length=total_tokens / doc_count if doc_count > 0 else 0.0,
            vocabulary_size=len(vocabulary),
            last_updated=datetime.now(),
        )
    
    # =========================================================================
    # SEARCH OPERATIONS
    # =========================================================================
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        where: dict[str, Any] | None = None,
    ) -> list[BM25SearchResult]:
        """
        BM25 검색
        
        Args:
            query: 검색 쿼리
            top_k: 반환할 최대 결과 수
            where: 메타데이터 필터 (예: {"keyword": "트럼프"})
            
        Returns:
            list[BM25SearchResult]: 검색 결과 (점수 내림차순)
        """
        if not self._corpus:
            logger.warning("BM25 index is empty")
            return []
        
        # 인덱스 확인
        self._ensure_bm25()
        
        # 쿼리 토큰화
        query_tokens = self._tokenizer.tokenize(query)
        
        if not query_tokens:
            logger.warning(f"Query '{query}' has no tokens after filtering")
            return []
        
        # BM25 점수 계산
        scores = self._bm25.get_scores(query_tokens)
        
        # 결과 생성
        results: list[BM25SearchResult] = []
        
        # 점수 기준 정렬된 인덱스
        sorted_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )
        
        rank = 0
        for idx in sorted_indices:
            if rank >= top_k:
                break
            
            doc_id = self._idx_to_doc_id.get(idx)
            if doc_id is None:
                continue
            
            doc = self._documents.get(doc_id)
            if doc is None:
                continue
            
            # 메타데이터 필터 적용
            if where:
                skip = False
                for key, value in where.items():
                    if doc.metadata.get(key) != value:
                        skip = True
                        break
                if skip:
                    continue
            
            rank += 1
            results.append(BM25SearchResult(
                doc_id=doc_id,
                document=doc.document,
                metadata=doc.metadata,
                score=float(scores[idx]),
                rank=rank,
            ))
        
        return results
    
    def search_by_keyword(
        self,
        query: str,
        keyword: str,
        top_k: int = 10,
    ) -> list[BM25SearchResult]:
        """
        키워드 필터링 BM25 검색
        
        Args:
            query: 검색 쿼리
            keyword: 필터링할 키워드
            top_k: 반환할 최대 결과 수
            
        Returns:
            list[BM25SearchResult]: 검색 결과
        """
        return self.search(query, top_k=top_k, where={"keyword": keyword})
    
    def get_scores_for_ids(
        self,
        query: str,
        doc_ids: list[str],
    ) -> dict[str, float]:
        """
        특정 문서 ID들에 대한 BM25 점수 반환
        
        Hybrid Search에서 RRF 융합 시 사용
        
        Args:
            query: 검색 쿼리
            doc_ids: 점수를 계산할 문서 ID 리스트
            
        Returns:
            dict[str, float]: {doc_id: score}
        """
        if not self._corpus:
            return {}
        
        self._ensure_bm25()
        
        query_tokens = self._tokenizer.tokenize(query)
        if not query_tokens:
            return {}
        
        scores = self._bm25.get_scores(query_tokens)
        
        result = {}
        for doc_id in doc_ids:
            idx = self._doc_id_to_idx.get(doc_id)
            if idx is not None:
                result[doc_id] = float(scores[idx])
        
        return result
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def get_stats(self) -> IndexStats:
        """인덱스 통계 반환"""
        return self._stats.model_copy()
    
    def get_document(self, doc_id: str) -> IndexedDocument | None:
        """문서 조회"""
        return self._documents.get(doc_id)
    
    def get_all_doc_ids(self) -> list[str]:
        """모든 문서 ID 반환"""
        return list(self._documents.keys())
    
    def clear(self) -> int:
        """
        인덱스 초기화
        
        Returns:
            int: 삭제된 문서 수
        """
        count = len(self._documents)
        
        self._documents.clear()
        self._doc_id_to_idx.clear()
        self._idx_to_doc_id.clear()
        self._corpus.clear()
        self._bm25 = None
        self._needs_rebuild = True
        self._stats = IndexStats()
        
        logger.warning(f"BM25 index cleared ({count} documents removed)")
        return count
    
    def contains(self, doc_id: str) -> bool:
        """문서 존재 여부 확인"""
        return doc_id in self._documents


# =============================================================================
# GLOBAL INSTANCE & FACTORY
# =============================================================================


_global_instance: BM25Index | None = None
_global_lock = threading.Lock()


def get_bm25_index() -> BM25Index:
    """
    전역 BM25Index 인스턴스 반환
    
    Returns:
        BM25Index: 싱글톤 인스턴스
        
    Usage:
        from trendops.search.bm25_index import get_bm25_index
        
        index = get_bm25_index()
        results = index.search("트럼프 관세", top_k=10)
    """
    global _global_instance
    
    if _global_instance is None:
        with _global_lock:
            if _global_instance is None:
                _global_instance = BM25Index()
    
    return _global_instance


def reset_bm25_index() -> None:
    """
    전역 인스턴스 초기화 (테스트용)
    
    ⚠️ 주의: 프로덕션에서는 사용 금지
    """
    global _global_instance
    
    with _global_lock:
        _global_instance = None
        BM25Index._instance = None
        BM25Index._initialized = False
    
    logger.warning("BM25Index global instance reset")


# =============================================================================
# CLI TEST
# =============================================================================


if __name__ == "__main__":
    print("=" * 70)
    print("  BM25Index Test")
    print("=" * 70)
    
    # 인스턴스 생성
    print("\n[1] Creating BM25Index...")
    index = get_bm25_index()
    print(f"    ✓ Document count: {index.document_count}")
    
    # 테스트 문서
    test_keyword = "__test_bm25__"
    test_docs = [
        ("doc1", "트럼프 대통령이 중국산 제품에 25% 관세를 부과한다고 발표했다."),
        ("doc2", "미국의 관세 정책이 세계 경제에 영향을 미친다."),
        ("doc3", "삼성전자 주가가 급등했다. 반도체 수요 증가 영향."),
        ("doc4", "비트코인 가격이 신고가를 경신했다."),
        ("doc5", "트럼프 행정부의 무역 정책에 대한 우려가 커지고 있다."),
    ]
    
    # 문서 추가
    print("\n[2] Adding test documents...")
    added = index.add_documents(
        doc_ids=[d[0] for d in test_docs],
        documents=[d[1] for d in test_docs],
        metadatas=[{"keyword": test_keyword} for _ in test_docs],
    )
    print(f"    ✓ Added: {added} documents")
    
    # 토큰화 확인
    print("\n[3] Tokenization test...")
    sample = "트럼프 대통령의 관세 정책이 발표되었다."
    tokens = index.tokenizer.tokenize(sample)
    print(f"    Input: {sample}")
    print(f"    Tokens: {tokens}")
    
    # 검색 테스트
    print("\n[4] Search test...")
    query = "트럼프 관세 정책"
    results = index.search(query, top_k=3)
    print(f"    Query: '{query}'")
    print(f"    Results ({len(results)}):")
    for r in results:
        print(f"      [{r.rank}] score={r.score:.3f}: {r.document[:50]}...")
    
    # 키워드 필터 검색
    print("\n[5] Search with keyword filter...")
    results = index.search_by_keyword("삼성전자", test_keyword, top_k=3)
    print(f"    Query: '삼성전자' (keyword={test_keyword})")
    print(f"    Results ({len(results)}):")
    for r in results:
        print(f"      [{r.rank}] score={r.score:.3f}: {r.document[:50]}...")
    
    # 통계
    print("\n[6] Index stats...")
    stats = index.get_stats()
    print(f"    ✓ Total documents: {stats.total_documents}")
    print(f"    ✓ Total tokens: {stats.total_tokens}")
    print(f"    ✓ Avg doc length: {stats.avg_doc_length:.1f}")
    print(f"    ✓ Vocabulary size: {stats.vocabulary_size}")
    
    # 정리
    print("\n[7] Cleanup...")
    cleared = index.clear()
    print(f"    ✓ Cleared {cleared} documents")
    
    print("\n" + "=" * 70)
    print("✅ BM25Index test completed!")
    print("=" * 70)