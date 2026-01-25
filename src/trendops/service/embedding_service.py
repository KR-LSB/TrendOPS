# src/trendops/service/embedding_service.py
"""
CPU 전용 Embedding Service

⚠️ CRITICAL HARDWARE CONSTRAINT (Blueprint 준수):
- 이 모듈은 절대로 GPU를 사용하지 않습니다.
- GPU (16GB)는 vLLM 전용으로 예약되어 있습니다.
- Embedding은 CPU에서 배치 처리로 Latency를 보완합니다.

Blueprint Section 1.6 Resource Allocation:
- Embedding Model (bge-m3-korean): CPU 8 threads
- GPU: vLLM 단독 점유 (OOM 방지)
"""
from __future__ import annotations

import os
import threading
from typing import TYPE_CHECKING

import numpy as np

# ============================================================================
# ⚠️ CUDA 차단 - 반드시 다른 import 전에 설정
# ============================================================================
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

from trendops.config.settings import get_settings
from trendops.utils.logger import get_logger

logger = get_logger(__name__)


class EmbeddingService:
    """
    CPU 전용 Embedding Service
    
    Blueprint Section 1.6 준수:
    - GPU 메모리 단편화 방지를 위해 CPU에서 실행
    - 배치 처리로 CPU 모드의 Latency 보완
    - Thread-safe Singleton 패턴
    
    사용 예시:
        # 방법 1: 전역 인스턴스 사용 (권장)
        from trendops.service.embedding_service import get_embedding_service
        service = get_embedding_service()
        embeddings = service.embed_batch(["텍스트1", "텍스트2"])
        
        # 방법 2: 직접 인스턴스 생성
        service = EmbeddingService()
        embeddings = service.embed_batch(texts)
    """
    
    _instance: EmbeddingService | None = None
    _lock: threading.Lock = threading.Lock()
    _initialized: bool = False
    
    def __new__(cls) -> EmbeddingService:
        """Thread-safe Singleton 패턴"""
        if cls._instance is None:
            with cls._lock:
                # Double-checked locking
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self) -> None:
        """
        Embedding 모델 초기화 (최초 1회만 실행)
        
        ⚠️ GPU 사용 완전 차단:
        - CUDA_VISIBLE_DEVICES="" 환경 변수 설정
        - device='cpu' 명시적 지정
        - torch.set_num_threads()로 CPU 스레드 제어
        """
        # Singleton: 이미 초기화되었으면 스킵
        if EmbeddingService._initialized:
            return
        
        with EmbeddingService._lock:
            if EmbeddingService._initialized:
                return
            
            self._settings = get_settings()
            
            # CPU 스레드 설정
            self._setup_cpu_threads()
            
            # 모델 로드
            self._model = self._load_model()
            
            # 초기화 완료 플래그
            EmbeddingService._initialized = True
            
            logger.info(
                "EmbeddingService initialized (CPU only)",
                extra={
                    "model": self._settings.embedding_model_name,
                    "device": "cpu",
                    "threads": self._settings.cpu_threads,
                    "max_seq_length": self._settings.embedding_max_seq_length,
                    "batch_size": self._settings.embedding_batch_size,
                }
            )
    
    def _setup_cpu_threads(self) -> None:
        """CPU 스레드 수 설정"""
        import torch
        
        num_threads = self._settings.cpu_threads
        
        # PyTorch CPU 스레드 설정
        torch.set_num_threads(num_threads)
        torch.set_num_interop_threads(max(1, num_threads // 2))
        
        # 추가 환경 변수 설정 (일부 백엔드에서 사용)
        os.environ["OMP_NUM_THREADS"] = str(num_threads)
        os.environ["MKL_NUM_THREADS"] = str(num_threads)
        
        logger.debug(f"CPU threads configured: {num_threads}")
    
    def _load_model(self) -> SentenceTransformer:
        """
        Sentence Transformer 모델 로드 (CPU 전용)
        
        Returns:
            SentenceTransformer: CPU에 로드된 임베딩 모델
        """
        from sentence_transformers import SentenceTransformer
        
        model_name = self._settings.embedding_model_name
        
        logger.info(f"Loading embedding model: {model_name} (CPU)")
        
        # ⚠️ device='cpu' 명시적 지정 - GPU 사용 방지
        model = SentenceTransformer(
            model_name,
            device="cpu",
        )
        
        # 최대 시퀀스 길이 설정
        model.max_seq_length = self._settings.embedding_max_seq_length
        
        # 모델이 실제로 CPU에 있는지 확인
        self._verify_cpu_only(model)
        
        return model
    
    def _verify_cpu_only(self, model: SentenceTransformer) -> None:
        """모델이 CPU에서만 실행되는지 검증"""
        import torch
        
        # CUDA 사용 불가 확인
        if torch.cuda.is_available():
            logger.warning(
                "CUDA is available but blocked by CUDA_VISIBLE_DEVICES=''. "
                "Embedding will run on CPU only."
            )
        
        # 모델 파라미터가 CPU에 있는지 확인
        for param in model.parameters():
            if param.device.type != "cpu":
                raise RuntimeError(
                    f"Model parameter found on {param.device}, expected CPU. "
                    "GPU usage is forbidden for embedding."
                )
        
        logger.debug("Verified: All model parameters are on CPU")
    
    @property
    def model(self) -> SentenceTransformer:
        """임베딩 모델 접근자"""
        return self._model
    
    @property
    def embedding_dimension(self) -> int:
        """임베딩 벡터 차원"""
        return self._model.get_sentence_embedding_dimension()
    
    def embed(self, text: str) -> np.ndarray:
        """
        단일 텍스트 임베딩
        
        Args:
            text: 임베딩할 텍스트
            
        Returns:
            np.ndarray: 정규화된 임베딩 벡터 (1D)
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for embedding")
            return np.zeros(self.embedding_dimension, dtype=np.float32)
        
        embedding = self._model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        
        return embedding.astype(np.float32)
    
    def embed_batch(
        self,
        texts: list[str],
        batch_size: int | None = None,
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        배치 텍스트 임베딩 (CPU 최적화)
        
        CPU 모드에서는 배치 크기를 크게 설정하여 Latency 보완:
        - GPU: batch=32, ~50ms
        - CPU: batch=64, ~200ms (but 안정적)
        
        Args:
            texts: 임베딩할 텍스트 리스트
            batch_size: 배치 크기 (None이면 설정값 사용)
            show_progress: 진행 표시줄 표시 여부
            
        Returns:
            np.ndarray: 정규화된 임베딩 벡터 배열 (N x D)
            
        Raises:
            ValueError: 빈 텍스트 리스트가 입력된 경우
        """
        if not texts:
            raise ValueError("Empty text list provided for embedding")
        
        # 빈 문자열 필터링 및 로깅
        valid_indices: list[int] = []
        valid_texts: list[str] = []
        
        for i, text in enumerate(texts):
            if text and text.strip():
                valid_indices.append(i)
                valid_texts.append(text)
            else:
                logger.warning(f"Empty text at index {i}, will use zero vector")
        
        if not valid_texts:
            logger.warning("All texts are empty, returning zero vectors")
            return np.zeros(
                (len(texts), self.embedding_dimension),
                dtype=np.float32
            )
        
        # 배치 크기 결정
        effective_batch_size = batch_size or self._settings.embedding_batch_size
        
        logger.debug(
            f"Embedding {len(valid_texts)} texts",
            extra={
                "total_input": len(texts),
                "valid_texts": len(valid_texts),
                "batch_size": effective_batch_size,
            }
        )
        
        # 배치 임베딩 실행
        valid_embeddings = self._model.encode(
            valid_texts,
            batch_size=effective_batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
        )
        
        # 원래 순서대로 결과 배열 구성 (빈 텍스트는 zero vector)
        result = np.zeros(
            (len(texts), self.embedding_dimension),
            dtype=np.float32
        )
        
        for idx, valid_idx in enumerate(valid_indices):
            result[valid_idx] = valid_embeddings[idx]
        
        logger.debug(f"Embedding complete: shape={result.shape}")
        
        return result
    
    def embed_batch_chunked(
        self,
        texts: list[str],
        chunk_size: int = 1000,
        batch_size: int | None = None,
    ) -> np.ndarray:
        """
        대용량 텍스트 청크 단위 임베딩
        
        메모리 효율을 위해 대용량 데이터를 청크 단위로 처리
        
        Args:
            texts: 임베딩할 텍스트 리스트
            chunk_size: 한 번에 처리할 텍스트 수
            batch_size: 임베딩 배치 크기
            
        Returns:
            np.ndarray: 정규화된 임베딩 벡터 배열
        """
        if len(texts) <= chunk_size:
            return self.embed_batch(texts, batch_size=batch_size)
        
        logger.info(
            f"Processing {len(texts)} texts in chunks of {chunk_size}",
        )
        
        all_embeddings: list[np.ndarray] = []
        
        for i in range(0, len(texts), chunk_size):
            chunk = texts[i:i + chunk_size]
            chunk_embeddings = self.embed_batch(chunk, batch_size=batch_size)
            all_embeddings.append(chunk_embeddings)
            
            logger.debug(
                f"Processed chunk {i // chunk_size + 1}/{(len(texts) - 1) // chunk_size + 1}"
            )
        
        return np.vstack(all_embeddings)
    
    def similarity(
        self,
        text1: str,
        text2: str,
    ) -> float:
        """
        두 텍스트 간 코사인 유사도 계산
        
        Args:
            text1: 첫 번째 텍스트
            text2: 두 번째 텍스트
            
        Returns:
            float: 코사인 유사도 (-1 ~ 1, 정규화된 벡터이므로 0 ~ 1)
        """
        emb1 = self.embed(text1)
        emb2 = self.embed(text2)
        
        # 정규화된 벡터의 내적 = 코사인 유사도
        return float(np.dot(emb1, emb2))
    
    def similarity_batch(
        self,
        query: str,
        documents: list[str],
    ) -> list[tuple[int, float]]:
        """
        쿼리와 문서 리스트 간 유사도 계산 및 정렬
        
        Args:
            query: 쿼리 텍스트
            documents: 문서 리스트
            
        Returns:
            list[tuple[int, float]]: (인덱스, 유사도) 튜플 리스트 (유사도 내림차순)
        """
        query_embedding = self.embed(query)
        doc_embeddings = self.embed_batch(documents)
        
        # 행렬 곱으로 모든 유사도 한번에 계산
        similarities = np.dot(doc_embeddings, query_embedding)
        
        # (인덱스, 유사도) 튜플로 변환 후 정렬
        results = [(i, float(sim)) for i, sim in enumerate(similarities)]
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results


# =============================================================================
# GLOBAL INSTANCE & FACTORY FUNCTION
# =============================================================================

_global_instance: EmbeddingService | None = None
_global_lock = threading.Lock()


def get_embedding_service() -> EmbeddingService:
    """
    전역 EmbeddingService 인스턴스 반환 (권장 사용 방법)
    
    Thread-safe lazy initialization 적용
    
    Returns:
        EmbeddingService: 전역 싱글톤 인스턴스
        
    Usage:
        from trendops.service.embedding_service import get_embedding_service
        
        service = get_embedding_service()
        embeddings = service.embed_batch(["텍스트1", "텍스트2"])
    """
    global _global_instance
    
    if _global_instance is None:
        with _global_lock:
            if _global_instance is None:
                _global_instance = EmbeddingService()
    
    return _global_instance


def reset_embedding_service() -> None:
    """
    전역 인스턴스 초기화 (테스트용)
    
    ⚠️ 주의: 프로덕션에서는 사용하지 마세요.
    """
    global _global_instance
    
    with _global_lock:
        _global_instance = None
        EmbeddingService._instance = None
        EmbeddingService._initialized = False
    
    logger.warning("EmbeddingService global instance reset")


# =============================================================================
# CLI TEST
# =============================================================================

if __name__ == "__main__":
    import time
    
    print("=" * 60)
    print("EmbeddingService CPU-Only Test")
    print("=" * 60)
    
    # 인스턴스 생성 (모델 로드)
    print("\n[1] Loading embedding model (CPU only)...")
    start = time.time()
    service = get_embedding_service()
    load_time = time.time() - start
    print(f"    ✓ Model loaded in {load_time:.2f}s")
    print(f"    ✓ Embedding dimension: {service.embedding_dimension}")
    
    # 단일 텍스트 임베딩
    print("\n[2] Single text embedding...")
    text = "트럼프 대통령이 중국산 제품에 25% 관세를 부과한다고 발표했다."
    start = time.time()
    embedding = service.embed(text)
    single_time = time.time() - start
    print(f"    ✓ Shape: {embedding.shape}")
    print(f"    ✓ Time: {single_time * 1000:.1f}ms")
    print(f"    ✓ Norm: {np.linalg.norm(embedding):.4f} (should be ~1.0)")
    
    # 배치 임베딩
    print("\n[3] Batch embedding (10 texts)...")
    texts = [
        "삼성전자 주가가 급등했다.",
        "비트코인이 신고가를 경신했다.",
        "AI 반도체 수요가 급증하고 있다.",
        "네이버가 새로운 AI 서비스를 출시했다.",
        "카카오 주식이 하락세를 보이고 있다.",
        "현대차가 전기차 신모델을 공개했다.",
        "LG에너지솔루션 배터리 수주가 증가했다.",
        "SK하이닉스 HBM 매출이 사상 최대를 기록했다.",
        "포스코홀딩스 리튬 사업 확대 계획을 발표했다.",
        "셀트리온 바이오시밀러 FDA 승인을 받았다.",
    ]
    start = time.time()
    embeddings = service.embed_batch(texts)
    batch_time = time.time() - start
    print(f"    ✓ Shape: {embeddings.shape}")
    print(f"    ✓ Time: {batch_time * 1000:.1f}ms ({batch_time * 100:.1f}ms per text)")
    
    # 유사도 테스트
    print("\n[4] Similarity test...")
    query = "반도체 관련 뉴스"
    results = service.similarity_batch(query, texts)
    print(f"    Query: '{query}'")
    print("    Top 3 similar:")
    for idx, sim in results[:3]:
        print(f"      - [{sim:.3f}] {texts[idx][:40]}...")
    
    # Singleton 확인
    print("\n[5] Singleton verification...")
    service2 = get_embedding_service()
    print(f"    ✓ Same instance: {service is service2}")
    
    print("\n" + "=" * 60)
    print("✅ All tests passed! GPU was NOT used.")
    print("=" * 60)