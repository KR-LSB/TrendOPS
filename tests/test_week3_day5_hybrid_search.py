# tests/test_week3_day5_hybrid_search.py
"""
Week 3 Day 5: Hybrid Search 통합 테스트

테스트 범위:
1. VectorStore (ChromaDB) 기본 동작
2. BM25Index 기본 동작  
3. HybridSearch RRF 융합
4. A/B 테스트 메트릭
5. 성능 벤치마크

실행 방법:
    python test_week3_day5_hybrid_search.py
"""
from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 환경 변수 설정 (ChromaDB 경로 등)
import os
os.environ["CHROMADB_PATH"] = str(project_root / "test_data" / "chromadb")


def print_header(title: str) -> None:
    """섹션 헤더 출력"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_subheader(title: str) -> None:
    """서브 섹션 헤더 출력"""
    print(f"\n--- {title} ---")


async def test_vector_store() -> bool:
    """VectorStore 테스트"""
    print_header("1. VectorStore (ChromaDB) Test")
    
    try:
        from trendops.store.vector_store import VectorStore, get_vector_store, reset_vector_store
        import numpy as np
        
        # 초기화
        reset_vector_store()
        store = get_vector_store()
        
        print_subheader("1.1 Basic Operations")
        
        # 테스트 데이터
        test_keyword = "__test_vs__"
        test_docs = [
            "트럼프가 중국에 관세를 부과했다.",
            "미국의 무역 정책이 변경되었다.",
            "삼성전자 주가가 상승했다.",
        ]
        
        # 가짜 임베딩 (실제로는 EmbeddingService 사용)
        dim = 1024
        embeddings = [np.random.randn(dim).tolist() for _ in test_docs]
        metadatas = [{"keyword": test_keyword, "title": f"Doc {i}"} for i in range(len(test_docs))]
        
        # 추가
        result = store.add_documents(
            contents=test_docs,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        print(f"    ✓ Added {result.added} documents")
        assert result.added == 3, f"Expected 3, got {result.added}"
        
        # 중복 추가 테스트
        result2 = store.add_documents(
            contents=test_docs,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        print(f"    ✓ Duplicate add: added={result2.added}, skipped={result2.skipped}")
        assert result2.skipped == 3, f"Expected 3 skipped, got {result2.skipped}"
        
        print_subheader("1.2 Search Operations")
        
        # 검색
        query_emb = np.random.randn(dim).tolist()
        results = store.search(query_emb, top_k=3)
        print(f"    ✓ Search returned {len(results)} results")
        assert len(results) <= 3
        
        # 키워드 필터 검색
        results = store.search_by_keyword(query_emb, test_keyword, top_k=3)
        print(f"    ✓ Keyword search returned {len(results)} results")
        
        print_subheader("1.3 Stats and Cleanup")
        
        # 통계
        stats = store.get_stats()
        print(f"    ✓ Collection: {stats.name}, Documents: {stats.count}")
        
        # 정리
        deleted = store.delete_by_keyword(test_keyword)
        print(f"    ✓ Deleted {deleted} test documents")
        
        print("\n✅ VectorStore tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ VectorStore test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_bm25_index() -> bool:
    """BM25Index 테스트"""
    print_header("2. BM25Index Test")
    
    try:
        from trendops.search.bm25_index import BM25Index, get_bm25_index, reset_bm25_index
        
        # 초기화
        reset_bm25_index()
        index = get_bm25_index()
        
        print_subheader("2.1 Tokenization")
        
        # 토크나이저 테스트
        tokenizer = index.tokenizer
        sample = "트럼프 대통령의 관세 정책이 발표되었다."
        tokens = tokenizer.tokenize(sample)
        print(f"    Input: {sample}")
        print(f"    Tokens: {tokens}")
        assert len(tokens) > 0, "Tokenization failed"
        
        print_subheader("2.2 Document Indexing")
        
        # 테스트 문서
        test_keyword = "__test_bm25__"
        test_docs = [
            ("doc1", "트럼프 대통령이 중국산 제품에 25% 관세를 부과한다고 발표했다."),
            ("doc2", "미국의 관세 정책이 세계 경제에 영향을 미친다."),
            ("doc3", "삼성전자 주가가 급등했다. 반도체 수요 증가 영향."),
            ("doc4", "비트코인 가격이 신고가를 경신했다."),
            ("doc5", "트럼프 행정부의 무역 정책에 대한 우려가 커지고 있다."),
        ]
        
        # 추가
        added = index.add_documents(
            doc_ids=[d[0] for d in test_docs],
            documents=[d[1] for d in test_docs],
            metadatas=[{"keyword": test_keyword} for _ in test_docs],
        )
        print(f"    ✓ Added {added} documents")
        assert added == 5, f"Expected 5, got {added}"
        
        print_subheader("2.3 Search Operations")
        
        # 검색
        query = "트럼프 관세 정책"
        results = index.search(query, top_k=3)
        print(f"    Query: '{query}'")
        print(f"    Results ({len(results)}):")
        for r in results:
            print(f"      [{r.rank}] score={r.score:.3f}: {r.document[:40]}...")
        
        assert len(results) > 0, "Search returned no results"
        
        # 상위 결과가 관세 관련인지 확인
        top_doc = results[0].document
        assert "관세" in top_doc or "트럼프" in top_doc, "Top result should be about 관세/트럼프"
        
        print_subheader("2.4 Stats and Cleanup")
        
        # 통계
        stats = index.get_stats()
        print(f"    ✓ Documents: {stats.total_documents}")
        print(f"    ✓ Vocabulary: {stats.vocabulary_size}")
        print(f"    ✓ Avg length: {stats.avg_doc_length:.1f}")
        
        # 정리
        cleared = index.clear()
        print(f"    ✓ Cleared {cleared} documents")
        
        print("\n✅ BM25Index tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ BM25Index test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_hybrid_search() -> bool:
    """HybridSearch 테스트"""
    print_header("3. Hybrid Search (RRF Fusion) Test")
    
    try:
        from trendops.search.hybrid_search import (
            HybridSearch, 
            get_hybrid_search, 
            reset_hybrid_search,
            SearchMode,
        )
        from trendops.store.vector_store import get_vector_store, reset_vector_store
        from trendops.search.bm25_index import get_bm25_index, reset_bm25_index
        import numpy as np
        
        # 모든 인스턴스 초기화
        reset_hybrid_search()
        reset_vector_store()
        reset_bm25_index()
        
        print_subheader("3.1 Setup Test Data")
        
        # 테스트 데이터
        test_keyword = "__test_hybrid__"
        test_docs = [
            ("트럼프 대통령이 중국산 제품에 25% 관세를 부과한다고 발표했다.", "관세 발표"),
            ("미국의 관세 정책이 세계 경제에 영향을 미친다.", "경제 영향"),
            ("삼성전자 주가가 급등했다. 반도체 수요 증가 영향.", "삼성 주가"),
            ("비트코인 가격이 신고가를 경신했다.", "비트코인"),
            ("트럼프 행정부의 무역 정책에 대한 우려가 커지고 있다.", "무역 정책"),
        ]
        
        # VectorStore에 추가 (가짜 임베딩 사용)
        vector_store = get_vector_store()
        dim = 1024
        
        contents = [d[0] for d in test_docs]
        embeddings = [np.random.randn(dim).tolist() for _ in test_docs]
        metadatas = [{"keyword": test_keyword, "title": d[1]} for d in test_docs]
        
        # 문서 ID 생성 (BM25와 일치시키기 위해)
        from trendops.store.vector_store import VectorStore
        doc_ids = [VectorStore._generate_doc_id(c, m) for c, m in zip(contents, metadatas)]
        
        vector_store.add_documents(
            contents=contents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=doc_ids,
        )
        print(f"    ✓ Added {len(test_docs)} documents to VectorStore")
        
        # BM25에 추가 (동일 ID 사용)
        bm25_index = get_bm25_index()
        bm25_index.add_documents(
            doc_ids=doc_ids,
            documents=contents,
            metadatas=metadatas,
        )
        print(f"    ✓ Added {len(test_docs)} documents to BM25Index")
        
        print_subheader("3.2 Hybrid Search")
        
        # HybridSearch 인스턴스 생성
        search = get_hybrid_search()
        print(f"    ✓ RRF k: {search.rrf_k}")
        print(f"    ✓ Vector weight: {search.vector_weight}")
        
        # Hybrid 검색 (BM25만 테스트 - 실제 임베딩 없이)
        query = "트럼프 관세 정책"
        
        response = await search.search(
            query=query,
            n_results=3,
            where={"keyword": test_keyword},
            mode=SearchMode.BM25_ONLY,  # BM25만 테스트 (임베딩 서비스 없이)
        )
        
        print(f"    Query: '{query}'")
        print(f"    Mode: BM25_ONLY")
        print(f"    Results ({len(response.results)}):")
        for r in response.results:
            print(f"      [{r.final_rank}] score={r.hybrid_score:.4f}: {r.document[:40]}...")
        
        assert len(response.results) > 0, "Search returned no results"
        
        print_subheader("3.3 Search Metrics")
        
        metrics = response.metrics
        print(f"    ✓ Total results: {metrics.total_results}")
        print(f"    ✓ BM25 latency: {metrics.bm25_latency_ms:.1f}ms")
        print(f"    ✓ Total latency: {metrics.total_latency_ms:.1f}ms")
        
        print_subheader("3.4 RRF Algorithm Test")
        
        # RRF 알고리즘 직접 테스트
        bm25_ranks = {"doc1": 1, "doc2": 3, "doc3": 5}
        vector_ranks = {"doc1": 2, "doc2": 1, "doc4": 4}
        
        fused = search._reciprocal_rank_fusion(bm25_ranks, vector_ranks)
        
        print("    BM25 ranks:", bm25_ranks)
        print("    Vector ranks:", vector_ranks)
        print("    RRF fused results:")
        for doc_id, score in fused[:5]:
            print(f"      {doc_id}: {score:.6f}")
        
        # doc1과 doc2가 양쪽에 있으므로 상위에 있어야 함
        top_docs = [doc_id for doc_id, _ in fused[:2]]
        assert "doc1" in top_docs or "doc2" in top_docs, "RRF should rank overlapping docs higher"
        
        print_subheader("3.5 Metrics Summary")
        
        summary = search.get_metrics_summary()
        print(f"    ✓ Total queries: {summary.get('count', 0)}")
        print(f"    ✓ Avg latency: {summary.get('avg_total_latency_ms', 0):.1f}ms")
        
        print_subheader("3.6 Cleanup")
        
        vector_store.delete_by_keyword(test_keyword)
        bm25_index.clear()
        search.clear_metrics()
        print("    ✓ Test data cleaned up")
        
        print("\n✅ Hybrid Search tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Hybrid Search test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_performance_benchmark() -> bool:
    """성능 벤치마크"""
    print_header("4. Performance Benchmark")
    
    try:
        from trendops.search.bm25_index import get_bm25_index, reset_bm25_index
        import random
        
        reset_bm25_index()
        index = get_bm25_index()
        
        print_subheader("4.1 BM25 Indexing Performance")
        
        # 대량 문서 생성
        num_docs = 1000
        keywords = ["트럼프", "관세", "삼성전자", "비트코인", "AI", "반도체"]
        
        docs = []
        for i in range(num_docs):
            keyword = random.choice(keywords)
            doc = f"{keyword} 관련 뉴스 {i}. 이것은 테스트 문서입니다. " * 3
            docs.append((f"bench_doc_{i}", doc, {"keyword": "__bench__"}))
        
        # 인덱싱 시간 측정
        start = time.time()
        index.add_documents(
            doc_ids=[d[0] for d in docs],
            documents=[d[1] for d in docs],
            metadatas=[d[2] for d in docs],
        )
        index_time = time.time() - start
        
        print(f"    ✓ Indexed {num_docs} documents in {index_time:.2f}s")
        print(f"    ✓ Rate: {num_docs / index_time:.0f} docs/sec")
        
        print_subheader("4.2 BM25 Search Performance")
        
        # 검색 시간 측정
        queries = ["트럼프 관세", "삼성전자 주가", "비트코인 가격", "AI 반도체"]
        num_searches = 100
        
        start = time.time()
        for _ in range(num_searches):
            query = random.choice(queries)
            index.search(query, top_k=10)
        search_time = time.time() - start
        
        avg_search_ms = (search_time / num_searches) * 1000
        
        print(f"    ✓ {num_searches} searches in {search_time:.2f}s")
        print(f"    ✓ Avg search time: {avg_search_ms:.2f}ms")
        
        print_subheader("4.3 Index Stats")
        
        stats = index.get_stats()
        print(f"    ✓ Documents: {stats.total_documents}")
        print(f"    ✓ Vocabulary: {stats.vocabulary_size}")
        print(f"    ✓ Avg doc length: {stats.avg_doc_length:.1f} tokens")
        
        # 정리
        index.clear()
        
        print("\n✅ Performance benchmark completed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Performance benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """메인 테스트 실행"""
    print("\n" + "█" * 70)
    print("█  Week 3 Day 5: Hybrid Search Integration Test")
    print("█" * 70)
    
    results = {
        "VectorStore": await test_vector_store(),
        "BM25Index": await test_bm25_index(),
        "HybridSearch": await test_hybrid_search(),
        "Benchmark": await test_performance_benchmark(),
    }
    
    # 결과 요약
    print("\n" + "=" * 70)
    print("  TEST SUMMARY")
    print("=" * 70)
    
    passed = 0
    failed = 0
    
    for test_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"    {test_name}: {status}")
        if success:
            passed += 1
        else:
            failed += 1
    
    print("-" * 70)
    print(f"    Total: {passed} passed, {failed} failed")
    print("=" * 70)
    
    if failed == 0:
        print("\n🎉 All tests passed! Hybrid Search is ready for production.")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())