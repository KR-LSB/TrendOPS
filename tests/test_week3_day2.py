#!/usr/bin/env python3
# tests/test_week3_day2.py
"""
Week 3 Day 2 통합 테스트: Semantic Deduplication

실행 방법:
    cd trendops
    poetry run python tests/test_week3_day2.py

테스트 항목:
1. VectorStore 기본 동작
2. SemanticDeduplicator 중복 제거
3. 유사도 임계값 검증
4. 배치 처리
5. 통계 확인
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path


def print_header(title: str) -> None:
    """섹션 헤더 출력"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_result(success: bool, message: str) -> None:
    """결과 출력"""
    icon = "✓" if success else "✗"
    print(f"    {icon} {message}")


async def test_vector_store() -> bool:
    """VectorStore 기본 동작 테스트"""
    print_header("Test 1: VectorStore Basic Operations")
    
    try:
        from trendops.store.vector_store import get_vector_store, VectorStore
        import numpy as np
        
        # 인스턴스 생성
        store = get_vector_store()
        print_result(True, f"VectorStore created: {type(store).__name__}")
        
        # 통계 확인
        stats = store.get_stats()
        print_result(True, f"Collection: {stats['collection_name']}")
        print_result(True, f"Initial count: {stats['total_documents']}")
        
        # 테스트 문서 추가
        keyword = "__test_vectorstore__"
        dim = 1024
        
        # 문서 추가
        test_text = "테스트 문서입니다."
        embedding = np.random.randn(dim).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)
        
        result = store.add_documents(
            contents=[test_text],
            embeddings=[embedding.tolist()],
            metadatas=[{"keyword": keyword, "title": "테스트", "source": "test"}],
        )
        print_result(result.added_count > 0, f"Document added: {result.doc_ids[0][:8] if result.doc_ids else 'N/A'}...")
        
        # 검색 테스트
        results = store.search_by_keyword(
            query_embedding=embedding.tolist(),
            keyword=keyword,
            top_k=1,
        )
        print_result(len(results) > 0, f"Search returned {len(results)} results")
        
        if results:
            print_result(
                results[0].similarity > 0.9,
                f"High similarity found: {results[0].similarity:.2%}"
            )
        
        # 클린업
        deleted = store.delete_by_keyword(keyword)
        print_result(deleted > 0, f"Cleanup: deleted {deleted} documents")
        
        # Singleton 확인
        store2 = get_vector_store()
        print_result(store is store2, "Singleton pattern verified")
        
        return True
        
    except Exception as e:
        print_result(False, f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_deduplicator_basic() -> bool:
    """SemanticDeduplicator 기본 동작 테스트"""
    print_header("Test 2: SemanticDeduplicator Basic Operations")
    
    try:
        from trendops.service.deduplicator import get_deduplicator, SemanticDeduplicator
        
        # 인스턴스 생성
        dedup = get_deduplicator()
        print_result(True, f"Deduplicator created: {type(dedup).__name__}")
        print_result(True, f"Threshold: {dedup.SIMILARITY_THRESHOLD:.0%}")
        
        # 초기 통계
        initial_stats = dedup.get_stats()
        print_result(True, f"Initial stats: processed={initial_stats.total_processed}")
        
        # 테스트 문서 추가
        keyword = "__test_dedup__"
        
        # 첫 번째 문서 (새로 추가)
        text1 = "트럼프 대통령이 중국산 제품에 25% 관세를 부과한다고 발표했다."
        is_added, reason = await dedup.add_if_unique(
            text=text1,
            metadata={"keyword": keyword, "title": "트럼프 관세", "source": "test"},
        )
        print_result(is_added, f"First document: {reason}")
        
        # 동일한 문서 (중복)
        is_added2, reason2 = await dedup.add_if_unique(
            text=text1,
            metadata={"keyword": keyword, "title": "트럼프 관세2", "source": "test"},
        )
        print_result(not is_added2, f"Same document: {reason2}")
        
        # 다른 주제 문서 (새로 추가)
        text3 = "삼성전자가 새로운 반도체 공장을 착공했다."
        is_added3, reason3 = await dedup.add_if_unique(
            text=text3,
            metadata={"keyword": keyword, "title": "삼성 반도체", "source": "test"},
        )
        print_result(is_added3, f"Different topic: {reason3}")
        
        # 통계 확인
        stats = dedup.get_stats()
        print_result(
            stats.total_processed > initial_stats.total_processed,
            f"Stats updated: processed={stats.total_processed}, "
            f"unique={stats.unique_added}, duplicates={stats.duplicates_filtered}"
        )
        print_result(True, f"Dedup ratio: {stats.dedup_ratio:.1%}")
        
        # 클린업
        store = dedup.vector_store
        deleted = store.delete_by_keyword(keyword)
        print_result(deleted > 0, f"Cleanup: deleted {deleted} documents")
        
        return True
        
    except Exception as e:
        print_result(False, f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_similarity_threshold() -> bool:
    """유사도 임계값 동작 테스트"""
    print_header("Test 3: Similarity Threshold Behavior")
    
    try:
        from trendops.service.deduplicator import get_deduplicator
        
        dedup = get_deduplicator()
        keyword = "__test_threshold__"
        
        # 원본 문서
        original = "미국 대통령이 새로운 무역 정책을 발표했습니다."
        
        # 원본 추가
        is_added, _ = await dedup.add_if_unique(
            text=original,
            metadata={"keyword": keyword, "title": "원본", "source": "test"},
        )
        print_result(is_added, f"Original added: {original[:40]}...")
        
        # 유사한 문서들 테스트
        similar_texts = [
            ("미국 대통령, 새 무역 정책 발표", "매우 유사"),
            ("삼성전자 주가 상승", "완전히 다름"),
        ]
        
        for text, expected in similar_texts:
            is_dup, similarity = await dedup.check_duplicate(text, keyword)
            sim_str = f"{similarity:.2%}" if similarity else "N/A"
            result_str = "Duplicate" if is_dup else "Unique"
            print_result(
                True,
                f"{expected}: [{result_str}] sim={sim_str} - {text[:30]}..."
            )
        
        # 클린업
        store = dedup.vector_store
        deleted = store.delete_by_keyword(keyword)
        print_result(deleted > 0, f"Cleanup: deleted {deleted} documents")
        
        return True
        
    except Exception as e:
        print_result(False, f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_batch_deduplication() -> bool:
    """배치 중복 제거 테스트"""
    print_header("Test 4: Batch Deduplication")
    
    try:
        from trendops.service.deduplicator import get_deduplicator
        
        dedup = get_deduplicator()
        keyword = "__test_batch__"
        
        # 배치 테스트 데이터
        batch_items = [
            ("AI 기술이 빠르게 발전하고 있다.", {"keyword": keyword, "title": "AI1", "source": "test"}),
            ("인공지능 기술이 급속히 발전 중이다.", {"keyword": keyword, "title": "AI2", "source": "test"}),
            ("전기차 시장이 성장하고 있다.", {"keyword": keyword, "title": "EV1", "source": "test"}),
            ("EV 시장이 확대되고 있다.", {"keyword": keyword, "title": "EV2", "source": "test"}),
            ("반도체 수요가 증가했다.", {"keyword": keyword, "title": "반도체", "source": "test"}),
        ]
        
        # 배치 처리
        results = await dedup.add_batch_unique(batch_items)
        
        added = sum(1 for r in results if r.is_added)
        duplicates = len(results) - added
        
        print_result(True, f"Batch processed: {len(results)} items")
        print_result(True, f"Added: {added}")
        print_result(True, f"Duplicates filtered: {duplicates}")
        
        # 개별 결과 출력
        for i, r in enumerate(results):
            status = "Added" if r.is_added else "Duplicate"
            print(f"        [{i+1}] {status}: {batch_items[i][0][:30]}...")
        
        # 클린업
        store = dedup.vector_store
        deleted = store.delete_by_keyword(keyword)
        print_result(deleted > 0, f"Cleanup: deleted {deleted} documents")
        
        return True
        
    except Exception as e:
        print_result(False, f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_storage_stats() -> bool:
    """저장소 절약 통계 테스트"""
    print_header("Test 5: Storage Savings Statistics")
    
    try:
        from trendops.service.deduplicator import get_deduplicator
        
        dedup = get_deduplicator()
        dedup.reset_stats()  # 통계 초기화
        
        keyword = "__test_stats__"
        
        # 의도적으로 중복 많은 데이터셋
        articles = [
            ("삼성전자가 신제품을 출시했다.", "삼성1"),
            ("삼성전자 신제품 출시 소식", "삼성2"),
            ("삼성전자, 새 제품 발표", "삼성3"),
            ("LG전자가 OLED TV를 공개했다.", "LG1"),
            ("LG전자 OLED TV 공개", "LG2"),
            ("SK하이닉스 HBM 수주 급증", "SK"),
            ("네이버가 AI 서비스를 시작했다.", "네이버1"),
            ("네이버 AI 서비스 런칭", "네이버2"),
        ]
        
        for text, title in articles:
            await dedup.add_if_unique(
                text=text,
                metadata={"keyword": keyword, "title": title, "source": "test"},
            )
        
        # 최종 통계
        stats = dedup.get_stats()
        
        print_result(True, f"Total processed: {stats.total_processed}")
        print_result(True, f"Unique added: {stats.unique_added}")
        print_result(True, f"Duplicates filtered: {stats.duplicates_filtered}")
        print_result(True, f"Dedup ratio: {stats.dedup_ratio:.1%}")
        print_result(True, f"Storage saved: {stats.storage_saved_percent:.1f}%")
        
        # 클린업
        store = dedup.vector_store
        deleted = store.delete_by_keyword(keyword)
        print_result(deleted > 0, f"Cleanup: deleted {deleted} documents")
        
        return True
        
    except Exception as e:
        print_result(False, f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """메인 테스트 실행"""
    print("\n" + "=" * 70)
    print("  Week 3 Day 2: Semantic Deduplication Test Suite")
    print("=" * 70)
    
    tests = [
        ("VectorStore Basic", test_vector_store),
        ("Deduplicator Basic", test_deduplicator_basic),
        ("Similarity Threshold", test_similarity_threshold),
        ("Batch Deduplication", test_batch_deduplication),
        ("Storage Statistics", test_storage_stats),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            success = await test_func()
            results.append((name, success))
        except Exception as e:
            print_result(False, f"{name} failed with exception: {e}")
            results.append((name, False))
    
    # 최종 결과 요약
    print_header("Test Results Summary")
    
    passed = sum(1 for _, s in results if s)
    failed = len(results) - passed
    
    for name, success in results:
        status = "PASS" if success else "FAIL"
        print(f"    [{status}] {name}")
    
    print()
    print(f"    Total: {len(results)} | Passed: {passed} | Failed: {failed}")
    
    if failed == 0:
        print("\n" + "=" * 70)
        print("  ✅ All tests passed! Semantic Deduplication is ready.")
        print("=" * 70 + "\n")
        return 0
    else:
        print("\n" + "=" * 70)
        print(f"  ❌ {failed} test(s) failed. Please review the errors above.")
        print("=" * 70 + "\n")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)