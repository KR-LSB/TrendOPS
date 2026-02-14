# src/trendops/pipeline.py
"""
TrendOps 통합 파이프라인

Blueprint Week 2: End-to-End MVP 파이프라인
전체 흐름: Trigger → Collect → Embed & Store → Retrieve → Analyze → Output

⚠️ 하드웨어 제약사항 (Blueprint Section 1.6):
- GPU (16GB): vLLM 단독 점유
- Embedding: CPU 전용 (8 threads)
- ChromaDB: SQLite 백엔드 (경량)

사용법:
    python -m trendops.pipeline "트럼프 관세"

    또는 Python에서:
        from trendops.pipeline import run_pipeline
        result = await run_pipeline("트럼프 관세")
"""
from __future__ import annotations

import asyncio
import time
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from trendops.config.settings import get_settings
from trendops.utils.logger import get_logger, setup_logging

logger = get_logger(__name__)


# =============================================================================
# Pipeline Result Models
# =============================================================================


class PipelineStageResult(BaseModel):
    """개별 단계 실행 결과"""

    stage: str = Field(..., description="단계 이름")
    success: bool = Field(..., description="성공 여부")
    duration_seconds: float = Field(..., description="소요 시간")
    data: dict[str, Any] = Field(default_factory=dict, description="단계별 데이터")
    error: str | None = Field(None, description="에러 메시지")


class PipelineResult(BaseModel):
    """파이프라인 전체 실행 결과"""

    keyword: str = Field(..., description="입력 키워드")
    success: bool = Field(..., description="전체 성공 여부")
    total_duration_seconds: float = Field(..., description="총 소요 시간")
    stages: list[PipelineStageResult] = Field(default_factory=list, description="단계별 결과")
    analysis: dict[str, Any] | None = Field(None, description="최종 분석 결과")
    started_at: datetime = Field(default_factory=datetime.now, description="시작 시간")

    def get_stage(self, stage_name: str) -> PipelineStageResult | None:
        """특정 단계 결과 조회"""
        for stage in self.stages:
            if stage.stage == stage_name:
                return stage
        return None


# =============================================================================
# Pipeline Implementation
# =============================================================================


async def run_pipeline(
    keyword: str,
    max_articles: int = 20,
    top_k_retrieve: int | None = None,
    skip_llm: bool = False,
) -> PipelineResult:
    """
    TrendOps 통합 파이프라인 실행

    실행 흐름:
    1. [Trigger] 키워드 입력 받음
    2. [Collect] RSS로 뉴스 수집
    3. [Embed & Store] 수집된 뉴스를 CPU로 임베딩하여 ChromaDB에 저장
    4. [Retrieve] ChromaDB에서 관련도 높은 문서 Top K 검색 (RAG 준비)
    5. [Analyze] 검색된 문서를 Context로 vLLM에 분석 요청
    6. [Output] 결과 반환

    Args:
        keyword: 분석할 키워드
        max_articles: 수집할 최대 기사 수 (기본 20)
        top_k_retrieve: RAG 검색 문서 수 (None이면 설정값 사용)
        skip_llm: LLM 분석 건너뛰기 (테스트용)

    Returns:
        PipelineResult: 파이프라인 실행 결과
    """
    settings = get_settings()

    if top_k_retrieve is None:
        top_k_retrieve = settings.pipeline_top_k_retrieve

    pipeline_start = time.time()
    stages: list[PipelineStageResult] = []

    logger.info("🚀 Pipeline started", extra={"keyword": keyword, "max_articles": max_articles})

    # =========================================================================
    # Stage 1: Trigger (키워드 입력)
    # =========================================================================
    stage_start = time.time()

    trigger_result = PipelineStageResult(
        stage="trigger",
        success=True,
        duration_seconds=round(time.time() - stage_start, 3),
        data={"keyword": keyword, "timestamp": datetime.now().isoformat()},
    )
    stages.append(trigger_result)

    logger.info(f"✅ [1/6] Trigger: keyword='{keyword}'")

    # =========================================================================
    # Stage 2: Collect (RSS 수집)
    # =========================================================================
    stage_start = time.time()

    try:
        # Lazy import to avoid circular dependencies
        from trendops.collector.collector_rss_google import GoogleNewsRSSCollector

        async with GoogleNewsRSSCollector() as collector:
            collection_result = await collector.fetch(keyword, max_results=max_articles)

        if not collection_result.success or not collection_result.articles:
            raise RuntimeError(
                f"RSS collection failed: {collection_result.error_message or 'No articles found'}"
            )

        articles = collection_result.articles

        collect_result = PipelineStageResult(
            stage="collect",
            success=True,
            duration_seconds=round(time.time() - stage_start, 3),
            data={
                "source": "google_news_rss",
                "article_count": len(articles),
                "articles": [
                    {"title": a.title, "link": a.link, "published": str(a.published)}
                    for a in articles
                ],
            },
        )
        stages.append(collect_result)

        logger.info(
            f"✅ [2/6] Collect: {len(articles)} articles from Google News RSS",
            extra={"article_count": len(articles)},
        )

    except Exception as e:
        logger.error(f"❌ [2/6] Collect failed: {e}")
        stages.append(
            PipelineStageResult(
                stage="collect",
                success=False,
                duration_seconds=round(time.time() - stage_start, 3),
                error=str(e),
            )
        )

        return PipelineResult(
            keyword=keyword,
            success=False,
            total_duration_seconds=round(time.time() - pipeline_start, 3),
            stages=stages,
        )

    # =========================================================================
    # Stage 3: Embed & Store (CPU 임베딩 → ChromaDB 저장)
    # =========================================================================
    stage_start = time.time()

    try:
        from trendops.service.embedding_service import get_embedding_service
        from trendops.store.vector_store import get_vector_store

        embedding_service = get_embedding_service()
        vector_store = get_vector_store()

        # 텍스트 준비 (제목 + 요약)
        texts = [
            f"{article.title}. {article.summary}" if article.summary else article.title
            for article in articles
        ]

        # 메타데이터 준비
        metadatas = [
            {
                "title": article.title,
                "source": article.source,
                "keyword": keyword,
                "link": article.link,
                "published_at": article.published.isoformat() if article.published else None,
            }
            for article in articles
        ]

        # CPU에서 배치 임베딩
        logger.info("  → Embedding on CPU...")
        embeddings = embedding_service.embed_batch(texts, show_progress=False)

        # ChromaDB에 저장
        logger.info("  → Storing in ChromaDB...")
        add_result = vector_store.add_documents(
            contents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            skip_duplicates=True,
        )

        embed_store_result = PipelineStageResult(
            stage="embed_store",
            success=True,
            duration_seconds=round(time.time() - stage_start, 3),
            data={
                "embedding_model": settings.embedding_model_name,
                "embedding_dimension": embedding_service.embedding_dimension,
                "documents_added": add_result.added_count,
                "documents_skipped": add_result.skipped_count,
                "total_in_store": vector_store.count,
            },
        )
        stages.append(embed_store_result)

        logger.info(
            f"✅ [3/6] Embed & Store: {add_result.added_count} added, "
            f"{add_result.skipped_count} skipped (total: {vector_store.count})",
            extra={
                "added": add_result.added_count,
                "skipped": add_result.skipped_count,
            },
        )

    except Exception as e:
        logger.error(f"❌ [3/6] Embed & Store failed: {e}")
        stages.append(
            PipelineStageResult(
                stage="embed_store",
                success=False,
                duration_seconds=round(time.time() - stage_start, 3),
                error=str(e),
            )
        )

        return PipelineResult(
            keyword=keyword,
            success=False,
            total_duration_seconds=round(time.time() - pipeline_start, 3),
            stages=stages,
        )

    # =========================================================================
    # Stage 4: Retrieve (RAG - 유사 문서 검색)
    # =========================================================================
    stage_start = time.time()

    try:
        # 키워드를 쿼리로 임베딩
        query_embedding = embedding_service.embed(keyword)

        # ChromaDB에서 검색
        search_results = vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k_retrieve,
        )

        if not search_results:
            logger.warning("No relevant documents found, using all collected articles")
            # Fallback: 수집된 기사 전체 사용
            retrieved_docs = [
                {"title": a.title, "content": f"{a.title}. {a.summary}", "similarity": 1.0}
                for a in articles[:top_k_retrieve]
            ]
        else:
            retrieved_docs = [
                {
                    "title": r.metadata.get("title", ""),
                    "content": r.content,
                    "similarity": r.similarity,
                    "source": r.metadata.get("source", "unknown"),
                }
                for r in search_results
            ]

        retrieve_result = PipelineStageResult(
            stage="retrieve",
            success=True,
            duration_seconds=round(time.time() - stage_start, 3),
            data={
                "top_k": top_k_retrieve,
                "retrieved_count": len(retrieved_docs),
                "documents": retrieved_docs,
            },
        )
        stages.append(retrieve_result)

        logger.info(
            f"✅ [4/6] Retrieve: Top {len(retrieved_docs)} documents retrieved",
            extra={"retrieved_count": len(retrieved_docs)},
        )

        # Top 문서 출력
        for i, doc in enumerate(retrieved_docs[:3], 1):
            logger.debug(f"    #{i} [{doc['similarity']:.3f}] {doc['title'][:50]}...")

    except Exception as e:
        logger.error(f"❌ [4/6] Retrieve failed: {e}")
        stages.append(
            PipelineStageResult(
                stage="retrieve",
                success=False,
                duration_seconds=round(time.time() - stage_start, 3),
                error=str(e),
            )
        )

        return PipelineResult(
            keyword=keyword,
            success=False,
            total_duration_seconds=round(time.time() - pipeline_start, 3),
            stages=stages,
        )

    # =========================================================================
    # Stage 5: Analyze (vLLM으로 분석)
    # =========================================================================
    stage_start = time.time()

    if skip_llm:
        logger.info("⏭️  [5/6] Analyze: Skipped (skip_llm=True)")
        stages.append(
            PipelineStageResult(
                stage="analyze",
                success=True,
                duration_seconds=0.0,
                data={"skipped": True},
            )
        )
        analysis_output = None
    else:
        try:
            from trendops.analyst.analyzer_llm import LLMAnalyzer

            # 검색된 문서를 LLM 분석용 형태로 변환
            analysis_articles = [
                {
                    "title": doc["title"],
                    "summary": doc["content"],
                    "source": doc.get("source", "unknown"),
                }
                for doc in retrieved_docs
            ]

            async with LLMAnalyzer() as analyzer:
                analysis_result = await analyzer.analyze(
                    keyword=keyword,
                    articles=analysis_articles,
                )

            analysis_output = {
                "main_cause": analysis_result.analysis.main_cause,
                "sentiment_ratio": {
                    "positive": analysis_result.analysis.sentiment_ratio.positive,
                    "negative": analysis_result.analysis.sentiment_ratio.negative,
                    "neutral": analysis_result.analysis.sentiment_ratio.neutral,
                },
                "key_opinions": analysis_result.analysis.key_opinions,
                "summary": analysis_result.analysis.summary,
                "model_version": analysis_result.model_version,
                "inference_time": analysis_result.inference_time_seconds,
                "is_valid": analysis_result.is_valid(),
            }

            analyze_stage = PipelineStageResult(
                stage="analyze",
                success=True,
                duration_seconds=round(time.time() - stage_start, 3),
                data=analysis_output,
            )
            stages.append(analyze_stage)

            logger.info(
                f"✅ [5/6] Analyze: LLM analysis complete "
                f"(inference: {analysis_result.inference_time_seconds:.2f}s)",
                extra={"inference_time": analysis_result.inference_time_seconds},
            )

        except Exception as e:
            logger.error(f"❌ [5/6] Analyze failed: {e}")
            stages.append(
                PipelineStageResult(
                    stage="analyze",
                    success=False,
                    duration_seconds=round(time.time() - stage_start, 3),
                    error=str(e),
                )
            )
            analysis_output = None

    # =========================================================================
    # Stage 6: Output (결과 반환)
    # =========================================================================
    stage_start = time.time()

    total_duration = round(time.time() - pipeline_start, 3)

    output_result = PipelineStageResult(
        stage="output",
        success=True,
        duration_seconds=round(time.time() - stage_start, 3),
        data={"total_duration": total_duration},
    )
    stages.append(output_result)

    # 최종 결과 생성
    pipeline_result = PipelineResult(
        keyword=keyword,
        success=all(s.success for s in stages),
        total_duration_seconds=total_duration,
        stages=stages,
        analysis=analysis_output,
    )

    logger.info(
        f"✅ [6/6] Output: Pipeline complete (total: {total_duration:.2f}s)",
        extra={"total_duration": total_duration, "success": pipeline_result.success},
    )

    return pipeline_result


# =============================================================================
# Pretty Print Functions
# =============================================================================


def print_pipeline_result(result: PipelineResult) -> None:
    """파이프라인 결과를 예쁘게 출력"""

    print("\n" + "=" * 70)
    print("  🎯 TrendOps Pipeline Result")
    print("=" * 70)

    # 기본 정보
    status = "✅ SUCCESS" if result.success else "❌ FAILED"
    print(f"\n  Keyword: {result.keyword}")
    print(f"  Status: {status}")
    print(f"  Total Time: {result.total_duration_seconds:.2f}s")
    print(f"  Started: {result.started_at.strftime('%Y-%m-%d %H:%M:%S')}")

    # 단계별 결과
    print("\n" + "-" * 70)
    print("  📊 Stage Results")
    print("-" * 70)

    stage_names = {
        "trigger": "1. Trigger",
        "collect": "2. Collect",
        "embed_store": "3. Embed & Store",
        "retrieve": "4. Retrieve",
        "analyze": "5. Analyze",
        "output": "6. Output",
    }

    for stage in result.stages:
        name = stage_names.get(stage.stage, stage.stage)
        status_icon = "✅" if stage.success else "❌"
        print(f"  {status_icon} {name}: {stage.duration_seconds:.3f}s")

        if stage.error:
            print(f"      └─ Error: {stage.error}")

    # 분석 결과 (있는 경우)
    if result.analysis:
        print("\n" + "-" * 70)
        print("  📝 Analysis Result")
        print("-" * 70)

        print("\n  🔍 핵심 원인:")
        print(f"     {result.analysis['main_cause']}")

        print("\n  📈 감성 비율:")
        sentiment = result.analysis["sentiment_ratio"]
        print(f"     🟢 긍정: {sentiment['positive']:.0%}")
        print(f"     🔴 부정: {sentiment['negative']:.0%}")
        print(f"     ⚪ 중립: {sentiment['neutral']:.0%}")

        print("\n  💬 핵심 의견:")
        for i, opinion in enumerate(result.analysis["key_opinions"], 1):
            print(f"     {i}. {opinion}")

        print("\n  📄 3줄 요약:")
        for line in result.analysis["summary"].split("\n"):
            print(f"     {line}")

        print(f"\n  ℹ️  Model: {result.analysis['model_version']}")
        print(f"     Inference: {result.analysis['inference_time']:.2f}s")
        print(f"     Valid: {'Yes' if result.analysis['is_valid'] else 'No'}")

    print("\n" + "=" * 70)


# =============================================================================
# CLI Entry Point
# =============================================================================


async def main(keyword: str, skip_llm: bool = False) -> None:
    """CLI 메인 함수"""

    # 로깅 설정
    setup_logging(level="INFO", enable_file=True)

    print("\n" + "=" * 70)
    print("  🚀 TrendOps Pipeline")
    print("=" * 70)
    print(f"\n  Keyword: {keyword}")
    print(f"  Skip LLM: {skip_llm}")
    print("\n  Starting pipeline...\n")

    try:
        result = await run_pipeline(
            keyword=keyword,
            skip_llm=skip_llm,
        )

        print_pipeline_result(result)

    except Exception as e:
        logger.exception(f"Pipeline failed with exception: {e}")
        raise


if __name__ == "__main__":
    import sys

    # 기본 키워드 또는 명령줄 인자
    test_keyword = sys.argv[1] if len(sys.argv) > 1 else "트럼프 관세"

    # --skip-llm 옵션 체크
    skip_llm = "--skip-llm" in sys.argv

    asyncio.run(main(test_keyword, skip_llm=skip_llm))
