# scripts/real_e2e_pipeline.py
"""
TrendOps Real End-to-End Pipeline (RAG Enabled)

Upgrade Week 5:
- Hybrid Search 기반 RAG (검색 증강 생성) 적용
- 전체 키워드(Top 10) 수집 및 분석 자동화
- 일일 리포트 데이터 자동 저장
"""
import asyncio
import json
import os
import sys
import time
from pathlib import Path

# 프로젝트 루트 경로 설정
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from trendops.collector.collector_rss import RSSCollector
from trendops.collector.collector_youtube import YouTubeCollector

# 검색 및 인덱싱 모듈
from trendops.search.bm25_index import get_bm25_index
from trendops.search.hybrid_search import SearchMode, get_hybrid_search
from trendops.service.deduplicator import get_deduplicator
from trendops.trigger.trigger_google import GoogleTrendTrigger
from trendops.utils.logger import get_logger

# [추가됨] 리포트 서비스 (일일 리포트 생성을 위한 데이터 저장)
try:
    from trendops.publisher.report_service import ReportService

    report_service = ReportService()
except ImportError:
    report_service = None

# Rich 라이브러리 설정 (Table 추가)
try:
    from rich import print as rprint
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.tree import Tree

    console = Console()
except ImportError:
    console = None

logger = get_logger("pipeline_e2e")


def print_stage(title: str):
    if console:
        console.print(Panel(f"[bold white]{title}[/bold white]", style="bold blue"))
    else:
        print(f"\n━━━ {title} ━━━")


def print_success(msg: str):
    if console:
        console.print(f"[bold green]✓ {msg}[/bold green]")
    else:
        print(f"✓ {msg}")


def print_error(msg: str):
    if console:
        console.print(f"[bold red]✗ {msg}[/bold red]")
    else:
        print(f"✗ {msg}")


# =============================================================================
# Stage 1: Trigger
# =============================================================================
async def stage_trigger(max_keywords: int = 10):
    print_stage("Stage 1: TRIGGER - 트렌드 감지")
    try:
        trigger = GoogleTrendTrigger()
        trend_keywords = await trigger.fetch_trends()

        # 전체 키워드 가져오기
        keywords = [
            {"keyword": tk.keyword, "score": tk.trend_score, "source": tk.source}
            for tk in trend_keywords[:max_keywords]
        ]

        print_success(f"Google Trends에서 {len(keywords)}개 키워드 감지")

        # 결과 테이블 출력
        if console and keywords:
            table = Table(show_header=True, header_style="bold cyan")
            table.add_column("Rank", width=6)
            table.add_column("Keyword", width=25)
            table.add_column("Score", width=10)
            table.add_column("Source", width=15)

            for i, kw in enumerate(keywords, 1):
                source_str = kw.get("source", "unknown")
                if hasattr(source_str, "value"):
                    source_str = source_str.value
                table.add_row(f"#{i}", kw["keyword"], f"{kw['score']:.1f}", str(source_str))

            console.print(table)

        return keywords
    except Exception as e:
        print_error(f"트리거 실패: {e}")
        return []


# =============================================================================
# Stage 2: Collector (Hybrid)
# =============================================================================
async def stage_collection(keywords: list[dict], max_articles: int = 15):
    print_stage("Stage 2: COLLECTOR - 뉴스/유튜브 수집")
    start = time.time()
    all_articles = []

    # [수정 1] 상위 3개 제한 제거 -> 전체 키워드 대상 수집
    target_keywords = [kw["keyword"] for kw in keywords]

    # 1. RSS 뉴스 수집 (전체)
    try:
        print(f"📡 RSS 뉴스 수집 시작... ({len(target_keywords)}개 키워드)")
        async with RSSCollector(max_results=max_articles) as rss:
            for kw in target_keywords:
                docs = await rss.fetch(kw)
                for doc in docs:
                    all_articles.append(doc_to_dict(doc))
                await asyncio.sleep(0.1)
    except Exception as e:
        print_error(f"RSS 수집 에러: {e}")

    # 2. YouTube 댓글 수집 (시간 관계상 상위 3개만)
    try:
        yt_targets = target_keywords[:3]  # 유튜브는 상위 3개만
        if yt_targets:
            print(f"🎬 YouTube 여론 수집 시작... (상위 {len(yt_targets)}개)")
            async with YouTubeCollector(headless=True) as yt:
                for kw in yt_targets:
                    # 키워드별로 순차 수집
                    yt_docs = await yt.fetch(keyword=kw, max_videos=2, comments_per_video=5)
                    if yt_docs:
                        for doc in yt_docs:
                            all_articles.append(doc_to_dict(doc))
                        print(f"   - '{kw}': 댓글 {len(yt_docs)}개")
    except Exception as e:
        print_error(f"YouTube 수집 에러: {e}")

    if console and all_articles:
        tree = Tree(f"📦 수집 결과 (총 {len(all_articles)}건)")
        sources = {}
        for a in all_articles:
            src = a.get("source", "unknown")
            sources[src] = sources.get(src, 0) + 1
        for src, cnt in sources.items():
            tree.add(f"[yellow]{src}[/]: {cnt}건")
        console.print(tree)

    return all_articles


def doc_to_dict(doc):
    return {
        "title": doc.title,
        "link": doc.link,
        "summary": doc.summary,
        "keyword": doc.keyword,
        "source": doc.source,
        "published": str(doc.published),
        "metadata": doc.metadata,
    }


# =============================================================================
# Stage 3: Deduplication & Indexing
# =============================================================================
async def stage_deduplication_and_indexing(articles: list[dict]):
    print_stage("Stage 3: DEDUPLICATION & INDEXING")
    if not articles:
        return []

    try:
        deduplicator = get_deduplicator()
        dedup_items = []

        for art in articles:
            text = f"{art.get('title','')} {art.get('summary','')}"
            meta = {
                "keyword": art.get("keyword", "unknown"),
                "title": art.get("title", "")[:100],
                "source": str(art.get("source", "unknown")),
                "link": art.get("link", ""),
                "published": str(art.get("published", "")),
            }
            if "metadata" in art and isinstance(art["metadata"], dict):
                for k, v in art["metadata"].items():
                    meta[k] = str(v)

            dedup_items.append((text, meta))

        results = await deduplicator.add_batch_unique(items=dedup_items)

        unique_articles = []
        new_doc_ids = []
        new_docs_content = []
        new_docs_meta = []

        for art, res, item in zip(articles, results, dedup_items):
            if res.is_added:
                unique_articles.append(art)
                if res.doc_id:
                    new_doc_ids.append(res.doc_id)
                    new_docs_content.append(item[0])
                    new_docs_meta.append(item[1])

        print_success(
            f"중복 제거: {len(articles)}건 → {len(unique_articles)}건 (신규 {len(unique_articles)}건)"
        )

        if new_doc_ids:
            bm25 = get_bm25_index()
            added_count = bm25.add_documents(
                doc_ids=new_doc_ids, documents=new_docs_content, metadatas=new_docs_meta
            )
            print_success(f"BM25 인덱싱: {added_count}건 추가 완료")

        return unique_articles

    except Exception as e:
        print_error(f"처리 실패: {e}")
        return articles


# =============================================================================
# Stage 3.5: RAG Search
# =============================================================================
async def stage_rag_search(keyword: str, current_articles: list[dict]):
    try:
        search_engine = get_hybrid_search()
        response = await search_engine.search(query=keyword, n_results=5, mode=SearchMode.HYBRID)

        context_docs = []
        if response.results:
            for res in response.results:
                context_docs.append(
                    {
                        "content": res.document[:200] + "...",
                        "source": res.metadata.get("source", "unknown"),
                        "date": res.metadata.get("published", "unknown"),
                    }
                )
        return context_docs
    except Exception:
        return []


# =============================================================================
# Stage 4: LLM Analysis (Self-Correction & Guardrail)
# =============================================================================
async def stage_llm_analysis(
    keyword: str,
    articles: list[dict],
    ollama_url: str = "http://localhost:11434",
    model: str = "exaone3.5",
    max_retries: int = 3,
) -> tuple[dict | None, dict]:
    start = time.time()

    # 1. 컨텍스트 구성
    context_parts = []
    for i, article in enumerate(articles[:10], 1):
        pub_date = article.get("published", "")[:10]
        context_parts.append(
            f"[{i}] {article['title']} ({pub_date})\n   - {article['summary'][:200]}"
        )
    context = "\n".join(context_parts)

    # 2. 시스템 프롬프트
    system_prompt = """당신은 냉철한 트렌드 분석 AI입니다.
객관적인 사실에 기반하여 분석하고, 감정적이거나 편향된 표현을 배제하세요.

[중요]
1. 입력된 뉴스가 영어라도, 분석 결과는 반드시 **한국어(Korean)**로 작성해야 합니다.
2. 전문 용어는 괄호 안에 영문을 병기할 수 있습니다.

반드시 아래 JSON 형식으로만 응답하세요:
{
  "main_cause": "핵심 원인 (1문장, 한국어)",
  "sentiment": {"positive": 0.0~1.0, "negative": 0.0~1.0, "neutral": 0.0~1.0},
  "key_opinions": ["핵심 여론 1 (한국어)", "핵심 여론 2 (한국어)", "핵심 여론 3 (한국어)"],
  "summary": "전체 요약 (3문장 이상, 한국어)"
}
"""
    base_user_prompt = (
        f"키워드: '{keyword}'\n\n[관련 문서]\n{context}\n\n위 문서를 분석하여 JSON으로 출력하세요."
    )

    try:
        from trendops.analyst.guardrail import ContentGuardrail, GuardrailAction

        guardrail = ContentGuardrail(use_mock=False)
    except ImportError:
        guardrail = None

    import aiohttp

    current_prompt = base_user_prompt
    final_analysis = None

    async with aiohttp.ClientSession() as session:
        for attempt in range(1, max_retries + 1):
            try:
                payload = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": current_prompt},
                    ],
                    "stream": False,
                    "format": "json",
                    "options": {"temperature": 0.2},
                }
                async with session.post(f"{ollama_url}/api/chat", json=payload) as resp:
                    if resp.status != 200:
                        raise Exception(f"API Error {resp.status}")
                    data = await resp.json()
                    response_text = data["message"]["content"]

                try:
                    parsed_data = json.loads(response_text)
                except json.JSONDecodeError:
                    current_prompt = (
                        base_user_prompt + "\n\n🚨 JSON 형식이 잘못되었습니다. 다시 작성하세요."
                    )
                    continue

                if guardrail:
                    check = await guardrail.check(parsed_data.get("summary", ""), keyword=keyword)
                    if check.action == GuardrailAction.REJECT:
                        current_prompt = (
                            base_user_prompt
                            + f"\n\n🚨 안전성 위배: {check.issue_summary}. 수정하세요."
                        )
                        continue
                    elif check.action == GuardrailAction.REVISE:
                        parsed_data["summary"] = check.revised_content

                final_analysis = parsed_data
                break
            except Exception as e:
                if attempt == max_retries:
                    print(f"   ⚠️ 분석 실패: {e}")

    duration = time.time() - start

    if final_analysis:
        if console:
            s = final_analysis.get("sentiment", {})
            pos, neg, neu = s.get("positive", 0), s.get("negative", 0), s.get("neutral", 0)
            console.print(
                Panel(
                    f"[bold]📌 핵심 원인[/]\n{final_analysis.get('main_cause', '-')}\n\n"
                    f"[bold]📝 요약[/]\n{final_analysis.get('summary', '-')}\n\n"
                    f"[bold]📊 감성 분포[/]\n😊 {pos:.0%} | 😠 {neg:.0%} | 😐 {neu:.0%}\n\n"
                    f"[bold]💡 핵심 포인트[/]\n"
                    + "\n".join(f"• {op}" for op in final_analysis.get("key_opinions", [])[:3]),
                    title=f"📊 분석 결과: {keyword}",
                    border_style="green",
                )
            )

    return final_analysis, {"duration": duration}


# =============================================================================
# Main Pipeline (수정된 핵심 로직)
# =============================================================================
async def run_real_pipeline(
    keywords: list[str] | None = None,
    max_keywords: int = 10,
    max_articles: int = 15,
    ollama_url: str = os.getenv("OLLAMA_HOST", "http://localhost:11434"),
    model: str = "exaone3.5",
    output_dir: Path = Path("./output/images"),
) -> dict:
    # 1. Trigger
    trend_keywords_data = await stage_trigger(max_keywords)
    if not trend_keywords_data:
        return {"success": False}

    # 2. Collection (전체 키워드 수집)
    articles = await stage_collection(trend_keywords_data, max_articles)
    if not articles:
        return {"success": False}

    # 3. Deduplication & Indexing
    await stage_deduplication_and_indexing(articles)

    # 4. Analysis Loop (전체 키워드 반복 분석)
    print_stage(f"Stage 4: LLM ANALYSIS - 전체 분석 ({len(trend_keywords_data)}개 토픽)")

    analysis_results = []

    for idx, kw_data in enumerate(trend_keywords_data, 1):
        target_keyword = kw_data["keyword"]
        print(f"\n[Topic {idx}/{len(trend_keywords_data)}] Analyzing: '{target_keyword}'...")

        # [수정 2] DB 중복 여부 상관없이 '현재 수집된 기사' 사용 (리포트 생성 보장)
        target_articles = [art for art in articles if art.get("keyword") == target_keyword]

        # 기사 내 단순 중복 제거 (제목 기준)
        seen = set()
        unique_target_articles = []
        for art in target_articles:
            if art["title"] not in seen:
                seen.add(art["title"])
                unique_target_articles.append(art)

        if not unique_target_articles:
            print(f"   ⚠️ 수집된 기사가 없습니다: '{target_keyword}' (Skipping)")
            continue

        # RAG 검색
        context_docs = await stage_rag_search(target_keyword, unique_target_articles)

        # 데이터 병합 (과거 문맥 + 현재 기사)
        rag_augmented_articles = []
        for ctx in context_docs:
            rag_augmented_articles.append(
                {
                    "title": f"[과거 문맥] {ctx.get('date', '')[:10]}",
                    "summary": ctx["content"],
                    "source": "TrendOps Memory",
                    "published": ctx.get("date", ""),
                    "keyword": target_keyword,
                }
            )
        rag_augmented_articles.extend(unique_target_articles)

        # LLM 분석
        analysis_result, _ = await stage_llm_analysis(
            keyword=target_keyword,
            articles=rag_augmented_articles,
            ollama_url=ollama_url,
            model=model,
        )

        if analysis_result:
            analysis_results.append(analysis_result)
            # [추가됨] 리포트 저장
            if report_service:
                report_service.save_analysis(target_keyword, analysis_result)
                print("   💾 리포트 데이터 저장 완료")

        await asyncio.sleep(1)

    if console:
        console.print(Panel("🎉 전체 파이프라인 완료!", style="bold green"))

    return {"success": True, "analysis_count": len(analysis_results)}


if __name__ == "__main__":
    asyncio.run(run_real_pipeline())
