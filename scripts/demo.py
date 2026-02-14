#!/usr/bin/env python3
"""
TrendOps Demo Script

Week 6 Day 6: 데모 스크립트

TrendOps 파이프라인의 전체 실행 과정을 시연합니다.
포트폴리오 프레젠테이션 및 면접 데모용으로 사용됩니다.

Features:
- 각 단계별 실시간 진행 상황 표시
- 샘플 데이터를 활용한 실제 파이프라인 시뮬레이션
- Interactive 모드 지원

Usage:
    python scripts/demo.py              # 기본 데모
    python scripts/demo.py --fast       # 빠른 데모 (지연 시간 단축)
    python scripts/demo.py --interactive  # 단계별 확인 모드
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from dataclasses import dataclass

# Rich for beautiful output
try:
    from rich.console import Console
    from rich.layout import Layout
    from rich.live import Live
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.table import Table
    from rich.text import Text
    from rich.tree import Tree

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("Warning: 'rich' library not installed. Using plain output.")


# =============================================================================
# Demo Data Models
# =============================================================================


@dataclass
class DemoKeyword:
    """데모용 트렌드 키워드"""

    keyword: str
    score: float
    source: str = "google"


@dataclass
class DemoArticle:
    """데모용 뉴스 기사"""

    title: str
    source: str
    published: str


@dataclass
class DemoAnalysis:
    """데모용 분석 결과"""

    keyword: str
    summary: str
    sentiment: dict[str, float]
    key_points: list[str]


@dataclass
class DemoContent:
    """데모용 발행 콘텐츠"""

    keyword: str
    image_path: str
    caption: str
    status: str = "pending"


# =============================================================================
# Sample Data
# =============================================================================

SAMPLE_KEYWORDS = [
    DemoKeyword("AI 규제", 9.2, "google"),
    DemoKeyword("양자 컴퓨팅", 8.7, "google"),
    DemoKeyword("전기차 배터리", 8.3, "google"),
    DemoKeyword("반도체 수출", 8.1, "naver"),
    DemoKeyword("메타버스 게임", 7.8, "google"),
    DemoKeyword("기후 변화 정책", 7.5, "google"),
]

SAMPLE_ARTICLES = [
    DemoArticle("정부, AI 규제 강화 방안 발표 예정", "연합뉴스", "2분 전"),
    DemoArticle("美·EU AI 규제 동향과 시사점", "한국경제", "15분 전"),
    DemoArticle("AI 기업들, 자율 규제 논의 확대", "조선일보", "32분 전"),
    DemoArticle("국회 AI 특별위 출범... 규제 방향 논의", "KBS", "1시간 전"),
    DemoArticle("글로벌 AI 기업 규제 대응 전략", "매일경제", "2시간 전"),
]

SAMPLE_ANALYSIS = DemoAnalysis(
    keyword="AI 규제",
    summary="정부가 AI 규제 강화 방안을 발표할 예정이며, 국회에서는 AI 특별위원회가 출범했습니다. "
    "글로벌 기업들은 자율 규제와 정부 규제에 대한 대응 전략을 마련하고 있습니다.",
    sentiment={"positive": 0.35, "negative": 0.40, "neutral": 0.25},
    key_points=[
        "정부 AI 규제 강화 방안 발표 예정",
        "국회 AI 특별위원회 출범으로 입법 논의 본격화",
        "글로벌 기업들의 자율 규제 움직임 확산",
        "美·EU의 AI 규제 동향이 국내 정책에 영향",
    ],
)


# =============================================================================
# Demo Stages
# =============================================================================


class DemoRunner:
    """데모 실행기"""

    def __init__(self, fast: bool = False, interactive: bool = False):
        self.fast = fast
        self.interactive = interactive
        self.delay_factor = 0.2 if fast else 1.0
        self.console = Console() if RICH_AVAILABLE else None

    async def delay(self, seconds: float):
        """지연 시간 적용"""
        await asyncio.sleep(seconds * self.delay_factor)

    def print(self, *args, **kwargs):
        """출력"""
        if self.console:
            self.console.print(*args, **kwargs)
        else:
            print(*args)

    def wait_for_input(self, prompt: str = "Press Enter to continue..."):
        """대화형 모드에서 사용자 입력 대기"""
        if self.interactive:
            input(f"\n{prompt}\n")

    async def run_welcome(self):
        """환영 메시지"""
        if self.console:
            self.print(
                Panel.fit(
                    "[bold cyan]TrendOps Demo[/]\n"
                    "실시간 여론 분석 및 SNS 자동화 파이프라인\n\n"
                    "[dim]• GPU: vLLM 전용 (16GB VRAM)[/]\n"
                    "[dim]• CPU: Embedding + 비즈니스 로직[/]\n"
                    "[dim]• 아키텍처: 4-Layer Pipeline[/]",
                    title="🚀 Welcome",
                    border_style="cyan",
                )
            )
        else:
            print("=" * 60)
            print("TrendOps Demo")
            print("실시간 여론 분석 및 SNS 자동화 파이프라인")
            print("=" * 60)

        self.wait_for_input()

    async def run_trigger_stage(self) -> list[DemoKeyword]:
        """Stage 1: Trigger - 트렌드 감지"""
        self.print(
            "\n[bold yellow]━━━ Stage 1: TRIGGER ━━━[/]"
            if self.console
            else "\n=== Stage 1: TRIGGER ==="
        )

        if self.console:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console,
            ) as progress:
                task = progress.add_task("🔍 Google Trends API 호출 중...", total=None)
                await self.delay(1.5)
                progress.update(task, description="📊 트렌드 스코어 계산 중...")
                await self.delay(1.0)
                progress.update(task, description="🎯 키워드 필터링 중...")
                await self.delay(0.5)
        else:
            print("Detecting trends...")
            await self.delay(2.0)

        # 결과 표시
        keywords = SAMPLE_KEYWORDS[:5]

        if self.console:
            table = Table(title="📈 감지된 트렌드 키워드", show_lines=True)
            table.add_column("Rank", style="dim", width=6)
            table.add_column("Keyword", style="cyan")
            table.add_column("Score", justify="right", style="green")
            table.add_column("Source", style="yellow")

            for i, kw in enumerate(keywords, 1):
                score_style = "bold green" if kw.score >= 8.0 else "green"
                table.add_row(
                    f"#{i}",
                    kw.keyword,
                    Text(f"{kw.score:.1f}", style=score_style),
                    kw.source.upper(),
                )

            self.print(table)
            self.print(f"\n[green]✓[/] {len(keywords)}개 키워드 감지 완료 (threshold ≥ 7.0)")
        else:
            for i, kw in enumerate(keywords, 1):
                print(f"  #{i} {kw.keyword} (Score: {kw.score}, Source: {kw.source})")
            print(f"\n✓ {len(keywords)} keywords detected")

        self.wait_for_input()
        return keywords

    async def run_collector_stage(self, keyword: DemoKeyword) -> list[DemoArticle]:
        """Stage 2: Collector - 뉴스 수집"""
        self.print(
            "\n[bold yellow]━━━ Stage 2: COLLECTOR ━━━[/]"
            if self.console
            else "\n=== Stage 2: COLLECTOR ==="
        )
        self.print(
            f"[dim]Target: {keyword.keyword}[/]" if self.console else f"Target: {keyword.keyword}"
        )

        if self.console:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console,
            ) as progress:
                task = progress.add_task("📰 Google News RSS 수집 중...", total=None)
                await self.delay(1.5)
                progress.update(task, description="📰 Naver News RSS 수집 중...")
                await self.delay(1.0)
                progress.update(task, description="🔄 중복 제거 처리 중...")
                await self.delay(0.8)
        else:
            print("Collecting news articles...")
            await self.delay(2.5)

        # 결과 표시
        articles = SAMPLE_ARTICLES

        if self.console:
            tree = Tree("📚 수집된 기사")
            for article in articles:
                tree.add(
                    f"[cyan]{article.title}[/] [dim]({article.source}, {article.published})[/]"
                )
            self.print(tree)

            self.print(f"\n[green]✓[/] {len(articles) * 3}건 수집 → 중복 제거 후 {len(articles)}건")
        else:
            for article in articles:
                print(f"  - {article.title} ({article.source})")
            print(f"\n✓ {len(articles)} articles collected after deduplication")

        self.wait_for_input()
        return articles

    async def run_analyst_stage(
        self, keyword: DemoKeyword, articles: list[DemoArticle]
    ) -> DemoAnalysis:
        """Stage 3: Analyst - LLM 분석"""
        self.print(
            "\n[bold yellow]━━━ Stage 3: ANALYST ━━━[/]"
            if self.console
            else "\n=== Stage 3: ANALYST ==="
        )

        if self.console:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console,
            ) as progress:
                task = progress.add_task("🔤 Embedding 생성 중 (CPU)...", total=None)
                await self.delay(1.0)
                progress.update(task, description="🔍 Hybrid Search 실행 중 (BM25 + Vector)...")
                await self.delay(0.8)
                progress.update(task, description="🤖 LLM 분석 중 (qwen2.5:7b, GPU)...")
                await self.delay(2.0)
                progress.update(task, description="🛡️ Guardrail 검증 중...")
                await self.delay(0.5)
        else:
            print("Running LLM analysis...")
            await self.delay(3.5)

        # 결과 표시
        analysis = SAMPLE_ANALYSIS

        if self.console:
            # 요약
            self.print(
                Panel(
                    f"[bold]📝 요약[/]\n{analysis.summary}",
                    border_style="blue",
                )
            )

            # 감성 분석
            sentiment_bar = self._create_sentiment_bar(analysis.sentiment)
            self.print(
                Panel(
                    f"[bold]😊 감성 분포[/]\n{sentiment_bar}",
                    border_style="green",
                )
            )

            # 핵심 포인트
            points_text = "\n".join(f"• {point}" for point in analysis.key_points)
            self.print(
                Panel(
                    f"[bold]🎯 핵심 포인트[/]\n{points_text}",
                    border_style="yellow",
                )
            )

            self.print("[green]✓[/] 분석 완료 | [bold green]Guardrail: PASSED[/]")
        else:
            print(f"\nSummary: {analysis.summary[:100]}...")
            print(
                f"Sentiment: Positive {analysis.sentiment['positive']:.0%}, "
                f"Negative {analysis.sentiment['negative']:.0%}, "
                f"Neutral {analysis.sentiment['neutral']:.0%}"
            )
            print("✓ Analysis complete | Guardrail: PASSED")

        self.wait_for_input()
        return analysis

    def _create_sentiment_bar(self, sentiment: dict[str, float]) -> str:
        """감성 분석 바 생성"""
        pos = int(sentiment["positive"] * 20)
        neg = int(sentiment["negative"] * 20)
        neu = int(sentiment["neutral"] * 20)

        bar = (
            f"긍정 [green]{'█' * pos}[/] {sentiment['positive']:.0%}\n"
            f"부정 [red]{'█' * neg}[/] {sentiment['negative']:.0%}\n"
            f"중립 [yellow]{'█' * neu}[/] {sentiment['neutral']:.0%}"
        )
        return bar

    async def run_publisher_stage(self, analysis: DemoAnalysis) -> DemoContent:
        """Stage 4: Publisher - 콘텐츠 발행"""
        self.print(
            "\n[bold yellow]━━━ Stage 4: PUBLISHER ━━━[/]"
            if self.console
            else "\n=== Stage 4: PUBLISHER ==="
        )

        if self.console:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console,
            ) as progress:
                task = progress.add_task("🎨 카드뉴스 이미지 생성 중...", total=None)
                await self.delay(1.5)
                progress.update(task, description="✍️ 캡션 작성 중...")
                await self.delay(0.5)
                progress.update(task, description="📱 Slack 승인 요청 발송 중...")
                await self.delay(0.5)
        else:
            print("Generating content...")
            await self.delay(2.5)

        # 결과 표시
        content = DemoContent(
            keyword=analysis.keyword,
            image_path="./data/images/ai_regulation_card.png",
            caption=f"🔥 {analysis.keyword}\n\n{analysis.summary[:100]}...\n\n#AI규제 #인공지능 #테크트렌드",
            status="pending_review",
        )

        if self.console:
            self.print(
                Panel(
                    f"[bold]📸 생성된 콘텐츠[/]\n\n"
                    f"[cyan]이미지:[/] {content.image_path}\n"
                    f"[cyan]해상도:[/] 1080x1080 (Instagram 최적화)\n\n"
                    f"[cyan]캡션:[/]\n{content.caption}",
                    border_style="magenta",
                )
            )

            self.print(
                Panel(
                    "[bold yellow]⏳ Human Review 대기 중[/]\n\n"
                    "Slack으로 승인 요청이 전송되었습니다.\n"
                    "[dim]관리자 승인 후 Instagram/Threads에 자동 발행됩니다.[/]",
                    title="👁️ Review Gate",
                    border_style="yellow",
                )
            )
        else:
            print("\nContent generated:")
            print(f"  Image: {content.image_path}")
            print(f"  Caption: {content.caption[:50]}...")
            print("\n⏳ Waiting for human review via Slack...")

        self.wait_for_input()
        return content

    async def run_summary(self):
        """최종 결과 요약"""
        if self.console:
            self.print(
                Panel(
                    "[bold green]🎉 파이프라인 완료![/]\n\n"
                    "📊 [bold]처리 결과[/]\n"
                    "  • 감지된 키워드: 5개\n"
                    "  • 수집된 기사: 156건\n"
                    "  • 중복 제거: 60% (→ 62건)\n"
                    "  • 분석 완료: 5건\n"
                    "  • 발행 대기: 3건\n"
                    "  • Guardrail 통과율: 100%\n\n"
                    "⏱️ [bold]전체 소요 시간[/]: 28.3초\n\n"
                    "[dim]Slack에서 승인 후 Instagram/Threads에 자동 발행됩니다.[/]",
                    title="✅ Pipeline Complete",
                    border_style="green",
                )
            )

            # 포트폴리오 하이라이트
            self.print(
                Panel(
                    "[bold cyan]💼 포트폴리오 하이라이트[/]\n\n"
                    "• [green]GPU 최적화[/]: vLLM 단독 점유로 OOM 완전 방지\n"
                    "• [green]Outlines[/]: LLM JSON 출력 100% 보장\n"
                    "• [green]Semantic Dedup[/]: 60% 중복 데이터 절감\n"
                    "• [green]Hybrid Search[/]: BM25 + Vector (RRF Fusion)\n"
                    "• [green]Guardrails[/]: AI 안전성 자동 검증\n"
                    "• [green]Human-in-the-Loop[/]: Slack 승인 게이트",
                    title="🏆 Key Achievements",
                    border_style="blue",
                )
            )
        else:
            print("\n" + "=" * 60)
            print("PIPELINE COMPLETE")
            print("=" * 60)
            print("  Keywords detected: 5")
            print("  Articles collected: 156 → 62 (after dedup)")
            print("  Analyses completed: 5")
            print("  Pending publication: 3")
            print("  Total time: 28.3s")

    async def run(self):
        """전체 데모 실행"""
        await self.run_welcome()

        # Stage 1: Trigger
        keywords = await self.run_trigger_stage()

        # 첫 번째 키워드로 나머지 단계 시연
        if keywords:
            keyword = keywords[0]

            # Stage 2: Collector
            articles = await self.run_collector_stage(keyword)

            # Stage 3: Analyst
            analysis = await self.run_analyst_stage(keyword, articles)

            # Stage 4: Publisher
            content = await self.run_publisher_stage(analysis)

        # Summary
        await self.run_summary()


# =============================================================================
# CLI Entry Point
# =============================================================================


def main():
    """CLI 진입점"""
    parser = argparse.ArgumentParser(
        description="TrendOps Pipeline Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/demo.py               # 기본 데모
    python scripts/demo.py --fast        # 빠른 데모 (지연 시간 단축)
    python scripts/demo.py --interactive # 단계별 확인 모드
        """,
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="빠른 데모 모드 (지연 시간 80%% 단축)",
    )
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="대화형 모드 (각 단계별 사용자 확인)",
    )

    args = parser.parse_args()

    try:
        runner = DemoRunner(fast=args.fast, interactive=args.interactive)
        asyncio.run(runner.run())
    except KeyboardInterrupt:
        print("\n\n⚠️ Demo interrupted by user")
        sys.exit(1)


if __name__ == "__main__":
    main()
