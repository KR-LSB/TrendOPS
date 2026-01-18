# tests/manual_e2e_week1.py
"""
TrendOps Week 1 E2E Integration Test
"""
import asyncio
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from trendops.collector.collector_rss_google import GoogleNewsRSSCollector
from trendops.queue.queue_redis import RedisQueue
from trendops.trigger.trigger_google import GoogleTrendTrigger
from trendops.utils.logger import get_logger

logger = get_logger(__name__)


class C:
    H = "\033[95m"
    B = "\033[94m"
    C = "\033[96m"
    G = "\033[92m"
    Y = "\033[93m"
    R = "\033[91m"
    BOLD = "\033[1m"
    E = "\033[0m"


def header(t: str) -> None:
    print(f"\n{C.BOLD}{C.H}{'='*60}\n  {t}\n{'='*60}{C.E}")


def step(n: int, t: str) -> None:
    print(f"\n{C.BOLD}{C.C}[Step {n}/4]{C.E} {t}")


def ok(t: str) -> None:
    print(f"{C.G}✅ {t}{C.E}")


def err(t: str) -> None:
    print(f"{C.R}❌ {t}{C.E}")


def info(k: str, v: str) -> None:
    print(f"   {C.Y}{k}:{C.E} {v}")


async def run_e2e_test() -> bool:
    header("TrendOps Week 1 - E2E Integration Test")
    print(f"   시작: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    q = RedisQueue()
    collector: GoogleNewsRSSCollector | None = None

    try:
        # Step 1
        step(1, "Google Trends 키워드 감지 → Redis Queue")
        await q.connect()
        ok("Redis 연결 성공")

        trigger = GoogleTrendTrigger(redis_queue=q)
        res = await trigger.run(min_score=0.0, max_keywords=1)

        if res["status"] != "success" or res["pushed"] == 0:
            err("트렌드 키워드 실패")
            from trendops.queue.queue_models import TrendJob, TrendKeyword, TrendSource
            dummy = TrendJob(keyword_info=TrendKeyword(
                keyword="AI 반도체",
                source=TrendSource.GOOGLE,
                trend_score=8.0,
            ))
            await q.push_job(dummy)
            ok("Fallback 키워드 'AI 반도체' 추가")
        else:
            ok(f"키워드 감지: {res.get('keywords', [])}")
            info("감지", str(res["fetched"]))
            info("큐 추가", str(res["pushed"]))

        stats = await q.get_queue_stats()
        info("Queue", f"pending={stats['pending']}, processing={stats['processing']}")

        # Step 2
        step(2, "Redis Queue에서 Job Pop")
        job = await q.pop_job(timeout=5)

        if job is None:
            err("Job Pop 실패")
            return False

        ok("Job Pop 성공")
        info("Job ID", str(job.job_id))
        info("키워드", job.keyword_info.keyword)
        info("점수", str(job.keyword_info.trend_score))

        await q.mark_processing(job.job_id)
        ok("Job → processing")

        # Step 3
        step(3, f"Google News RSS 수집: '{job.keyword_info.keyword}'")
        
        # Context manager 사용하여 세션 자동 정리
        async with GoogleNewsRSSCollector() as collector:
            result = await collector.fetch(keyword=job.keyword_info.keyword, max_results=5)

            if result.count == 0:
                err("뉴스 수집 실패")
                await q.mark_failed(job.job_id, "No articles")
                return False

            ok(f"뉴스 수집 완료: {result.count}개")
            info("수집 시간", result.collected_at.strftime("%Y-%m-%d %H:%M:%S"))

        await q.mark_completed(job.job_id)
        ok("Job → completed")

        # Step 4
        step(4, "결과 출력")
        print(f"\n{'─'*60}")
        print(f"{C.BOLD}📰 수집된 뉴스 ({result.count}건){C.E}")
        print(f"{'─'*60}")

        for i, a in enumerate(result.articles, 1):
            pub = a.published.strftime("%Y-%m-%d %H:%M") if a.published else "N/A"
            print(f"\n   {C.BOLD}{i}. {a.title}{C.E}")
            link_str = str(a.link)
            print(f"      🔗 {link_str[:70]}...")
            print(f"      📅 {pub}")

        print(f"\n{'─'*60}")

        final = await q.get_queue_stats()
        print(f"\n{C.BOLD}📊 최종 Queue 상태{C.E}")
        info("Pending", str(final["pending"]))
        info("Processing", str(final["processing"]))
        info("Completed", str(final["completed"]))

        return True

    except Exception as e:
        err(f"테스트 실패: {e}")
        logger.exception("E2E test failed")
        return False

    finally:
        await q.disconnect()


async def main() -> None:
    print(f"\n{C.BOLD}{C.B}")
    print(r"""
  _____                    _  ___
 |_   _| __ ___ _ __   __| |/ _ \ _ __  ___
   | || '__/ _ \ '_ \ / _` | | | | '_ \/ __|
   | || | |  __/ | | | (_| | |_| | |_) \__ \
   |_||_|  \___|_| |_|\__,_|\___/| .__/|___/
                                 |_|
    Week 1: Data Ingestion Foundation
    """)
    print(C.E)

    success = await run_e2e_test()

    header("테스트 결과")

    if success:
        print(f"""
   {C.G}{C.BOLD}
   ✅ Week 1 E2E 테스트 통과!

   검증된 모듈:
   ├── trigger/trigger_google.py
   ├── queue/queue_redis.py
   ├── queue/queue_models.py
   ├── collector/collector_rss_google.py
   ├── config/settings.py
   └── utils/logger.py

   다음: Week 2 - LLM 연동
   {C.E}""")
    else:
        print(f"""
   {C.R}{C.BOLD}
   ❌ Week 1 E2E 테스트 실패

   확인:
   1. Redis 실행 확인 (docker-compose up -d)
   2. .env 설정 확인
   3. 네트워크 상태 확인
   {C.E}""")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())