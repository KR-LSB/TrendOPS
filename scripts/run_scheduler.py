# scripts/run_scheduler.py
import asyncio
import sys
import logging
from datetime import datetime
from pathlib import Path

# APScheduler 임포트
try:
    from apscheduler.schedulers.asyncio import AsyncIOScheduler
    from apscheduler.triggers.interval import IntervalTrigger
except ImportError:
    print("❌ APScheduler가 설치되지 않았습니다. 설치해주세요: pip install apscheduler")
    sys.exit(1)

# 프로젝트 루트 및 스크립트 경로 설정
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.append(str(current_dir))
sys.path.append(str(project_root))

# [수정 완료] 파이프라인 함수 임포트
from real_e2e_pipeline import run_real_pipeline

# [수정 완료] setup_logger -> get_logger
from trendops.utils.logger import get_logger

# [수정 완료] 로거 초기화
logger = get_logger("scheduler")

async def job_function():
    """주기적으로 실행될 작업"""
    logger.info("⏰ Scheduled Job Started: TrendOps E2E Pipeline")
    try:
        # 파이프라인 실행
        result = await run_real_pipeline(
            max_keywords=10,
            max_articles=15,
            model="exaone3.5"
        )
        
        status = "SUCCESS" if result.get("success") else "FAILED"
        # total_time_seconds 키가 없으면 0으로 처리
        duration = result.get("total_time_seconds", 0)
        logger.info(f"✅ Job Finished: {status}")
        
    except Exception as e:
        logger.error(f"❌ Job Execution Failed: {e}")

async def main():
    # 스케줄러 설정
    scheduler = AsyncIOScheduler(timezone="Asia/Seoul")
    
    # 실행 주기 설정 (예: 30분)
    INTERVAL_MINUTES = 30
    
    # 작업 등록
    scheduler.add_job(
        job_function,
        trigger=IntervalTrigger(minutes=INTERVAL_MINUTES),
        id="trendops_pipeline",
        name="TrendOps Pipeline",
        replace_existing=True,
        # 앱 시작 시 5초 후 첫 실행
        next_run_time=datetime.now().replace(microsecond=0) 
    )
    
    # 시작
    print(f"\n🚀 TrendOps Automation Started")
    print(f"   - Interval: Every {INTERVAL_MINUTES} minutes")
    print(f"   - Target: scripts/real_e2e_pipeline.py")
    print("   - Press Ctrl+C to stop.\n")
    
    scheduler.start()
    
    # 무한 대기
    try:
        while True:
            await asyncio.sleep(1)
    except (KeyboardInterrupt, SystemExit):
        print("\n🛑 Scheduler stopped.")
        scheduler.shutdown()

if __name__ == "__main__":
    asyncio.run(main())