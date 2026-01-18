# src/trendops/trigger/trigger_google.py
import asyncio
import random
from datetime import datetime
from typing import Any

import aiohttp

from trendops.config.settings import get_settings
from trendops.queue.queue_models import JobStatus, TrendKeyword, TrendJob, TrendSource
from trendops.queue.queue_redis import RedisQueue
from trendops.utils.logger import get_logger

logger = get_logger(__name__)


class RetryConfig:
    """재시도 설정"""
    max_attempts: int = 3
    base_delay: float = 2.0
    max_delay: float = 60.0
    exponential_base: float = 2.0


class GoogleTrendTrigger:
    """
    Google Trends 실시간 트렌드 감지기
    
    Blueprint Week 1: 트렌드 감지 → Job Queue 추가
    pytrends 대신 직접 HTTP 요청으로 안정성 확보
    """
    
    # Google Trends Daily Trends API (비공식)
    TRENDS_API_URL = "https://trends.google.com/trends/api/dailytrends"
    
    BASE_TREND_SCORE = 7.0
    RANK_SCORE_BONUS = 2.0
    
    def __init__(
        self,
        redis_queue: RedisQueue | None = None,
        retry_config: RetryConfig | None = None,
    ):
        self._redis_queue = redis_queue
        self._retry_config = retry_config or RetryConfig()
        self._settings = get_settings()
    
    async def _fetch_trending_direct(self) -> list[str]:
        """
        Google Trends에서 직접 트렌드 키워드 조회
        pytrends 호환성 문제 우회
        """
        params = {
            "hl": "ko",
            "tz": "-540",
            "geo": "KR",
            "ns": "15",
        }
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "application/json",
        }
        
        timeout = aiohttp.ClientTimeout(total=30)
        
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(
                    self.TRENDS_API_URL,
                    params=params,
                    headers=headers,
                ) as response:
                    if response.status != 200:
                        logger.warning(
                            f"Google Trends API returned status {response.status}"
                        )
                        return []
                    
                    text = await response.text()
                    
                    # Google API는 ")]}'" 접두사를 붙임
                    if text.startswith(")]}'"):
                        text = text[5:]
                    
                    import json
                    data = json.loads(text)
                    
                    keywords: list[str] = []
                    
                    # dailytrends 응답 파싱
                    trending_days = data.get("default", {}).get("trendingSearchesDays", [])
                    
                    for day in trending_days:
                        searches = day.get("trendingSearches", [])
                        for search in searches:
                            title = search.get("title", {}).get("query", "")
                            if title:
                                keywords.append(title)
                    
                    return keywords
                    
        except Exception as e:
            logger.warning(f"Direct trends fetch failed: {e}")
            return []
    
    async def _fetch_with_pytrends(self) -> list[str]:
        """
        pytrends 라이브러리를 사용한 조회 (Fallback)
        """
        try:
            from pytrends.request import TrendReq
            
            loop = asyncio.get_event_loop()
            
            def fetch_sync() -> list[str]:
                pytrends = TrendReq(hl="ko", tz=540)
                df = pytrends.trending_searches(pn="south_korea")
                if df is None or df.empty:
                    return []
                return df[0].tolist()
            
            return await loop.run_in_executor(None, fetch_sync)
            
        except Exception as e:
            logger.warning(f"pytrends fetch failed: {e}")
            return []
    
    async def _fetch_trending_with_backoff(self) -> list[str]:
        """
        Exponential backoff를 적용한 트렌드 키워드 조회
        """
        for attempt in range(self._retry_config.max_attempts):
            # 1차: 직접 API 호출
            keywords = await self._fetch_trending_direct()
            
            if keywords:
                logger.info(
                    f"Trending keywords fetched via direct API",
                    extra={"count": len(keywords), "attempt": attempt + 1}
                )
                return keywords
            
            # 2차: pytrends fallback
            keywords = await self._fetch_with_pytrends()
            
            if keywords:
                logger.info(
                    f"Trending keywords fetched via pytrends",
                    extra={"count": len(keywords), "attempt": attempt + 1}
                )
                return keywords
            
            if attempt < self._retry_config.max_attempts - 1:
                delay = min(
                    self._retry_config.base_delay * (self._retry_config.exponential_base ** attempt),
                    self._retry_config.max_delay
                )
                delay = delay * (0.5 + random.random())
                
                logger.warning(
                    f"Trends fetch failed, retrying in {delay:.1f}s",
                    extra={"attempt": attempt + 1}
                )
                await asyncio.sleep(delay)
        
        logger.error("All trend fetch methods failed")
        return await self._get_fallback_keywords()
    
    async def _get_fallback_keywords(self) -> list[str]:
        """
        Fallback 키워드 목록 반환
        실제 운영에서는 Redis 캐시에서 최근 트렌드를 가져옴
        """
        logger.warning("Using fallback keywords")
        return [
            "AI 반도체",
            "트럼프",
            "삼성전자",
            "비트코인",
            "날씨",
        ]
    
    def _calculate_trend_score(self, rank: int, total: int) -> float:
        """순위 기반 트렌드 점수 계산"""
        if total == 0:
            return self.BASE_TREND_SCORE
        
        rank_ratio = 1 - (rank / total)
        bonus = self.RANK_SCORE_BONUS * rank_ratio + 0.1
        score = min(self.BASE_TREND_SCORE + bonus, 10.0)
        return round(score, 2)
    
    async def fetch_trends(self) -> list[TrendKeyword]:
        """트렌드 키워드 조회 및 TrendKeyword 변환"""
        keywords = await self._fetch_trending_with_backoff()
        
        if not keywords:
            logger.warning("No trending keywords found")
            return []
        
        trend_keywords: list[TrendKeyword] = []
        total = len(keywords)
        discovered_at = datetime.now()
        
        for rank, keyword in enumerate(keywords):
            trend_score = self._calculate_trend_score(rank, total)
            
            trend_keyword = TrendKeyword(
                keyword=keyword,
                source=TrendSource.GOOGLE,
                trend_score=trend_score,
                discovered_at=discovered_at,
            )
            trend_keywords.append(trend_keyword)
        
        logger.info(f"Converted {len(trend_keywords)} trend keywords")
        return trend_keywords
    
    async def push_to_queue(
        self,
        trend_keywords: list[TrendKeyword],
        min_score: float = 7.0,
    ) -> list[str]:
        """트렌드 키워드를 Job Queue에 추가"""
        if self._redis_queue is None:
            raise RuntimeError("Redis queue not initialized")
        
        job_ids: list[str] = []
        skipped_count = 0
        
        for trend_keyword in trend_keywords:
            if trend_keyword.trend_score < min_score:
                skipped_count += 1
                continue
            
            job = TrendJob(
                keyword_info=trend_keyword,
                status=JobStatus.PENDING,
            )
            
            job_id = await self._redis_queue.push_job(job)
            job_ids.append(job_id)
        
        logger.info(
            f"Pushed {len(job_ids)} jobs to queue (skipped {skipped_count})",
            extra={"pushed": len(job_ids), "skipped": skipped_count}
        )
        return job_ids
    
    async def run(
        self,
        min_score: float = 7.0,
        max_keywords: int | None = None,
    ) -> dict[str, Any]:
        """트렌드 감지 및 Job Queue 추가 실행"""
        start_time = datetime.now()
        
        logger.info(
            f"Google Trend Trigger started",
            extra={"min_score": min_score, "max_keywords": max_keywords}
        )
        
        trend_keywords = await self.fetch_trends()
        
        if not trend_keywords:
            return {
                "status": "no_trends",
                "fetched": 0,
                "pushed": 0,
                "duration_seconds": (datetime.now() - start_time).total_seconds(),
            }
        
        if max_keywords is not None:
            trend_keywords = trend_keywords[:max_keywords]
        
        job_ids = await self.push_to_queue(trend_keywords, min_score=min_score)
        
        duration = (datetime.now() - start_time).total_seconds()
        
        result = {
            "status": "success",
            "fetched": len(trend_keywords),
            "pushed": len(job_ids),
            "keywords": [tk.keyword for tk in trend_keywords[:5]],
            "duration_seconds": duration,
        }
        
        logger.info(f"Google Trend Trigger completed", extra=result)
        
        return result


async def run_trigger(
    min_score: float = 7.0,
    max_keywords: int | None = 10,
) -> dict[str, Any]:
    """독립 실행용 트리거 함수"""
    redis_queue = RedisQueue()
    
    try:
        await redis_queue.connect()
        
        trigger = GoogleTrendTrigger(redis_queue=redis_queue)
        result = await trigger.run(
            min_score=min_score,
            max_keywords=max_keywords,
        )
        
        return result
        
    finally:
        await redis_queue.disconnect()


if __name__ == "__main__":
    async def main():
        result = await run_trigger(min_score=7.0, max_keywords=10)
        print(f"\nGoogle Trend Trigger Result:")
        print(f"  Status: {result['status']}")
        print(f"  Fetched: {result['fetched']} keywords")
        print(f"  Pushed: {result['pushed']} jobs")
        if result.get('keywords'):
            print(f"  Top Keywords: {result['keywords']}")
    
    asyncio.run(main())