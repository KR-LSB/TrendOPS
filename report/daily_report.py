import json
from pathlib import Path
from datetime import datetime, timedelta
from jinja2 import Environment, FileSystemLoader
import logging

logger = logging.getLogger("daily_report")

class DailyReportGenerator:
    def __init__(self, template_dir: str = "src/trendops/report/templates"):
        self.template_dir = Path(template_dir)
        # Jinja2 환경 설정
        self.env = Environment(loader=FileSystemLoader(self.template_dir))

    def generate(self, log_file: Path, output_file: Path) -> str | None:
        """로그 파일을 읽어 리포트 생성"""
        if not log_file.exists():
            logger.warning("No log file found.")
            return None

        # 1. 데이터 로딩 및 통계 계산
        data = self._process_logs(log_file)
        
        # 2. 템플릿 렌더링
        try:
            template = self.env.get_template("daily_report.md.j2")
            rendered_report = template.render(**data)
            
            # 3. 파일 저장
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(rendered_report)
            
            return str(output_file)
        except Exception as e:
            logger.error(f"Failed to render report: {e}")
            return None

    def _process_logs(self, log_file: Path) -> dict:
        """JSONL 로그를 파싱하여 통계 데이터 생성"""
        entries = []
        with open(log_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    entries.append(json.loads(line))

        # --- 통계 계산 로직 (Mockup 포함) ---
        
        # 1. Top 5 트렌드 (최신순 5개 추출)
        top_trends = []
        for entry in entries[-5:]: # 뒤에서 5개
            an = entry["analysis"]
            
            # 감성 포맷팅
            pos = an.get("sentiment", {}).get("positive", 0)
            neg = an.get("sentiment", {}).get("negative", 0)
            if pos > 0.5: sent = f"🟢 긍정 {int(pos*100)}%"
            elif neg > 0.5: sent = f"🔴 부정 {int(neg*100)}%"
            else: sent = f"⚪ 중립 {int((1-pos-neg)*100)}%"

            top_trends.append({
                "keyword": entry["keyword"],
                "sentiment": sent,
                "cause": an.get("main_cause", "-")[:40] + "..." # 길이 제한
            })
        top_trends.reverse() # 1위가 위로 오게

        # 2. 시스템 통계 (실제 데이터 + 일부 Mockup)
        # 실제로는 DB나 모니터링 툴에서 가져와야 하지만, 지금은 로그 개수로 추정합니다.
        total_collected = sum([an.get("source_count", 0) for an in [e["analysis"] for e in entries]])
        
        stats = {
            "total_collected": total_collected,
            "news_count": int(total_collected * 0.7),   # 예시 비율
            "youtube_count": int(total_collected * 0.3),# 예시 비율
            "dedup_ratio": 73.2, # (나중에 Dedup 모듈에서 받아와야 함)
            "run_count": len(entries),
            "success_rate": 100,
            "avg_duration": 38.2, # (Log에 duration 추가 시 계산 가능)
            "log_count": len(entries),
            "issues": [] # 이슈가 있으면 여기에 추가
        }

        # 3. 샘플 데이터
        sample = {"keyword": "-", "summary": "-"}
        if entries:
            last = entries[-1]
            sample = {
                "keyword": last["keyword"],
                "summary": last["analysis"].get("summary", "-")
            }

        # 날짜 포맷팅
        now = datetime.now()
        days = ["월", "화", "수", "목", "금", "토", "일"]
        
        return {
            "date": now.strftime("%Y-%m-%d"),
            "day_of_week": days[now.weekday()],
            "top_trends": top_trends,
            "stats": stats,
            "sample": sample,
            "next_run": (now + timedelta(days=1)).strftime("%Y-%m-%d 00:00")
        }