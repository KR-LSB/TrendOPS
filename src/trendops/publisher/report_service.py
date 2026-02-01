# src/trendops/publisher/report_service.py
"""
TrendOps Report Service
Week 5: 분석 결과 저장 및 일일 리포트 생성
"""
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

# 로거 설정
import logging
logger = logging.getLogger("report_service")

class ReportService:
    """리포트 관련 기능을 담당하는 서비스 클래스"""
    
    def __init__(self, base_dir: str = "data/reports"):
        self.base_dir = Path(base_dir)
        self.log_dir = self.base_dir / "logs"      # 원본 데이터 저장 (JSONL)
        self.output_dir = self.base_dir / "daily"  # 최종 리포트 저장 (Markdown)
        
        # 디렉토리 생성
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _get_log_file(self) -> Path:
        """오늘 날짜의 로그 파일 경로 반환"""
        today = datetime.now().strftime("%Y-%m-%d")
        return self.log_dir / f"{today}_analysis.jsonl"

    def save_analysis(self, keyword: str, analysis_data: dict) -> None:
        """
        분석 결과를 JSONL 파일에 추가 저장 (Append)
        """
        log_file = self._get_log_file()
        
        # 저장할 데이터 구조
        entry = {
            "timestamp": datetime.now().isoformat(),
            "keyword": keyword,
            "analysis": analysis_data
        }
        
        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            logger.info(f"Analysis saved for keyword: {keyword}")
        except Exception as e:
            logger.error(f"Failed to save analysis log: {e}")

    def generate_daily_report(self) -> str | None:
        """
        오늘 쌓인 로그를 읽어서 Markdown 리포트 생성
        """
        log_file = self._get_log_file()
        if not log_file.exists():
            logger.warning("No analysis logs found for today.")
            return None
            
        # 1. 로그 읽기
        entries = []
        try:
            with open(log_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        entries.append(json.loads(line))
        except Exception as e:
            logger.error(f"Failed to read logs: {e}")
            return None

        if not entries:
            return None

        # 2. 리포트 작성 (Markdown)
        today_str = datetime.now().strftime("%Y년 %m월 %d일")
        report = [f"# 📰 TrendOps 일일 트렌드 리포트 ({today_str})", ""]
        report.append(f"> **오늘 분석된 토픽:** {len(entries)}건")
        report.append(f"> **발행 시간:** {datetime.now().strftime('%H:%M:%S')}")
        report.append("---\n")

        for idx, entry in enumerate(entries, 1):
            data = entry["analysis"]
            keyword = entry["keyword"]
            sentiment = data.get("sentiment", {})
            
            # 이모지 감성
            pos = sentiment.get("positive", 0)
            mood = "😊 긍정적" if pos > 0.6 else ("😠 부정적" if sentiment.get("negative", 0) > 0.6 else "😐 중립적")

            report.append(f"## {idx}. {keyword} {mood}")
            report.append(f"**핵심 원인:** {data.get('main_cause', '-')}\n")
            
            report.append("### 📝 요약")
            report.append(f"{data.get('summary', '-')}\n")
            
            report.append("### 💡 주요 여론")
            for op in data.get("key_opinions", [])[:3]:
                report.append(f"- {op}")
            
            report.append(f"\n*(감성지수: 긍정 {int(pos*100)}% / 부정 {int(sentiment.get('negative', 0)*100)}%)*")
            report.append("\n---\n")

        # 3. 파일 저장
        report_content = "\n".join(report)
        output_file = self.output_dir / f"Daily_Report_{datetime.now().strftime('%Y-%m-%d')}.md"
        
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(report_content)
            logger.info(f"Daily report generated: {output_file}")
            return str(output_file)
        except Exception as e:
            logger.error(f"Failed to save report file: {e}")
            return None