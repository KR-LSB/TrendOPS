# src/trendops/analyst/__init__.py
"""
TrendOps Analyst Module

Blueprint Week 2: LLM 기반 뉴스 분석
- vLLM 서버 연동 (OpenAI API 호환)
- 중립적 분석가 페르소나
- JSON 구조화 출력

Week 4에서 Outlines 적용으로 JSON 100% 보장 예정
"""

from trendops.analyst.analyzer_llm import (
    AnalysisOutput,
    AnalysisResult,
    LLMAnalyzer,
    SentimentRatio,
    analyze_keyword,
)

__all__ = [
    "LLMAnalyzer",
    "AnalysisResult",
    "AnalysisOutput",
    "SentimentRatio",
    "analyze_keyword",
]
