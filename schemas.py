# src/trendops/schemas.py
"""
TrendOps 공통 데이터 스키마
"""
from enum import Enum
from dataclasses import dataclass, field
from typing import Any

class GuardrailIssueType(str, Enum):
    """검출된 이슈 유형"""
    PROFANITY = "profanity"           # 욕설
    HATE_SPEECH = "hate_speech"       # 혐오 발언
    SENSATIONALISM = "sensationalism" # 선정성
    POLITICAL_BIAS = "political_bias" # 정치적 편향
    PERSONAL_INFO = "personal_info"   # 개인정보
    UNVERIFIED_CLAIM = "unverified_claim" # 미검증 주장
    UNKNOWN = "unknown"

class GuardrailAction(str, Enum):
    """검사 결과 액션"""
    PASS = "pass"       # 통과
    REVISE = "revise"   # 수정 후 통과
    REVIEW = "review"   # 사람 검토 필요
    REJECT = "reject"   # 거부

@dataclass
class GuardrailIssue:
    """발견된 문제점"""
    issue_type: GuardrailIssueType
    severity: str  # low, medium, high, critical
    description: str
    location: str | None = None
    suggestion: str | None = None

@dataclass
class GuardrailResult:
    """최종 검사 결과"""
    content_id: str
    action: GuardrailAction
    is_safe: bool
    confidence: float
    issues: list[GuardrailIssue] = field(default_factory=list)
    original_content: str = ""
    revised_content: str | None = None
    review_reason: str | None = None
    
    @property
    def issue_summary(self) -> str:
        if not self.issues:
            return "No issues found"
        return ", ".join(f"{i.issue_type.value}({i.severity})" for i in self.issues)

@dataclass
class GuardrailCheckRequest:
    content: str
    keyword: str | None = None