# src/trendops/analyst/guardrail.py
"""
Week 4 Day 3: 콘텐츠 안전성 검증 시스템 (Guardrail)

Blueprint Week 4: Self-Correction Loop의 핵심 컴포넌트

구조:
1. RuleBasedChecker - 정규식/키워드 기반 빠른 검사 (1차 필터)
2. LLMBasedChecker - LLM 기반 심층 검사 (2차 검증)
3. ContentGuardrail - 통합 Guardrail 클래스
4. AutoReviser - 자동 수정 시도 (선택적)

검사 기준 (Blueprint Section 1.2.3):
- 정치적 편향: 특정 정당/인물 일방적 비난/옹호
- 욕설/비속어: 부적절한 표현
- 허위 정보: 검증되지 않은 사실 단정
- 개인정보: 특정 개인 식별 가능 정보
- 선정적 표현: 자극적/과장된 표현
- 혐오 발언: 특정 집단 비하

사용법:
    guardrail = ContentGuardrail()
    result = await guardrail.check(content, keyword="트럼프 관세")

    if result.action == GuardrailAction.PASS:
        # 안전한 콘텐츠
        pass
    elif result.action == GuardrailAction.REVISE:
        # 수정된 콘텐츠 사용
        revised = result.revised_content
    else:
        # 거부됨
        logger.warning(f"Content rejected: {result.issue_summary}")
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

# Day 2에서 생성한 스키마 import
try:
    from trendops.schemas import (
        GuardrailAction,
        GuardrailCheckRequest,
        GuardrailIssue,
        GuardrailIssueType,
        GuardrailResult,
    )
except ImportError:
    # 단독 실행 시 fallback (테스트용)
    from schemas import (
        GuardrailAction,
        GuardrailCheckRequest,
        GuardrailIssue,
        GuardrailIssueType,
        GuardrailResult,
    )


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class GuardrailConfig:
    """Guardrail 설정"""

    # 검사 모드
    enable_rule_based: bool = True
    enable_llm_based: bool = True
    strict_mode: bool = False  # True면 더 엄격한 검사

    # LLM 설정
    llm_model: str = "exaone3.5"
    llm_base_url: str = "http://localhost:11434"
    llm_temperature: float = 0.1  # 일관된 판정을 위해 낮은 temperature

    # 임계값
    confidence_threshold: float = 0.7  # 이 이상이면 확신
    auto_revise_threshold: float = 0.5  # 이 이상이면 자동 수정 시도

    # 재시도
    max_retries: int = 2
    retry_delay: float = 1.0


# =============================================================================
# Rule-Based Patterns (1차 필터)
# =============================================================================


@dataclass
class PatternRule:
    """패턴 매칭 규칙"""

    pattern: re.Pattern
    issue_type: GuardrailIssueType
    severity: str  # low, medium, high, critical
    description: str
    suggestion: str | None = None


class RulePatterns:
    """검사용 패턴 정의"""

    # 욕설/비속어 패턴 (한국어)
    PROFANITY_PATTERNS = [
        r"시[0-9]*발",
        r"씨[0-9]*발",
        r"ㅅㅂ",
        r"ㅆㅂ",
        r"병[0-9]*신",
        r"ㅂㅅ",
        r"지[0-9]*랄",
        r"ㅈㄹ",
        r"개[새세]끼",
        r"ㄱㅅㄲ",
        r"미친[놈년]",
        r"또라이",
        r"찐따",
        r"멍청이",
        r"바보같은",
    ]

    # 혐오 표현 패턴
    HATE_SPEECH_PATTERNS = [
        r"틀딱",
        r"한남",
        r"한녀",
        r"페미나치",
        r"일베충",
        r"좌좀",
        r"우좀",
        r"홍어",
        r"쪽바리",
        r"짱깨",
        r"깜둥이",
    ]

    # 선정적/자극적 표현 패턴
    SENSATIONALISM_PATTERNS = [
        r"충격[\s!]*[적의]?",
        r"경악",
        r"발칵\s*뒤집",
        r"완전\s*폭망",
        r"대박\s*사건",
        r"역대급",
        r"초강력",
        r"초특급",
        r"[!]{2,}",  # 느낌표 과다
        r"[?]{2,}",  # 물음표 과다
    ]

    # 정치적 편향 키워드 (주의 필요)
    POLITICAL_BIAS_KEYWORDS = [
        # 일방적 평가 표현
        r"무능한\s*(정부|대통령|정권)",
        r"최악의\s*(정부|대통령|정권)",
        r"최고의\s*(정부|대통령|정권)",
        r"훌륭한\s*(정부|대통령|정권)",
        # 극단적 표현
        r"빨갱이",
        r"수꼴",
        r"친일파",
        r"매국노",
        r"종북",
    ]

    # 개인정보 패턴
    PERSONAL_INFO_PATTERNS = [
        r"\d{3}[-.\s]?\d{4}[-.\s]?\d{4}",  # 전화번호
        r"\d{6}[-]?\d{7}",  # 주민등록번호
        r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",  # 이메일
        r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}",  # IP 주소
    ]

    # 검증되지 않은 주장 패턴
    UNVERIFIED_CLAIM_PATTERNS = [
        r"확실히\s+~[이가]다",
        r"틀림없이",
        r"100%\s+확실",
        r"명백한\s+사실",
        r"누구나\s+알고\s+있",
        r"모두가\s+인정",
    ]

    @classmethod
    def get_all_rules(cls) -> list[PatternRule]:
        """모든 패턴 규칙 반환"""
        rules = []

        # 욕설
        for pattern in cls.PROFANITY_PATTERNS:
            rules.append(
                PatternRule(
                    pattern=re.compile(pattern, re.IGNORECASE),
                    issue_type=GuardrailIssueType.PROFANITY,
                    severity="high",
                    description="욕설 또는 비속어가 포함되어 있습니다.",
                    suggestion="해당 표현을 삭제하거나 순화된 표현으로 수정하세요.",
                )
            )

        # 혐오 표현
        for pattern in cls.HATE_SPEECH_PATTERNS:
            rules.append(
                PatternRule(
                    pattern=re.compile(pattern, re.IGNORECASE),
                    issue_type=GuardrailIssueType.HATE_SPEECH,
                    severity="critical",
                    description="혐오 발언이 포함되어 있습니다.",
                    suggestion="해당 표현을 완전히 삭제하세요.",
                )
            )

        # 선정적 표현
        for pattern in cls.SENSATIONALISM_PATTERNS:
            rules.append(
                PatternRule(
                    pattern=re.compile(pattern, re.IGNORECASE),
                    issue_type=GuardrailIssueType.SENSATIONALISM,
                    severity="medium",
                    description="선정적이거나 자극적인 표현이 포함되어 있습니다.",
                    suggestion="객관적이고 중립적인 표현으로 수정하세요.",
                )
            )

        # 정치적 편향
        for pattern in cls.POLITICAL_BIAS_KEYWORDS:
            rules.append(
                PatternRule(
                    pattern=re.compile(pattern, re.IGNORECASE),
                    issue_type=GuardrailIssueType.POLITICAL_BIAS,
                    severity="high",
                    description="정치적으로 편향된 표현이 포함되어 있습니다.",
                    suggestion="중립적 관점에서 사실만을 서술하세요.",
                )
            )

        # 개인정보
        for pattern in cls.PERSONAL_INFO_PATTERNS:
            rules.append(
                PatternRule(
                    pattern=re.compile(pattern),
                    issue_type=GuardrailIssueType.PERSONAL_INFO,
                    severity="critical",
                    description="개인정보가 포함되어 있습니다.",
                    suggestion="개인정보를 완전히 삭제하세요.",
                )
            )

        # 검증되지 않은 주장
        for pattern in cls.UNVERIFIED_CLAIM_PATTERNS:
            rules.append(
                PatternRule(
                    pattern=re.compile(pattern, re.IGNORECASE),
                    issue_type=GuardrailIssueType.UNVERIFIED_CLAIM,
                    severity="medium",
                    description="검증되지 않은 주장이 포함되어 있습니다.",
                    suggestion="'~한 것으로 알려졌다', '~라는 의견이 있다' 형태로 수정하세요.",
                )
            )

        return rules


# =============================================================================
# Checker Interface
# =============================================================================


class BaseChecker(ABC):
    """검사기 추상 클래스"""

    @abstractmethod
    async def check(
        self,
        content: str,
        keyword: str | None = None,
        strict_mode: bool = False,
    ) -> list[GuardrailIssue]:
        """
        콘텐츠 검사 수행

        Returns:
            발견된 이슈 목록
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """검사기 이름"""
        pass


# =============================================================================
# Rule-Based Checker (1차 필터)
# =============================================================================


class RuleBasedChecker(BaseChecker):
    """
    규칙 기반 검사기

    정규식 패턴 매칭을 통한 빠른 1차 필터링
    - 장점: 빠름, 확정적, 리소스 적음
    - 단점: 문맥 파악 불가, 우회 가능
    """

    def __init__(self, custom_rules: list[PatternRule] | None = None):
        self.rules = RulePatterns.get_all_rules()
        if custom_rules:
            self.rules.extend(custom_rules)

    async def check(
        self,
        content: str,
        keyword: str | None = None,
        strict_mode: bool = False,
    ) -> list[GuardrailIssue]:
        """규칙 기반 검사 수행"""
        issues = []

        for rule in self.rules:
            matches = rule.pattern.findall(content)
            if matches:
                # 매칭된 텍스트의 위치 찾기
                match = rule.pattern.search(content)
                location = None
                if match:
                    start = max(0, match.start() - 20)
                    end = min(len(content), match.end() + 20)
                    location = f"...{content[start:end]}..."

                issues.append(
                    GuardrailIssue(
                        issue_type=rule.issue_type,
                        severity=rule.severity,
                        description=rule.description,
                        location=location,
                        suggestion=rule.suggestion,
                    )
                )

        return issues

    def get_name(self) -> str:
        return "rule-based-checker"


# =============================================================================
# LLM-Based Checker (2차 검증)
# =============================================================================


class LLMGuardrailOutput(BaseModel):
    """LLM Guardrail 출력 스키마"""

    is_safe: bool = Field(..., description="콘텐츠 안전 여부")
    confidence: float = Field(..., ge=0.0, le=1.0, description="판정 신뢰도")
    issues: list[dict] = Field(default_factory=list, description="발견된 이슈 목록")
    revision_needed: bool = Field(default=False, description="수정 필요 여부")
    revised_content: str | None = Field(default=None, description="수정된 콘텐츠")
    review_reason: str | None = Field(default=None, description="추가 검토 필요 사유")


# LLM Guardrail 시스템 프롬프트
LLM_GUARDRAIL_SYSTEM_PROMPT = """당신은 콘텐츠 검수 전문가입니다. 
제공된 콘텐츠가 공개 발행에 적합한지 검토하세요.

## 검토 기준:

1. **정치적 편향 (political_bias)**
   - 특정 정당이나 정치인을 일방적으로 비난하거나 옹호하는 표현
   - "최악의 정부", "무능한 대통령" 등의 편향적 평가
   - 특정 정치 성향을 비하하는 표현

2. **욕설/비속어 (profanity)**
   - 욕설, 비속어, 비하 표현
   - 인신공격성 표현

3. **허위 정보 (misinformation)**
   - 검증되지 않은 사실을 단정적으로 서술
   - "확실히 ~이다", "틀림없이 ~이다" 형태의 검증 불가 주장

4. **개인정보 (personal_info)**
   - 특정 개인을 식별할 수 있는 정보 (이름+직위 외의 연락처, 주소 등)
   - 전화번호, 이메일, 주민등록번호 등

5. **혐오 발언 (hate_speech)**
   - 특정 집단(성별, 나이, 지역, 인종 등)을 비하하는 표현
   - 차별적 표현

6. **선정적 표현 (sensationalism)**
   - "충격!", "경악!", "발칵 뒤집혀" 등 과장된 표현
   - 과도한 감탄사나 느낌표 사용

7. **검증되지 않은 주장 (unverified_claim)**
   - 출처 없이 "모두가 알고 있다", "누구나 인정한다" 형태의 주장

## 응답 형식:
반드시 아래 JSON 형식으로만 응답하세요.

```json
{
    "is_safe": true/false,
    "confidence": 0.0~1.0,
    "issues": [
        {
            "type": "issue_type",
            "severity": "low/medium/high/critical",
            "description": "이슈 설명",
            "location": "문제 위치 (원문 일부)",
            "suggestion": "수정 제안"
        }
    ],
    "revision_needed": true/false,
    "revised_content": "수정된 콘텐츠 (수정 필요시)",
    "review_reason": "추가 검토 필요 사유 (해당시)"
}
```

## 중요:
- 중립적이고 사실 기반의 콘텐츠는 안전(is_safe: true)합니다.
- "~한 것으로 알려졌다", "~라는 분석이 있다" 형태는 적절합니다.
- 통계 데이터 인용은 적절합니다.
- 애매한 경우 is_safe: false, review_reason에 사유를 기재하세요.
"""


class LLMBasedChecker(BaseChecker):
    """
    LLM 기반 검사기

    LLM을 사용한 심층적 문맥 이해 검사
    - 장점: 문맥 이해, 뉘앙스 파악
    - 단점: 느림, 비용, 비확정적
    """

    def __init__(
        self,
        model_name: str = "qwen2.5:7b-instruct",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.1,
    ):
        self.model_name = model_name
        self.base_url = base_url
        self.temperature = temperature
        self._client = None

    def _get_client(self):
        """Ollama 클라이언트 lazy loading"""
        if self._client is None:
            try:
                from ollama import AsyncClient

                self._client = AsyncClient(host=self.base_url)
            except ImportError:
                raise ImportError("ollama 라이브러리가 필요합니다: pip install ollama")
        return self._client

    async def check(
        self,
        content: str,
        keyword: str | None = None,
        strict_mode: bool = False,
    ) -> list[GuardrailIssue]:
        """LLM 기반 검사 수행"""
        client = self._get_client()

        # 프롬프트 구성
        user_prompt = f"""## 검토 대상 콘텐츠:
{content}

"""
        if keyword:
            user_prompt += f"## 관련 키워드: {keyword}\n\n"

        if strict_mode:
            user_prompt += "## 모드: 엄격 검사 (의심스러운 표현도 이슈로 보고)\n\n"

        user_prompt += "위 콘텐츠를 검토하고 JSON 형식으로 결과를 제공하세요."

        try:
            response = await client.chat(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": LLM_GUARDRAIL_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                format="json",
                options={
                    "temperature": self.temperature,
                    "num_predict": 2048,
                },
            )

            result_text = response["message"]["content"]
            result_data = json.loads(result_text)

            # 이슈 파싱
            issues = []
            for issue_data in result_data.get("issues", []):
                try:
                    issue_type = GuardrailIssueType(issue_data.get("type", "unknown"))
                except ValueError:
                    issue_type = GuardrailIssueType.UNVERIFIED_CLAIM  # fallback

                issues.append(
                    GuardrailIssue(
                        issue_type=issue_type,
                        severity=issue_data.get("severity", "medium"),
                        description=issue_data.get("description", ""),
                        location=issue_data.get("location"),
                        suggestion=issue_data.get("suggestion"),
                    )
                )

            # 추가 메타데이터 저장 (나중에 사용)
            self._last_result = result_data

            return issues

        except Exception as e:
            # LLM 실패 시 빈 결과 반환 (Rule-based만 사용)
            print(f"[WARNING] LLM Guardrail 실패: {e}")
            self._last_result = None
            return []

    def get_last_result(self) -> dict | None:
        """마지막 LLM 검사 결과 반환"""
        return getattr(self, "_last_result", None)

    def get_name(self) -> str:
        return f"llm-checker:{self.model_name}"


# =============================================================================
# Mock Checker (테스트용)
# =============================================================================


class MockLLMChecker(BaseChecker):
    """
    테스트용 Mock LLM 검사기

    Ollama 없이 테스트할 때 사용
    """

    # 테스트용 응답 정의
    MOCK_RESPONSES: dict[str, list[dict]] = {
        "정치적편향테스트": [
            {
                "type": "political_bias",
                "severity": "high",
                "description": "정치적으로 편향된 표현이 포함되어 있습니다.",
                "suggestion": "중립적 표현으로 수정하세요.",
            }
        ],
        "욕설테스트": [
            {
                "type": "profanity",
                "severity": "critical",
                "description": "욕설이 포함되어 있습니다.",
                "suggestion": "해당 표현을 삭제하세요.",
            }
        ],
    }

    async def check(
        self,
        content: str,
        keyword: str | None = None,
        strict_mode: bool = False,
    ) -> list[GuardrailIssue]:
        """Mock 검사 수행"""
        await asyncio.sleep(0.1)  # 시뮬레이션

        issues = []

        # 키워드별 Mock 응답
        if keyword and keyword in self.MOCK_RESPONSES:
            for issue_data in self.MOCK_RESPONSES[keyword]:
                issues.append(
                    GuardrailIssue(
                        issue_type=GuardrailIssueType(issue_data["type"]),
                        severity=issue_data["severity"],
                        description=issue_data["description"],
                        suggestion=issue_data.get("suggestion"),
                    )
                )

        return issues

    def get_name(self) -> str:
        return "mock-llm-checker"


# =============================================================================
# Content Guardrail (통합 클래스)
# =============================================================================


class ContentGuardrail:
    """
    콘텐츠 안전성 검증 통합 클래스

    Blueprint Week 4: 2-Stage Guardrail
    - Stage 1: Rule-based 빠른 필터링
    - Stage 2: LLM 기반 심층 검사

    Usage:
        guardrail = ContentGuardrail()
        result = await guardrail.check(content, keyword="트럼프 관세")

        if result.action == GuardrailAction.PASS:
            print("안전한 콘텐츠")
        elif result.action == GuardrailAction.REVISE:
            print(f"수정된 콘텐츠: {result.revised_content}")
        else:
            print(f"거부됨: {result.issue_summary}")
    """

    def __init__(
        self,
        config: GuardrailConfig | None = None,
        use_mock: bool = False,
    ):
        """
        Args:
            config: Guardrail 설정
            use_mock: True면 Mock LLM 사용 (테스트용)
        """
        self.config = config or GuardrailConfig()

        # 검사기 초기화
        self._rule_checker: BaseChecker | None = None
        self._llm_checker: BaseChecker | None = None
        self._use_mock = use_mock

        if self.config.enable_rule_based:
            self._rule_checker = RuleBasedChecker()

        if self.config.enable_llm_based:
            if use_mock:
                self._llm_checker = MockLLMChecker()
            else:
                self._llm_checker = LLMBasedChecker(
                    model_name=self.config.llm_model,
                    base_url=self.config.llm_base_url,
                    temperature=self.config.llm_temperature,
                )

    async def check(
        self,
        content: str,
        keyword: str | None = None,
        content_type: str = "summary",
        strict_mode: bool | None = None,
    ) -> GuardrailResult:
        """
        콘텐츠 안전성 검사 수행

        Args:
            content: 검사할 콘텐츠
            keyword: 관련 키워드 (컨텍스트 제공)
            content_type: 콘텐츠 유형 (summary, opinion, full)
            strict_mode: 엄격 모드 (None이면 config 따름)

        Returns:
            GuardrailResult: 검사 결과
        """
        if strict_mode is None:
            strict_mode = self.config.strict_mode

        content_id = f"guardrail-{uuid4().hex[:8]}"
        all_issues: list[GuardrailIssue] = []

        # Stage 1: Rule-based 검사
        if self._rule_checker:
            rule_issues = await self._rule_checker.check(content, keyword, strict_mode)
            all_issues.extend(rule_issues)

        # Stage 2: LLM 기반 검사 (Rule-based에서 critical 없을 때만)
        has_critical = any(i.severity == "critical" for i in all_issues)

        if self._llm_checker and not has_critical:
            llm_issues = await self._llm_checker.check(content, keyword, strict_mode)
            # 중복 제거 (같은 issue_type은 하나만)
            existing_types = {i.issue_type for i in all_issues}
            for issue in llm_issues:
                if issue.issue_type not in existing_types:
                    all_issues.append(issue)

        # 결과 판정
        action, is_safe, confidence = self._determine_action(all_issues)

        # 수정 시도 (REVISE일 때)
        revised_content = None
        review_reason = None

        if action == GuardrailAction.REVISE:
            revised_content = await self._attempt_revision(content, all_issues)
        elif action == GuardrailAction.REVIEW:
            review_reason = self._generate_review_reason(all_issues)

        return GuardrailResult(
            content_id=content_id,
            action=action,
            is_safe=is_safe,
            confidence=confidence,
            issues=all_issues,
            original_content=content,
            revised_content=revised_content,
            review_reason=review_reason,
        )

    def _determine_action(
        self, issues: list[GuardrailIssue]
    ) -> tuple[GuardrailAction, bool, float]:
        """
        이슈 목록을 기반으로 액션 결정

        Returns:
            (action, is_safe, confidence)
        """
        if not issues:
            return GuardrailAction.PASS, True, 0.95

        # 심각도별 분류
        critical_count = sum(1 for i in issues if i.severity == "critical")
        high_count = sum(1 for i in issues if i.severity == "high")
        medium_count = sum(1 for i in issues if i.severity == "medium")
        low_count = sum(1 for i in issues if i.severity == "low")

        # Critical이 있으면 무조건 REJECT
        if critical_count > 0:
            return GuardrailAction.REJECT, False, 0.99

        # High가 있으면 REVIEW (사람 검토 필요)
        if high_count > 0:
            if high_count >= 2:
                return GuardrailAction.REJECT, False, 0.90
            return GuardrailAction.REVIEW, False, 0.80

        # Medium만 있으면 REVISE 시도
        if medium_count > 0:
            confidence = max(0.6, 0.9 - medium_count * 0.1)
            return GuardrailAction.REVISE, False, confidence

        # Low만 있으면 PASS (경고만)
        if low_count > 0:
            return GuardrailAction.PASS, True, 0.85

        return GuardrailAction.PASS, True, 0.95

    async def _attempt_revision(self, content: str, issues: list[GuardrailIssue]) -> str | None:
        """자동 수정 시도"""
        revised = content

        for issue in issues:
            if issue.issue_type == GuardrailIssueType.SENSATIONALISM:
                # 느낌표/물음표 과다 수정
                revised = re.sub(r"[!]{2,}", "!", revised)
                revised = re.sub(r"[?]{2,}", "?", revised)
                # 자극적 표현 순화
                revised = re.sub(r"충격[\s!]*[적의]?", "주목할 만한 ", revised)
                revised = re.sub(r"경악", "놀라운", revised)

            elif issue.issue_type == GuardrailIssueType.UNVERIFIED_CLAIM:
                # 단정적 표현 → 추정 표현
                revised = re.sub(r"확실히\s+", "~한 것으로 알려진 ", revised)
                revised = re.sub(r"틀림없이\s+", "~한 것으로 보이는 ", revised)

        # 수정이 실제로 이루어졌는지 확인
        if revised != content:
            return revised
        return None

    def _generate_review_reason(self, issues: list[GuardrailIssue]) -> str:
        """검토 필요 사유 생성"""
        reasons = []
        for issue in issues:
            if issue.severity in ("high", "critical"):
                reasons.append(f"- {issue.issue_type.value}: {issue.description}")
        return "\n".join(reasons) if reasons else "수동 검토가 필요합니다."

    async def check_batch(
        self,
        contents: list[str],
        keyword: str | None = None,
    ) -> list[GuardrailResult]:
        """여러 콘텐츠 일괄 검사"""
        tasks = [self.check(content, keyword) for content in contents]
        return await asyncio.gather(*tasks)


# =============================================================================
# Convenience Functions
# =============================================================================


async def check_content_safety(
    content: str,
    keyword: str | None = None,
    strict_mode: bool = False,
    use_mock: bool = False,
) -> GuardrailResult:
    """
    콘텐츠 안전성 검사 편의 함수

    Usage:
        result = await check_content_safety(
            content="트럼프 대통령이 관세 정책을 발표했습니다.",
            keyword="트럼프 관세"
        )
        print(f"Safe: {result.is_safe}, Action: {result.action}")
    """
    config = GuardrailConfig(strict_mode=strict_mode)
    guardrail = ContentGuardrail(config=config, use_mock=use_mock)
    return await guardrail.check(content, keyword)


# =============================================================================
# CLI Test
# =============================================================================

if __name__ == "__main__":

    async def main():
        """테스트 실행"""
        print("\n" + "=" * 70)
        print("  Week 4 Day 3: Guardrail Test")
        print("=" * 70)

        # 테스트 케이스
        test_cases = [
            {
                "name": "안전한 콘텐츠",
                "content": "트럼프 대통령이 중국산 제품에 관세를 부과한다고 발표했습니다. 이에 따라 국내 수출 기업들이 대응에 나섰으며, 전문가들은 경제에 영향이 있을 것으로 분석하고 있습니다.",
                "expected_safe": True,
            },
            {
                "name": "선정적 표현",
                "content": "충격!! 트럼프의 관세 폭탄으로 전 세계가 발칵 뒤집혔다!!! 이건 완전 대박 사건이다!!!",
                "expected_safe": False,
            },
            {
                "name": "정치적 편향",
                "content": "무능한 정부의 최악의 대응으로 국민들이 고통받고 있다. 이 정권은 역대 최악이다.",
                "expected_safe": False,
            },
            {
                "name": "개인정보 포함",
                "content": "관계자 김OO씨(010-1234-5678)에게 연락하면 자세한 내용을 알 수 있습니다.",
                "expected_safe": False,
            },
            {
                "name": "검증되지 않은 주장",
                "content": "이 정책이 실패할 것은 100% 확실하다. 틀림없이 경제가 망할 것이다.",
                "expected_safe": False,
            },
        ]

        # Mock 모드로 테스트 (Ollama 없이)
        guardrail = ContentGuardrail(use_mock=True)

        for i, test in enumerate(test_cases, 1):
            print(f"\n{'─' * 60}")
            print(f"Test {i}: {test['name']}")
            print(f"{'─' * 60}")
            print(f"Content: {test['content'][:60]}...")

            result = await guardrail.check(test["content"], keyword="테스트")

            status = "✅" if result.is_safe == test["expected_safe"] else "❌"
            print(f"\n{status} Result:")
            print(f"   Action: {result.action.value}")
            print(f"   Safe: {result.is_safe}")
            print(f"   Confidence: {result.confidence:.2f}")
            print(f"   Issues: {len(result.issues)}")

            for issue in result.issues:
                print(
                    f"     - [{issue.severity}] {issue.issue_type.value}: {issue.description[:40]}..."
                )

            if result.revised_content:
                print(f"   Revised: {result.revised_content[:60]}...")

        print("\n" + "=" * 70)
        print("  Test Complete!")
        print("=" * 70)

    asyncio.run(main())
