# src/trendops/core/error_handler.py
"""
Week 4 Day 5: 에러 핸들링 + Circuit Breaker 구현

Blueprint Week 4 Goal: "99% 가동률" 달성을 위한 에러 복구 시스템

구조:
1. Circuit Breaker - 연속 실패 시 자동 차단
2. Exponential Backoff - 지수 백오프 재시도
3. Error Classifier - 에러 분류 및 복구 전략
4. Decorators - 파이프라인 통합용 데코레이터
5. Error Reporter - 로깅 및 알림

Circuit Breaker 상태 전이:
┌────────┐  failure >= threshold  ┌────────┐
│ CLOSED │ ────────────────────▶ │  OPEN  │
│(정상)   │                       │ (차단)  │
└────┬───┘                       └────┬───┘
     │                                │
     │ success                        │ timeout
     │                                ▼
     │                          ┌──────────┐
     └──────────────────────────│HALF_OPEN │
                     success    │ (테스트)  │
                                └──────────┘

사용법:
    # 데코레이터 방식
    @with_retry(max_attempts=3, backoff_base=1.0)
    async def fetch_data():
        ...

    # Circuit Breaker 방식
    breaker = CircuitBreaker(failure_threshold=5)

    async with breaker:
        result = await risky_operation()

    # 통합 방식
    @with_error_handling(
        retry_config=RetryConfig(max_attempts=3),
        circuit_breaker=breaker,
    )
    async def safe_operation():
        ...
"""

from __future__ import annotations

import asyncio
import functools
import logging
import random
import traceback
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Generic, ParamSpec, TypeVar

# Day 2 스키마
try:
    from trendops.schemas import (
        ErrorCategory,
        ErrorSeverity,
        PipelineError,
    )
except ImportError:
    # 단독 실행 시 fallback (테스트용)
    from schemas import (
        ErrorCategory,
        ErrorSeverity,
        PipelineError,
    )


# =============================================================================
# Type Variables
# =============================================================================

T = TypeVar("T")
P = ParamSpec("P")

# Logger
logger = logging.getLogger("trendops.error_handler")


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class RetryConfig:
    """재시도 설정"""

    max_attempts: int = 3
    backoff_base: float = 1.0  # 기본 대기 시간 (초)
    backoff_max: float = 60.0  # 최대 대기 시간 (초)
    backoff_multiplier: float = 2.0  # 지수 백오프 배수
    jitter: bool = True  # 랜덤 지터 추가
    jitter_range: float = 0.5  # 지터 범위 (±50%)

    # 재시도 가능한 에러 카테고리
    retryable_categories: set[ErrorCategory] = field(
        default_factory=lambda: {
            ErrorCategory.NETWORK,
            ErrorCategory.RATE_LIMIT,
            ErrorCategory.LLM,
            ErrorCategory.DATABASE,
        }
    )

    # 재시도 불가능한 예외 타입
    non_retryable_exceptions: tuple[type[Exception], ...] = field(
        default_factory=lambda: (
            ValueError,
            TypeError,
            KeyError,
            AttributeError,
        )
    )

    def calculate_delay(self, attempt: int) -> float:
        """백오프 대기 시간 계산"""
        delay = min(self.backoff_base * (self.backoff_multiplier**attempt), self.backoff_max)

        if self.jitter:
            jitter_factor = 1 + random.uniform(-self.jitter_range, self.jitter_range)
            delay *= jitter_factor

        return delay


@dataclass
class CircuitBreakerConfig:
    """Circuit Breaker 설정"""

    failure_threshold: int = 5  # 연속 실패 임계값
    success_threshold: int = 2  # HALF_OPEN에서 CLOSED로 전환 필요 성공 횟수
    timeout_seconds: float = 30.0  # OPEN 상태 유지 시간
    half_open_max_calls: int = 3  # HALF_OPEN에서 허용할 최대 호출 수


# =============================================================================
# Error Classification
# =============================================================================


class ErrorClassifier:
    """
    에러 분류기

    예외를 분석하여 카테고리, 심각도, 복구 가능 여부 판단
    """

    # 예외 타입별 카테고리 매핑
    EXCEPTION_CATEGORY_MAP: dict[type[Exception], ErrorCategory] = {
        ConnectionError: ErrorCategory.NETWORK,
        TimeoutError: ErrorCategory.NETWORK,
        OSError: ErrorCategory.NETWORK,
        # aiohttp 관련
        # asyncio.TimeoutError: ErrorCategory.NETWORK,
    }

    # 에러 메시지 패턴별 카테고리 매핑
    MESSAGE_PATTERNS: list[tuple[str, ErrorCategory]] = [
        ("rate limit", ErrorCategory.RATE_LIMIT),
        ("too many requests", ErrorCategory.RATE_LIMIT),
        ("429", ErrorCategory.RATE_LIMIT),
        ("connection refused", ErrorCategory.NETWORK),
        ("timeout", ErrorCategory.NETWORK),
        ("timed out", ErrorCategory.NETWORK),
        ("json", ErrorCategory.PARSING),
        ("parse", ErrorCategory.PARSING),
        ("decode", ErrorCategory.PARSING),
        ("validation", ErrorCategory.VALIDATION),
        ("invalid", ErrorCategory.VALIDATION),
        ("llm", ErrorCategory.LLM),
        ("model", ErrorCategory.LLM),
        ("inference", ErrorCategory.LLM),
        ("database", ErrorCategory.DATABASE),
        ("redis", ErrorCategory.DATABASE),
        ("auth", ErrorCategory.AUTHENTICATION),
        ("unauthorized", ErrorCategory.AUTHENTICATION),
        ("forbidden", ErrorCategory.AUTHENTICATION),
    ]

    # 카테고리별 기본 심각도
    CATEGORY_SEVERITY: dict[ErrorCategory, ErrorSeverity] = {
        ErrorCategory.NETWORK: ErrorSeverity.WARNING,
        ErrorCategory.RATE_LIMIT: ErrorSeverity.WARNING,
        ErrorCategory.PARSING: ErrorSeverity.ERROR,
        ErrorCategory.VALIDATION: ErrorSeverity.ERROR,
        ErrorCategory.LLM: ErrorSeverity.ERROR,
        ErrorCategory.DATABASE: ErrorSeverity.ERROR,
        ErrorCategory.AUTHENTICATION: ErrorSeverity.CRITICAL,
        ErrorCategory.UNKNOWN: ErrorSeverity.ERROR,
    }

    # 카테고리별 복구 가능 여부
    CATEGORY_RECOVERABLE: dict[ErrorCategory, bool] = {
        ErrorCategory.NETWORK: True,
        ErrorCategory.RATE_LIMIT: True,
        ErrorCategory.PARSING: False,
        ErrorCategory.VALIDATION: False,
        ErrorCategory.LLM: True,
        ErrorCategory.DATABASE: True,
        ErrorCategory.AUTHENTICATION: False,
        ErrorCategory.UNKNOWN: True,
    }

    @classmethod
    def classify(cls, exception: Exception) -> tuple[ErrorCategory, ErrorSeverity, bool]:
        """
        예외 분류

        Returns:
            (카테고리, 심각도, 복구 가능 여부)
        """
        # 1. 예외 타입으로 분류
        for exc_type, category in cls.EXCEPTION_CATEGORY_MAP.items():
            if isinstance(exception, exc_type):
                return (
                    category,
                    cls.CATEGORY_SEVERITY.get(category, ErrorSeverity.ERROR),
                    cls.CATEGORY_RECOVERABLE.get(category, True),
                )

        # 2. 에러 메시지 패턴으로 분류
        error_msg = str(exception).lower()
        for pattern, category in cls.MESSAGE_PATTERNS:
            if pattern in error_msg:
                return (
                    category,
                    cls.CATEGORY_SEVERITY.get(category, ErrorSeverity.ERROR),
                    cls.CATEGORY_RECOVERABLE.get(category, True),
                )

        # 3. 기본값
        return (ErrorCategory.UNKNOWN, ErrorSeverity.ERROR, True)

    @classmethod
    def create_pipeline_error(
        cls, exception: Exception, stage: str, keyword: str | None = None, **extra
    ) -> PipelineError:
        """예외로부터 PipelineError 생성"""
        category, severity, recoverable = cls.classify(exception)

        return PipelineError(
            category=category,
            severity=severity,
            message=str(exception),
            stage=stage,
            keyword=keyword,
            recoverable=recoverable,
            stack_trace=traceback.format_exc(),
            details=extra,
        )


# =============================================================================
# Circuit Breaker
# =============================================================================


class CircuitState(str, Enum):
    """Circuit Breaker 상태"""

    CLOSED = "closed"  # 정상 - 모든 요청 허용
    OPEN = "open"  # 차단 - 모든 요청 거부
    HALF_OPEN = "half_open"  # 테스트 - 일부 요청만 허용


class CircuitBreakerError(Exception):
    """Circuit Breaker가 열려있을 때 발생하는 예외"""

    def __init__(self, message: str, state: CircuitState, retry_after: float | None = None):
        super().__init__(message)
        self.state = state
        self.retry_after = retry_after


class CircuitBreaker:
    """
    Circuit Breaker 패턴 구현

    연속적인 실패 발생 시 일시적으로 요청을 차단하여
    시스템 과부하를 방지하고 복구 시간을 확보

    Usage:
        breaker = CircuitBreaker(failure_threshold=5)

        # Context Manager 방식
        async with breaker:
            result = await risky_operation()

        # 명시적 호출 방식
        if breaker.can_execute():
            try:
                result = await risky_operation()
                breaker.record_success()
            except Exception as e:
                breaker.record_failure(e)
                raise
    """

    def __init__(
        self,
        name: str = "default",
        config: CircuitBreakerConfig | None = None,
    ):
        self.name = name
        self.config = config or CircuitBreakerConfig()

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: datetime | None = None
        self._half_open_calls = 0

        # 이벤트 콜백
        self._on_state_change: Callable[[CircuitState, CircuitState], None] | None = None
        self._on_failure: Callable[[Exception], None] | None = None

    @property
    def state(self) -> CircuitState:
        """현재 상태 (타임아웃 체크 포함)"""
        if self._state == CircuitState.OPEN:
            if self._should_try_reset():
                self._transition_to(CircuitState.HALF_OPEN)
        return self._state

    @property
    def failure_count(self) -> int:
        return self._failure_count

    @property
    def is_closed(self) -> bool:
        return self.state == CircuitState.CLOSED

    @property
    def is_open(self) -> bool:
        return self.state == CircuitState.OPEN

    def _should_try_reset(self) -> bool:
        """OPEN -> HALF_OPEN 전환 여부 확인"""
        if self._last_failure_time is None:
            return False

        elapsed = (datetime.now() - self._last_failure_time).total_seconds()
        return elapsed >= self.config.timeout_seconds

    def _transition_to(self, new_state: CircuitState):
        """상태 전환"""
        old_state = self._state
        self._state = new_state

        if new_state == CircuitState.HALF_OPEN:
            self._half_open_calls = 0
            self._success_count = 0
        elif new_state == CircuitState.CLOSED:
            self._failure_count = 0
            self._success_count = 0

        if self._on_state_change:
            self._on_state_change(old_state, new_state)

        logger.info(
            f"CircuitBreaker[{self.name}] state changed: {old_state.value} -> {new_state.value}"
        )

    def can_execute(self) -> bool:
        """실행 가능 여부 확인"""
        state = self.state

        if state == CircuitState.CLOSED:
            return True

        if state == CircuitState.HALF_OPEN:
            if self._half_open_calls < self.config.half_open_max_calls:
                self._half_open_calls += 1
                return True
            return False

        # OPEN
        return False

    def record_success(self):
        """성공 기록"""
        if self._state == CircuitState.HALF_OPEN:
            self._success_count += 1
            if self._success_count >= self.config.success_threshold:
                self._transition_to(CircuitState.CLOSED)
        elif self._state == CircuitState.CLOSED:
            # 연속 실패 카운트 리셋
            self._failure_count = 0

    def record_failure(self, exception: Exception | None = None):
        """실패 기록"""
        self._failure_count += 1
        self._last_failure_time = datetime.now()

        if self._on_failure and exception:
            self._on_failure(exception)

        if self._state == CircuitState.HALF_OPEN:
            # HALF_OPEN에서 실패하면 다시 OPEN
            self._transition_to(CircuitState.OPEN)
        elif self._state == CircuitState.CLOSED:
            if self._failure_count >= self.config.failure_threshold:
                self._transition_to(CircuitState.OPEN)

    def on_state_change(self, callback: Callable[[CircuitState, CircuitState], None]):
        """상태 변경 콜백 등록"""
        self._on_state_change = callback

    def on_failure(self, callback: Callable[[Exception], None]):
        """실패 콜백 등록"""
        self._on_failure = callback

    def reset(self):
        """상태 초기화"""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = None
        self._half_open_calls = 0

    def get_retry_after(self) -> float | None:
        """재시도 가능 시간 반환"""
        if self._state != CircuitState.OPEN:
            return None

        if self._last_failure_time is None:
            return None

        elapsed = (datetime.now() - self._last_failure_time).total_seconds()
        remaining = self.config.timeout_seconds - elapsed

        return max(0, remaining)

    async def __aenter__(self) -> CircuitBreaker:
        """Context Manager 진입"""
        if not self.can_execute():
            retry_after = self.get_retry_after()
            raise CircuitBreakerError(
                f"CircuitBreaker[{self.name}] is {self.state.value}",
                self.state,
                retry_after,
            )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context Manager 종료"""
        if exc_type is None:
            self.record_success()
        else:
            self.record_failure(exc_val)
        return False  # 예외 전파

    def get_stats(self) -> dict[str, Any]:
        """통계 반환"""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self._failure_count,
            "success_count": self._success_count,
            "last_failure": self._last_failure_time.isoformat()
            if self._last_failure_time
            else None,
            "retry_after": self.get_retry_after(),
        }


# =============================================================================
# Retry Logic
# =============================================================================


class RetryExhaustedError(Exception):
    """재시도 횟수 초과 예외"""

    def __init__(self, message: str, attempts: int, last_exception: Exception):
        super().__init__(message)
        self.attempts = attempts
        self.last_exception = last_exception


@dataclass
class RetryResult(Generic[T]):
    """재시도 결과"""

    success: bool
    value: T | None = None
    attempts: int = 0
    total_delay: float = 0.0
    last_error: Exception | None = None
    errors: list[Exception] = field(default_factory=list)


async def retry_async(
    func: Callable[[], Awaitable[T]],
    config: RetryConfig | None = None,
    on_retry: Callable[[int, Exception, float], None] | None = None,
) -> RetryResult[T]:
    """
    비동기 함수 재시도 실행

    Args:
        func: 실행할 비동기 함수
        config: 재시도 설정
        on_retry: 재시도 시 콜백 (attempt, exception, delay)

    Returns:
        RetryResult: 재시도 결과
    """
    config = config or RetryConfig()

    errors: list[Exception] = []
    total_delay = 0.0

    for attempt in range(config.max_attempts):
        try:
            result = await func()
            return RetryResult(
                success=True,
                value=result,
                attempts=attempt + 1,
                total_delay=total_delay,
                errors=errors,
            )
        except config.non_retryable_exceptions as e:
            # 재시도 불가능한 예외는 즉시 실패
            errors.append(e)
            return RetryResult(
                success=False,
                attempts=attempt + 1,
                total_delay=total_delay,
                last_error=e,
                errors=errors,
            )
        except Exception as e:
            errors.append(e)

            # 마지막 시도면 실패 반환
            if attempt == config.max_attempts - 1:
                return RetryResult(
                    success=False,
                    attempts=attempt + 1,
                    total_delay=total_delay,
                    last_error=e,
                    errors=errors,
                )

            # 대기 시간 계산
            delay = config.calculate_delay(attempt)
            total_delay += delay

            # 콜백 호출
            if on_retry:
                on_retry(attempt + 1, e, delay)

            logger.warning(
                f"Retry attempt {attempt + 1}/{config.max_attempts} " f"after {delay:.2f}s: {e}"
            )

            # 대기
            await asyncio.sleep(delay)

    # 여기에 도달하면 안 됨
    return RetryResult(
        success=False,
        attempts=config.max_attempts,
        total_delay=total_delay,
        last_error=errors[-1] if errors else None,
        errors=errors,
    )


# =============================================================================
# Decorators
# =============================================================================


def with_retry(
    max_attempts: int = 3,
    backoff_base: float = 1.0,
    backoff_max: float = 60.0,
    retryable_exceptions: tuple[type[Exception], ...] | None = None,
):
    """
    재시도 데코레이터

    Usage:
        @with_retry(max_attempts=3, backoff_base=1.0)
        async def fetch_data():
            ...
    """

    def decorator(func: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]:
        @functools.wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            config = RetryConfig(
                max_attempts=max_attempts,
                backoff_base=backoff_base,
                backoff_max=backoff_max,
            )

            if retryable_exceptions:
                # 지정된 예외만 재시도
                async def inner():
                    try:
                        return await func(*args, **kwargs)
                    except retryable_exceptions:
                        raise
                    except Exception:
                        raise

                result = await retry_async(inner, config)
            else:
                result = await retry_async(
                    lambda: func(*args, **kwargs),
                    config,
                )

            if result.success:
                return result.value  # type: ignore
            else:
                raise RetryExhaustedError(
                    f"Retry exhausted after {result.attempts} attempts",
                    result.attempts,
                    result.last_error,  # type: ignore
                )

        return wrapper

    return decorator


def with_circuit_breaker(
    breaker: CircuitBreaker,
):
    """
    Circuit Breaker 데코레이터

    Usage:
        breaker = CircuitBreaker(name="api")

        @with_circuit_breaker(breaker)
        async def call_api():
            ...
    """

    def decorator(func: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]:
        @functools.wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            async with breaker:
                return await func(*args, **kwargs)

        return wrapper

    return decorator


def with_error_handling(
    stage: str,
    retry_config: RetryConfig | None = None,
    circuit_breaker: CircuitBreaker | None = None,
    on_error: Callable[[PipelineError], None] | None = None,
):
    """
    통합 에러 핸들링 데코레이터

    재시도 + Circuit Breaker + 에러 분류를 모두 적용

    Usage:
        @with_error_handling(
            stage="collector",
            retry_config=RetryConfig(max_attempts=3),
            circuit_breaker=breaker,
        )
        async def collect_data(keyword: str):
            ...
    """

    def decorator(func: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]:
        @functools.wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            # Circuit Breaker 체크
            if circuit_breaker and not circuit_breaker.can_execute():
                retry_after = circuit_breaker.get_retry_after()
                error = PipelineError(
                    category=ErrorCategory.NETWORK,
                    severity=ErrorSeverity.WARNING,
                    message=f"Circuit breaker is {circuit_breaker.state.value}",
                    stage=stage,
                    recoverable=True,
                    retry_after_seconds=int(retry_after) if retry_after else None,
                )
                if on_error:
                    on_error(error)
                raise CircuitBreakerError(
                    "Circuit breaker is open",
                    circuit_breaker.state,
                    retry_after,
                )

            # 재시도 로직
            async def execute():
                try:
                    result = await func(*args, **kwargs)
                    if circuit_breaker:
                        circuit_breaker.record_success()
                    return result
                except Exception as e:
                    if circuit_breaker:
                        circuit_breaker.record_failure(e)
                    raise

            if retry_config:

                def on_retry(attempt: int, exc: Exception, delay: float):
                    error = ErrorClassifier.create_pipeline_error(exc, stage)
                    logger.warning(
                        f"[{stage}] Retry {attempt}: {error.message} " f"(waiting {delay:.1f}s)"
                    )

                result = await retry_async(execute, retry_config, on_retry)

                if result.success:
                    return result.value  # type: ignore
                else:
                    error = ErrorClassifier.create_pipeline_error(
                        result.last_error,  # type: ignore
                        stage,
                    )
                    if on_error:
                        on_error(error)
                    raise result.last_error  # type: ignore
            else:
                try:
                    return await execute()
                except Exception as e:
                    error = ErrorClassifier.create_pipeline_error(e, stage)
                    if on_error:
                        on_error(error)
                    raise

        return wrapper

    return decorator


# =============================================================================
# Error Reporter
# =============================================================================


class ErrorReporter(ABC):
    """에러 리포터 추상 클래스"""

    @abstractmethod
    async def report(self, error: PipelineError) -> None:
        """에러 보고"""
        pass


class LoggingErrorReporter(ErrorReporter):
    """로깅 기반 에러 리포터"""

    def __init__(self, logger_name: str = "trendops.errors"):
        self.logger = logging.getLogger(logger_name)

    async def report(self, error: PipelineError) -> None:
        """에러를 로그로 기록"""
        log_level = {
            ErrorSeverity.DEBUG: logging.DEBUG,
            ErrorSeverity.INFO: logging.INFO,
            ErrorSeverity.WARNING: logging.WARNING,
            ErrorSeverity.ERROR: logging.ERROR,
            ErrorSeverity.CRITICAL: logging.CRITICAL,
        }.get(error.severity, logging.ERROR)

        self.logger.log(
            log_level,
            f"[{error.stage}] {error.category.value}: {error.message}",
            extra=error.to_log_dict(),
        )


class CompositeErrorReporter(ErrorReporter):
    """여러 리포터를 조합한 복합 리포터"""

    def __init__(self, reporters: list[ErrorReporter]):
        self.reporters = reporters

    async def report(self, error: PipelineError) -> None:
        """모든 리포터에 에러 보고"""
        tasks = [reporter.report(error) for reporter in self.reporters]
        await asyncio.gather(*tasks, return_exceptions=True)


# =============================================================================
# Error Handler Manager
# =============================================================================


class ErrorHandlerManager:
    """
    에러 핸들러 관리자

    여러 서비스의 Circuit Breaker와 에러 리포터를 통합 관리

    Usage:
        manager = ErrorHandlerManager()

        # Circuit Breaker 등록
        manager.register_breaker("ollama", CircuitBreaker(name="ollama"))
        manager.register_breaker("redis", CircuitBreaker(name="redis"))

        # 에러 리포터 등록
        manager.add_reporter(LoggingErrorReporter())

        # 사용
        breaker = manager.get_breaker("ollama")
        async with breaker:
            ...
    """

    def __init__(self):
        self._breakers: dict[str, CircuitBreaker] = {}
        self._reporters: list[ErrorReporter] = []
        self._error_history: list[PipelineError] = []
        self._max_history: int = 1000

    def register_breaker(
        self,
        name: str,
        breaker: CircuitBreaker | None = None,
        config: CircuitBreakerConfig | None = None,
    ) -> CircuitBreaker:
        """Circuit Breaker 등록"""
        if breaker is None:
            breaker = CircuitBreaker(name=name, config=config)
        self._breakers[name] = breaker
        return breaker

    def get_breaker(self, name: str) -> CircuitBreaker | None:
        """Circuit Breaker 조회"""
        return self._breakers.get(name)

    def add_reporter(self, reporter: ErrorReporter):
        """에러 리포터 추가"""
        self._reporters.append(reporter)

    async def report_error(self, error: PipelineError):
        """에러 보고 및 기록"""
        # 히스토리 기록
        self._error_history.append(error)
        if len(self._error_history) > self._max_history:
            self._error_history = self._error_history[-self._max_history :]

        # 리포터들에게 보고
        for reporter in self._reporters:
            try:
                await reporter.report(error)
            except Exception as e:
                logger.error(f"Error reporter failed: {e}")

    def get_all_breaker_stats(self) -> dict[str, dict]:
        """모든 Circuit Breaker 통계"""
        return {name: breaker.get_stats() for name, breaker in self._breakers.items()}

    def get_error_summary(self) -> dict[str, Any]:
        """에러 요약"""
        if not self._error_history:
            return {"total": 0, "by_category": {}, "by_stage": {}}

        by_category: dict[str, int] = {}
        by_stage: dict[str, int] = {}
        by_severity: dict[str, int] = {}

        for error in self._error_history:
            cat = error.category.value
            by_category[cat] = by_category.get(cat, 0) + 1

            by_stage[error.stage] = by_stage.get(error.stage, 0) + 1

            sev = error.severity.value
            by_severity[sev] = by_severity.get(sev, 0) + 1

        return {
            "total": len(self._error_history),
            "by_category": by_category,
            "by_stage": by_stage,
            "by_severity": by_severity,
        }

    def reset_all_breakers(self):
        """모든 Circuit Breaker 초기화"""
        for breaker in self._breakers.values():
            breaker.reset()

    def clear_history(self):
        """에러 히스토리 초기화"""
        self._error_history.clear()


# =============================================================================
# Global Instance
# =============================================================================

# 전역 에러 핸들러 매니저
_global_manager: ErrorHandlerManager | None = None


def get_error_manager() -> ErrorHandlerManager:
    """전역 에러 핸들러 매니저 반환"""
    global _global_manager
    if _global_manager is None:
        _global_manager = ErrorHandlerManager()
        _global_manager.add_reporter(LoggingErrorReporter())
    return _global_manager


# =============================================================================
# CLI Test
# =============================================================================

if __name__ == "__main__":

    async def main():
        """테스트 실행"""
        print("\n" + "=" * 70)
        print("  Week 4 Day 5: Error Handler Test")
        print("=" * 70)

        # 로깅 설정
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
        )

        # 1. Circuit Breaker 테스트
        print("\n" + "─" * 60)
        print("Test 1: Circuit Breaker")
        print("─" * 60)

        breaker = CircuitBreaker(
            name="test",
            config=CircuitBreakerConfig(
                failure_threshold=3,
                timeout_seconds=5,
            ),
        )

        # 상태 변경 콜백
        def on_state_change(old: CircuitState, new: CircuitState):
            print(f"  State: {old.value} -> {new.value}")

        breaker.on_state_change(on_state_change)

        # 실패 시뮬레이션
        for i in range(4):
            try:
                async with breaker:
                    raise ConnectionError("Simulated failure")
            except CircuitBreakerError as e:
                print(f"  Blocked by Circuit Breaker: {e.state.value}")
            except ConnectionError:
                pass

        print(f"  Final state: {breaker.state.value}")
        print(f"  Stats: {breaker.get_stats()}")

        # 2. Retry 테스트
        print("\n" + "─" * 60)
        print("Test 2: Retry with Backoff")
        print("─" * 60)

        attempt_count = 0

        async def flaky_operation():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise TimeoutError(f"Attempt {attempt_count} failed")
            return "Success!"

        config = RetryConfig(
            max_attempts=5,
            backoff_base=0.1,
            backoff_multiplier=2,
        )

        result = await retry_async(flaky_operation, config)
        print(f"  Result: {result.success}")
        print(f"  Value: {result.value}")
        print(f"  Attempts: {result.attempts}")
        print(f"  Total delay: {result.total_delay:.2f}s")

        # 3. Error Classifier 테스트
        print("\n" + "─" * 60)
        print("Test 3: Error Classifier")
        print("─" * 60)

        test_exceptions = [
            ConnectionError("Connection refused"),
            TimeoutError("Request timed out"),
            ValueError("Invalid JSON"),
            Exception("rate limit exceeded"),
            Exception("unknown error"),
        ]

        for exc in test_exceptions:
            category, severity, recoverable = ErrorClassifier.classify(exc)
            print(f"  {type(exc).__name__}: {exc}")
            print(f"    -> {category.value}, {severity.value}, recoverable={recoverable}")

        # 4. 데코레이터 테스트
        print("\n" + "─" * 60)
        print("Test 4: Decorators")
        print("─" * 60)

        @with_retry(max_attempts=3, backoff_base=0.1)
        async def decorated_operation():
            return "Decorated success!"

        result = await decorated_operation()
        print(f"  @with_retry result: {result}")

        # 5. Error Manager 테스트
        print("\n" + "─" * 60)
        print("Test 5: Error Handler Manager")
        print("─" * 60)

        manager = get_error_manager()
        manager.register_breaker("api", CircuitBreaker(name="api"))
        manager.register_breaker("db", CircuitBreaker(name="db"))

        # 에러 보고
        error = PipelineError(
            category=ErrorCategory.NETWORK,
            severity=ErrorSeverity.WARNING,
            message="Test error",
            stage="test",
        )
        await manager.report_error(error)

        print(f"  Breaker stats: {manager.get_all_breaker_stats()}")
        print(f"  Error summary: {manager.get_error_summary()}")

        print("\n" + "=" * 70)
        print("  Test Complete!")
        print("=" * 70)

    asyncio.run(main())
