# src/trendops/utils/logger.py
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class JSONFormatter(logging.Formatter):
    """
    Grafana Loki / ELK 연동을 위한 JSON 포맷 로거
    
    Blueprint Section 7.2: 구조화된 로깅
    Phase 3에서 모니터링 시스템 연동 시 바로 사용 가능
    """
    
    def format(self, record: logging.LogRecord) -> str:
        log_data: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "message": record.getMessage(),
        }
        
        # extra 필드 추가
        if hasattr(record, "extra") and record.extra:
            log_data["extra"] = record.extra
        
        # 예외 정보 추가
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_data, ensure_ascii=False, default=str)


class ExtraAdapter(logging.LoggerAdapter):
    """
    extra 필드를 지원하는 LoggerAdapter
    
    사용 예시:
        logger.info("Message", extra={"key": "value"})
    """
    
    def process(self, msg: str, kwargs: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        extra = kwargs.pop("extra", {})
        
        if extra:
            kwargs["extra"] = {"extra": extra}
        
        return msg, kwargs


class ColorFormatter(logging.Formatter):
    """콘솔 출력용 컬러 포매터"""
    
    COLORS = {
        "DEBUG": "\033[36m",     # Cyan
        "INFO": "\033[32m",      # Green
        "WARNING": "\033[33m",   # Yellow
        "ERROR": "\033[31m",     # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"
    BOLD = "\033[1m"
    
    def format(self, record: logging.LogRecord) -> str:
        # 기본 포맷
        log_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        color = self.COLORS.get(record.levelname, self.RESET)
        
        # 기본 메시지
        formatted = (
            f"{self.BOLD}[{log_time}]{self.RESET} "
            f"{color}{record.levelname:8}{self.RESET} "
            f"\033[90m{record.name}\033[0m: "
            f"{record.getMessage()}"
        )
        
        # extra 필드 추가
        if hasattr(record, "extra") and record.extra:
            extra_str = " ".join(f"{k}={v}" for k, v in record.extra.items())
            formatted += f" \033[90m| {extra_str}\033[0m"
        
        # 예외 정보 추가
        if record.exc_info:
            formatted += f"\n{self.formatException(record.exc_info)}"
        
        return formatted


def setup_logging(
    level: str = "INFO",
    log_dir: str | Path = "logs",
    enable_file: bool = True,
    enable_console: bool = True,
) -> None:
    """
    전역 로깅 설정
    
    Args:
        level: 로그 레벨
        log_dir: 로그 파일 디렉토리
        enable_file: 파일 로깅 활성화
        enable_console: 콘솔 로깅 활성화
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))
    
    # 기존 핸들러 제거
    root_logger.handlers.clear()
    
    # 콘솔 핸들러
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(ColorFormatter())
        root_logger.addHandler(console_handler)
    
    # 파일 핸들러 (JSON Lines)
    if enable_file:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(
            log_path / "trendops.jsonl",
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(JSONFormatter())
        root_logger.addHandler(file_handler)


def get_logger(name: str) -> ExtraAdapter:
    """
    모듈별 로거 생성
    
    Args:
        name: 로거 이름 (보통 __name__ 사용)
        
    Returns:
        ExtraAdapter 인스턴스
        
    사용 예시:
        from trendops.utils.logger import get_logger
        logger = get_logger(__name__)
        logger.info("Keyword detected", extra={"keyword": "트럼프", "score": 8.5})
    """
    # 최초 호출 시 기본 설정 적용
    if not logging.getLogger().handlers:
        setup_logging()
    
    logger = logging.getLogger(name)
    return ExtraAdapter(logger, {})


# 모듈 레벨 기본 로거
_default_logger: ExtraAdapter | None = None


def get_default_logger() -> ExtraAdapter:
    """기본 로거 반환 (싱글톤)"""
    global _default_logger
    if _default_logger is None:
        _default_logger = get_logger("trendops")
    return _default_logger


# 편의를 위한 모듈 레벨 함수들
def debug(msg: str, **kwargs: Any) -> None:
    get_default_logger().debug(msg, **kwargs)


def info(msg: str, **kwargs: Any) -> None:
    get_default_logger().info(msg, **kwargs)


def warning(msg: str, **kwargs: Any) -> None:
    get_default_logger().warning(msg, **kwargs)


def error(msg: str, **kwargs: Any) -> None:
    get_default_logger().error(msg, **kwargs)


def critical(msg: str, **kwargs: Any) -> None:
    get_default_logger().critical(msg, **kwargs)


def exception(msg: str, **kwargs: Any) -> None:
    get_default_logger().exception(msg, **kwargs)