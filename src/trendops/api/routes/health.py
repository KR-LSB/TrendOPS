# src/trendops/api/routes/health.py
"""
TrendOps Health Check 엔드포인트

Week 6 Day 4: 시스템 상태 확인 API
"""
from datetime import datetime
from typing import Literal

from fastapi import APIRouter, Response
from pydantic import BaseModel, Field

from trendops.database.connection import get_database


router = APIRouter()


# =============================================================================
# Schemas
# =============================================================================

class ComponentHealth(BaseModel):
    """개별 컴포넌트 헬스 상태"""
    status: Literal["healthy", "unhealthy", "degraded"]
    latency_ms: float | None = None
    error: str | None = None


class HealthStatus(BaseModel):
    """전체 헬스 상태 응답"""
    status: Literal["healthy", "unhealthy", "degraded"]
    timestamp: str
    version: str = "1.0.0"
    uptime_seconds: float | None = None
    checks: dict[str, ComponentHealth]


class LivenessResponse(BaseModel):
    """Liveness Probe 응답"""
    status: Literal["alive"] = "alive"
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class ReadinessResponse(BaseModel):
    """Readiness Probe 응답"""
    status: Literal["ready", "not_ready"]
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    dependencies: dict[str, bool] = Field(default_factory=dict)


# =============================================================================
# Startup Time Tracking
# =============================================================================

_startup_time: datetime | None = None


def set_startup_time() -> None:
    """애플리케이션 시작 시간 기록"""
    global _startup_time
    _startup_time = datetime.now()


def get_uptime_seconds() -> float | None:
    """업타임 계산 (초)"""
    if _startup_time:
        return (datetime.now() - _startup_time).total_seconds()
    return None


# =============================================================================
# Endpoints
# =============================================================================

@router.get(
    "/health",
    response_model=HealthStatus,
    summary="시스템 전체 상태 확인",
    description="모든 의존 서비스의 상태를 확인합니다.",
)
async def health_check() -> HealthStatus:
    """
    시스템 상태 확인
    
    - **database**: PostgreSQL 연결 상태
    - **redis**: Redis 연결 상태
    - **chromadb**: ChromaDB 연결 상태 (선택적)
    """
    checks: dict[str, ComponentHealth] = {}
    overall_status: Literal["healthy", "unhealthy", "degraded"] = "healthy"
    
    db = get_database()
    
    # PostgreSQL 체크
    try:
        start = datetime.now()
        pg_result = await db.check_postgres()
        latency = (datetime.now() - start).total_seconds() * 1000
        
        if pg_result["status"] == "healthy":
            checks["database"] = ComponentHealth(
                status="healthy",
                latency_ms=round(latency, 2),
            )
        else:
            checks["database"] = ComponentHealth(
                status="unhealthy",
                error=pg_result.get("error"),
            )
            overall_status = "unhealthy"
    except Exception as e:
        checks["database"] = ComponentHealth(
            status="unhealthy",
            error=str(e),
        )
        overall_status = "unhealthy"
    
    # Redis 체크
    try:
        start = datetime.now()
        redis_result = await db.check_redis()
        latency = (datetime.now() - start).total_seconds() * 1000
        
        if redis_result["status"] == "healthy":
            checks["redis"] = ComponentHealth(
                status="healthy",
                latency_ms=round(latency, 2),
            )
        else:
            checks["redis"] = ComponentHealth(
                status="unhealthy",
                error=redis_result.get("error"),
            )
            # Redis 실패는 degraded로 처리 (비필수)
            if overall_status == "healthy":
                overall_status = "degraded"
    except Exception as e:
        checks["redis"] = ComponentHealth(
            status="unhealthy",
            error=str(e),
        )
        if overall_status == "healthy":
            overall_status = "degraded"
    
    return HealthStatus(
        status=overall_status,
        timestamp=datetime.now().isoformat(),
        uptime_seconds=get_uptime_seconds(),
        checks=checks,
    )


@router.get(
    "/health/liveness",
    response_model=LivenessResponse,
    summary="Liveness Probe",
    description="Kubernetes Liveness Probe용 엔드포인트",
)
async def liveness() -> LivenessResponse:
    """
    Kubernetes Liveness Probe
    
    애플리케이션이 살아있는지 확인합니다.
    실패 시 컨테이너가 재시작됩니다.
    """
    return LivenessResponse()


@router.get(
    "/health/readiness",
    response_model=ReadinessResponse,
    summary="Readiness Probe",
    description="Kubernetes Readiness Probe용 엔드포인트",
)
async def readiness(response: Response) -> ReadinessResponse:
    """
    Kubernetes Readiness Probe
    
    모든 의존성이 준비되었는지 확인합니다.
    실패 시 트래픽이 라우팅되지 않습니다.
    """
    db = get_database()
    dependencies: dict[str, bool] = {}
    all_ready = True
    
    # PostgreSQL 확인
    try:
        pg_result = await db.check_postgres()
        dependencies["postgres"] = pg_result["status"] == "healthy"
        if not dependencies["postgres"]:
            all_ready = False
    except Exception:
        dependencies["postgres"] = False
        all_ready = False
    
    # Redis 확인
    try:
        redis_result = await db.check_redis()
        dependencies["redis"] = redis_result["status"] == "healthy"
        # Redis는 선택적이므로 all_ready에 영향 안 줌
    except Exception:
        dependencies["redis"] = False
    
    if not all_ready:
        response.status_code = 503
    
    return ReadinessResponse(
        status="ready" if all_ready else "not_ready",
        dependencies=dependencies,
    )


@router.get(
    "/health/startup",
    summary="Startup Probe",
    description="Kubernetes Startup Probe용 엔드포인트",
)
async def startup_check() -> dict:
    """
    Kubernetes Startup Probe
    
    애플리케이션이 시작 완료되었는지 확인합니다.
    """
    return {
        "status": "started",
        "timestamp": datetime.now().isoformat(),
    }