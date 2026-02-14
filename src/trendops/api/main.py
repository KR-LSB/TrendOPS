# src/trendops/api/main.py
"""
TrendOps FastAPI 메인 애플리케이션

Week 6 Day 4: REST API 서버 구현
"""
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from trendops.api.metrics import init_metrics
from trendops.api.routes import analysis, health, keywords, pipeline, publications
from trendops.api.routes.health import set_startup_time
from trendops.config.settings import get_settings
from trendops.database.connection import get_database

# =============================================================================
# Lifespan Management
# =============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    애플리케이션 수명 주기 관리

    Startup:
        - 데이터베이스 연결
        - 메트릭 초기화
        - 시작 시간 기록

    Shutdown:
        - 데이터베이스 연결 해제
    """
    # === Startup ===
    settings = get_settings()
    db = get_database(settings)

    try:
        await db.connect()
        print(f"✅ Database connected (env: {settings.env})")
    except Exception as e:
        print(f"⚠️ Database connection failed: {e}")

    init_metrics()
    set_startup_time()
    print("✅ TrendOps API Server started")

    yield

    # === Shutdown ===
    await db.disconnect()
    print("✅ Database disconnected")
    print("✅ TrendOps API Server stopped")


# =============================================================================
# FastAPI Application
# =============================================================================

app = FastAPI(
    title="TrendOps API",
    description="""
## TrendOps: 실시간 여론 분석 및 SNS 자동화 파이프라인

### 주요 기능
- 🔍 **트렌드 감지**: Google Trends, Naver DataLab 연동
- 📰 **뉴스 수집**: RSS 기반 자동 수집
- 🤖 **AI 분석**: Local LLM 기반 분석
- 📱 **SNS 발행**: Instagram, Threads 자동 발행
- 📊 **모니터링**: Prometheus + Grafana 대시보드

### API 구조
- `/health` - 시스템 상태 확인
- `/api/keywords` - 키워드 관리
- `/api/analysis` - LLM 분석 결과
- `/api/publications` - SNS 발행 관리
- `/api/pipeline` - 파이프라인 실행 및 모니터링
""",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
    contact={
        "name": "TrendOps Team",
        "url": "https://github.com/trendops",
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    },
)


# =============================================================================
# Middleware
# =============================================================================

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Production에서는 특정 도메인만 허용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request Logging Middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """요청 로깅 미들웨어"""
    start_time = datetime.now()

    response = await call_next(request)

    duration = (datetime.now() - start_time).total_seconds() * 1000

    # 로깅 (개발 환경에서만)
    settings = get_settings()
    if settings.is_development:
        print(
            f"{request.method} {request.url.path} " f"- {response.status_code} ({duration:.1f}ms)"
        )

    return response


# =============================================================================
# Exception Handlers
# =============================================================================


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """전역 예외 핸들러"""
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_server_error",
            "message": str(exc),
            "path": str(request.url.path),
            "timestamp": datetime.now().isoformat(),
        },
    )


# =============================================================================
# Routers
# =============================================================================

# Health Check (prefix 없음)
app.include_router(
    health.router,
    tags=["Health"],
)

# API Routes
app.include_router(
    keywords.router,
    prefix="/api/keywords",
    tags=["Keywords"],
)

app.include_router(
    analysis.router,
    prefix="/api/analysis",
    tags=["Analysis"],
)

app.include_router(
    publications.router,
    prefix="/api/publications",
    tags=["Publications"],
)

app.include_router(
    pipeline.router,
    prefix="/api/pipeline",
    tags=["Pipeline"],
)


# =============================================================================
# Root Endpoints
# =============================================================================


@app.get(
    "/",
    summary="API 정보",
    description="TrendOps API 기본 정보를 반환합니다.",
)
async def root():
    """Root endpoint"""
    return {
        "name": "TrendOps API",
        "version": "1.0.0",
        "status": "running",
        "timestamp": datetime.now().isoformat(),
        "docs": "/docs",
        "health": "/health",
    }


@app.get(
    "/info",
    summary="상세 정보",
    description="API 상세 정보를 반환합니다.",
)
async def info():
    """API 상세 정보"""
    settings = get_settings()
    return {
        "name": "TrendOps API",
        "version": "1.0.0",
        "environment": settings.env,
        "endpoints": {
            "keywords": "/api/keywords",
            "analysis": "/api/analysis",
            "publications": "/api/publications",
            "pipeline": "/api/pipeline",
            "health": "/health",
            "metrics": "/metrics",
            "docs": "/docs",
        },
        "timestamp": datetime.now().isoformat(),
    }


# =============================================================================
# Prometheus Metrics Endpoint
# =============================================================================

try:
    from prometheus_fastapi_instrumentator import Instrumentator

    # FastAPI 자동 계측
    instrumentator = Instrumentator(
        should_group_status_codes=True,
        should_ignore_untemplated=True,
        should_respect_env_var=True,
        should_instrument_requests_inprogress=True,
        excluded_handlers=["/health/*", "/metrics"],
        inprogress_name="trendops_http_requests_inprogress",
        inprogress_labels=True,
    )

    instrumentator.instrument(app).expose(app, endpoint="/metrics")
    print("✅ Prometheus metrics enabled at /metrics")

except ImportError:
    print("⚠️ prometheus-fastapi-instrumentator not installed, metrics disabled")


# =============================================================================
# CLI Entry Point
# =============================================================================


def run_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    reload: bool = False,
):
    """
    서버 실행 (개발용)

    Usage:
        python -m trendops.api.main
    """
    import uvicorn

    uvicorn.run(
        "trendops.api.main:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )


if __name__ == "__main__":
    run_server(reload=True)
