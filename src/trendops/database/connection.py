# src/trendops/database/connection.py
"""
TrendOps 데이터베이스 연결 관리

Week 6 Day 2-4: PostgreSQL 연결 + Connection Pool
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import redis.asyncio as redis
from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from trendops.config.settings import Settings, get_settings


class DatabaseManager:
    """
    데이터베이스 연결 관리자
    
    PostgreSQL, Redis, ChromaDB 연결을 중앙에서 관리
    """
    
    _instance: DatabaseManager | None = None
    
    def __init__(self, settings: Settings | None = None):
        self._settings = settings or get_settings()
        self._engine: AsyncEngine | None = None
        self._session_factory: async_sessionmaker[AsyncSession] | None = None
        self._redis_client: redis.Redis | None = None
        self._connected = False
    
    @classmethod
    def get_instance(cls, settings: Settings | None = None) -> DatabaseManager:
        """싱글톤 인스턴스 반환"""
        if cls._instance is None:
            cls._instance = cls(settings)
        return cls._instance
    
    @classmethod
    def reset_instance(cls) -> None:
        """인스턴스 리셋 (테스트용)"""
        if cls._instance is not None:
            asyncio.get_event_loop().run_until_complete(cls._instance.disconnect())
        cls._instance = None
    
    @property
    def is_connected(self) -> bool:
        """연결 상태 확인"""
        return self._connected
    
    async def connect(self) -> None:
        """모든 데이터베이스 연결"""
        if self._connected:
            return
        
        # PostgreSQL 연결
        postgres_url = self._settings.postgres_url.replace(
            "postgresql://", "postgresql+asyncpg://"
        )
        self._engine = create_async_engine(
            postgres_url,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,
            echo=self._settings.is_development,
        )
        self._session_factory = async_sessionmaker(
            self._engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )
        
        # Redis 연결
        self._redis_client = redis.from_url(
            self._settings.redis_url,
            encoding="utf-8",
            decode_responses=True,
        )
        
        self._connected = True
    
    async def disconnect(self) -> None:
        """모든 데이터베이스 연결 해제"""
        if not self._connected:
            return
        
        if self._engine:
            await self._engine.dispose()
            self._engine = None
        
        if self._redis_client:
            await self._redis_client.close()
            self._redis_client = None
        
        self._session_factory = None
        self._connected = False
    
    @asynccontextmanager
    async def session(self) -> AsyncGenerator[AsyncSession, None]:
        """
        데이터베이스 세션 컨텍스트 매니저
        
        Usage:
            async with db.session() as session:
                result = await session.execute(query)
        """
        if not self._session_factory:
            raise RuntimeError("Database not connected. Call connect() first.")
        
        async with self._session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
    
    async def execute_sql(self, sql: str) -> None:
        """Raw SQL 실행 (마이그레이션, 테스트용)"""
        async with self.session() as session:
            await session.execute(text(sql))
    
    async def check_postgres(self) -> dict:
        """PostgreSQL 연결 상태 확인"""
        try:
            async with self.session() as session:
                result = await session.execute(text("SELECT 1"))
                result.fetchone()
            return {"status": "healthy", "type": "postgres"}
        except Exception as e:
            return {"status": "unhealthy", "type": "postgres", "error": str(e)}
    
    async def check_redis(self) -> dict:
        """Redis 연결 상태 확인"""
        try:
            if self._redis_client:
                await self._redis_client.ping()
                return {"status": "healthy", "type": "redis"}
            return {"status": "unhealthy", "type": "redis", "error": "Not connected"}
        except Exception as e:
            return {"status": "unhealthy", "type": "redis", "error": str(e)}
    
    @property
    def redis(self) -> redis.Redis:
        """Redis 클라이언트 반환"""
        if not self._redis_client:
            raise RuntimeError("Redis not connected")
        return self._redis_client
    
    @property
    def engine(self) -> AsyncEngine:
        """SQLAlchemy 엔진 반환"""
        if not self._engine:
            raise RuntimeError("Database not connected")
        return self._engine


# =============================================================================
# Global Instance
# =============================================================================

def get_database(settings: Settings | None = None) -> DatabaseManager:
    """
    전역 DatabaseManager 인스턴스 반환
    
    Usage:
        db = get_database()
        async with db.session() as session:
            ...
    """
    return DatabaseManager.get_instance(settings)


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI Dependency로 사용할 세션 제공자
    
    Usage:
        @router.get("/")
        async def endpoint(session: AsyncSession = Depends(get_session)):
            ...
    """
    db = get_database()
    async with db.session() as session:
        yield session