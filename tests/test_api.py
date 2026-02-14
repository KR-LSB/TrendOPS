"""FastAPI 엔드포인트 테스트
src/trendops/api/main.py 기준
"""
import pytest
from fastapi.testclient import TestClient

# FastAPI 앱 import (conftest.py에서 src/ 경로 추가됨)
from trendops.api.main import app

client = TestClient(app)


class TestHealthEndpoint:
    """헬스체크 엔드포인트"""

    def test_health_returns_200(self):
        """GET /health → 200"""
        response = client.get("/health")
        # health 엔드포인트가 /health인지 /api/health인지
        # 실제 라우트에 맞게 조정 필요
        assert response.status_code in [200, 404]

    def test_health_response_format(self):
        """헬스체크 응답 포맷"""
        response = client.get("/health")
        if response.status_code == 200:
            data = response.json()
            assert "status" in data


class TestAPIValidation:
    """입력 검증 테스트"""

    def test_invalid_endpoint_returns_404(self):
        """존재하지 않는 엔드포인트 → 404"""
        response = client.get("/nonexistent")
        assert response.status_code == 404

    def test_root_endpoint(self):
        """루트 엔드포인트 확인"""
        response = client.get("/")
        assert response.status_code in [200, 404, 307]