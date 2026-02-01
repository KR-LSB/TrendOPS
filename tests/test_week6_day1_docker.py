# tests/test_week6_day1_docker.py
"""
TrendOps Week 6 Day 1: Docker Compose 통합 테스트
"""
import json
from pathlib import Path

import pytest
import yaml


@pytest.fixture
def project_root() -> Path:
    """프로젝트 루트 디렉토리"""
    return Path(__file__).parent.parent


def read_file(path: Path) -> str:
    """UTF-8로 파일 읽기 (Windows 호환)"""
    return path.read_text(encoding="utf-8")


# =========================================================================
# FILE EXISTENCE TESTS
# =========================================================================

class TestFileExistence:
    """필수 파일 존재 여부 테스트"""
    
    def test_docker_compose_exists(self, project_root: Path):
        assert (project_root / "docker-compose.yml").exists()
    
    def test_dockerfile_exists(self, project_root: Path):
        assert (project_root / "Dockerfile").exists()
    
    def test_init_db_sql_exists(self, project_root: Path):
        assert (project_root / "scripts" / "init_db.sql").exists()
    
    def test_prometheus_yml_exists(self, project_root: Path):
        assert (project_root / "monitoring" / "prometheus.yml").exists()
    
    def test_grafana_provisioning_exists(self, project_root: Path):
        assert (project_root / "monitoring" / "grafana" / "provisioning" / "datasources" / "prometheus.yml").exists()
        assert (project_root / "monitoring" / "grafana" / "provisioning" / "dashboards" / "default.yml").exists()
    
    def test_api_main_exists(self, project_root: Path):
        assert (project_root / "src" / "trendops" / "api" / "main.py").exists()


# =========================================================================
# DOCKER COMPOSE TESTS
# =========================================================================

class TestDockerCompose:
    """Docker Compose 파일 테스트"""
    
    def test_docker_compose_syntax(self, project_root: Path):
        content = read_file(project_root / "docker-compose.yml")
        config = yaml.safe_load(content)
        assert config is not None
        assert "services" in config
    
    def test_required_services_defined(self, project_root: Path):
        content = read_file(project_root / "docker-compose.yml")
        config = yaml.safe_load(content)
        services = config.get("services", {})
        required = ["api", "redis", "postgres", "chromadb", "prometheus", "grafana"]
        for service in required:
            assert service in services, f"Missing: {service}"
    
    def test_api_service_configuration(self, project_root: Path):
        content = read_file(project_root / "docker-compose.yml")
        config = yaml.safe_load(content)
        api = config["services"]["api"]
        assert "build" in api
        assert any("8000" in str(p) for p in api.get("ports", []))
        assert "depends_on" in api
    
    def test_healthchecks_defined(self, project_root: Path):
        content = read_file(project_root / "docker-compose.yml")
        config = yaml.safe_load(content)
        assert "healthcheck" in config["services"]["redis"]
        assert "healthcheck" in config["services"]["postgres"]
    
    def test_volumes_defined(self, project_root: Path):
        content = read_file(project_root / "docker-compose.yml")
        config = yaml.safe_load(content)
        volumes = config.get("volumes", {})
        required = ["redis_data", "postgres_data", "chroma_data", "prometheus_data", "grafana_data"]
        for vol in required:
            assert vol in volumes, f"Missing volume: {vol}"


# =========================================================================
# DOCKERFILE TESTS
# =========================================================================

class TestDockerfile:
    """Dockerfile 테스트"""
    
    def test_dockerfile_has_from(self, project_root: Path):
        content = read_file(project_root / "Dockerfile")
        assert "FROM" in content
    
    def test_dockerfile_base_image(self, project_root: Path):
        content = read_file(project_root / "Dockerfile")
        assert "python:3.10" in content
    
    def test_dockerfile_has_healthcheck(self, project_root: Path):
        content = read_file(project_root / "Dockerfile")
        assert "HEALTHCHECK" in content
    
    def test_dockerfile_has_cmd(self, project_root: Path):
        content = read_file(project_root / "Dockerfile")
        assert "CMD" in content
        assert "uvicorn" in content
    
    def test_dockerfile_installs_fonts(self, project_root: Path):
        content = read_file(project_root / "Dockerfile")
        assert "fonts-noto-cjk" in content


# =========================================================================
# PROMETHEUS & GRAFANA TESTS
# =========================================================================

class TestMonitoring:
    """모니터링 설정 테스트"""
    
    def test_prometheus_scrape_configs(self, project_root: Path):
        content = read_file(project_root / "monitoring" / "prometheus.yml")
        config = yaml.safe_load(content)
        assert "scrape_configs" in config
        job_names = [job.get("job_name") for job in config["scrape_configs"]]
        assert "trendops-api" in job_names
    
    def test_grafana_datasource(self, project_root: Path):
        content = read_file(project_root / "monitoring" / "grafana" / "provisioning" / "datasources" / "prometheus.yml")
        config = yaml.safe_load(content)
        assert config is not None
        assert "datasources" in config
        assert any(ds.get("type") == "prometheus" for ds in config["datasources"])
    
    def test_grafana_dashboard_json(self, project_root: Path):
        content = read_file(project_root / "monitoring" / "grafana" / "dashboards" / "trendops-pipeline.json")
        dashboard = json.loads(content)
        assert "title" in dashboard
        assert "panels" in dashboard


# =========================================================================
# SUMMARY TEST
# =========================================================================

class TestDay1Summary:
    """Day 1 종합 테스트"""
    
    def test_all_day1_files_present(self, project_root: Path):
        required = [
            "Dockerfile",
            "docker-compose.yml",
            "scripts/init_db.sql",
            "monitoring/prometheus.yml",
            "monitoring/grafana/provisioning/datasources/prometheus.yml",
            "monitoring/grafana/provisioning/dashboards/default.yml",
            "monitoring/grafana/dashboards/trendops-pipeline.json",
            "src/trendops/api/main.py",
        ]
        missing = [f for f in required if not (project_root / f).exists()]
        assert len(missing) == 0, f"Missing: {missing}"