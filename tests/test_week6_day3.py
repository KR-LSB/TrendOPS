"""
TrendOps Week 6 Day 3 Tests
Grafana Dashboard Configuration Tests

테스트 항목:
1. Dashboard JSON 유효성 검증
2. Prometheus 메트릭 모듈 동작 확인
3. 프로비저닝 설정 검증
4. Alert Rules 구문 검증
"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import pytest

# 프로젝트 경로 설정
PROJECT_ROOT = Path(__file__).parent.parent
MONITORING_DIR = PROJECT_ROOT / "monitoring"
GRAFANA_DASHBOARDS_DIR = MONITORING_DIR / "grafana" / "dashboards"

# 메트릭 모듈 임포트
sys.path.insert(0, str(MONITORING_DIR))


class TestDashboardJSON:
    """Dashboard JSON 파일 검증 테스트"""
    
    @pytest.fixture
    def overview_path(self) -> Path:
        return GRAFANA_DASHBOARDS_DIR / "trendops-pipeline.json"
    
    @pytest.fixture
    def detail_path(self) -> Path:
        return GRAFANA_DASHBOARDS_DIR / "trendops-detail.json"
    
    def test_overview_file_exists(self, overview_path: Path) -> None:
        """Overview 대시보드 파일 존재 확인"""
        assert overview_path.exists(), f"Overview dashboard not found: {overview_path}"
    
    def test_detail_file_exists(self, detail_path: Path) -> None:
        """Detail 대시보드 파일 존재 확인"""
        assert detail_path.exists(), f"Detail dashboard not found: {detail_path}"
    
    def test_overview_valid_json(self, overview_path: Path) -> None:
        """Overview 유효한 JSON 형식 확인"""
        with open(overview_path, "r", encoding="utf-8") as f:
            dashboard = json.load(f)
        assert isinstance(dashboard, dict)
    
    def test_detail_valid_json(self, detail_path: Path) -> None:
        """Detail 유효한 JSON 형식 확인"""
        with open(detail_path, "r", encoding="utf-8") as f:
            dashboard = json.load(f)
        assert isinstance(dashboard, dict)
    
    def test_overview_required_fields(self, overview_path: Path) -> None:
        """Overview 필수 필드 존재 확인"""
        with open(overview_path, "r", encoding="utf-8") as f:
            dashboard = json.load(f)
        
        required_fields = ["title", "uid", "panels", "schemaVersion"]
        for field in required_fields:
            assert field in dashboard, f"Missing required field: {field}"
    
    def test_detail_required_fields(self, detail_path: Path) -> None:
        """Detail 필수 필드 존재 확인"""
        with open(detail_path, "r", encoding="utf-8") as f:
            dashboard = json.load(f)
        
        required_fields = ["title", "uid", "panels", "schemaVersion"]
        for field in required_fields:
            assert field in dashboard, f"Missing required field: {field}"
    
    def test_overview_title_and_uid(self, overview_path: Path) -> None:
        """Overview 대시보드 제목/UID 확인"""
        with open(overview_path, "r", encoding="utf-8") as f:
            dashboard = json.load(f)
        
        assert dashboard["title"] == "TrendOps Overview"
        assert dashboard["uid"] == "trendops-overview"
    
    def test_detail_title_and_uid(self, detail_path: Path) -> None:
        """Detail 대시보드 제목/UID 확인"""
        with open(detail_path, "r", encoding="utf-8") as f:
            dashboard = json.load(f)
        
        assert dashboard["title"] == "TrendOps Detail"
        assert dashboard["uid"] == "trendops-detail"
    
    def test_uid_no_conflict(self, overview_path: Path, detail_path: Path) -> None:
        """UID 충돌 없음 확인"""
        with open(overview_path, "r", encoding="utf-8") as f:
            overview = json.load(f)
        with open(detail_path, "r", encoding="utf-8") as f:
            detail = json.load(f)
        
        assert overview["uid"] != detail["uid"], "UID conflict between dashboards!"
    
    def test_overview_has_panels(self, overview_path: Path) -> None:
        """Overview 패널 존재 확인"""
        with open(overview_path, "r", encoding="utf-8") as f:
            dashboard = json.load(f)
        
        assert len(dashboard["panels"]) >= 7, "Overview should have at least 7 panels"
    
    def test_detail_has_panels(self, detail_path: Path) -> None:
        """Detail 패널 존재 확인"""
        with open(detail_path, "r", encoding="utf-8") as f:
            dashboard = json.load(f)
        
        assert len(dashboard["panels"]) >= 10, "Detail should have at least 10 panels"
    
    def test_overview_links_to_detail(self, overview_path: Path) -> None:
        """Overview에서 Detail로 링크 확인"""
        with open(overview_path, "r", encoding="utf-8") as f:
            dashboard = json.load(f)
        
        links = dashboard.get("links", [])
        detail_link = any("trendops-detail" in link.get("url", "") for link in links)
        assert detail_link, "Overview should have link to Detail dashboard"
    
    def test_detail_links_to_overview(self, detail_path: Path) -> None:
        """Detail에서 Overview로 링크 확인"""
        with open(detail_path, "r", encoding="utf-8") as f:
            dashboard = json.load(f)
        
        links = dashboard.get("links", [])
        overview_link = any("trendops-overview" in link.get("url", "") for link in links)
        assert overview_link, "Detail should have link to Overview dashboard"
    
    def test_detail_panel_titles(self, detail_path: Path) -> None:
        """Detail 주요 패널 제목 확인"""
        with open(detail_path, "r", encoding="utf-8") as f:
            dashboard = json.load(f)
        
        panel_titles = [p.get("title", "") for p in dashboard["panels"]]
        
        # 블루프린트 4.3 기준 필수 패널들
        expected_patterns = [
            "Keywords",      # Today's Keywords
            "Documents",     # Documents Collected
            "Latency",       # Pipeline Latency
            "GPU",           # GPU Memory
            "Success Rate",  # Success Rate by Stage
        ]
        
        for pattern in expected_patterns:
            found = any(pattern in title for title in panel_titles)
            assert found, f"Missing panel containing '{pattern}' in title"
    
    def test_dashboards_datasource_references(self, overview_path: Path, detail_path: Path) -> None:
        """모든 패널의 데이터소스 참조 확인"""
        for path in [overview_path, detail_path]:
            with open(path, "r", encoding="utf-8") as f:
                dashboard = json.load(f)
            
            for panel in dashboard["panels"]:
                if "targets" in panel:
                    for target in panel["targets"]:
                        if "datasource" in target:
                            ds = target["datasource"]
                            assert ds.get("type") == "prometheus", (
                                f"Panel '{panel.get('title')}' has non-Prometheus datasource"
                            )
    
    def test_dashboards_refresh_interval(self, overview_path: Path, detail_path: Path) -> None:
        """자동 새로고침 설정 확인"""
        for path in [overview_path, detail_path]:
            with open(path, "r", encoding="utf-8") as f:
                dashboard = json.load(f)
            
            assert "refresh" in dashboard
            assert dashboard["refresh"], "Refresh interval should be set"


class TestProvisioningConfigs:
    """프로비저닝 설정 파일 검증 테스트"""
    
    def test_datasources_yml_exists(self) -> None:
        """Datasources YAML 파일 존재 확인"""
        path = MONITORING_DIR / "provisioning" / "datasources" / "datasources.yml"
        assert path.exists()
    
    def test_dashboards_yml_exists(self) -> None:
        """Dashboards YAML 파일 존재 확인"""
        path = MONITORING_DIR / "provisioning" / "dashboards" / "dashboards.yml"
        assert path.exists()
    
    def test_datasources_yml_valid(self) -> None:
        """Datasources YAML 유효성 확인"""
        import yaml
        
        path = MONITORING_DIR / "provisioning" / "datasources" / "datasources.yml"
        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        
        assert "apiVersion" in config
        assert "datasources" in config
        assert len(config["datasources"]) > 0
    
    def test_datasources_prometheus_config(self) -> None:
        """Prometheus 데이터소스 설정 확인"""
        import yaml
        
        path = MONITORING_DIR / "provisioning" / "datasources" / "datasources.yml"
        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        
        prometheus_ds = None
        for ds in config["datasources"]:
            if ds["name"] == "Prometheus":
                prometheus_ds = ds
                break
        
        assert prometheus_ds is not None, "Prometheus datasource not found"
        assert prometheus_ds["type"] == "prometheus"
        assert prometheus_ds["isDefault"] is True
    
    def test_dashboards_yml_valid(self) -> None:
        """Dashboards YAML 유효성 확인"""
        import yaml
        
        path = MONITORING_DIR / "provisioning" / "dashboards" / "dashboards.yml"
        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        
        assert "apiVersion" in config
        assert "providers" in config
        assert len(config["providers"]) > 0


class TestPrometheusConfig:
    """Prometheus 설정 파일 검증 테스트"""
    
    def test_prometheus_yml_exists(self) -> None:
        """Prometheus YAML 파일 존재 확인"""
        path = MONITORING_DIR / "prometheus.yml"
        assert path.exists()
    
    def test_prometheus_yml_valid(self) -> None:
        """Prometheus YAML 유효성 확인"""
        import yaml
        
        path = MONITORING_DIR / "prometheus.yml"
        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        
        assert "global" in config
        assert "scrape_configs" in config
    
    def test_prometheus_scrape_configs(self) -> None:
        """스크래핑 대상 설정 확인"""
        import yaml
        
        path = MONITORING_DIR / "prometheus.yml"
        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        
        job_names = [sc["job_name"] for sc in config["scrape_configs"]]
        
        # 필수 스크래핑 대상
        assert "trendops-api" in job_names
        assert "vllm" in job_names


class TestAlertRules:
    """알림 규칙 검증 테스트"""
    
    def test_alerts_file_exists(self) -> None:
        """알림 규칙 파일 존재 확인"""
        path = MONITORING_DIR / "alerts" / "trendops_alerts.yml"
        assert path.exists()
    
    def test_alerts_yml_valid(self) -> None:
        """알림 규칙 YAML 유효성 확인"""
        import yaml
        
        path = MONITORING_DIR / "alerts" / "trendops_alerts.yml"
        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        
        assert "groups" in config
        assert len(config["groups"]) > 0
    
    def test_alerts_have_required_fields(self) -> None:
        """알림 규칙 필수 필드 확인"""
        import yaml
        
        path = MONITORING_DIR / "alerts" / "trendops_alerts.yml"
        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        
        for group in config["groups"]:
            assert "name" in group
            assert "rules" in group
            
            for rule in group["rules"]:
                assert "alert" in rule, f"Missing 'alert' in rule"
                assert "expr" in rule, f"Missing 'expr' in rule: {rule.get('alert')}"
                assert "labels" in rule, f"Missing 'labels' in rule: {rule.get('alert')}"
                assert "annotations" in rule, f"Missing 'annotations' in rule: {rule.get('alert')}"
    
    def test_alerts_severity_levels(self) -> None:
        """알림 심각도 수준 확인"""
        import yaml
        
        path = MONITORING_DIR / "alerts" / "trendops_alerts.yml"
        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        
        valid_severities = {"info", "warning", "critical"}
        
        for group in config["groups"]:
            for rule in group["rules"]:
                severity = rule["labels"].get("severity")
                assert severity in valid_severities, (
                    f"Invalid severity '{severity}' in rule '{rule['alert']}'"
                )


class TestMetricsModule:
    """메트릭 모듈 기능 테스트"""
    
    @pytest.fixture
    def metrics_module(self) -> Any:
        """메트릭 모듈 임포트"""
        try:
            from monitoring.metrics import (
                metrics,
                KEYWORDS_DETECTED,
                DOCUMENTS_COLLECTED,
                TRIGGER_DURATION,
                get_metrics_response,
                TrendOpsMetrics,
            )
            return {
                "metrics": metrics,
                "KEYWORDS_DETECTED": KEYWORDS_DETECTED,
                "DOCUMENTS_COLLECTED": DOCUMENTS_COLLECTED,
                "TRIGGER_DURATION": TRIGGER_DURATION,
                "get_metrics_response": get_metrics_response,
                "TrendOpsMetrics": TrendOpsMetrics,
            }
        except ImportError as e:
            pytest.skip(f"Metrics module not available: {e}")
    
    def test_metrics_module_import(self, metrics_module: dict) -> None:
        """메트릭 모듈 임포트 확인"""
        assert metrics_module is not None
    
    def test_trendops_metrics_class(self, metrics_module: dict) -> None:
        """TrendOpsMetrics 클래스 인스턴스 확인"""
        metrics_instance = metrics_module["metrics"]
        assert metrics_instance is not None
    
    def test_record_keyword_detected(self, metrics_module: dict) -> None:
        """키워드 감지 기록 테스트"""
        metrics_instance = metrics_module["metrics"]
        # Should not raise
        metrics_instance.record_keyword_detected("google", count=5)
        metrics_instance.record_keyword_detected("naver", count=3)
    
    def test_record_document_collected(self, metrics_module: dict) -> None:
        """문서 수집 기록 테스트"""
        metrics_instance = metrics_module["metrics"]
        metrics_instance.record_document_collected("youtube", count=100)
        metrics_instance.record_document_collected("news_rss", count=50)
    
    def test_update_queue_size(self, metrics_module: dict) -> None:
        """큐 크기 업데이트 테스트"""
        metrics_instance = metrics_module["metrics"]
        metrics_instance.update_queue_size("pending", 10)
        metrics_instance.update_queue_size("processing", 3)
        metrics_instance.update_queue_size("completed", 47)
        metrics_instance.update_queue_size("failed", 2)
    
    def test_update_gpu_memory(self, metrics_module: dict) -> None:
        """GPU 메모리 업데이트 테스트"""
        metrics_instance = metrics_module["metrics"]
        metrics_instance.update_gpu_memory(vllm_gb=11.5, embedding_gb=0.0)
    
    def test_get_metrics_response(self, metrics_module: dict) -> None:
        """메트릭 응답 생성 테스트"""
        get_response = metrics_module["get_metrics_response"]
        body, content_type = get_response()
        
        assert isinstance(body, bytes)
        assert "text/plain" in content_type or "text/plain" in str(content_type)
    
    def test_metrics_output_format(self, metrics_module: dict) -> None:
        """메트릭 출력 형식 확인"""
        get_response = metrics_module["get_metrics_response"]
        body, _ = get_response()
        
        output = body.decode()
        
        # Prometheus 형식 확인 (# HELP, # TYPE 주석 존재)
        assert "# HELP" in output or "trendops_" in output
    
    @pytest.mark.asyncio
    async def test_track_stage_context_manager(self, metrics_module: dict) -> None:
        """스테이지 추적 컨텍스트 매니저 테스트"""
        metrics_instance = metrics_module["metrics"]
        
        with metrics_instance.track_stage("trigger"):
            await asyncio.sleep(0.01)  # 짧은 지연
    
    @pytest.mark.asyncio
    async def test_track_llm_context_manager(self, metrics_module: dict) -> None:
        """LLM 추적 컨텍스트 매니저 테스트"""
        metrics_instance = metrics_module["metrics"]
        
        with metrics_instance.track_llm("qwen2.5-7b", "summarize"):
            await asyncio.sleep(0.01)


class TestMetricNames:
    """메트릭 이름 규칙 검증 테스트"""
    
    @pytest.fixture
    def metrics_module(self) -> Any:
        try:
            from monitoring.metrics import get_metrics_response
            return get_metrics_response
        except ImportError:
            pytest.skip("Metrics module not available")
    
    def test_metrics_prefix(self, metrics_module: Any) -> None:
        """모든 메트릭이 'trendops_' 접두사를 가지는지 확인"""
        body, _ = metrics_module()
        output = body.decode()
        
        # 메트릭 라인만 추출 (주석 제외)
        metric_lines = [
            line for line in output.split('\n')
            if line and not line.startswith('#') and not line.startswith(' ')
        ]
        
        for line in metric_lines:
            metric_name = line.split('{')[0].split(' ')[0]
            # 빌트인 메트릭 제외
            if metric_name and not metric_name.startswith(('python_', 'process_')):
                assert metric_name.startswith('trendops_'), (
                    f"Metric '{metric_name}' should start with 'trendops_'"
                )


class TestDashboardQueries:
    """대시보드 쿼리 유효성 검증 테스트"""
    
    @pytest.fixture
    def dashboard_queries(self) -> list[dict]:
        """대시보드에서 모든 쿼리 추출"""
        queries = []
        
        for dashboard_file in ["trendops-pipeline.json", "trendops-detail.json"]:
            path = GRAFANA_DASHBOARDS_DIR / dashboard_file
            with open(path, "r", encoding="utf-8") as f:
                dashboard = json.load(f)
            
            for panel in dashboard["panels"]:
                if "targets" in panel:
                    for target in panel["targets"]:
                        if "expr" in target:
                            queries.append({
                                "dashboard": dashboard_file,
                                "panel": panel.get("title", "Unknown"),
                                "expr": target["expr"],
                                "legendFormat": target.get("legendFormat", ""),
                            })
        return queries
    
    def test_queries_exist(self, dashboard_queries: list[dict]) -> None:
        """쿼리 존재 확인"""
        assert len(dashboard_queries) > 0
    
    def test_queries_reference_trendops_metrics(self, dashboard_queries: list[dict]) -> None:
        """쿼리가 trendops_ 메트릭을 참조하는지 확인"""
        for query in dashboard_queries:
            expr = query["expr"]
            # 단순 PromQL 표현식에서 메트릭 이름 확인
            assert "trendops_" in expr, (
                f"Query in panel '{query['panel']}' should reference trendops_ metrics: {expr}"
            )
    
    def test_queries_syntax_basic(self, dashboard_queries: list[dict]) -> None:
        """기본적인 PromQL 구문 확인"""
        for query in dashboard_queries:
            expr = query["expr"]
            
            # 괄호 균형 확인
            assert expr.count('(') == expr.count(')'), (
                f"Unbalanced parentheses in query: {expr}"
            )
            
            # 대괄호 균형 확인
            assert expr.count('[') == expr.count(']'), (
                f"Unbalanced brackets in query: {expr}"
            )
            
            # 중괄호 균형 확인
            assert expr.count('{') == expr.count('}'), (
                f"Unbalanced braces in query: {expr}"
            )


class TestDirectoryStructure:
    """모니터링 디렉토리 구조 검증 테스트"""
    
    def test_monitoring_dir_exists(self) -> None:
        """monitoring 디렉토리 존재 확인"""
        assert MONITORING_DIR.exists()
    
    def test_grafana_dashboards_dir_exists(self) -> None:
        """grafana/dashboards 디렉토리 존재 확인"""
        assert GRAFANA_DASHBOARDS_DIR.exists()
    
    def test_provisioning_dir_exists(self) -> None:
        """provisioning 디렉토리 존재 확인"""
        assert (MONITORING_DIR / "provisioning").exists()
    
    def test_provisioning_subdirs_exist(self) -> None:
        """provisioning 하위 디렉토리 존재 확인"""
        assert (MONITORING_DIR / "provisioning" / "datasources").exists()
        assert (MONITORING_DIR / "provisioning" / "dashboards").exists()
    
    def test_alerts_dir_exists(self) -> None:
        """alerts 디렉토리 존재 확인"""
        assert (MONITORING_DIR / "alerts").exists()
    
    def test_dashboard_files_exist(self) -> None:
        """대시보드 파일 존재 확인"""
        assert (GRAFANA_DASHBOARDS_DIR / "trendops-pipeline.json").exists()
        assert (GRAFANA_DASHBOARDS_DIR / "trendops-detail.json").exists()


# =============================================================================
# Integration Tests
# =============================================================================

class TestMonitoringIntegration:
    """모니터링 통합 테스트"""
    
    def test_all_config_files_present(self) -> None:
        """모든 설정 파일 존재 확인"""
        required_files = [
            MONITORING_DIR / "prometheus.yml",
            GRAFANA_DASHBOARDS_DIR / "trendops-pipeline.json",
            GRAFANA_DASHBOARDS_DIR / "trendops-detail.json",
            MONITORING_DIR / "provisioning" / "datasources" / "datasources.yml",
            MONITORING_DIR / "provisioning" / "dashboards" / "dashboards.yml",
            MONITORING_DIR / "alerts" / "trendops_alerts.yml",
            MONITORING_DIR / "metrics.py",
        ]
        
        for file_path in required_files:
            assert file_path.exists(), f"Missing required file: {file_path}"
    
    def test_config_files_are_valid(self) -> None:
        """모든 설정 파일이 유효한 형식인지 확인"""
        import yaml
        
        yaml_files = [
            MONITORING_DIR / "prometheus.yml",
            MONITORING_DIR / "provisioning" / "datasources" / "datasources.yml",
            MONITORING_DIR / "provisioning" / "dashboards" / "dashboards.yml",
            MONITORING_DIR / "alerts" / "trendops_alerts.yml",
        ]
        
        for yaml_file in yaml_files:
            with open(yaml_file, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            assert config is not None, f"Empty or invalid YAML: {yaml_file}"
        
        # JSON 파일
        json_files = [
            GRAFANA_DASHBOARDS_DIR / "trendops-pipeline.json",
            GRAFANA_DASHBOARDS_DIR / "trendops-detail.json",
        ]
        
        for json_file in json_files:
            with open(json_file, "r", encoding="utf-8") as f:
                dashboard = json.load(f)
            assert dashboard is not None, f"Empty or invalid JSON: {json_file}"


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])