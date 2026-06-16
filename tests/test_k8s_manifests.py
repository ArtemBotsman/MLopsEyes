"""Static checks for Kubernetes monitoring manifests."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
K8S_DIR = ROOT / "k8s" / "monitoring"


def test_frontend_deployment_exists():
    assert (K8S_DIR / "frontend-deployment.yaml").is_file()


def test_frontend_service_exists():
    assert (K8S_DIR / "frontend-service.yaml").is_file()


def test_frontend_deployment_image_pull_policy_never():
    content = (K8S_DIR / "frontend-deployment.yaml").read_text(encoding="utf-8")
    assert "imagePullPolicy: Never" in content


def test_frontend_deployment_api_url_points_to_backend_service():
    content = (K8S_DIR / "frontend-deployment.yaml").read_text(encoding="utf-8")
    assert "http://backend-service:8000" in content


def test_mlflow_deployment_exists():
    assert (K8S_DIR / "mlflow-deployment.yaml").is_file()


def test_mlflow_service_exists():
    assert (K8S_DIR / "mlflow-service.yaml").is_file()


def test_mlflow_deployment_allows_service_host():
    content = (K8S_DIR / "mlflow-deployment.yaml").read_text(encoding="utf-8")
    assert "--allowed-hosts" in content
    assert "mlflow-service" in content


def test_backend_deployment_sets_mlflow_request_timeout():
    content = (K8S_DIR / "backend-deployment.yaml").read_text(encoding="utf-8")
    assert "MLFLOW_REQUEST_TIMEOUT" in content
