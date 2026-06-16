"""Static checks for DVC and MinIO setup."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_dvc_config_exists():
    assert (ROOT / ".dvc" / "config").is_file()


def test_docker_compose_minio_exists():
    assert (ROOT / "docker-compose.minio.yml").is_file()


def test_dvc_minio_docs_exist():
    assert (ROOT / "docs" / "dvc_minio.md").is_file()


def test_eyes_dataset_dvc_exists():
    assert (ROOT / "EyesDataset.dvc").is_file()


def test_eyes_dataset2_dvc_exists():
    assert (ROOT / "EyesDataset2.dvc").is_file()


def test_weights_dvc_exists():
    assert (ROOT / "eye_cnn_best_val_final.pth.dvc").is_file()


def test_dvc_config_has_no_secrets():
    content = (ROOT / ".dvc" / "config").read_text(encoding="utf-8")
    assert "minioadmin" not in content
    assert "access_key_id" not in content
    assert "secret_access_key" not in content


def test_dvc_yaml_exists():
    assert (ROOT / "dvc.yaml").is_file()


def test_dvc_remote_points_to_minio_bucket():
    content = (ROOT / ".dvc" / "config").read_text(encoding="utf-8")
    assert "s3://mlops-eyes/dvc" in content
    assert "http://localhost:9000" in content


def test_physical_artifacts_still_exist_locally():
    assert (ROOT / "EyesDataset" / "opened").is_dir()
    assert (ROOT / "EyesDataset2").is_dir()
    assert (ROOT / "eye_cnn_best_val_final.pth").is_file()
