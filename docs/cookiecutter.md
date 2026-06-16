# Cookiecutter project template

This repository includes a **Cookiecutter** template that scaffolds a typical MLOps project layout without copying production datasets, model weights, or runtime artifacts.

## Why Cookiecutter?

Cookiecutter turns the Open Eyes MLOps demo into a **reproducible starter kit**. New projects get the same folder structure, service wiring, and documentation patterns as the reference implementation, while remaining lightweight and safe to share.

## What is templated?

The template `cookiecutter-mlops-eyes/` generates:

| Area | Contents |
|------|----------|
| Backend | FastAPI app with `/health` smoke endpoint |
| Frontend | Streamlit UI showing project title and `API_URL` |
| Docker | `Dockerfile.backend`, `Dockerfile.frontend`, `docker-compose.yml` |
| MLOps | `dvc.yaml`, `params.yaml`, MLflow service in compose |
| Monitoring | README placeholders for Prometheus/Grafana |
| Kubernetes | README placeholder for manifests |
| Tests | `tests/test_smoke.py` |
| Tooling | `requirements.txt`, `requirements-dev.txt`, `pyproject.toml` |

Template variables (from `cookiecutter.json`):

- `project_name`, `project_slug`, `author_name`
- `python_version`, `model_name`
- `backend_port`, `frontend_port`, `mlflow_port`, `prometheus_port`, `grafana_port`

## What is intentionally excluded?

The skeleton does **not** include:

- image datasets (`EyesDataset`, DVC-tracked data)
- trained weights (`.pth` files)
- MLflow run history (`mlruns/`)
- production drift reports
- full CNN inference code
- CI/CD workflows or Argo CD manifests

Those belong in the real project or are added after generation. This keeps the template small, fast to clone, and free of large binaries.

## Generate a new project

Install Cookiecutter (also listed in `requirements-dev.txt`):

```bash
python -m pip install cookiecutter
```

From the repository root:

```bash
cookiecutter cookiecutter-mlops-eyes
```

Non-interactive generation with defaults:

```bash
cookiecutter --no-input cookiecutter-mlops-eyes --output-dir /tmp
```

Cookiecutter will prompt for parameters (or use defaults from `cookiecutter.json` with `--no-input`).

## Run the generated project

```bash
cd open_eyes_mlops
python -m pip install -r requirements-dev.txt

# Backend
uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000

# Frontend (separate terminal)
API_URL=http://localhost:8000 streamlit run frontend/streamlit_app.py

# Full stack
docker compose up --build

# Tests
pytest -v
```

See the generated `README.md` inside the new project for port numbers and DVC/MLflow setup hints.

## Relation to this repository

The main repo (`Open_eyes_classifier MLops`) is the **reference implementation** with real data, training, drift, and monitoring. The Cookiecutter template captures the **structure and conventions** so the same MLOps patterns can be bootstrapped elsewhere.
