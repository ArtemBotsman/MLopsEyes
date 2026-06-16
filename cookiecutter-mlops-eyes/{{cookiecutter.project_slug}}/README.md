# {{cookiecutter.project_name}}

MLOps skeleton generated from the **cookiecutter-mlops-eyes** template.

Author: **{{cookiecutter.author_name}}**  
Model name: **{{cookiecutter.model_name}}**  
Python: **{{cookiecutter.python_version}}+**

## Install

```bash
python -m pip install -r requirements-dev.txt
```

## Run backend

```bash
uvicorn backend.app.main:app --reload --host 0.0.0.0 --port {{cookiecutter.backend_port}}
```

Health check: http://localhost:{{cookiecutter.backend_port}}/health  
OpenAPI docs: http://localhost:{{cookiecutter.backend_port}}/docs

## Run frontend

```bash
API_URL=http://localhost:{{cookiecutter.backend_port}} streamlit run frontend/streamlit_app.py --server.port {{cookiecutter.frontend_port}}
```

UI: http://localhost:{{cookiecutter.frontend_port}}

## Run docker compose

```bash
docker compose up --build
```

Services:

| Service | URL |
|---------|-----|
| Backend | http://localhost:{{cookiecutter.backend_port}}/docs |
| Frontend | http://localhost:{{cookiecutter.frontend_port}} |
| MLflow | http://localhost:{{cookiecutter.mlflow_port}} |
| Prometheus | http://localhost:{{cookiecutter.prometheus_port}} |
| Grafana | http://localhost:{{cookiecutter.grafana_port}} (`admin` / `admin`) |

Stop:

```bash
docker compose down
```

## Run tests

```bash
pytest -v
```

## Connect DVC and MLflow

1. Initialize DVC (if not done yet):

```bash
dvc init
```

2. Configure remote storage (S3/MinIO example):

```bash
dvc remote add -d minio s3://mlops-bucket/data
dvc remote modify minio endpointurl http://localhost:9000
```

3. Point training to MLflow:

```bash
export MLFLOW_TRACKING_URI=http://localhost:{{cookiecutter.mlflow_port}}
python -m backend.src.train --epochs 3
```

4. Reproduce the pipeline:

```bash
dvc repro
```

## Project layout

```text
backend/app/main.py      # FastAPI /health smoke endpoint
backend/src/train.py     # training placeholder
frontend/streamlit_app.py
docker/                  # Dockerfiles
k8s/                     # Kubernetes manifests placeholder
monitoring/              # Prometheus/Grafana configs placeholder
tests/test_smoke.py
dvc.yaml / params.yaml   # DVC pipeline skeleton
```
