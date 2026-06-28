# Руководство по запуску — Open Eyes Classifier MLOps

Полная шпаргалка: как поднять весь проект и как запускать каждый компонент отдельно.

**Требования:** Python 3.10+, Docker (опционально), Minikube (для k8s-demo).

---

## 1. Первичная настройка

```bash
git clone https://github.com/ArtemBotsman/MLopsEyes.git
cd MLopsEyes

python -m pip install -r requirements-dev.txt
```

### Датасет и веса (DVC)

Тяжёлые файлы не в git — только `.dvc` метаданные. После clone:

```bash
# MinIO для DVC remote (если ещё не запущен)
docker compose -f docker-compose.minio.yml up -d

# Скачать датасет и веса
dvc pull
```

Без DVC: положите `EyesDataset/` и `eye_cnn_best_val_final.pth` в корень вручную.

---

## 2. Запуск всего стека (рекомендуется)

### Docker Compose — полный MLOps-стек

Backend + Frontend + MLflow + Prometheus + Grafana:

```bash
docker compose up --build
```

| Сервис | URL |
|--------|-----|
| Backend API / OpenAPI | http://localhost:8000/docs |
| Streamlit UI | http://localhost:8501 |
| MLflow | http://localhost:5001 |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 (admin/admin) |

Остановка:

```bash
docker compose down
```

### Kubernetes / Minikube — финальная демонстрация

```bash
minikube start
eval $(minikube docker-env)

docker build -f docker/Dockerfile.backend -t mlops-eyes-backend:local .
docker build -f docker/Dockerfile.frontend -t mlops-eyes-frontend:local .

kubectl apply -f k8s/monitoring/namespace.yaml
kubectl apply -f k8s/monitoring/
kubectl get pods -n mlops-eyes
```

Port-forward (в отдельных терминалах):

```bash
kubectl port-forward svc/backend-service 8000:8000 -n mlops-eyes
kubectl port-forward svc/frontend-service 8501:8501 -n mlops-eyes
kubectl port-forward svc/mlflow-service 5001:5000 -n mlops-eyes
kubectl port-forward svc/prometheus-service 9090:9090 -n mlops-eyes
kubectl port-forward svc/grafana-service 3000:3000 -n mlops-eyes
```

---

## 3. Запуск каждого файла / компонента

### CLI — инференс одного изображения

| Файл | Назначение | Команда |
|------|------------|---------|
| `open_eyes_classifier.py` | Предсказание opened/closed | `python open_eyes_classifier.py EyesDataset/opened/003035.jpg` |
| | С другими весами | `python open_eyes_classifier.py image.jpg --weights path/to/weights.pth` |
| | Справка | `python open_eyes_classifier.py --help` |

### Backend API (FastAPI)

| Файл | Назначение | Команда |
|------|------------|---------|
| `backend/app/main.py` | REST API | `uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000` |

OpenAPI: http://localhost:8000/docs

Основные эндпоинты:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/predictions
curl http://localhost:8000/experiments
curl http://localhost:8000/models
curl http://localhost:8000/metrics
curl -X POST http://localhost:8000/retrain -H "Content-Type: application/json" -d '{"epochs": 2}'
curl http://localhost:8000/retrain/status
curl -X POST http://localhost:8000/drift/run
curl http://localhost:8000/drift/latest
```

### Frontend (Streamlit Web UI)

| Файл | Назначение | Команда |
|------|------------|---------|
| `frontend/streamlit_app.py` | Web UI | `API_URL=http://localhost:8000 streamlit run frontend/streamlit_app.py` |

Страницы: Inference, Predictions, Drift, Experiments, Retraining, System status.

### Обучение и переобучение

| Файл | Назначение | Команда |
|------|------------|---------|
| `backend/src/train.py` | Полный train + MLflow | `python -m backend.src.train --config configs/train.yaml` |
| | Быстрый smoke (16 img/class) | `python -m backend.src.train --config configs/train.yaml --fast-dev-run` |
| `scripts/retrain.py` | Быстрый retrain (1–3 эпохи) | `python scripts/retrain.py --epochs 2` |
| `configs/train.yaml` | Конфиг полного обучения | используется через `--config` |
| `configs/retrain.yaml` | Короткий конфиг retrain | используется через `scripts/retrain.py` |

MLflow UI (при запущенном compose):

```bash
MLFLOW_TRACKING_URI=http://localhost:5001 python -m backend.src.train --config configs/train.yaml --fast-dev-run
```

### Drift monitoring

| Файл | Назначение | Команда |
|------|------------|---------|
| `backend/src/drift.py` | Data/target/concept drift | `python -m backend.src.drift --reference EyesDataset --current data/incoming --output reports/drift` |
| `backend/src/reports.py` | Генерация JSON/HTML отчётов | вызывается из `drift.py` |
| `dvc.yaml` (stage drift) | DVC pipeline | `dvc repro drift` |

Отчёты: `reports/drift/latest_drift_report.json`, `.html`

### DVC + MinIO

| Файл | Назначение | Команда |
|------|------------|---------|
| `docker-compose.minio.yml` | MinIO S3 storage | `docker compose -f docker-compose.minio.yml up -d` |
| `.dvc/config` | DVC remote | `dvc push` / `dvc pull` / `dvc status` |
| `dvc.yaml` | Pipeline train + drift | `dvc repro` |

MinIO console: http://localhost:9001 (minioadmin/minioadmin)

### Тесты и линтер

| Файл | Назначение | Команда |
|------|------------|---------|
| `tests/` | Все тесты | `pytest -v` |
| `pyproject.toml` | Ruff config | `ruff check .` |
| `tests/test_cli.py` | CLI тесты | `pytest tests/test_cli.py -v` |
| `tests/test_backend_api.py` | API тесты | `pytest tests/test_backend_api.py -v` |
| `tests/test_retrain_script.py` | Retrain CLI | `pytest tests/test_retrain_script.py -v` |

### Ручная оценка модели

| Файл | Назначение | Команда |
|------|------------|---------|
| `test_model_accuracy.py` | Accuracy на EyesDataset | `python test_model_accuracy.py` |
| `test_model_eyesdataset2.py` | Accuracy на EyesDataset2 | `python test_model_eyesdataset2.py` |
| `project.ipynb` | Обучение в Jupyter | открыть в Jupyter/VS Code |

### Docker-образы

| Файл | Назначение | Команда |
|------|------------|---------|
| `Dockerfile` | CLI-образ | `docker build -t open-eyes-classifier .` |
| `docker/Dockerfile.backend` | Backend API | `docker build -f docker/Dockerfile.backend -t mlops-eyes-backend:local .` |
| `docker/Dockerfile.frontend` | Streamlit UI | `docker build -f docker/Dockerfile.frontend -t mlops-eyes-frontend:local .` |

### Cookiecutter-шаблон

| Файл | Назначение | Команда |
|------|------------|---------|
| `cookiecutter-mlops-eyes/` | MLOps skeleton | `cookiecutter cookiecutter-mlops-eyes --output-dir /tmp` |

Подробнее: `docs/cookiecutter.md`

### Kubernetes manifests

| Путь | Назначение | Команда |
|------|------------|---------|
| `k8s/monitoring/` | Все k8s ресурсы | `kubectl apply -f k8s/monitoring/` |

Файлы: `backend-deployment.yaml`, `frontend-deployment.yaml`, `mlflow-deployment.yaml`, `prometheus-*.yaml`, `grafana-*.yaml`

### CI/CD

| Файл | Назначение |
|------|------------|
| `.github/workflows/ci-cd.yml` | GitHub Actions: ruff, pytest, docker build, GHCR publish |

Запускается автоматически на PR/push в `main`.

---

## 4. Переменные окружения

| Переменная | По умолчанию | Описание |
|------------|--------------|----------|
| `MODEL_WEIGHTS_PATH` | `eye_cnn_best_val_final.pth` | Путь к весам модели |
| `MLFLOW_TRACKING_URI` | `http://localhost:5000` | MLflow server |
| `MLFLOW_REQUEST_TIMEOUT` | `5` | Timeout MLflow API (сек) |
| `API_URL` | `http://localhost:8000` | Backend URL для Streamlit |
| `REFERENCE_DATASET` | `EyesDataset` | Reference для drift |
| `CURRENT_DATASET` | `data/incoming` | Current data для drift |

---

## 5. Быстрая проверка «всё работает»

```bash
ruff check .
pytest -v
python open_eyes_classifier.py EyesDataset/opened/003035.jpg
curl http://localhost:8000/health          # после uvicorn или compose
curl http://localhost:8000/experiments       # после MLflow
python scripts/retrain.py --epochs 1         # быстрый retrain (локально)
```

---

## 6. Дополнительная документация

| Документ | Содержание |
|----------|------------|
| `README.md` | Обзор проекта |
| `docs/mlflow.md` | MLflow tracking и registry |
| `docs/dvc_minio.md` | DVC + MinIO |
| `docs/drift_monitoring.md` | Drift + Prometheus + Grafana |
| `docs/cookiecutter.md` | Cookiecutter template |
| `frontend/README.md` | Streamlit UI |
| `backend/README.md` | Backend API |
