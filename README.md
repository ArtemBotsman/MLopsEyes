# Open Eyes Classifier — MLOps CI/CD demo

## Описание проекта

Бинарный классификатор изображений глаз по двум классам:

- **opened** — глаз открыт
- **closed** — глаз закрыт

Модель (**MediumEyeCNN**) возвращает **score** от **0.0** до **1.0**:

- ближе к **1.0** — глаз открыт
- ближе к **0.0** — глаз закрыт

Порог по умолчанию для текстовой интерпретации в CLI: **0.5**.

## Архитектура и демонстрация

| Режим | Назначение |
|-------|------------|
| **Docker Compose** | Локальная отладка всего стека |
| **Kubernetes / Minikube** | **Финальная демонстрация** — backend, frontend, MLflow, Prometheus, Grafana через `k8s/monitoring/` |

Важно:

- **Старый CLI не сломан:** `python open_eyes_classifier.py <image>` работает как раньше.
- **Retrain:** быстрое переобучение через `scripts/retrain.py` (1–3 эпохи) и кнопку в UI (`POST /retrain`).

## Порты (локально / docker compose)

| Сервис | Host | Внутри Docker / k8s |
|--------|------|---------------------|
| Backend (FastAPI) | 8000 | `backend:8000` / `backend-service:8000` |
| Streamlit | 8501 | `frontend:8501` / `frontend-service:8501` |
| MLflow UI | 5001 | `mlflow:5000` / `mlflow-service:5000` |
| Prometheus | 9090 | `prometheus:9090` / `prometheus-service:9090` |
| Grafana | 3000 | `grafana:3000` / `grafana-service:3000` |
| MinIO API / Console | 9000 / 9001 | `localhost:9000` / `localhost:9001` |

## Структура проекта

| Файл / папка | Назначение |
|--------------|------------|
| `open_eyes_classifier.py` | CLI и класс `OpenEyesClassificator` для инференса |
| `eye_cnn_best_val_final.pth` | Актуальные веса обученной модели |
| `EyesDataset/` | Датасет для примеров (`opened/`, `closed/`), DVC-tracked (`EyesDataset.dvc`) |
| `tests/test_cli.py` | Автотесты CLI (help, predict, диапазон score) |
| `Dockerfile` | Образ для запуска классификатора в контейнере |
| `.github/workflows/ci-cd.yml` | GitHub Actions: lint, test, build, publish |
| `requirements.txt` | Runtime-зависимости (PyTorch, Pillow) |
| `requirements-dev.txt` | Dev-зависимости (pytest, ruff) |
| `backend/app/main.py` | FastAPI backend (инференс, drift, metrics) |
| `frontend/` | Streamlit Web UI |
| `docker-compose.yml` | Backend + frontend + MLflow + Prometheus + Grafana |
| `k8s/monitoring/` | Manifests для minikube |
| `docs/drift_monitoring.md` | Документация по drift и мониторингу |
| `docs/run_guide.md` | **Как запустить всё и каждый компонент** |
| `scripts/retrain.py` | Быстрый retrain (1–3 эпохи) |
| `configs/retrain.yaml` | Короткий конфиг для retrain |

Дополнительно в репозитории: `project.ipynb` (обучение), скрипты ручной оценки `test_model_*.py`.

## Быстрый старт

Полное руководство по запуску каждого компонента: **[docs/run_guide.md](docs/run_guide.md)**

```bash
# 1. Зависимости
python -m pip install -r requirements-dev.txt
dvc pull   # датасет и веса

# 2. Весь стек
docker compose up --build

# 3. Проверка
curl http://localhost:8000/health
open http://localhost:8501
```

## Локальный запуск

Требования: **Python 3.10+**.

```bash
python -m pip install -r requirements.txt
python open_eyes_classifier.py EyesDataset/opened/003035.jpg
python open_eyes_classifier.py --help
```

По умолчанию используются веса `eye_cnn_best_val_final.pth`. Другой файл весов:

```bash
python open_eyes_classifier.py путь/к/изображению.jpg --weights путь/к/весам.pth
```

## Запуск тестов и линтера

```bash
python -m pip install -r requirements-dev.txt
ruff check .
pytest -v
```

## Docker

Сборка образа:

```bash
docker build -t open-eyes-classifier .
```

Запуск с монтированием датасета:

```bash
docker run --rm -v "$(pwd)/EyesDataset:/app/EyesDataset" open-eyes-classifier EyesDataset/opened/003035.jpg
```

## CI/CD

Используется **GitHub Actions** (workflow: `.github/workflows/ci-cd.yml`).

**CI** запускается:

- при **pull_request** в `main`
- при **push** в `main`

**CI** выполняет:

- установку зависимостей (`requirements-dev.txt`)
- `ruff check .`
- `pytest -v`
- сборку Docker-образа (`docker build`, без публикации на PR)

**CD** выполняет:

- после **push** / merge в `main` — сборку Docker-образа и публикацию в **GitHub Container Registry (GHCR)**

Опубликованный образ:

```text
ghcr.io/artembotsman/mlopseyes:latest
```

Репозиторий: https://github.com/ArtemBotsman/MLopsEyes

## Что уже реализовано

- GitHub repository
- GitHub Actions workflow
- Ruff lint
- Pytest tests
- Docker build
- Docker publish to GHCR

## Retrain (быстрое переобучение)

```bash
python scripts/retrain.py --epochs 2
# или через API:
curl -X POST http://localhost:8000/retrain -H "Content-Type: application/json" -d '{"epochs": 2}'
```

Конфиг: `configs/retrain.yaml` (1–3 эпохи). Обучение логируется в MLflow.

## Что будет добавлено позже

- Retrain в k8s (Job + DVC dataset в кластере)
- Argo CD (CD в самом конце проекта)

## Backend API

```bash
python -m pip install -r requirements-dev.txt
uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

OpenAPI: http://localhost:8000/docs

Основные эндпоинты:

- `GET /health`
- `POST /predict`
- `GET /predictions`
- `POST /drift/run`
- `GET /drift/latest`
- `GET /metrics`
- `POST /retrain` — быстрый retrain (1–3 эпохи, body: `{"epochs": 2}`)
- `GET /retrain/status`
- `GET /experiments`
- `GET /models`

## MLflow tracking and registry

```bash
docker compose up --build
MLFLOW_TRACKING_URI=http://localhost:5001 python -m backend.src.train --config configs/train.yaml --fast-dev-run
```

Открыть MLflow UI: http://localhost:5001

Подробнее: `docs/mlflow.md`

## DVC + MinIO

Тяжёлые артефакты версионируются через DVC, remote storage — MinIO (S3-compatible):

```bash
docker compose -f docker-compose.minio.yml up -d
dvc status
dvc push
dvc pull
dvc repro
```

- MinIO console: http://localhost:9001 (`minioadmin` / `minioadmin`)
- Bucket: `mlops-eyes`
- DVC-tracked: `EyesDataset`, `EyesDataset2`, `eye_cnn_best_val_final.pth`

Подробнее: `docs/dvc_minio.md`

## Drift reports

```bash
python -m backend.src.drift --reference EyesDataset --current data/incoming --output reports/drift
```

Отчёты сохраняются в:

- `reports/drift/latest_drift_report.json`
- `reports/drift/latest_drift_report.html`

Подробнее: `docs/drift_monitoring.md`

## Prometheus / Grafana (docker compose)

```bash
docker compose up --build
```

- Backend: http://localhost:8000/docs
- Frontend: http://localhost:8501
- MLflow: http://localhost:5001
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (`admin` / `admin`)

Остановить:

```bash
docker compose down
```

## Minikube monitoring

Ручной запуск monitoring-стека и Web UI в Kubernetes (это **не** CD и **не** Argo CD):

```bash
minikube start
eval $(minikube docker-env)
docker build -f docker/Dockerfile.backend -t mlops-eyes-backend:local .
docker build -f docker/Dockerfile.frontend -t mlops-eyes-frontend:local .
kubectl apply -f k8s/monitoring/namespace.yaml
kubectl apply -f k8s/monitoring/
kubectl get pods -n mlops-eyes
kubectl get svc -n mlops-eyes
```

Port-forward:

```bash
kubectl port-forward svc/backend-service 8000:8000 -n mlops-eyes
kubectl port-forward svc/frontend-service 8501:8501 -n mlops-eyes
kubectl port-forward svc/mlflow-service 5000:5000 -n mlops-eyes
kubectl port-forward svc/prometheus-service 9090:9090 -n mlops-eyes
kubectl port-forward svc/grafana-service 3000:3000 -n mlops-eyes
```

Открыть:

- Frontend: http://localhost:8501
- MLflow: http://localhost:5000
- Backend docs: http://localhost:8000/docs
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000

## Web UI

UI реализован на **Streamlit** и доступен по http://localhost:8501.

Страницы:

- inference (загрузка изображения, score, label);
- latest predictions (таблица с фильтрами);
- anomaly flags;
- drift notifications;
- retraining button (1–3 эпохи через API);
- experiments placeholder (под MLflow);
- system status (health + ссылки на мониторинг).

Backend API: http://localhost:8000/docs

Локально:

```bash
API_URL=http://localhost:8000 streamlit run frontend/streamlit_app.py
```

Через docker compose:

```bash
docker compose up --build
```

Подробнее: `frontend/README.md`

## Cookiecutter project template

Шаблон Cookiecutter для воспроизведения типовой MLOps-структуры (backend, frontend, docker, DVC, MLflow, monitoring) без датасетов и весов модели.

Подробнее: [docs/cookiecutter.md](docs/cookiecutter.md)
