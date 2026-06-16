# Open Eyes Classifier — MLOps CI/CD demo

## Описание проекта

Бинарный классификатор изображений глаз по двум классам:

- **opened** — глаз открыт
- **closed** — глаз закрыт

Модель (**MediumEyeCNN**) возвращает **score** от **0.0** до **1.0**:

- ближе к **1.0** — глаз открыт
- ближе к **0.0** — глаз закрыт

Порог по умолчанию для текстовой интерпретации в CLI: **0.5**.

## Структура проекта

| Файл / папка | Назначение |
|--------------|------------|
| `open_eyes_classifier.py` | CLI и класс `OpenEyesClassificator` для инференса |
| `eye_cnn_best_val_final.pth` | Актуальные веса обученной модели |
| `EyesDataset/` | Датасет для примеров и ручной проверки (`opened/`, `closed/`) |
| `tests/test_cli.py` | Автотесты CLI (help, predict, диапазон score) |
| `Dockerfile` | Образ для запуска классификатора в контейнере |
| `.github/workflows/ci-cd.yml` | GitHub Actions: lint, test, build, publish |
| `requirements.txt` | Runtime-зависимости (PyTorch, Pillow) |
| `requirements-dev.txt` | Dev-зависимости (pytest, ruff) |
| `backend/app/main.py` | FastAPI backend (инференс, drift, metrics) |
| `frontend/` | Заготовка под Web UI (следующий этап) |
| `docker-compose.yml` | Backend + Prometheus + Grafana |
| `k8s/monitoring/` | Manifests для minikube |
| `docs/drift_monitoring.md` | Документация по drift и мониторингу |

Дополнительно в репозитории: `project.ipynb` (обучение), скрипты ручной оценки `test_model_*.py`.

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

## Что будет добавлено позже

- DVC
- MLflow
- Полноценный Web UI в `frontend/`
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
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (`admin` / `admin`)

Остановить:

```bash
docker compose down
```

## Minikube monitoring

Ручной запуск monitoring-стека в Kubernetes (это **не** CD и **не** Argo CD):

```bash
minikube start
eval $(minikube docker-env)
docker build -f docker/Dockerfile.backend -t mlops-eyes-backend:local .
kubectl apply -f k8s/monitoring/namespace.yaml
kubectl apply -f k8s/monitoring/
kubectl get pods -n mlops-eyes
kubectl get svc -n mlops-eyes
```

Port-forward:

```bash
kubectl port-forward svc/backend-service 8000:8000 -n mlops-eyes
kubectl port-forward svc/prometheus-service 9090:9090 -n mlops-eyes
kubectl port-forward svc/grafana-service 3000:3000 -n mlops-eyes
```

## Frontend

Папка `frontend/` подготовлена отдельно. Полноценный UI будет добавлен следующим этапом.
