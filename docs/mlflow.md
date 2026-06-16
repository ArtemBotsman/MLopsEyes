# MLflow tracking and model registry

## Зачем MLflow в проекте

MLflow используется для:

- трекинга экспериментов обучения;
- логирования параметров, метрик и артефактов;
- регистрации обученной модели в Model Registry (`open-eyes-cnn`);
- просмотра результатов через Web UI и страницу **Experiments** в Streamlit.

## Что логируется

### Параметры

- `epochs`
- `batch_size`
- `learning_rate`
- `fast_dev_run`
- `dataset_path`

### Метрики

- `train_loss`
- `val_loss`
- `accuracy`
- `precision`
- `recall`
- `f1`

### Артефакты

- `confusion_matrix.png`
- trained weights `.pth`
- `metrics.json`
- PyTorch model artifact в MLflow

## Запуск MLflow через docker compose

```bash
docker compose up --build
```

UI: http://localhost:5001

> Порт **5001** на хосте используется вместо 5000, чтобы избежать конфликта с AirPlay Receiver на macOS. Внутри docker network MLflow доступен как `http://mlflow:5000`.

## Fast-dev training

Быстрый демо-запуск обучения на маленьком subset:

```bash
python -m pip install -r requirements-dev.txt
MLFLOW_TRACKING_URI=http://localhost:5001 python -m backend.src.train --config configs/train.yaml --fast-dev-run
```

Или без config:

```bash
python -m backend.src.train --data EyesDataset --epochs 1 --fast-dev-run
```

## Как посмотреть эксперимент

1. Откройте http://localhost:5001
2. Выберите experiment `open-eyes-classifier`
3. Откройте последний run и посмотрите metrics / artifacts

Также можно через backend API:

```bash
curl http://localhost:8000/experiments
```

И через Streamlit UI → страница **Experiments**.

## Как посмотреть registered model

1. В MLflow UI откройте **Models**
2. Найдите модель `open-eyes-cnn`

Или через API:

```bash
curl http://localhost:8000/models
```

Если Model Registry недоступен (например, при локальном `file:./mlruns` без MLflow server), модель всё равно логируется как artifact, а ограничение выводится в консоль training script.

## MLflow в minikube

```bash
minikube start
eval $(minikube docker-env)
docker build -f docker/Dockerfile.backend -t mlops-eyes-backend:local .
docker build -f docker/Dockerfile.frontend -t mlops-eyes-frontend:local .
kubectl apply -f k8s/monitoring/namespace.yaml
kubectl apply -f k8s/monitoring/
kubectl port-forward svc/mlflow-service 5000:5000 -n mlops-eyes
```

Открыть: http://localhost:5000

## Связь с Web UI

Страница **Experiments** в Streamlit:

- показывает ссылку на MLflow UI (`MLFLOW_URL`);
- запрашивает `GET /experiments` и `GET /models` у backend;
- отображает warning, если MLflow server недоступен.
