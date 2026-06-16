# Drift monitoring and observability

## Data drift

В этом проекте **data drift** — это изменение распределения признаков изображений между reference и current датасетами.

Сравниваются признаки 24x24 grayscale-изображений:

- `mean_pixel`
- `std_pixel`
- `min_pixel`
- `max_pixel`
- `dark_pixel_ratio` (пиксели < 50)
- `bright_pixel_ratio` (пиксели > 200)
- 8 histogram bins (`hist_0` ... `hist_7`)

Reference по умолчанию: `EyesDataset`  
Current по умолчанию: `data/incoming`

Для каждого признака используется **KS-test** (`scipy`). Если `p_value < 0.05`, признак считается дрейфующим.

## Target drift

**Target drift** — изменение распределения предсказаний модели (opened/closed) между reference и current.

Алгоритм MVP:

1. Получить предсказания модели на reference и current.
2. Посчитать долю `opened`.
3. Если `|reference_opened_ratio - current_opened_ratio| > 0.2`, target drift detected.

## Concept drift

**Concept drift** возможен только если у current данных есть истинные метки.

Если `data/incoming` содержит подпапки:

- `opened/`
- `closed/`

то считаются метрики:

- accuracy
- precision
- recall
- f1

Если accuracy < 0.85, concept drift detected.

Если меток нет:

```json
{
  "concept_drift_status": "not_available",
  "reason": "true labels are not available for current data"
}
```

## Отчёты

После каждого запуска drift detection сохраняются:

- `reports/drift/latest_drift_report.json`
- `reports/drift/latest_drift_report.html`

Если current пустой, отчёт всё равно создаётся со статусом `not_enough_data`.

## Запуск drift report

### Через API

```bash
uvicorn backend.app.main:app --reload
curl -X POST http://localhost:8000/drift/run
curl http://localhost:8000/drift/latest
```

Если отчёт ещё не создан, `GET /drift/latest` возвращает:

```json
{
  "status": "not_available",
  "message": "drift report not generated yet"
}
```

После генерации отчёта ответ содержит `"status": "available"` и поле `report`.

### Через CLI

```bash
python -m backend.src.drift --reference EyesDataset --current data/incoming --output reports/drift
```

## Prometheus

Backend отдаёт метрики на:

```text
http://localhost:8000/metrics
```

Основные метрики:

- `mlops_predictions_total`
- `mlops_opened_predictions_total`
- `mlops_closed_predictions_total`
- `mlops_anomaly_predictions_total`
- `mlops_prediction_latency_seconds`
- `mlops_prediction_score`
- `mlops_drift_runs_total`
- `mlops_drift_alerts_total`
- `mlops_data_drift_detected`
- `mlops_target_drift_detected`
- `mlops_concept_drift_detected`

## Grafana

При запуске через docker compose:

- URL: http://localhost:3000
- login: `admin`
- password: `admin`

Dashboard: `MLOps Eyes Dashboard`

## Docker Compose

```bash
docker compose up --build
```

Открыть:

- Backend OpenAPI: http://localhost:8000/docs
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000

Остановить:

```bash
docker compose down
```

## Kubernetes / minikube

Ручной запуск monitoring в Kubernetes (это **не** CD и **не** Argo CD):

```bash
minikube start
eval $(minikube docker-env)
docker build -f docker/Dockerfile.backend -t mlops-eyes-backend:local .
kubectl apply -f k8s/monitoring/namespace.yaml
kubectl apply -f k8s/monitoring/
kubectl get pods -n mlops-eyes
```

Port-forward:

```bash
kubectl port-forward svc/backend-service 8000:8000 -n mlops-eyes
kubectl port-forward svc/prometheus-service 9090:9090 -n mlops-eyes
kubectl port-forward svc/grafana-service 3000:3000 -n mlops-eyes
```

Удалить ресурсы:

```bash
kubectl delete namespace mlops-eyes
```

## Frontend

Папка `frontend/` подготовлена как отдельный модуль. Полноценный Web UI будет добавлен на следующем этапе и будет использовать backend API.
