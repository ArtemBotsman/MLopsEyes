# Frontend — Streamlit Web UI

Web-интерфейс для MLOps-проекта классификации открытых/закрытых глаз.

## Возможности

- **Inference** — загрузка изображения и получение score / label / anomaly flag;
- **Predictions** — таблица последних предсказаний с фильтрами и метриками;
- **Drift** — запуск drift report и уведомления о дрейфе;
- **Experiments** — заготовка под интеграцию с MLflow;
- **Retraining** — кнопка «Запустить переобучение» (MVP mock);
- **System status** — health backend и ссылки на мониторинг.

## Локальный запуск

Сначала запустите backend:

```bash
uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

Затем frontend:

```bash
python -m pip install -r frontend/requirements.txt
API_URL=http://localhost:8000 streamlit run frontend/streamlit_app.py
```

Откройте: http://localhost:8501

## Переменные окружения

| Переменная | По умолчанию | Назначение |
|------------|--------------|------------|
| `API_URL` | `http://localhost:8000` | Backend API |
| `PROMETHEUS_URL` | `http://localhost:9090` | Prometheus UI |
| `GRAFANA_URL` | `http://localhost:3000` | Grafana UI |
| `MLFLOW_URL` | _(не задан)_ | MLflow UI (опционально) |

## Docker Compose

```bash
docker compose up --build
```

Сервисы:

- Frontend: http://localhost:8501
- Backend docs: http://localhost:8000/docs
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000

Остановка:

```bash
docker compose down
```
