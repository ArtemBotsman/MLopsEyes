# Backend API for inference, drift, and Prometheus metrics.

## Endpoints

- `GET /health`
- `POST /predict`
- `GET /predictions`
- `POST /drift/run`
- `GET /drift/latest`
- `GET /metrics`
- OpenAPI: `http://localhost:8000/docs`

## Local run

From project root:

```bash
python -m pip install -r requirements-dev.txt
uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

## Drift CLI

```bash
python -m backend.src.drift --reference EyesDataset --current data/incoming --output reports/drift
```
