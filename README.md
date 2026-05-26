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
| `pyproject.toml` | Настройки Ruff и pytest |

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

- FastAPI + OpenAPI
- DVC
- MLflow
- drift detection
- Prometheus + Grafana
- Web UI
- Kubernetes/minikube
- Argo CD
