# DVC and MinIO

## Зачем DVC в проекте

DVC (Data Version Control) версионирует **тяжёлые артефакты** отдельно от Git:

- датасеты `EyesDataset`, `EyesDataset2`;
- веса модели `eye_cnn_best_val_final.pth`;
- кэш выходов pipeline (`dvc.yaml`).

Git хранит **код**, `.dvc`-метаданные и pipeline-описание. Фактические данные — в DVC remote (MinIO).

## Почему датасет и веса не в Git

- Git не предназначен для тысяч бинарных файлов и сотен мегабайт весов.
- Клонирование репозитория остаётся быстрым.
- Версии данных и модели управляются через DVC + remote storage.

## MinIO как S3-compatible remote

MinIO выступает локальным S3-совместимым хранилищем для DVC:

| Параметр | Значение |
|----------|----------|
| Bucket | `mlops-eyes` |
| DVC remote path | `s3://mlops-eyes/dvc` |
| S3 endpoint | `http://localhost:9000` |
| Console | http://localhost:9001 |
| Login | `minioadmin` |
| Password | `minioadmin` |

Секреты (`access_key_id`, `secret_access_key`) хранятся только в `.dvc/config.local` (не коммитится).

## Запуск MinIO

```bash
docker compose -f docker-compose.minio.yml up -d
```

Console: http://localhost:9001

Остановка:

```bash
docker compose -f docker-compose.minio.yml down
```

## DVC remote (уже настроен в проекте)

```bash
dvc remote add -d minio s3://mlops-eyes/dvc
dvc remote modify minio endpointurl http://localhost:9000
dvc remote modify minio use_ssl false
dvc remote modify --local minio access_key_id minioadmin
dvc remote modify --local minio secret_access_key minioadmin
```

## Основные команды

### Проверить статус

```bash
dvc status
```

### Загрузить данные в MinIO

```bash
dvc push
```

### Скачать данные с MinIO

```bash
dvc pull
```

### Запустить pipeline

```bash
dvc repro
```

Стадии в `dvc.yaml`:

- **train** — fast-dev обучение с MLflow logging;
- **drift** — генерация drift report.

Параметры — в `params.yaml`.

## Что отслеживается DVC

| Артефакт | `.dvc` файл |
|----------|-------------|
| `EyesDataset/` | `EyesDataset.dvc` |
| `EyesDataset2/` | `EyesDataset2.dvc` |
| `eye_cnn_best_val_final.pth` | `eye_cnn_best_val_final.pth.dvc` |

## CI / GitHub Actions

Локальный MinIO на `localhost:9000` **недоступен** из GitHub Actions.

Для CI нужны:

- внешний S3/MinIO endpoint (AWS S3, managed MinIO, etc.);
- GitHub Secrets для `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` или DVC remote credentials;
- `dvc pull` в workflow перед train/test.

## Связь с остальным проектом

- **Git** — код, `.dvc`, `dvc.yaml`, `params.yaml`, docs.
- **DVC + MinIO** — датасеты, веса, кэш pipeline outputs.
- **MLflow** — эксперименты и Model Registry (отдельно от DVC data remote).
