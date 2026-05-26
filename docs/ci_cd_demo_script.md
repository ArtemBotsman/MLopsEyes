# Сценарий демонстрации CI/CD для проекта Open Eyes Classifier

**Репозиторий:** https://github.com/ArtemBotsman/MLopsEyes  
**Образ в GHCR:** `ghcr.io/artembotsman/mlopseyes:latest`

---

## 1. Цель демонстрации

У нас есть ML-проект — бинарный классификатор изображений глаз (**opened / closed**). Модель возвращает **score от 0.0 до 1.0** (чем ближе к 1.0, тем вероятнее, что глаз открыт).

Завтра на защите мы демонстрируем **не полный MLOps-стек**, а только **CI/CD-контур**:

- **CI** проверяет качество проекта при `pull_request` и `push` в `main`.
- **CD** после попадания изменений в `main` собирает и публикует **Docker-образ** в **GitHub Container Registry (GHCR)**.

---

## 2. Кратко: что такое CI/CD

- **CI (Continuous Integration)** — автоматическая проверка проекта при изменениях в репозитории.
- **CD (Continuous Delivery/Deployment)** — доставка артефакта после успешных проверок. В нашем случае это **публикация Docker-образа в GHCR**.

В ML-проектах это важно, потому что:

- воспроизводимость (одинаковая версия кода/весов в контейнере);
- быстрый фидбек (ошибки ловятся сразу);
- готовность к развёртыванию.

---

## 3. Что уже реализовано в проекте

- GitHub repository: https://github.com/ArtemBotsman/MLopsEyes
- GitHub Actions workflow: `.github/workflows/ci-cd.yml`
- Ruff lint: `ruff check .`
- Тесты: `pytest -v` (категория smoke-тестов CLI)
- Dockerfile: собирает образ с `open_eyes_classifier.py` и весами `eye_cnn_best_val_final.pth`
- CI: на PR и push собирается Docker-образ (без публикации)
- CD: на push/merge в `main` образ публикуется в GHCR
- Образ: `ghcr.io/artembotsman/mlopseyes:latest`

---

## 4. Что открыть на GitHub

1. Откройте репозиторий: https://github.com/ArtemBotsman/MLopsEyes
2. Перейдите во вкладку **Actions**
3. Выберите workflow **CI/CD**
4. Откройте последний **workflow run**
5. Покажите job’ы:
   - `lint-and-test` (Ruff + Pytest)
   - `docker-build` (docker build)
   - `docker-publish` (появляется только на push/merge в `main`)
6. Затем покажите **Packages / GHCR** и найдите образ `mlopseyes:latest`

---

## 4. Как запустить CI/CD — пошагово (команды и что они делают)

Ниже сценарий, который можно буквально повторять на защите.

### Часть A. Подготовка: перейти в папку проекта

1) **Узнать текущую папку**
```bash
pwd
```
Что делает: показывает, где вы сейчас находитесь.

2) **Перейти в папку проекта**
```bash
cd "/Users/artem/Work/Open_eyes_classifier MLops"
```
Что делает: переносит вас в корень проекта.

3) **Проверить, что вы в проекте**
```bash
ls
```
Что должно появиться: `open_eyes_classifier.py`, `Dockerfile`, `tests/`, `.github/`.

### Часть B. Локально повторить то, что делает CI

Выполняйте команды сверху вниз.

4) **Установить dev-зависимости**
```bash
python -m pip install -r requirements-dev.txt
```
Что делает: ставит ruff и pytest (и зависимости проекта).

5) **Запустить линтер Ruff**
```bash
ruff check .
```
Что должно получиться: строка `All checks passed!`

6) **Запустить тесты**
```bash
pytest -v
```
Что должно получиться: `2 passed`

### Часть C. Показать CI на GitHub (через Pull Request)

7) **Убедиться, что вы на main**
```bash
git checkout main
```
Что делает: переключает вас на ветку `main`.

8) **Создать тестовую ветку для демонстрации**
```bash
git checkout -b demo-ci-check
```
Что делает: создаёт новую ветку.

9) **Сделать маленькое изменение (для запуска CI)**
```bash
echo "# CI/CD demo check" >> DEMO.md
```
Что делает: добавляет одну строку в `DEMO.md`.

10) **Коммит**
```bash
git add DEMO.md
git commit -m "docs: add ci cd demo note"
```
Что делает: фиксирует изменения.

11) **Отправить ветку на GitHub**
```bash
git push -u origin demo-ci-check
```
Что делает: после push создаётся PR → стартует CI.

Дальше в браузере:

12) Нажмите **Compare & pull request** → **Create pull request** (цель: `main`).
13) Вкладка **Checks / Actions** → дождитесь зелёных галочек:
   - `lint-and-test`
   - `docker-build`

Важно: на PR **публикация в GHCR не выполняется**, это делает CD только после попадания в `main`.

### Часть D. Показать CD (после попадания в main)

Есть два варианта:

Вариант 1 (проще для защиты): **Merge PR** в `main`.

Что происходит после merge:
- запускается `docker-publish`
- образ публикуется в GHCR

Вариант 2 (если нужно прямо сейчас): **push в main**.

Тогда на GitHub в Actions откройте последний run на `main` и покажите job `docker-publish`.

---

## 5. Что говорить на защите (готовый текст на 2–3 минуты)

На текущем этапе мы показываем CI/CD-контур для проекта Open Eyes Classifier.

При pull request в ветку main автоматически запускаются линтер Ruff, тесты Pytest и сборка Docker-образа — это CI. Мы проверяем, что проект проходит качество и собирается.

После того как изменения попадают в main, запускается docker-publish — это CD. В этом job Docker-образ публикуется в GitHub Container Registry, и его можно использовать как готовый артефакт.

Для демонстрации я локально повторяю проверки и затем показываю на GitHub вкладку Actions и результат в GHCR. Полный MLOps-стек с MLflow, drift detection, UI и мониторингом будет добавлен на следующих этапах.

---

## 6. Команды локальной проверки (коротко)

```bash
pwd
ls
git status
python -m pip install -r requirements-dev.txt
ruff check .
pytest -v
docker build -t open-eyes-classifier .
docker images
docker run --rm -v "$(pwd)/EyesDataset:/app/EyesDataset" open-eyes-classifier EyesDataset/opened/003035.jpg
```

---

## 7. Честное состояние проекта

**Уже сделано:** dataset + baseline model, GitHub, GitHub Actions, ruff lint, pytest tests, Docker build, Docker publish в GHCR.  
**Будет сделано позже:** FastAPI + OpenAPI, DVC, MLflow, drift detection, Prometheus + Grafana, Web UI, Kubernetes/minikube, Argo CD.

---

## 8. Финальный чек-лист перед показом

- [ ] Локально проходит `ruff check .`
- [ ] Локально проходит `pytest -v`
- [ ] В Actions на GitHub зелёный `lint-and-test` и `docker-build` (для PR)
- [ ] После merge/push в main зелёный `docker-publish`
- [ ] Образ виден в GHCR: `ghcr.io/artembotsman/mlopseyes:latest`

