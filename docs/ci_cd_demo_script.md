# Сценарий демонстрации CI/CD для проекта Open Eyes Classifier

**Репозиторий:** https://github.com/ArtemBotsman/MLopsEyes  
**Образ в GHCR:** `ghcr.io/artembotsman/mlopseyes:latest`

---

## 1. Цель демонстрации

У нас есть **ML-проект** — классификатор изображений глаз на два состояния: **opened** (открыт) и **closed** (закрыт). Модель выдаёт **score от 0.0 до 1.0** (чем ближе к 1.0, тем вероятнее, что глаз открыт).

**Завтра на защите мы показываем не весь MLOps**, а **именно CI/CD-контур** — как код и модель автоматически проверяются и доставляются в виде Docker-образа.

- **CI (Continuous Integration)** — при каждом изменении в репозитории автоматически проверяется качество проекта: линтер, тесты, сборка образа.
- **CD (Continuous Delivery/Deployment)** — после успешной проверки и попадания изменений в ветку **main** образ **собирается и публикуется** в **GitHub Container Registry (GHCR)**, откуда его можно скачать и запустить на любой машине.

Так преподаватель видит: проект не «лежит на ноутбуке», а **воспроизводим** и **готов к развёртыванию**.

---

## 2. Кратко: что такое CI/CD

### CI — Continuous Integration

**Простыми словами:** при **pull request** или **push** в GitHub автоматически запускается pipeline, который:

- ставит зависимости;
- проверяет стиль кода (**ruff**);
- запускает тесты (**pytest**);
- пробует собрать **Docker-образ**.

Если что-то сломалось — вы сразу видите красный крестик в Actions, а не узнаёте об этом на защите.

### CD — Continuous Delivery

**В нашем проекте:** после **push в main** (или merge PR) дополнительно запускается job **docker-publish** — образ пушится в **GHCR** с тегами `latest` и `sha` (хеш коммита).

**Зачем это в ML-проекте:**

- одна и та же версия модели и кода в контейнере;
- коллега или сервер может запустить `docker pull` без ручной установки PyTorch;
- дальше этот же образ можно подключить к **docker-compose**, **Kubernetes**, **Argo CD** (на следующих этапах курса).

---

## 3. Что уже реализовано в проекте

| Компонент | Где находится |
|-----------|----------------|
| GitHub repository | https://github.com/ArtemBotsman/MLopsEyes |
| GitHub Actions workflow | `.github/workflows/ci-cd.yml` |
| Ruff lint | job `lint-and-test`, команда `ruff check .` |
| Pytest tests | `tests/test_cli.py`, job `lint-and-test` |
| Dockerfile | корень репозитория |
| Docker build | job `docker-build` (на PR и push) |
| Docker publish в GHCR | job `docker-publish` (только push в `main`) |
| Образ | `ghcr.io/artembotsman/mlopseyes:latest` |

---

## 4. Как запустить CI/CD — пошаговая инструкция

Ниже — **порядок действий на защите**: сначала локальная проверка (как на CI), затем запуск pipeline на GitHub (CI на PR, CD на main).

### Часть A. Перейти в папку проекта

| Шаг | Команда | Что делает |
|-----|---------|------------|
| 1 | `pwd` | Показывает, в какой папке вы сейчас. Должно быть что-то вроде `.../Open_eyes_classifier MLops`. |
| 2 | `cd "/Users/artem/Work/Open_eyes_classifier MLops"` | Переходит в папку проекта (путь подставьте свой, если отличается). |
| 3 | `pwd` | Проверка: вы в каталоге проекта. |
| 4 | `ls` | Список файлов: должны быть `open_eyes_classifier.py`, `Dockerfile`, `tests/`. |

### Часть B. Локальная проверка (то же, что делает CI)

Выполняйте команды **по очереди**, сверху вниз.

| Шаг | Команда | Что делает | Ожидаемый результат |
|-----|---------|------------|---------------------|
| 1 | `python -m pip install -r requirements-dev.txt` | Ставит pytest и ruff + зависимости из `requirements.txt`. | `Successfully installed` или `Requirement already satisfied`. |
| 2 | `ruff check .` | **Линтер** — проверка стиля и импортов (как job lint-and-test на GitHub). | `All checks passed!` |
| 3 | `pytest -v` | **Тесты** — smoke-тесты CLI (как Pytest в CI). | `2 passed` |
| 4 | `python open_eyes_classifier.py EyesDataset/opened/003035.jpg` | Запуск модели **без Docker** — проверка инференса. | Первая строка: число `0.xxxx` или `1.0000`. |
| 5 | `docker build -t open-eyes-classifier .` | **Сборка образа** (как docker-build в CI). Нужен запущенный Docker Desktop. | `FINISHED`, образ `open-eyes-classifier:latest`. |
| 6 | `docker run --rm -v "$(pwd)/EyesDataset:/app/EyesDataset" open-eyes-classifier EyesDataset/opened/003035.jpg` | **Запуск в контейнере** — тот же инференс, что в проде. | Score и текст «ГЛАЗ ОТКРЫТ/ЗАКРЫТ». |

**Одной строкой (копировать блоком):**

```bash
cd "/Users/artem/Work/Open_eyes_classifier MLops"
python -m pip install -r requirements-dev.txt
ruff check .
pytest -v
python open_eyes_classifier.py EyesDataset/opened/003035.jpg
docker build -t open-eyes-classifier .
docker run --rm -v "$(pwd)/EyesDataset:/app/EyesDataset" open-eyes-classifier EyesDataset/opened/003035.jpg
```

### Часть C. Запустить CI на GitHub (через Pull Request)

CI **запускается автоматически** при создании PR в `main`. Вручную кнопку «Run workflow» настраивать не нужно.

| Шаг | Команда / действие | Что делает |
|-----|-------------------|------------|
| 1 | `git status` | Показывает ветку и незакоммиченные файлы. |
| 2 | `git checkout -b demo-ci-check` | Создаёт новую ветку для демо. |
| 3 | `echo "# CI/CD demo check" >> DEMO.md` | Создаёт маленькое изменение для PR. |
| 4 | `git add DEMO.md` | Добавляет файл в коммит. |
| 5 | `git commit -m "docs: add ci cd demo note"` | Коммит (conventional commits). |
| 6 | `git push -u origin demo-ci-check` | Отправляет ветку на GitHub → **стартует CI**. |
| 7 | Браузер: https://github.com/ArtemBotsman/MLopsEyes | Открыть репозиторий. |
| 8 | Нажать **Compare & pull request** → **Create pull request** | Создать PR в `main`. |
| 9 | Вкладка PR → **Checks** или **Actions** | Смотреть pipeline. |

**Что запустится на CI (PR):**

1. **lint-and-test** — `pip install`, `ruff check .`, `pytest -v`
2. **docker-build** — сборка образа без публикации
3. **docker-publish** — **НЕ запускается** на PR (только на push в main)

Дождитесь **зелёных галочек** у `lint-and-test` и `docker-build`.

### Часть D. Запустить CD на GitHub (публикация образа)

CD **запускается автоматически** при **push в main** или **merge PR в main**.

| Шаг | Команда / действие | Что делает |
|-----|-------------------|------------|
| 1 | На GitHub: **Merge pull request** (или локально `git checkout main` + `git push`) | Код попадает в `main` → стартует полный pipeline. |
| 2 | **Actions** → последний run на ветке `main` | Открыть workflow run. |
| 3 | Проверить job **docker-publish** | Сборка и **push** образа в GHCR. |
| 4 | **Packages** (справа на странице репо) | Образ `ghcr.io/artembotsman/mlopseyes:latest`. |

**Альтернатива без PR** — прямой push в main (у вас уже был):

```bash
git checkout main
git add README.md
git commit -m "docs: update readme with ci cd instructions"
git push origin main
```

После `git push origin main` на GitHub запускаются **все три job**: lint-and-test → docker-build → **docker-publish**.

### Часть E. Схема: что когда запускается

| Событие | lint-and-test | docker-build | docker-publish |
|---------|---------------|--------------|----------------|
| Pull request → main | да | да | **нет** |
| Push → main | да | да | **да** |

### Часть F. Как понять, что CI/CD прошёл успешно

| Где смотреть | Признак успеха |
|--------------|----------------|
| Терминал (локально) | `All checks passed!`, `2 passed`, Docker вывел score |
| GitHub Actions | Зелёная галочка у workflow run |
| Job lint-and-test | Все шаги зелёные |
| Job docker-build | Build completed |
| Job docker-publish | Build and push completed (только main) |
| GHCR Packages | Есть пакет с тегом `latest` |

---

## 5. Что открыть на GitHub

### Шаг 1. Репозиторий

1. Откройте браузер.
2. Перейдите: **https://github.com/ArtemBotsman/MLopsEyes**
3. **Что сказать:** «Вот репозиторий проекта, код и CI/CD лежат здесь».

### Шаг 2. Вкладка Actions

1. Вверху репозитория нажмите **Actions**.
2. Слева выберите workflow **CI/CD**.
3. Справа — список **workflow runs** (запусков).
4. **Что сказать:** «При каждом push и pull request в main запускается этот pipeline».

### Шаг 3. Открыть один успешный run

1. Кликните на последний run с **зелёной галочкой** (например, после коммита README).
2. Увидите три job’а:
   - **lint-and-test**
   - **docker-build**
   - **docker-publish** (только если это был push в `main`, не PR)

### Шаг 4. Логи job

1. Кликните **lint-and-test** → откройте шаги **Ruff** и **Pytest**.
2. Кликните **docker-build** → шаг **Build Docker image**.
3. Если был push в main — **docker-publish** → **Build and push**.
4. **Что сказать:** «Вот автоматическая проверка кода и сборка образа на сервере GitHub».

### Шаг 5. GitHub Packages / GHCR

**Вариант А (удобнее):**

1. На **главной странице** репозитория справа найдите блок **Packages**.
2. Кликните пакет **mlopseyes** (или похожее имя).
3. **Что сказать:** «После merge в main образ публикуется сюда — GitHub Container Registry».

**Вариант Б:**

1. Кликните аватар GitHub → **Your profile** (или **Your organizations**).
2. Вкладка **Packages**.
3. Найдите **mlopseyes**.

### Шаг 6. README

1. Вкладка **Code** → файл **README.md**.
2. **Что сказать:** «В README описаны локальный запуск, Docker и схема CI/CD».

---

## 6. Что говорить на защите (2–3 минуты)

> Добрый день. Я представляю проект **Open Eyes Classifier** — бинарный классификатор изображений глаз: открыт или закрыт. Модель возвращает score от 0 до 1.
>
> **На текущем этапе мы показываем CI/CD-контур** для этого ML-проекта, а не полный MLOps со всеми сервисами.
>
> Код хранится в GitHub, репозиторий **ArtemBotsman/MLopsEyes**. При изменениях автоматически запускается pipeline в GitHub Actions.
>
> **При pull request** в ветку main запускаются **линтер Ruff**, **тесты Pytest** и **сборка Docker-образа** без публикации — это этап CI: мы проверяем, что проект собирается и тесты проходят до слияния.
>
> **После попадания изменений в main** дополнительно запускается job **docker-publish** — **публикация Docker-образа в GitHub Container Registry**. Образ доступен как `ghcr.io/artembotsman/mlopseyes:latest`.
>
> **Этот образ далее можно использовать** для запуска через docker-compose, развёртывания в **Kubernetes/minikube** и доставки через **Argo CD** — это запланировано на следующих этапах.
>
> Локально я могу показать те же шаги: `ruff`, `pytest`, `docker build` и `docker run` с примером изображения из датасета EyesDataset.
>
> **Полный MLOps-контур** с **MLflow**, **drift detection**, **Web UI** и мониторингом (**Prometheus + Grafana**) **будет добавлен на следующих этапах** дисциплины.
>
> Спасибо за внимание, готов ответить на вопросы.

---

## 7. Команды локальной проверки

| Команда | Что делает | Что должно получиться |
|---------|------------|------------------------|
| `pwd` | Показывает текущую папку | Путь к `Open_eyes_classifier MLops` |
| `ls` | Список файлов в папке | Видны `open_eyes_classifier.py`, `Dockerfile`, `tests/`, … |
| `git status` | Статус git | Ветка `main`, чистое дерево или список изменений |
| `python -m pip install -r requirements-dev.txt` | Установка dev-зависимостей | `Successfully installed` или `Requirement already satisfied` |
| `ruff check .` | Проверка стиля кода | `All checks passed!` |
| `pytest -v` | Запуск тестов | `2 passed` |
| `docker build -t open-eyes-classifier .` | Сборка образа | `FINISHED`, тег `open-eyes-classifier:latest` |
| `docker images` | Список образов | Строка `open-eyes-classifier` |
| `docker run --rm -v "$(pwd)/EyesDataset:/app/EyesDataset" open-eyes-classifier EyesDataset/opened/003035.jpg` | Запуск инференса в контейнере | `1.0000` (или другой score) и строка «ГЛАЗ ОТКРЫТ/ЗАКРЫТ» |
| `docker ps` | Запущенные контейнеры | Пусто, если контейнер уже завершился (`--rm`) |
| `docker ps -a` | Все контейнеры | Список; с `--rm` контейнер может не остаться |

**Важно:** перед Docker-командами должен быть **запущен Docker Desktop** (Engine running).

---

## 8. Как запускать Docker

### docker build

Собирает **image** (образ) по инструкции из `Dockerfile`: Python, PyTorch CPU, скрипт, веса модели.

```bash
docker build -t open-eyes-classifier .
```

`-t open-eyes-classifier` — имя (тег) образа.

### docker run

Запускает **container** (контейнер) из образа — изолированную среду, где выполняется одна команда.

```bash
docker run --rm -v "$(pwd)/EyesDataset:/app/EyesDataset" open-eyes-classifier EyesDataset/opened/003035.jpg
```

| Часть команды | Значение |
|---------------|----------|
| `--rm` | После завершения контейнер **удаляется** (не засоряет систему) |
| `-v "$(pwd)/EyesDataset:/app/EyesDataset"` | **Проброс папки**: локальная `EyesDataset` → `/app/EyesDataset` внутри контейнера |
| `open-eyes-classifier` | Имя образа |
| `EyesDataset/opened/003035.jpg` | Аргумент для ENTRYPOINT (путь к картинке **внутри** контейнера) |

**Почему пробрасываем EyesDataset:** в образ **не кладём** весь датасет (он большой и в `.dockerignore`). Для демо монтируем папку с хоста — так можно проверить реальное фото без пересборки образа.

### Термины

- **Image (образ)** — «шаблон» с ОС, Python, кодом и весами (как ISO или снимок).
- **Container (контейнер)** — запущенный экземпляр образа; после `docker run` он стартует, печатает score и завершается.

---

## 9. Как остановить Docker

### Вариант 1. Контейнер в терминале «висит»

Нажмите **Ctrl + C** в том же окне терминала.

### Вариант 2. Контейнер в фоне

```bash
docker ps
docker stop <CONTAINER_ID>
```

Подставьте ID из первого столбца (например `6b71a6f98ff8`).

### Вариант 3. Удалить остановленные контейнеры

```bash
docker ps -a
docker rm <CONTAINER_ID>
```

### Вариант 4. В будущем с docker-compose

```bash
docker compose down
```

(Сейчас в проекте compose нет — для справки на следующих этапах.)

### Вариант 5. Docker Desktop

1. Откройте **Docker Desktop**.
2. Вкладка **Containers**.
3. Нажмите **Stop** (корзина — удалить).

Чтобы **полностью выгрузить** Docker с Mac: меню Docker → **Quit Docker Desktop**.

---

## 10. Как сделать тестовый Pull Request для демонстрации

В терминале, в папке проекта:

```bash
git checkout -b demo-ci-check
echo "# CI/CD demo check" >> DEMO.md
git add DEMO.md
git commit -m "docs: add ci cd demo note"
git push -u origin demo-ci-check
```

На GitHub:

1. Появится жёлтая полоска **Compare & pull request** — нажмите.
2. Base: **main**, Compare: **demo-ci-check**.
3. **Create pull request**.
4. Откройте вкладку **Checks** или **Actions** у PR.
5. Дождитесь зелёных галочек: **lint-and-test**, **docker-build**.
6. **Не обязательно** merge до защиты — достаточно показать CI на PR.

**Что сказать:** «На pull request CI проверяет код и собирает образ, но не публикует в registry — это безопасная проверка до merge».

---

## 11. Как показать CD

1. Сделайте **Merge pull request** в `main` (или `git push` напрямую в `main`).
2. Откройте **Actions** → последний run на ветке `main`.
3. Покажите job **docker-publish** (зелёный).
4. Откройте **Packages** / GHCR → образ **`ghcr.io/artembotsman/mlopseyes:latest`**.
5. **Что сказать:** «CD доставил артефакт — готовый Docker-образ — в registry».

---

## 12. Что делать, если что-то пошло не так

| Проблема | Что означает | Что проверить | Что сказать преподавателю |
|----------|--------------|---------------|---------------------------|
| **ruff упал** | Стиль/импорты не прошли проверку | Лог job lint-and-test, локально `ruff check .` | «Исправим замечания линтера и перезапустим pipeline» |
| **pytest упал** | Тесты не прошли | Лог Pytest, локально `pytest -v` | «Проверим CLI и веса модели, тесты smoke на predict» |
| **docker build упал** | Образ не собрался | Лог docker-build; локально `docker build` | «На CI сборка идёт на Linux; локально проверили отдельно» |
| **package не в GHCR** | publish не сработал | Был ли **push в main**? Job docker-publish зелёный? Права Packages | «Publish только после merge в main, не на PR» |
| **Docker Desktop не запущен** | `docker.sock` недоступен | Запустить Docker Desktop, `docker version` | «Локально Docker был выключен; на GitHub Actions сборка прошла» |
| **docker: command not found** | CLI не установлен | Установить Docker Desktop | «Покажу сборку в GitHub Actions» |
| **Веса слишком большие** | Долгий clone/push | `eye_cnn_best_val_final.pth` ~800 KB — в git; если LFS — проверить | «Веса в репозитории, образ включает их при build» |
| **Не видно Packages** | Пакет приватный или первый push | Профиль → Packages; Settings репо → Actions permissions | «Образ в GHCR, открою вкладку Packages у репозитория» |

---

## 13. Честное состояние проекта

| Уже сделано | Будет сделано позже |
|-------------|---------------------|
| dataset + baseline model | FastAPI + OpenAPI |
| GitHub | DVC |
| GitHub Actions | MLflow |
| lint (ruff) | drift detection |
| tests (pytest) | Prometheus + Grafana |
| Docker build | Web UI |
| Docker publish в GHCR | Kubernetes/minikube |
| README с CI/CD | Argo CD |

---

## 14. Финальный чек-лист перед показом

- [ ] Docker Desktop запущен (если показываете Docker локально)
- [ ] Локально: `ruff check .` — OK
- [ ] Локально: `pytest -v` — 2 passed
- [ ] Локально: `docker build` и `docker run` — score на экране
- [ ] GitHub Actions — последний run **зелёный**
- [ ] Есть **PR** для показа CI (ветка `demo-ci-check`) или готов создать на месте
- [ ] После merge в main — job **docker-publish** зелёный
- [ ] **Package** виден в GHCR: `ghcr.io/artembotsman/mlopseyes:latest`
- [ ] **README** на GitHub обновлён

---

*Документ подготовлен для защиты по дисциплине MLOps. Репозиторий: ArtemBotsman/MLopsEyes.*
