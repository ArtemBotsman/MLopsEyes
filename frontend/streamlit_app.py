"""Streamlit Web UI for Open Eyes Classifier MLOps."""

from __future__ import annotations

import os

import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
PROMETHEUS_URL = os.getenv("PROMETHEUS_URL", "http://localhost:9090").rstrip("/")
GRAFANA_URL = os.getenv("GRAFANA_URL", "http://localhost:3000").rstrip("/")
MLFLOW_URL = os.getenv("MLFLOW_URL", "").rstrip("/")

PAGES = [
    "Inference",
    "Predictions",
    "Drift",
    "Experiments",
    "Retraining",
    "System status",
]


def api_get(path: str, timeout: int = 30) -> tuple[dict | list | None, str | None]:
    try:
        response = requests.get(f"{API_URL}{path}", timeout=timeout)
        response.raise_for_status()
        return response.json(), None
    except requests.RequestException as exc:
        return None, str(exc)


def api_post(
    path: str,
    files: dict | None = None,
    json_body: dict | None = None,
    timeout: int = 120,
) -> tuple[dict | None, str | None]:
    try:
        response = requests.post(
            f"{API_URL}{path}",
            files=files,
            json=json_body,
            timeout=timeout,
        )
        response.raise_for_status()
        return response.json(), None
    except requests.RequestException as exc:
        return None, str(exc)


def page_inference() -> None:
    st.header("Inference")
    st.caption("Загрузите изображение глаза для классификации opened / closed.")

    uploaded = st.file_uploader("Изображение (jpg/png)", type=["jpg", "jpeg", "png"])
    if uploaded is None:
        return

    st.image(uploaded, caption=uploaded.name, use_container_width=True)

    if st.button("Получить предсказание", type="primary"):
        with st.spinner("Отправка в backend..."):
            data, error = api_post(
                "/predict",
                files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type or "image/jpeg")},
            )
        if error:
            st.error(f"Ошибка backend: {error}")
            return
        if not data:
            return

        label_text = "Глаз открыт" if data["label"] == "opened" else "Глаз закрыт"
        st.success(label_text)
        col1, col2, col3 = st.columns(3)
        col1.metric("Score", f"{data['score']:.4f}")
        col2.metric("Label", data["label"])
        col3.metric("Anomaly", "да" if data["is_anomaly"] else "нет")
        st.write(f"**Model version:** {data['model_version']}")
        st.write(f"**Created at:** {data['created_at']}")

        if data["is_anomaly"]:
            st.warning(
                "Обнаружен флаг аномалии: модель не уверена или изображение имеет нетипичную яркость."
            )


def page_predictions() -> None:
    st.header("Predictions")
    data, error = api_get("/predictions?limit=100")
    if error:
        st.error(f"Не удалось загрузить предсказания: {error}")
        return
    if not data:
        return

    predictions = data.get("predictions", [])
    if not predictions:
        st.info("Предсказаний пока нет. Выполните инференс на странице Inference.")
        return

    df = pd.DataFrame(predictions)
    filter_option = st.selectbox(
        "Фильтр",
        ["все", "только anomaly", "только opened", "только closed"],
    )
    if filter_option == "только anomaly":
        df = df[df["is_anomaly"]]
    elif filter_option == "только opened":
        df = df[df["label"] == "opened"]
    elif filter_option == "только closed":
        df = df[df["label"] == "closed"]

    total = len(predictions)
    opened = sum(1 for p in predictions if p["label"] == "opened")
    closed = sum(1 for p in predictions if p["label"] == "closed")
    anomalies = sum(1 for p in predictions if p["is_anomaly"])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Всего предсказаний", total)
    c2.metric("Opened", opened)
    c3.metric("Closed", closed)
    c4.metric("Anomalies", anomalies)

    display_df = df.copy()
    if "is_anomaly" in display_df.columns:
        display_df["is_anomaly"] = display_df["is_anomaly"].map({True: "⚠️ да", False: "нет"})
    st.dataframe(display_df, use_container_width=True)


def page_drift() -> None:
    st.header("Drift")
    st.caption("Мониторинг data / target / concept drift.")

    if st.button("Запустить drift report", type="primary"):
        with st.spinner("Генерация отчёта..."):
            result, error = api_post("/drift/run", timeout=300)
        if error:
            st.error(f"Ошибка запуска drift report: {error}")
        elif result:
            st.success(f"Drift report: {result.get('status', 'ok')}")
            if result.get("message"):
                st.info(result["message"])

    data, error = api_get("/drift/latest")
    if error:
        st.error(f"Не удалось получить drift report: {error}")
        return
    if not data:
        return

    if data.get("status") == "not_available":
        st.info("Отчёт о дрейфе ещё не был сгенерирован.")
        return

    report = data.get("report") or {}
    st.subheader("Последний отчёт")
    st.write(f"**Status:** {report.get('status', data.get('status'))}")
    st.write(f"**Generated at:** {report.get('generated_at', '—')}")

    data_drift = report.get("data_drift", {})
    target_drift = report.get("target_drift", {})
    concept_drift = report.get("concept_drift", {})

    c1, c2, c3 = st.columns(3)
    c1.metric("Data drift", "detected" if data_drift.get("data_drift_detected") else "ok")
    c2.metric("Target drift", "detected" if target_drift.get("target_drift_detected") else "ok")
    c3.metric("Concept drift", concept_drift.get("concept_drift_status", "—"))

    any_drift = (
        data_drift.get("data_drift_detected")
        or target_drift.get("target_drift_detected")
        or concept_drift.get("concept_drift_detected")
        or report.get("status") == "warning"
    )
    if any_drift:
        st.error("⚠️ Обнаружен drift! Проверьте отчёт и входящие данные.")

    features = data_drift.get("features", [])
    if features:
        st.subheader("Data drift features")
        st.dataframe(pd.DataFrame(features), use_container_width=True)

    st.info("HTML report: `reports/drift/latest_drift_report.html`")


def page_experiments() -> None:
    st.header("Experiments")
    st.write(
        "MLflow используется для трекинга экспериментов и регистрации обученных моделей "
        "в Model Registry (`open-eyes-cnn`)."
    )

    if MLFLOW_URL:
        st.link_button("Открыть MLflow UI", MLFLOW_URL)
    else:
        st.warning("MLflow URL is not configured yet.")

    data, error = api_get("/experiments")
    if error:
        st.warning(f"Не удалось получить эксперименты: {error}")
        return
    if not data:
        return

    if data.get("status") == "not_available":
        st.warning(data.get("message", "MLflow tracking server is not available"))
        return

    experiments = data.get("experiments") or []
    if experiments:
        st.subheader("Эксперименты")
        st.dataframe(pd.DataFrame(experiments), use_container_width=True)
    else:
        st.info("Эксперименты пока не найдены. Запустите обучение с MLflow tracking.")

    models_data, models_error = api_get("/models")
    if not models_error and models_data and models_data.get("status") == "ok":
        models = models_data.get("models") or []
        if models:
            st.subheader("Registered models")
            st.dataframe(pd.DataFrame(models), use_container_width=True)


def page_retraining() -> None:
    st.header("Retraining")
    st.caption(
        "Быстрое переобучение через отдельный скрипт `scripts/retrain.py` "
        "и конфиг `configs/retrain.yaml` (1–3 эпохи)."
    )

    epochs = st.selectbox("Эпохи", options=[1, 2, 3], index=1)
    st.code(f"python scripts/retrain.py --epochs {epochs}", language="bash")

    if st.button("Запустить переобучение", type="primary"):
        with st.spinner("Отправка запроса в backend..."):
            result, error = api_post("/retrain", json_body={"epochs": epochs})
        if error:
            st.error(f"Ошибка: {error}")
        elif result:
            if result.get("status") == "not_available":
                st.warning(result.get("message", "Скрипт недоступен в контейнере"))
                st.info("Запустите команду выше локально из корня репозитория.")
            else:
                st.success(result.get("message", "Запрос принят"))
            st.json(result)

    st.subheader("История запросов")
    status_data, error = api_get("/retrain/status")
    if error:
        st.warning(f"Не удалось загрузить статус: {error}")
        return
    if status_data and status_data.get("events"):
        st.dataframe(pd.DataFrame(status_data["events"]), use_container_width=True)
    else:
        st.info("Запросов на переобучение пока нет.")


def page_system_status() -> None:
    st.header("System status")

    health, health_error = api_get("/health")
    backend_ok = health_error is None and health is not None

    st.metric("Backend", "доступен" if backend_ok else "недоступен")
    if health:
        st.write(f"**Health status:** {health.get('status', 'unknown')}")
    elif health_error:
        st.error(health_error)

    st.subheader("Ссылки")
    st.markdown(f"- [OpenAPI docs]({API_URL}/docs)")
    st.markdown(f"- [Prometheus]({PROMETHEUS_URL})")
    st.markdown(f"- [Grafana]({GRAFANA_URL})")
    if MLFLOW_URL:
        st.markdown(f"- [MLflow]({MLFLOW_URL})")

    if backend_ok:
        try:
            response = requests.get(f"{API_URL}/metrics", timeout=10)
            if response.ok:
                st.caption("Prometheus /metrics endpoint доступен")
                if "mlops_retrain_requests_total" in response.text:
                    st.caption("Метрика mlops_retrain_requests_total зарегистрирована")
        except requests.RequestException as exc:
            st.warning(f"Не удалось прочитать /metrics: {exc}")


def main() -> None:
    st.set_page_config(page_title="Open Eyes Classifier", page_icon="👁️", layout="wide")
    st.sidebar.title("Open Eyes Classifier")
    st.sidebar.caption(f"API: {API_URL}")
    page = st.sidebar.radio("Навигация", PAGES)

    if page == "Inference":
        page_inference()
    elif page == "Predictions":
        page_predictions()
    elif page == "Drift":
        page_drift()
    elif page == "Experiments":
        page_experiments()
    elif page == "Retraining":
        page_retraining()
    elif page == "System status":
        page_system_status()


if __name__ == "__main__":
    main()
