"""Streamlit frontend skeleton for {{cookiecutter.project_name}}."""

from __future__ import annotations

import os

import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:{{cookiecutter.backend_port}}").rstrip("/")

st.set_page_config(page_title="{{cookiecutter.project_name}}", layout="wide")
st.title("{{cookiecutter.project_name}}")
st.caption("MLOps UI skeleton — connect to the FastAPI backend for inference.")
st.write(f"**API URL:** `{API_URL}`")
