"""FastAPI backend skeleton for {{cookiecutter.project_name}}."""

from fastapi import FastAPI

app = FastAPI(
    title="{{cookiecutter.project_name}} API",
    description="MLOps backend skeleton by {{cookiecutter.author_name}}",
    version="0.1.0",
)


@app.get("/health")
def health() -> dict[str, str]:
    """Smoke health endpoint."""
    return {
        "status": "ok",
        "project": "{{cookiecutter.project_slug}}",
        "model": "{{cookiecutter.model_name}}",
    }
