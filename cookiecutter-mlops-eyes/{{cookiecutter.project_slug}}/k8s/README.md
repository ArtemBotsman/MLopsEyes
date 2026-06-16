# Kubernetes manifests

Placeholder for Kubernetes deployments and services.

Suggested next steps:

1. Add `namespace.yaml` for `{{cookiecutter.project_slug}}`.
2. Add `backend-deployment.yaml` and `backend-service.yaml` (port {{cookiecutter.backend_port}}).
3. Add `frontend-deployment.yaml` and `frontend-service.yaml` (port {{cookiecutter.frontend_port}}).
4. Add monitoring stack manifests (Prometheus, Grafana, MLflow).

Apply example:

```bash
kubectl apply -f k8s/
```
