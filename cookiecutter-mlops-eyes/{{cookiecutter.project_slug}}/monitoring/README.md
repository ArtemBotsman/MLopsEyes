# Monitoring

Placeholder for Prometheus and Grafana configuration.

Suggested files:

- `prometheus.yml` — scrape backend metrics on port {{cookiecutter.backend_port}}
- `grafana/provisioning/datasources/prometheus.yml`
- `grafana/dashboards/` — project dashboards

Local ports (docker compose defaults):

- Prometheus: {{cookiecutter.prometheus_port}}
- Grafana: {{cookiecutter.grafana_port}}
