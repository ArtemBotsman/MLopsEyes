"""HTML/JSON drift report generation."""

from __future__ import annotations

import json
from html import escape
from pathlib import Path
from typing import Any


def _render_features_table(features: list[dict[str, Any]]) -> str:
    if not features:
        return "<p>No feature statistics available.</p>"
    rows = []
    for item in features:
        rows.append(
            "<tr>"
            f"<td>{escape(str(item.get('feature', '')))}</td>"
            f"<td>{escape(str(item.get('statistic', '')))}</td>"
            f"<td>{escape(str(item.get('p_value', '')))}</td>"
            f"<td>{escape(str(item.get('drift_detected', '')))}</td>"
            "</tr>"
        )
    return (
        "<table border='1' cellpadding='6' cellspacing='0'>"
        "<tr><th>Feature</th><th>KS statistic</th><th>p-value</th><th>Drift</th></tr>"
        + "".join(rows)
        + "</table>"
    )


def render_drift_html(report: dict[str, Any]) -> str:
    data = report.get("data_drift", {})
    target = report.get("target_drift", {})
    concept = report.get("concept_drift", {})
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Drift Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; }}
    h1, h2 {{ color: #1f2937; }}
    .status-ok {{ color: #15803d; font-weight: bold; }}
    .status-warning {{ color: #b45309; font-weight: bold; }}
    .box {{ border: 1px solid #d1d5db; border-radius: 8px; padding: 16px; margin-bottom: 16px; }}
  </style>
</head>
<body>
  <h1>Drift Monitoring Report</h1>
  <p><strong>Generated at:</strong> {escape(str(report.get('generated_at', '')))}</p>
  <p><strong>Overall status:</strong>
    <span class="{'status-warning' if 'WARNING' in str(report.get('overall_status', '')) else 'status-ok'}">
      {escape(str(report.get('overall_status', report.get('status', 'unknown'))))}
    </span>
  </p>

  <div class="box">
    <h2>Data drift</h2>
    <p><strong>Detected:</strong> {escape(str(data.get('data_drift_detected', False)))}</p>
    {_render_features_table(data.get('features', []))}
  </div>

  <div class="box">
    <h2>Target drift</h2>
    <p><strong>Reference opened ratio:</strong> {escape(str(target.get('reference_opened_ratio')))}</p>
    <p><strong>Current opened ratio:</strong> {escape(str(target.get('current_opened_ratio')))}</p>
    <p><strong>Delta:</strong> {escape(str(target.get('delta')))}</p>
    <p><strong>Detected:</strong> {escape(str(target.get('target_drift_detected', False)))}</p>
  </div>

  <div class="box">
    <h2>Concept drift</h2>
    <p><strong>Status:</strong> {escape(str(concept.get('concept_drift_status', 'unknown')))}</p>
    <p><strong>Reason:</strong> {escape(str(concept.get('reason', '-')))}</p>
    <p><strong>Accuracy:</strong> {escape(str(concept.get('accuracy', '-')))}</p>
    <p><strong>Precision:</strong> {escape(str(concept.get('precision', '-')))}</p>
    <p><strong>Recall:</strong> {escape(str(concept.get('recall', '-')))}</p>
    <p><strong>F1:</strong> {escape(str(concept.get('f1', '-')))}</p>
    <p><strong>Detected:</strong> {escape(str(concept.get('concept_drift_detected', False)))}</p>
  </div>
</body>
</html>
"""


def save_drift_report(report: dict[str, Any], output_dir: str | Path) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "latest_drift_report.json"
    html_path = output / "latest_drift_report.html"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    html_path.write_text(render_drift_html(report), encoding="utf-8")
    return json_path
