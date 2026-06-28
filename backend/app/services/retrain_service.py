"""Background launcher for scripts/retrain.py."""

from __future__ import annotations

import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

from backend.app.storage import save_retrain_event

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RETRAIN_SCRIPT = PROJECT_ROOT / "scripts" / "retrain.py"
MIN_EPOCHS = 1
MAX_EPOCHS = 3


def _finish_event(status: str, message: str, mode: str) -> None:
    save_retrain_event(
        status=status,
        message=message,
        mode=mode,
        created_at=datetime.now(timezone.utc).isoformat(),
    )


def _run_retrain_subprocess(epochs: int) -> None:
    mode = "quick"
    cmd = [sys.executable, str(RETRAIN_SCRIPT), "--epochs", str(epochs)]
    try:
        result = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            _finish_event(
                status="completed",
                message=f"Quick retrain finished ({epochs} epoch(s))",
                mode=mode,
            )
            return
        stderr = (result.stderr or result.stdout or "unknown error").strip()
        _finish_event(
            status="failed",
            message=f"Quick retrain failed: {stderr[:500]}",
            mode=mode,
        )
    except Exception as exc:  # noqa: BLE001
        _finish_event(
            status="failed",
            message=f"Quick retrain failed: {exc}",
            mode=mode,
        )


def schedule_quick_retrain(epochs: int) -> tuple[str, str]:
    if not MIN_EPOCHS <= epochs <= MAX_EPOCHS:
        raise ValueError(f"epochs must be between {MIN_EPOCHS} and {MAX_EPOCHS}")
    if not RETRAIN_SCRIPT.is_file():
        raise FileNotFoundError(f"Retrain script not found: {RETRAIN_SCRIPT}")

    message = f"Quick retrain scheduled for {epochs} epoch(s)"
    save_retrain_event(
        status="started",
        message=message,
        mode="quick",
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    thread = threading.Thread(
        target=_run_retrain_subprocess,
        args=(epochs,),
        daemon=True,
    )
    thread.start()
    return "started", message
