"""SQLite storage for prediction history."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

DEFAULT_DB_PATH = Path("data/predictions.db")


def _connect(db_path: Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Path | None = None) -> None:
    db_path = db_path or DEFAULT_DB_PATH
    with _connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                score REAL NOT NULL,
                label TEXT NOT NULL,
                is_anomaly INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                model_version TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS retrain_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                status TEXT NOT NULL,
                message TEXT NOT NULL,
                mode TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.commit()


def save_prediction(
    filename: str,
    score: float,
    label: str,
    is_anomaly: bool,
    created_at: str,
    model_version: str,
    db_path: Path | None = None,
) -> int:
    db_path = db_path or DEFAULT_DB_PATH
    with _connect(db_path) as conn:
        cursor = conn.execute(
            """
            INSERT INTO predictions (filename, score, label, is_anomaly, created_at, model_version)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (filename, score, label, int(is_anomaly), created_at, model_version),
        )
        conn.commit()
        return int(cursor.lastrowid)


def list_predictions(limit: int = 50, db_path: Path | None = None) -> list[dict[str, Any]]:
    db_path = db_path or DEFAULT_DB_PATH
    with _connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT id, filename, score, label, is_anomaly, created_at, model_version
            FROM predictions
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [
        {
            "id": row["id"],
            "filename": row["filename"],
            "score": row["score"],
            "label": row["label"],
            "is_anomaly": bool(row["is_anomaly"]),
            "created_at": row["created_at"],
            "model_version": row["model_version"],
        }
        for row in rows
    ]


def save_retrain_event(
    status: str,
    message: str,
    mode: str,
    created_at: str,
    db_path: Path | None = None,
) -> int:
    db_path = db_path or DEFAULT_DB_PATH
    with _connect(db_path) as conn:
        cursor = conn.execute(
            """
            INSERT INTO retrain_events (status, message, mode, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (status, message, mode, created_at),
        )
        conn.commit()
        return int(cursor.lastrowid)


def list_retrain_events(limit: int = 20, db_path: Path | None = None) -> list[dict[str, Any]]:
    db_path = db_path or DEFAULT_DB_PATH
    with _connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT id, status, message, mode, created_at
            FROM retrain_events
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [
        {
            "id": row["id"],
            "status": row["status"],
            "message": row["message"],
            "mode": row["mode"],
            "created_at": row["created_at"],
        }
        for row in rows
    ]
