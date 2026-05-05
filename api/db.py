"""Small SQLite job store."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import sqlite3
from typing import Any, Iterator


class JobStore:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    input_path TEXT NOT NULL,
                    result_json_path TEXT,
                    result_txt_path TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    error_code TEXT,
                    error_message TEXT,
                    include_ivr INTEGER NOT NULL DEFAULT 0,
                    mask_pii INTEGER NOT NULL DEFAULT 1
                )
                """
            )
            self._ensure_column(conn, "include_ivr", "INTEGER NOT NULL DEFAULT 0")
            self._ensure_column(conn, "mask_pii", "INTEGER NOT NULL DEFAULT 1")

    def _ensure_column(self, conn: sqlite3.Connection, name: str, definition: str) -> None:
        columns = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(jobs)").fetchall()
        }
        if name not in columns:
            conn.execute(f"ALTER TABLE jobs ADD COLUMN {name} {definition}")

    def create_job(
        self,
        job_id: str,
        filename: str,
        input_path: Path,
        *,
        include_ivr: bool,
        mask_pii: bool,
        status: str = "queued",
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO jobs (job_id, status, filename, input_path, include_ivr, mask_pii)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (job_id, status, filename, str(input_path), int(include_ivr), int(mask_pii)),
            )

    def set_status(
        self,
        job_id: str,
        status: str,
        *,
        result_json_path: Path | None = None,
        result_txt_path: Path | None = None,
        error_code: str | None = None,
        error_message: str | None = None,
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE jobs
                SET status = ?,
                    result_json_path = COALESCE(?, result_json_path),
                    result_txt_path = COALESCE(?, result_txt_path),
                    error_code = ?,
                    error_message = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE job_id = ?
                """,
                (
                    status,
                    str(result_json_path) if result_json_path else None,
                    str(result_txt_path) if result_txt_path else None,
                    error_code,
                    error_message,
                    job_id,
                ),
            )

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        return dict(row) if row else None

    def count_active_jobs(self) -> int:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM jobs WHERE status IN ('queued', 'processing')"
            ).fetchone()
        return int(row["n"])

    def mark_uploaded_as_queued(self, job_ids: list[str]) -> list[str]:
        started: list[str] = []
        with self.connect() as conn:
            for job_id in job_ids:
                row = conn.execute(
                    "SELECT status FROM jobs WHERE job_id = ?",
                    (job_id,),
                ).fetchone()
                if row is None:
                    continue
                if row["status"] != "uploaded":
                    continue
                conn.execute(
                    """
                    UPDATE jobs
                    SET status = 'queued', updated_at = CURRENT_TIMESTAMP
                    WHERE job_id = ?
                    """,
                    (job_id,),
                )
                started.append(job_id)
        return started

    def queue_position(self, job_id: str) -> int | None:
        job = self.get_job(job_id)
        if job is None or job["status"] not in {"queued", "processing"}:
            return None
        if job["status"] == "processing":
            return 0

        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT COUNT(*) AS n
                FROM jobs
                WHERE status = 'processing'
                   OR (
                        status = 'queued'
                        AND (
                            created_at < ?
                            OR (created_at = ? AND job_id <= ?)
                        )
                   )
                """,
                (job["created_at"], job["created_at"], job_id),
            ).fetchone()
        return int(row["n"])

    def queued_jobs(self) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM jobs
                WHERE status = 'queued'
                ORDER BY created_at ASC
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def terminal_jobs(self) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM jobs
                WHERE status IN ('done', 'failed')
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def reset_processing_to_queued(self) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE jobs
                SET status = 'queued', updated_at = CURRENT_TIMESTAMP
                WHERE status = 'processing'
                """
            )
