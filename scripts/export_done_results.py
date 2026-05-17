"""Export completed TXT results from API storage using original filenames."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sqlite3
from zipfile import ZIP_DEFLATED, ZipFile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--storage-dir", default="storage", help="API storage directory")
    parser.add_argument("--output-dir", default="out/exported_txt", help="TXT export directory")
    parser.add_argument("--zip-path", default="out/exported_txt.zip", help="Optional ZIP output path")
    parser.add_argument("--created-at-prefix", help="Export jobs whose created_at starts with this value")
    parser.add_argument("--no-zip", action="store_true", help="Do not create ZIP archive")
    return parser.parse_args()


def txt_name(filename: str, used_names: set[str]) -> str:
    base = Path(filename).stem or "transcript"
    candidate = f"{base}.txt"
    index = 2
    while candidate.lower() in used_names:
        candidate = f"{base} ({index}).txt"
        index += 1
    used_names.add(candidate.lower())
    return candidate


def load_jobs(db_path: Path, created_at_prefix: str | None) -> list[sqlite3.Row]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        sql = """
        SELECT job_id, filename, status, result_txt_path, error_code, error_message
        FROM jobs
        """
        params: tuple[str, ...] = ()
        if created_at_prefix:
            sql += " WHERE created_at LIKE ?"
            params = (f"{created_at_prefix}%",)
        sql += " ORDER BY created_at ASC, job_id ASC"
        return conn.execute(sql, params).fetchall()
    finally:
        conn.close()


def main() -> None:
    args = parse_args()
    storage_dir = Path(args.storage_dir)
    db_path = storage_dir / "jobs.sqlite3"
    output_dir = Path(args.output_dir)
    zip_path = Path(args.zip_path)

    if not db_path.exists():
        raise SystemExit(f"DB not found: {db_path}")

    jobs = load_jobs(db_path, args.created_at_prefix)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_zip:
        zip_path.parent.mkdir(parents=True, exist_ok=True)

    used_names: set[str] = set()
    exported: list[Path] = []
    done = 0
    failed = 0
    missing = 0

    for job in jobs:
        if job["status"] == "failed":
            failed += 1
        if job["status"] != "done":
            continue

        done += 1
        source = Path(job["result_txt_path"] or "")
        if not source.exists():
            missing += 1
            continue

        target = output_dir / txt_name(job["filename"], used_names)
        shutil.copy2(source, target)
        exported.append(target)

    if not args.no_zip:
        with ZipFile(zip_path, "w", compression=ZIP_DEFLATED) as archive:
            for path in exported:
                archive.write(path, path.name)

    print(f"done_jobs={done}")
    print(f"failed_jobs={failed}")
    print(f"missing_txt={missing}")
    print(f"exported_txt={len(exported)}")
    print(f"output_dir={output_dir}")
    if not args.no_zip:
        print(f"zip_path={zip_path}")


if __name__ == "__main__":
    main()
