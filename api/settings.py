"""Runtime settings for the API service."""

from dataclasses import dataclass
import os
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


def _int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value else default


def _float_env(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value else default


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    storage_dir: Path = Path(os.getenv("TRANSCRIBE_STORAGE_DIR", "storage"))
    device: str | None = os.getenv("TRANSCRIBE_DEVICE", "cuda")
    model_name: str = os.getenv("TRANSCRIBE_MODEL", "v3_e2e_rnnt")

    max_upload_mb: int = _int_env("TRANSCRIBE_MAX_UPLOAD_MB", 500)
    min_free_disk_mb: int = _int_env("TRANSCRIBE_MIN_FREE_DISK_MB", 2048)
    max_queued_jobs: int = _int_env("TRANSCRIBE_MAX_QUEUED_JOBS", 100)

    batch_size: int = _int_env("TRANSCRIBE_BATCH_SIZE", 8)
    pause_threshold: float = _float_env("TRANSCRIBE_PAUSE_THRESHOLD", 2.0)
    strict_limit_duration: float = _float_env("TRANSCRIBE_STRICT_LIMIT_DURATION", 30.0)
    hold_threshold: float = _float_env("TRANSCRIBE_HOLD_THRESHOLD", 15.0)
    operator_channel: int = _int_env("TRANSCRIBE_OPERATOR_CHANNEL", 1)
    mask_pii: bool = _bool_env("TRANSCRIBE_MASK_PII", True)
    spacy_model: str = os.getenv("TRANSCRIBE_SPACY_MODEL", "ru_core_news_lg")

    @property
    def uploads_dir(self) -> Path:
        return self.storage_dir / "uploads"

    @property
    def results_dir(self) -> Path:
        return self.storage_dir / "results"

    @property
    def db_path(self) -> Path:
        return self.storage_dir / "jobs.sqlite3"

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def min_free_disk_bytes(self) -> int:
        return self.min_free_disk_mb * 1024 * 1024


settings = Settings()
