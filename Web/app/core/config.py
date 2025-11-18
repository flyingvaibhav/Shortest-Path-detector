"""Application configuration helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


@dataclass
class Settings:
    """Strongly-typed configuration for the FastAPI app."""

    app_name: str
    base_dir: Path
    static_dir: Path
    templates_dir: Path
    uploads_dir: Path
    results_dir: Path
    models_dir: Path
    model_filename: str
    worker_pool_size: int
    detection_confidence: float
    detection_img_size: int

    @property
    def model_path(self) -> Path:
        return self.models_dir / self.model_filename


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load settings from env vars with sensible defaults."""

    base_dir = Path(os.getenv("APP_BASE_DIR", Path(__file__).resolve().parents[2]))
    static_dir = Path(os.getenv("APP_STATIC_DIR", base_dir / "static"))
    templates_dir = Path(os.getenv("APP_TEMPLATES_DIR", base_dir / "templates"))
    uploads_dir = Path(os.getenv("APP_UPLOAD_DIR", base_dir / "uploads"))
    results_dir = Path(os.getenv("APP_RESULTS_DIR", base_dir / "result"))
    models_dir = Path(os.getenv("APP_MODELS_DIR", base_dir / "models"))

    uploads_dir.mkdir(parents=True, exist_ok=True)
    static_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    return Settings(
        app_name=os.getenv("APP_NAME", "TreeSense"),
        base_dir=base_dir,
        static_dir=static_dir,
        templates_dir=templates_dir,
        uploads_dir=uploads_dir,
        results_dir=results_dir,
        models_dir=models_dir,
        model_filename=os.getenv("YOLO_MODEL_FILENAME", "best.pt"),
        worker_pool_size=int(os.getenv("WORKER_POOL_SIZE", 2)),
        detection_confidence=float(os.getenv("DETECTION_CONFIDENCE", 0.25)),
        detection_img_size=int(os.getenv("DETECTION_IMG_SIZE", 640)),
    )
