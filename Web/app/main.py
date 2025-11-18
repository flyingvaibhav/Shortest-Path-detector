"""FastAPI application wiring for TreeSense."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.api.routes import router as api_router
from app.core.config import get_settings
from app.core.jobs import JobStore
from app.services.detection import DetectionService


def create_app() -> FastAPI:
    settings = get_settings()
    job_store = JobStore()
    detection_service = DetectionService(settings, job_store)

    app = FastAPI(title=settings.app_name)

    app.mount("/static", StaticFiles(directory=settings.static_dir), name="static")
    app.mount("/results", StaticFiles(directory=settings.results_dir), name="results")
    templates = Jinja2Templates(directory=settings.templates_dir)

    app.state.settings = settings
    app.state.job_store = job_store
    app.state.detections = detection_service
    app.state.templates = templates

    app.include_router(api_router)

    return app


app = create_app()
