"""HTTP routes for the TreeSense FastAPI app."""

from __future__ import annotations

import asyncio
import json
from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from app.core.config import Settings
from app.core.jobs import JobStore
from app.services.detection import DetectionService

router = APIRouter()


def _get_dependencies(request: Request) -> tuple[Settings, JobStore, DetectionService]:
    try:
        return (
            request.app.state.settings,
            request.app.state.job_store,
            request.app.state.detections,
        )
    except AttributeError as exc:  # pragma: no cover - defensive guard
        raise HTTPException(status_code=500, detail="Application not initialized") from exc


@router.post("/upload")
async def upload_file(request: Request, file: UploadFile = File(...)) -> dict[str, str]:
    settings, job_store, detection_service = _get_dependencies(request)

    if not file.filename:
        raise HTTPException(status_code=400, detail="File name missing")

    job_id = str(uuid4())
    target_path = settings.uploads_dir / f"{job_id}_{file.filename}"
    with open(target_path, "wb") as buffer:
        buffer.write(await file.read())

    job_store.create(job_id)
    asyncio.create_task(detection_service.run(job_id, target_path))

    return {"job_id": job_id}


@router.get("/", response_class=HTMLResponse)
async def landing_page(request: Request):
    templates = request.app.state.templates
    return templates.TemplateResponse("index.html", {"request": request})


@router.get("/detect", response_class=HTMLResponse)
async def detect_ui(request: Request):
    templates = request.app.state.templates
    return templates.TemplateResponse("urban.html", {"request": request})


@router.get("/urbananalysis", response_class=HTMLResponse)
async def urban_analysis(request: Request):
    templates = request.app.state.templates
    return templates.TemplateResponse("urbananalysis.html", {"request": request})


@router.get("/progress/{job_id}")
async def progress_stream(request: Request, job_id: str):
    _, job_store, _ = _get_dependencies(request)

    async def event_stream():
        last_log_len = 0
        while True:
            snapshot = job_store.snapshot(job_id)
            if not snapshot:
                yield f"data: {json.dumps({'error': 'Invalid job'})}\n\n"
                break

            logs = snapshot["logs"][last_log_len:]
            last_log_len = len(snapshot["logs"])

            for log in logs:
                payload = {
                    "status": snapshot["status"],
                    "progress": round(snapshot["progress"], 3),
                    "log": log,
                }
                yield f"data: {json.dumps(payload)}\n\n"

            if snapshot["status"] in {"done", "error"}:
                yield f"data: {json.dumps({'status': snapshot['status'], 'done': True})}\n\n"
                break

            await asyncio.sleep(1)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.get("/result/{job_id}")
async def result(request: Request, job_id: str):
    _, job_store, _ = _get_dependencies(request)

    timeout = 600
    waited = 0
    while waited < timeout:
        snapshot = job_store.snapshot(job_id)
        if not snapshot:
            return JSONResponse({"error": "Invalid job id"}, status_code=404)

        if snapshot["status"] in {"done", "error"}:
            if snapshot["status"] == "error":
                return JSONResponse(
                    {"error": "Detection failed", "logs": snapshot["logs"]},
                    status_code=500,
                )
            if snapshot["result"]:
                return JSONResponse(snapshot["result"])
            return JSONResponse(
                {"error": "No result generated", "logs": snapshot["logs"]},
                status_code=500,
            )

        await asyncio.sleep(1)
        waited += 1

    final_snapshot = job_store.snapshot(job_id)
    logs = final_snapshot["logs"] if final_snapshot else []
    return JSONResponse(
        {"error": "Detection timeout", "logs": logs},
        status_code=504,
    )
