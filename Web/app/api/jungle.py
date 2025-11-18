"""Routes dedicated to Jungle Mode's path planning experience."""

from __future__ import annotations

import uuid
from pathlib import Path
import time

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, Request, UploadFile, WebSocket
from fastapi.responses import HTMLResponse

from app.services.jungle_pipeline import (
    JungleJobStore,
    base_image_response,
    current_image_response,
    run_pipeline,
    stream_logs,
    stream_video_response,
    summary_response,
    video_response,
)
from app.services.jungle_pipeline import load_image, ndarray_to_png_bytes

router = APIRouter(prefix="/jungle", tags=["jungle"])


def _get_dependencies(request: Request) -> tuple[JungleJobStore, Path]:
    try:
        store = request.app.state.jungle_store
        settings = request.app.state.settings
        return store, settings.uploads_dir / "jungle"
    except AttributeError as exc:  # pragma: no cover - defensive guard
        raise HTTPException(status_code=500, detail="Jungle store missing") from exc


@router.get("/", response_class=HTMLResponse)
async def jungle_home(request: Request):
    templates = request.app.state.templates
    return templates.TemplateResponse("jungle/index.html", {"request": request})


@router.post("/upload", response_class=HTMLResponse)
async def jungle_upload(request: Request, image: UploadFile = File(...)):
    templates = request.app.state.templates
    store, uploads_dir = _get_dependencies(request)

    if not image.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    suffix = Path(image.filename).suffix.lower()
    if suffix not in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
        raise HTTPException(status_code=400, detail="Unsupported format")

    uploads_dir.mkdir(parents=True, exist_ok=True)
    dst = uploads_dir / f"{uuid.uuid4().hex}{suffix}"
    dst.write_bytes(await image.read())

    job = store.create(dst)
    try:
        rgb = load_image(dst, max_dim=900)
        job.rgb = rgb
        job.original_rgb = rgb.copy()
        job.current_frame = rgb.copy()
        base_png = ndarray_to_png_bytes(rgb)
        base_dir = job.output_root / "image"
        base_dir.mkdir(parents=True, exist_ok=True)
        (base_dir / "base.png").write_bytes(base_png)
    except Exception as exc:  # pragma: no cover - surface to user
        job.error = str(exc)
        return HTMLResponse(f"Image load failed: {exc}", status_code=500)

    return templates.TemplateResponse(
        "jungle/select_point.html",
        {"request": request, "jobId": job.job_id, "ts": int(time.time() * 1000)},
    )


@router.get("/image/{job_id}/base.png")
async def jungle_base(job_id: str, request: Request):
    store, _ = _get_dependencies(request)
    job = store.get(job_id)
    return base_image_response(job)


@router.get("/image/{job_id}/current.png")
async def jungle_current(job_id: str, request: Request):
    store, _ = _get_dependencies(request)
    job = store.get(job_id)
    return current_image_response(job)


@router.post("/start", response_class=HTMLResponse)
async def jungle_start(
    request: Request,
    background_tasks: BackgroundTasks,
    jobId: str = Form(...),
    startY: int = Form(...),
    startX: int = Form(...),
    goalY: int = Form(...),
    goalX: int = Form(...),
    expansionsPerFrame: int = Form(25),
):
    templates = request.app.state.templates
    store, _ = _get_dependencies(request)
    job = store.get(jobId)

    if job.rgb is None:
        raise HTTPException(status_code=400, detail="Image not ready")

    h, w = job.rgb.shape[:2]
    sY = max(0, min(int(startY), h - 1))
    sX = max(0, min(int(startX), w - 1))
    gY = max(0, min(int(goalY), h - 1))
    gX = max(0, min(int(goalX), w - 1))
    job.start = (sY, sX)
    job.goal = (gY, gX)
    job.log(f"Start set to {job.start}")
    job.log(f"Goal set to {job.goal}")
    job.log(f"Expansions per frame: {expansionsPerFrame}")

    background_tasks.add_task(run_pipeline, job, expansionsPerFrame, 12)
    return templates.TemplateResponse(
        "jungle/run.html",
        {"request": request, "jobId": jobId, "ts": int(time.time() * 1000)},
    )


@router.get("/status/{job_id}")
async def jungle_status(job_id: str, request: Request):
    store, _ = _get_dependencies(request)
    job = store.get(job_id)
    summary_available = job.summary_image_path and job.summary_image_path.exists()
    video_available = job.final_video_path and job.final_video_path.exists()
    return {
        "done": job.done,
        "error": job.error,
        "summaryPath": f"/jungle/summary/{job_id}.png" if summary_available else None,
        "videoPath": f"/jungle/stream/video/{job_id}" if video_available else None,
    }


@router.get("/summary/{job_id}.png")
async def jungle_summary(job_id: str, request: Request):
    store, _ = _get_dependencies(request)
    job = store.get(job_id)
    return summary_response(job)


@router.get("/video/{job_id}.mp4")
async def jungle_video(job_id: str, request: Request):
    store, _ = _get_dependencies(request)
    job = store.get(job_id)
    return video_response(job)


@router.get("/stream/video/{job_id}")
async def jungle_video_stream(job_id: str, request: Request):
    store, _ = _get_dependencies(request)
    job = store.get(job_id)
    return stream_video_response(job)


@router.websocket("/ws/{job_id}")
async def jungle_ws(ws: WebSocket, job_id: str):
    try:
        store = ws.app.state.jungle_store  # type: ignore[attr-defined]
    except AttributeError:  # pragma: no cover - defensive guard
        await ws.accept()
        await ws.send_text("{\"type\": \"log\", \"msg\": \"Server misconfigured\"}")
        await ws.close()
        return

    try:
        job = store.get(job_id)
    except KeyError:
        await ws.accept()
        await ws.send_text("{\"type\": \"log\", \"msg\": \"Job not found\"}")
        await ws.close()
        return
    await stream_logs(ws, job)


@router.get("/result/{job_id}", response_class=HTMLResponse)
async def jungle_result(request: Request, job_id: str):
    templates = request.app.state.templates
    store, _ = _get_dependencies(request)
    job = store.get(job_id)

    if not job.done:
        return HTMLResponse(f"<h3>Job {job_id} still running...</h3>", status_code=202)

    summary_available = job.summary_image_path and job.summary_image_path.exists()
    video_available = job.final_video_path and job.final_video_path.exists()

    return templates.TemplateResponse(
        "jungle/result.html",
        {
            "request": request,
            "jobId": job_id,
            "summaryAvailable": bool(summary_available),
            "videoAvailable": bool(video_available),
            "ts": int(time.time() * 1000),
        },
    )
