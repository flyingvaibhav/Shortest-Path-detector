import os
# -------------------------------------------------------------------
# Environment setup (before importing torch/cv2)
# -------------------------------------------------------------------
# Prevent OpenMP runtime conflict
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from uuid import uuid4
from ultralytics import YOLO
import asyncio, cv2, time, json, base64, concurrent.futures
from pathlib import Path
from typing import Dict
from functools import partial

# -------------------------------------------------------------------
# Basic setup
# -------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
app = FastAPI()

app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")

model_path = BASE_DIR / "models" / "best.pt"
model = YOLO(str(model_path))

# Global job state
JOBS: Dict[str, Dict] = {}
EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=2)


# -------------------------------------------------------------------
# Heavy detection work (runs inside thread pool)
# -------------------------------------------------------------------
def run_detection_sync(job_id: str, path: str):
    try:
        JOBS[job_id]["status"] = "running"
        JOBS[job_id]["logs"].append("Starting detection...")
        ext = Path(path).suffix.lower()
        start = time.time()

        # ---------------- IMAGE ----------------
        if ext in [".jpg", ".jpeg", ".png"]:
            JOBS[job_id]["logs"].append("Reading image...")
            img = cv2.imread(path)
            if img is None:
                raise ValueError("Invalid image format.")
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            JOBS[job_id]["logs"].append("Running YOLO model...")
            results = model.predict(rgb, conf=0.25, imgsz=640, verbose=False)
            result = results[0]
            boxes = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else []
            tree_count = len(boxes)

            annotated = rgb.copy()
            for (x1, y1, x2, y2) in boxes.astype(int):
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

            annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
            output_dir = BASE_DIR / "static"
            output_dir.mkdir(exist_ok=True)
            image_name = f"annotated_{job_id}_{Path(path).stem}.png"
            image_path = output_dir / image_name
            cv2.imwrite(str(image_path), annotated_bgr)

            _, buffer = cv2.imencode(".png", annotated_bgr)
            encoded_img = base64.b64encode(buffer.tobytes()).decode("utf-8")

            JOBS[job_id]["result"] = {
                "type": "image",
                "count": tree_count,
                "annotated_image_base64": f"data:image/png;base64,{encoded_img}",
                "output_image_url": f"/static/{image_name}",
            }
            JOBS[job_id]["logs"].append(f"Detected {tree_count} trees.")
            JOBS[job_id]["status"] = "done"
            JOBS[job_id]["logs"].append(f"Completed in {time.time() - start:.2f}s")
            return

        # ---------------- VIDEO ----------------
        cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
        if not cap.isOpened():
            raise RuntimeError("Could not open video.")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            JOBS[job_id]["logs"].append("Frame count unavailable, estimating dynamically...")
            total_frames = 1  # fallback

        JOBS[job_id]["logs"].append(f"Processing video... estimated {total_frames} frames")

        output_dir = BASE_DIR / "static"
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / f"annotated_{Path(path).stem}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height), apiPreference=cv2.CAP_FFMPEG)
        if not writer.isOpened():
            raise RuntimeError("Failed to open video writer.")

        frame_id = 0
        seen_ids = set()
        unique_trees = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_id += 1

            results = model.track(frame, conf=0.25, imgsz=640, persist=True, verbose=False)
            annotated = frame.copy()

            if results and results[0].boxes is not None:
                boxes = results[0].boxes
                xyxy = boxes.xyxy.cpu().numpy().astype(int)
                confs = boxes.conf.cpu().numpy()
                cls_ids = boxes.cls.cpu().numpy().astype(int)
                track_ids = (
                    boxes.id.cpu().numpy().astype(int)
                    if getattr(boxes, "id", None) is not None
                    else []
                )

                for (x1, y1, x2, y2), conf, cls_id, track_id in zip(xyxy, confs, cls_ids, track_ids):
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(annotated, f"Tree {track_id} ({conf:.2f})", (x1, max(y1 - 10, 0)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    if track_id not in seen_ids:
                        seen_ids.add(int(track_id))
                        unique_trees += 1

            cv2.putText(annotated, f"Unique Trees: {unique_trees}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            writer.write(annotated)

            if frame_id % 10 == 0:
                eta = (time.time() - start) / frame_id * (max(1, total_frames - frame_id))
                JOBS[job_id]["logs"].append(
                    f"Frame {frame_id} | ETA: {eta:.1f}s | Unique trees: {unique_trees}"
                )
                JOBS[job_id]["progress"] = frame_id / max(1, total_frames)

        cap.release()
        writer.release()

        JOBS[job_id]["result"] = {
            "type": "video",
            "frames_processed": frame_id,
            "unique_tree_count": unique_trees,
            "output_video_url": f"/static/{output_path.name}",
        }
        JOBS[job_id]["status"] = "done"
        JOBS[job_id]["logs"].append(f"Video processed. Unique trees: {unique_trees}")
        JOBS[job_id]["logs"].append(f"Completed in {time.time() - start:.2f}s")

    except Exception as e:
        JOBS[job_id]["status"] = "error"
        JOBS[job_id]["logs"].append(f"Error: {str(e)}")


# -------------------------------------------------------------------
# Background task wrapper
# -------------------------------------------------------------------
async def detect_job(job_id: str, path: str):
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(EXECUTOR, partial(run_detection_sync, job_id, path))


# -------------------------------------------------------------------
# Routes
# -------------------------------------------------------------------
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    job_id = str(uuid4())
    upload_dir = BASE_DIR / "uploads"
    upload_dir.mkdir(exist_ok=True)
    temp_path = upload_dir / f"{job_id}_{file.filename}"
    with open(temp_path, "wb") as f:
        f.write(await file.read())

    JOBS[job_id] = {"status": "queued", "logs": [], "result": {}, "progress": 0.0}
    asyncio.create_task(detect_job(job_id, str(temp_path)))
    return {"job_id": job_id}


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("landing.html", {"request": request})


@app.get("/detect", response_class=HTMLResponse)
async def detect_ui(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/progress/{job_id}")
async def progress_stream(job_id: str):
    async def event_stream():
        last_len = 0
        while True:
            job = JOBS.get(job_id)
            if not job:
                yield f"data: {json.dumps({'error': 'Invalid job'})}\n\n"
                break

            new_logs = job["logs"][last_len:]
            last_len = len(job["logs"])

            for log in new_logs:
                payload = {
                    "status": job["status"],
                    "progress": round(job["progress"], 3),
                    "log": log,
                }
                yield f"data: {json.dumps(payload)}\n\n"

            # keep alive ping every 5s
            if job["status"] in ("done", "error"):
                yield f"data: {json.dumps({'status': job['status'], 'done': True})}\n\n"
                break

            await asyncio.sleep(1)

    return StreamingResponse(event_stream(), media_type="text/event-stream")

@app.get("/result/{job_id}")
async def result(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        return JSONResponse({"error": "Invalid job id"}, status_code=404)

    # Wait for background detection to finish (max 10 min)
    timeout = 600  # seconds
    waited = 0
    while job["status"] not in ("done", "error") and waited < timeout:
        await asyncio.sleep(1)
        waited += 1

    # Job errored
    if job["status"] == "error":
        return JSONResponse(
            {"error": "Detection failed", "logs": job.get("logs", [])},
            status_code=500,
        )

    # Timeout safety
    if waited >= timeout:
        return JSONResponse(
            {"error": "Detection timeout", "logs": job.get("logs", [])},
            status_code=504,
        )

    # Job done but result empty
    if not job.get("result"):
        return JSONResponse(
            {"error": "No result generated", "logs": job.get("logs", [])},
            status_code=500,
        )

    # Success
    return JSONResponse(job["result"])

