from pathlib import Path
import io
import base64
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

import numpy as np
from PIL import Image
import cv2

try:
    from ultralytics import YOLO
except Exception as e:
    YOLO = None


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "yolov8n.pt"

app = FastAPI(title="TreeSense YOLOv8 Inference")

# Serve the single HTML page from /
app.mount("/", StaticFiles(directory=str(BASE_DIR / "static"), html=True), name="static")


def load_model():
    if YOLO is None:
        raise RuntimeError("ultralytics package is not installed. Install from requirements.txt")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    # Load model once
    model = YOLO(str(MODEL_PATH))
    return model


# Load model at import time so subsequent calls are fast
try:
    model = load_model()
except Exception:
    model = None


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


def _numpy_from_upload(contents: bytes) -> np.ndarray:
    arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Accept an image upload, run YOLOv8 inference and return annotated image and tree count."""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded on server. Check server logs.")

    contents = await file.read()
    img = _numpy_from_upload(contents)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image")

    # Run inference (uses model defaults). We pass the numpy BGR image directly.
    results = model.predict(img, imgsz=640, conf=0.25, verbose=False)
    if not results:
        raise HTTPException(status_code=500, detail="Model did not return results")

    r = results[0]

    # Extract class names mapping from model
    names = {}
    try:
        names = {int(k): str(v).lower() for k, v in model.names.items()}
    except Exception:
        # fallback: try list
        try:
            names = {i: str(n).lower() for i, n in enumerate(model.names)}
        except Exception:
            names = {}

    # Find index for class named 'tree' if present
    tree_idx: Optional[int] = None
    for k, v in names.items():
        if v == "tree":
            tree_idx = k
            break

    # Get detected classes
    classes = []
    try:
        # res.boxes.cls may be a tensor
        classes = r.boxes.cls.cpu().numpy().astype(int)
    except Exception:
        try:
            classes = np.array(r.boxes.cls).astype(int)
        except Exception:
            classes = []

    # Count trees (if class name exists), otherwise count all detections
    if tree_idx is not None and len(classes) > 0:
        count = int((classes == tree_idx).sum())
    else:
        count = int(len(classes))

    # Create annotated image
    try:
        annotated = r.plot()
        # r.plot() often returns an RGB numpy array; ensure conversion to RGB PIL image
        if isinstance(annotated, np.ndarray):
            # if image has 3 channels assume it's in RGB or BGR; convert BGR->RGB by checking dtype
            if annotated.shape[-1] == 3:
                # heuristic: convert BGR->RGB using OpenCV
                annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            else:
                annotated_rgb = annotated
            pil_img = Image.fromarray(annotated_rgb)
        else:
            # If it's already a PIL Image
            pil_img = annotated
    except Exception:
        # fallback: draw nothing and return original upload
        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")

    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    data_uri = f"data:image/png;base64,{b64}"

    return JSONResponse({"count": count, "annotated_image": data_uri})
