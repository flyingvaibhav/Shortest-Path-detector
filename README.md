# TreeSense

TreeSense is an end-to-end, YOLOv8-powered toolkit for measuring urban canopy health. It couples reproducible training notebooks, a background inference API, carbon/oxygen estimation utilities, and an interactive "Jungle Mode" path planner for exploring vegetation-rich routes. The project was built to answer a single question: *How many trees surround this location and how do they impact air quality?*

## Highlights

- **Dataset-backed training pipeline** – fine-tune YOLOv8 on the [Tree-Top-View v6](https://universe.roboflow.com/treedataset-clsqo/tree-top-view/dataset/6) dataset (CC BY 4.0) using `ModelTraining.ipynb`.
- **Live inference notebooks** – `LiveTreeSnese.ipynb` provides camera/image/video annotation helpers plus carbon, oxygen, and particulate-matter estimates per detection.
- **FastAPI inference server** – the `Web/` folder exposes a multi-page UI and JSON API. Upload an image/video, follow the Server-Sent Events (SSE) stream, and download annotated outputs powered by `ultralytics` + OpenCV.
- **Jungle Mode path planning** – an A*-based visualizer carves least-cost paths across vegetation masks, streaming progress over WebSockets and exporting summary plots + MP4 animations.
- **Environmental metrics** – shared `services/carbon.py` helpers approximate CO₂ sequestration, oxygen output, and PM capture using the detected tree count so every workflow returns actionable context.

## Repository layout

```
TreeSense/
├── Tree-Top-View-6/         # Dataset clone from Roboflow (train/val/test splits + data.yaml)
├── runs/                    # YOLO training artifacts (args, metrics, weights)
├── Web/                     # FastAPI application (templates, static, services)
├── ModelTraining.ipynb      # Main YOLO fine-tuning + evaluation notebook
├── LiveTreeSnese.ipynb      # Notebook for ad-hoc image/video annotation + metrics
├── installingDataset.ipynb  # Helper walkthrough to sync Roboflow data locally
├── installingYoloV8.ipynb   # Environment bootstrap instructions for Ultralytics
├── test/                    # Sample imagery for local inference tests
├── yolov8n.pt               # Base Ultralytics checkpoint used for transfer learning
└── README.md                # You are here
```

## Requirements

- Python 3.10+
- CUDA-capable GPU recommended (CPU works for small batches)
- FFmpeg is required for Jungle Mode video exports (installed automatically via `imageio-ffmpeg`)
- See [`requirement.txt`](requirement.txt) for the consolidated dependency list

## Quick start

1. **Create a virtual environment**
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\activate
   ```
2. **Install dependencies**
   ```powershell
   pip install --upgrade pip
   pip install -r requirement.txt
   ```
3. **Download / verify the dataset**
   - Grab Tree-Top-View v6 from Roboflow (link above) and extract it so that `Tree-Top-View-6/data.yaml` matches the provided file.
   - The dataset uses Roboflow's standard YOLO format (train/valid/test folders with `images/` + `labels/`).
4. **Verify pretrained weights**
   - The base `yolov8n.pt` checkpoint ships with the repo.
   - After training, copy your `runs/tree_canopy/<experiment>/weights/best.pt` into `Web/models/best.pt` so the API serves the latest model.

## Training workflow (`ModelTraining.ipynb`)

The notebook orchestrates the entire fine-tuning loop:

1. **Environment prep** – optional cell to install `ultralytics`, `opencv-python`, and plotting libraries.
2. **Imports + model load** – instantiates `YOLO("yolov8n.pt")` for transfer learning.
3. **Data config** – verifies `Tree-Top-View-6/data.yaml` and builds a `train_settings` dict (80 epochs, 640px images, batch 16, patience 20, GPU `device=0`).
4. **Training call** – `model.train(**train_settings)` writes logs, metrics, and weights to `runs/tree_canopy/yolov8n-tree`.
5. **Validation** – `model.val(split="val")` surfaces mAP@50:95 and mAP@50 for quick regression tracking.
6. **Batch prediction loop** – iterates through `Tree-Top-View-6/test/images`, stores annotated JPEGs under `runs/tree_canopy/predictions`, and previews the first three detections inline.
7. **Utility function** – `detect_trees_in_image(...)` accepts any path, draws detections, returns the annotated array plus the tree count (also reused by the web server).

> Tip: The notebook is structured so you can re-run Cells 4–7 whenever you update hyperparameters without re-importing libraries.

## Live annotation + ESG metrics (`LiveTreeSnese.ipynb`)

This notebook loads `runs/tree_canopy/yolov8n-tree/weights/best.pt` (fallbacks to `yolov8n.pt`) and exposes:

- `annotate_trees_in_image(...)` – fast ad-hoc previews with Matplotlib overlays and optional file output.
- `annotate_trees_in_video(...)` – processes any OpenCV-readable stream, embeds the running tree count per frame, and writes MP4 results.
- `annotate_unique_trees_in_video(...)` – leverages `model.track(..., persist=True)` to count each track ID exactly once.
- Sustainability helpers from `services/carbon.py`:
  - `estimate_carbon_sequestration(tree_count)` – rough stock + annual CO₂e estimates.
  - `estimate_oxygen_output(tree_count, age_group)` – kg of O₂ per year for young/medium/mature stands.
  - `estimate_pm_capture(tree_count, ...)` – PM₂.₅ removal approximations via rule-of-thumb or deposition models.

Use the final cell as a template: it loads `./test/hq720.jpg`, prints the detected tree total, and dumps all three reports to the console.

## FastAPI inference server (`Web/`)

The `Web` folder contains a self-served UI plus JSON endpoints:

- `app/main.py` wires settings, job stores, and detection services.
- `app/api/routes.py` exposes:
  - `GET /` – landing page
  - `GET /detect` – upload UI for the standard pipeline
  - `POST /upload` – accepts an image or video, spins up a background detection job
  - `GET /progress/{job_id}` – SSE stream of job logs/progress
  - `GET /result/{job_id}` – returns the finished payload (counts, environmental reports, file URLs)
- `app/services/detection.py` handles CPU/GPU inference in a thread pool, writes annotated imagery, and enriches outputs with ESG metrics.

### Running the server

```powershell
cd Web
python -m venv .venv
.\.venv\Scripts\activate
pip install -r ..\requirement.txt
uvicorn app:app --reload --port 8000
```

Visit `http://localhost:8000/detect` to use the browser workflow. Uploads land in `Web/uploads/`, annotated assets are written to `Web/result/`, and the SSE console streams messages such as "Frame 120 | ETA: 3.5s | Unique trees: 42".

### Environment configuration

`app/core/config.py` reads the following optional variables:

- `APP_BASE_DIR` – overrides the directory containing `static/`, `templates/`, `models/`
- `APP_UPLOAD_DIR`, `APP_RESULTS_DIR` – custom storage roots
- `YOLO_MODEL_FILENAME` – alternate weights name (default `best.pt`)
- `WORKER_POOL_SIZE` – threads for background inference
- `DETECTION_CONFIDENCE`, `DETECTION_IMG_SIZE` – inference thresholds shared by notebooks & API

## Jungle Mode path planning

Accessible under `http://localhost:8000/jungle`, this workflow lets you:

1. Upload a high-res aerial image.
2. Pick start/goal points interactively.
3. Watch an A*-driven search explore the vegetation mask (trees are higher-cost cells built via the Excess Green index).
4. Download summary plots and a rendered MP4 showing the exploration, best path, and unique-tree corridors.

Implementation details live in `app/services/jungle_pipeline.py`, which pairs FFmpeg streaming with WebSocket log updates for a fluid UX even on large rasters.

## References

- *Ultralytics YOLOv8*: https://docs.ultralytics.com/
- *Tree-Top-View Dataset v6 (CC BY 4.0)*: https://universe.roboflow.com/treedataset-clsqo/tree-top-view

Feel free to adapt the notebooks, retrain with different hyperparameters, or plug TreeSense into other ESG dashboards—the codebase is intentionally modular so you can swap models, datasets, or sustainability heuristics as your needs evolve.
