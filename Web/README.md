# TreeSense

TreeSense is a FastAPI powered YOLO inference service that counts and tracks trees from uploaded imagery or video. The backend offloads heavy inference work to a thread pool and streams job progress to the browser through Server-Sent Events (SSE).

## Project layout

```
app/
  api/          # FastAPI routers and request handlers
  core/         # Configuration helpers and job store utilities
  services/     # Long-running detection services (YOLO, OpenCV)
models/         # YOLO weight files (ignored from repo history)
static/         # Frontend assets
templates/      # Jinja2 HTML templates
uploads/        # Temporary user uploads (auto-created)
```

## Getting started

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app:app --reload
```

Open http://localhost:8000 to view the landing page, then switch to `/detect` to upload imagery. The UI will stream live logs and provide annotated downloads once inference finishes.
