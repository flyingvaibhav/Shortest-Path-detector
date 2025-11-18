"""YOLO powered detection service that runs inside a thread pool."""

from __future__ import annotations

import asyncio
import base64
import concurrent.futures
import time
from pathlib import Path
from typing import Dict, Tuple

import cv2
from ultralytics import YOLO

from app.core.config import Settings
from app.core.jobs import JobStore
from app.services.carbon import (
    estimate_carbon_sequestration,
    estimate_oxygen_output,
    estimate_pm_capture,
)


class DetectionService:
    """Wraps the heavy YOLO inference logic with background execution."""

    def __init__(self, settings: Settings, job_store: JobStore) -> None:
        self.settings = settings
        self.job_store = job_store
        self.model = YOLO(str(settings.model_path))
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=settings.worker_pool_size
        )

    async def run(self, job_id: str, source_path: Path) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            self.executor, lambda: self._run_sync(job_id, Path(source_path))
        )

    def _run_sync(self, job_id: str, source_path: Path) -> None:
        start = time.time()
        ext = source_path.suffix.lower()
        self.job_store.set_status(job_id, "running")
        self.job_store.append_log(job_id, "Starting detection...")

        try:
            if ext in {".jpg", ".jpeg", ".png"}:
                result = self._process_image(job_id, source_path)
            else:
                result = self._process_video(job_id, source_path)

            self.job_store.set_result(job_id, result)
            self.job_store.set_status(job_id, "done")
            self.job_store.append_log(
                job_id, f"Completed in {time.time() - start:.2f}s"
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            self.job_store.set_status(job_id, "error")
            self.job_store.append_log(job_id, f"Error: {exc}")

    def _process_image(self, job_id: str, path: Path) -> Dict[str, object]:
        self.job_store.append_log(job_id, "Reading image...")
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError("Invalid image provided.")

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.job_store.append_log(job_id, "Running YOLO model...")
        results = self.model.predict(
            rgb_image,
            conf=self.settings.detection_confidence,
            imgsz=self.settings.detection_img_size,
            verbose=False,
        )
        result = results[0]
        boxes = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else []
        tree_count = len(boxes)

        carbon_report = estimate_carbon_sequestration(tree_count)
        oxygen_report = estimate_oxygen_output(tree_count)
        pm_report = estimate_pm_capture(tree_count)

        annotated = rgb_image.copy()
        for (x1, y1, x2, y2) in boxes.astype(int):
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

        annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
        output_path = self._write_image(job_id, path, annotated_bgr)

        _, buffer = cv2.imencode(".png", annotated_bgr)
        encoded_img = base64.b64encode(buffer.tobytes()).decode("utf-8")

        self.job_store.append_log(job_id, f"Detected {tree_count} trees.")
        self.job_store.set_progress(job_id, 1.0)

        return {
            "type": "image",
            "count": tree_count,
            "tree_count": tree_count,
            "annotated_image_base64": f"data:image/png;base64,{encoded_img}",
            "output_image_url": f"/results/{output_path.name}",
            "carbon_report": carbon_report,
            "oxygen_report": oxygen_report,
            "pm_report": pm_report,
        }

    def _process_video(self, job_id: str, path: Path) -> Dict[str, object]:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            self.job_store.append_log(
                job_id, "Initial VideoCapture failed, retrying with CAP_FFMPEG..."
            )
            cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
        if not cap.isOpened():
            raise RuntimeError("Could not open video source.")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            self.job_store.append_log(
                job_id, "Frame count unavailable, estimating dynamically..."
            )
            total_frames = 1

        writer, output_path = self._prepare_video_writer(job_id, path, cap)

        frame_id = 0
        seen_ids = set()
        unique_trees = 0
        video_start = time.time()

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_id += 1

            results = self.model.track(
                frame,
                conf=self.settings.detection_confidence,
                imgsz=self.settings.detection_img_size,
                persist=True,
                verbose=False,
            )
            annotated = frame.copy()

            if results and results[0].boxes is not None:
                boxes = results[0].boxes
                xyxy = boxes.xyxy.cpu().numpy().astype(int)
                confs = boxes.conf.cpu().numpy()
                raw_ids = (
                    boxes.id.cpu().numpy().astype(int)
                    if getattr(boxes, "id", None) is not None
                    else [None] * len(xyxy)
                )

                for (x1, y1, x2, y2), conf, track_id in zip(xyxy, confs, raw_ids):
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"Tree {track_id}" if track_id is not None else "Tree"
                    cv2.putText(
                        annotated,
                        f"{label} ({conf:.2f})",
                        (x1, max(y1 - 10, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1,
                    )
                    if track_id is not None and track_id not in seen_ids:
                        seen_ids.add(int(track_id))
                        unique_trees += 1

            cv2.putText(
                annotated,
                f"Unique Trees: {unique_trees}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
            )
            writer.write(annotated)

            if frame_id % 10 == 0:
                elapsed = time.time() - video_start
                remaining = (
                    (elapsed / frame_id) * max(total_frames - frame_id, 1)
                    if frame_id
                    else 0
                )
                self.job_store.append_log(
                    job_id,
                    f"Frame {frame_id} | ETA: {remaining:.1f}s | Unique trees: {unique_trees}",
                )
                self.job_store.set_progress(job_id, frame_id / max(total_frames, 1))

        cap.release()
        writer.release()
        self.job_store.set_progress(job_id, 1.0)

        avg_tree_count = unique_trees / max(frame_id, 1)

        carbon_report = estimate_carbon_sequestration(unique_trees)
        oxygen_report = estimate_oxygen_output(unique_trees)
        pm_report = estimate_pm_capture(unique_trees)

        return {
            "type": "video",
            "frames_processed": frame_id,
            "unique_tree_count": unique_trees,
            "tree_count": unique_trees,
            "avg_tree_count": avg_tree_count,
            "output_video_url": f"/results/{output_path.name}",
            "carbon_report": carbon_report,
            "oxygen_report": oxygen_report,
            "pm_report": pm_report,
        }

    def _write_image(self, job_id: str, source_path: Path, image) -> Path:
        output_path = self.settings.results_dir / f"annotated_{job_id}_{source_path.stem}.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), image)
        return output_path

    def _prepare_video_writer(
        self, job_id: str, source_path: Path, cap: cv2.VideoCapture
    ) -> Tuple[cv2.VideoWriter, Path]:
        fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480

        output_path = self.settings.results_dir / f"annotated_{job_id}_{source_path.stem}.mp4"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError("Failed to open video writer.")

        return writer, output_path
