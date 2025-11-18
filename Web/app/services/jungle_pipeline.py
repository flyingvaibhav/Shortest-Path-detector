"""Jungle-mode path planning pipeline utilities."""

from __future__ import annotations

import asyncio
import heapq
import io
import json
import subprocess
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import imageio_ffmpeg
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (Agg backend must be set first)
import numpy as np
from fastapi import WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, PlainTextResponse, StreamingResponse
from PIL import Image

Point = Tuple[int, int]


# ---------------------------- I/O helpers -----------------------------
def load_image(path: Path, max_dim: int = 900) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    width, height = img.size
    scale = min(max_dim / width, max_dim / height, 1.0)
    if scale < 1.0:
        new_size = (int(width * scale), int(height * scale))
        try:
            img = img.resize(new_size, Image.Resampling.LANCZOS)
        except AttributeError:  # pragma: no cover - older Pillow fallback
            img = img.resize(new_size, Image.LANCZOS)
    return np.array(img)


def compute_tree_mask(rgb: np.ndarray, threshold: float = 0.15) -> np.ndarray:
    r = rgb[..., 0].astype(np.float32)
    g = rgb[..., 1].astype(np.float32)
    b = rgb[..., 2].astype(np.float32)
    max_rb = np.maximum(r, b)
    exg = (g - max_rb) / (g + max_rb + 1e-6)
    return (exg > threshold).astype(np.int32)


def build_cost_map(tree_mask: np.ndarray, tree_cost: int = 20, open_cost: int = 1) -> np.ndarray:
    return np.where(tree_mask == 1, tree_cost, open_cost).astype(np.int32)


def ndarray_to_png_bytes(arr: np.ndarray) -> bytes:
    im = Image.fromarray(arr)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    buf.seek(0)
    return buf.read()


# --------------------------- A* animator ------------------------------
class AStarAnimator:
    def __init__(
        self,
        rgb: np.ndarray,
        cost_map: np.ndarray,
        start: Point,
        goal: Point,
        expansions_per_frame: int = 10,
    ) -> None:
        self.rgb = rgb
        self.cost = cost_map
        self.start = start
        self.goal = goal
        self.expansions_per_frame = max(1, expansions_per_frame)

        self.neighbors = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        self.g = {start: 0}
        self.f = {start: self._heuristic(start, goal)}
        self.parents: Dict[Point, Point] = {}
        self.open: List[Tuple[float, Point]] = [(self.f[start], start)]
        self.closed: set[Point] = set()
        self.done = False
        self.path: List[Point] = []
        self.last_current: Optional[Point] = None

        self.base = rgb.copy()
        mask = cost_map > cost_map.min()
        green_tint = np.zeros_like(self.base)
        green_tint[:] = [0, 140, 0]
        self.base[mask] = (
            0.4 * self.base[mask].astype(np.float32)
            + 0.6 * green_tint[mask].astype(np.float32)
        ).astype(np.uint8)

        self.frame_buffer = np.empty_like(self.base)
        self.visited_mask = np.zeros(cost_map.shape, dtype=bool)
        self.frontier_mask = np.zeros(cost_map.shape, dtype=bool)

    @staticmethod
    def _heuristic(a: Point, b: Point) -> float:
        return float(np.hypot(a[0] - b[0], a[1] - b[1]))

    def _advance(self) -> None:
        for _ in range(self.expansions_per_frame):
            if self.done or not self.open:
                self.done = self.done or not self.open
                return

            _, current = heapq.heappop(self.open)
            if current in self.closed:
                continue

            self.closed.add(current)
            self.visited_mask[current] = True
            self.last_current = current

            if current == self.goal:
                path: List[Point] = [current]
                while current in self.parents:
                    current = self.parents[current]
                    path.append(current)
                self.path = list(reversed(path))
                self.done = True
                return

            current_cost = self.g[current]
            for dy, dx in self.neighbors:
                ny, nx = current[0] + dy, current[1] + dx
                if 0 <= ny < self.cost.shape[0] and 0 <= nx < self.cost.shape[1]:
                    tentative = current_cost + int(self.cost[ny, nx])
                    neighbor = (ny, nx)
                    if tentative < self.g.get(neighbor, float("inf")):
                        self.parents[neighbor] = current
                        self.g[neighbor] = tentative
                        self.f[neighbor] = tentative + self._heuristic(neighbor, self.goal)
                        heapq.heappush(self.open, (self.f[neighbor], neighbor))

    def step(self) -> np.ndarray:
        self._advance()
        np.copyto(self.frame_buffer, self.base)

        if self.visited_mask.any():
            self.frame_buffer[self.visited_mask] = [100, 149, 237]

        self.frontier_mask.fill(False)
        for _, node in self.open:
            if node not in self.closed:
                self.frontier_mask[node] = True
        if self.frontier_mask.any():
            self.frame_buffer[self.frontier_mask] = [255, 165, 0]

        if self.last_current is not None and not self.done:
            cy, cx = self.last_current
            self.frame_buffer[cy, cx] = [65, 105, 225]

        sy, sx = self.start
        gy, gx = self.goal
        self.frame_buffer[sy, sx] = [255, 0, 255]
        self.frame_buffer[gy, gx] = [255, 255, 0]

        if self.path:
            for py, px in self.path:
                self.frame_buffer[py, px] = [255, 0, 0]
        return self.frame_buffer


# --------------------------- Summary plots ----------------------------
def plot_paths_summary(anim: AStarAnimator, rgb: np.ndarray, original: np.ndarray, cost_map: np.ndarray):
    if not anim.path:
        return None
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    rows = [pt[0] for pt in anim.path]
    cols = [pt[1] for pt in anim.path]

    axes[0, 0].imshow(original)
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis("off")

    cost_vis = cost_map.astype(np.float32)
    axes[0, 1].imshow(cost_vis, cmap="viridis")
    axes[0, 1].set_title("Cost Map (Higher = Trees)")
    axes[0, 1].axis("off")

    axes[1, 0].imshow(rgb)
    axes[1, 0].plot(cols, rows, color="red", linewidth=3, alpha=0.9)
    axes[1, 0].set_title("Best Path Overlay")
    axes[1, 0].axis("off")
    axes[1, 0].set_ylim(rgb.shape[0], 0)

    axes[1, 1].imshow(rgb)
    explored_y, explored_x = np.where(anim.visited_mask)
    if explored_y.size:
        axes[1, 1].scatter(explored_x, explored_y, s=2, c="gold", alpha=0.3, label="Explored")
    axes[1, 1].plot(cols, rows, color="red", linewidth=3, alpha=0.9, label="Best Path")
    axes[1, 1].set_title("Explored vs Best Path")
    axes[1, 1].axis("off")
    axes[1, 1].set_ylim(rgb.shape[0], 0)
    axes[1, 1].legend(loc="upper right")

    fig.subplots_adjust(hspace=0.095, wspace=0.062, top=0.945, bottom=0.015, left=0.010, right=0.990)
    return fig


def fallback_straight_path(start: Point, goal: Point) -> List[Point]:
    y0, x0 = start
    y1, x1 = goal
    path: List[Point] = []

    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy

    while True:
        path.append((y0, x0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy
    return path


# ------------------------------ Video ---------------------------------
def save_animation(
    anim: AStarAnimator,
    interval_ms: int,
    output_path: Path,
    hold_frames: int = 15,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    fps = max(1, int(round(1000 / interval_ms)))
    h, w = anim.rgb.shape[:2]
    size = f"{w}x{h}"

    process = subprocess.Popen(  # noqa: S603, S607  (ffmpeg controlled input)
        [
            ffmpeg_path,
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            size,
            "-r",
            str(fps),
            "-i",
            "-",
            "-an",
            "-vcodec",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(output_path),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )

    try:
        frame = anim.step()
        process.stdin.write(frame.tobytes())

        while not anim.done:
            frame = anim.step()
            process.stdin.write(frame.tobytes())

        for _ in range(hold_frames):
            process.stdin.write(anim.frame_buffer.tobytes())

        process.stdin.close()
        stderr_output = process.stderr.read().decode("utf-8", errors="ignore")
        process.wait()

        if process.returncode != 0:
            raise RuntimeError(f"ffmpeg failed (code {process.returncode}): {stderr_output}")

        if not output_path.exists() or output_path.stat().st_size == 0:
            raise RuntimeError("ffmpeg wrote no output or empty file")

    except Exception as exc:  # pragma: no cover - defensive cleanup
        try:
            process.kill()
        except Exception:
            pass
        raise RuntimeError(f"Failed to save animation video: {exc}") from exc
    finally:
        try:
            if process.poll() is None:
                process.terminate()
            process.stderr.close()
        except Exception:
            pass

    return output_path


# ------------------------------ Jobs ----------------------------------
@dataclass
class JungleJob:
    job_id: str
    upload_path: Path
    output_root: Path
    rgb: Optional[np.ndarray] = None
    original_rgb: Optional[np.ndarray] = None
    tree_mask: Optional[np.ndarray] = None
    cost_map: Optional[np.ndarray] = None
    start: Optional[Point] = None
    goal: Optional[Point] = None
    animator: Optional[AStarAnimator] = None
    final_video_path: Optional[Path] = None
    summary_image_path: Optional[Path] = None
    error: Optional[str] = None
    done: bool = False
    logs: Deque[str] = field(default_factory=lambda: deque(maxlen=5000))
    current_frame: Optional[np.ndarray] = None
    lock: threading.Lock = field(default_factory=threading.Lock)

    def log(self, msg: str) -> None:
        with self.lock:
            ts = time.strftime("%H:%M:%S")
            self.logs.append(f"[{ts}] {msg}")


class JungleJobStore:
    def __init__(self, output_root: Path) -> None:
        self._jobs: Dict[str, JungleJob] = {}
        self._lock = threading.Lock()
        self._output_root = output_root
        self._output_root.mkdir(parents=True, exist_ok=True)

    def create(self, upload_path: Path) -> JungleJob:
        job_id = uuid.uuid4().hex[:12]
        out_root = self._output_root / job_id
        job = JungleJob(job_id=job_id, upload_path=upload_path, output_root=out_root)
        with self._lock:
            self._jobs[job_id] = job
        return job

    def get(self, job_id: str) -> JungleJob:
        with self._lock:
            if job_id not in self._jobs:
                raise KeyError("Job not found")
            return self._jobs[job_id]


# --------------------------- Pipeline run ------------------------------
def run_pipeline(job: JungleJob, expansions_per_frame: int = 25, interval_ms: int = 12) -> None:
    try:
        job.log("Loading image")
        rgb = load_image(job.upload_path, max_dim=900)
        job.rgb = rgb
        job.original_rgb = rgb.copy()
        job.current_frame = rgb.copy()

        job.log("Computing vegetation mask")
        tree_mask = compute_tree_mask(rgb, threshold=0.15)
        job.tree_mask = tree_mask

        job.log("Building traversal cost map")
        cost_map = build_cost_map(tree_mask, tree_cost=20, open_cost=1)
        job.cost_map = cost_map

        if job.start is None or job.goal is None:
            job.error = "Start and goal not set"
            job.log("Error: start and goal not set")
            job.done = True
            return

        job.log(f"Starting A* search from {job.start} to {job.goal}")
        animator = AStarAnimator(rgb, cost_map, job.start, job.goal, expansions_per_frame=expansions_per_frame)
        job.animator = animator

        video_dir = job.output_root / "video"
        image_dir = job.output_root / "image"
        video_dir.mkdir(parents=True, exist_ok=True)
        image_dir.mkdir(parents=True, exist_ok=True)
        video_path = video_dir / "search_animation.mp4"

        try:
            job.log("Writing video with imageio-ffmpeg")
            final_path = save_animation(animator, interval_ms=interval_ms, output_path=video_path, hold_frames=15)
            job.final_video_path = final_path
            job.log(f"Video saved: {final_path}")
        except Exception as exc:
            job.log(f"Video export failed: {exc}")
            while not animator.done:
                frame = animator.step()
                job.current_frame = frame.copy()

        if animator.done:
            job.current_frame = animator.frame_buffer.copy()

        if animator.path:
            trees_crossed = int(np.sum(tree_mask[tuple(np.array(animator.path).T)]))
            job.log(f"Path length: {len(animator.path)}")
            job.log(f"Estimated trees crossed: {trees_crossed}")

            job.log("Creating summary figure")
            fig2 = plot_paths_summary(animator, animator.base, job.original_rgb, cost_map)
            if fig2 is not None:
                summary_path = job.output_root / "image" / "best_path_summary.png"
                fig2.savefig(summary_path, bbox_inches="tight", dpi=160)
                plt.close(fig2)
                job.summary_image_path = summary_path
                job.log(f"Summary saved: {summary_path}")
        else:
            job.log("No viable path found. Generating straight-line fallback")
            direct = fallback_straight_path(job.start, job.goal)
            trees_crossed = int(np.sum(tree_mask[tuple(np.array(direct).T)]))
            job.log(f"Direct path length: {len(direct)}")
            job.log(f"Estimated trees encountered on direct path: {trees_crossed}")

        job.done = True
        job.log("Processing complete")
    except Exception as exc:  # pragma: no cover - defensive logging
        job.error = str(exc)
        job.done = True
        job.log(f"Fatal error: {exc}")


# ------------------------------ WebSocket ------------------------------
async def stream_logs(ws: WebSocket, job: JungleJob) -> None:
    await ws.accept()
    last_len = 0
    try:
        for msg in list(job.logs):
            await ws.send_text(json.dumps({"type": "log", "msg": msg}))

        while True:
            await asyncio.sleep(0.3)
            with job.lock:
                if len(job.logs) > last_len:
                    for idx in range(last_len, len(job.logs)):
                        await ws.send_text(json.dumps({"type": "log", "msg": job.logs[idx]}))
                    last_len = len(job.logs)
            await ws.send_text(json.dumps({"type": "frame"}))
            if job.done:
                await ws.send_text(json.dumps({"type": "done"}))
                break
    except WebSocketDisconnect:
        return
    except Exception as exc:  # pragma: no cover - websocket best effort
        try:
            await ws.send_text(json.dumps({"type": "log", "msg": f"WebSocket error: {exc}"}))
        except Exception:
            pass
        finally:
            try:
                await ws.close()
            except Exception:
                pass


# --------------------------- Response helpers -------------------------
def base_image_response(job: JungleJob):
    if job.rgb is None:
        return PlainTextResponse("Base image not ready", status_code=404)
    png = ndarray_to_png_bytes(job.rgb)
    return StreamingResponse(io.BytesIO(png), media_type="image/png")


def current_image_response(job: JungleJob):
    arr = job.current_frame if job.current_frame is not None else job.rgb
    if arr is None:
        return PlainTextResponse("Current frame not ready", status_code=404)
    png = ndarray_to_png_bytes(arr)
    return StreamingResponse(io.BytesIO(png), media_type="image/png")


def summary_response(job: JungleJob):
    if not job.summary_image_path or not job.summary_image_path.exists():
        return PlainTextResponse("Summary not available", status_code=404)
    return FileResponse(job.summary_image_path, media_type="image/png")


def video_response(job: JungleJob):
    if not job.final_video_path or not job.final_video_path.exists():
        return PlainTextResponse("Video not available", status_code=404)
    return FileResponse(job.final_video_path, media_type="video/mp4", filename="search_animation.mp4")


def stream_video_response(job: JungleJob):
    if not job.final_video_path or not job.final_video_path.exists():
        return PlainTextResponse("Video not found", status_code=404)
    return FileResponse(
        job.final_video_path,
        media_type="video/mp4",
        filename=f"{job.job_id}.mp4",
        headers={"Accept-Ranges": "bytes"},
    )
