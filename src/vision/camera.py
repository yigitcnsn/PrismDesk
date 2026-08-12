"""Robust USB camera capture for Raspberry Pi (V4L2 + MJPG).

Smoke-test lessons baked in:
- Use integer device indices with cv2.CAP_V4L2 (never '/dev/videoN' strings on arm64).
- Prefer MJPG @ 1080p50 so the camera ISP decodes (60 can be unstable on this USB cam).
- Silence OpenCV backend spam via OPENCV_LOG_LEVEL.
- Detect corrupt/empty frames and re-open with a short sleep to avoid CPU spin.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import yaml

# Must be set before VideoCapture opens backends that log loudly on Pi.
os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")
try:
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
except Exception:
    pass


@dataclass
class CameraConfig:
    device_indices: List[int] = field(default_factory=lambda: [1, 0, 2])
    width: int = 1920
    height: int = 1080
    fps: int = 50
    fourcc: str = "MJPG"
    # Skip many corrupt MJPEG frames before reopening the device (USB reopen is costly/flaky on Pi).
    max_consecutive_bad_frames: int = 24
    reopen_sleep_sec: float = 0.5
    read_timeout_retries: int = 8
    # How many cheap skip-reads to try before counting toward reopen.
    skip_before_reopen: int = 16
    # Mount orientation: 0, 90, 180, or 270 (overhead cams are often mounted upside-down).
    rotate_degrees: int = 0
    model: str = "fisheye"
    camera_matrix: Optional[np.ndarray] = None
    dist_coeffs: Optional[np.ndarray] = None
    image_size: Optional[Tuple[int, int]] = None


def load_camera_config(path: str | Path) -> CameraConfig:
    path = Path(path)
    if not path.is_file():
        return CameraConfig()
    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    indices = raw.get("device_indices", [1, 0, 2])
    cfg = CameraConfig(
        device_indices=[int(i) for i in indices],
        width=int(raw.get("width", 1920)),
        height=int(raw.get("height", 1080)),
        fps=int(raw.get("fps", 50)),
        fourcc=str(raw.get("fourcc", "MJPG")),
        max_consecutive_bad_frames=int(raw.get("max_consecutive_bad_frames", 24)),
        reopen_sleep_sec=float(raw.get("reopen_sleep_sec", 0.5)),
        read_timeout_retries=int(raw.get("read_timeout_retries", 8)),
        skip_before_reopen=int(raw.get("skip_before_reopen", 16)),
        rotate_degrees=int(raw.get("rotate_degrees", 0)),
        model=str(raw.get("model", "fisheye")),
    )
    if raw.get("camera_matrix") is not None:
        cfg.camera_matrix = np.asarray(raw["camera_matrix"], dtype=np.float64)
    if raw.get("dist_coeffs") is not None:
        cfg.dist_coeffs = np.asarray(raw["dist_coeffs"], dtype=np.float64).reshape(-1)
    if raw.get("image_size") is not None:
        wh = raw["image_size"]
        cfg.image_size = (int(wh[0]), int(wh[1]))
    return cfg


def save_camera_config(path: str | Path, cfg: CameraConfig) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "device_indices": list(cfg.device_indices),
        "width": cfg.width,
        "height": cfg.height,
        "fps": cfg.fps,
        "fourcc": cfg.fourcc,
        "max_consecutive_bad_frames": cfg.max_consecutive_bad_frames,
        "reopen_sleep_sec": cfg.reopen_sleep_sec,
        "read_timeout_retries": cfg.read_timeout_retries,
        "skip_before_reopen": cfg.skip_before_reopen,
        "rotate_degrees": int(cfg.rotate_degrees),
        "model": cfg.model,
        "camera_matrix": cfg.camera_matrix.tolist() if cfg.camera_matrix is not None else None,
        "dist_coeffs": cfg.dist_coeffs.tolist() if cfg.dist_coeffs is not None else None,
        "image_size": list(cfg.image_size) if cfg.image_size is not None else None,
    }
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, default_flow_style=False, sort_keys=False)


def is_frame_ok(frame: Optional[np.ndarray], expect_w: int, expect_h: int) -> bool:
    """Reject None, wrong shape, empty, or classic USB 'corrupt' frames.

    USB MJPEG on Pi often emits a few green/black/flat frames — we skip those.
    Size check is loose so renegotiated modes (e.g. 800x600 vs 960x540) still pass.
    """
    if frame is None:
        return False
    if not isinstance(frame, np.ndarray):
        return False
    if frame.ndim != 3 or frame.shape[2] != 3:
        return False
    h, w = frame.shape[:2]
    if h < 16 or w < 16:
        return False
    # Allow large renegotiation (USB cams often ignore requested WxH).
    if expect_w > 0 and expect_h > 0:
        if abs(w - expect_w) > expect_w * 0.5 or abs(h - expect_h) > expect_h * 0.5:
            return False
    # Sample a center crop — cheaper and less sensitive to edge decode glitches.
    y0, y1 = h // 4, 3 * h // 4
    x0, x1 = w // 4, 3 * w // 4
    sample = frame[y0:y1, x0:x1]
    if sample.size == 0:
        return False
    mean = float(sample.mean())
    std = float(sample.std())
    # Near-constant / all-black / all-green = classic corrupt MJPEG decode
    if std < 1.25:
        return False
    if mean < 1.0 or mean > 254.0:
        return False
    # Strong green cast with almost no red/blue is a common V4L2 fail frame.
    b, g, r = sample[:, :, 0].mean(), sample[:, :, 1].mean(), sample[:, :, 2].mean()
    if g > 120 and g > (r + b) * 1.8 and std < 12.0:
        return False
    return True


class Camera:
    """V4L2 capture with index fallback, MJPG setup, and corrupt-frame recovery."""

    def __init__(
        self,
        config: Optional[CameraConfig] = None,
        device_index: Optional[int] = None,
    ) -> None:
        self.config = config or CameraConfig()
        if device_index is not None:
            self.config.device_indices = [int(device_index)] + [
                i for i in self.config.device_indices if i != int(device_index)
            ]
        self._cap: Optional[cv2.VideoCapture] = None
        self._active_index: Optional[int] = None
        self._bad_streak = 0
        self._last_good: Optional[np.ndarray] = None
        self._corrupt_skips = 0

    @property
    def active_index(self) -> Optional[int]:
        return self._active_index

    def open(self) -> int:
        """Open first working device. Returns active index. Raises RuntimeError on failure."""
        self.close()
        last_err = "no devices tried"
        for index in self.config.device_indices:
            try:
                cap = self._open_index(index)
            except Exception as exc:  # noqa: BLE001
                last_err = str(exc)
                time.sleep(self.config.reopen_sleep_sec)
                continue
            if cap is None:
                last_err = f"index {index} failed to open"
                time.sleep(self.config.reopen_sleep_sec)
                continue
            # Warm up + validate a few frames (use negotiated size, not request)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or self.config.width
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or self.config.height
            ok = False
            for _ in range(max(5, self.config.read_timeout_retries)):
                grabbed, frame = cap.read()
                if grabbed and is_frame_ok(frame, w, h):
                    ok = True
                    break
                time.sleep(0.05)
            if not ok:
                cap.release()
                last_err = f"index {index} opened but frames corrupt/empty"
                time.sleep(self.config.reopen_sleep_sec)
                continue
            self._cap = cap
            self._active_index = index
            self._bad_streak = 0
            # Lock expectation to what the driver actually delivered.
            self.config.width = w
            self.config.height = h
            return index
        raise RuntimeError(
            f"Could not open USB camera on indices {self.config.device_indices}: {last_err}"
        )

    def _open_index(self, index: int) -> Optional[cv2.VideoCapture]:
        # Integer index — V4L2 on Linux/Pi; default backend elsewhere (macOS/Windows debug).
        if sys.platform.startswith("linux"):
            cap = cv2.VideoCapture(int(index), cv2.CAP_V4L2)
        else:
            cap = cv2.VideoCapture(int(index))
        if not cap.isOpened():
            cap.release()
            return None
        try:
            fourcc = cv2.VideoWriter_fourcc(*self.config.fourcc[:4])
            cap.set(cv2.CAP_PROP_FOURCC, fourcc)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.config.width))
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.config.height))
            cap.set(cv2.CAP_PROP_FPS, float(self.config.fps))
        except Exception:
            pass
        # Reduce internal buffering so we don't process stale/corrupt queues
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        return cap

    def _orient(self, frame: np.ndarray) -> np.ndarray:
        """Apply mount rotation from config (0/90/180/270)."""
        deg = int(self.config.rotate_degrees or 0) % 360
        if deg == 0:
            return frame
        if deg == 180:
            return cv2.rotate(frame, cv2.ROTATE_180)
        if deg == 90:
            return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        if deg == 270:
            return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return frame

    def read(self) -> np.ndarray:
        """
        Return a good BGR frame.

        Strategy for intermittent MJPEG corruption on Pi USB cams:
        1) Skip bad frames cheaply (no reopen)
        2) Briefly reuse last good frame while skipping
        3) Only reopen the device after a long consecutive bad streak
        """
        if self._cap is None:
            self.open()
        assert self._cap is not None

        skip_budget = max(1, int(self.config.skip_before_reopen))
        for _ in range(skip_budget):
            grabbed, frame = self._cap.read()
            if grabbed and is_frame_ok(frame, self.config.width, self.config.height):
                self._bad_streak = 0
                oriented = self._orient(frame)
                self._last_good = oriented
                return oriented
            self._bad_streak += 1
            self._corrupt_skips += 1
            # Prefer holding last good frame over tearing down USB for a blip.
            if self._last_good is not None and self._bad_streak < self.config.max_consecutive_bad_frames:
                if self._bad_streak == 1 or self._bad_streak % 8 == 0:
                    print(
                        f"camera: skipped corrupt frame "
                        f"(streak={self._bad_streak}, total_skips={self._corrupt_skips})",
                        flush=True,
                    )
                return self._last_good
            time.sleep(0.005)

        recover_attempts = 4
        for recover_i in range(recover_attempts):
            print(
                f"camera: bad streak={self._bad_streak} — "
                f"reopen attempt {recover_i + 1}/{recover_attempts}",
                flush=True,
            )
            try:
                self._recover(backoff_sec=self.config.reopen_sleep_sec * (1 + recover_i))
            except RuntimeError as exc:
                print(f"camera: reopen failed: {exc}", flush=True)
                if self._last_good is not None:
                    return self._last_good
                time.sleep(self.config.reopen_sleep_sec * (1 + recover_i))
                continue
            assert self._cap is not None
            # Drain a few frames after reopen — first ones are often junk.
            for _ in range(5):
                grabbed, frame = self._cap.read()
                if grabbed and is_frame_ok(frame, self.config.width, self.config.height):
                    self._bad_streak = 0
                    oriented = self._orient(frame)
                    self._last_good = oriented
                    print("camera: recovered", flush=True)
                    return oriented
                time.sleep(0.01)
            if self._last_good is not None:
                print("camera: reopen still noisy — using last good frame", flush=True)
                return self._last_good

        raise RuntimeError(
            f"Camera index {self._active_index} delivering corrupt frames "
            f"(bad_streak={self._bad_streak}, skips={self._corrupt_skips}) "
            f"after {recover_attempts} reopen attempts"
        )

    def _recover(self, *, backoff_sec: Optional[float] = None) -> None:
        """Release and re-open with backoff (prevents CPU lockup on USB drop)."""
        idx = self._active_index
        self.close()
        time.sleep(float(backoff_sec if backoff_sec is not None else self.config.reopen_sleep_sec))
        if idx is not None:
            # Prefer the last good index first
            self.config.device_indices = [idx] + [i for i in self.config.device_indices if i != idx]
        self.open()

    def close(self) -> None:
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
        self._cap = None
        # Keep _last_good across soft recoveries; cleared only when caller replaces Camera.

    def __enter__(self) -> "Camera":
        self.open()
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def negotiated(self) -> Tuple[int, int, float]:
        """Return (width, height, fps) reported by the driver after open."""
        if self._cap is None:
            raise RuntimeError("Camera not open")
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(self._cap.get(cv2.CAP_PROP_FPS))
        return w, h, fps
