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
    max_consecutive_bad_frames: int = 8
    reopen_sleep_sec: float = 0.35
    read_timeout_retries: int = 3
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
        max_consecutive_bad_frames=int(raw.get("max_consecutive_bad_frames", 8)),
        reopen_sleep_sec=float(raw.get("reopen_sleep_sec", 0.35)),
        read_timeout_retries=int(raw.get("read_timeout_retries", 3)),
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
        "model": cfg.model,
        "camera_matrix": cfg.camera_matrix.tolist() if cfg.camera_matrix is not None else None,
        "dist_coeffs": cfg.dist_coeffs.tolist() if cfg.dist_coeffs is not None else None,
        "image_size": list(cfg.image_size) if cfg.image_size is not None else None,
    }
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, default_flow_style=False, sort_keys=False)


def is_frame_ok(frame: Optional[np.ndarray], expect_w: int, expect_h: int) -> bool:
    """Reject None, wrong shape, empty, or classic USB 'corrupt' frames."""
    if frame is None:
        return False
    if not isinstance(frame, np.ndarray):
        return False
    if frame.ndim != 3 or frame.shape[2] != 3:
        return False
    h, w = frame.shape[:2]
    if h < 16 or w < 16:
        return False
    # Allow slight mismatch if driver renegotiated, but flag huge mismatches
    if expect_w > 0 and expect_h > 0:
        if abs(w - expect_w) > expect_w * 0.25 or abs(h - expect_h) > expect_h * 0.25:
            return False
    # All-black / all-green / near-constant = common corrupt MJPEG decode
    mean = float(frame.mean())
    std = float(frame.std())
    if std < 2.0:
        return False
    if mean < 1.5 or mean > 253.0:
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
            # Warm up + validate a few frames
            ok = False
            for _ in range(max(5, self.config.read_timeout_retries)):
                grabbed, frame = cap.read()
                if grabbed and is_frame_ok(frame, self.config.width, self.config.height):
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

    def read(self) -> np.ndarray:
        """
        Return a good BGR frame. Re-opens the device after repeated bad frames.
        """
        if self._cap is None:
            self.open()
        assert self._cap is not None

        for attempt in range(self.config.read_timeout_retries):
            grabbed, frame = self._cap.read()
            if grabbed and is_frame_ok(frame, self.config.width, self.config.height):
                self._bad_streak = 0
                return frame
            self._bad_streak += 1
            time.sleep(0.01)

            if self._bad_streak >= self.config.max_consecutive_bad_frames:
                self._recover()
                assert self._cap is not None
                grabbed, frame = self._cap.read()
                if grabbed and is_frame_ok(frame, self.config.width, self.config.height):
                    self._bad_streak = 0
                    return frame

        # Last chance: recover once more then fail clearly
        self._recover()
        assert self._cap is not None
        grabbed, frame = self._cap.read()
        if grabbed and is_frame_ok(frame, self.config.width, self.config.height):
            self._bad_streak = 0
            return frame
        raise RuntimeError(
            f"Camera index {self._active_index} delivering corrupt frames "
            f"(bad_streak={self._bad_streak})"
        )

    def _recover(self) -> None:
        """Release and re-open with backoff (prevents CPU lockup on USB drop)."""
        idx = self._active_index
        self.close()
        time.sleep(self.config.reopen_sleep_sec)
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
