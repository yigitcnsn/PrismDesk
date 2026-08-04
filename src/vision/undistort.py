"""Undistort helpers for fisheye / pinhole USB cameras."""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from .camera import CameraConfig


class Undistorter:
    """Lazy remap tables from CameraConfig calibration."""

    def __init__(self, config: CameraConfig) -> None:
        self.config = config
        self._map1: Optional[np.ndarray] = None
        self._map2: Optional[np.ndarray] = None
        self._size: Optional[Tuple[int, int]] = None

    @property
    def enabled(self) -> bool:
        return (
            self.config.camera_matrix is not None
            and self.config.dist_coeffs is not None
            and self.config.camera_matrix.size > 0
        )

    def apply(self, frame: np.ndarray) -> np.ndarray:
        if not self.enabled:
            return frame
        h, w = frame.shape[:2]
        if self._map1 is None or self._size != (w, h):
            self._build_maps(w, h)
        assert self._map1 is not None and self._map2 is not None
        return cv2.remap(frame, self._map1, self._map2, interpolation=cv2.INTER_LINEAR)

    def _build_maps(self, width: int, height: int) -> None:
        k = np.asarray(self.config.camera_matrix, dtype=np.float64)
        d = np.asarray(self.config.dist_coeffs, dtype=np.float64).reshape(-1, 1)
        size = (width, height)
        model = (self.config.model or "fisheye").lower()
        if model == "fisheye":
            # OpenCV fisheye expects 4 distortion coeffs
            d4 = np.zeros((4, 1), dtype=np.float64)
            n = min(4, d.size)
            d4[:n, 0] = d.reshape(-1)[:n]
            new_k = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                k, d4, size, np.eye(3), balance=0.0
            )
            self._map1, self._map2 = cv2.fisheye.initUndistortRectifyMap(
                k, d4, np.eye(3), new_k, size, cv2.CV_16SC2
            )
        else:
            new_k, _ = cv2.getOptimalNewCameraMatrix(k, d, size, 0, size)
            self._map1, self._map2 = cv2.initUndistortRectifyMap(
                k, d, None, new_k, size, cv2.CV_16SC2
            )
        self._size = size
