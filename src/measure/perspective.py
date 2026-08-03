"""Homography warp from photo mat corners to a top-down mat plane."""

from __future__ import annotations

import numpy as np
import cv2

from .mat import MatConfig, order_corners


def warp_to_mat_plane(
    image: np.ndarray,
    corners: np.ndarray,
    config: MatConfig,
) -> tuple[np.ndarray, float]:
    """
    Warp `image` so the mat fills a top-down rectangle of known size.

    Returns (warped_bgr, px_per_cm).
    """
    ordered = order_corners(np.asarray(corners, dtype=np.float32))
    width_px = int(round(config.width_cm * config.px_per_cm))
    height_px = int(round(config.height_cm * config.px_per_cm))
    if width_px < 1 or height_px < 1:
        raise ValueError("Warped mat dimensions must be positive")

    dst = np.array(
        [
            [0.0, 0.0],
            [width_px - 1.0, 0.0],
            [width_px - 1.0, height_px - 1.0],
            [0.0, height_px - 1.0],
        ],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(ordered, dst)
    warped = cv2.warpPerspective(image, matrix, (width_px, height_px))
    return warped, float(config.px_per_cm)
