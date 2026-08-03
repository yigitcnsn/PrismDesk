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

    Chooses whether the longer physical side (width_cm) maps to the image
    horizontal or vertical based on measured opposite-side lengths, so portrait
    and landscape photos keep correct centimetre scale.

    Returns (warped_bgr, px_per_cm).
    """
    ordered = order_corners(np.asarray(corners, dtype=np.float32))
    top = float(np.linalg.norm(ordered[1] - ordered[0]))
    bottom = float(np.linalg.norm(ordered[2] - ordered[3]))
    left = float(np.linalg.norm(ordered[3] - ordered[0]))
    right = float(np.linalg.norm(ordered[2] - ordered[1]))
    horiz = 0.5 * (top + bottom)
    vert = 0.5 * (left + right)

    long_cm = max(config.width_cm, config.height_cm)
    short_cm = min(config.width_cm, config.height_cm)

    # Longer image edge pair ↔ longer physical side
    if horiz >= vert:
        out_w_cm, out_h_cm = long_cm, short_cm
    else:
        out_w_cm, out_h_cm = short_cm, long_cm

    width_px = int(round(out_w_cm * config.px_per_cm))
    height_px = int(round(out_h_cm * config.px_per_cm))
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
