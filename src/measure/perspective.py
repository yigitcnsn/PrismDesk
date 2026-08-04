"""Homography warp from photo mat corners to a top-down mat plane."""

from __future__ import annotations

from typing import List, Sequence, Tuple

import cv2
import numpy as np

from .mat import MatConfig, order_corners

Point = Tuple[float, float]


def mat_plane_destination(
    corners: np.ndarray,
    config: MatConfig,
) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    """
    Build camera→mat-plane perspective map.

    Returns (ordered_corners, dst_px, px_per_cm, out_w_cm, out_h_cm).
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
    return ordered, dst, float(config.px_per_cm), float(out_w_cm), float(out_h_cm)


def mat_homography(corners: np.ndarray, config: MatConfig) -> tuple[np.ndarray, float, float, float]:
    """Return (H 3x3 camera→mat-px, px_per_cm, out_w_cm, out_h_cm)."""
    ordered, dst, ppc, w_cm, h_cm = mat_plane_destination(corners, config)
    matrix = cv2.getPerspectiveTransform(ordered, dst)
    return matrix, ppc, w_cm, h_cm


def image_points_to_mat_cm(
    points: Sequence[Point],
    corners: np.ndarray,
    config: MatConfig,
) -> List[Point]:
    """Map image pixel points onto the mat plane in centimetres."""
    if not points:
        return []
    H, ppc, _, _ = mat_homography(corners, config)
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    mapped = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
    return [(float(x) / ppc, float(y) / ppc) for x, y in mapped]


def mat_plane_points_to_image(
    points: Sequence[Point],
    corners: np.ndarray,
    config: MatConfig,
) -> List[Point]:
    """Map mat-plane pixel points back into the camera image."""
    if not points:
        return []
    H, _, _, _ = mat_homography(corners, config)
    ok, H_inv = cv2.invert(H)
    if not ok:
        raise RuntimeError("Failed to invert mat homography")
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    mapped = cv2.perspectiveTransform(pts, H_inv).reshape(-1, 2)
    return [(float(x), float(y)) for x, y in mapped]


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
    ordered, dst, ppc, _, _ = mat_plane_destination(corners, config)
    matrix = cv2.getPerspectiveTransform(ordered, dst)
    width_px = int(dst[1, 0]) + 1
    height_px = int(dst[2, 1]) + 1
    warped = cv2.warpPerspective(image, matrix, (width_px, height_px))
    return warped, ppc
