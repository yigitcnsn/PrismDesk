"""Auto-find object silhouettes on a warped top-down mat image.

Pipeline (aligned with common OpenCV measurement projects):
  1. Scan warped mat
  2. Build one silhouette (mat-subtraction + edges + chroma when useful)
  3. Failsafe: if chroma only caught part of a thin multi-material object, use full mat-subtraction
  4. Sample colors inside silhouette (metadata only; adjacent colors stay one object)
  5. Classify shape → edges / circle r+Ø / fillets / thin length×width
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np

from .mat import MatConfig
from .shape import ObjectAnalysis, analyze_silhouette


Point = Tuple[float, float]


def detect_object_outline(
    warped: np.ndarray,
    config: MatConfig,
) -> Optional[List[Point]]:
    """Back-compat: return simplified outline points only."""
    analysis = analyze_object(warped, config)
    if analysis is None:
        return None
    return list(analysis.outline_points)


def analyze_object(
    warped: np.ndarray,
    config: MatConfig,
) -> Optional[ObjectAnalysis]:
    """Full pipeline: silhouette → colors → shape measurements."""
    contour = find_object_silhouette(warped, config)
    if contour is None:
        return None
    return analyze_silhouette(warped, contour, float(config.px_per_cm))


def find_object_silhouette(
    warped: np.ndarray,
    config: MatConfig,
) -> Optional[np.ndarray]:
    """
    One outer silhouette for the dominant object on the mat.

    Combines:
    - Lab distance from mat border (works for multi-color / metal+plastic)
    - Lab chroma vs mat (tight on solid colorful objects like coaster/box)
    - Canny edges (helps low-chroma metallic segments)

    Failsafe: never keep a short chroma stub when mat-subtraction finds a longer
    thin object that contains it (classic pencil / multi-material case).
    """
    h, w = warped.shape[:2]
    margin = max(2, int(config.object_border_margin_px))
    blur = cv2.GaussianBlur(warped, (5, 5), 0)
    lab = cv2.cvtColor(blur, cv2.COLOR_BGR2LAB).astype(np.float32)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    img_area = float(h * w)

    sample = _border_sample_mask(h, w, margin)
    samples = lab[sample]
    if samples.size == 0:
        return None

    med = np.median(samples, axis=0)
    mad = np.maximum(np.median(np.abs(samples - med), axis=0), 1.5)
    dist = np.sqrt(np.sum(((lab - med) / mad) ** 2, axis=2))
    border_dist = dist[sample]
    med_d = float(np.median(border_dist))
    mad_d = float(np.median(np.abs(border_dist - med_d)))
    dist_thr = max(med_d + max(4.0, 5.0 * max(mad_d, 0.5)), float(config.object_v_min) / 8.0)

    chroma = np.sqrt((lab[:, :, 1] - med[1]) ** 2 + (lab[:, :, 2] - med[2]) ** 2)
    border_ch = chroma[sample]
    ch_med = float(np.median(border_ch))
    ch_mad = float(np.median(np.abs(border_ch - ch_med)))
    chroma_thr = ch_med + max(12.0, 8.0 * max(ch_mad, 0.5))

    mask_dist = (dist >= dist_thr).astype(np.uint8) * 255
    mask_chroma = (chroma >= chroma_thr).astype(np.uint8) * 255

    edges = cv2.Canny(gray, 40, 120)
    edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
    mask_edge = cv2.bitwise_and(edges, mask_dist)
    mask_edge = cv2.morphologyEx(
        mask_edge,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
        iterations=2,
    )

    # Full multi-material candidate: dist + edges, then bridge small gaps
    mask_full = cv2.bitwise_or(mask_dist, mask_edge)
    mask_full = _clear_border(mask_full, margin)
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_full = cv2.morphologyEx(mask_full, cv2.MORPH_OPEN, k5, iterations=1)
    mask_full = cv2.morphologyEx(mask_full, cv2.MORPH_CLOSE, k5, iterations=2)
    # Extra close helps join metal tip/grip to plastic barrel
    k_bridge = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask_full = cv2.morphologyEx(mask_full, cv2.MORPH_CLOSE, k_bridge, iterations=2)

    mask_ch = _clear_border(mask_chroma.copy(), margin)
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_ch = cv2.morphologyEx(mask_ch, cv2.MORPH_OPEN, k3, iterations=1)
    mask_ch = cv2.morphologyEx(mask_ch, cv2.MORPH_CLOSE, k3, iterations=1)

    chroma_c = _largest_valid_contour(mask_ch, config, img_area)
    full_c = _largest_valid_contour(mask_full, config, img_area)

    chosen = _select_silhouette(chroma_c, full_c, config.px_per_cm)
    return chosen


def _select_silhouette(
    chroma_c: Optional[np.ndarray],
    full_c: Optional[np.ndarray],
    px_per_cm: float,
) -> Optional[np.ndarray]:
    """
    Choose chroma (tight colorful objects) vs full mat-subtraction silhouette.

    Failsafe: if chroma is only a fragment of a longer thin full silhouette
    (pencil barrel vs whole pencil), keep the full silhouette.
    """
    if chroma_c is None and full_c is None:
        return None
    if chroma_c is None:
        return full_c
    if full_c is None:
        return chroma_c

    ch_len, ch_wid, ch_aspect = _min_area_dims(chroma_c)
    full_len, full_wid, full_aspect = _min_area_dims(full_c)
    ch_area = float(cv2.contourArea(chroma_c))
    full_area = float(cv2.contourArea(full_c))

    # Compact colorful object (coaster/box): chroma is usually better
    chroma_compact = ch_aspect < 4.0 and ch_area >= 0.45 * full_area

    # Thin multi-material failsafe
    full_is_thin = full_aspect >= 4.0
    chroma_is_fragment = (
        full_is_thin
        and full_len > ch_len * 1.25  # meaningfully longer
        and _contour_center_inside(chroma_c, full_c)
        and ch_area < 0.85 * full_area
    )

    # Also: chroma itself thin but much shorter than full thin object
    chroma_thin_stub = (
        ch_aspect >= 4.0
        and full_is_thin
        and full_len > ch_len * 1.2
        and _contour_center_inside(chroma_c, full_c)
    )

    if chroma_thin_stub or chroma_is_fragment:
        return full_c

    if chroma_compact:
        return chroma_c

    # Default: larger plausible silhouette wins (still one object)
    if full_area >= ch_area * 0.9:
        # Prefer full when similar size — safer for multi-color
        # unless chroma is clearly tighter rectangle for a compact object
        if ch_aspect < 3.0 and full_aspect < 3.0 and ch_area >= 0.7 * full_area:
            return chroma_c
        return full_c
    return chroma_c


def _min_area_dims(contour: np.ndarray) -> Tuple[float, float, float]:
    (_cx, _cy), (rw, rh), _angle = cv2.minAreaRect(contour)
    length = float(max(rw, rh))
    width = float(max(min(rw, rh), 1e-3))
    return length, width, length / width


def _contour_center_inside(inner: np.ndarray, outer: np.ndarray) -> bool:
    """True if centroid of inner lies inside outer contour."""
    m = cv2.moments(inner)
    if abs(m["m00"]) < 1e-6:
        return False
    cx = m["m10"] / m["m00"]
    cy = m["m01"] / m["m00"]
    return cv2.pointPolygonTest(outer, (float(cx), float(cy)), False) >= 0


def _clear_border(mask: np.ndarray, margin: int) -> np.ndarray:
    clear = max(margin, 6)
    mask = mask.copy()
    mask[:clear, :] = 0
    mask[-clear:, :] = 0
    mask[:, :clear] = 0
    mask[:, -clear:] = 0
    return mask


def _border_sample_mask(h: int, w: int, margin: int) -> np.ndarray:
    border = np.zeros((h, w), dtype=bool)
    border[:margin, :] = True
    border[-margin:, :] = True
    border[:, :margin] = True
    border[:, -margin:] = True
    m2 = margin * 2
    if m2 * 2 < min(h, w):
        ring = np.zeros((h, w), dtype=bool)
        ring[margin:m2, margin : w - margin] = True
        ring[-m2:-margin, margin : w - margin] = True
        ring[margin : h - margin, margin:m2] = True
        ring[margin : h - margin, -m2:-margin] = True
        return border | ring
    return border


def _largest_valid_contour(
    mask: np.ndarray,
    config: MatConfig,
    img_area: float,
) -> Optional[np.ndarray]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    best_area = 0.0
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < config.object_min_area_ratio * img_area:
            continue
        if area > config.object_max_area_ratio * img_area:
            continue
        if area > best_area:
            best_area = area
            best = contour
    return best
