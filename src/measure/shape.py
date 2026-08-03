"""Shape classification and geometric measurements for object silhouettes."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .geometry import segment_lengths_cm, to_cm


Point = Tuple[float, float]


@dataclass
class ObjectAnalysis:
    """One physical object = one silhouette = one measurement set."""

    shape: str  # circle | polygon | thin
    outline_points: List[Point]
    contour: np.ndarray
    colors: List[str] = field(default_factory=list)
    edge_cm: List[float] = field(default_factory=list)
    radius_cm: Optional[float] = None
    diameter_cm: Optional[float] = None
    fillet_radii_cm: List[float] = field(default_factory=list)
    center: Optional[Point] = None
    length_cm: Optional[float] = None
    width_cm: Optional[float] = None


def analyze_silhouette(
    image_bgr: np.ndarray,
    contour: np.ndarray,
    px_per_cm: float,
) -> ObjectAnalysis:
    """Classify silhouette and compute shape-appropriate measurements."""
    area = float(cv2.contourArea(contour))
    peri = float(cv2.arcLength(contour, True))
    circularity = (4.0 * math.pi * area / (peri * peri)) if peri > 1e-6 else 0.0

    colors = sample_dominant_colors(image_bgr, contour, max_colors=3)

    (cx, cy), radius_px = cv2.minEnclosingCircle(contour)
    circle_area = math.pi * radius_px * radius_px
    area_fill = area / circle_area if circle_area > 1 else 0.0

    rect = cv2.minAreaRect(contour)
    (rw, rh) = rect[1]
    rw = float(max(rw, 1e-3))
    rh = float(max(rh, 1e-3))
    aspect = max(rw, rh) / min(rw, rh)
    box = np.asarray(cv2.boxPoints(rect), dtype=np.float32)
    rectangularity = area / float(abs(cv2.contourArea(box))) if cv2.contourArea(box) > 1 else 0.0

    # Circle / disc (OpenCV docs: minEnclosingCircle + circularity/area fill)
    if circularity >= 0.82 and area_fill >= 0.75:
        r_cm = to_cm(radius_px, px_per_cm)
        pts = _sample_circle_points(cx, cy, radius_px, n=64)
        return ObjectAnalysis(
            shape="circle",
            outline_points=pts,
            contour=contour,
            colors=colors,
            radius_cm=r_cm,
            diameter_cm=2.0 * r_cm,
            center=(float(cx), float(cy)),
        )

    # Thin stick-like objects (pencil)
    if aspect >= 6.0 and rectangularity >= 0.55:
        length_px = max(rw, rh)
        width_px = min(rw, rh)
        ordered = _order_box(box)
        return ObjectAnalysis(
            shape="thin",
            outline_points=ordered,
            contour=contour,
            colors=colors,
            edge_cm=segment_lengths_cm(ordered + [ordered[0]], px_per_cm),
            length_cm=to_cm(length_px, px_per_cm),
            width_cm=to_cm(width_px, px_per_cm),
            center=(float(rect[0][0]), float(rect[0][1])),
        )

    # Polygon / rounded rectangle
    outline = _polygon_outline(contour, rectangularity)
    edge_cm = segment_lengths_cm(outline + [outline[0]], px_per_cm) if len(outline) >= 2 else []
    fillets = estimate_fillet_radii_cm(contour, outline, px_per_cm)
    # Only keep fillets that look like rounded corners (small vs edge length)
    if edge_cm and fillets:
        min_edge = min(edge_cm)
        fillets = [f for f in fillets if 0.05 * min_edge <= f <= 0.22 * min_edge]
    return ObjectAnalysis(
        shape="polygon",
        outline_points=outline,
        contour=contour,
        colors=colors,
        edge_cm=edge_cm,
        fillet_radii_cm=fillets,
        center=(float(cx), float(cy)),
    )


def sample_dominant_colors(
    image_bgr: np.ndarray,
    contour: np.ndarray,
    max_colors: int = 3,
) -> List[str]:
    """
    Sample colors inside the silhouette (metadata only — does not split measurements).

    Uses k-means in Lab; returns human-readable labels with hex.
    """
    h, w = image_bgr.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, thickness=-1)
    # Shrink slightly to avoid mat bleed at the rim
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.erode(mask, k, iterations=1)

    ys, xs = np.where(mask > 0)
    if len(xs) < 50:
        return []

    # Subsample for speed
    idx = np.linspace(0, len(xs) - 1, num=min(2000, len(xs)), dtype=np.int32)
    pixels = image_bgr[ys[idx], xs[idx]].astype(np.float32)

    k_count = int(min(max_colors, max(1, len(pixels) // 100)))
    if k_count < 1:
        return []

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _compactness, labels, centers = cv2.kmeans(
        pixels, k_count, None, criteria, 3, cv2.KMEANS_PP_CENTERS
    )
    labels = labels.reshape(-1)
    counts = np.bincount(labels, minlength=k_count).astype(np.float32)
    order = np.argsort(-counts)

    # Drop tiny clusters (<8%) — usually noise / specular
    total = float(counts.sum()) or 1.0
    results: List[str] = []
    for i in order:
        if counts[i] / total < 0.08:
            continue
        b, g, r = [int(round(v)) for v in centers[i]]
        name = _color_name(r, g, b)
        results.append(f"{name} #{r:02X}{g:02X}{b:02X}")
    return results


def estimate_fillet_radii_cm(
    contour: np.ndarray,
    outline: List[Point],
    px_per_cm: float,
    window_frac: float = 0.12,
) -> List[float]:
    """
    Estimate rounded-corner fillet radii by fitting local circles near each
    polygon vertex (common approach for rounded-rect metrology).
    """
    if len(outline) < 3:
        return []

    pts = contour.reshape(-1, 2).astype(np.float32)
    if len(pts) < 20:
        return []

    peri = float(cv2.arcLength(contour, True))
    window = max(8, int(window_frac * peri / max(len(outline), 1)))
    fillets: List[float] = []

    for corner in outline:
        cxy = np.asarray(corner, dtype=np.float32)
        d = np.linalg.norm(pts - cxy, axis=1)
        nearest = int(np.argmin(d))
        # Take an arc neighborhood around the nearest contour point
        indices = [(nearest + i) % len(pts) for i in range(-window, window + 1)]
        arc = pts[indices]
        fitted = _fit_circle_least_squares(arc)
        if fitted is None:
            continue
        _center, radius_px = fitted
        # Fillet radii are small vs object size; reject absurd fits
        if radius_px < 0.05 * px_per_cm or radius_px > 0.35 * peri:
            continue
        fillets.append(to_cm(radius_px, px_per_cm))

    return fillets


def _polygon_outline(contour: np.ndarray, rectangularity: float) -> List[Point]:
    if rectangularity >= 0.82:
        box = np.asarray(cv2.boxPoints(cv2.minAreaRect(contour)), dtype=np.float32)
        return _order_box(box)

    peri = cv2.arcLength(contour, True)
    best = None
    for eps in (0.01, 0.015, 0.02, 0.03, 0.04, 0.06):
        approx = cv2.approxPolyDP(contour, eps * peri, True)
        if 3 <= len(approx) <= 10:
            best = approx
            if len(approx) <= 6:
                break
    if best is None:
        box = np.asarray(cv2.boxPoints(cv2.minAreaRect(contour)), dtype=np.float32)
        return _order_box(box)
    return [(float(p[0][0]), float(p[0][1])) for p in best]


def _order_box(box: np.ndarray) -> List[Point]:
    pts = np.asarray(box, dtype=np.float32).reshape(4, 2)
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    ordered = pts[np.argsort(angles)]
    start = int(np.argmin(ordered.sum(axis=1)))
    ordered = np.roll(ordered, -start, axis=0)
    return [(float(x), float(y)) for x, y in ordered]


def _sample_circle_points(cx: float, cy: float, radius: float, n: int = 64) -> List[Point]:
    angles = np.linspace(0, 2 * math.pi, n, endpoint=False)
    return [(float(cx + radius * math.cos(a)), float(cy + radius * math.sin(a))) for a in angles]


def _fit_circle_least_squares(points: np.ndarray) -> Optional[Tuple[Point, float]]:
    """Algebraic circle fit (Kåsa). Returns (center, radius) or None."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(pts) < 5:
        return None
    x = pts[:, 0]
    y = pts[:, 1]
    A = np.column_stack([2 * x, 2 * y, np.ones_like(x)])
    b = x * x + y * y
    try:
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None
    cx, cy, c = sol
    r2 = c + cx * cx + cy * cy
    if r2 <= 1.0:
        return None
    return (float(cx), float(cy)), float(math.sqrt(r2))


def _color_name(r: int, g: int, b: int) -> str:
    mx = max(r, g, b)
    mn = min(r, g, b)
    if mx < 40:
        return "black"
    if mn > 210 and (mx - mn) < 30:
        return "white"
    if (mx - mn) < 25:
        return "gray"
    # Hue-ish from RGB
    rf, gf, bf = r / 255.0, g / 255.0, b / 255.0
    hsv = cv2.cvtColor(np.uint8([[[b, g, r]]]), cv2.COLOR_BGR2HSV)[0, 0]
    h = int(hsv[0])
    s = int(hsv[1])
    v = int(hsv[2])
    if s < 40:
        return "gray"
    if v < 50:
        return "black"
    if h < 10 or h >= 170:
        return "red"
    if h < 25:
        return "orange"
    if h < 35:
        return "yellow"
    if h < 85:
        return "green"
    if h < 130:
        return "blue"
    if h < 160:
        return "purple"
    return "red"
