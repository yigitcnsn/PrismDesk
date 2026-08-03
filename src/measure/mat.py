"""Mat config loading, black-mat detection, and manual corner fallback."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml


@dataclass(frozen=True)
class MatConfig:
    width_cm: float
    height_cm: float
    px_per_cm: float
    v_max: int
    min_area_ratio: float
    max_area_ratio: float
    aspect_tolerance: float
    morph_kernel: int
    detect_max_dim: int
    # Object auto-find on warped mat
    object_border_margin_px: int
    object_min_area_ratio: float
    object_max_area_ratio: float
    object_approx_epsilon: float
    object_v_min: int


def load_mat_config(path: str | Path) -> MatConfig:
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return MatConfig(
        width_cm=float(raw.get("width_cm", 40.0)),
        height_cm=float(raw.get("height_cm", 30.0)),
        px_per_cm=float(raw.get("px_per_cm", 40.0)),
        v_max=int(raw.get("v_max", 70)),
        min_area_ratio=float(raw.get("min_area_ratio", 0.02)),
        max_area_ratio=float(raw.get("max_area_ratio", 0.85)),
        aspect_tolerance=float(raw.get("aspect_tolerance", 0.55)),
        morph_kernel=int(raw.get("morph_kernel", 7)),
        detect_max_dim=int(raw.get("detect_max_dim", 1280)),
        object_border_margin_px=int(raw.get("object_border_margin_px", 40)),
        object_min_area_ratio=float(raw.get("object_min_area_ratio", 0.005)),
        object_max_area_ratio=float(raw.get("object_max_area_ratio", 0.55)),
        object_approx_epsilon=float(raw.get("object_approx_epsilon", 0.02)),
        object_v_min=int(raw.get("object_v_min", 45)),
    )


def order_corners(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as TL, TR, BR, BL."""
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    ordered = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)
    ordered[0] = pts[np.argmin(s)]  # TL
    ordered[2] = pts[np.argmax(s)]  # BR
    ordered[1] = pts[np.argmin(diff)]  # TR
    ordered[3] = pts[np.argmax(diff)]  # BL
    return ordered


def _resize_for_detect(image: np.ndarray, max_dim: int) -> tuple[np.ndarray, float]:
    h, w = image.shape[:2]
    longest = max(h, w)
    if longest <= max_dim:
        return image, 1.0
    scale = max_dim / float(longest)
    resized = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return resized, scale


def _sample_edge_contrast(
    gray: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    samples: int = 24,
    offset: float = 6.0,
) -> float:
    """
    Score one side: mat interior should be darker than exterior just outside the edge.
    Positive = dark inside / bright outside (good mat edge).
    """
    h, w = gray.shape[:2]
    edge = b.astype(np.float32) - a.astype(np.float32)
    length = float(np.linalg.norm(edge))
    if length < 1e-3:
        return -1e9
    tangent = edge / length
    # Outward normal (rotate 90°); sign resolved by comparing means
    normal = np.array([-tangent[1], tangent[0]], dtype=np.float32)

    ts = np.linspace(0.1, 0.9, samples, dtype=np.float32)
    pts = a.astype(np.float32)[None, :] + ts[:, None] * edge[None, :]
    inside = pts - normal * offset
    outside = pts + normal * offset

    def gather(points: np.ndarray) -> np.ndarray:
        xs = np.clip(np.round(points[:, 0]).astype(np.int32), 0, w - 1)
        ys = np.clip(np.round(points[:, 1]).astype(np.int32), 0, h - 1)
        return gray[ys, xs].astype(np.float32)

    vin = gather(inside)
    vout = gather(outside)
    # Pick normal orientation that puts darker side "inside"
    if float(vin.mean()) > float(vout.mean()):
        vin, vout = vout, vin
    return float(vout.mean() - vin.mean())


def _quad_edge_score(gray: np.ndarray, quad: np.ndarray) -> float:
    ordered = order_corners(quad)
    scores = []
    for i in range(4):
        scores.append(_sample_edge_contrast(gray, ordered[i], ordered[(i + 1) % 4]))
    if min(scores) < 8.0:
        # Reject if any side lacks a clear dark→light transition
        return -1e9
    return float(sum(scores))


def _quad_aspect_error(quad: np.ndarray, target_aspect: float) -> float:
    """
    Perspective-tolerant aspect error.

    Uses average opposite-side lengths (more stable than TL-TR / TL-BL under skew).
    Also considers swapped orientation.
    """
    ordered = order_corners(quad)
    top = float(np.linalg.norm(ordered[1] - ordered[0]))
    bottom = float(np.linalg.norm(ordered[2] - ordered[3]))
    left = float(np.linalg.norm(ordered[3] - ordered[0]))
    right = float(np.linalg.norm(ordered[2] - ordered[1]))
    width = 0.5 * (top + bottom)
    height = 0.5 * (left + right)
    if width < 1e-3 or height < 1e-3:
        return 1e9
    aspect = width / height
    aspect_alt = height / width
    return float(
        min(
            abs(aspect - target_aspect) / target_aspect,
            abs(aspect_alt - target_aspect) / target_aspect,
        )
    )


def _interior_dark_ratio(mask: np.ndarray, quad: np.ndarray) -> float:
    ordered = order_corners(quad).astype(np.int32)
    poly_mask = np.zeros(mask.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(poly_mask, ordered, 255)
    area = float(np.count_nonzero(poly_mask))
    if area < 1:
        return 0.0
    return float(np.count_nonzero(cv2.bitwise_and(mask, poly_mask))) / area


def _line_from_points(p1: np.ndarray, p2: np.ndarray) -> np.ndarray | None:
    """Ax + By + C = 0 from two points."""
    a = float(p1[1] - p2[1])
    b = float(p2[0] - p1[0])
    c = float(p1[0] * p2[1] - p2[0] * p1[1])
    norm = (a * a + b * b) ** 0.5
    if norm < 1e-6:
        return None
    return np.array([a / norm, b / norm, c / norm], dtype=np.float32)


def _intersect_lines(l1: np.ndarray, l2: np.ndarray) -> np.ndarray | None:
    a1, b1, c1 = l1
    a2, b2, c2 = l2
    det = a1 * b2 - a2 * b1
    if abs(det) < 1e-6:
        return None
    x = (b1 * c2 - b2 * c1) / det
    y = (c1 * a2 - c2 * a1) / det
    return np.array([x, y], dtype=np.float32)


def _quad_from_hough(mask: np.ndarray, contour: np.ndarray) -> list[np.ndarray]:
    """
    Fit a perspective quad by intersecting 4 dominant edge lines (Hough).
    Helps under strong foreshortening where min-area rect is a bad fit.
    """
    x, y, bw, bh = cv2.boundingRect(contour)
    pad = 8
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(mask.shape[1], x + bw + pad)
    y1 = min(mask.shape[0], y + bh + pad)
    roi = mask[y0:y1, x0:x1]
    edges = cv2.Canny(roi, 40, 120)
    min_len = max(30, int(0.25 * min(bw, bh)))
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=40, minLineLength=min_len, maxLineGap=20)
    if lines is None or len(lines) < 4:
        return []

    # Represent each segment as angle + line equation
    segs = []
    for raw in lines:
        line = raw.reshape(-1)
        x1, y1, x2, y2 = map(float, line[:4])
        p1 = np.array([x1 + x0, y1 + y0], dtype=np.float32)
        p2 = np.array([x2 + x0, y2 + y0], dtype=np.float32)
        length = float(np.linalg.norm(p2 - p1))
        angle = float(np.arctan2(p2[1] - p1[1], p2[0] - p1[0])) % np.pi
        eq = _line_from_points(p1, p2)
        if eq is None:
            continue
        segs.append((angle, length, eq, 0.5 * (p1 + p2)))

    if len(segs) < 4:
        return []

    # Cluster into up to 4 angle groups (opposite sides share angle)
    segs.sort(key=lambda s: s[0])
    clusters: list[list] = []
    for seg in segs:
        placed = False
        for cluster in clusters:
            if abs(seg[0] - cluster[0][0]) < np.deg2rad(18) or abs(abs(seg[0] - cluster[0][0]) - np.pi) < np.deg2rad(18):
                cluster.append(seg)
                placed = True
                break
        if not placed:
            clusters.append([seg])
    clusters = sorted(clusters, key=lambda c: sum(s[1] for s in c), reverse=True)[:4]
    if len(clusters) < 2:
        return []

    # From the two strongest orientation clusters, pick two parallel lines each (farthest apart)
    def two_sides(cluster: list) -> list[np.ndarray]:
        # Rank by length, keep diverse midpoints
        cluster = sorted(cluster, key=lambda s: s[1], reverse=True)
        primary = cluster[0]
        best = None
        best_dist = -1.0
        for other in cluster[1:]:
            # Distance between parallel lines ~ |C1 - C2| for normalized (A,B)
            dist = abs(float(primary[2][2] - other[2][2]))
            if dist > best_dist:
                best_dist = dist
                best = other
        if best is None:
            return [primary[2]]
        return [primary[2], best[2]]

    side_lines: list[np.ndarray] = []
    for cluster in clusters[:2]:
        side_lines.extend(two_sides(cluster))
    if len(side_lines) < 4 and len(clusters) >= 3:
        side_lines.extend(two_sides(clusters[2])[: 4 - len(side_lines)])
    if len(side_lines) < 4:
        return []

    # Intersect consecutive orientation pairs: assume lines 0,1 one orientation; 2,3 the other
    if len(side_lines) >= 4:
        l_a, l_b, l_c, l_d = side_lines[:4]
        corners = []
        for la in (l_a, l_b):
            for lb in (l_c, l_d):
                pt = _intersect_lines(la, lb)
                if pt is not None:
                    corners.append(pt)
        if len(corners) == 4:
            return [np.asarray(corners, dtype=np.float32)]
    return []


def _candidate_quads(contour: np.ndarray, mask: np.ndarray | None = None) -> list[np.ndarray]:
    """Prefer true perspective quads (approxPolyDP / Hough); min-area rect last."""
    quads: list[np.ndarray] = []
    peri = cv2.arcLength(contour, True)
    for eps in (0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.06, 0.08):
        approx = cv2.approxPolyDP(contour, eps * peri, True)
        if len(approx) == 4:
            quads.append(approx.reshape(4, 2).astype(np.float32))
    hull = cv2.convexHull(contour)
    peri_h = cv2.arcLength(hull, True)
    for eps in (0.015, 0.02, 0.03, 0.04, 0.05, 0.07):
        approx = cv2.approxPolyDP(hull, eps * peri_h, True)
        if len(approx) == 4:
            quads.append(approx.reshape(4, 2).astype(np.float32))
        elif len(approx) > 4:
            pts = approx.reshape(-1, 2).astype(np.float32)
            center = pts.mean(axis=0)
            dists = np.linalg.norm(pts - center, axis=1)
            idx = np.argsort(dists)[-4:]
            quads.append(pts[idx])
    if mask is not None:
        quads.extend(_quad_from_hough(mask, contour))
    box = cv2.boxPoints(cv2.minAreaRect(contour))
    quads.append(np.asarray(box, dtype=np.float32))
    return quads


def _refine_corners_on_edges(gray: np.ndarray, quad: np.ndarray, search: int = 12) -> np.ndarray:
    """Nudge each corner along normals of adjacent sides toward strongest intensity step."""
    ordered = order_corners(quad)
    refined = ordered.copy()
    h, w = gray.shape[:2]
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    for i in range(4):
        prev_pt = ordered[(i - 1) % 4]
        curr = ordered[i]
        next_pt = ordered[(i + 1) % 4]
        v1 = curr - prev_pt
        v2 = next_pt - curr
        n1 = np.array([-v1[1], v1[0]], dtype=np.float32)
        n2 = np.array([-v2[1], v2[0]], dtype=np.float32)
        for n in (n1, n2):
            norm = float(np.linalg.norm(n))
            if norm > 1e-3:
                n /= norm
        # Bisector pointing roughly outward from quad center
        center = ordered.mean(axis=0)
        outward = curr - center
        outward /= max(float(np.linalg.norm(outward)), 1e-3)
        best_pt = curr
        best_score = -1e9
        for delta in range(-search, search + 1):
            pt = curr + outward * float(delta)
            x, y = int(round(pt[0])), int(round(pt[1]))
            if not (1 <= x < w - 1 and 1 <= y < h - 1):
                continue
            # Local gradient magnitude as edge strength
            gx = float(blur[y, min(x + 1, w - 1)]) - float(blur[y, max(x - 1, 0)])
            gy = float(blur[min(y + 1, h - 1), x]) - float(blur[max(y - 1, 0), x])
            score = abs(gx) + abs(gy)
            if score > best_score:
                best_score = score
                best_pt = np.array([x, y], dtype=np.float32)
        refined[i] = best_pt
    return order_corners(refined)


def detect_mat_corners(image: np.ndarray, config: MatConfig) -> np.ndarray | None:
    """
    Find the black mat as a perspective quadrilateral on a crowded tabletop.

    Uses multi-threshold dark masks, prefers approxPolyDP quads over min-area
    rectangles, and ranks candidates by edge contrast (dark inside / bright outside)
    so skewed camera angles still lock onto the true mat edges.
    """
    small, scale = _resize_for_detect(image, config.detect_max_dim)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    value = hsv[:, :, 2]
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    k = max(3, int(config.morph_kernel) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    img_area = float(small.shape[0] * small.shape[1])
    target_aspect = config.width_cm / config.height_cm
    best: tuple[float, np.ndarray] | None = None

    base = int(config.v_max)
    v_candidates = sorted({max(15, base + d) for d in (-30, -20, -10, 0, 10, 20, 30, 40, 50)})

    for v_max in v_candidates:
        mask = cv2.inRange(value, 0, v_max)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < config.min_area_ratio * img_area:
                continue
            if area > config.max_area_ratio * img_area:
                continue

            hull = cv2.convexHull(contour)
            hull_area = float(cv2.contourArea(hull)) or 1.0
            solidity = area / hull_area
            if solidity < 0.70:
                continue

            for quad in _candidate_quads(contour, mask):
                ordered = order_corners(quad)
                aspect_err = _quad_aspect_error(ordered, target_aspect)
                # Soft gate only — perspective foreshortening inflates image aspect error
                if aspect_err > config.aspect_tolerance:
                    continue

                edge_score = _quad_edge_score(gray, ordered)
                if edge_score < 0:
                    continue

                fill = _interior_dark_ratio(mask, ordered)
                if fill < 0.55:
                    continue

                # Prefer strong edges + dark fill; mild area/aspect terms
                score = (
                    edge_score * 10.0
                    + fill * 40.0
                    + solidity * 10.0
                    + (area / img_area) * 20.0
                    - aspect_err * 15.0
                )
                if best is None or score > best[0]:
                    refined = _refine_corners_on_edges(gray, ordered)
                    # Keep refined only if it still scores well
                    refined_score = _quad_edge_score(gray, refined)
                    chosen = refined if refined_score >= edge_score * 0.85 else ordered
                    best = (score, chosen)

    if best is None:
        return None

    corners = best[1] / scale
    return order_corners(corners)


def click_mat_corners(image: np.ndarray, window_name: str = "Select mat corners") -> np.ndarray | None:
    """
    Interactive fallback: click 4 mat corners (any order), then Enter to accept.

    Keys: u undo, r reset, Enter accept, q/Esc cancel.
    """
    clone = image.copy()
    display_scale = 1.0
    max_dim = 1400
    h, w = clone.shape[:2]
    if max(h, w) > max_dim:
        display_scale = max_dim / float(max(h, w))
        clone = cv2.resize(clone, (int(w * display_scale), int(h * display_scale)))

    points: list[tuple[int, int]] = []

    def on_mouse(event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((x, y))

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, on_mouse)

    while True:
        frame = clone.copy()
        for i, (px, py) in enumerate(points):
            cv2.circle(frame, (px, py), 6, (0, 255, 255), -1)
            cv2.putText(
                frame,
                str(i + 1),
                (px + 8, py - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )
        if len(points) == 4:
            ordered = order_corners(np.array(points, dtype=np.float32))
            cv2.polylines(frame, [ordered.astype(np.int32)], True, (0, 255, 0), 2)

        hint = "Click 4 mat corners | u undo | r reset | Enter accept | q cancel"
        cv2.putText(frame, hint, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow(window_name, frame)
        key = cv2.waitKey(20) & 0xFF
        if key in (ord("q"), 27):
            cv2.destroyWindow(window_name)
            return None
        if key == ord("u") and points:
            points.pop()
        if key == ord("r"):
            points.clear()
        if key in (13, 10) and len(points) == 4:
            cv2.destroyWindow(window_name)
            pts = np.array(points, dtype=np.float32) / display_scale
            return order_corners(pts)


def confirm_or_override_corners(
    image: np.ndarray,
    corners: np.ndarray | None,
    window_name: str = "Confirm mat",
) -> np.ndarray | None:
    """
    Show detected corners. Enter accept, m manual, q cancel.
    If corners is None, jump straight to manual.
    """
    if corners is None:
        return click_mat_corners(image)

    display = image.copy()
    scale = 1.0
    max_dim = 1400
    h, w = display.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        display = cv2.resize(display, (int(w * scale), int(h * scale)))

    pts = (corners * scale).astype(np.int32)
    overlay = display.copy()
    cv2.polylines(overlay, [pts], True, (0, 255, 0), 3)
    for i, (x, y) in enumerate(pts):
        cv2.circle(overlay, (int(x), int(y)), 7, (0, 255, 255), -1)
        cv2.putText(
            overlay,
            str(i + 1),
            (int(x) + 8, int(y) - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )
    cv2.putText(
        overlay,
        "Enter accept | m manual corners | q cancel",
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, overlay)

    while True:
        key = cv2.waitKey(0) & 0xFF
        if key in (13, 10):
            cv2.destroyWindow(window_name)
            return order_corners(corners)
        if key == ord("m"):
            cv2.destroyWindow(window_name)
            return click_mat_corners(image)
        if key in (ord("q"), 27):
            cv2.destroyWindow(window_name)
            return None
