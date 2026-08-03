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

    @property
    def target_aspect(self) -> float:
        return self.width_cm / self.height_cm


def load_mat_config(path: str | Path) -> MatConfig:
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return MatConfig(
        width_cm=float(raw.get("width_cm", 40.0)),
        height_cm=float(raw.get("height_cm", 30.0)),
        px_per_cm=float(raw.get("px_per_cm", 10.0)),
        v_max=int(raw.get("v_max", 60)),
        min_area_ratio=float(raw.get("min_area_ratio", 0.02)),
        max_area_ratio=float(raw.get("max_area_ratio", 0.85)),
        aspect_tolerance=float(raw.get("aspect_tolerance", 0.35)),
        morph_kernel=int(raw.get("morph_kernel", 5)),
        detect_max_dim=int(raw.get("detect_max_dim", 1280)),
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


def _candidate_quads(contour: np.ndarray) -> list[np.ndarray]:
    """Generate plausible 4-corner approximations for a mat-like contour."""
    quads: list[np.ndarray] = []
    peri = cv2.arcLength(contour, True)
    for eps in (0.015, 0.02, 0.03, 0.04, 0.05, 0.06):
        approx = cv2.approxPolyDP(contour, eps * peri, True)
        if len(approx) == 4:
            quads.append(approx.reshape(4, 2).astype(np.float32))
    hull = cv2.convexHull(contour)
    peri_h = cv2.arcLength(hull, True)
    for eps in (0.02, 0.04, 0.06):
        approx = cv2.approxPolyDP(hull, eps * peri_h, True)
        if len(approx) == 4:
            quads.append(approx.reshape(4, 2).astype(np.float32))
    # Rounded mats often need min-area rect as a stable fallback
    box = cv2.boxPoints(cv2.minAreaRect(contour))
    quads.append(np.asarray(box, dtype=np.float32))
    return quads


def detect_mat_corners(image: np.ndarray, config: MatConfig) -> np.ndarray | None:
    """
    Find the black mat as a dark quadrilateral on a crowded tabletop.

    Sweeps several darkness thresholds for robustness under varied lighting.
    Returns ordered corners (TL, TR, BR, BL) in original image coords, or None.
    """
    small, scale = _resize_for_detect(image, config.detect_max_dim)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    value = hsv[:, :, 2]

    k = max(3, int(config.morph_kernel) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    img_area = float(small.shape[0] * small.shape[1])
    target_aspect = config.target_aspect
    best: tuple[float, np.ndarray] | None = None

    # Sweep V thresholds around config.v_max (handles glare / dark tables)
    base = int(config.v_max)
    v_candidates = sorted({max(20, base + d) for d in (-20, -10, 0, 10, 20, 30, 40)})

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
            if solidity < 0.75:
                continue

            for quad in _candidate_quads(contour):
                ordered = order_corners(quad)
                width = float(np.linalg.norm(ordered[1] - ordered[0]))
                height = float(np.linalg.norm(ordered[3] - ordered[0]))
                if height < 1e-3 or width < 1e-3:
                    continue
                aspect = width / height
                aspect_alt = height / width
                err = min(
                    abs(aspect - target_aspect) / target_aspect,
                    abs(aspect_alt - target_aspect) / target_aspect,
                )
                if err > config.aspect_tolerance:
                    continue

                score = area * solidity * (1.0 - err)
                if best is None or score > best[0]:
                    best = (score, ordered)

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
