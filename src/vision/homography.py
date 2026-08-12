"""Camera ↔ projector plane homography (replace stretch HUD mapping)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

Point = Tuple[float, float]
Size = Tuple[int, int]


@dataclass
class CamProjectorHomography:
    """Maps camera pixels → projector canvas pixels at calibration sizes."""

    matrix: np.ndarray  # 3x3 float64, cam → proj
    cam_size: Size
    proj_size: Size
    reprojection_error_px: Optional[float] = None

    def composed_for(
        self,
        *,
        src_size: Size,
        hud_size: Size,
    ) -> np.ndarray:
        """3x3 mapping current camera frame → current HUD/projector canvas."""
        cam_w, cam_h = self.cam_size
        proj_w, proj_h = self.proj_size
        src_w, src_h = src_size
        hud_w, hud_h = hud_size
        if min(cam_w, cam_h, proj_w, proj_h, src_w, src_h, hud_w, hud_h) <= 0:
            raise ValueError("sizes must be positive")

        # Scale current cam → calib cam, apply H, scale calib proj → hud.
        s_x = float(cam_w) / float(src_w)
        s_y = float(cam_h) / float(src_h)
        t_x = float(hud_w) / float(proj_w)
        t_y = float(hud_h) / float(proj_h)
        s = np.array([[s_x, 0.0, 0.0], [0.0, s_y, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        t = np.array([[t_x, 0.0, 0.0], [0.0, t_y, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        return t @ self.matrix @ s

    def map_points(
        self,
        points: Sequence[Point],
        *,
        src_size: Size,
        hud_size: Size,
    ) -> List[Tuple[int, int]]:
        if not points:
            return []
        m = self.composed_for(src_size=src_size, hud_size=hud_size)
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
        mapped = cv2.perspectiveTransform(pts, m).reshape(-1, 2)
        return [(int(round(x)), int(round(y))) for x, y in mapped]


def stretch_points(
    points: Sequence[Point],
    *,
    src_size: Size,
    hud_size: Size,
) -> List[Tuple[int, int]]:
    """Legacy camera→HUD stretch (used when no homography is saved)."""
    src_w, src_h = src_size
    hud_w, hud_h = hud_size
    if src_w <= 0 or src_h <= 0:
        return [(int(x), int(y)) for x, y in points]
    return [
        (int(round(x * hud_w / src_w)), int(round(y * hud_h / src_h)))
        for x, y in points
    ]


def map_cam_to_hud(
    points: Sequence[Point],
    *,
    src_size: Size,
    hud_size: Size,
    homography: Optional[CamProjectorHomography] = None,
) -> List[Tuple[int, int]]:
    if homography is None:
        return stretch_points(points, src_size=src_size, hud_size=hud_size)
    return homography.map_points(points, src_size=src_size, hud_size=hud_size)


def make_chessboard_pattern(
    width: int,
    height: int,
    board_size: Tuple[int, int] = (9, 6),
    *,
    margin_frac: float = 0.01,
) -> Tuple[np.ndarray, np.ndarray]:
    """Nearly full-canvas chessboard for projector calib.

    Cells may be rectangular (not square) so the board fills the projector
    frame; homography only needs known projector-pixel corner coords.

    Returns (BGR image, projector-pixel coords of inner corners Nx2).
    Corner order matches OpenCV findChessboardCorners (row-major).
    Clean B/W only — no HUD text (that breaks detection).
    """
    cols, rows = board_size
    if cols < 2 or rows < 2:
        raise ValueError("board_size needs at least 2x2 inner corners")
    squares_x = cols + 1
    squares_y = rows + 1

    # Keep a thin border so edge corners stay inside the projected light field.
    margin_x = max(4, int(round(width * margin_frac)))
    margin_y = max(4, int(round(height * margin_frac)))
    usable_w = max(squares_x, width - 2 * margin_x)
    usable_h = max(squares_y, height - 2 * margin_y)
    # Stretch cells to fill usable area (non-square OK for plane H).
    cell_w = usable_w / float(squares_x)
    cell_h = usable_h / float(squares_y)
    ox = float(margin_x)
    oy = float(margin_y)

    img = np.zeros((height, width, 3), dtype=np.uint8)
    for j in range(squares_y):
        for i in range(squares_x):
            if (i + j) % 2 == 0:
                x0 = int(round(ox + i * cell_w))
                y0 = int(round(oy + j * cell_h))
                x1 = int(round(ox + (i + 1) * cell_w))
                y1 = int(round(oy + (j + 1) * cell_h))
                img[y0:y1, x0:x1] = 255

    proj_corners = np.zeros((rows * cols, 2), dtype=np.float64)
    idx = 0
    for r in range(rows):
        for c in range(cols):
            proj_corners[idx, 0] = ox + (c + 1) * cell_w
            proj_corners[idx, 1] = oy + (r + 1) * cell_h
            idx += 1
    return img, proj_corners


def estimate_homography(
    cam_points: np.ndarray,
    proj_points: np.ndarray,
) -> Tuple[np.ndarray, float, int]:
    """Estimate cam→proj H. Returns (H, mean_reproj_err_px, inliers)."""
    cam = np.asarray(cam_points, dtype=np.float64).reshape(-1, 2)
    proj = np.asarray(proj_points, dtype=np.float64).reshape(-1, 2)
    if cam.shape[0] < 4 or cam.shape != proj.shape:
        raise ValueError("need >=4 matching cam/proj points")
    h, mask = cv2.findHomography(cam, proj, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    if h is None:
        raise RuntimeError("findHomography failed")
    mapped = cv2.perspectiveTransform(cam.reshape(-1, 1, 2), h).reshape(-1, 2)
    err = np.linalg.norm(mapped - proj, axis=1)
    if mask is not None:
        inl = mask.ravel().astype(bool)
        inliers = int(inl.sum())
        mean_err = float(err[inl].mean()) if inliers else float(err.mean())
    else:
        inliers = int(cam.shape[0])
        mean_err = float(err.mean())
    return np.asarray(h, dtype=np.float64), mean_err, inliers


def homography_from_dict(raw: Optional[dict]) -> Optional[CamProjectorHomography]:
    if not raw or not isinstance(raw, dict):
        return None
    matrix = raw.get("cam_to_proj")
    cam_size = raw.get("cam_size")
    proj_size = raw.get("proj_size")
    if matrix is None or cam_size is None or proj_size is None:
        return None
    m = np.asarray(matrix, dtype=np.float64)
    if m.shape != (3, 3):
        return None
    cw, ch = int(cam_size[0]), int(cam_size[1])
    pw, ph = int(proj_size[0]), int(proj_size[1])
    if min(cw, ch, pw, ph) <= 0:
        return None
    err = raw.get("reprojection_error_px")
    return CamProjectorHomography(
        matrix=m,
        cam_size=(cw, ch),
        proj_size=(pw, ph),
        reprojection_error_px=float(err) if err is not None else None,
    )


def homography_to_dict(h: CamProjectorHomography) -> dict:
    payload: dict = {
        "cam_to_proj": h.matrix.tolist(),
        "cam_size": [int(h.cam_size[0]), int(h.cam_size[1])],
        "proj_size": [int(h.proj_size[0]), int(h.proj_size[1])],
    }
    if h.reprojection_error_px is not None:
        payload["reprojection_error_px"] = float(h.reprojection_error_px)
    return payload


def parse_board_size(text: str) -> Tuple[int, int]:
    a, b = str(text).lower().split("x", 1)
    return int(a), int(b)


def _refine_corners(gray: np.ndarray, corners: np.ndarray) -> np.ndarray:
    refined = cv2.cornerSubPix(
        gray,
        corners.reshape(-1, 1, 2).astype(np.float32),
        (11, 11),
        (-1, -1),
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001),
    )
    return refined.reshape(-1, 2).astype(np.float64)


def _try_find_chessboard(
    gray: np.ndarray,
    board_size: Tuple[int, int],
) -> Tuple[bool, Optional[np.ndarray]]:
    cols, rows = board_size
    flags = (
        cv2.CALIB_CB_ADAPTIVE_THRESH
        + cv2.CALIB_CB_NORMALIZE_IMAGE
        + cv2.CALIB_CB_FILTER_QUADS
    )
    found, corners = cv2.findChessboardCorners(gray, (cols, rows), flags)
    if found and corners is not None:
        return True, _refine_corners(gray, corners)

    # SB is slower but much better on soft / projected boards.
    if hasattr(cv2, "findChessboardCornersSB"):
        sb_flags = 0
        for name in ("CALIB_CB_EXHAUSTIVE", "CALIB_CB_ACCURACY"):
            sb_flags |= int(getattr(cv2, name, 0))
        try:
            found, corners = cv2.findChessboardCornersSB(gray, (cols, rows), sb_flags)
            if found and corners is not None:
                return True, corners.reshape(-1, 2).astype(np.float64)
        except cv2.error:
            pass
    return False, None


def find_projected_chessboard(
    frame_bgr: np.ndarray,
    board_size: Tuple[int, int],
) -> Tuple[bool, Optional[np.ndarray]]:
    """Detect projected chessboard corners in a camera frame.

    Tries CLAHE + inverted polarity; prefers findChessboardCornersSB when present.
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    # Boost local contrast (projector wash / uneven desk lighting).
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    for candidate in (enhanced, gray, cv2.bitwise_not(enhanced), cv2.bitwise_not(gray)):
        found, corners = _try_find_chessboard(candidate, board_size)
        if found and corners is not None:
            return True, corners
    return False, None

