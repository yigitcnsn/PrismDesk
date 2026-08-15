"""Live desk HUD: mat outline + object measure on the projector canvas."""

from __future__ import annotations

from datetime import datetime
from typing import Optional, Tuple

import cv2
import numpy as np

from src.core.home_hub import OverlayFlags
from src.measure.mat import MatConfig, order_corners
from src.measure.perspective import mat_plane_points_to_image
from src.measure.shape import ObjectAnalysis
from src.vision.homography import CamProjectorHomography, map_cam_to_hud

Point = Tuple[float, float]

# Dim cyan (BGR) — quieter than live HUD cyan so idle stays cheap visually.
IDLE_TIME_COLOR = (0, 140, 140)


def format_object_metrics(analysis: ObjectAnalysis) -> str:
    """Short one-line measurement string for the HUD / CLI."""
    if analysis.shape == "circle":
        parts = []
        if analysis.diameter_cm is not None:
            parts.append(f"Ø{analysis.diameter_cm:.1f}")
        if analysis.radius_cm is not None:
            parts.append(f"r{analysis.radius_cm:.1f}")
        body = " ".join(parts) if parts else "circle"
    elif analysis.shape == "thin":
        if analysis.length_cm is not None and analysis.width_cm is not None:
            body = f"{analysis.length_cm:.1f}x{analysis.width_cm:.1f}"
        else:
            body = "thin"
    else:
        if analysis.edge_cm:
            body = " ".join(f"{e:.1f}" for e in analysis.edge_cm)
        else:
            body = "polygon"
    colors = ",".join(analysis.colors[:2]) if analysis.colors else ""
    if colors:
        return f"{analysis.shape} {body}cm [{colors}]"
    return f"{analysis.shape} {body}cm"


def _to_hud(
    x: float,
    y: float,
    src_w: int,
    src_h: int,
    hud_w: int,
    hud_h: int,
    homography: Optional[CamProjectorHomography] = None,
) -> Tuple[int, int]:
    mapped = map_cam_to_hud(
        [(x, y)],
        src_size=(src_w, src_h),
        hud_size=(hud_w, hud_h),
        homography=homography,
    )
    return mapped[0]


def _draw_mat(
    canvas: np.ndarray,
    mat_corners: np.ndarray,
    mat_config: MatConfig,
    src_size: Tuple[int, int],
    homography: Optional[CamProjectorHomography] = None,
) -> None:
    hud_h, hud_w = canvas.shape[:2]
    src_w, src_h = src_size
    ordered = order_corners(mat_corners)
    pts = [
        _to_hud(float(x), float(y), src_w, src_h, hud_w, hud_h, homography)
        for x, y in ordered
    ]
    for i in range(4):
        cv2.line(canvas, pts[i], pts[(i + 1) % 4], (0, 255, 255), 2, cv2.LINE_AA)
    for p in pts:
        cv2.circle(canvas, p, 5, (0, 200, 255), -1, cv2.LINE_AA)
    cv2.putText(
        canvas,
        f"mat {mat_config.width_cm:.0f}x{mat_config.height_cm:.0f}cm",
        (pts[0][0] + 8, max(24, pts[0][1] - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )


def _draw_object(
    canvas: np.ndarray,
    analysis: ObjectAnalysis,
    mat_corners: np.ndarray,
    measure_cfg: MatConfig,
    src_size: Tuple[int, int],
    homography: Optional[CamProjectorHomography] = None,
) -> None:
    if not analysis.outline_points:
        return
    hud_h, hud_w = canvas.shape[:2]
    src_w, src_h = src_size
    try:
        cam_pts = mat_plane_points_to_image(
            analysis.outline_points, mat_corners, measure_cfg
        )
    except Exception:
        return
    hud_obj = [
        _to_hud(x, y, src_w, src_h, hud_w, hud_h, homography) for x, y in cam_pts
    ]
    if len(hud_obj) >= 2:
        for i in range(len(hud_obj)):
            cv2.line(
                canvas,
                hud_obj[i],
                hud_obj[(i + 1) % len(hud_obj)],
                (0, 165, 255),
                2,
                cv2.LINE_AA,
            )
    label = format_object_metrics(analysis)
    anchor = hud_obj[0] if hud_obj else (16, hud_h - 20)
    cv2.putText(
        canvas,
        label,
        (max(8, anchor[0]), min(hud_h - 12, anchor[1] + 22)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 165, 255),
        1,
        cv2.LINE_AA,
    )
    if analysis.shape == "polygon" and analysis.edge_cm and len(hud_obj) >= 2:
        n = min(len(analysis.edge_cm), len(hud_obj))
        for i in range(n):
            a = hud_obj[i]
            b = hud_obj[(i + 1) % len(hud_obj)]
            mid = ((a[0] + b[0]) // 2, (a[1] + b[1]) // 2)
            cv2.putText(
                canvas,
                f"{analysis.edge_cm[i]:.1f}",
                mid,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 200, 255),
                1,
                cv2.LINE_AA,
            )


def format_idle_time(now: Optional[datetime] = None) -> str:
    """24h wall clock without seconds (idle HUD v1)."""
    stamp = now or datetime.now()
    return stamp.strftime("%H:%M")


def draw_idle_hud(
    canvas_bgr: np.ndarray,
    *,
    now: Optional[datetime] = None,
) -> np.ndarray:
    """Cheap idle projector HUD: near-black canvas + top-left dim-cyan time only."""
    canvas_bgr[:] = 0
    text = format_idle_time(now)
    cv2.putText(
        canvas_bgr,
        text,
        (24, 64),
        cv2.FONT_HERSHEY_SIMPLEX,
        2.0,
        IDLE_TIME_COLOR,
        2,
        cv2.LINE_AA,
    )
    return canvas_bgr


def draw_desk_hud(
    canvas_bgr: np.ndarray,
    *,
    mat_corners: Optional[np.ndarray],
    mat_config: MatConfig,
    src_size: Tuple[int, int],
    fps_live: float,
    analysis: Optional[ObjectAnalysis] = None,
    measure_config: Optional[MatConfig] = None,
    overlays: Optional[OverlayFlags] = None,
    homography: Optional[CamProjectorHomography] = None,
) -> np.ndarray:
    """Dark projector HUD: mat outline + object outline+cm.

    Measure/work mode replaces idle entirely — do not composite the idle clock here.
    When ``homography`` is set, overlays lock to the desk via cam↔projector H;
    otherwise stretch mapping is used.
    """
    flags = overlays or OverlayFlags()
    canvas_bgr[:] = 0
    measure_cfg = measure_config or mat_config

    if flags.mat and mat_corners is not None:
        _draw_mat(canvas_bgr, mat_corners, mat_config, src_size, homography)

    if flags.object and analysis is not None and mat_corners is not None:
        _draw_object(
            canvas_bgr, analysis, mat_corners, measure_cfg, src_size, homography
        )

    # Projector HUD: FPS number only (no status chrome).
    cv2.putText(
        canvas_bgr,
        f"{fps_live:.0f}",
        (16, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 0),
        2,
        cv2.LINE_AA,
    )
    return canvas_bgr


def draw_debug_camera(
    frame_bgr: np.ndarray,
    *,
    mat_corners: Optional[np.ndarray],
    mat_config: MatConfig,
    fps_live: float,
    mat_ok: bool,
    analysis: Optional[ObjectAnalysis] = None,
    measure_config: Optional[MatConfig] = None,
    overlays: Optional[OverlayFlags] = None,
) -> np.ndarray:
    """Real camera frame + overlays for home-hub debug UI."""
    flags = overlays or OverlayFlags()
    out = frame_bgr.copy()
    src_size = (out.shape[1], out.shape[0])
    measure_cfg = measure_config or mat_config

    if flags.mat and mat_corners is not None:
        _draw_mat(out, mat_corners, mat_config, src_size)
    if flags.object and analysis is not None and mat_corners is not None:
        _draw_object(out, analysis, mat_corners, measure_cfg, src_size)

    status = "mat:lock" if mat_ok else "mat:--"
    obj = "obj:yes" if analysis is not None else "obj:--"
    cv2.putText(
        out,
        f"fps={fps_live:.1f}  {status}  {obj}",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return out
