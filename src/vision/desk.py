"""Live desk HUD: mat outline + object measure + hands on the projector canvas."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

from src.measure.mat import MatConfig, order_corners
from src.measure.perspective import image_points_to_mat_cm, mat_plane_points_to_image
from src.measure.shape import ObjectAnalysis
from src.vision.hands import HAND_CONNECTIONS, HandResult

Point = Tuple[float, float]


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
) -> Tuple[int, int]:
    if src_w <= 0 or src_h <= 0:
        return int(x), int(y)
    return int(x * hud_w / src_w), int(y * hud_h / src_h)


def draw_desk_hud(
    canvas_bgr: np.ndarray,
    *,
    hands: Sequence[HandResult],
    mat_corners: Optional[np.ndarray],
    mat_config: MatConfig,
    src_size: Tuple[int, int],
    fps_live: float,
    track_fps: float,
    mat_ok: bool,
    analysis: Optional[ObjectAnalysis] = None,
    measure_config: Optional[MatConfig] = None,
) -> np.ndarray:
    """Dark projector HUD: mat, object outline+cm, hand skeleton."""
    hud_h, hud_w = canvas_bgr.shape[:2]
    src_w, src_h = src_size
    canvas_bgr[:] = 0
    measure_cfg = measure_config or mat_config

    if mat_corners is not None:
        ordered = order_corners(mat_corners)
        pts = [
            _to_hud(float(x), float(y), src_w, src_h, hud_w, hud_h) for x, y in ordered
        ]
        for i in range(4):
            cv2.line(canvas_bgr, pts[i], pts[(i + 1) % 4], (0, 255, 255), 2, cv2.LINE_AA)
        for p in pts:
            cv2.circle(canvas_bgr, p, 5, (0, 200, 255), -1, cv2.LINE_AA)
        cv2.putText(
            canvas_bgr,
            f"mat {mat_config.width_cm:.0f}x{mat_config.height_cm:.0f}cm",
            (pts[0][0] + 8, max(24, pts[0][1] - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

    if analysis is not None and mat_corners is not None and analysis.outline_points:
        try:
            cam_pts = mat_plane_points_to_image(
                analysis.outline_points, mat_corners, measure_cfg
            )
            hud_obj = [
                _to_hud(x, y, src_w, src_h, hud_w, hud_h) for x, y in cam_pts
            ]
            if len(hud_obj) >= 2:
                for i in range(len(hud_obj)):
                    cv2.line(
                        canvas_bgr,
                        hud_obj[i],
                        hud_obj[(i + 1) % len(hud_obj)],
                        (0, 165, 255),
                        2,
                        cv2.LINE_AA,
                    )
            label = format_object_metrics(analysis)
            anchor = hud_obj[0] if hud_obj else (16, hud_h - 20)
            cv2.putText(
                canvas_bgr,
                label,
                (max(8, anchor[0]), min(hud_h - 12, anchor[1] + 22)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 165, 255),
                1,
                cv2.LINE_AA,
            )
            # Edge labels for polygons
            if analysis.shape == "polygon" and analysis.edge_cm and len(hud_obj) >= 2:
                n = min(len(analysis.edge_cm), len(hud_obj))
                for i in range(n):
                    a = hud_obj[i]
                    b = hud_obj[(i + 1) % len(hud_obj)]
                    mid = ((a[0] + b[0]) // 2, (a[1] + b[1]) // 2)
                    cv2.putText(
                        canvas_bgr,
                        f"{analysis.edge_cm[i]:.1f}",
                        mid,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (0, 200, 255),
                        1,
                        cv2.LINE_AA,
                    )
        except Exception:
            pass

    bone = (0, 255, 255)
    joint = (255, 0, 255)
    tip_color = (0, 255, 0)
    for hand in hands:
        pts_px = hand.landmarks_px
        hud_pts = [
            _to_hud(x, y, src_w, src_h, hud_w, hud_h) for x, y in pts_px
        ]
        for a, b in HAND_CONNECTIONS:
            if a < len(hud_pts) and b < len(hud_pts):
                cv2.line(canvas_bgr, hud_pts[a], hud_pts[b], bone, 2, cv2.LINE_AA)
        for p in hud_pts:
            cv2.circle(canvas_bgr, p, 4, joint, -1, cv2.LINE_AA)
        if not hud_pts:
            continue
        tip = hud_pts[8] if len(hud_pts) > 8 else hud_pts[-1]
        cv2.circle(canvas_bgr, tip, 12, tip_color, 2, cv2.LINE_AA)
        label = hand.handedness
        if mat_corners is not None:
            try:
                cm = image_points_to_mat_cm([hand.index_tip], mat_corners, mat_config)[0]
                label = f"{hand.handedness} {cm[0]:.1f},{cm[1]:.1f}cm"
            except Exception:
                pass
        cv2.putText(
            canvas_bgr,
            label,
            (tip[0] + 14, tip[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            tip_color,
            1,
            cv2.LINE_AA,
        )

    status = "mat:lock" if mat_ok else "mat:--"
    obj = "obj:yes" if analysis is not None else "obj:--"
    cv2.putText(
        canvas_bgr,
        f"fps={fps_live:.1f}  track={track_fps:.1f}Hz  hands={len(hands)}  {status}  {obj}",
        (16, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 0),
        2,
        cv2.LINE_AA,
    )
    return canvas_bgr
