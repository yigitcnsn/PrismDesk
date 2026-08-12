"""MediaPipe Hand Landmarker (Tasks API) for live undistorted frames.

MediaPipe >= 0.10.30 removed mp.solutions; this module uses HandLandmarker.
"""

from __future__ import annotations

import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .homography import CamProjectorHomography, map_cam_to_hud

Point = Tuple[float, float]

# Classic 21-point hand skeleton (MediaPipe Hands).
HAND_CONNECTIONS: Tuple[Tuple[int, int], ...] = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
)

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parents[2] / "models" / "hand_landmarker.task"
)


@dataclass
class HandResult:
    landmarks_px: List[Point]
    landmarks_norm: List[Point]
    index_tip: Point
    thumb_tip: Point
    handedness: str


def ensure_hand_model(path: Optional[str | Path] = None) -> Path:
    """Return path to hand_landmarker.task, downloading once if missing."""
    dest = Path(path) if path is not None else DEFAULT_MODEL_PATH
    if dest.is_file() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".task.partial")
    print(f"Downloading MediaPipe hand model → {dest}")
    print(f"  {MODEL_URL}")
    try:
        urllib.request.urlretrieve(MODEL_URL, tmp)
        tmp.replace(dest)
    except Exception:
        if tmp.is_file():
            tmp.unlink(missing_ok=True)
        raise
    return dest


def _import_mediapipe():
    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_tasks
        from mediapipe.tasks.python import vision as mp_vision
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "mediapipe is required for hand tracking. Install with: pip install mediapipe"
        ) from exc
    return mp, mp_tasks, mp_vision


class HandTracker:
    def __init__(
        self,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.6,
        min_tracking_confidence: float = 0.5,
        model_path: Optional[str | Path] = None,
        infer_size: Optional[Tuple[int, int]] = (640, 360),
    ) -> None:
        mp, mp_tasks, mp_vision = _import_mediapipe()
        self._mp = mp
        model = ensure_hand_model(model_path)
        options = mp_vision.HandLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=str(model)),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = mp_vision.HandLandmarker.create_from_options(options)
        self._ts_ms = 0
        self._t0 = time.monotonic()
        # Downscale before inference on Pi; None / (0,0) = full frame.
        if infer_size is None or infer_size[0] <= 0 or infer_size[1] <= 0:
            self.infer_size: Optional[Tuple[int, int]] = None
        else:
            self.infer_size = (int(infer_size[0]), int(infer_size[1]))

    def process(self, frame_bgr: np.ndarray) -> List[HandResult]:
        h, w = frame_bgr.shape[:2]
        infer = frame_bgr
        if self.infer_size is not None:
            tw, th = self.infer_size
            if w != tw or h != th:
                infer = cv2.resize(frame_bgr, (tw, th), interpolation=cv2.INTER_AREA)

        rgb = cv2.cvtColor(infer, cv2.COLOR_BGR2RGB)
        if not rgb.flags["C_CONTIGUOUS"]:
            rgb = np.ascontiguousarray(rgb)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)

        # Timestamps must be monotonically increasing (ms).
        now_ms = int((time.monotonic() - self._t0) * 1000.0)
        if now_ms <= self._ts_ms:
            now_ms = self._ts_ms + 1
        self._ts_ms = now_ms

        result = self._landmarker.detect_for_video(mp_image, now_ms)
        hands: List[HandResult] = []
        landmarks_list = result.hand_landmarks or []
        handedness_list = result.handedness or []
        for i, lm_list in enumerate(landmarks_list):
            # Normalized coords are FOV-relative; map px back to the source frame.
            norms = [(float(lm.x), float(lm.y)) for lm in lm_list]
            pts = [(x * w, y * h) for x, y in norms]
            label = "Unknown"
            if i < len(handedness_list) and handedness_list[i]:
                label = handedness_list[i][0].category_name
            tip = pts[8] if len(pts) > 8 else (pts[0] if pts else (0.0, 0.0))
            thumb = pts[4] if len(pts) > 4 else tip
            hands.append(
                HandResult(
                    landmarks_px=pts,
                    landmarks_norm=norms,
                    index_tip=tip,
                    thumb_tip=thumb,
                    handedness=label,
                )
            )
        return hands

    def draw(self, frame_bgr: np.ndarray, hands: List[HandResult]) -> np.ndarray:
        out = frame_bgr
        bone = (0, 255, 255)
        joint = (255, 0, 255)
        for hand in hands:
            pts = [(int(x), int(y)) for x, y in hand.landmarks_px]
            for a, b in HAND_CONNECTIONS:
                if a < len(pts) and b < len(pts):
                    cv2.line(out, pts[a], pts[b], bone, 2, cv2.LINE_AA)
            for p in pts:
                cv2.circle(out, p, 4, joint, -1, cv2.LINE_AA)
            tip = (int(hand.index_tip[0]), int(hand.index_tip[1]))
            cv2.circle(out, tip, 12, (0, 255, 0), 2)
            cv2.putText(
                out,
                f"{hand.handedness}",
                (tip[0] + 14, tip[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
        return out

    def draw_hud(
        self,
        canvas_bgr: np.ndarray,
        hands: List[HandResult],
        *,
        src_size: Optional[Tuple[int, int]] = None,
        homography: Optional[CamProjectorHomography] = None,
    ) -> np.ndarray:
        """Draw bright hand skeleton on a dark projector canvas.

        Uses cam↔projector homography when provided; otherwise stretch / norm mapping.
        """
        h, w = canvas_bgr.shape[:2]
        bone = (0, 255, 255)
        joint = (255, 0, 255)
        tip_color = (0, 255, 0)
        for hand in hands:
            pts = _landmarks_to_canvas(
                hand, w, h, src_size, homography=homography
            )
            if not pts:
                continue
            for a, b in HAND_CONNECTIONS:
                if a < len(pts) and b < len(pts):
                    cv2.line(canvas_bgr, pts[a], pts[b], bone, 3, cv2.LINE_AA)
            for p in pts:
                cv2.circle(canvas_bgr, p, 5, joint, -1, cv2.LINE_AA)
            tip = pts[8] if len(pts) > 8 else pts[-1]
            cv2.circle(canvas_bgr, tip, 14, tip_color, 2, cv2.LINE_AA)
            cv2.putText(
                canvas_bgr,
                hand.handedness,
                (tip[0] + 16, tip[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                tip_color,
                2,
                cv2.LINE_AA,
            )
        return canvas_bgr

    def close(self) -> None:
        self._landmarker.close()


def _landmarks_to_canvas(
    hand: HandResult,
    canvas_w: int,
    canvas_h: int,
    src_size: Optional[Tuple[int, int]],
    *,
    homography: Optional[CamProjectorHomography] = None,
) -> List[Tuple[int, int]]:
    if homography is not None:
        if src_size is None or src_size[0] <= 0 or src_size[1] <= 0:
            return []
        sw, sh = src_size
        if hand.landmarks_px:
            pts = hand.landmarks_px
        elif hand.landmarks_norm:
            pts = [(x * sw, y * sh) for x, y in hand.landmarks_norm]
        else:
            return []
        return map_cam_to_hud(
            pts,
            src_size=src_size,
            hud_size=(canvas_w, canvas_h),
            homography=homography,
        )
    if hand.landmarks_norm:
        return [
            (int(x * canvas_w), int(y * canvas_h)) for x, y in hand.landmarks_norm
        ]
    if not hand.landmarks_px:
        return []
    if src_size is None or src_size[0] <= 0 or src_size[1] <= 0:
        return [(int(x), int(y)) for x, y in hand.landmarks_px]
    sw, sh = src_size
    sx = canvas_w / float(sw)
    sy = canvas_h / float(sh)
    return [(int(x * sx), int(y * sy)) for x, y in hand.landmarks_px]
