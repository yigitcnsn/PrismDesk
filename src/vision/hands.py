"""MediaPipe Hands wrapper for live undistorted frames."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import cv2
import numpy as np

try:
    import mediapipe as mp
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "mediapipe is required for hand tracking. Install with: pip install mediapipe"
    ) from exc


Point = Tuple[float, float]


@dataclass
class HandResult:
    landmarks_px: List[Point]
    index_tip: Point
    handedness: str
    raw_landmarks: Any = None


class HandTracker:
    def __init__(
        self,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.6,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        self._mp_hands = mp.solutions.hands
        self._drawer = mp.solutions.drawing_utils
        self._styles = mp.solutions.drawing_styles
        self._hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            model_complexity=0,  # lighter on Pi 5
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def process(self, frame_bgr: np.ndarray) -> List[HandResult]:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        result = self._hands.process(rgb)
        rgb.flags.writeable = True
        h, w = frame_bgr.shape[:2]
        hands: List[HandResult] = []
        if not result.multi_hand_landmarks:
            return hands
        handedness = result.multi_handedness or []
        for i, lm_list in enumerate(result.multi_hand_landmarks):
            pts = [(lm.x * w, lm.y * h) for lm in lm_list.landmark]
            label = "Unknown"
            if i < len(handedness):
                label = handedness[i].classification[0].label
            tip = pts[8]  # INDEX_FINGER_TIP
            hands.append(
                HandResult(
                    landmarks_px=pts,
                    index_tip=tip,
                    handedness=label,
                    raw_landmarks=lm_list,
                )
            )
        return hands

    def draw(self, frame_bgr: np.ndarray, hands: List[HandResult]) -> np.ndarray:
        out = frame_bgr
        for hand in hands:
            if hand.raw_landmarks is not None:
                self._drawer.draw_landmarks(
                    out,
                    hand.raw_landmarks,
                    self._mp_hands.HAND_CONNECTIONS,
                    self._styles.get_default_hand_landmarks_style(),
                    self._styles.get_default_hand_connections_style(),
                )
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
    ) -> np.ndarray:
        """Draw bright hand skeleton on a dark projector canvas.

        Landmarks are mapped with a simple stretch from the camera frame to the
        canvas (cam↔projector homography comes later). Prefer MediaPipe normalized
        coords when raw_landmarks are present.
        """
        h, w = canvas_bgr.shape[:2]
        bone = (0, 255, 255)
        joint = (255, 0, 255)
        tip_color = (0, 255, 0)
        for hand in hands:
            pts = _landmarks_to_canvas(hand, w, h, src_size)
            if not pts:
                continue
            for a, b in self._mp_hands.HAND_CONNECTIONS:
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
        self._hands.close()


def _landmarks_to_canvas(
    hand: HandResult,
    canvas_w: int,
    canvas_h: int,
    src_size: Optional[Tuple[int, int]],
) -> List[Tuple[int, int]]:
    if hand.raw_landmarks is not None:
        return [
            (int(lm.x * canvas_w), int(lm.y * canvas_h))
            for lm in hand.raw_landmarks.landmark
        ]
    if not hand.landmarks_px:
        return []
    if src_size is None or src_size[0] <= 0 or src_size[1] <= 0:
        return [(int(x), int(y)) for x, y in hand.landmarks_px]
    sw, sh = src_size
    sx = canvas_w / float(sw)
    sy = canvas_h / float(sh)
    return [(int(x * sx), int(y * sy)) for x, y in hand.landmarks_px]
