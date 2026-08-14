"""Pinch detection from MediaPipe hand landmarks (2D only; no depth)."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

from src.vision.hands import HandResult
from src.vision.homography import CamProjectorHomography, map_cam_to_hud

Point = Tuple[float, float]
HudPoint = Tuple[int, int]

# MediaPipe indices
_WRIST = 0
_THUMB_TIP = 4
_INDEX_TIP = 8
_MIDDLE_MCP = 9


@dataclass
class PinchSnapshot:
    is_pinched: bool
    just_closed: bool
    just_opened: bool
    index_hud: Optional[HudPoint]
    norm_gap: Optional[float]


class PinchTracker:
    """Edge-triggered pinch with scale-normalized thumb↔index gap."""

    def __init__(
        self,
        *,
        close_thresh: float = 0.38,
        open_thresh: float = 0.48,
        cooldown_s: float = 0.30,
    ) -> None:
        self.close_thresh = float(close_thresh)
        self.open_thresh = float(open_thresh)
        self.cooldown_s = float(cooldown_s)
        self._pinched = False
        self._last_fire = 0.0

    @property
    def is_pinched(self) -> bool:
        return self._pinched

    def reset(self) -> None:
        self._pinched = False

    def update(
        self,
        hands: Sequence[HandResult],
        *,
        src_size: Tuple[int, int],
        hud_size: Tuple[int, int],
        homography: Optional[CamProjectorHomography] = None,
    ) -> PinchSnapshot:
        hand = _prefer_hand(hands)
        if hand is None:
            was = self._pinched
            self._pinched = False
            return PinchSnapshot(
                is_pinched=False,
                just_closed=False,
                just_opened=was,
                index_hud=None,
                norm_gap=None,
            )

        gap = _normalized_pinch_gap(hand)
        index_hud = _tip_to_hud(
            hand.index_tip, src_size=src_size, hud_size=hud_size, homography=homography
        )

        just_closed = False
        just_opened = False
        if gap is not None:
            if not self._pinched and gap <= self.close_thresh:
                self._pinched = True
                just_closed = True
            elif self._pinched and gap >= self.open_thresh:
                self._pinched = False
                just_opened = True

        return PinchSnapshot(
            is_pinched=self._pinched,
            just_closed=just_closed,
            just_opened=just_opened,
            index_hud=index_hud,
            norm_gap=gap,
        )

    def consume_click(self, just_opened: bool) -> bool:
        """Return True once on pinch-up if cooldown allows."""
        if not just_opened:
            return False
        now = time.monotonic()
        if now - self._last_fire < self.cooldown_s:
            return False
        self._last_fire = now
        return True


def _prefer_hand(hands: Sequence[HandResult]) -> Optional[HandResult]:
    if not hands:
        return None
    for hand in hands:
        if str(hand.handedness).lower().startswith("right"):
            return hand
    return hands[0]


def _normalized_pinch_gap(hand: HandResult) -> Optional[float]:
    pts = hand.landmarks_px
    if len(pts) <= _MIDDLE_MCP:
        # Fallback using stored tips only.
        scale = 80.0
        gap = _dist(hand.thumb_tip, hand.index_tip)
        return float(gap / scale)
    wrist = pts[_WRIST]
    mid = pts[_MIDDLE_MCP]
    scale = _dist(wrist, mid)
    if scale < 1e-3:
        return None
    tip_a = pts[_THUMB_TIP] if len(pts) > _THUMB_TIP else hand.thumb_tip
    tip_b = pts[_INDEX_TIP] if len(pts) > _INDEX_TIP else hand.index_tip
    return float(_dist(tip_a, tip_b) / scale)


def _dist(a: Point, b: Point) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def _tip_to_hud(
    tip: Point,
    *,
    src_size: Tuple[int, int],
    hud_size: Tuple[int, int],
    homography: Optional[CamProjectorHomography],
) -> HudPoint:
    mapped = map_cam_to_hud(
        [tip],
        src_size=src_size,
        hud_size=hud_size,
        homography=homography,
    )
    return mapped[0]
