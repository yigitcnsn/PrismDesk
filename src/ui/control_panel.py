"""Projector control panel: Visual toggles + Sound meter, pinch to click."""

from __future__ import annotations

import time
from typing import Callable, Optional, Sequence, Tuple

import cv2
import numpy as np

from src.core.home_hub import OverlayFlags
from src.ui.pinch import PinchTracker
from src.ui.widgets import Button, LevelMeter, Toggle, WidgetStyle
from src.vision.hands import HandResult
from src.vision.homography import CamProjectorHomography

OnVisualChange = Callable[[OverlayFlags], None]
HudPoint = Tuple[int, int]


class ControlPanel:
    """Bottom-right Controls chip + optional left strip panel."""

    def __init__(
        self,
        *,
        on_visual_change: Optional[OnVisualChange] = None,
        flash_s: float = 0.25,
    ) -> None:
        self._on_visual_change = on_visual_change
        self._flash_s = float(flash_s)
        self._open = False
        self._flags = OverlayFlags()
        self._pinch = PinchTracker()
        self._style = WidgetStyle()
        self._controls = Button("controls", "Controls", (0, 0, 160, 56))
        self._close = Button("close", "Close", (0, 0, 160, 56))
        self._toggles = {
            "mat": Toggle("mat", "mat", (0, 0, 220, 64), on=True),
            "object": Toggle("object", "object", (0, 0, 220, 64), on=True),
            "hands": Toggle("hands", "hands", (0, 0, 220, 64), on=True),
        }
        self._meter = LevelMeter((0, 0, 220, 36), label="mic")
        self._last_tip: Optional[HudPoint] = None
        self._layout_for((1280, 720))

    @property
    def open(self) -> bool:
        return self._open

    @property
    def flags(self) -> OverlayFlags:
        return OverlayFlags(
            mat=self._flags.mat,
            object=self._flags.object,
            hands=self._flags.hands,
        )

    def set_flags(self, flags: OverlayFlags) -> None:
        self._flags = OverlayFlags(
            mat=bool(flags.mat),
            object=bool(flags.object),
            hands=bool(flags.hands),
        )
        self._toggles["mat"].on = self._flags.mat
        self._toggles["object"].on = self._flags.object
        self._toggles["hands"].on = self._flags.hands

    def set_level(self, level: float, *, available: bool) -> None:
        self._meter.level = float(level)
        self._meter.available = bool(available)

    def update(
        self,
        hands: Sequence[HandResult],
        *,
        src_size: Tuple[int, int],
        hud_size: Tuple[int, int],
        homography: Optional[CamProjectorHomography] = None,
    ) -> None:
        self._layout_for(hud_size)
        snap = self._pinch.update(
            hands,
            src_size=src_size,
            hud_size=hud_size,
            homography=homography,
        )
        self._last_tip = snap.index_hud
        if not self._pinch.consume_click(snap.just_opened):
            return
        if snap.index_hud is None:
            return
        tip = snap.index_hud
        now = time.monotonic()

        if self._controls.hit(tip):
            self._open = not self._open
            self._controls.flash_until = now + self._flash_s
            return

        if not self._open:
            return

        if self._close.hit(tip):
            self._open = False
            self._close.flash_until = now + self._flash_s
            return

        for key, toggle in self._toggles.items():
            if toggle.hit(tip):
                toggle.on = not toggle.on
                toggle.flash_until = now + self._flash_s
                if key == "mat":
                    self._flags.mat = toggle.on
                elif key == "object":
                    self._flags.object = toggle.on
                else:
                    self._flags.hands = toggle.on
                if self._on_visual_change is not None:
                    self._on_visual_change(self.flags)
                return

    def draw(self, canvas: np.ndarray) -> None:
        now = time.monotonic()
        h, w = canvas.shape[:2]
        self._layout_for((w, h))
        self._controls.draw(canvas, now=now, style=self._style)
        if self._last_tip is not None:
            color = (0, 255, 180) if self._pinch.is_pinched else (200, 200, 200)
            cv2.circle(canvas, self._last_tip, 16, color, 2, cv2.LINE_AA)
        if not self._open:
            return

        panel_w = 260
        overlay = canvas[:, :panel_w].copy()
        cv2.rectangle(overlay, (0, 0), (panel_w - 1, h - 1), (18, 18, 18), -1)
        cv2.addWeighted(overlay, 0.85, canvas[:, :panel_w], 0.15, 0, canvas[:, :panel_w])
        cv2.putText(
            canvas,
            "Visual",
            (24, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            self._style.on_color,
            2,
            cv2.LINE_AA,
        )
        for toggle in self._toggles.values():
            toggle.draw(canvas, now=now, style=self._style)
        cv2.putText(
            canvas,
            "Sound",
            (24, self._meter.rect[1] - 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            self._style.on_color,
            2,
            cv2.LINE_AA,
        )
        self._meter.draw(canvas, style=self._style)
        self._close.draw(canvas, now=now, style=self._style)

    def _layout_for(self, hud_size: Tuple[int, int]) -> None:
        w, h = int(hud_size[0]), int(hud_size[1])
        chip_w, chip_h = 160, 56
        self._controls.rect = (
            max(8, w - chip_w - 24),
            max(8, h - chip_h - 24),
            chip_w,
            chip_h,
        )

        x0, y0 = 24, 72
        tw, th, gap = 220, 64, 18
        for i, key in enumerate(("mat", "object", "hands")):
            self._toggles[key].rect = (x0, y0 + i * (th + gap), tw, th)
        meter_y = y0 + 3 * (th + gap) + 40
        self._meter.rect = (x0, meter_y, tw, 36)
        self._close.rect = (x0, min(h - chip_h - 24, meter_y + 70), tw, chip_h)
