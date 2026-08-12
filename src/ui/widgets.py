"""Simple projector HUD widgets (hit-test + draw)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

HudPoint = Tuple[int, int]
Rect = Tuple[int, int, int, int]  # x, y, w, h


@dataclass
class WidgetStyle:
    on_color: Tuple[int, int, int] = (0, 220, 220)
    off_color: Tuple[int, int, int] = (70, 70, 70)
    text_color: Tuple[int, int, int] = (240, 240, 240)
    border: Tuple[int, int, int] = (180, 180, 180)
    flash_color: Tuple[int, int, int] = (0, 255, 180)


def point_in_rect(pt: HudPoint, rect: Rect) -> bool:
    x, y, w, h = rect
    return x <= pt[0] < x + w and y <= pt[1] < y + h


@dataclass
class Button:
    id: str
    label: str
    rect: Rect
    flash_until: float = 0.0

    def hit(self, pt: HudPoint) -> bool:
        return point_in_rect(pt, self.rect)

    def draw(self, canvas: np.ndarray, *, now: float, style: Optional[WidgetStyle] = None) -> None:
        style = style or WidgetStyle()
        x, y, w, h = self.rect
        color = style.flash_color if now < self.flash_until else style.on_color
        cv2.rectangle(canvas, (x, y), (x + w - 1, y + h - 1), color, 2, cv2.LINE_AA)
        cv2.putText(
            canvas,
            self.label,
            (x + 14, y + h // 2 + 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            style.text_color,
            2,
            cv2.LINE_AA,
        )


@dataclass
class Toggle:
    id: str
    label: str
    rect: Rect
    on: bool = True
    flash_until: float = 0.0

    def hit(self, pt: HudPoint) -> bool:
        return point_in_rect(pt, self.rect)

    def draw(self, canvas: np.ndarray, *, now: float, style: Optional[WidgetStyle] = None) -> None:
        style = style or WidgetStyle()
        x, y, w, h = self.rect
        if now < self.flash_until:
            fill = style.flash_color
        elif self.on:
            fill = style.on_color
        else:
            fill = style.off_color
        cv2.rectangle(canvas, (x, y), (x + w - 1, y + h - 1), fill, -1, cv2.LINE_AA)
        cv2.rectangle(canvas, (x, y), (x + w - 1, y + h - 1), style.border, 2, cv2.LINE_AA)
        state = "ON" if self.on else "OFF"
        cv2.putText(
            canvas,
            f"{self.label}  {state}",
            (x + 16, y + h // 2 + 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (10, 10, 10) if self.on else style.text_color,
            2,
            cv2.LINE_AA,
        )


@dataclass
class LevelMeter:
    rect: Rect
    level: float = 0.0  # 0..1
    available: bool = False
    label: str = "mic"

    def draw(self, canvas: np.ndarray, *, style: Optional[WidgetStyle] = None) -> None:
        style = style or WidgetStyle()
        x, y, w, h = self.rect
        cv2.rectangle(canvas, (x, y), (x + w - 1, y + h - 1), style.border, 2, cv2.LINE_AA)
        inner = max(0, w - 8)
        if self.available:
            fill_w = int(inner * max(0.0, min(1.0, self.level)))
            if fill_w > 0:
                cv2.rectangle(
                    canvas,
                    (x + 4, y + 4),
                    (x + 4 + fill_w, y + h - 5),
                    style.on_color,
                    -1,
                    cv2.LINE_AA,
                )
            db = 20.0 * np.log10(max(self.level, 1e-6))
            text = f"{self.label}  {db:.0f} dBFS"
        else:
            text = f"{self.label}:--"
        cv2.putText(
            canvas,
            text,
            (x + 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            style.text_color,
            1,
            cv2.LINE_AA,
        )
