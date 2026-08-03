"""Interactive outline edge measurement on a warped mat image."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .geometry import path_length_cm, segment_lengths_cm


@dataclass
class OutlineResult:
    points: list[tuple[float, float]]
    segment_cm: list[float]
    total_cm: float


class OutlineSession:
    """Click outline vertices on a warped top-down mat image."""

    def __init__(self, image: np.ndarray, px_per_cm: float, window_name: str = "Measure outline") -> None:
        self.image = image
        self.px_per_cm = float(px_per_cm)
        self.window_name = window_name
        self.points: list[tuple[float, float]] = []

    def run(self) -> OutlineResult | None:
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self._on_mouse)

        while True:
            frame = self._draw()
            cv2.imshow(self.window_name, frame)
            key = cv2.waitKey(20) & 0xFF
            if key in (ord("q"), 27):
                cv2.destroyWindow(self.window_name)
                return None
            if key == ord("u") and self.points:
                self.points.pop()
            if key == ord("r"):
                self.points.clear()
            if key in (13, 10, ord("n")) and len(self.points) >= 2:
                cv2.destroyWindow(self.window_name)
                segs = segment_lengths_cm(self.points, self.px_per_cm)
                total = path_length_cm(self.points, self.px_per_cm)
                return OutlineResult(points=list(self.points), segment_cm=segs, total_cm=total)

    def _on_mouse(self, event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((float(x), float(y)))

    def _draw(self) -> np.ndarray:
        frame = self.image.copy()
        segs = segment_lengths_cm(self.points, self.px_per_cm) if len(self.points) >= 2 else []
        total = path_length_cm(self.points, self.px_per_cm) if len(self.points) >= 2 else 0.0

        if len(self.points) >= 2:
            pts = np.array(self.points, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(frame, [pts], False, (0, 255, 255), 2)

        for i, (x, y) in enumerate(self.points):
            cv2.circle(frame, (int(x), int(y)), 5, (255, 0, 255), -1)
            if i > 0:
                mid = (
                    int((self.points[i - 1][0] + x) / 2),
                    int((self.points[i - 1][1] + y) / 2),
                )
                label = f"{segs[i - 1]:.1f} cm"
                cv2.putText(frame, label, mid, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        hint = "Click outline | u undo | r reset | Enter/n finish | q quit"
        cv2.putText(frame, hint, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        cv2.putText(
            frame,
            f"Total: {total:.2f} cm  |  points: {len(self.points)}",
            (8, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
        )
        return frame
