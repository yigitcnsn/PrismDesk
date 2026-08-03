"""Interactive outline edge measurement on a warped mat image."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .geometry import path_length_cm, segment_lengths_cm
from .mat import MatConfig
from .object import analyze_object
from .shape import ObjectAnalysis


Point = Tuple[float, float]


@dataclass
class OutlineResult:
    points: List[Point]
    segment_cm: List[float]
    total_cm: float
    closed: bool
    analysis: Optional[ObjectAnalysis] = None


class OutlineSession:
    """Click or auto-find outline on a warped top-down mat image."""

    def __init__(
        self,
        image: np.ndarray,
        px_per_cm: float,
        config: Optional[MatConfig] = None,
        initial_points: Optional[Sequence[Point]] = None,
        closed: bool = False,
        analysis: Optional[ObjectAnalysis] = None,
        window_name: str = "Measure outline",
    ) -> None:
        self.image = image
        self.px_per_cm = float(px_per_cm)
        self.config = config
        self.window_name = window_name
        self.analysis = analysis
        if analysis is not None and not initial_points:
            self.points: List[Point] = list(analysis.outline_points)
            self.closed = True
        else:
            self.points = [tuple(map(float, p)) for p in (initial_points or [])]  # type: ignore[misc]
            self.closed = bool(closed and len(self.points) >= 3)

    def run(self) -> Optional[OutlineResult]:
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
                self.closed = False
                self.analysis = None
            if key == ord("r"):
                self.points.clear()
                self.closed = False
                self.analysis = None
            if key == ord("c") and len(self.points) >= 3:
                self.closed = True
            if key == ord("a"):
                self._auto_find()
            if key in (13, 10, ord("n")) and (len(self.points) >= 2 or self.analysis is not None):
                cv2.destroyWindow(self.window_name)
                return self._result()

    def _auto_find(self) -> None:
        if self.config is None:
            return
        found = analyze_object(self.image, self.config)
        if found is None:
            return
        self.analysis = found
        self.points = list(found.outline_points)
        self.closed = len(self.points) >= 3

    def _result(self) -> OutlineResult:
        if self.analysis is not None and self.analysis.shape == "circle":
            return OutlineResult(
                points=list(self.points),
                segment_cm=[],
                total_cm=float(self.analysis.diameter_cm or 0.0),
                closed=True,
                analysis=self.analysis,
            )
        pts = list(self.points)
        measure_pts = pts + [pts[0]] if self.closed and len(pts) >= 3 else pts
        segs = segment_lengths_cm(measure_pts, self.px_per_cm)
        total = path_length_cm(measure_pts, self.px_per_cm)
        return OutlineResult(
            points=pts,
            segment_cm=segs,
            total_cm=total,
            closed=self.closed,
            analysis=self.analysis,
        )

    def _on_mouse(self, event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((float(x), float(y)))
            self.closed = False
            self.analysis = None

    def _draw(self) -> np.ndarray:
        frame = self.image.copy()
        analysis = self.analysis

        if analysis is not None and analysis.shape == "circle" and analysis.center and analysis.radius_cm:
            cx, cy = int(analysis.center[0]), int(analysis.center[1])
            r_px = analysis.radius_cm * self.px_per_cm
            cv2.circle(frame, (cx, cy), int(round(r_px)), (0, 255, 255), 2)
            cv2.circle(frame, (cx, cy), 4, (255, 0, 255), -1)
            cv2.putText(
                frame,
                f"r={analysis.radius_cm:.2f} cm  Ø={analysis.diameter_cm:.2f} cm",
                (cx + 10, cy - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
        else:
            pts = list(self.points)
            measure_pts = pts + [pts[0]] if self.closed and len(pts) >= 3 else pts
            segs = (
                analysis.edge_cm
                if analysis is not None and analysis.edge_cm
                else (segment_lengths_cm(measure_pts, self.px_per_cm) if len(measure_pts) >= 2 else [])
            )
            if len(pts) >= 2:
                arr = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
                cv2.polylines(frame, [arr], self.closed, (0, 255, 255), 2)
            for i, (x, y) in enumerate(pts):
                cv2.circle(frame, (int(x), int(y)), 5, (255, 0, 255), -1)
                cv2.putText(
                    frame,
                    str(i + 1),
                    (int(x) + 6, int(y) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 0, 255),
                    1,
                )
            for i in range(len(segs)):
                if i + 1 >= len(measure_pts):
                    break
                a = measure_pts[i]
                b = measure_pts[i + 1]
                mid = (int((a[0] + b[0]) / 2), int((a[1] + b[1]) / 2))
                cv2.putText(
                    frame,
                    f"E{i + 1}: {segs[i]:.1f} cm",
                    mid,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

            if analysis is not None and analysis.fillet_radii_cm:
                y0 = 72
                for i, fr in enumerate(analysis.fillet_radii_cm):
                    cv2.putText(
                        frame,
                        f"fillet{i + 1}: {fr:.2f} cm",
                        (8, y0 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 200, 255),
                        2,
                    )

            if analysis is not None and analysis.shape == "thin":
                cv2.putText(
                    frame,
                    f"L={analysis.length_cm:.2f} cm  W={analysis.width_cm:.2f} cm",
                    (8, 72),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0),
                    2,
                )

        if analysis is not None and analysis.colors:
            cv2.putText(
                frame,
                "colors: " + ", ".join(analysis.colors[:3]),
                (8, frame.shape[0] - 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
            )

        hint = "a auto-find | c close | u undo | r reset | Enter/n finish | q quit"
        cv2.putText(frame, hint, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        shape = analysis.shape if analysis is not None else ("closed" if self.closed else "open")
        cv2.putText(
            frame,
            f"shape: {shape}  |  points: {len(self.points)}",
            (8, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            2,
        )
        return frame
