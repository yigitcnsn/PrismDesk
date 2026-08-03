"""Pure geometry helpers for edge measurements."""

from __future__ import annotations

import math
from typing import Sequence, Tuple, Union

Point = Union[Tuple[float, float], Sequence[float]]


def segment_length_px(a: Point, b: Point) -> float:
    """Euclidean distance between two points in pixels."""
    return math.hypot(float(b[0]) - float(a[0]), float(b[1]) - float(a[1]))


def path_length_px(points: Sequence[Point]) -> float:
    """Sum of consecutive segment lengths in pixels."""
    if len(points) < 2:
        return 0.0
    total = 0.0
    for i in range(1, len(points)):
        total += segment_length_px(points[i - 1], points[i])
    return total


def segment_lengths_px(points: Sequence[Point]) -> list[float]:
    """Length of each consecutive segment in pixels."""
    if len(points) < 2:
        return []
    return [segment_length_px(points[i - 1], points[i]) for i in range(1, len(points))]


def to_cm(length_px: float, px_per_cm: float) -> float:
    """Convert a pixel length to centimetres."""
    if px_per_cm <= 0:
        raise ValueError("px_per_cm must be positive")
    return length_px / px_per_cm


def segment_lengths_cm(points: Sequence[Point], px_per_cm: float) -> list[float]:
    """Length of each consecutive segment in centimetres."""
    return [to_cm(px, px_per_cm) for px in segment_lengths_px(points)]


def path_length_cm(points: Sequence[Point], px_per_cm: float) -> float:
    """Total path length in centimetres."""
    return to_cm(path_length_px(points), px_per_cm)
