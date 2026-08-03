"""Load images (including HEIC) as OpenCV BGR arrays."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

_HEIF_REGISTERED = False


def _ensure_heif() -> None:
    global _HEIF_REGISTERED
    if _HEIF_REGISTERED:
        return
    from pillow_heif import register_heif_opener

    register_heif_opener()
    _HEIF_REGISTERED = True


def load_image(path: str | Path) -> np.ndarray:
    """Load an image file as a BGR uint8 ndarray for OpenCV."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")

    suffix = path.suffix.lower()
    if suffix in {".heic", ".heif"}:
        _ensure_heif()
        with Image.open(path) as pil_img:
            rgb = pil_img.convert("RGB")
            arr = np.asarray(rgb)
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        # Fallback for formats OpenCV may miss
        with Image.open(path) as pil_img:
            rgb = pil_img.convert("RGB")
            arr = np.asarray(rgb)
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return image
