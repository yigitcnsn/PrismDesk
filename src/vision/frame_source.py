"""Frame sources that look like Camera: live USB or still images from disk."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

from src.measure.io import load_image

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".heic", ".heif"}


def list_example_images(folder: str | Path) -> List[Path]:
    """Sorted image paths under folder (non-recursive)."""
    root = Path(folder)
    if not root.is_dir():
        return []
    files = [
        p
        for p in root.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    ]
    return sorted(files, key=lambda p: p.name.lower())


class ImageFolderSource:
    """Replay stills as a Camera-like source for local debug without the Pi."""

    def __init__(self, paths: Sequence[str | Path]) -> None:
        self._paths = [Path(p) for p in paths]
        if not self._paths:
            raise ValueError("ImageFolderSource needs at least one image path")
        self._index = 0
        self._frame: Optional[np.ndarray] = None
        self._active = False

    @classmethod
    def from_folder(cls, folder: str | Path) -> "ImageFolderSource":
        paths = list_example_images(folder)
        if not paths:
            raise FileNotFoundError(
                f"No images in {folder} "
                f"(supported: {', '.join(sorted(IMAGE_SUFFIXES))})"
            )
        return cls(paths)

    @property
    def path(self) -> Path:
        return self._paths[self._index]

    @property
    def index(self) -> int:
        return self._index

    @property
    def count(self) -> int:
        return len(self._paths)

    @property
    def active_index(self) -> Optional[int]:
        return self._index if self._active else None

    def open(self) -> int:
        self._load(self._index)
        self._active = True
        return self._index

    def _load(self, index: int) -> None:
        self._index = int(index) % len(self._paths)
        self._frame = load_image(self._paths[self._index])

    def next(self) -> Path:
        self._load(self._index + 1)
        return self.path

    def prev(self) -> Path:
        self._load(self._index - 1)
        return self.path

    def goto(self, index: int) -> Path:
        self._load(index)
        return self.path

    def read(self) -> np.ndarray:
        if self._frame is None:
            self.open()
        assert self._frame is not None
        return self._frame.copy()

    def negotiated(self) -> Tuple[int, int, float]:
        if self._frame is None:
            raise RuntimeError("ImageFolderSource not open")
        h, w = self._frame.shape[:2]
        return w, h, 1.0

    def close(self) -> None:
        self._frame = None
        self._active = False

    def __enter__(self) -> "ImageFolderSource":
        self.open()
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
