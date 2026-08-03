"""Photo-based mat detection and edge measurement utilities."""

from .geometry import path_length_cm, segment_lengths_cm, to_cm
from .io import load_image
from .mat import MatConfig, detect_mat_corners, load_mat_config
from .outline import OutlineResult, OutlineSession
from .perspective import warp_to_mat_plane

__all__ = [
    "MatConfig",
    "OutlineResult",
    "OutlineSession",
    "detect_mat_corners",
    "load_image",
    "load_mat_config",
    "path_length_cm",
    "segment_lengths_cm",
    "to_cm",
    "warp_to_mat_plane",
]
