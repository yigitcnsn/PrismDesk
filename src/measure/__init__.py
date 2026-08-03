"""Photo-based mat detection and edge measurement utilities."""

from .geometry import path_length_cm, segment_lengths_cm, to_cm
from .io import load_image
from .mat import MatConfig, detect_mat_corners, load_mat_config
from .object import analyze_object, detect_object_outline
from .outline import OutlineResult, OutlineSession
from .perspective import warp_to_mat_plane
from .shape import ObjectAnalysis, analyze_silhouette

__all__ = [
    "MatConfig",
    "ObjectAnalysis",
    "OutlineResult",
    "OutlineSession",
    "analyze_object",
    "analyze_silhouette",
    "detect_mat_corners",
    "detect_object_outline",
    "load_image",
    "load_mat_config",
    "path_length_cm",
    "segment_lengths_cm",
    "to_cm",
    "warp_to_mat_plane",
]
