"""
PrismDesk entry point.

Runs the photo edge measurement utility:
load photo → find mat → warp → analyze object (silhouette/colors/shape) → UI.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.measure.io import load_image
from src.measure.mat import confirm_or_override_corners, detect_mat_corners, load_mat_config
from src.measure.object import analyze_object
from src.measure.outline import OutlineSession
from src.measure.perspective import warp_to_mat_plane


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure object edges on a known-size mat from a photo.",
    )
    parser.add_argument(
        "photo",
        type=Path,
        help="Path to a photo (HEIC/JPEG/PNG) of the mat on the table",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config" / "mat.yaml",
        help="Path to mat config YAML (default: config/mat.yaml)",
    )
    parser.add_argument(
        "--auto-accept",
        action="store_true",
        help="Accept detected mat corners without confirmation UI",
    )
    parser.add_argument(
        "--no-auto-object",
        action="store_true",
        help="Skip automatic object outline detection (manual clicks only)",
    )
    return parser.parse_args()


def _print_analysis(result) -> None:
    analysis = result.analysis
    if analysis is None:
        label = "Edge lengths (cm)" if result.closed else "Segment lengths (cm)"
        print(f"{label}:")
        for i, length in enumerate(result.segment_cm, start=1):
            print(f"  E{i}: {length:.2f}")
        print(f"Total path: {result.total_cm:.2f} cm")
        return

    print(f"Shape: {analysis.shape}")
    if analysis.colors:
        print("Colors: " + ", ".join(analysis.colors))

    if analysis.shape == "circle":
        print(f"  radius: {analysis.radius_cm:.2f} cm")
        print(f"  diameter: {analysis.diameter_cm:.2f} cm")
        return

    if analysis.shape == "thin":
        print(f"  length: {analysis.length_cm:.2f} cm")
        print(f"  width: {analysis.width_cm:.2f} cm")

    if analysis.edge_cm:
        print("Edge lengths (cm):")
        for i, length in enumerate(analysis.edge_cm, start=1):
            print(f"  E{i}: {length:.2f}")

    if analysis.fillet_radii_cm:
        print("Fillet radii (cm):")
        for i, fr in enumerate(analysis.fillet_radii_cm, start=1):
            print(f"  F{i}: {fr:.2f}")

    if result.segment_cm and not analysis.edge_cm:
        print("Segment lengths (cm):")
        for i, length in enumerate(result.segment_cm, start=1):
            print(f"  E{i}: {length:.2f}")
        print(f"Total path: {result.total_cm:.2f} cm")


def main() -> int:
    args = parse_args()
    config = load_mat_config(args.config)
    image = load_image(args.photo)

    detected = detect_mat_corners(image, config)
    if args.auto_accept and detected is not None:
        corners = detected
        print("Using auto-detected mat corners.")
    else:
        corners = confirm_or_override_corners(image, detected)
        if corners is None:
            print("Cancelled: no mat corners selected.")
            return 1

    warped, px_per_cm = warp_to_mat_plane(image, corners, config)
    print(
        f"Warped mat plane: {warped.shape[1]}x{warped.shape[0]} px "
        f"({config.width_cm}x{config.height_cm} cm @ {px_per_cm} px/cm)"
    )

    analysis = None
    if not args.no_auto_object:
        analysis = analyze_object(warped, config)
        if analysis:
            print(
                f"Auto-found {analysis.shape} "
                f"({len(analysis.outline_points)} outline pts"
                + (f", colors={analysis.colors}" if analysis.colors else "")
                + ")."
            )
        else:
            print("Auto-find: no object silhouette detected (click manually or press a).")

    result = OutlineSession(
        warped,
        px_per_cm,
        config=config,
        analysis=analysis,
    ).run()
    if result is None:
        print("Cancelled: no outline measured.")
        return 1

    _print_analysis(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
