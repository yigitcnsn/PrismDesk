"""
PrismDesk entry point.

Modes:
  measure           Photo edge measurement (existing)
  calibrate-camera  Chessboard fisheye/pinhole calibration for USB cam
  hands             Live hand tracking preview (V4L2 MJPG → undistort → MediaPipe)
  projector-list    List Wayland outputs via wlr-randr
  projector-test    Fullscreen alignment pattern on HY300 (HDMI-A-1)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_CAMERA_CONFIG = ROOT / "config" / "camera.yaml"
EXAMPLE_CAMERA_CONFIG = ROOT / "config" / "camera.example.yaml"
DEFAULT_PROJECTOR_CONFIG = ROOT / "config" / "projector.yaml"
EXAMPLE_PROJECTOR_CONFIG = ROOT / "config" / "projector.example.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PrismDesk")
    sub = parser.add_subparsers(dest="command", required=True)

    measure = sub.add_parser("measure", help="Measure object edges from a photo")
    measure.add_argument("photo", type=Path, help="Photo path (HEIC/JPEG/PNG)")
    measure.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config" / "mat.yaml",
        help="Mat config YAML",
    )
    measure.add_argument("--auto-accept", action="store_true")
    measure.add_argument("--no-auto-object", action="store_true")

    calib = sub.add_parser("calibrate-camera", help="Calibrate USB camera (chessboard)")
    calib.add_argument(
        "--camera-config",
        type=Path,
        default=DEFAULT_CAMERA_CONFIG,
        help="Input/output camera YAML (default: config/camera.yaml)",
    )
    calib.add_argument("--device", type=int, default=None, help="Force V4L2 index")
    calib.add_argument("--board", default="9x6", help="Inner corners WxH (default 9x6)")
    calib.add_argument("--samples", type=int, default=20)
    calib.add_argument(
        "--model",
        choices=("fisheye", "pinhole"),
        default=None,
        help="Override distortion model",
    )

    hands = sub.add_parser("hands", help="Live MediaPipe hand tracking preview")
    hands.add_argument(
        "--camera-config",
        type=Path,
        default=DEFAULT_CAMERA_CONFIG,
        help="Camera YAML with optional calibration",
    )
    hands.add_argument("--device", type=int, default=None, help="Force V4L2 index")
    hands.add_argument("--no-undistort", action="store_true")

    sub.add_parser("projector-list", help="List Wayland outputs (wlr-randr)")

    proj = sub.add_parser("projector-test", help="Fullscreen test pattern on HY300")
    proj.add_argument(
        "--projector-config",
        type=Path,
        default=DEFAULT_PROJECTOR_CONFIG,
        help="Projector YAML (default: config/projector.yaml)",
    )
    proj.add_argument(
        "--output",
        default=None,
        help="Override output name (default from config: HDMI-A-1)",
    )
    return parser


def cmd_measure(args: argparse.Namespace) -> int:
    from src.measure.io import load_image
    from src.measure.mat import confirm_or_override_corners, detect_mat_corners, load_mat_config
    from src.measure.object import analyze_object
    from src.measure.outline import OutlineSession
    from src.measure.perspective import warp_to_mat_plane

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

    analysis = result.analysis
    if analysis is None:
        print("Segment lengths (cm):")
        for i, length in enumerate(result.segment_cm, start=1):
            print(f"  E{i}: {length:.2f}")
        print(f"Total path: {result.total_cm:.2f} cm")
        return 0

    print(f"Shape: {analysis.shape}")
    if analysis.colors:
        print("Colors: " + ", ".join(analysis.colors))
    if analysis.shape == "circle":
        print(f"  radius: {analysis.radius_cm:.2f} cm")
        print(f"  diameter: {analysis.diameter_cm:.2f} cm")
    elif analysis.shape == "thin":
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
    return 0


def _load_or_bootstrap_camera_config(path: Path):
    from src.vision.camera import CameraConfig, load_camera_config

    if path.is_file():
        return load_camera_config(path)
    if EXAMPLE_CAMERA_CONFIG.is_file():
        print(f"{path} missing — loading defaults from {EXAMPLE_CAMERA_CONFIG}")
        return load_camera_config(EXAMPLE_CAMERA_CONFIG)
    return CameraConfig()


def cmd_calibrate_camera(args: argparse.Namespace) -> int:
    from src.vision.calibrate import run_calibration

    cfg = _load_or_bootstrap_camera_config(args.camera_config)
    if args.device is not None:
        cfg.device_indices = [args.device] + [i for i in cfg.device_indices if i != args.device]
    w_s, h_s = args.board.lower().split("x")
    run_calibration(
        cfg,
        output_path=args.camera_config,
        board_size=(int(w_s), int(h_s)),
        target_samples=args.samples,
        model=args.model,
    )
    return 0


def cmd_hands(args: argparse.Namespace) -> int:
    import cv2

    from src.vision.camera import Camera
    from src.vision.hands import HandTracker
    from src.vision.undistort import Undistorter

    cfg = _load_or_bootstrap_camera_config(args.camera_config)
    if args.device is not None:
        cfg.device_indices = [args.device] + [i for i in cfg.device_indices if i != args.device]

    und = Undistorter(cfg)
    if args.no_undistort:
        from src.vision.camera import CameraConfig as CC

        und = Undistorter(CC())  # no K/D → passthrough

    cam = Camera(cfg)
    tracker = HandTracker()
    window = "prismdesk-hands"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    idx = cam.open()
    w, h, fps = cam.negotiated()
    print(
        f"Hands: camera index={idx} negotiated={w}x{h}@{fps:.1f} "
        f"undistort={'on' if und.enabled else 'off (calibrate-camera first)'}"
    )
    print("q quit")

    frames = 0
    t0 = time.time()
    try:
        while True:
            frame = cam.read()
            if und.enabled and not args.no_undistort:
                frame = und.apply(frame)
            hands = tracker.process(frame)
            view = tracker.draw(frame, hands)
            frames += 1
            elapsed = max(time.time() - t0, 1e-6)
            fps_live = frames / elapsed
            cv2.putText(
                view,
                f"idx={idx}  fps={fps_live:.1f}  hands={len(hands)}  q=quit",
                (12, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )
            cv2.imshow(window, view)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        tracker.close()
        cam.close()
        cv2.destroyAllWindows()
    return 0


def _load_or_bootstrap_projector_config(path: Path):
    from src.vision.projector import ProjectorConfig, load_projector_config

    if path.is_file():
        return load_projector_config(path)
    if EXAMPLE_PROJECTOR_CONFIG.is_file():
        print(f"{path} missing — loading defaults from {EXAMPLE_PROJECTOR_CONFIG}")
        return load_projector_config(EXAMPLE_PROJECTOR_CONFIG)
    return ProjectorConfig()


def cmd_projector_list(_args: argparse.Namespace) -> int:
    from src.vision.projector import list_outputs, wlr_randr_available

    if not wlr_randr_available():
        print("wlr-randr not found. Install it on Pi OS Wayland/labwc, or run under a Wayland session.")
        return 1
    outputs = list_outputs()
    if not outputs:
        print("No outputs parsed from wlr-randr.")
        return 1
    for out in outputs:
        print(
            f"{out.name}: {out.width}x{out.height}@{out.refresh_hz:.3f}Hz "
            f"pos=({out.x},{out.y}) enabled={out.enabled} make={out.make!r}"
        )
        for w, h, hz in out.modes[:8]:
            mark = " *" if (w, h) == (out.width, out.height) and abs(hz - out.refresh_hz) < 0.01 else ""
            print(f"  mode {w}x{h}@{hz:.3f}Hz{mark}")
        if len(out.modes) > 8:
            print(f"  … {len(out.modes) - 8} more modes")
    return 0


def cmd_projector_test(args: argparse.Namespace) -> int:
    import cv2

    from src.vision.projector import ProjectorSurface, make_alignment_pattern

    cfg = _load_or_bootstrap_projector_config(args.projector_config)
    if args.output:
        cfg.output_name = args.output

    surface = ProjectorSurface(cfg)
    try:
        info = surface.prepare()
        print(
            f"Projector: {info.name} {info.width}x{info.height}@{info.refresh_hz:.3f}Hz "
            f"pos=({info.x},{info.y})"
        )
        pattern = make_alignment_pattern(cfg.width, cfg.height)
        surface.open()
        print("Showing alignment pattern on projector — press q in the window to quit")
        while True:
            surface.show(pattern)
            if cv2.waitKey(50) & 0xFF == ord("q"):
                break
    finally:
        surface.close()
        cv2.destroyAllWindows()
    return 0


def main() -> int:
    # Backward compatible: `python main.py photo.HEIC` still works as measure
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        known = {
            "measure",
            "calibrate-camera",
            "hands",
            "projector-list",
            "projector-test",
        }
        if sys.argv[1] not in known:
            sys.argv.insert(1, "measure")

    parser = build_parser()
    args = parser.parse_args()
    if args.command == "measure":
        return cmd_measure(args)
    if args.command == "calibrate-camera":
        return cmd_calibrate_camera(args)
    if args.command == "hands":
        return cmd_hands(args)
    if args.command == "projector-list":
        return cmd_projector_list(args)
    if args.command == "projector-test":
        return cmd_projector_test(args)
    parser.error(f"Unknown command {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
