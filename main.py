"""
PrismDesk entry point.

Modes:
  measure           Photo edge measurement (existing)
  calibrate-camera  Chessboard fisheye/pinhole calibration for USB cam
  hands             Live hand tracking (optional --project HUD on HY300)
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

    hands = sub.add_parser(
        "hands",
        help="Live MediaPipe hand tracking (optionally project HUD to HY300)",
    )
    hands.add_argument(
        "--camera-config",
        type=Path,
        default=DEFAULT_CAMERA_CONFIG,
        help="Camera YAML with optional calibration",
    )
    hands.add_argument("--device", type=int, default=None, help="Force V4L2 index")
    hands.add_argument("--no-undistort", action="store_true")
    hands.add_argument(
        "--project",
        action="store_true",
        help="Project hand HUD fullscreen on HY300 (dark canvas + skeleton)",
    )
    hands.add_argument(
        "--projector-config",
        type=Path,
        default=DEFAULT_PROJECTOR_CONFIG,
        help="Projector YAML (default: config/projector.yaml)",
    )
    hands.add_argument(
        "--output",
        default=None,
        help="Override projector output name (e.g. HDMI-A-1)",
    )
    hands.add_argument(
        "--show",
        choices=("auto", "mpv", "opencv"),
        default="auto",
        help="Projector sink when --project (default: auto → mpv then OpenCV)",
    )
    hands.add_argument(
        "--preview",
        action="store_true",
        help="Also show local OpenCV camera preview (off by default with --project)",
    )
    hands.add_argument(
        "--no-preview",
        action="store_true",
        help="Never open local OpenCV preview window",
    )
    hands.add_argument(
        "--track-size",
        default="640x360",
        help="MediaPipe inference size WxH (default 640x360; use full for camera native)",
    )
    hands.add_argument(
        "--capture",
        default=None,
        help="Optional camera capture WxH override (e.g. 1280x720) for more FPS",
    )
    hands.add_argument(
        "--hud-size",
        default="1280x720",
        help="Projector HUD / video-sink size WxH (default 1280x720; use full for projector native)",
    )

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
    proj.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Write pattern PNG (no GUI required)",
    )
    proj.add_argument(
        "--show",
        choices=("opencv", "mpv", "feh", "auto"),
        default="auto",
        help="How to display: auto tries mpv then OpenCV (default: auto)",
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


def _parse_size(text: str | None, *, allow_full: bool = True) -> tuple[int, int] | None:
    """Parse '1920x1080' or 'full' / '0x0' → None (native)."""
    if text is None:
        return None
    s = str(text).strip().lower()
    if allow_full and s in ("full", "native", "0", "0x0"):
        return None
    if "x" not in s:
        raise ValueError(f"size must look like 640x360, got {text!r}")
    a, b = s.split("x", 1)
    w, h = int(a), int(b)
    if w <= 0 or h <= 0:
        return None
    return w, h


def cmd_hands(args: argparse.Namespace) -> int:
    import cv2
    import numpy as np

    from src.vision.camera import Camera
    from src.vision.hands import HandTracker
    from src.vision.projector import (
        MpvFrameSink,
        ProjectorSurface,
        ensure_gui_env,
        opencv_gui_hint,
    )
    from src.vision.undistort import Undistorter

    cfg = _load_or_bootstrap_camera_config(args.camera_config)
    if args.device is not None:
        cfg.device_indices = [args.device] + [i for i in cfg.device_indices if i != args.device]
    try:
        capture = _parse_size(args.capture, allow_full=True) if args.capture else None
        track_size = _parse_size(args.track_size, allow_full=True)
        hud_size = _parse_size(args.hud_size, allow_full=True)
    except ValueError as exc:
        print(exc)
        return 1
    if capture is not None:
        cfg.width, cfg.height = capture

    und = Undistorter(cfg)
    if args.no_undistort:
        from src.vision.camera import CameraConfig as CC

        und = Undistorter(CC())  # no K/D → passthrough

    project = bool(args.project)
    # With --project, skip local OpenCV window unless --preview (Qt/xcb often aborts on Pi).
    want_preview = (not project and not args.no_preview) or (project and args.preview)
    if args.no_preview:
        want_preview = False

    proj_cfg = None
    surface: ProjectorSurface | None = None
    mpv: MpvFrameSink | None = None
    proj_w = proj_h = 0
    hud_w = hud_h = 0
    show = "mpv"
    if project:
        proj_cfg = _load_or_bootstrap_projector_config(args.projector_config)
        if args.output:
            proj_cfg.output_name = args.output
        env = ensure_gui_env()
        if env.get("fixed"):
            print("auto-set:", ", ".join(env["fixed"]))
        surface = ProjectorSurface(proj_cfg)
        info = surface.prepare()
        proj_w, proj_h = int(proj_cfg.width), int(proj_cfg.height)
        if hud_size is None:
            hud_w, hud_h = proj_w, proj_h
        else:
            hud_w, hud_h = hud_size
        print(
            f"Projector: {info.name} {info.width}x{info.height}@{info.refresh_hz:.3f}Hz "
            f"hud={hud_w}x{hud_h} source={info.source}"
        )
        show = args.show if args.show != "auto" else "mpv"

    cam = Camera(cfg)
    tracker = HandTracker(infer_size=track_size)
    window = "prismdesk-hands"
    if want_preview:
        try:
            cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        except Exception as exc:  # noqa: BLE001
            print(f"local preview unavailable: {exc}")
            print(opencv_gui_hint())
            want_preview = False
            if not project:
                return 1

    idx = cam.open()
    w, h, fps = cam.negotiated()
    track_label = (
        f"{tracker.infer_size[0]}x{tracker.infer_size[1]}"
        if tracker.infer_size
        else f"{w}x{h} (full)"
    )
    print(
        f"Hands: camera index={idx} negotiated={w}x{h}@{fps:.1f} "
        f"track={track_label} "
        f"undistort={'on' if und.enabled else 'off (calibrate-camera first)'} "
        f"project={'on' if project else 'off'} preview={'on' if want_preview else 'off'}"
    )

    # Start video sink only after model + camera are ready.
    if project:
        sink_fps = float(fps) if fps and fps > 1 else float(cfg.fps or 30)
        if show == "mpv":
            try:
                mpv = MpvFrameSink(hud_w, hud_h, fps=sink_fps)
            except Exception as exc:  # noqa: BLE001
                print(f"video sink failed: {exc}")
                print(
                    "OpenCV fullscreen is disabled by default on Pi "
                    "(pip Qt/xcb aborts). Fix the sink or install: sudo apt install ffmpeg"
                )
                print(opencv_gui_hint())
                tracker.close()
                cam.close()
                return 1
        elif show == "opencv":
            # Explicit only — pip OpenCV often aborts with no catchable exception.
            try:
                surface.open()
            except Exception as exc:  # noqa: BLE001
                print(opencv_gui_hint())
                print(f"error: {exc}")
                tracker.close()
                cam.close()
                return 1
        print("HUD uses stretch mapping (cam↔projector homography later). Ctrl+C or q quit")
    else:
        print("q quit")

    frames = 0
    t0 = time.time()
    hud = None
    try:
        while True:
            frame = cam.read()
            if und.enabled and not args.no_undistort:
                frame = und.apply(frame)
            hands = tracker.process(frame)
            frames += 1
            elapsed = max(time.time() - t0, 1e-6)
            fps_live = frames / elapsed

            if project:
                if hud is None or hud.shape[:2] != (hud_h, hud_w):
                    hud = np.zeros((hud_h, hud_w, 3), dtype=np.uint8)
                else:
                    hud[:] = 0
                tracker.draw_hud(hud, hands, src_size=(frame.shape[1], frame.shape[0]))
                cv2.putText(
                    hud,
                    f"fps={fps_live:.1f}  hands={len(hands)}  track={track_label}",
                    (24, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 0),
                    2,
                )
                if mpv is not None:
                    mpv.show(hud)
                elif surface is not None:
                    surface.show(hud)

            if want_preview:
                view = tracker.draw(frame, hands)
                cv2.putText(
                    view,
                    f"idx={idx}  fps={fps_live:.1f}  hands={len(hands)}  "
                    f"track={track_label}  q=quit",
                    (12, 32),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )
                cv2.imshow(window, view)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            elif mpv is not None and not mpv.alive:
                print("mpv closed — exiting")
                break
            else:
                time.sleep(0.001)
    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        tracker.close()
        cam.close()
        if mpv is not None:
            mpv.close()
        if surface is not None:
            surface.close()
        if want_preview:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
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
    from src.vision.projector import (
        active_discovery_source,
        discovery_backend,
        ensure_gui_env,
        list_outputs,
    )

    env = ensure_gui_env()
    print(f"discovery binaries: {discovery_backend()}")
    print(f"active source: {active_discovery_source()}")
    print(
        f"session DISPLAY={env.get('DISPLAY')!r} "
        f"WAYLAND_DISPLAY={env.get('WAYLAND_DISPLAY')!r} "
        f"XDG_RUNTIME_DIR={env.get('XDG_RUNTIME_DIR')!r}"
    )
    if env.get("fixed"):
        print("auto-set:", ", ".join(env["fixed"]))
    outputs = list_outputs()
    if not outputs:
        print("No connected outputs found.")
        return 1
    for out in outputs:
        print(
            f"{out.name}: {out.width}x{out.height}@{out.refresh_hz:.3f}Hz "
            f"pos=({out.x},{out.y}) enabled={out.enabled} "
            f"source={out.source} make={out.make!r}"
        )
        for w, h, hz in out.modes[:8]:
            mark = (
                " *"
                if (w, h) == (out.width, out.height) and (hz == 0 or abs(hz - out.refresh_hz) < 0.01)
                else ""
            )
            print(f"  mode {w}x{h}@{hz:.3f}Hz{mark}")
        if len(out.modes) > 8:
            print(f"  … {len(out.modes) - 8} more modes")
    return 0


def cmd_projector_test(args: argparse.Namespace) -> int:
    import cv2

    from src.vision.projector import (
        ProjectorSurface,
        ensure_gui_env,
        make_alignment_pattern,
        opencv_gui_hint,
        show_image_external,
    )

    cfg = _load_or_bootstrap_projector_config(args.projector_config)
    if args.output:
        cfg.output_name = args.output

    env = ensure_gui_env()
    if env.get("fixed"):
        print("auto-set:", ", ".join(env["fixed"]))
    print(
        f"session DISPLAY={env.get('DISPLAY')!r} "
        f"WAYLAND_DISPLAY={env.get('WAYLAND_DISPLAY')!r}"
    )

    surface = ProjectorSurface(cfg)
    info = surface.prepare()
    print(
        f"Projector: {info.name} {info.width}x{info.height}@{info.refresh_hz:.3f}Hz "
        f"pos=({info.x},{info.y}) source={info.source}"
    )
    pattern = make_alignment_pattern(cfg.width, cfg.height)

    save_path = Path(args.save) if args.save else Path("/tmp/prismdesk-projector-test.png")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), pattern)
    print(f"Wrote pattern: {save_path}")

    if args.show in ("mpv", "feh", "auto"):
        try:
            tool = "auto" if args.show == "auto" else args.show
            return int(show_image_external(save_path, tool=tool))
        except Exception as exc:  # noqa: BLE001
            print(f"external show failed: {exc}")
            if args.show != "auto":
                print(opencv_gui_hint())
                return 1
            print("falling back to OpenCV window…")

    try:
        surface.open()
        print("Showing alignment pattern — press q to quit")
        while True:
            surface.show(pattern)
            if cv2.waitKey(50) & 0xFF == ord("q"):
                break
    except cv2.error as exc:
        print(opencv_gui_hint())
        print(f"cv2 error: {exc}")
        return 1
    except Exception as exc:  # noqa: BLE001
        print(opencv_gui_hint())
        print(f"error: {exc}")
        return 1
    finally:
        surface.close()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
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
