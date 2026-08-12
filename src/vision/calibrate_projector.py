"""Project a chessboard on the HY300 and solve camera↔projector homography.

No OpenCV GUI windows (pip Qt/xcb aborts on Pi Wayland). Pattern goes to
ffplay/mpv; control is terminal keys + optional auto-sample.
"""

from __future__ import annotations

import select
import sys
import termios
import time
import tty
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .camera import Camera, CameraConfig
from .homography import (
    CamProjectorHomography,
    estimate_homography,
    find_projected_chessboard,
    make_chessboard_pattern,
)
from .projector import (
    MpvFrameSink,
    ProjectorConfig,
    ProjectorSurface,
    ensure_gui_env,
    opencv_gui_hint,
    save_projector_config,
)
from .undistort import Undistorter


class _TerminalKeys:
    """Non-blocking single-key reads in cbreak mode (no Enter needed)."""

    def __init__(self) -> None:
        self._fd: Optional[int] = None
        self._old: Optional[list] = None
        self._ok = False

    def __enter__(self) -> "_TerminalKeys":
        if not sys.stdin.isatty():
            print("stdin is not a TTY — use --auto-save (no interactive keys)")
            return self
        self._fd = sys.stdin.fileno()
        self._old = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        self._ok = True
        return self

    def __exit__(self, *args: object) -> None:
        if self._fd is not None and self._old is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old)

    def poll(self) -> Optional[str]:
        if not self._ok or self._fd is None:
            return None
        if select.select([sys.stdin], [], [], 0)[0]:
            ch = sys.stdin.read(1)
            return ch if ch else None
        return None


def _annotate_cam(
    frame: np.ndarray,
    *,
    board_size: Tuple[int, int],
    found: bool,
    corners: Optional[np.ndarray],
    samples: int,
    stable: int,
    stable_need: int,
) -> np.ndarray:
    cols, rows = board_size
    display = frame.copy()
    if found and corners is not None:
        cv2.drawChessboardCorners(
            display,
            (cols, rows),
            corners.reshape(-1, 1, 2).astype(np.float32),
            True,
        )
    status = "LOCK" if found else "seeking"
    cv2.putText(
        display,
        f"{status}  samples={samples}  stable={stable}/{stable_need}  "
        f"[space]=sample  [c]=save  [q]=quit",
        (12, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255) if found else (0, 165, 255),
        2,
        cv2.LINE_AA,
    )
    return display


def run_projector_homography_calibration(
    camera_config: CameraConfig,
    projector_config: ProjectorConfig,
    output_path: str | Path,
    *,
    board_size: Tuple[int, int] = (9, 6),
    show: str = "mpv",
    cam_preview: bool = False,
    auto_save: bool = False,
    stable_frames: int = 20,
    no_undistort: bool = False,
) -> CamProjectorHomography:
    """Projected-chessboard calib without OpenCV windows.

    Keys (terminal): SPACE sample | c compute+save | q cancel
    After ``stable_frames`` consecutive detections, auto-captures one sample.
    With ``auto_save``, saves as soon as one sample exists.
    Optional ``cam_preview`` opens a *windowed* ffplay/mpv of the annotated
    camera (not fullscreen — that would cover the projected pattern).
    """
    cols, rows = board_size
    pattern, proj_corners = make_chessboard_pattern(
        projector_config.width,
        projector_config.height,
        board_size,
    )

    und = Undistorter(camera_config)
    if no_undistort:
        und = Undistorter(CameraConfig())

    env = ensure_gui_env()
    if env.get("fixed"):
        print("auto-set:", ", ".join(env["fixed"]))

    surface = ProjectorSurface(projector_config)
    info = surface.prepare()
    print(
        f"Projector: {info.name} {info.width}x{info.height}@{info.refresh_hz:.3f}Hz "
        f"canvas={projector_config.width}x{projector_config.height} source={info.source}"
    )
    print(
        f"Pattern: chessboard inner corners {cols}x{rows} "
        f"(full projector canvas visible in camera)"
    )

    mpv: MpvFrameSink | None = None
    cam_sink: MpvFrameSink | None = None
    use_opencv_proj = False
    show_mode = show if show != "auto" else "mpv"

    if show_mode == "opencv":
        print(
            "WARNING: --show opencv often aborts on Pi (pip Qt/xcb). Prefer default mpv."
        )
        try:
            surface.open()
            surface.show(pattern)
            use_opencv_proj = True
        except Exception as exc:  # noqa: BLE001
            print(opencv_gui_hint())
            print(f"error: {exc}")
            raise
    else:
        try:
            mpv = MpvFrameSink(
                projector_config.width,
                projector_config.height,
                fps=float(projector_config.refresh_hz or 30),
            )
            mpv.show(pattern)
        except Exception as exc:  # noqa: BLE001
            print(f"video sink failed: {exc}")
            print(opencv_gui_hint())
            raise RuntimeError(
                "Need ffplay/mpv for projected pattern (OpenCV window disabled by default)"
            ) from exc

    cam = Camera(camera_config)
    idx = cam.open()
    cam_w, cam_h, cam_fps = cam.negotiated()
    print(
        f"Camera: index={idx} negotiated={cam_w}x{cam_h}@{cam_fps:.1f} "
        f"undistort={'on' if und.enabled else 'off'}"
    )

    if cam_preview:
        # Windowed ffplay/mpv only — fullscreen would cover the projected board.
        try:
            cam_sink = MpvFrameSink(
                cam_w,
                cam_h,
                fps=float(cam_fps) if cam_fps > 1 else 30.0,
                fullscreen=False,
            )
            print("Camera preview: windowed ffplay/mpv (annotated). No OpenCV window.")
        except Exception as exc:  # noqa: BLE001
            print(f"camera preview sink unavailable: {exc} — continuing terminal-only")
            cam_sink = None
    else:
        print("Camera preview: off (terminal status). Pass --cam-preview for windowed view.")

    stable_need = max(1, int(stable_frames))
    print(
        "Controls (terminal): SPACE=sample  c=save  q=quit  "
        f"(auto-sample after {stable_need} locked frames"
        + ("; --auto-save on" if auto_save else "")
        + ")"
    )

    cam_samples: List[np.ndarray] = []
    proj_samples: List[np.ndarray] = []
    image_size: Optional[Tuple[int, int]] = None
    stable = 0
    frames = 0
    last_status = ""
    t_status = 0.0
    done = False

    try:
        with _TerminalKeys() as keys:
            while not done:
                if mpv is not None:
                    mpv.show(pattern)
                    if not mpv.alive:
                        raise SystemExit("projector video sink closed")
                elif use_opencv_proj:
                    surface.show(pattern)

                frame = und.apply(cam.read())
                image_size = (frame.shape[1], frame.shape[0])
                found, corners = find_projected_chessboard(frame, board_size)
                frames += 1

                if found and corners is not None:
                    stable += 1
                else:
                    stable = 0

                # Auto-sample once when lock is stable.
                if (
                    found
                    and corners is not None
                    and stable >= stable_need
                    and len(cam_samples) == 0
                ):
                    cam_samples.append(corners.copy())
                    proj_samples.append(proj_corners.copy())
                    print(f"  auto-captured sample {len(cam_samples)} (stable lock)")
                    if auto_save:
                        done = True

                display = _annotate_cam(
                    frame,
                    board_size=board_size,
                    found=found,
                    corners=corners,
                    samples=len(cam_samples),
                    stable=stable,
                    stable_need=stable_need,
                )
                if cam_sink is not None:
                    cam_sink.show(display)
                    if not cam_sink.alive:
                        print("camera preview closed — continuing without it")
                        cam_sink.close()
                        cam_sink = None

                now = time.time()
                status = (
                    f"board={'yes' if found else 'no'}  stable={stable}/{stable_need}  "
                    f"samples={len(cam_samples)}  frame={frames}"
                )
                if status != last_status and (now - t_status) > 0.5:
                    print(status, flush=True)
                    last_status = status
                    t_status = now

                ch = keys.poll()
                if ch is None:
                    continue
                if ch in ("q", "Q", "\x03"):
                    raise SystemExit("Projector homography calibration cancelled")
                if ch == " " and found and corners is not None:
                    cam_samples.append(corners.copy())
                    proj_samples.append(proj_corners.copy())
                    print(f"  captured sample {len(cam_samples)}")
                    if auto_save:
                        done = True
                elif ch in ("c", "C"):
                    if len(cam_samples) < 1:
                        print("Need at least 1 sample (wait for LOCK / press SPACE)")
                        continue
                    done = True
    finally:
        cam.close()
        if cam_sink is not None:
            cam_sink.close()
        if mpv is not None:
            mpv.close()
        if use_opencv_proj:
            surface.close()

    if not cam_samples:
        raise SystemExit("No samples captured — board never locked")
    assert image_size is not None
    cam_pts = np.concatenate(cam_samples, axis=0)
    proj_pts = np.concatenate(proj_samples, axis=0)
    h_mat, mean_err, inliers = estimate_homography(cam_pts, proj_pts)
    result = CamProjectorHomography(
        matrix=h_mat,
        cam_size=image_size,
        proj_size=(projector_config.width, projector_config.height),
        reprojection_error_px=mean_err,
    )
    projector_config.homography = result
    save_projector_config(output_path, projector_config)
    print(
        f"Saved cam↔proj H  mean_reproj={mean_err:.2f}px  inliers≈{inliers} "
        f"cam={image_size[0]}x{image_size[1]} "
        f"proj={projector_config.width}x{projector_config.height} → {output_path}"
    )
    return result
