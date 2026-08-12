"""Project a chessboard on the HY300 and solve camera↔projector homography."""

from __future__ import annotations

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


def run_projector_homography_calibration(
    camera_config: CameraConfig,
    projector_config: ProjectorConfig,
    output_path: str | Path,
    *,
    board_size: Tuple[int, int] = (9, 6),
    show: str = "mpv",
    preview: bool = True,
    no_undistort: bool = False,
) -> CamProjectorHomography:
    """Interactive projected-chessboard calib.

    Assumes the full projector canvas is visible in the camera frame.
    Keys (camera preview): SPACE sample | c compute+save | q cancel
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
    use_opencv_proj = False
    show_mode = show if show != "auto" else "mpv"
    if show_mode == "mpv":
        try:
            mpv = MpvFrameSink(
                projector_config.width,
                projector_config.height,
                fps=float(projector_config.refresh_hz or 30),
            )
            mpv.show(pattern)
        except Exception as exc:  # noqa: BLE001
            print(f"mpv sink failed: {exc}")
            print("falling back to OpenCV projector window…")
            show_mode = "opencv"
    if show_mode == "opencv":
        try:
            surface.open()
            surface.show(pattern)
            use_opencv_proj = True
        except Exception as exc:  # noqa: BLE001
            print(opencv_gui_hint())
            print(f"error: {exc}")
            raise

    cam = Camera(camera_config)
    idx = cam.open()
    cam_w, cam_h, cam_fps = cam.negotiated()
    print(
        f"Camera: index={idx} negotiated={cam_w}x{cam_h}@{cam_fps:.1f} "
        f"undistort={'on' if und.enabled else 'off'}"
    )
    print("SPACE=sample when corners found  c=save  q=quit")

    preview_win = "calibrate-projector"
    want_preview = preview
    if want_preview:
        try:
            cv2.namedWindow(preview_win, cv2.WINDOW_NORMAL)
        except Exception as exc:  # noqa: BLE001
            print(f"camera preview window unavailable: {exc}")
            want_preview = False

    cam_samples: List[np.ndarray] = []
    proj_samples: List[np.ndarray] = []
    image_size: Optional[Tuple[int, int]] = None

    try:
        while True:
            if mpv is not None:
                mpv.show(pattern)
            elif use_opencv_proj:
                surface.show(pattern)

            frame = und.apply(cam.read())
            image_size = (frame.shape[1], frame.shape[0])
            found, corners = find_projected_chessboard(frame, board_size)
            display = frame.copy()
            if found and corners is not None:
                cv2.drawChessboardCorners(
                    display,
                    (cols, rows),
                    corners.reshape(-1, 1, 2).astype(np.float32),
                    True,
                )
            cv2.putText(
                display,
                f"samples={len(cam_samples)}  SPACE=capture  c=save  q=quit",
                (12, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )
            if want_preview:
                cv2.imshow(preview_win, display)
                key = cv2.waitKey(1) & 0xFF
            else:
                # Headless-ish: still need a key path — poll waitKey on a dummy if possible.
                key = cv2.waitKey(1) & 0xFF
                if found and corners is not None and len(cam_samples) == 0:
                    # Auto-take first good lock so SSH/no-GUI can still finish with 'c'
                    # if someone attaches keys later; print status each ~30 frames.
                    pass

            if key == ord("q"):
                raise SystemExit("Projector homography calibration cancelled")
            if key == ord(" ") and found and corners is not None:
                cam_samples.append(corners.copy())
                proj_samples.append(proj_corners.copy())
                print(f"  captured sample {len(cam_samples)}")
            if key == ord("c"):
                if len(cam_samples) < 1:
                    print("Need at least 1 sample (SPACE when corners are green)")
                    continue
                break
    finally:
        cam.close()
        if mpv is not None:
            mpv.close()
        if use_opencv_proj:
            surface.close()
        if want_preview:
            try:
                cv2.destroyWindow(preview_win)
            except Exception:
                pass

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
