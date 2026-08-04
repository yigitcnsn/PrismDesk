"""Chessboard camera calibration (fisheye or pinhole) using the live USB camera."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .camera import Camera, CameraConfig, save_camera_config
from .undistort import Undistorter


def run_calibration(
    config: CameraConfig,
    output_path: str | Path,
    board_size: Tuple[int, int] = (9, 6),
    square_size: float = 1.0,
    target_samples: int = 20,
    model: Optional[str] = None,
) -> CameraConfig:
    """
    Interactive chessboard calibration.

    Keys: SPACE capture sample when corners found | c compute+save | q cancel
    board_size = inner corner counts (cols, rows).
    """
    model = (model or config.model or "fisheye").lower()
    cols, rows = board_size
    objp = np.zeros((rows * cols, 1, 3), np.float64)
    grid = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp[:, 0, :2] = grid * square_size

    obj_points: List[np.ndarray] = []
    img_points: List[np.ndarray] = []
    image_size: Optional[Tuple[int, int]] = None

    cam = Camera(config)
    cam.open()
    print(
        f"Camera open on index {cam.active_index} "
        f"negotiated={cam.negotiated()} model={model}"
    )
    print("Show chessboard. SPACE=sample  c=calibrate+save  q=quit")

    window = "calibrate-camera"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    try:
        while True:
            frame = cam.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            image_size = (gray.shape[1], gray.shape[0])
            flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
            found, corners = cv2.findChessboardCorners(gray, (cols, rows), flags)
            display = frame.copy()
            if found:
                corners = cv2.cornerSubPix(
                    gray,
                    corners,
                    (11, 11),
                    (-1, -1),
                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
                )
                cv2.drawChessboardCorners(display, (cols, rows), corners, found)
            cv2.putText(
                display,
                f"samples={len(img_points)}/{target_samples}  SPACE=capture  c=save  q=quit",
                (12, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )
            cv2.imshow(window, display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                raise SystemExit("Calibration cancelled")
            if key == ord(" ") and found:
                obj_points.append(objp.copy())
                img_points.append(corners.reshape(-1, 1, 2).astype(np.float64))
                print(f"  captured sample {len(img_points)}")
            if key == ord("c"):
                if len(img_points) < 5:
                    print("Need at least 5 samples")
                    continue
                break
    finally:
        cam.close()
        cv2.destroyWindow(window)

    assert image_size is not None
    if model == "fisheye":
        k = np.zeros((3, 3))
        d = np.zeros((4, 1))
        # fisheye.calibrate wants object points as list of (N,1,3)
        rms, k, d, _rvecs, _tvecs = cv2.fisheye.calibrate(
            obj_points,
            img_points,
            image_size,
            k,
            d,
            flags=cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
            + cv2.fisheye.CALIB_CHECK_COND
            + cv2.fisheye.CALIB_FIX_SKEW,
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6),
        )
        dist = d.reshape(-1)
    else:
        rms, k, d, _r, _t = cv2.calibrateCamera(
            [p.reshape(-1, 3) for p in obj_points],
            [p.reshape(-1, 1, 2) for p in img_points],
            image_size,
            None,
            None,
        )
        dist = d.reshape(-1)

    config.model = model
    config.camera_matrix = np.asarray(k, dtype=np.float64)
    config.dist_coeffs = np.asarray(dist, dtype=np.float64)
    config.image_size = image_size
    save_camera_config(output_path, config)
    print(f"Saved calibration RMS={rms:.4f} → {output_path}")

    # Quick preview of undistort
    cam = Camera(config)
    und = Undistorter(config)
    cam.open()
    try:
        for _ in range(60):
            frame = und.apply(cam.read())
            cv2.putText(
                frame,
                "Undistort preview — q to finish",
                (12, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            cv2.imshow(window, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cam.close()
        cv2.destroyAllWindows()
    return config
