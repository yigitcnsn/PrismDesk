"""Unit tests for camera ↔ projector homography helpers."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.vision.homography import (
    CamProjectorHomography,
    estimate_homography,
    make_chessboard_pattern,
    map_cam_to_hud,
    stretch_points,
)
from src.vision.projector import ProjectorConfig, load_projector_config, save_projector_config


class TestHomography(unittest.TestCase):
    def test_make_chessboard_pattern_shape_and_corners(self):
        pattern, proj = make_chessboard_pattern(640, 360, (5, 4))
        self.assertEqual(pattern.shape, (360, 640, 3))
        self.assertEqual(proj.shape, (20, 2))
        self.assertEqual(proj.dtype, np.float64)
        self.assertGreater(proj[:, 0].min(), 0)
        self.assertGreater(proj[:, 1].min(), 0)
        self.assertLess(proj[:, 0].max(), 640)
        self.assertLess(proj[:, 1].max(), 360)

    def test_estimate_homography_recovers_synthetic_warp(self):
        _pattern, proj = make_chessboard_pattern(640, 360, (5, 4))
        h_true = np.array(
            [[1.1, 0.05, 40.0], [0.02, 0.95, 30.0], [0.0001, 0.0002, 1.0]],
            dtype=np.float64,
        )
        cam = cv2.perspectiveTransform(
            proj.reshape(-1, 1, 2), np.linalg.inv(h_true)
        ).reshape(-1, 2)

        h_est, err, inliers = estimate_homography(cam, proj)
        self.assertLess(err, 0.5)
        self.assertGreaterEqual(inliers, 16)

        hp = CamProjectorHomography(h_est, (640, 360), (640, 360), err)
        mapped = hp.map_points(
            [(float(cam[0, 0]), float(cam[0, 1]))],
            src_size=(640, 360),
            hud_size=(640, 360),
        )
        self.assertLess(abs(mapped[0][0] - proj[0, 0]), 1)
        self.assertLess(abs(mapped[0][1] - proj[0, 1]), 1)

    def test_composed_for_scales_to_smaller_hud(self):
        _pattern, proj = make_chessboard_pattern(640, 360, (5, 4))
        h_true = np.array(
            [[1.1, 0.05, 40.0], [0.02, 0.95, 30.0], [0.0001, 0.0002, 1.0]],
            dtype=np.float64,
        )
        cam = cv2.perspectiveTransform(
            proj.reshape(-1, 1, 2), np.linalg.inv(h_true)
        ).reshape(-1, 2)
        h_est, err, _inliers = estimate_homography(cam, proj)
        hp = CamProjectorHomography(h_est, (640, 360), (640, 360), err)

        mapped = hp.map_points(
            [(float(cam[0, 0]), float(cam[0, 1]))],
            src_size=(640, 360),
            hud_size=(320, 180),
        )
        self.assertLess(abs(mapped[0][0] - proj[0, 0] / 2), 1)
        self.assertLess(abs(mapped[0][1] - proj[0, 1] / 2), 1)

    def test_stretch_fallback_when_no_homography(self):
        pts = map_cam_to_hud(
            [(100.0, 50.0)],
            src_size=(200, 100),
            hud_size=(400, 200),
            homography=None,
        )
        self.assertEqual(pts, [(200, 100)])
        self.assertEqual(
            stretch_points(
                [(100.0, 50.0)], src_size=(200, 100), hud_size=(400, 200)
            ),
            [(200, 100)],
        )

    def test_projector_config_homography_yaml_roundtrip(self):
        _pattern, proj = make_chessboard_pattern(640, 360, (5, 4))
        h_true = np.eye(3, dtype=np.float64)
        h_true[0, 2] = 10.0
        cam = cv2.perspectiveTransform(
            proj.reshape(-1, 1, 2), np.linalg.inv(h_true)
        ).reshape(-1, 2)
        h_est, err, _ = estimate_homography(cam, proj)
        hp = CamProjectorHomography(h_est, (640, 360), (640, 360), err)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "projector.yaml"
            cfg = ProjectorConfig(width=640, height=360, homography=hp)
            save_projector_config(path, cfg)
            loaded = load_projector_config(path)

        self.assertIsNotNone(loaded.homography)
        assert loaded.homography is not None
        self.assertEqual(loaded.homography.cam_size, (640, 360))
        self.assertEqual(loaded.homography.proj_size, (640, 360))
        self.assertTrue(np.allclose(loaded.homography.matrix, h_est))
        self.assertIsNotNone(loaded.homography.reprojection_error_px)


if __name__ == "__main__":
    unittest.main()
