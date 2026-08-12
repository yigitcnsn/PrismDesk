"""Unit tests for threaded camera wrapper."""

from __future__ import annotations

import time
import unittest
from unittest.mock import MagicMock

import numpy as np

from src.vision.camera import ThreadedCamera


class TestThreadedCamera(unittest.TestCase):
    def test_read_returns_latest_frame(self):
        frames = [
            np.full((16, 16, 3), i, dtype=np.uint8) for i in range(1, 6)
        ]
        idx = {"i": 0}

        def fake_read():
            i = min(idx["i"], len(frames) - 1)
            idx["i"] += 1
            time.sleep(0.01)
            return frames[i]

        cam = MagicMock()
        cam.open.return_value = 0
        cam.read.side_effect = fake_read
        cam.negotiated.return_value = (16, 16, 30.0)
        cam.active_index = 0

        threaded = ThreadedCamera(cam)
        self.assertEqual(threaded.open(), 0)
        time.sleep(0.08)
        frame = threaded.read()
        self.assertEqual(frame.shape, (16, 16, 3))
        self.assertGreaterEqual(int(frame[0, 0, 0]), 1)
        threaded.close()
        cam.close.assert_called()


if __name__ == "__main__":
    unittest.main()
