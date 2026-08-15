"""Unit tests for home-hub overlay payload helpers."""

from __future__ import annotations

import unittest

from src.core.home_hub import OverlayFlags, overlay_config_payload


class TestOverlayPayload(unittest.TestCase):
    def test_overlay_config_payload_mirrors(self) -> None:
        flags = OverlayFlags(mat=False, object=True)
        payload = overlay_config_payload(flags)
        self.assertEqual(payload["projector"], payload["browser"])
        self.assertEqual(payload["overlays"], payload["projector"])
        self.assertFalse(payload["browser"]["mat"])
        self.assertTrue(payload["browser"]["object"])
        self.assertNotIn("hands", payload["projector"])


if __name__ == "__main__":
    unittest.main()
