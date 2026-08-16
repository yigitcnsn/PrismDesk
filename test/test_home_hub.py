"""Unit tests for home-hub overlay payload and phone-control parse."""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from src.core.home_hub import (
    HUB_COMMAND_STOP,
    OverlayFlags,
    overlay_config_payload,
    parse_hub_control,
)


def _iso(delta_s: float = 0.0) -> str:
    stamp = datetime.now(timezone.utc) + timedelta(seconds=delta_s)
    return stamp.isoformat().replace("+00:00", "Z")


class TestOverlayPayload(unittest.TestCase):
    def test_overlay_config_payload_mirrors(self) -> None:
        flags = OverlayFlags(mat=False, object=True)
        payload = overlay_config_payload(flags)
        self.assertEqual(payload["projector"], payload["browser"])
        self.assertEqual(payload["overlays"], payload["projector"])
        self.assertFalse(payload["browser"]["mat"])
        self.assertTrue(payload["browser"]["object"])
        self.assertNotIn("hands", payload["projector"])


class TestHubControlParse(unittest.TestCase):
    def test_defaults(self) -> None:
        ctrl = parse_hub_control(None)
        self.assertEqual(ctrl.mode, "desk")
        self.assertIsNone(ctrl.command)
        self.assertTrue(ctrl.projector.mat)
        self.assertTrue(ctrl.projector.object)

    def test_idle_mode(self) -> None:
        ctrl = parse_hub_control({"mode": "idle", "projector": {"mat": False, "object": True}})
        self.assertEqual(ctrl.mode, "idle")
        self.assertFalse(ctrl.projector.mat)
        self.assertTrue(ctrl.projector.object)

    def test_unknown_mode_falls_back_to_desk(self) -> None:
        ctrl = parse_hub_control({"mode": "dance"})
        self.assertEqual(ctrl.mode, "desk")

    def test_stop_fresh(self) -> None:
        ctrl = parse_hub_control({"command": HUB_COMMAND_STOP, "command_at": _iso()})
        self.assertEqual(ctrl.command, HUB_COMMAND_STOP)

    def test_stop_stale_ignored(self) -> None:
        ctrl = parse_hub_control({"command": HUB_COMMAND_STOP, "command_at": _iso(-60)})
        self.assertIsNone(ctrl.command)

    def test_stop_without_timestamp_ignored(self) -> None:
        ctrl = parse_hub_control({"command": HUB_COMMAND_STOP})
        self.assertIsNone(ctrl.command)


if __name__ == "__main__":
    unittest.main()
