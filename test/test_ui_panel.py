"""Unit tests for pinch UI, widgets, and hub overlay payload."""

from __future__ import annotations

import time
import unittest

import numpy as np

from src.core.home_hub import OverlayFlags, overlay_config_payload
from src.ui.control_panel import ControlPanel
from src.ui.pinch import PinchTracker, _normalized_pinch_gap
from src.ui.widgets import Button, Toggle, point_in_rect
from src.vision.hands import HandResult


def _hand_with_gap(norm_gap: float) -> HandResult:
    """Synthetic hand: wrist at 0, middle MCP at 100 → scale=100; tips spaced by gap*100."""
    scale = 100.0
    wrist = (0.0, 0.0)
    mid = (scale, 0.0)
    thumb = (50.0, 0.0)
    index = (50.0 + norm_gap * scale, 0.0)
    pts = [(0.0, 0.0)] * 21
    pts[0] = wrist
    pts[4] = thumb
    pts[8] = index
    pts[9] = mid
    return HandResult(
        landmarks_px=pts,
        landmarks_norm=[(0.0, 0.0)] * 21,
        index_tip=index,
        thumb_tip=thumb,
        handedness="Right",
    )


class TestPinch(unittest.TestCase):
    def test_normalized_gap(self):
        hand = _hand_with_gap(0.2)
        gap = _normalized_pinch_gap(hand)
        self.assertIsNotNone(gap)
        self.assertAlmostEqual(gap, 0.2, places=2)

    def test_pinch_edges(self):
        tracker = PinchTracker(close_thresh=0.38, open_thresh=0.48, cooldown_s=0.0)
        open_hand = _hand_with_gap(0.7)
        closed = _hand_with_gap(0.2)
        snap = tracker.update([open_hand], src_size=(200, 200), hud_size=(400, 400))
        self.assertFalse(snap.is_pinched)
        snap = tracker.update([closed], src_size=(200, 200), hud_size=(400, 400))
        self.assertTrue(snap.just_closed)
        self.assertTrue(snap.is_pinched)
        snap = tracker.update([open_hand], src_size=(200, 200), hud_size=(400, 400))
        self.assertTrue(snap.just_opened)
        self.assertTrue(tracker.consume_click(True))


class TestWidgets(unittest.TestCase):
    def test_point_in_rect(self):
        self.assertTrue(point_in_rect((10, 10), (0, 0, 20, 20)))
        self.assertFalse(point_in_rect((25, 10), (0, 0, 20, 20)))

    def test_button_hit(self):
        btn = Button("x", "X", (100, 100, 50, 40))
        self.assertTrue(btn.hit((120, 120)))
        self.assertFalse(btn.hit((10, 10)))


class TestControlPanel(unittest.TestCase):
    def test_toggle_flips_flags_and_hub_payload(self):
        seen = []

        def on_change(flags: OverlayFlags) -> None:
            seen.append(overlay_config_payload(flags))

        panel = ControlPanel(on_visual_change=on_change, flash_s=0.0)
        panel._pinch.cooldown_s = 0.0
        self.assertTrue(panel.flags.mat)

        def pinch_at(x: float, y: float, *, closed: bool) -> HandResult:
            gap = 0.15 if closed else 0.7
            hand = _hand_with_gap(gap)
            # Keep thumb near index so gap stays meaningful at the hit point.
            hand.index_tip = (x, y)
            hand.thumb_tip = (x - gap * 100.0, y)
            hand.landmarks_px[8] = hand.index_tip
            hand.landmarks_px[4] = hand.thumb_tip
            hand.landmarks_px[0] = (x - 50.0, y)
            hand.landmarks_px[9] = (x + 50.0, y)
            return hand

        chip = panel._controls.rect
        # Force layout for 400x400 before reading chip rect.
        panel._layout_for((400, 400))
        chip = panel._controls.rect
        cx = float(chip[0] + chip[2] // 2)
        cy = float(chip[1] + chip[3] // 2)

        panel.update([pinch_at(cx, cy, closed=True)], src_size=(400, 400), hud_size=(400, 400))
        panel.update([pinch_at(cx, cy, closed=False)], src_size=(400, 400), hud_size=(400, 400))
        self.assertTrue(panel.open)

        mat = panel._toggles["mat"].rect
        mx = float(mat[0] + mat[2] // 2)
        my = float(mat[1] + mat[3] // 2)
        panel.update([pinch_at(mx, my, closed=True)], src_size=(400, 400), hud_size=(400, 400))
        panel.update([pinch_at(mx, my, closed=False)], src_size=(400, 400), hud_size=(400, 400))
        self.assertFalse(panel.flags.mat)
        self.assertEqual(len(seen), 1)
        payload = seen[0]
        self.assertEqual(payload["projector"], payload["browser"])
        self.assertFalse(payload["projector"]["mat"])
        self.assertTrue(payload["projector"]["object"])
        self.assertTrue(payload["projector"]["hands"])


class TestOverlayPayload(unittest.TestCase):
    def test_overlay_config_payload_mirrors(self):
        flags = OverlayFlags(mat=False, object=True, hands=False)
        payload = overlay_config_payload(flags)
        self.assertEqual(payload["projector"], payload["browser"])
        self.assertEqual(payload["overlays"], payload["projector"])
        self.assertFalse(payload["browser"]["mat"])


if __name__ == "__main__":
    unittest.main()
