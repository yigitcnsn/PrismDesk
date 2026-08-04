"""Publish annotated JPEG + telemetry to home-hub PrismDesk debug module."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import cv2
import numpy as np
import yaml

MAX_FRAME_BYTES = 800 * 1024


@dataclass
class HomeHubConfig:
    enabled: bool = False
    base_url: str = "http://127.0.0.1:3000"
    publish_every: int = 3
    config_every: int = 30
    jpeg_quality: int = 75
    max_bytes: int = MAX_FRAME_BYTES
    timeout_sec: float = 1.5
    # Max long edge for debug JPEG (keeps under 800 KB on Pi)
    max_dim: int = 960


@dataclass
class OverlayFlags:
    mat: bool = True
    object: bool = True
    hands: bool = True

    def as_list(self) -> list[str]:
        out = []
        if self.mat:
            out.append("mat")
        if self.object:
            out.append("object")
        if self.hands:
            out.append("hands")
        return out


def load_home_hub_config(path: str | Path) -> HomeHubConfig:
    path = Path(path)
    if not path.is_file():
        return HomeHubConfig()
    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return HomeHubConfig(
        enabled=bool(raw.get("enabled", False)),
        base_url=str(raw.get("base_url", "http://127.0.0.1:3000")).rstrip("/"),
        publish_every=max(1, int(raw.get("publish_every", 3))),
        config_every=max(1, int(raw.get("config_every", 30))),
        jpeg_quality=int(raw.get("jpeg_quality", 75)),
        max_bytes=min(MAX_FRAME_BYTES, int(raw.get("max_bytes", MAX_FRAME_BYTES))),
        timeout_sec=float(raw.get("timeout_sec", 1.5)),
        max_dim=int(raw.get("max_dim", 960)),
    )


def overlays_from_config_payload(payload: Mapping[str, Any] | None) -> OverlayFlags:
    flags = OverlayFlags()
    if not payload:
        return flags
    overlays = payload.get("overlays") if isinstance(payload, Mapping) else None
    if not isinstance(overlays, Mapping):
        return flags
    if isinstance(overlays.get("mat"), bool):
        flags.mat = bool(overlays["mat"])
    if isinstance(overlays.get("object"), bool):
        flags.object = bool(overlays["object"])
    if isinstance(overlays.get("hands"), bool):
        flags.hands = bool(overlays["hands"])
    return flags


class HomeHubPublisher:
    """POST latest annotated JPEG + state; poll overlay config."""

    def __init__(self, config: Optional[HomeHubConfig] = None) -> None:
        self.config = config or HomeHubConfig()
        self.overlays = OverlayFlags()
        self._fail_streak = 0
        self._last_err = ""

    @property
    def enabled(self) -> bool:
        return bool(self.config.enabled)

    def fetch_config(self) -> OverlayFlags:
        url = f"{self.config.base_url}/api/prismdesk/config"
        try:
            raw = self._request_json("GET", url)
            self.overlays = overlays_from_config_payload(raw)
            self._fail_streak = 0
        except Exception as exc:  # noqa: BLE001
            self._note_fail(exc)
        return self.overlays

    def publish(self, frame_bgr: np.ndarray, state: Dict[str, Any]) -> bool:
        """Encode frame as JPEG and POST frame + state. Returns True on success."""
        if not self.enabled:
            return False
        jpeg = encode_jpeg_capped(
            frame_bgr,
            quality=self.config.jpeg_quality,
            max_bytes=self.config.max_bytes,
            max_dim=self.config.max_dim,
        )
        if jpeg is None:
            self._note_fail(RuntimeError("JPEG encode failed / over size cap"))
            return False
        try:
            self._post_bytes(
                f"{self.config.base_url}/api/prismdesk/frame",
                jpeg,
                content_type="image/jpeg",
            )
            payload = dict(state)
            payload.setdefault("overlays", self.overlays.as_list())
            self._post_json(f"{self.config.base_url}/api/prismdesk/state", payload)
            self._fail_streak = 0
            return True
        except Exception as exc:  # noqa: BLE001
            self._note_fail(exc)
            return False

    def _note_fail(self, exc: BaseException) -> None:
        self._fail_streak += 1
        msg = str(exc)
        # Avoid spamming the desk loop; print occasionally.
        if msg != self._last_err or self._fail_streak in (1, 5, 20):
            print(f"home-hub publish failed ({self._fail_streak}): {msg}")
            self._last_err = msg

    def _post_bytes(self, url: str, body: bytes, *, content_type: str) -> None:
        req = urllib.request.Request(
            url,
            data=body,
            method="POST",
            headers={"Content-Type": content_type, "Content-Length": str(len(body))},
        )
        with urllib.request.urlopen(req, timeout=self.config.timeout_sec) as resp:
            if resp.status >= 400:
                raise RuntimeError(f"HTTP {resp.status} posting to {url}")

    def _post_json(self, url: str, payload: Dict[str, Any]) -> None:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            method="POST",
            headers={"Content-Type": "application/json", "Content-Length": str(len(data))},
        )
        with urllib.request.urlopen(req, timeout=self.config.timeout_sec) as resp:
            if resp.status >= 400:
                raise RuntimeError(f"HTTP {resp.status} posting JSON to {url}")

    def _request_json(self, method: str, url: str) -> Any:
        req = urllib.request.Request(url, method=method)
        with urllib.request.urlopen(req, timeout=self.config.timeout_sec) as resp:
            body = resp.read().decode("utf-8")
            return json.loads(body) if body else {}


def encode_jpeg_capped(
    frame_bgr: np.ndarray,
    *,
    quality: int = 75,
    max_bytes: int = MAX_FRAME_BYTES,
    max_dim: int = 960,
) -> Optional[bytes]:
    """Encode BGR frame as JPEG under max_bytes (quality + downscale fallback)."""
    img = frame_bgr
    h, w = img.shape[:2]
    longest = max(h, w)
    if max_dim > 0 and longest > max_dim:
        scale = max_dim / float(longest)
        img = cv2.resize(
            img,
            (max(1, int(w * scale)), max(1, int(h * scale))),
            interpolation=cv2.INTER_AREA,
        )

    q = max(30, min(95, int(quality)))
    for _ in range(6):
        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        if not ok:
            return None
        data = buf.tobytes()
        if len(data) <= max_bytes:
            return data
        q = max(30, q - 12)
        # Still too big: shrink further
        h, w = img.shape[:2]
        img = cv2.resize(img, (max(1, w * 3 // 4), max(1, h * 3 // 4)), interpolation=cv2.INTER_AREA)
    return None
