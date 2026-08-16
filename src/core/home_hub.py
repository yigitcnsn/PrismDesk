"""Publish annotated JPEG + telemetry to home-hub PrismDesk debug module.

Supports multi-layer debug feeds:
  raw | mat | object | final

Desk POSTs each enabled layer as JPEG, then POSTs state JSON.
Backward compatible: publish(frame, state) still posts as layer \"final\"
and also to legacy /api/prismdesk/frame.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import cv2
import numpy as np
import yaml

MAX_FRAME_BYTES = 800 * 1024

# Layer ids used by home-hub multi-panel debug UI.
LAYER_IDS: tuple[str, ...] = ("raw", "mat", "object", "final")
HUB_MODES: tuple[str, ...] = ("desk", "idle")
HUB_COMMAND_STOP = "stop"
HUB_COMMAND_TTL_S = 15.0


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
    # Which layer JPEGs to publish (subset of LAYER_IDS)
    layers: List[str] = field(default_factory=lambda: list(LAYER_IDS))


@dataclass
class OverlayFlags:
    mat: bool = True
    object: bool = True

    def as_list(self) -> list[str]:
        out = []
        if self.mat:
            out.append("mat")
        if self.object:
            out.append("object")
        return out

    def as_dict(self) -> dict[str, bool]:
        return {
            "mat": bool(self.mat),
            "object": bool(self.object),
        }


@dataclass
class HubControl:
    """Phone / home-hub control snapshot polled by the desk."""

    projector: OverlayFlags = field(default_factory=OverlayFlags)
    browser: OverlayFlags = field(default_factory=OverlayFlags)
    mode: str = "desk"
    command: Optional[str] = None


def overlay_config_payload(flags: OverlayFlags) -> dict[str, dict[str, bool]]:
    """Hub PUT body: projector + browser mirrored (desk Visual toggles)."""
    block = flags.as_dict()
    return {
        "projector": dict(block),
        "browser": dict(block),
        "overlays": dict(block),
    }


def _normalize_layers(raw: Any) -> List[str]:
    if raw is None:
        return list(LAYER_IDS)
    if isinstance(raw, str):
        items = [raw]
    elif isinstance(raw, Sequence) and not isinstance(raw, (bytes, bytearray)):
        items = list(raw)
    else:
        return list(LAYER_IDS)
    out: List[str] = []
    for item in items:
        name = str(item).strip().lower()
        if name in LAYER_IDS and name not in out:
            out.append(name)
    return out or list(LAYER_IDS)


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
        layers=_normalize_layers(raw.get("layers")),
    )


def _flags_from_mapping(raw: Any) -> OverlayFlags:
    flags = OverlayFlags()
    if not isinstance(raw, Mapping):
        return flags
    if isinstance(raw.get("mat"), bool):
        flags.mat = bool(raw["mat"])
    if isinstance(raw.get("object"), bool):
        flags.object = bool(raw["object"])
    return flags


def split_overlays_from_config(
    payload: Mapping[str, Any] | None,
) -> tuple[OverlayFlags, OverlayFlags]:
    """Parse projector vs browser overlay toggles from hub config.

    New shape:
      { "projector": {mat,object}, "browser": {mat,object} }
    Legacy:
      { "overlays": {mat,object} }  → applied to BOTH surfaces
    """
    if not payload:
        return OverlayFlags(), OverlayFlags()

    legacy = payload.get("overlays") if isinstance(payload.get("overlays"), Mapping) else None
    proj_raw = payload.get("projector") if isinstance(payload.get("projector"), Mapping) else None
    browser_raw = payload.get("browser") if isinstance(payload.get("browser"), Mapping) else None

    if proj_raw is None and browser_raw is None and legacy is not None:
        shared = _flags_from_mapping(legacy)
        return shared, OverlayFlags(mat=shared.mat, object=shared.object)

    projector = _flags_from_mapping(proj_raw if proj_raw is not None else legacy)
    browser = _flags_from_mapping(browser_raw if browser_raw is not None else legacy)
    return projector, browser


def _parse_command_at(raw: Any) -> Optional[float]:
    if not isinstance(raw, str) or not raw.strip():
        return None
    text = raw.strip().replace("Z", "+00:00")
    try:
        stamp = datetime.fromisoformat(text)
    except ValueError:
        return None
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    return stamp.timestamp()


def parse_hub_control(payload: Mapping[str, Any] | None) -> HubControl:
    """Parse GET /api/prismdesk/config into overlays + phone commands."""
    projector, browser = split_overlays_from_config(payload)
    if not payload:
        return HubControl(projector=projector, browser=browser)
    mode = str(payload.get("mode") or "desk").strip().lower()
    if mode not in HUB_MODES:
        mode = "desk"
    command = None
    raw_cmd = payload.get("command")
    if raw_cmd == HUB_COMMAND_STOP:
        issued = _parse_command_at(payload.get("command_at") or payload.get("commandAt"))
        if issued is not None and (time.time() - issued) <= HUB_COMMAND_TTL_S:
            command = HUB_COMMAND_STOP
    return HubControl(projector=projector, browser=browser, mode=mode, command=command)


def overlays_from_config_payload(payload: Mapping[str, Any] | None) -> OverlayFlags:
    """Legacy helper: projector overlays (falls back to flat overlays)."""
    projector, _browser = split_overlays_from_config(payload)
    return projector


class HomeHubPublisher:
    """POST layer JPEGs + state; poll overlay / phone-control config."""

    def __init__(self, config: Optional[HomeHubConfig] = None) -> None:
        self.config = config or HomeHubConfig()
        # Separate surfaces: projector HUD vs browser debug composition.
        self.projector_overlays = OverlayFlags()
        self.browser_overlays = OverlayFlags()
        self.mode = "desk"
        self._fail_streak = 0
        self._last_err = ""
        self._lock = threading.RLock()
        self._local_edit_until = 0.0

    @property
    def enabled(self) -> bool:
        return bool(self.config.enabled)

    @property
    def enabled_layers(self) -> List[str]:
        return _normalize_layers(self.config.layers)

    @property
    def overlays(self) -> OverlayFlags:
        """Backward-compatible alias for projector overlays."""
        return self.projector_overlays

    def mark_local_edit(self, hold_s: float = 1.0) -> None:
        """Ignore hub poll overwrites briefly after a desk-side toggle."""
        with self._lock:
            self._local_edit_until = time.monotonic() + max(0.0, float(hold_s))

    def fetch_control(self) -> HubControl:
        """GET hub config: overlays, HUD mode, one-shot phone commands."""
        url = f"{self.config.base_url}/api/prismdesk/config"
        held = HubControl(
            projector=self.projector_overlays,
            browser=self.browser_overlays,
            mode=self.mode,
        )
        try:
            raw = self._request_json("GET", url)
            with self._lock:
                if time.monotonic() < self._local_edit_until:
                    return held
                ctrl = parse_hub_control(raw if isinstance(raw, Mapping) else None)
                self.projector_overlays = ctrl.projector
                self.browser_overlays = ctrl.browser
                self.mode = ctrl.mode
                self._fail_streak = 0
                return ctrl
        except Exception as exc:  # noqa: BLE001
            self._note_fail(exc)
        return held

    def fetch_config(self) -> OverlayFlags:
        return self.fetch_control().projector

    def patch_config(self, payload: Mapping[str, Any]) -> bool:
        """PUT a partial config update (home-hub merges into current)."""
        if not self.enabled:
            return False
        url = f"{self.config.base_url}/api/prismdesk/config"
        try:
            with self._lock:
                self._put_json(url, dict(payload))
                self._fail_streak = 0
            return True
        except Exception as exc:  # noqa: BLE001
            self._note_fail(exc)
            return False

    def clear_command(self) -> bool:
        """ACK a one-shot phone command so it does not replay."""
        return self.patch_config({"command": None})

    def push_config(
        self,
        flags: OverlayFlags,
        *,
        mirror_browser: bool = True,
        hold_s: float = 1.0,
    ) -> bool:
        """PUT overlay config so the webpage matches desk Visual toggles."""
        if not self.enabled:
            with self._lock:
                self.projector_overlays = OverlayFlags(
                    mat=flags.mat, object=flags.object
                )
                if mirror_browser:
                    self.browser_overlays = OverlayFlags(
                        mat=flags.mat, object=flags.object
                    )
            return False
        browser = flags if mirror_browser else self.browser_overlays
        payload = {
            "projector": flags.as_dict(),
            "browser": browser.as_dict() if not mirror_browser else flags.as_dict(),
            "overlays": flags.as_dict(),
        }
        url = f"{self.config.base_url}/api/prismdesk/config"
        try:
            with self._lock:
                self.projector_overlays = OverlayFlags(
                    mat=flags.mat, object=flags.object
                )
                if mirror_browser:
                    self.browser_overlays = OverlayFlags(
                        mat=flags.mat, object=flags.object
                    )
                self._local_edit_until = time.monotonic() + max(0.0, float(hold_s))
                self._put_json(url, payload)
                self._fail_streak = 0
            return True
        except Exception as exc:  # noqa: BLE001
            self._note_fail(exc)
            return False

    def publish(self, frame_bgr: np.ndarray, state: Dict[str, Any]) -> bool:
        """Backward-compatible: publish a single frame as the final layer."""
        return self.publish_layers({"final": frame_bgr}, state)

    def publish_layer_frame(self, layer: str, frame_bgr: np.ndarray) -> bool:
        """POST one layer JPEG only (safe to call from a dedicated layer thread)."""
        if not self.enabled:
            return False
        name = str(layer).strip().lower()
        if name not in set(self.enabled_layers):
            return False
        if frame_bgr is None or not isinstance(frame_bgr, np.ndarray) or frame_bgr.size == 0:
            return False
        jpeg = encode_jpeg_capped(
            frame_bgr,
            quality=self.config.jpeg_quality,
            max_bytes=self.config.max_bytes,
            max_dim=self.config.max_dim,
        )
        if jpeg is None:
            self._note_fail(RuntimeError(f"JPEG encode failed for layer={name}"))
            return False
        try:
            with self._lock:
                try:
                    self._post_bytes(
                        f"{self.config.base_url}/api/prismdesk/frame/{name}",
                        jpeg,
                        content_type="image/jpeg",
                    )
                except RuntimeError as exc:
                    if "404" not in str(exc):
                        raise
                    if name != "final":
                        return False
                if name == "final":
                    self._post_bytes(
                        f"{self.config.base_url}/api/prismdesk/frame",
                        jpeg,
                        content_type="image/jpeg",
                    )
                self._fail_streak = 0
            return True
        except Exception as exc:  # noqa: BLE001
            self._note_fail(exc)
            return False

    def publish_state(self, state: Dict[str, Any]) -> bool:
        """POST telemetry JSON only."""
        if not self.enabled:
            return False
        try:
            with self._lock:
                payload = dict(state)
                payload.setdefault("overlays", self.projector_overlays.as_list())
                payload.setdefault(
                    "projector_overlays", self.projector_overlays.as_list()
                )
                payload.setdefault("browser_overlays", self.browser_overlays.as_list())
                payload.setdefault("layers", list(self.enabled_layers))
                self._post_json(f"{self.config.base_url}/api/prismdesk/state", payload)
                self._fail_streak = 0
            return True
        except Exception as exc:  # noqa: BLE001
            self._note_fail(exc)
            return False

    def publish_layers(
        self,
        layers: Mapping[str, np.ndarray],
        state: Dict[str, Any],
    ) -> bool:
        """Encode and POST each enabled layer JPEG, then POST state."""
        if not self.enabled:
            return False

        enabled = set(self.enabled_layers)
        posted: List[str] = []
        try:
            with self._lock:
                for name, frame in layers.items():
                    layer = str(name).strip().lower()
                    if layer not in LAYER_IDS or layer not in enabled:
                        continue
                    if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
                        continue
                    jpeg = encode_jpeg_capped(
                        frame,
                        quality=self.config.jpeg_quality,
                        max_bytes=self.config.max_bytes,
                        max_dim=self.config.max_dim,
                    )
                    if jpeg is None:
                        self._note_fail(RuntimeError(f"JPEG encode failed for layer={layer}"))
                        continue
                    try:
                        self._post_bytes(
                            f"{self.config.base_url}/api/prismdesk/frame/{layer}",
                            jpeg,
                            content_type="image/jpeg",
                        )
                        posted.append(layer)
                    except RuntimeError as exc:
                        if "404" not in str(exc):
                            raise
                        if layer != "final":
                            continue
                    if layer == "final":
                        self._post_bytes(
                            f"{self.config.base_url}/api/prismdesk/frame",
                            jpeg,
                            content_type="image/jpeg",
                        )
                        if "final" not in posted:
                            posted.append("final")

                payload = dict(state)
                payload.setdefault("overlays", self.projector_overlays.as_list())
                payload.setdefault(
                    "projector_overlays", self.projector_overlays.as_list()
                )
                payload.setdefault("browser_overlays", self.browser_overlays.as_list())
                payload["layers"] = posted
                self._post_json(f"{self.config.base_url}/api/prismdesk/state", payload)
                self._fail_streak = 0
            return bool(posted)
        except Exception as exc:  # noqa: BLE001
            self._note_fail(exc)
            return False

    def _note_fail(self, exc: BaseException) -> None:
        with self._lock:
            self._fail_streak += 1
            streak = self._fail_streak
            msg = str(exc)
            should_print = msg != self._last_err or streak in (1, 5, 20)
            if should_print:
                self._last_err = msg
        if should_print:
            print(f"home-hub publish failed ({streak}): {msg}")

    def _post_bytes(self, url: str, body: bytes, *, content_type: str) -> None:
        req = urllib.request.Request(
            url,
            data=body,
            method="POST",
            headers={"Content-Type": content_type, "Content-Length": str(len(body))},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.config.timeout_sec) as resp:
                if resp.status >= 400:
                    raise RuntimeError(f"HTTP {resp.status} POST {url}")
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"HTTP {exc.code} POST {url}") from exc

    def _post_json(self, url: str, payload: Dict[str, Any]) -> None:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            method="POST",
            headers={"Content-Type": "application/json", "Content-Length": str(len(data))},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.config.timeout_sec) as resp:
                if resp.status >= 400:
                    raise RuntimeError(f"HTTP {resp.status} POST {url}")
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"HTTP {exc.code} POST {url}") from exc

    def _put_json(self, url: str, payload: Dict[str, Any]) -> None:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            method="PUT",
            headers={"Content-Type": "application/json", "Content-Length": str(len(data))},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.config.timeout_sec) as resp:
                if resp.status >= 400:
                    raise RuntimeError(f"HTTP {resp.status} PUT {url}")
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"HTTP {exc.code} PUT {url}") from exc

    def _request_json(self, method: str, url: str) -> Any:
        req = urllib.request.Request(url, method=method)
        try:
            with urllib.request.urlopen(req, timeout=self.config.timeout_sec) as resp:
                body = resp.read().decode("utf-8")
                return json.loads(body) if body else {}
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"HTTP {exc.code} {method} {url}") from exc


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
