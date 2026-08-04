"""HY300 / HDMI projector output helpers for Raspberry Pi (Wayland / labwc).

Uses `wlr-randr` to discover outputs and optionally force 1920x1080@50.
OpenCV fullscreen window is moved onto the projector's desktop geometry.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import yaml


@dataclass
class ProjectorConfig:
    output_name: str = "HDMI-A-1"
    width: int = 1920
    height: int = 1080
    refresh_hz: float = 50.0
    window_name: str = "prismdesk-projector"
    apply_mode: bool = True


@dataclass
class OutputInfo:
    name: str
    make: str
    width: int
    height: int
    refresh_hz: float
    x: int
    y: int
    enabled: bool
    modes: List[Tuple[int, int, float]]  # w, h, hz


def load_projector_config(path: str | Path) -> ProjectorConfig:
    path = Path(path)
    if not path.is_file():
        return ProjectorConfig()
    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return ProjectorConfig(
        output_name=str(raw.get("output_name", "HDMI-A-1")),
        width=int(raw.get("width", 1920)),
        height=int(raw.get("height", 1080)),
        refresh_hz=float(raw.get("refresh_hz", 50)),
        window_name=str(raw.get("window_name", "prismdesk-projector")),
        apply_mode=bool(raw.get("apply_mode", True)),
    )


def save_projector_config(path: str | Path, cfg: ProjectorConfig) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "output_name": cfg.output_name,
        "width": cfg.width,
        "height": cfg.height,
        "refresh_hz": cfg.refresh_hz,
        "window_name": cfg.window_name,
        "apply_mode": cfg.apply_mode,
    }
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, default_flow_style=False, sort_keys=False)


def wlr_randr_available() -> bool:
    return shutil.which("wlr-randr") is not None


def list_outputs() -> List[OutputInfo]:
    """Parse `wlr-randr` text output into OutputInfo list."""
    if not wlr_randr_available():
        return []
    proc = subprocess.run(
        ["wlr-randr"],
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return []
    return _parse_wlr_randr(proc.stdout)


def _parse_wlr_randr(text: str) -> List[OutputInfo]:
    outputs: List[OutputInfo] = []
    current: Optional[dict] = None
    mode_re = re.compile(
        r"^\s+(\d+)x(\d+)\s+px,\s+([\d.]+)\s+Hz(.*)$"
    )
    # Alternate formats seen on some builds:
    alt_mode_re = re.compile(r"^\s+(\d+)x(\d+)\s+@\s+([\d.]+)\s+Hz(.*)$")

    def flush() -> None:
        nonlocal current
        if current is None:
            return
        modes = current.get("modes") or []
        w = int(current.get("width") or (modes[0][0] if modes else 0))
        h = int(current.get("height") or (modes[0][1] if modes else 0))
        hz = float(current.get("refresh_hz") or (modes[0][2] if modes else 0.0))
        outputs.append(
            OutputInfo(
                name=str(current["name"]),
                make=str(current.get("make") or ""),
                width=w,
                height=h,
                refresh_hz=hz,
                x=int(current.get("x") or 0),
                y=int(current.get("y") or 0),
                enabled=bool(current.get("enabled", True)),
                modes=list(modes),
            )
        )
        current = None

    for line in text.splitlines():
        if not line.startswith(" ") and line.strip():
            # New output header: "HDMI-A-1 \"...\""
            flush()
            name = line.split()[0].strip()
            current = {"name": name, "modes": [], "enabled": True}
            continue
        if current is None:
            continue
        s = line.strip()
        if s.startswith("Make:"):
            current["make"] = s.split(":", 1)[1].strip()
        elif s.startswith("Position:"):
            # Position: 1920,0
            m = re.search(r"(-?\d+)\s*,\s*(-?\d+)", s)
            if m:
                current["x"], current["y"] = int(m.group(1)), int(m.group(2))
        elif s.startswith("Enabled:"):
            current["enabled"] = "yes" in s.lower() or "true" in s.lower()
        else:
            m = mode_re.match(line) or alt_mode_re.match(line)
            if m:
                mw, mh, mhz = int(m.group(1)), int(m.group(2)), float(m.group(3))
                rest = m.group(4)
                current["modes"].append((mw, mh, mhz))
                if "current" in rest:
                    current["width"], current["height"], current["refresh_hz"] = mw, mh, mhz
    flush()
    return outputs


def find_output(name: str, outputs: Optional[List[OutputInfo]] = None) -> Optional[OutputInfo]:
    outputs = outputs if outputs is not None else list_outputs()
    for out in outputs:
        if out.name == name or out.name.endswith(name):
            return out
    # Fuzzy: HDMI-A-1 vs card1-HDMI-A-1
    needle = name.replace("card1-", "").replace("card0-", "")
    for out in outputs:
        if needle in out.name or out.name in needle:
            return out
    return None


def pick_mode(
    output: OutputInfo,
    width: int,
    height: int,
    prefer_hz: float,
) -> Optional[Tuple[int, int, float]]:
    """Prefer exact WxH near prefer_hz; fall back to any WxH, then preferred current."""
    candidates = [m for m in output.modes if m[0] == width and m[1] == height]
    if not candidates and output.width == width and output.height == height:
        return (width, height, output.refresh_hz or prefer_hz)
    if not candidates:
        return None

    def score(m: Tuple[int, int, float]) -> float:
        return abs(m[2] - prefer_hz)

    candidates.sort(key=score)
    return candidates[0]


def apply_output_mode(
    output_name: str,
    width: int,
    height: int,
    refresh_hz: float,
) -> bool:
    """Run wlr-randr --output NAME --mode WxH@Hz. Returns True on success."""
    if not wlr_randr_available():
        return False
    # Try exact refresh first, then without Hz
    mode_str = f"{width}x{height}@{refresh_hz:.3f}Hz"
    # Compact forms some builds accept
    trials = [
        ["wlr-randr", "--output", output_name, "--mode", f"{width}x{height}@{refresh_hz}Hz"],
        ["wlr-randr", "--output", output_name, "--mode", f"{width}x{height}"],
        ["wlr-randr", "--output", output_name, "--on"],
    ]
    for cmd in trials:
        proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
        if proc.returncode == 0:
            time.sleep(0.2)
            return True
    return False


class ProjectorSurface:
    """Fullscreen OpenCV window parked on the projector output."""

    def __init__(self, config: Optional[ProjectorConfig] = None) -> None:
        self.config = config or ProjectorConfig()
        self.output: Optional[OutputInfo] = None
        self._open = False

    def prepare(self) -> OutputInfo:
        outputs = list_outputs()
        out = find_output(self.config.output_name, outputs)
        if out is None:
            names = [o.name for o in outputs] or ["<none — is wlr-randr installed / Wayland active?>"]
            raise RuntimeError(
                f"Projector output '{self.config.output_name}' not found. "
                f"Available: {names}"
            )
        if self.config.apply_mode:
            mode = pick_mode(out, self.config.width, self.config.height, self.config.refresh_hz)
            if mode is not None:
                apply_output_mode(out.name, mode[0], mode[1], mode[2])
                # Refresh geometry after mode change
                out = find_output(self.config.output_name) or out
        self.output = out
        return out

    def open(self) -> None:
        out = self.prepare()
        name = self.config.window_name
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        # Place on projector desktop coords then fullscreen
        cv2.moveWindow(name, int(out.x), int(out.y))
        cv2.resizeWindow(name, int(self.config.width), int(self.config.height))
        cv2.setWindowProperty(name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        self._open = True

    def show(self, frame_bgr: np.ndarray) -> None:
        if not self._open:
            self.open()
        h, w = frame_bgr.shape[:2]
        tw, th = self.config.width, self.config.height
        if (w, h) != (tw, th):
            frame_bgr = cv2.resize(frame_bgr, (tw, th), interpolation=cv2.INTER_LINEAR)
        cv2.imshow(self.config.window_name, frame_bgr)

    def close(self) -> None:
        if self._open:
            try:
                cv2.destroyWindow(self.config.window_name)
            except Exception:
                pass
        self._open = False

    def __enter__(self) -> "ProjectorSurface":
        self.open()
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


def make_alignment_pattern(width: int, height: int) -> np.ndarray:
    """High-contrast desk alignment / smoke-test pattern (cyan/magenta on dark)."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:] = (20, 20, 20)
    # Outer border
    cv2.rectangle(img, (4, 4), (width - 5, height - 5), (255, 255, 0), 4)
    # Crosshair
    cx, cy = width // 2, height // 2
    cv2.line(img, (cx, 0), (cx, height), (255, 0, 255), 2)
    cv2.line(img, (0, cy), (width, cy), (255, 0, 255), 2)
    # Corner markers
    arm = min(width, height) // 12
    for x, y in ((0, 0), (width - 1, 0), (0, height - 1), (width - 1, height - 1)):
        x0 = 0 if x == 0 else width - arm
        y0 = 0 if y == 0 else height - arm
        cv2.rectangle(img, (x0, y0), (x0 + arm - 1, y0 + arm - 1), (255, 255, 0), 3)
    cv2.putText(
        img,
        "PrismDesk projector — q quit",
        (40, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.4,
        (255, 255, 0),
        3,
    )
    cv2.putText(
        img,
        f"{width}x{height}",
        (40, 140),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 0, 255),
        2,
    )
    return img
