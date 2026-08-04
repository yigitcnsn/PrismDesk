"""HY300 / HDMI projector output helpers for Raspberry Pi.

Discovery order:
  1) wlr-randr (Wayland / labwc)
  2) xrandr (X11)
  3) /sys/class/drm (connected connectors, no geometry)
  4) fullscreen fallback at (0,0) using config width/height

Mode apply prefers wlr-randr, then xrandr.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, field
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
    # If discovery tools missing, still open fullscreen (HY300 as primary HDMI)
    allow_fullscreen_fallback: bool = True


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
    modes: List[Tuple[int, int, float]] = field(default_factory=list)
    source: str = ""  # wlr-randr | xrandr | drm | fallback


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
        allow_fullscreen_fallback=bool(raw.get("allow_fullscreen_fallback", True)),
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
        "allow_fullscreen_fallback": cfg.allow_fullscreen_fallback,
    }
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, default_flow_style=False, sort_keys=False)


def wlr_randr_available() -> bool:
    return shutil.which("wlr-randr") is not None


def xrandr_available() -> bool:
    return shutil.which("xrandr") is not None


def discovery_backend() -> str:
    """What tool binaries exist (not necessarily usable without a session)."""
    if wlr_randr_available():
        return "wlr-randr"
    if xrandr_available():
        return "xrandr"
    if Path("/sys/class/drm").is_dir():
        return "drm"
    return "none"


def active_discovery_source() -> str:
    """What list_outputs() actually used."""
    outs = list_outputs()
    if not outs:
        return discovery_backend()
    return outs[0].source or discovery_backend()


def list_outputs() -> List[OutputInfo]:
    """List displays via wlr-randr → xrandr → DRM sysfs."""
    if wlr_randr_available():
        proc = subprocess.run(["wlr-randr"], check=False, capture_output=True, text=True)
        if proc.returncode == 0 and proc.stdout.strip():
            outs = _parse_wlr_randr(proc.stdout)
            for o in outs:
                o.source = "wlr-randr"
            if outs:
                return outs
    if xrandr_available():
        proc = subprocess.run(["xrandr", "--query"], check=False, capture_output=True, text=True)
        if proc.returncode == 0 and proc.stdout.strip():
            outs = _parse_xrandr(proc.stdout)
            for o in outs:
                o.source = "xrandr"
            if outs:
                return outs
    outs = _list_drm_outputs()
    for o in outs:
        o.source = "drm"
    return outs


def _parse_wlr_randr(text: str) -> List[OutputInfo]:
    outputs: List[OutputInfo] = []
    current: Optional[dict] = None
    mode_re = re.compile(r"^\s+(\d+)x(\d+)\s+px,\s+([\d.]+)\s+Hz(.*)$")
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


def _parse_xrandr(text: str) -> List[OutputInfo]:
    """Parse `xrandr --query` connected outputs."""
    outputs: List[OutputInfo] = []
    # HDMI-1 connected primary 1920x1080+0+0 ...
    conn_re = re.compile(
        r"^(\S+)\s+connected(?:\s+primary)?(?:\s+(\d+)x(\d+)\+(\d+)\+(\d+))?(.*)$"
    )
    mode_re = re.compile(r"^\s+(\d+)x(\d+)\s+([\d.]+)([*+ ])?")
    current: Optional[dict] = None

    def flush() -> None:
        nonlocal current
        if current is None:
            return
        modes = current.get("modes") or []
        outputs.append(
            OutputInfo(
                name=str(current["name"]),
                make="",
                width=int(current.get("width") or 0),
                height=int(current.get("height") or 0),
                refresh_hz=float(current.get("refresh_hz") or 0.0),
                x=int(current.get("x") or 0),
                y=int(current.get("y") or 0),
                enabled=True,
                modes=list(modes),
            )
        )
        current = None

    for line in text.splitlines():
        m = conn_re.match(line)
        if m:
            flush()
            current = {
                "name": m.group(1),
                "modes": [],
                "width": int(m.group(2) or 0),
                "height": int(m.group(3) or 0),
                "x": int(m.group(4) or 0),
                "y": int(m.group(5) or 0),
                "refresh_hz": 0.0,
            }
            continue
        if line.startswith(" ") and current is not None:
            mm = mode_re.match(line)
            if mm:
                w, h, hz = int(mm.group(1)), int(mm.group(2)), float(mm.group(3))
                current["modes"].append((w, h, hz))
                flags = mm.group(4) or ""
                if "*" in flags or (current["refresh_hz"] == 0.0 and current["width"] == w):
                    current["refresh_hz"] = hz
                    if current["width"] == 0:
                        current["width"], current["height"] = w, h
    flush()
    return outputs


def _list_drm_outputs() -> List[OutputInfo]:
    """Read connected connectors from /sys/class/drm (no desktop geometry)."""
    root = Path("/sys/class/drm")
    if not root.is_dir():
        return []
    outputs: List[OutputInfo] = []
    for entry in sorted(root.iterdir()):
        # card1-HDMI-A-1
        name = entry.name
        if "-" not in name or name.startswith("card") and name.count("-") < 1:
            continue
        if not name.startswith("card"):
            continue
        status = entry / "status"
        if not status.is_file():
            continue
        if status.read_text(encoding="utf-8").strip() != "connected":
            continue
        # Strip cardN- prefix → HDMI-A-1
        short = name.split("-", 1)[1] if name.startswith("card") else name
        modes: List[Tuple[int, int, float]] = []
        modes_file = entry / "modes"
        if modes_file.is_file():
            for line in modes_file.read_text(encoding="utf-8").splitlines():
                m = re.match(r"^(\d+)x(\d+)", line.strip())
                if m:
                    modes.append((int(m.group(1)), int(m.group(2)), 0.0))
        w = h = 0
        if modes:
            w, h = modes[0][0], modes[0][1]
        outputs.append(
            OutputInfo(
                name=short,
                make=name,
                width=w or 1920,
                height=h or 1080,
                refresh_hz=0.0,
                x=0,
                y=0,
                enabled=True,
                modes=modes,
            )
        )
    return outputs


def find_output(name: str, outputs: Optional[List[OutputInfo]] = None) -> Optional[OutputInfo]:
    outputs = outputs if outputs is not None else list_outputs()
    for out in outputs:
        if out.name == name or out.name.endswith(name):
            return out
    needle = name.replace("card1-", "").replace("card0-", "")
    for out in outputs:
        if needle in out.name or out.name in needle or needle in out.make:
            return out
    # HDMI fuzzy: HDMI-A-1 vs HDMI-1
    compact = needle.replace("-", "").lower()
    for out in outputs:
        if out.name.replace("-", "").lower() == compact:
            return out
        if "hdmi" in out.name.lower() and "hdmi" in needle.lower():
            return out
    return None


def pick_mode(
    output: OutputInfo,
    width: int,
    height: int,
    prefer_hz: float,
) -> Optional[Tuple[int, int, float]]:
    candidates = [m for m in output.modes if m[0] == width and m[1] == height]
    if not candidates and output.width == width and output.height == height:
        return (width, height, output.refresh_hz or prefer_hz)
    if not candidates:
        return None

    def score(m: Tuple[int, int, float]) -> float:
        if m[2] <= 0:
            return abs(prefer_hz) + 100
        return abs(m[2] - prefer_hz)

    candidates.sort(key=score)
    return candidates[0]


def apply_output_mode(
    output_name: str,
    width: int,
    height: int,
    refresh_hz: float,
) -> bool:
    if wlr_randr_available():
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
    if xrandr_available():
        # X11 names often HDMI-1 / HDMI-2
        names = [output_name]
        if "HDMI-A-1" in output_name:
            names.append("HDMI-1")
        if "HDMI-A-2" in output_name:
            names.append("HDMI-2")
        for name in names:
            trials = [
                ["xrandr", "--output", name, "--mode", f"{width}x{height}", "--rate", str(refresh_hz)],
                ["xrandr", "--output", name, "--mode", f"{width}x{height}"],
            ]
            for cmd in trials:
                proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
                if proc.returncode == 0:
                    time.sleep(0.2)
                    return True
    return False


class ProjectorSurface:
    """Fullscreen OpenCV window parked on the projector output when possible."""

    def __init__(self, config: Optional[ProjectorConfig] = None) -> None:
        self.config = config or ProjectorConfig()
        self.output: Optional[OutputInfo] = None
        self._open = False

    def prepare(self) -> OutputInfo:
        outputs = list_outputs()
        out = find_output(self.config.output_name, outputs)

        if out is None and outputs:
            # Prefer any connected HDMI
            hdmi = [o for o in outputs if "hdmi" in o.name.lower()]
            out = hdmi[0] if hdmi else outputs[0]
            print(f"Note: using discovered output {out.name!r} (requested {self.config.output_name!r})")

        if out is None:
            if not self.config.allow_fullscreen_fallback:
                raise RuntimeError(
                    f"Projector output '{self.config.output_name}' not found. "
                    f"backend={discovery_backend()} DISPLAY={os.environ.get('DISPLAY')!r} "
                    f"WAYLAND_DISPLAY={os.environ.get('WAYLAND_DISPLAY')!r}. "
                    "Install wlr-randr (Wayland) or xrandr (X11), or enable allow_fullscreen_fallback."
                )
            print(
                f"Warning: no output discovery ({discovery_backend()}). "
                "Falling back to fullscreen at (0,0) — plug HY300 as the active display."
            )
            out = OutputInfo(
                name=self.config.output_name,
                make="fallback",
                width=self.config.width,
                height=self.config.height,
                refresh_hz=self.config.refresh_hz,
                x=0,
                y=0,
                enabled=True,
                modes=[(self.config.width, self.config.height, self.config.refresh_hz)],
                source="fallback",
            )

        if self.config.apply_mode and out.source in ("wlr-randr", "xrandr"):
            mode = pick_mode(out, self.config.width, self.config.height, self.config.refresh_hz)
            if mode is not None:
                apply_output_mode(out.name, mode[0], mode[1], mode[2])
                refreshed = find_output(out.name)
                if refreshed is not None:
                    out = refreshed

        # Prefer configured canvas size for drawing even if EDID reports differently
        if out.width <= 0:
            out.width = self.config.width
        if out.height <= 0:
            out.height = self.config.height

        self.output = out
        return out

    def open(self) -> None:
        out = self.output or self.prepare()
        name = self.config.window_name
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
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
    cv2.rectangle(img, (4, 4), (width - 5, height - 5), (255, 255, 0), 4)
    cx, cy = width // 2, height // 2
    cv2.line(img, (cx, 0), (cx, height), (255, 0, 255), 2)
    cv2.line(img, (0, cy), (width, cy), (255, 0, 255), 2)
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


def ensure_gui_env() -> dict:
    """
    Best-effort: set XDG_RUNTIME_DIR / WAYLAND_DISPLAY / DISPLAY for local Pi session.

    Returns a small status dict for logging. Does not guarantee OpenCV can open a window
    (pip opencv-python ships Qt/xcb only — often broken on pure Wayland).
    """
    status = {
        "uid": os.getuid(),
        "DISPLAY": os.environ.get("DISPLAY"),
        "WAYLAND_DISPLAY": os.environ.get("WAYLAND_DISPLAY"),
        "XDG_RUNTIME_DIR": os.environ.get("XDG_RUNTIME_DIR"),
        "fixed": [],
    }
    uid = status["uid"]
    runtime = Path(f"/run/user/{uid}")
    if not os.environ.get("XDG_RUNTIME_DIR") and runtime.is_dir():
        os.environ["XDG_RUNTIME_DIR"] = str(runtime)
        status["fixed"].append(f"XDG_RUNTIME_DIR={runtime}")
        status["XDG_RUNTIME_DIR"] = str(runtime)

    runtime_dir = Path(os.environ.get("XDG_RUNTIME_DIR", f"/run/user/{uid}"))
    if not os.environ.get("WAYLAND_DISPLAY"):
        for candidate in ("wayland-0", "wayland-1"):
            if (runtime_dir / candidate).exists():
                os.environ["WAYLAND_DISPLAY"] = candidate
                status["fixed"].append(f"WAYLAND_DISPLAY={candidate}")
                status["WAYLAND_DISPLAY"] = candidate
                break

    if not os.environ.get("DISPLAY"):
        # Xwayland / X11 local seat
        if Path("/tmp/.X11-unix/X0").exists():
            os.environ["DISPLAY"] = ":0"
            status["fixed"].append("DISPLAY=:0")
            status["DISPLAY"] = ":0"

    return status


def gui_session_present() -> bool:
    ensure_gui_env()
    if os.environ.get("WAYLAND_DISPLAY") or os.environ.get("DISPLAY"):
        return True
    return False


def opencv_gui_hint() -> str:
    return (
        "OpenCV window failed (no usable display / Qt xcb).\n"
        "Fixes on Pi 5:\n"
        "  1) Run from the desktop session (not plain SSH), or:\n"
        "       export XDG_RUNTIME_DIR=/run/user/$(id -u)\n"
        "       export WAYLAND_DISPLAY=wayland-0   # or DISPLAY=:0 if X11/Xwayland\n"
        "  2) Prefer system OpenCV (GTK) over pip Qt build:\n"
        "       pip uninstall -y opencv-python opencv-python-headless\n"
        "       sudo apt install -y python3-opencv\n"
        "  3) Or skip GUI: python main.py projector-test --save /tmp/proj.png --show mpv"
    )


def show_image_external(path: str | Path, tool: str = "mpv") -> int:
    """
    Fullscreen show via external player (works better than OpenCV Qt on Pi HDMI).

    tool: mpv | feh | auto
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    ensure_gui_env()

    def run(cmd: list[str]) -> int:
        print("exec:", " ".join(cmd))
        return subprocess.run(cmd, check=False).returncode

    tools = []
    if tool == "auto":
        tools = ["mpv", "feh"]
    else:
        tools = [tool]

    errors = []
    for name in tools:
        if not shutil.which(name):
            errors.append(f"{name} not installed")
            continue
        if name == "mpv":
            # DRM/Wayland/X — mpv picks a working VO when session env is set
            code = run(
                [
                    "mpv",
                    "--fs",
                    "--image-display-duration=inf",
                    "--loop-file=inf",
                    str(path),
                ]
            )
            return code
        if name == "feh":
            code = run(["feh", "--fullscreen", "--auto-zoom", str(path)])
            return code
    raise RuntimeError(
        "No external viewer worked ("
        + "; ".join(errors)
        + "). Install with: sudo apt install mpv"
    )


class MpvFrameSink:
    """Live fullscreen frames via mpv rawvideo stdin (reliable on Pi HDMI)."""

    def __init__(self, width: int, height: int, fps: float = 30.0) -> None:
        if not shutil.which("mpv"):
            raise RuntimeError("mpv not found — install with: sudo apt install mpv")
        ensure_gui_env()
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps) if fps > 0 else 30.0
        self._frame_bytes = self.width * self.height * 3
        cmd = [
            "mpv",
            "--fs",
            "--no-cache",
            "--untimed",
            "--quiet",
            "--no-terminal",
            f"--demuxer-rawvideo-w={self.width}",
            f"--demuxer-rawvideo-h={self.height}",
            f"--demuxer-rawvideo-fps={self.fps:.3f}",
            "--demuxer-rawvideo-mp=bgr24",
            "--demuxer=rawvideo",
            "-",
        ]
        print("exec:", " ".join(cmd))
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            bufsize=0,
        )

    @property
    def alive(self) -> bool:
        return self._proc.poll() is None

    def show(self, frame_bgr: np.ndarray) -> None:
        if not self.alive or self._proc.stdin is None:
            raise RuntimeError("mpv process ended")
        h, w = frame_bgr.shape[:2]
        if (w, h) != (self.width, self.height):
            frame_bgr = cv2.resize(
                frame_bgr, (self.width, self.height), interpolation=cv2.INTER_LINEAR
            )
        if not frame_bgr.flags["C_CONTIGUOUS"] or frame_bgr.dtype != np.uint8:
            frame_bgr = np.ascontiguousarray(frame_bgr, dtype=np.uint8)
        try:
            self._proc.stdin.write(frame_bgr.tobytes())
        except BrokenPipeError as exc:
            raise RuntimeError("mpv pipe closed") from exc

    def close(self) -> None:
        if self._proc.stdin is not None:
            try:
                self._proc.stdin.close()
            except Exception:
                pass
        try:
            self._proc.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            self._proc.wait(timeout=1.0)

    def __enter__(self) -> "MpvFrameSink":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
