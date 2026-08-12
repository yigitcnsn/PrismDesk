"""Background mic level meter (PortAudio / sounddevice when available)."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import yaml


@dataclass
class AudioConfig:
    device: Optional[str | int] = None
    sample_rate: int = 16000
    block_size: int = 1024
    channels: int = 1
    ema: float = 0.35


def load_audio_config(path: str | Path) -> AudioConfig:
    path = Path(path)
    if not path.is_file():
        return AudioConfig()
    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    device = raw.get("device", None)
    if device is not None and str(device).strip() == "":
        device = None
    return AudioConfig(
        device=device,
        sample_rate=int(raw.get("sample_rate", 16000)),
        block_size=int(raw.get("block_size", 1024)),
        channels=max(1, int(raw.get("channels", 1))),
        ema=float(raw.get("ema", 0.35)),
    )


class AudioLevelMeter:
    """Daemon thread publishing a smoothed 0..1 mic level."""

    def __init__(self, config: Optional[AudioConfig] = None) -> None:
        self.config = config or AudioConfig()
        self._lock = threading.Lock()
        self._level = 0.0
        self._available = False
        self._error: Optional[str] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    @property
    def level(self) -> float:
        with self._lock:
            return float(self._level)

    @property
    def available(self) -> bool:
        with self._lock:
            return bool(self._available)

    @property
    def error(self) -> Optional[str]:
        with self._lock:
            return self._error

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, name="prismdesk-audio", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
        self._thread = None

    def _set(self, *, level: Optional[float] = None, available: Optional[bool] = None, error: Optional[str] = None) -> None:
        with self._lock:
            if level is not None:
                self._level = float(level)
            if available is not None:
                self._available = bool(available)
            if error is not None:
                self._error = error

    def _loop(self) -> None:
        try:
            import sounddevice as sd
        except Exception as exc:  # noqa: BLE001
            self._set(available=False, level=0.0, error=f"sounddevice missing: {exc}")
            return

        cfg = self.config
        kwargs = max(0.0, min(1.0, float(cfg.ema)))
        smooth = 0.0
        try:
            with sd.InputStream(
                samplerate=cfg.sample_rate,
                blocksize=cfg.block_size,
                channels=cfg.channels,
                dtype="float32",
                device=cfg.device,
            ) as stream:
                self._set(available=True, error=None)
                while not self._stop.is_set():
                    block, _overflowed = stream.read(cfg.block_size)
                    if block is None or len(block) == 0:
                        time.sleep(0.01)
                        continue
                    mono = np.asarray(block, dtype=np.float32)
                    if mono.ndim > 1:
                        mono = mono.mean(axis=1)
                    rms = float(np.sqrt(np.mean(np.square(mono)) + 1e-12))
                    # Soft-clip mapping so speech sits visibly on the bar.
                    mapped = max(0.0, min(1.0, rms * 8.0))
                    smooth = (1.0 - kappa) * smooth + kappa * mapped
                    self._set(level=smooth, available=True)
        except Exception as exc:  # noqa: BLE001
            self._set(available=False, level=0.0, error=str(exc))
