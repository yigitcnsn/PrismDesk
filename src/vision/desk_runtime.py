"""Multi-threaded live desk pipeline.

Layers (each on its own thread):
  frame   — camera read + undistort
  mat     — mat corner detect
  object  — silhouette measure
  display — projector HUD
  hub-*   — home-hub debug layers (raw/mat/object/final), one thread each
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional

import cv2
import numpy as np

from src.core.home_hub import HUB_COMMAND_STOP, HomeHubPublisher, OverlayFlags
from src.measure.mat import MatConfig, detect_mat_corners
from src.measure.object import analyze_object
from src.measure.perspective import warp_to_mat_plane
from src.measure.shape import ObjectAnalysis
from src.vision.desk import draw_debug_camera, draw_desk_hud, draw_idle_hud, format_object_metrics
from src.vision.homography import CamProjectorHomography
from src.vision.undistort import Undistorter


ShowFn = Callable[[np.ndarray], None]


@dataclass
class DeskRuntimeConfig:
    mat_config: MatConfig
    measure_config: MatConfig
    undistorter: Undistorter
    mat_every: int = 12
    object_every: int = 15
    do_object: bool = True
    homography: Optional[CamProjectorHomography] = None
    rotate_degrees: int = 180
    hub_publish_every: float = 0.2  # seconds between hub layer rebuilds
    hub_config_every: float = 1.0


@dataclass
class _Shared:
    stop: threading.Event = field(default_factory=threading.Event)
    # Latest undistorted camera frame
    frame_lock: threading.Lock = field(default_factory=threading.Lock)
    frame: Optional[np.ndarray] = None
    frame_id: int = 0
    cam_error: Optional[str] = None
    # Mat
    mat_lock: threading.Lock = field(default_factory=threading.Lock)
    mat_corners: Optional[np.ndarray] = None
    # Object
    object_lock: threading.Lock = field(default_factory=threading.Lock)
    analysis: Optional[ObjectAnalysis] = None
    last_metrics: str = ""
    # Projector / browser Visual flags (hub + phone)
    overlay_lock: threading.Lock = field(default_factory=threading.Lock)
    projector_overlays: OverlayFlags = field(default_factory=OverlayFlags)
    browser_overlays: OverlayFlags = field(default_factory=OverlayFlags)
    mode: str = "desk"
    # Display pacing
    display_count: int = 0
    t0: float = field(default_factory=time.time)


class DeskRuntime:
    """Owns worker threads for desk vision + optional home-hub layer publishers."""

    def __init__(
        self,
        *,
        camera: Any,
        cfg: DeskRuntimeConfig,
        hud_size: tuple[int, int],
        show_fn: ShowFn,
        alive_fn: Optional[Callable[[], bool]] = None,
        hub: Optional[HomeHubPublisher] = None,
    ) -> None:
        self._cam = camera
        self._cfg = cfg
        self._hud_w, self._hud_h = hud_size
        self._show = show_fn
        self._alive = alive_fn or (lambda: True)
        self._hub = hub
        self._shared = _Shared()
        self._threads: List[threading.Thread] = []
        if hub is not None:
            self._shared.projector_overlays = OverlayFlags(
                mat=hub.projector_overlays.mat,
                object=hub.projector_overlays.object,
            )
            self._shared.browser_overlays = OverlayFlags(
                mat=hub.browser_overlays.mat,
                object=hub.browser_overlays.object,
            )
            self._shared.mode = hub.mode if hub.mode in ("desk", "idle") else "desk"

    def start(self) -> None:
        self._shared.t0 = time.time()
        workers = [
            ("desk-frame", self._frame_loop),
            ("desk-mat", self._mat_loop),
            ("desk-object", self._object_loop),
            ("desk-display", self._display_loop),
        ]
        if self._hub is not None:
            for layer in self._hub.enabled_layers:
                workers.append((f"desk-hub-{layer}", self._hub_layer_loop_factory(layer)))
            workers.append(("desk-hub-config", self._hub_config_loop))
            workers.append(("desk-hub-state", self._hub_state_loop))
        for name, target in workers:
            t = threading.Thread(target=target, name=name, daemon=True)
            self._threads.append(t)
            t.start()
        print(
            "Desk threads: "
            + ", ".join(t.name.replace("desk-", "") for t in self._threads),
            flush=True,
        )

    def stop(self) -> None:
        self._shared.stop.set()
        for t in self._threads:
            t.join(timeout=2.0)
        self._threads.clear()

    def run_until_stop(self) -> None:
        """Block until Ctrl+C / sink death / stop flag."""
        try:
            while not self._shared.stop.is_set():
                if not self._alive():
                    print("video sink closed — exiting", flush=True)
                    break
                time.sleep(0.05)
        except KeyboardInterrupt:
            print("\nInterrupted", flush=True)
        finally:
            self.stop()

    def _snapshot_frame(self) -> tuple[Optional[np.ndarray], int]:
        with self._shared.frame_lock:
            if self._shared.frame is None:
                return None, self._shared.frame_id
            return self._shared.frame.copy(), self._shared.frame_id

    def _fps(self) -> float:
        elapsed = max(time.time() - self._shared.t0, 1e-6)
        return self._shared.display_count / elapsed

    def _frame_loop(self) -> None:
        und = self._cfg.undistorter
        while not self._shared.stop.is_set():
            try:
                frame = self._cam.read()
                if und.enabled:
                    frame = und.apply(frame)
                with self._shared.frame_lock:
                    self._shared.frame = frame
                    self._shared.frame_id += 1
                    self._shared.cam_error = None
            except Exception as vis_exc:  # noqa: BLE001
                with self._shared.frame_lock:
                    self._shared.cam_error = str(vis_exc)
                time.sleep(0.2)

    def _mat_loop(self) -> None:
        last_id = -1
        every = max(1, int(self._cfg.mat_every))
        seen = 0
        while not self._shared.stop.is_set():
            frame, fid = self._snapshot_frame()
            if frame is None or fid == last_id:
                time.sleep(0.002)
                continue
            last_id = fid
            seen += 1
            if (seen - 1) % every != 0:
                continue
            try:
                found = detect_mat_corners(frame, self._cfg.mat_config)
            except Exception as vis_exc:  # noqa: BLE001
                print(f"mat detect skipped: {vis_exc}", flush=True)
                continue
            if found is not None:
                with self._shared.mat_lock:
                    self._shared.mat_corners = found

    def _object_loop(self) -> None:
        if not self._cfg.do_object:
            return
        last_id = -1
        every = max(1, int(self._cfg.object_every))
        seen = 0
        while not self._shared.stop.is_set():
            frame, fid = self._snapshot_frame()
            if frame is None or fid == last_id:
                time.sleep(0.002)
                continue
            last_id = fid
            seen += 1
            if (seen - 1) % every != 0:
                continue
            with self._shared.mat_lock:
                mat = (
                    None
                    if self._shared.mat_corners is None
                    else self._shared.mat_corners.copy()
                )
            if mat is None:
                continue
            try:
                warped, _ = warp_to_mat_plane(frame, mat, self._cfg.measure_config)
                analysis = analyze_object(warped, self._cfg.measure_config)
            except Exception as vis_exc:  # noqa: BLE001
                print(f"object measure skipped: {vis_exc}", flush=True)
                continue
            with self._shared.object_lock:
                self._shared.analysis = analysis
                if analysis is not None:
                    metrics = format_object_metrics(analysis)
                    if metrics != self._shared.last_metrics:
                        print(metrics, flush=True)
                        self._shared.last_metrics = metrics

    def _display_loop(self) -> None:
        hud = np.zeros((self._hud_h, self._hud_w, 3), dtype=np.uint8)
        while not self._shared.stop.is_set():
            if not self._alive():
                self._shared.stop.set()
                break
            frame, _fid = self._snapshot_frame()
            with self._shared.frame_lock:
                cam_err = self._shared.cam_error
            if frame is None:
                hud[:] = 0
                if cam_err:
                    cv2.putText(
                        hud,
                        "camera reconnecting…",
                        (24, 64),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 140, 255),
                        2,
                        cv2.LINE_AA,
                    )
                try:
                    self._show(hud)
                except Exception:
                    pass
                time.sleep(0.05)
                continue

            with self._shared.mat_lock:
                mat = (
                    None
                    if self._shared.mat_corners is None
                    else self._shared.mat_corners.copy()
                )
            with self._shared.object_lock:
                analysis = self._shared.analysis
            with self._shared.overlay_lock:
                mode = self._shared.mode
                overlays = OverlayFlags(
                    mat=self._shared.projector_overlays.mat,
                    object=self._shared.projector_overlays.object,
                )
            fps_live = self._fps()
            if mode == "idle":
                draw_idle_hud(hud)
            else:
                draw_desk_hud(
                    hud,
                    mat_corners=mat,
                    mat_config=self._cfg.mat_config,
                    src_size=(frame.shape[1], frame.shape[0]),
                    fps_live=fps_live,
                    analysis=analysis if self._cfg.do_object else None,
                    measure_config=self._cfg.measure_config,
                    overlays=overlays,
                    homography=self._cfg.homography,
                )
            try:
                self._show(hud)
            except Exception as vis_exc:  # noqa: BLE001
                print(f"display failed: {vis_exc}", flush=True)
                self._shared.stop.set()
                break
            self._shared.display_count += 1
            time.sleep(0.001)

    def _hub_config_loop(self) -> None:
        assert self._hub is not None
        every = max(0.2, float(self._cfg.hub_config_every))
        while not self._shared.stop.is_set():
            try:
                ctrl = self._hub.fetch_control()
                with self._shared.overlay_lock:
                    self._shared.projector_overlays = OverlayFlags(
                        mat=ctrl.projector.mat, object=ctrl.projector.object
                    )
                    self._shared.browser_overlays = OverlayFlags(
                        mat=ctrl.browser.mat, object=ctrl.browser.object
                    )
                    self._shared.mode = ctrl.mode
                if ctrl.command == HUB_COMMAND_STOP:
                    print("home-hub: stop from phone", flush=True)
                    self._hub.clear_command()
                    self._shared.stop.set()
                    break
            except Exception:
                pass
            self._shared.stop.wait(every)

    def _hub_state_loop(self) -> None:
        assert self._hub is not None
        every = max(0.1, float(self._cfg.hub_publish_every))
        while not self._shared.stop.is_set():
            frame, _fid = self._snapshot_frame()
            with self._shared.mat_lock:
                mat_ok = self._shared.mat_corners is not None
            with self._shared.object_lock:
                analysis = self._shared.analysis
            with self._shared.overlay_lock:
                local = OverlayFlags(
                    mat=self._shared.projector_overlays.mat,
                    object=self._shared.projector_overlays.object,
                )
                browser = OverlayFlags(
                    mat=self._shared.browser_overlays.mat,
                    object=self._shared.browser_overlays.object,
                )
                mode = self._shared.mode
            self._hub.projector_overlays = local
            self._hub.browser_overlays = browser
            self._hub.mode = mode
            fps_live = self._fps()
            proj = self._hub.projector_overlays
            state = {
                "fps": round(fps_live, 2),
                "mode": mode,
                "mat_locked": mat_ok,
                "object": (
                    format_object_metrics(analysis) if analysis is not None else None
                ),
                "capture": (
                    f"{frame.shape[1]}x{frame.shape[0]}" if frame is not None else None
                ),
                "overlays": proj.as_list(),
                "projector_overlays": proj.as_list(),
                "browser_overlays": browser.as_list(),
                "config": {
                    "projector": proj.as_dict(),
                    "browser": browser.as_dict(),
                    "overlays": proj.as_dict(),
                    "mode": mode,
                },
                "rotate": int(self._cfg.rotate_degrees),
            }
            try:
                self._hub.publish_state(state)
            except Exception:
                pass
            self._shared.stop.wait(every)

    def _hub_layer_loop_factory(self, layer: str) -> Callable[[], None]:
        def _loop() -> None:
            assert self._hub is not None
            every = max(0.05, float(self._cfg.hub_publish_every))
            while not self._shared.stop.is_set():
                t_start = time.time()
                try:
                    self._publish_one_layer(layer)
                except Exception as vis_exc:  # noqa: BLE001
                    print(f"hub layer {layer} skipped: {vis_exc}", flush=True)
                elapsed = time.time() - t_start
                self._shared.stop.wait(max(0.0, every - elapsed))

        return _loop

    def _publish_one_layer(self, layer: str) -> None:
        assert self._hub is not None
        frame, _fid = self._snapshot_frame()
        if frame is None:
            return
        with self._shared.mat_lock:
            mat = (
                None
                if self._shared.mat_corners is None
                else self._shared.mat_corners.copy()
            )
        with self._shared.object_lock:
            analysis = self._shared.analysis
        with self._shared.overlay_lock:
            browser = OverlayFlags(
                mat=self._shared.browser_overlays.mat,
                object=self._shared.browser_overlays.object,
            )
        fps_live = self._fps()
        measure_cfg = self._cfg.measure_config
        mat_cfg = self._cfg.mat_config
        do_object = self._cfg.do_object

        if layer == "raw":
            img = frame
        elif layer == "mat":
            img = draw_debug_camera(
                frame,
                mat_corners=mat,
                mat_config=mat_cfg,
                fps_live=fps_live,
                mat_ok=mat is not None,
                analysis=None,
                measure_config=measure_cfg,
                overlays=OverlayFlags(mat=True, object=False),
            )
        elif layer == "object":
            img = draw_debug_camera(
                frame,
                mat_corners=mat,
                mat_config=mat_cfg,
                fps_live=fps_live,
                mat_ok=mat is not None,
                analysis=analysis if do_object else None,
                measure_config=measure_cfg,
                overlays=OverlayFlags(mat=False, object=True),
            )
        else:  # final
            img = draw_debug_camera(
                frame,
                mat_corners=mat,
                mat_config=mat_cfg,
                fps_live=fps_live,
                mat_ok=mat is not None,
                analysis=analysis if do_object else None,
                measure_config=measure_cfg,
                overlays=browser,
            )
        self._hub.publish_layer_frame(layer, img)
