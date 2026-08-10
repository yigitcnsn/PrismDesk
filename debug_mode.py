"""PrismDesk local debug GUI (Tkinter).

Browse example stills or optionally use a live camera. Test idle / hands / desk
without a Pi or projector.

Run:
  python debug_mode.py
  python main.py debug

Logs:
  terminal (stderr) + debug_dumps/debug_mode.log + on-screen Log panel
"""

from __future__ import annotations

import logging
import os
import sys
import time
import traceback
import tkinter as tk
from dataclasses import replace
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from tkinter import filedialog, messagebox
from typing import Optional, Tuple, Union

# macOS system Tk deprecation noise
os.environ.setdefault("TK_SILENCE_DEPRECATION", "1")

import cv2
import numpy as np
from PIL import Image, ImageTk

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.home_hub import OverlayFlags
from src.measure.mat import MatConfig, detect_mat_corners, load_mat_config
from src.measure.object import analyze_object
from src.measure.perspective import warp_to_mat_plane
from src.vision.camera import Camera, CameraConfig, load_camera_config
from src.vision.desk import (
    draw_debug_camera,
    draw_desk_hud,
    draw_idle_hud,
    format_object_metrics,
)
from src.vision.frame_source import ImageFolderSource, list_example_images
from src.vision.hands import HandTracker
from src.vision.undistort import Undistorter

DEFAULT_EXAMPLES = ROOT / "examples"
DEFAULT_DUMPS = ROOT / "debug_dumps"
DEFAULT_LOG = DEFAULT_DUMPS / "debug_mode.log"
MODES = ("idle", "hands", "desk")
PREVIEW_MAX = 900

# Explicit colors — avoid ttk/Aqua blank-window bug on macOS system Tk.
BG = "#1c1c1c"
PANEL = "#252525"
FG = "#f2f2f2"
MUTED = "#9a9a9a"
BTN = "#3d3d3d"
ENTRY = "#111111"
ACCENT = "#4aa3ff"
LOG_FG = "#b6f5b6"

LOG = logging.getLogger("prismdesk.debug_mode")

FrameSource = Union[Camera, ImageFolderSource]


def setup_logging(log_path: Path = DEFAULT_LOG) -> Path:
    """Configure console + rotating file logging. Safe to call more than once."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    LOG.setLevel(logging.DEBUG)
    LOG.handlers.clear()
    LOG.propagate = False

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    sh = logging.StreamHandler(sys.stderr)
    sh.setLevel(logging.DEBUG)
    sh.setFormatter(fmt)
    LOG.addHandler(sh)

    fh = RotatingFileHandler(
        log_path, maxBytes=1_000_000, backupCount=3, encoding="utf-8"
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    LOG.addHandler(fh)

    LOG.debug("logging ready → %s", log_path)
    return log_path


class TextWidgetHandler(logging.Handler):
    """Push log lines into a Tk Text widget (thread-safe via after)."""

    def __init__(self, root: tk.Misc, widget: tk.Text) -> None:
        super().__init__(level=logging.DEBUG)
        self._root = root
        self._widget = widget
        self.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")
        )

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record) + "\n"

            def _append() -> None:
                try:
                    self._widget.configure(state="normal")
                    self._widget.insert("end", msg)
                    self._widget.see("end")
                    # Cap length so the widget does not grow forever.
                    line_count = int(self._widget.index("end-1c").split(".")[0])
                    if line_count > 800:
                        self._widget.delete("1.0", f"{line_count - 600}.0")
                    self._widget.configure(state="disabled")
                except tk.TclError:
                    pass

            self._root.after(0, _append)
        except Exception:
            self.handleError(record)


def _btn(parent: tk.Misc, **kw: object) -> tk.Button:
    opts = {
        "bg": BTN,
        "fg": FG,
        "activebackground": "#555555",
        "activeforeground": FG,
        "relief": "raised",
        "bd": 1,
        "highlightthickness": 0,
        "padx": 8,
        "pady": 4,
    }
    opts.update(kw)
    return tk.Button(parent, **opts)  # type: ignore[arg-type]


def _label(parent: tk.Misc, **kw: object) -> tk.Label:
    opts = {"bg": PANEL, "fg": FG, "anchor": "w"}
    opts.update(kw)
    return tk.Label(parent, **opts)  # type: ignore[arg-type]


def _check(parent: tk.Misc, **kw: object) -> tk.Checkbutton:
    opts = {
        "bg": PANEL,
        "fg": FG,
        "activebackground": PANEL,
        "activeforeground": FG,
        "selectcolor": ENTRY,
        "highlightthickness": 0,
        "anchor": "w",
    }
    opts.update(kw)
    return tk.Checkbutton(parent, **opts)  # type: ignore[arg-type]


def _radio(parent: tk.Misc, **kw: object) -> tk.Radiobutton:
    opts = {
        "bg": PANEL,
        "fg": FG,
        "activebackground": PANEL,
        "activeforeground": FG,
        "selectcolor": ENTRY,
        "highlightthickness": 0,
        "anchor": "w",
    }
    opts.update(kw)
    return tk.Radiobutton(parent, **opts)  # type: ignore[arg-type]


class DebugGui:
    """Tkinter harness: images by default; camera is optional."""

    def __init__(
        self,
        *,
        examples_dir: Path = DEFAULT_EXAMPLES,
        mat_config: Optional[MatConfig] = None,
        camera_config: Optional[CameraConfig] = None,
        device: Optional[int] = None,
        dump_dir: Path = DEFAULT_DUMPS,
        measure_px_per_cm: float = 20.0,
        no_undistort: bool = False,
        start_source: str = "images",
        start_mode: str = "desk",
    ) -> None:
        self.examples_dir = Path(examples_dir)
        self.examples_dir.mkdir(parents=True, exist_ok=True)
        self.dump_dir = Path(dump_dir)
        self.mat_config = mat_config or load_mat_config(ROOT / "config" / "mat.yaml")
        self._base_mat = self.mat_config
        self.measure_px_per_cm = max(5.0, float(measure_px_per_cm))
        self._rebuild_measure_config()

        self.camera_config = camera_config or CameraConfig()
        self._preferred_device = device
        if device is not None:
            self.camera_config.device_indices = [int(device)] + [
                i for i in self.camera_config.device_indices if i != int(device)
            ]

        self.no_undistort = bool(no_undistort)
        self.undistorter = Undistorter(self.camera_config)

        self.source_kind = "images" if start_source != "camera" else "camera"
        self.mode = start_mode if start_mode in MODES else "desk"
        self.view = "camera"
        self.show_object = True
        self.show_hands = True
        self.show_mat = True

        self._source: Optional[FrameSource] = None
        self._tracker: Optional[HandTracker] = None
        self._hands: list = []
        self._mat_corners = None
        self._analysis = None
        self._last_metrics = ""
        self._frames = 0
        self._t0 = time.time()
        self._photo: Optional[ImageTk.PhotoImage] = None
        self._running = False
        self._tick_after: Optional[str] = None
        self._ui_log_handler: Optional[TextWidgetHandler] = None
        self._tick_errors = 0

        LOG.info("creating Tk root (python=%s tk=%s)", sys.version.split()[0], tk.TkVersion)
        self.root = tk.Tk()
        self.root.title("PrismDesk Debug")
        self.root.geometry("1280x820")
        self.root.minsize(1000, 640)
        self.root.configure(bg=BG)
        LOG.info("Tk root ok display=%r", self.root.winfo_screenwidth())

        LOG.info("building UI…")
        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_quit)
        kids = self.root.winfo_children()
        LOG.info(
            "UI built: %d top-level children, left_w=%s preview=%s",
            len(kids),
            getattr(self, "_left_frame", None),
            getattr(self, "preview_label", None),
        )
        for i, child in enumerate(kids):
            LOG.debug(
                "  child[%d]=%s manager=%s",
                i,
                child.winfo_class(),
                child.winfo_manager(),
            )

        self._refresh_image_list()
        if self.source_kind == "camera":
            LOG.info("initial source=camera")
            if not self._try_open_camera(quiet=True):
                LOG.warning("camera failed at startup — falling back to images")
                self.source_kind = "images"
                self.var_source.set("images")
                self._try_open_images(quiet=True)
        else:
            LOG.info("initial source=images dir=%s", self.examples_dir)
            self._try_open_images(quiet=True)
        self._sync_controls()
        if self._source is None:
            LOG.warning("no frame source yet — showing placeholder")
            self._show_placeholder(
                "Add JPG/PNG/HEIC to examples/\nthen click Refresh list\nCamera is optional"
            )
        LOG.info("DebugGui init complete")

    def _rebuild_measure_config(self) -> None:
        ppc = max(5.0, float(self.measure_px_per_cm))
        self.measure_config = replace(self._base_mat, px_per_cm=ppc)
        if self._base_mat.px_per_cm > 0:
            scale = ppc / float(self._base_mat.px_per_cm)
            self.measure_config = replace(
                self.measure_config,
                object_border_margin_px=max(
                    4, int(round(self._base_mat.object_border_margin_px * scale))
                ),
            )

    def _section(self, parent: tk.Misc, title: str) -> tk.Frame:
        wrap = tk.Frame(parent, bg=PANEL)
        wrap.pack(fill="x", padx=10, pady=(10, 2))
        tk.Label(
            wrap,
            text=title,
            bg=PANEL,
            fg=ACCENT,
            font=("", 12, "bold"),
            anchor="w",
        ).pack(fill="x")
        body = tk.Frame(wrap, bg=PANEL)
        body.pack(fill="x", pady=(4, 0))
        return body

    def _build_ui(self) -> None:
        # Left panel — pure tk widgets (ttk/Aqua draws blank on macOS system Tk).
        left = tk.Frame(self.root, bg=PANEL)
        left.pack(side="left", fill="y")
        self._left_frame = left
        # Keep a readable sidebar width without collapsing content height.
        left_inner = tk.Frame(left, bg=PANEL, width=320)
        left_inner.pack(fill="both", expand=True)
        left_inner.pack_propagate(True)
        LOG.debug("left panel created")

        tk.Label(
            left_inner,
            text="PrismDesk Debug",
            bg=PANEL,
            fg=FG,
            font=("", 14, "bold"),
            anchor="w",
        ).pack(fill="x", padx=12, pady=(12, 4))
        tk.Label(
            left_inner,
            text="Camera is optional",
            bg=PANEL,
            fg=MUTED,
            anchor="w",
        ).pack(fill="x", padx=12, pady=(0, 8))

        # --- Source ---
        src = self._section(left_inner, "Source")
        self.var_source = tk.StringVar(value=self.source_kind)
        _radio(
            src,
            text="Images (examples/)",
            value="images",
            variable=self.var_source,
            command=self._on_source_change,
        ).pack(fill="x")
        _radio(
            src,
            text="Camera (optional)",
            value="camera",
            variable=self.var_source,
            command=self._on_source_change,
        ).pack(fill="x")

        # --- Examples ---
        ex = self._section(left_inner, "Examples folder")
        self.var_folder = tk.StringVar(value=str(self.examples_dir))
        folder_row = tk.Frame(ex, bg=PANEL)
        folder_row.pack(fill="x")
        tk.Entry(
            folder_row,
            textvariable=self.var_folder,
            bg=ENTRY,
            fg=FG,
            insertbackground=FG,
            relief="flat",
        ).pack(side="left", fill="x", expand=True)
        _btn(folder_row, text="…", width=3, command=self._browse_folder).pack(
            side="left", padx=(4, 0)
        )
        _btn(ex, text="Refresh list", command=self._refresh_image_list).pack(
            fill="x", pady=(6, 4)
        )

        self.list_images = tk.Listbox(
            ex,
            height=6,
            exportselection=False,
            bg=ENTRY,
            fg=FG,
            selectbackground="#3a5a80",
            selectforeground=FG,
            highlightthickness=0,
            relief="flat",
        )
        self.list_images.pack(fill="x")
        self.list_images.bind("<<ListboxSelect>>", self._on_image_select)
        nav = tk.Frame(ex, bg=PANEL)
        nav.pack(fill="x", pady=(4, 0))
        _btn(nav, text="Prev", command=self._prev_image).pack(
            side="left", expand=True, fill="x"
        )
        _btn(nav, text="Next", command=self._next_image).pack(
            side="left", expand=True, fill="x", padx=(4, 0)
        )

        # --- Camera ---
        cam = self._section(left_inner, "Camera (optional)")
        cam_row = tk.Frame(cam, bg=PANEL)
        cam_row.pack(fill="x")
        _label(cam_row, text="Device index").pack(side="left")
        self.var_device = tk.StringVar(
            value="" if self._preferred_device is None else str(self._preferred_device)
        )
        tk.Entry(
            cam_row,
            textvariable=self.var_device,
            width=6,
            bg=ENTRY,
            fg=FG,
            insertbackground=FG,
            relief="flat",
        ).pack(side="left", padx=8)
        self.var_undistort = tk.BooleanVar(value=not self.no_undistort)
        _check(
            cam, text="Undistort if calibrated", variable=self.var_undistort
        ).pack(fill="x", pady=(4, 0))

        # --- Mode ---
        mode = self._section(left_inner, "Mode")
        self.var_mode = tk.StringVar(value=self.mode)
        for m in MODES:
            _radio(
                mode,
                text=m,
                value=m,
                variable=self.var_mode,
                command=self._on_mode_change,
            ).pack(fill="x")

        # --- View ---
        view = self._section(left_inner, "View")
        self.var_view = tk.StringVar(value=self.view)
        _radio(
            view,
            text="Camera debug",
            value="camera",
            variable=self.var_view,
            command=self._on_view_change,
        ).pack(fill="x")
        _radio(
            view,
            text="Projector HUD",
            value="hud",
            variable=self.var_view,
            command=self._on_view_change,
        ).pack(fill="x")

        # --- Overlays ---
        ov = self._section(left_inner, "Overlays")
        self.var_mat = tk.BooleanVar(value=True)
        self.var_object = tk.BooleanVar(value=True)
        self.var_hands = tk.BooleanVar(value=True)
        _check(ov, text="Mat", variable=self.var_mat, command=self._on_overlay).pack(
            fill="x"
        )
        _check(
            ov, text="Object measure", variable=self.var_object, command=self._on_overlay
        ).pack(fill="x")
        _check(
            ov, text="Hands", variable=self.var_hands, command=self._on_overlay
        ).pack(fill="x")

        # --- Measure ---
        meas = self._section(left_inner, "Measure")
        _label(meas, text="measure_px_per_cm").pack(fill="x")
        self.var_ppc = tk.DoubleVar(value=self.measure_px_per_cm)
        ppc_row = tk.Frame(meas, bg=PANEL)
        ppc_row.pack(fill="x", pady=2)
        tk.Entry(
            ppc_row,
            textvariable=self.var_ppc,
            width=8,
            bg=ENTRY,
            fg=FG,
            insertbackground=FG,
            relief="flat",
        ).pack(side="left")
        _btn(ppc_row, text="Apply", command=self._on_ppc).pack(side="left", padx=6)

        # --- Actions ---
        actions = self._section(left_inner, "Actions")
        _btn(actions, text="Save frame", command=self._save_frame).pack(
            fill="x", pady=2
        )
        _btn(actions, text="Reprocess", command=self._force_tick).pack(
            fill="x", pady=2
        )
        _btn(actions, text="Quit", command=self._on_quit).pack(fill="x", pady=2)

        self.status = tk.StringVar(value="Ready")
        tk.Label(
            left_inner,
            textvariable=self.status,
            bg=PANEL,
            fg=MUTED,
            wraplength=280,
            justify="left",
            anchor="nw",
        ).pack(fill="x", padx=12, pady=10)

        # Right: preview + log
        right = tk.Frame(self.root, bg=BG)
        right.pack(side="left", fill="both", expand=True)
        self.preview_label = tk.Label(right, bg=BG, fg=FG)
        self.preview_label.pack(fill="both", expand=True, padx=8, pady=8)
        self.metrics = tk.StringVar(value="")
        tk.Label(
            right,
            textvariable=self.metrics,
            bg=BG,
            fg=ACCENT,
            anchor="w",
        ).pack(fill="x", padx=12, pady=(0, 4))

        log_wrap = tk.Frame(right, bg=BG)
        log_wrap.pack(fill="x", padx=8, pady=(0, 8))
        tk.Label(
            log_wrap, text="Log (also → debug_dumps/debug_mode.log)", bg=BG, fg=MUTED, anchor="w"
        ).pack(fill="x")
        log_fr = tk.Frame(log_wrap, bg=ENTRY)
        log_fr.pack(fill="x")
        self.log_text = tk.Text(
            log_fr,
            height=8,
            bg=ENTRY,
            fg=LOG_FG,
            insertbackground=FG,
            relief="flat",
            highlightthickness=0,
            state="disabled",
            wrap="word",
        )
        log_scroll = tk.Scrollbar(log_fr, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scroll.set)
        self.log_text.pack(side="left", fill="both", expand=True)
        log_scroll.pack(side="right", fill="y")

        self._ui_log_handler = TextWidgetHandler(self.root, self.log_text)
        LOG.addHandler(self._ui_log_handler)
        LOG.info("on-screen log panel attached")

        self.root.update_idletasks()
        LOG.debug(
            "geometry after build: %s req=%sx%s",
            self.root.geometry(),
            self.root.winfo_reqwidth(),
            self.root.winfo_reqheight(),
        )

    def _sync_controls(self) -> None:
        self.var_source.set(self.source_kind)
        self.var_mode.set(self.mode)
        self.var_view.set(self.view)
        self.var_mat.set(self.show_mat)
        self.var_object.set(self.show_object)
        self.var_hands.set(self.show_hands)
        self.var_ppc.set(self.measure_px_per_cm)
        self.var_undistort.set(not self.no_undistort)
        self.var_folder.set(str(self.examples_dir))

    def _set_status(self, text: str) -> None:
        self.status.set(text)
        LOG.info("status: %s", text)

    def _show_placeholder(self, message: str) -> None:
        LOG.debug("placeholder: %s", message.replace("\n", " | "))
        blank = np.zeros((440, 760, 3), dtype=np.uint8)
        blank[:] = (28, 28, 28)
        for i, line in enumerate(message.split("\n")):
            cv2.putText(
                blank,
                line,
                (28, 90 + i * 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                (0, 200, 255),
                2,
                cv2.LINE_AA,
            )
        self._show_bgr(blank)

    def _browse_folder(self) -> None:
        path = filedialog.askdirectory(
            initialdir=str(self.examples_dir),
            title="Select examples folder",
        )
        if not path:
            return
        self.examples_dir = Path(path)
        self.var_folder.set(str(self.examples_dir))
        self._refresh_image_list()
        if self.source_kind == "images":
            self._try_open_images()

    def _refresh_image_list(self) -> None:
        folder = Path(self.var_folder.get().strip() or self.examples_dir)
        self.examples_dir = folder
        folder.mkdir(parents=True, exist_ok=True)
        paths = list_example_images(folder)
        self.list_images.delete(0, tk.END)
        for p in paths:
            self.list_images.insert(tk.END, p.name)
        if not paths:
            self._set_status(f"No images in {folder} — drop files then Refresh")
        else:
            self._set_status(f"{len(paths)} image(s) in {folder.name}/")

    def _on_image_select(self, _event: object = None) -> None:
        if self.source_kind != "images" or not isinstance(self._source, ImageFolderSource):
            return
        sel = self.list_images.curselection()
        if not sel:
            return
        self._source.goto(int(sel[0]))
        self._mat_corners = None
        self._analysis = None
        self._set_status(
            f"image {self._source.path.name} ({self._source.index + 1}/{self._source.count})"
        )
        self._force_tick()

    def _prev_image(self) -> None:
        if isinstance(self._source, ImageFolderSource):
            self._source.prev()
            self._select_list_index(self._source.index)
            self._mat_corners = None
            self._analysis = None
            self._set_status(
                f"image {self._source.path.name} ({self._source.index + 1}/{self._source.count})"
            )
            self._force_tick()

    def _next_image(self) -> None:
        if isinstance(self._source, ImageFolderSource):
            self._source.next()
            self._select_list_index(self._source.index)
            self._mat_corners = None
            self._analysis = None
            self._set_status(
                f"image {self._source.path.name} ({self._source.index + 1}/{self._source.count})"
            )
            self._force_tick()

    def _select_list_index(self, index: int) -> None:
        self.list_images.selection_clear(0, tk.END)
        if 0 <= index < self.list_images.size():
            self.list_images.selection_set(index)
            self.list_images.see(index)

    def _on_source_change(self) -> None:
        if self.var_source.get() == "camera":
            if not self._try_open_camera():
                self.var_source.set("images")
                self.source_kind = "images"
        else:
            self._try_open_images()

    def _on_mode_change(self) -> None:
        self.mode = self.var_mode.get()
        self._mat_corners = None
        self._analysis = None
        self._set_status(f"mode={self.mode}")
        self._force_tick()

    def _on_view_change(self) -> None:
        self.view = self.var_view.get()
        self._force_tick()

    def _on_overlay(self) -> None:
        self.show_mat = bool(self.var_mat.get())
        self.show_object = bool(self.var_object.get())
        self.show_hands = bool(self.var_hands.get())
        self._force_tick()

    def _on_ppc(self) -> None:
        try:
            self.measure_px_per_cm = float(self.var_ppc.get())
        except (TypeError, ValueError, tk.TclError):
            return
        self._rebuild_measure_config()
        self._analysis = None
        self._force_tick()

    def _close_source(self) -> None:
        if self._source is not None:
            try:
                self._source.close()
            except Exception:
                pass
        self._source = None

    def _apply_device_from_ui(self) -> None:
        raw = self.var_device.get().strip()
        if not raw:
            return
        try:
            idx = int(raw)
        except ValueError:
            return
        self.camera_config.device_indices = [idx] + [
            i for i in self.camera_config.device_indices if i != idx
        ]

    def _try_open_images(self, *, quiet: bool = False) -> bool:
        LOG.info("open images from %s", self.var_folder.get())
        self._close_source()
        self.examples_dir = Path(self.var_folder.get().strip() or self.examples_dir)
        self.examples_dir.mkdir(parents=True, exist_ok=True)
        self._refresh_image_list()
        try:
            src = ImageFolderSource.from_folder(self.examples_dir)
            src.open()
        except Exception as exc:  # noqa: BLE001
            LOG.warning("images open failed: %s", exc)
            self.source_kind = "images"
            self._set_status(f"No images yet — {exc}")
            if not quiet:
                messagebox.showinfo(
                    "Images",
                    f"Drop stills into:\n{self.examples_dir}\n\nThen click Refresh list.",
                )
            self._show_placeholder(
                "No images in examples/\nAdd JPG/PNG/HEIC then Refresh"
            )
            return False
        self._source = src
        self.source_kind = "images"
        self.var_source.set("images")
        self._select_list_index(src.index)
        LOG.info("images ok: %s (%d/%d)", src.path.name, src.index + 1, src.count)
        self._set_status(f"images: {src.path.name} ({src.index + 1}/{src.count})")
        self._force_tick()
        return True

    def _try_open_camera(self, *, quiet: bool = False) -> bool:
        LOG.info(
            "open camera indices=%s device_field=%r",
            self.camera_config.device_indices,
            self.var_device.get(),
        )
        self._close_source()
        self._apply_device_from_ui()
        try:
            cam = Camera(self.camera_config)
            cam.open()
        except Exception as exc:  # noqa: BLE001
            LOG.warning("camera open failed: %s", exc)
            self._set_status(f"Camera unavailable (optional): {exc}")
            if not quiet:
                messagebox.showinfo(
                    "Camera optional",
                    "Could not open a camera.\n"
                    "Use Images, or allow Camera for Terminal/Python in\n"
                    "System Settings → Privacy & Security → Camera.\n\n"
                    f"{exc}",
                )
            return False
        self._source = cam
        self.source_kind = "camera"
        self.var_source.set("camera")
        w, h, fps = cam.negotiated()
        LOG.info("camera ok idx=%s %sx%s @%.1f", cam.active_index, w, h, fps)
        self._set_status(f"camera idx={cam.active_index} {w}x{h}@{fps:.1f}")
        self._force_tick()
        return True

    def _ensure_tracker(self) -> HandTracker:
        if self._tracker is None:
            self._tracker = HandTracker(infer_size=(480, 270))
        return self._tracker

    def _read_frame(self) -> Optional[np.ndarray]:
        if self._source is None:
            return None
        frame = self._source.read()
        use_undistort = bool(self.var_undistort.get()) and not self.no_undistort
        if (
            use_undistort
            and self.source_kind == "camera"
            and self.undistorter.enabled
        ):
            frame = self.undistorter.apply(frame)
        return frame

    def _process(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        h, w = frame.shape[:2]
        hud = np.zeros((h, w, 3), dtype=np.uint8)
        elapsed = max(time.time() - self._t0, 1e-6)
        fps_live = self._frames / elapsed

        self.show_mat = bool(self.var_mat.get())
        self.show_object = bool(self.var_object.get())
        self.show_hands = bool(self.var_hands.get())
        self.mode = self.var_mode.get()
        self.view = self.var_view.get()

        if self.mode == "idle":
            draw_idle_hud(hud)
            return frame.copy(), hud

        if self.mode in ("hands", "desk") and self.show_hands:
            try:
                self._hands = self._ensure_tracker().process(frame)
            except Exception as exc:  # noqa: BLE001
                self._hands = []
                self._set_status(f"hands error: {exc}")
        else:
            self._hands = []

        if self.mode == "desk" and self.show_mat:
            found = detect_mat_corners(frame, self.mat_config)
            if found is not None:
                self._mat_corners = found
        elif self.mode != "desk":
            self._mat_corners = None

        if self.mode == "desk" and self.show_object and self._mat_corners is not None:
            try:
                warped, _ = warp_to_mat_plane(
                    frame, self._mat_corners, self.measure_config
                )
                self._analysis = analyze_object(warped, self.measure_config)
                if self._analysis is not None:
                    metrics = format_object_metrics(self._analysis)
                    if metrics != self._last_metrics:
                        self._last_metrics = metrics
                        self.metrics.set(metrics)
            except Exception as exc:  # noqa: BLE001
                self._analysis = None
                self._set_status(f"object skip: {exc}")
        else:
            self._analysis = None
            if self.mode != "desk":
                self.metrics.set("")

        overlays = OverlayFlags(
            mat=self.show_mat and self.mode == "desk",
            object=self.show_object and self.mode == "desk",
            hands=self.show_hands and self.mode in ("hands", "desk"),
        )
        cam_view = draw_debug_camera(
            frame,
            hands=self._hands,
            mat_corners=self._mat_corners if self.mode == "desk" else None,
            mat_config=self.mat_config,
            fps_live=fps_live,
            track_fps=fps_live,
            mat_ok=self._mat_corners is not None and self.mode == "desk",
            analysis=self._analysis,
            measure_config=self.measure_config,
            overlays=overlays,
        )
        draw_desk_hud(
            hud,
            hands=self._hands,
            mat_corners=self._mat_corners if self.mode == "desk" else None,
            mat_config=self.mat_config,
            src_size=(w, h),
            fps_live=fps_live,
            track_fps=fps_live,
            mat_ok=self._mat_corners is not None and self.mode == "desk",
            analysis=self._analysis,
            measure_config=self.measure_config,
            overlays=overlays,
        )
        return cam_view, hud

    def _show_bgr(self, bgr: np.ndarray) -> None:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        scale = min(1.0, PREVIEW_MAX / max(w, h))
        if scale < 1.0:
            rgb = cv2.resize(
                rgb,
                (max(1, int(w * scale)), max(1, int(h * scale))),
                interpolation=cv2.INTER_AREA,
            )
        image = Image.fromarray(rgb)
        self._photo = ImageTk.PhotoImage(image=image)
        self.preview_label.configure(image=self._photo)

    def _force_tick(self) -> None:
        if self._tick_after is not None:
            try:
                self.root.after_cancel(self._tick_after)
            except Exception:
                pass
            self._tick_after = None
        if self._running:
            self._tick()
        else:
            # Before mainloop / single shot for init
            frame = self._read_frame()
            if frame is not None:
                cam_view, hud = self._process(frame)
                show = cam_view if self.view == "camera" else hud
                self._show_bgr(show)

    def _tick(self) -> None:
        if not self._running:
            return
        try:
            frame = self._read_frame()
            if frame is not None:
                self._frames += 1
                cam_view, hud = self._process(frame)
                show = cam_view if self.view == "camera" else hud
                self._show_bgr(show)
                if self._frames == 1 or self._frames % 30 == 0:
                    LOG.debug(
                        "tick frame=%d mode=%s src=%s view=%s shape=%s",
                        self._frames,
                        self.mode,
                        self.source_kind,
                        self.view,
                        tuple(frame.shape),
                    )
                delay = 30 if self.source_kind == "camera" else 120
            else:
                if self._frames == 0:
                    LOG.debug("tick: no frame yet")
                delay = 200
            self._tick_after = self.root.after(delay, self._tick)
        except Exception:
            self._tick_errors += 1
            LOG.exception("tick failed (#%d)", self._tick_errors)
            if self._tick_errors <= 5:
                self._set_status(f"tick error — see log (#{self._tick_errors})")
            self._tick_after = self.root.after(500, self._tick)

    def _save_frame(self) -> None:
        frame = self._read_frame()
        if frame is None:
            messagebox.showinfo("Save", "No frame to save")
            return
        cam_view, hud = self._process(frame)
        show = cam_view if self.view == "camera" else hud
        self.dump_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        path = self.dump_dir / f"debug-{self.mode}-{self.source_kind}-{stamp}.jpg"
        cv2.imwrite(str(path), show)
        LOG.info("saved frame → %s", path)
        self._set_status(f"saved {path}")

    def _on_quit(self) -> None:
        LOG.info("quit requested")
        self._running = False
        if self._tick_after is not None:
            try:
                self.root.after_cancel(self._tick_after)
            except Exception:
                pass
        if self._ui_log_handler is not None:
            try:
                LOG.removeHandler(self._ui_log_handler)
            except Exception:
                pass
            self._ui_log_handler = None
        if self._tracker is not None:
            try:
                self._tracker.close()
            except Exception:
                pass
        self._close_source()
        self.root.destroy()
        LOG.info("window destroyed")

    def run(self) -> int:
        LOG.info("entering mainloop")
        self._running = True
        self._t0 = time.time()
        self._frames = 0
        self._tick()
        self.root.mainloop()
        LOG.info("mainloop exited frames=%d", self._frames)
        return 0


def run_debug_gui(**kwargs: object) -> int:
    return DebugGui(**kwargs).run()  # type: ignore[arg-type]


def main() -> int:
    import argparse

    log_path = setup_logging()
    print(f"PrismDesk debug_mode — logs: {log_path}", flush=True)
    LOG.info("==== debug_mode start ====")
    LOG.info("cwd=%s root=%s", Path.cwd(), ROOT)
    LOG.info("python=%s executable=%s", sys.version.replace("\n", " "), sys.executable)
    LOG.info("tkinter TkVersion=%s TclVersion=%s", tk.TkVersion, tk.TclVersion)

    parser = argparse.ArgumentParser(description="PrismDesk local debug GUI (Tkinter)")
    parser.add_argument("--examples", type=Path, default=DEFAULT_EXAMPLES)
    parser.add_argument(
        "--source", choices=("images", "camera"), default="images"
    )
    parser.add_argument(
        "--mode", choices=("idle", "hands", "desk"), default="desk"
    )
    parser.add_argument(
        "--camera-config", type=Path, default=ROOT / "config" / "camera.yaml"
    )
    parser.add_argument(
        "--mat-config", type=Path, default=ROOT / "config" / "mat.yaml"
    )
    parser.add_argument("--device", type=int, default=None)
    parser.add_argument("--no-undistort", action="store_true")
    parser.add_argument("--measure-px-per-cm", type=float, default=20.0)
    parser.add_argument("--dump-dir", type=Path, default=DEFAULT_DUMPS)
    args = parser.parse_args()
    LOG.info("args=%s", vars(args))

    try:
        cam_path = args.camera_config
        if not cam_path.is_file():
            example = ROOT / "config" / "camera.example.yaml"
            LOG.info(
                "camera.yaml missing — using %s",
                example if example.is_file() else cam_path,
            )
            cam_cfg = load_camera_config(example if example.is_file() else cam_path)
        else:
            cam_cfg = load_camera_config(cam_path)

        gui = DebugGui(
            examples_dir=args.examples,
            mat_config=load_mat_config(args.mat_config),
            camera_config=cam_cfg,
            device=args.device,
            dump_dir=args.dump_dir,
            measure_px_per_cm=float(args.measure_px_per_cm),
            no_undistort=bool(args.no_undistort),
            start_source=str(args.source),
            start_mode=str(args.mode),
        )
        return gui.run()
    except Exception:
        LOG.exception("fatal error in debug_mode")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
