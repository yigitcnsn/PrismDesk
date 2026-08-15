# PrismDesk

Spatial AR workbench for Raspberry Pi 5: overhead camera + HY300 projector turn a desk into a measured surface. Local AI / voice come later.

**Repo:** https://github.com/yigitcnsn/PrismDesk

---

## What’s working (alpha)

| Mode | Command | Notes |
|------|---------|--------|
| Photo measure | `python main.py measure PHOTO` | Detect 40×30 cm mat → warp → silhouette → cm |
| Camera calib | `python main.py calibrate-camera` | Chessboard → `config/camera.yaml` |
| Projector calib | `python main.py calibrate-projector` | Projected chessboard → cam↔proj H in `config/projector.yaml` |
| Desk (all-in-one) | `python main.py desk` | Defaults: capture 960x540, rotate 180, home-hub on, threaded layers |
| Desk → home-hub | `python main.py desk` | Same — hub on by default (`--no-home-hub` to disable) |
| Projector list | `python main.py projector-list` | DRM / wlr-randr / xrandr discovery |
| Projector test | `python main.py projector-test` | Fullscreen alignment pattern |

Live HUD uses **ffplay/mpv** (pip OpenCV Qt/xcb often aborts on Pi). After `calibrate-projector`, overlays use the saved cam↔projector homography; otherwise stretch.

---

## Hardware (current desk)

- Raspberry Pi 5
- USB camera (V4L2 MJPG, prefer 1080p @ 50 FPS when stable)
- HY300 projector on HDMI (`HDMI-A-1` / `card1-HDMI-A-1`)
- Black reference mat **40×30 cm**
- Desk ~159×65 cm; mount height ~186 cm (target for final overhead rig)

---

## Architecture (target)

```text
               [ Raspberry Pi 5 ]
                          │
         ┌────────────────┴────────────────┐
         ▼                                 ▼
[ Overhead camera ]                 [ HY300 projector ]
 (mat / objects)                     (spatial HUD)
         │                                 │
         └─────────────┬───────────────────┘
                       ▼
            [ PrismDesk core ]
         ├── Vision: mat detect + cam↔projector homography
         ├── Measure: silhouette → shape metrics (cm)
         ├── HUD: dark high-contrast overlay
         ├── home-hub debug bridge
         └── Later: pi-llm, voice
```

---

## Project layout

```text
PrismDesk/
├── main.py                 # CLI entry
├── config/
│   ├── mat.yaml            # 40×30 cm mat + detect knobs
│   ├── camera.example.yaml # copy → camera.yaml (gitignored)
│   ├── projector.example.yaml
│   └── home_hub.example.yaml
├── src/
│   ├── measure/            # Photo + mat-plane measurement
│   ├── vision/             # Camera, calib, projector, desk HUD
│   └── core/               # home-hub publisher
├── test/                   # Unit tests
├── requirements.txt
└── README.md
```

---

## Setup (Pi 5)

```bash
git clone https://github.com/yigitcnsn/PrismDesk.git
cd PrismDesk
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# System packages for live display / Wayland outputs
sudo apt install -y ffmpeg mpv wlr-randr

cp config/camera.example.yaml config/camera.yaml
cp config/projector.example.yaml config/projector.yaml
```

On aarch64, prefer system OpenCV if pip Qt fails:

```bash
pip uninstall -y opencv-python opencv-python-headless
sudo apt install -y python3-opencv
```

If SSH’ing into a desktop session:

```bash
export XDG_RUNTIME_DIR=/run/user/$(id -u)
export WAYLAND_DISPLAY=wayland-0   # or DISPLAY=:0
```

---

## Usage

### Photo measurement

```bash
python main.py measure path/to/photo.HEIC
# or JPEG/PNG
```

### Projector smoke test

```bash
python main.py projector-list
python main.py projector-test
```

### Camera ↔ projector homography

Assumes the full projector canvas is visible in the overhead camera. Projects a chessboard, detects it in the camera, saves `homography` into `config/projector.yaml`.

```bash
# Prefer same capture size + rotate you use for desk
python main.py calibrate-projector --capture 960x540 --rotate 180
# Pattern on projector via ffplay/mpv; status in terminal (no OpenCV GUI)
# Wait for board=yes → auto-sample → press c   (or add --auto-save)
# Optional windowed cam view: --cam-preview
python main.py desk
```

### Desk: mat + object measure

```bash
python main.py desk
# baked defaults: --capture 960x540 --rotate 180 --home-hub
# --no-home-hub / --no-object / --rotate 0 if needed
```

HUD shows FPS number only, plus mat/object overlays. Vision + home-hub layers each run on their own thread.

---

## Roadmap

### Done / in progress

- [x] Repo layout, venv, public-safe `.gitignore`
- [x] Photo mat detect → warp → object silhouette measure (cm)
- [x] USB camera capture + fisheye/pinhole calibration path
- [x] Projector HUD via ffplay/mpv
- [x] Live `desk` mode: mat + object measure
- [x] home-hub PrismDesk debug publisher (`--home-hub`)
- [x] Camera ↔ projector homography (projected chessboard)

### Next

- [ ] Higher live FPS
- [ ] Final overhead mount (Cam 3 Wide + HY300)
- [ ] `pi-llm` bridge
- [ ] Wake-word, STT, TTS

---

## License

MIT © [Ahmet Yiğitcan Şen](https://github.com/yigitcnsn)
