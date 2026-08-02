```markdown
# PrismDesk 📐

An interactive, spatial AR workbench running on Raspberry Pi 5. It uses an overhead camera and projector to transform a physical desk into a dynamic, gesture-controlled UI backed by local AI and voice control.

---

## Architecture Overview

```text
               [ Raspberry Pi 5 (8GB) ]
                          │
         ┌────────────────┴────────────────┐
         ▼                                 ▼
[ RPi Cam 3 Wide ]                 [ HY300 Projector ]
 (Desk & Hand Tracking)             (Spatial HUD Projection)
         │                                 │
         └─────────────┬───────────────────┘
                       ▼
            [ PrismDesk Core Engine ]
         ├── Vision: MediaPipe + Homography Matrix
         ├── UI: Fast GPU Rendering (Raylib / Pygame)
         └── Integrations: pi-llm & home-hub APIs

```

---

## Roadmap & Milestones

### Phase 1: Spatial Calibration

- [ ] Set up project structure and virtual environment (`uv` / `venv`).
- [ ] Mount RPi Camera 3 Wide and HY300 projector overhead (~186 cm above desk).
- [ ] Implement $3 \times 3$ Homography matrix calibration (`src/calibration/homography.py`) to map camera pixels directly to projected coordinates.

### Phase 2: Hand Tracking & Spatial UI

- [ ] Build multi-threaded MediaPipe hand-tracking pipeline to hit 30+ FPS without choking the main thread.
- [ ] Implement dark-mode rendering loop with high-contrast UI assets (`#00FFFF` Cyan / `#FF00FF` Magenta) designed specifically for light wood surfaces.
- [ ] Add basic spatial input detection (hover, pinch, virtual button press).

### Phase 3: Local Brain Integration

- [ ] Wire async HTTP/WebSocket handlers to stream responses from local `pi-llm` (`Qwen 2.5 3B` via Ollama).
- [ ] Add trigger endpoints to control local automation via `home-hub`.

### Phase 4: Voice Control

- [ ] Add lightweight wake-word engine (Porcupine/Precise) on a dedicated background thread.
- [ ] Pipe audio into `whisper.cpp` (INT8 quantized) for low-latency STT.
- [ ] Integrate `Piper TTS` to stream voice responses back through speakers.

---

## Project Structure

```text
PrismDesk/
├── src/
│   ├── vision/          # Camera capture & MediaPipe tracking
│   ├── calibration/     # Homography calibration scripts
│   ├── ui/              # Spatial HUD rendering loop
│   ├── voice/           # Wake-word, STT, and TTS modules
│   └── core/            # pi-llm and home-hub API bridges
├── config/              # Camera settings & homography matrices
├── docs/                # Mount specs & calibration guides
├── main.py              # Application entry point
└── requirements.txt

```

---

## Getting Started

```bash

```



# Clone

git clone [https://github.com/yigitcnsn/PrismDesk.git](https://github.com/yigitcnsn/PrismDesk.git)
cd PrismDesk

# Setup environment

python3 -m venv .venv
source .venv/bin/activate

# Install dependencies

pip install -r requirements.txt

# Run camera-projector calibration

python3 src/calibration/homography.py

```

---

## License

MIT © [Ahmet Yiğitcan Şen](https://www.google.com/search?q=https://github.com/yigitcnsn)

```

```

```

