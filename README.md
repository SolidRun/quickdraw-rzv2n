# Quick Draw

> **Notice:** This project is under active development. Configuration values, default parameters, and documented behavior may not reflect the latest state of the code. Users are responsible for verifying all settings against the source files before use in production. Bug reports and corrections are welcome.

Real-time sketch recognition on the **SolidRun RZ/V2N** board.

Draw on a touchscreen and the DRP-AI3 accelerator classifies your sketch from **345 categories** in ~1 ms. Single C++ binary — no server, no Python, no sockets at runtime.

| | |
|---|---|
| **Board** | RZ/V2N SR SOM (SolidRun HummingBoard iIoT) |
| **CPU** | ARM Cortex-A55 |
| **Accelerator** | DRP-AI3 (AI-MAC @ 1 GHz) |
| **Display** | 1024 x 600 touchscreen (Wayland) |
| **Model** | MobileNetV2, 345 classes, INT8 |
| **Inference** | ~1 ms |

## Quick Start

```bash
# Build (from host — Docker cross-compiles for ARM64)
cd board_app
./docker_build.sh

# Deploy
./deploy.sh <board-ip>

# Run (on board)
ssh root@<board-ip>
cd /home/root/quickdraw && ./run.sh
```

## Documentation

| Document | Contents |
|----------|----------|
| [docs/INSTALL.md](docs/INSTALL.md) | System setup: Docker image, Renesas toolchain, board prerequisites |
| [docs/TRAINING.md](docs/TRAINING.md) | Dataset download, model training, ONNX export |
| [docs/BUILD.md](docs/BUILD.md) | DRP-AI compilation, C++ build, packaging, deployment |
| [docs/APP.md](docs/APP.md) | Application architecture, configuration, controls |

## Project Layout

```
quickdraw/
+-- train/
|   +-- download_ndjson.py       Download + render strokes at 128x128
|   +-- train.py                 Train MobileNetV2, export ONNX
|   +-- data_128/                345 .npy files
|
+-- calibration/                 1,725 PNG images for INT8 quantization
+-- drpai_model/                 Compiled DRP-AI model
+-- best_model.pt                Trained weights (14 MB)
+-- qd_model.onnx                ONNX model (14 MB)
+-- categories.txt               345 class names
+-- generate_calibration.py      Calibration image generator
|
+-- board_app/
|   +-- src/                     C++ source (gui, inference, preprocessing)
|   +-- config.ini               DRP-AI frequencies and model path
|   +-- config.json              UI layout, colors, AI comments
|   +-- labels.txt               345 class names
|   +-- docker_build.sh          Build from host via Docker
|   +-- package.sh               Create deploy/ folder
|   +-- deploy.sh                SCP to board
|   +-- compile_model.sh         DRP-AI TVM compilation
|   +-- run.sh                   Board startup script
|   +-- lib/                     MERA2 runtime libraries
|   +-- deploy/                  Ready-to-copy package (46 MB)
|
+-- docs/
    +-- INSTALL.md
    +-- TRAINING.md
    +-- BUILD.md
    +-- APP.md
```
