# Build and Deploy

## Prerequisites

See [INSTALL.md](INSTALL.md) for full setup instructions (Docker image build, Renesas toolchain, board setup).

**Summary:** You need a running DRP-AI TVM Docker container with the RZ/V LP SDK, DRP-AI Translator, and MERA2 compiler installed.

---

## Step 1: Compile Model for DRP-AI

Run inside the Docker container.

```bash
cd board_app
./compile_model.sh \
    ../qd_model.onnx \
    ../calibration \
    qd_mobilenetv2 \
    1725
```

### What the script patches

The default DRP-AI TVM compile script has ImageNet defaults that are wrong for sketch data. `compile_model.sh` applies these patches:

| Patch | What changes |
|-------|-------------|
| Normalization | `mean=[0.485, 0.456, 0.406]` -> `mean=[0.0, 0.0, 0.0]` |
| | `stdev=[0.229, 0.224, 0.225]` -> `stdev=[1.0, 1.0, 1.0]` |
| Calibration resize | `resize(256) + center_crop(224)` -> `resize(128)` |
| Preprocessing shape | `[1, 480, 640, 3]` -> `[1, 128, 128, 3]` |
| Product check | `V2H` only -> `V2H` or `V2N` |

### Quantization settings

- Method: Percentile 99.99
- CPU operations: float32
- Output format: MERA2

### Output

```
drpai_model/qd_mobilenetv2/
+-- sub_0000__CPU_DRP_TVM/
|   +-- deploy.so              INT8 compiled model
|   +-- deploy.json            Model metadata
|   +-- deploy.params          Quantized parameters
+-- preprocess/                DRP-AI preprocessing config (10 files)
+-- mera.plan
```

---

## Step 2: Build C++ Application

From your host machine (not inside Docker).

```bash
cd board_app
./docker_build.sh
```

### What docker_build.sh does

1. Copies these items into the container: `src`, `toolchain`, `CMakeLists.txt`, `build.sh`, `config.ini`, `config.json`, `labels.txt`
2. Sources the RZ/V LP SDK environment
3. Runs CMake + make
4. Copies compiled binaries back to host `build/`
5. Runs `package.sh` to create `deploy/`

Use `--clean` for a full rebuild:

```bash
./docker_build.sh --clean
```

### Build configuration

| Setting | Value |
|---------|-------|
| C++ standard | 17 |
| Compiler flags | `-O3 -mtune=cortex-a55 -Wall` |
| Compile definitions | `V2H`, `V2N`, `KDLDRPAI` |
| Toolchain | `toolchain/runtime.cmake` |

### Targets

| Binary | Purpose | Libraries |
|--------|---------|-----------|
| `app_quickdraw_gui` | GTK3 GUI + DRP-AI inference | mera2_runtime, mera2_plan_io, drp_tvm_rt, pthread, gtk-3, gdk-3, cairo, glib-2.0, gobject-2.0, pango-1.0, pangocairo-1.0, gdk_pixbuf-2.0, atk-1.0, gio-2.0 |
| `app_quickdraw` | Socket server (legacy) | mera2_runtime, mera2_plan_io, drp_tvm_rt, pthread |

### Source files

| File | Purpose |
|------|---------|
| `src/main_gui.cpp` | Entry point: parses config.ini, loads DRP-AI model, launches GUI |
| `src/gui.cpp` / `gui.h` | GTK3 fullscreen canvas, touch+mouse input, predictions panel, AI commentary |
| `src/drpai_inference.cpp` / `.h` | MERA2 runtime wrapper: load model, set frequencies, run inference |
| `src/classification.cpp` / `.h` | Softmax (with overflow guard) + top-K extraction |
| `src/preprocessing.cpp` / `.h` | Ink detection, crop, pad, invert, area-resize, normalize |
| `src/define.h` | Default constants |
| `src/main.cpp` | Socket server entry point (legacy, for Python GUI) |

---

## Step 3: Package

Runs automatically after `docker_build.sh`. To run manually:

```bash
cd board_app
./package.sh
```

Creates `deploy/` containing:

```
deploy/                           46 MB total
+-- app_quickdraw_gui             13 MB   ARM64 binary
+-- config.ini                            DRP-AI config
+-- config.json                           UI config
+-- labels.txt                            345 class names
+-- run.sh                                Board startup script
+-- solidrun_logo.png                     Title bar logo
+-- model/qd_mobilenetv2/                 Compiled DRP-AI model
|   +-- sub_0000__CPU_DRP_TVM/
|   |   +-- deploy.so            12 MB
|   |   +-- deploy.json
|   |   +-- deploy.params
|   +-- preprocess/
|   +-- mera.plan
+-- lib/                                  MERA2 runtime libraries
    +-- libmera2_runtime.so
    +-- libmera2_plan_io.so
    +-- libdrp_tvm_rt.so
    +-- libdrp_rt.so
    +-- libarm_compute.so
    +-- libarm_compute_core.so
    +-- libarm_compute_graph.so
    +-- libacl_rt.so
```

---

## Step 4: Deploy to Board

```bash
cd board_app
./deploy.sh <board-ip>
```

### deploy.sh flags

| Flag | Effect |
|------|--------|
| `--build` / `-b` | Run `docker_build.sh --clean` before packaging |
| `--run` / `-r` | Start the app on the board after deploying |
| `--no-package` | Skip packaging, deploy existing `deploy/` |

### Manual deploy

```bash
scp -r board_app/deploy/ root@<board-ip>:/home/root/quickdraw
```

---

## Step 5: Run on Board

```bash
ssh root@<board-ip>
cd /home/root/quickdraw
./run.sh
```

### What run.sh does

1. Checks for root access (elevates with `su` if needed)
2. Checks that `app_quickdraw_gui` and `config.ini` exist
3. Installs runtime libraries from `lib/` to `/usr/lib64/` (first run only, runs `ldconfig`)
4. Detects Wayland socket (creates symlink for root user if needed)
5. Runs `app_quickdraw_gui --config config.ini`

### Override config

```bash
./run.sh --config /path/to/alternate.ini
```

---

## Full Pipeline

```bash
# 1. Download dataset (PC)
cd train
python download_ndjson.py --categories ../categories.txt --output ./data_128 --max-samples 8000

# 2. Train (PC with GPU)
python train.py

# 3. Calibration images (PC)
cd ..
python generate_calibration.py --per-class 5

# 4. Compile for DRP-AI (inside Docker)
cd board_app
./compile_model.sh ../qd_model.onnx ../calibration qd_mobilenetv2 1725

# 5. Build + package (from host)
./docker_build.sh

# 6. Deploy + run
./deploy.sh <board-ip> --run
```
