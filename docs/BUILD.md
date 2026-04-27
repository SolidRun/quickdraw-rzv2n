# Build and Deploy

End-to-end guide: from a trained ONNX model to a running app on the RZ/V2N board.

This guide is verified against a working install. See [INSTALL.md](INSTALL.md) for the one-time toolchain setup.

---

## Prerequisites

Before you start, you need:

| Requirement | How to verify |
|---|---|
| DRP-AI TVM Docker container running | `docker ps \| grep drp-ai_tvm` |
| Trained ONNX model: `qd_model.onnx` | `ls qd_model.onnx` in project root |
| Calibration images: `calibration/*.png` (1725 files) | `ls calibration/*.png \| wc -l` |
| Renesas RZ/V2N board reachable over SSH | `ssh root@<board-ip> ls /dev/drpai0` |

If any of these are missing, see [INSTALL.md](INSTALL.md) (toolchain) or [TRAINING.md](TRAINING.md) (model + calibration).

---

## Pipeline Overview

```
┌──────────────────────────────────────────────────┐
│ Step 1: Compile model     → drpai_model/         │  (inside Docker)
│ Step 2: Cross-compile C++ → board_app/build/     │  (host → Docker)
│ Step 3: Package           → board_app/deploy/    │  (host, automatic)
│ Step 4: Deploy to board   → /home/root/quickdraw │  (scp)
│ Step 5: Run on board      → DRP-AI inference     │  (ssh)
└──────────────────────────────────────────────────┘
```

---

## Step 1 — Compile the Model for DRP-AI

Convert the ONNX model into the INT8 quantized DRP-AI runtime format.

**Where this runs:** files are pushed into the container with `docker cp`, the script runs via `docker exec`, then results are pulled back to the host with `docker cp`.

### 1a. Push the model + calibration into the container

From your host shell, in the project root:

```bash
CONTAINER=drp-ai_tvm_v2n_container_$(whoami)

# Push ONNX model and calibration images into a working dir inside the container
docker exec $CONTAINER mkdir -p /quickdraw
docker cp qd_model.onnx       $CONTAINER:/quickdraw/qd_model.onnx
docker cp calibration         $CONTAINER:/quickdraw/calibration
docker cp board_app/compile_model.sh $CONTAINER:/quickdraw/compile_model.sh
```

### 1b. Run the compile script inside the container

```bash
docker exec -it $CONTAINER bash -c '
    cd /quickdraw
    chmod +x compile_model.sh
    ./compile_model.sh /quickdraw/qd_model.onnx /quickdraw/calibration qd_mobilenetv2 1725
'
```

**Arguments** (in order):
1. `qd_model.onnx` — the FP32 ONNX model from training
2. `calibration/` — directory of representative PNGs for INT8 quantization (1725 images = 5 per class × 345 classes)
3. `qd_mobilenetv2` — output directory name (under `/drp-ai_tvm/tutorials/` inside container)
4. `1725` — number of calibration images to use

### 1c. Pull the compiled model back to the host

```bash
# Pull from container into a temp name first so we can verify before replacing
docker cp $CONTAINER:/drp-ai_tvm/tutorials/qd_mobilenetv2 ./drpai_model_new

# Verify it's the proper DRP-AI compile (~12 MB), not the CPU fallback (~27 MB)
ls -lh drpai_model_new/sub_0000__CPU_DRP_TVM/deploy.so

# If size is correct (~12 MB), replace the previous drpai_model
sudo rm -rf drpai_model            # sudo because docker cp pulled files as root
mv drpai_model_new drpai_model

# Optional: chown to your user so future operations don't need sudo
sudo chown -R $(id -u):$(id -g) drpai_model
```

> **Note on `docker cp` ownership**: files pulled out of the container come back owned by `root` (because the container's processes run as root). The `sudo rm` and `sudo chown` above are needed once. Afterwards your user can read/write freely.

**What the script does:**
1. Patches `compile_onnx_model_quant.py` for sketch-specific preprocessing (see table below)
2. Runs INT8 quantization with Percentile 99.99 method on calibration images
3. Compiles via DRP-AI Translator + MERA2 backend
4. Outputs `deploy.so`, `deploy.json`, `deploy.params`, `mera.plan`, and a `preprocess/` directory

**Patches the script applies** (sketch model defaults differ from the Renesas YOLO/ImageNet defaults):

| Patch | Original (ImageNet) | Patched (Sketch) |
|-------|---------------------|------------------|
| Normalization mean | `[0.485, 0.456, 0.406]` | `[0.0, 0.0, 0.0]` |
| Normalization stdev | `[0.229, 0.224, 0.225]` | `[1.0, 1.0, 1.0]` |
| Calibration resize | `resize(256) + center_crop(224)` | `resize(128)` |
| Preprocess shape | `[1, 480, 640, 3]` | `[1, 128, 128, 3]` |
| PRODUCT check | `V2H` only | `V2H` or `V2N` |

**Expected output** (under `/drp-ai_tvm/tutorials/qd_mobilenetv2/` inside container):

```
qd_mobilenetv2/
├── sub_0000__CPU_DRP_TVM/
│   ├── deploy.so          ~12 MB    INT8 compiled DRP-AI model
│   ├── deploy.json        ~2.5 KB   Model metadata
│   └── deploy.params      ~3 KB     Quantized parameters
├── preprocess/            (10 files) DRP-AI preprocessing config
└── mera.plan              ~1.4 KB
```

### ⚠️ Watch out for silent CPU fallback

When this guide was written, running the compile produced these errors in the log:

```
KeyError: 'zero_in'
[ERROR A004] Failed at parse operation
TVMError: drp quantizer translator toolchain failed
```

Despite the errors, the script printed `COMPILATION SUCCESSFUL` and exited 0 — but the produced `deploy.so` was **~27 MB instead of ~12 MB**, and the `preprocess/` directory was empty. That output is a CPU-only fallback that will run but won't use the DRP-AI3 accelerator.

The root cause is not yet confirmed (it could be a translator/quantizer version mismatch, a Python dependency issue, or something specific to this model). What is verified:
- The errors above appear in the compile log when the failure occurs
- `deploy.so` size and the contents of `preprocess/` are reliable indicators
- An older successfully-compiled `drpai_model/` (12 MB, with full `preprocess/`) is included in this repo for reference

**Always verify after compile**. There are two strong signals:

```bash
# 1. deploy.so size — the most reliable single check
ls -lh drpai_model/sub_0000__CPU_DRP_TVM/deploy.so
# ~12 MB → DRP-AI accelerated (correct)
# ~27 MB → CPU fallback (translator failed silently — see log for KeyError)

# 2. preprocess/ directory — must contain 10 hardware-config files
ls drpai_model/preprocess/
# Expected (DRP-AI accelerated):
#   addr_map.txt  aimac_cmd.bin  aimac_desc.bin  aimac_param_cmd.bin
#   aimac_param_desc.bin  drp_config.mem  drp_desc.bin  drp_param.bin
#   drp_param_info.txt  weight.bin
# Empty or missing → CPU fallback (the model literally won't run on DRP-AI3 hardware)
```

If `preprocess/` is missing, the C++ app on the board cannot load the DRP-AI hardware preprocessing pipeline — only `deploy.so` would run, on the CPU.

If you got the CPU fallback, check the compile log for `KeyError: 'zero_in'`. The root cause and a clean fix are not yet established. As a workaround, the existing 12 MB `drpai_model/` in the repo (compiled earlier when the toolchain was working) is what's used by `package.sh` in Step 3 — so the rest of the build pipeline still produces a working app. Investigation directions worth trying if you need to recompile fresh: check if Renesas has updated the Translator/Quantizer, try different quantizer options (e.g., asymmetric mode `-az`), or compare the Python package versions against a known-working environment.

### Quantization settings

| Setting | Value |
|---------|-------|
| Method | Percentile 99.99 |
| CPU operations | float32 |
| Output format | MERA2 |
| Calibration data | 1725 PNG images, 50% normal resize / 50% board-style crop+pad |

---

## Step 2 — Cross-compile the C++ Application

Build the ARM64 binaries for the board, using the cross-compiler inside the Docker container.

**Where this runs:** host shell. The script orchestrates Docker for you.

```bash
cd board_app
./docker_build.sh           # Incremental build
./docker_build.sh --clean   # Clean rebuild (use after editing CMakeLists or toolchain)
```

**What `docker_build.sh` does:**
1. Auto-detects the running DRP-AI TVM container (or starts a stopped one with confirmation)
2. Copies these source items into `/tmp/board_app/` inside the container: `src/`, `toolchain/`, `CMakeLists.txt`, `build.sh`, `config.ini`, `config.json`, `labels.txt`
3. Inside container: sources the SDK (`/opt/rz-vlp/5.0.6/environment-setup-cortexa55-poky-linux`), runs CMake + `make -j$(nproc)`
4. Copies the compiled binaries back to host `board_app/build/`
5. Auto-runs `package.sh` to create `board_app/deploy/`

**Build configuration** (set in `board_app/CMakeLists.txt`):

| Setting | Value |
|---------|-------|
| C++ standard | 17 |
| Compiler flags | `-O3 -mtune=cortex-a55 -Wall` |
| Compile definitions | `V2H`, `V2N`, `KDLDRPAI` |
| Toolchain | `toolchain/runtime.cmake` |

**Targets built:**

| Binary | Size | Purpose |
|--------|------|---------|
| `app_quickdraw_gui` | ~13 MB | GTK3 GUI + DRP-AI inference (the main app) |
| `app_quickdraw` | ~11 MB | Socket server (legacy, for Python GUI) |

**Verify the binary is ARM64** (not the host's x86_64):

```bash
file board_app/build/app_quickdraw_gui
# Expected: ELF 64-bit LSB pie executable, ARM aarch64, ...
```

**Source files** (in `board_app/src/`):

| File | Purpose |
|------|---------|
| `main_gui.cpp` | Entry point: parses `config.ini`, loads DRP-AI model, launches GUI |
| `gui.cpp` / `gui.h` | GTK3 fullscreen canvas, touch+mouse input, predictions panel, AI commentary |
| `drpai_inference.cpp` / `.h` | MERA2 runtime wrapper: load model, set frequencies, run inference |
| `classification.cpp` / `.h` | Softmax (with overflow guard) + top-K extraction |
| `preprocessing.cpp` / `.h` | Ink detection, crop, pad, invert, area-resize, normalize |
| `define.h` | Default constants |
| `main.cpp` | Socket server entry point (legacy) |

---

## Step 3 — Package for Deployment

Runs **automatically** at the end of `docker_build.sh`. To run manually:

```bash
cd board_app
./package.sh
```

The script auto-detects the compiled model from these locations (in order):
1. `/drp-ai_tvm/tutorials/qd_mobilenetv2/` (inside container — Step 1 output)
2. `../drpai_model/` (host project root)
3. `./drpai_model/` (board_app local)

It creates `board_app/deploy/` (~46 MB total):

```
deploy/
├── app_quickdraw_gui                       13 MB   ARM64 binary
├── config.ini                                       DRP-AI config (frequencies, model path)
├── config.json                                      UI config (colors, comments, layout)
├── labels.txt                                       345 class names
├── run.sh                                           Board startup script
├── solidrun_logo.png                                Title bar logo
├── model/qd_mobilenetv2/                            Compiled DRP-AI model
│   ├── sub_0000__CPU_DRP_TVM/
│   │   ├── deploy.so           ~12 MB
│   │   ├── deploy.json
│   │   └── deploy.params
│   ├── preprocess/             (10 files)
│   └── mera.plan
└── lib/                                             MERA2 runtime libraries
    ├── libmera2_runtime.so
    ├── libmera2_plan_io.so
    ├── libdrp_tvm_rt.so
    ├── libdrp_rt.so
    ├── libarm_compute.so
    ├── libarm_compute_core.so
    ├── libarm_compute_graph.so
    └── libacl_rt.so
```

**Custom paths** (if your model isn't in the default location):

```bash
./package.sh --model-dir <name>             # Default: qd_mobilenetv2
./package.sh --compiled /path/to/compiled   # Override compiled model path
./package.sh --output /custom/deploy        # Override output dir
./package.sh --binary  /custom/binary       # Override binary path
```

---

## Step 4 — Deploy to the Board

```bash
cd board_app
./deploy.sh <board-ip>
```

**Flags:**

| Flag | Effect |
|------|--------|
| `--build` / `-b` | Run `docker_build.sh --clean` before packaging |
| `--run` / `-r` | Start the app on the board after deploying |
| `--no-package` | Skip packaging, deploy existing `deploy/` |

**Examples:**

```bash
./deploy.sh 192.168.1.100                 # Package + scp
./deploy.sh 192.168.1.100 --run           # Package + scp + launch app
./deploy.sh 192.168.1.100 --build --run   # Full: rebuild + package + scp + launch
```

**Manual deploy** (if you prefer not to use the script):

```bash
scp -r board_app/deploy/ root@<board-ip>:/home/root/quickdraw
```

The board destination is `/home/root/quickdraw`.

---

## Step 5 — Run on the Board

```bash
ssh root@<board-ip>
cd /home/root/quickdraw
./run.sh
```

**What `run.sh` does:**
1. Checks for root access (DRP-AI device `/dev/drpai0` requires root). Auto-elevates with `su` if needed
2. Verifies `app_quickdraw_gui` and `config.ini` exist
3. **First run only**: copies MERA2 runtime libraries from `lib/` to `/usr/lib64/` and runs `ldconfig`
4. Auto-detects the Wayland compositor socket (creates a symlink for the root user if needed)
5. Sets `LD_LIBRARY_PATH=/usr/lib64`, `XDG_RUNTIME_DIR`, `WAYLAND_DISPLAY`
6. Launches `app_quickdraw_gui --config config.ini`

**Override the config file:**

```bash
./run.sh --config /path/to/alternate.ini
```

**Stop the app:** close the window, press `Esc`, or `Ctrl+C` from the terminal.

---

## Full Pipeline (typical workflow)

```bash
# ── On host PC ──

# 1. Train model + export ONNX (see TRAINING.md)
cd train
python download_ndjson.py --categories ../categories.txt --output ./data_128 --max-samples 8000
python train.py

# 2. Generate calibration images
cd ..
python generate_calibration.py --per-class 5

# ── Compile model: push files in, run, pull back (host shell) ──

# 3. Compile model for DRP-AI
CONTAINER=drp-ai_tvm_v2n_container_$(whoami)
docker exec $CONTAINER mkdir -p /quickdraw
docker cp qd_model.onnx $CONTAINER:/quickdraw/qd_model.onnx
docker cp calibration   $CONTAINER:/quickdraw/calibration
docker cp board_app/compile_model.sh $CONTAINER:/quickdraw/compile_model.sh
docker exec $CONTAINER bash -c 'cd /quickdraw && chmod +x compile_model.sh && ./compile_model.sh /quickdraw/qd_model.onnx /quickdraw/calibration qd_mobilenetv2 1725'
docker cp $CONTAINER:/drp-ai_tvm/tutorials/qd_mobilenetv2 ./drpai_model_new
ls -lh drpai_model_new/sub_0000__CPU_DRP_TVM/deploy.so   # verify ~12 MB
sudo rm -rf drpai_model && mv drpai_model_new drpai_model
sudo chown -R $(id -u):$(id -g) drpai_model

# 4. Build C++ app + package (automatic) + deploy + run
cd board_app
./deploy.sh <board-ip> --build --run
```

---

## Troubleshooting

| Symptom | Cause / Fix |
|---------|-------------|
| `compile_model.sh` reports "COMPILATION SUCCESSFUL" but `deploy.so` is ~27 MB | DRP-AI Translator silently fell back to CPU. Check log for `KeyError: 'zero_in'`. See "Known issue" in Step 1. |
| `compile_model.sh: ONNX model not found` | You forgot Step 1a — push files into the container with `docker cp` first |
| `rm: cannot remove '...': Permission denied` after Step 1c | `docker cp` pulls files as root. Use `sudo rm -rf drpai_model` then `sudo chown -R $(id -u):$(id -g) drpai_model` |
| `docker_build.sh: No DRP-AI TVM container found` | Container is not running: `docker start drp-ai_tvm_v2n_container_<user>` |
| `package.sh: Compiled model not found` | Run Step 1 first, or copy model output to `<project>/drpai_model/` |
| Build error: `aarch64-poky-linux-gcc: not found` | SDK env not sourced — `docker_build.sh` handles this; if running `build.sh` manually, `unset LD_LIBRARY_PATH` then `source /opt/rz-vlp/5.0.6/environment-setup-cortexa55-poky-linux` |
| Board: `Cannot open /dev/drpai0` | Run as root. `run.sh` auto-elevates with `su` |
| Board: `libmera2_runtime.so not found` | First run installs them automatically. Manual fallback: `cp lib/*.so* /usr/lib64/ && ldconfig` |
| Board: app shows but predictions are random | INT8 quantization mismatch — verify `deploy.so` is ~12 MB (DRP-AI), not 27 MB (CPU fallback) |
