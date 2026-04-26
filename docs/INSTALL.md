# System Installation Guide — Renesas RZ/V2N DRP-AI TVM

Complete guide for setting up the full development environment: from a fresh Ubuntu machine to deploying AI models on the RZ/V2N board.

---

## Table of Contents

1. [Overview](#overview)
2. [Host Machine Requirements](#host-machine-requirements)
3. [Obtain Renesas Packages](#obtain-renesas-packages)
4. [Build the Docker Image](#build-the-docker-image)
5. [Verify Docker Environment](#verify-docker-environment)
6. [Host Python Environment (Training)](#host-python-environment-training)
7. [Board Setup (RZ/V2N)](#board-setup-rzv2n)
8. [End-to-End Workflow](#end-to-end-workflow)
9. [Troubleshooting](#troubleshooting)

---

## Overview

The DRP-AI TVM pipeline uses **three environments**:

| Environment | Purpose | Runs on |
|-------------|---------|---------|
| **Host (Python venv)** | Train models, export ONNX, generate calibration images | Your PC (GPU recommended) |
| **Docker container** | Cross-compile C++ for ARM64, compile ONNX → DRP-AI model (INT8 quantization) | Your PC (x86_64) |
| **RZ/V2N board** | Run inference with DRP-AI hardware accelerator | Renesas RZ/V2N EVK or SolidRun HummingBoard |

The Docker container is the core of the Renesas toolchain. It bundles:
- **RZ/V LP SDK** — Yocto cross-compiler and sysroot for ARM Cortex-A55
- **DRP-AI Translator + Quantizer** — converts ONNX to INT8 DRP-AI format
- **DRP-AI TVM (MERA2)** — TVM-based compiler targeting DRP-AI3 accelerator

---

## Host Machine Requirements

- Ubuntu 20.04 / 22.04 (or any Linux with Docker support)
- Docker Engine 20.10+
- Python 3.8+
- Git

### Install Docker (if not already installed)

```bash
# Docker Engine
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io

# Allow running Docker without sudo
sudo usermod -aG docker $USER
newgrp docker
```

---

## Obtain Renesas Packages

You need **two proprietary packages** from Renesas. These are NOT publicly downloadable — you must register and download them from the Renesas website.

### 1. RZ/V Linux Package SDK (cross-compiler)

- **Website**: https://www.renesas.com/software-tool/rzv-verified-linux-package
- **What to download**: The SDK installer shell script (`.sh` file)
  - Example filename: `rz-vlp-glibc-x86_64-core-image-weston-cortexa55-rzv2n-evk-toolchain-5.0.6.sh`
- **Version used**: 5.0.6 (for RZ/V2N)
- **Installs to**: `/opt/rz-vlp/5.0.6/` inside Docker

> **Note**: The SDK provides the GCC cross-compiler (`aarch64-poky-linux-gcc`), sysroot with ARM64 headers/libraries, and the `environment-setup-*poky-linux` script that sets all cross-compilation environment variables.

### 2. DRP-AI Translator i8 (quantizer + translator)

- **Website**: https://www.renesas.com/software-tool/drp-ai-translator-i8
- **What to download**: The Linux x86_64 installer binary
  - Example filename: `DRP-AI_Translator_i8-v1.11-Linux-x86_64-Install`
- **Version used**: v1.11
- **Installs to**: `/opt/DRP-AI_Translator_i8/` inside Docker

> **Note**: This package provides the DRP-AI Quantizer (INT8 quantization) and the DRP-AI Translator (maps operators to DRP-AI hardware). Both are required for model compilation.

### 3. DRP-AI TVM Repository (open source)

This is cloned automatically during the Docker build from GitHub:

```
https://github.com/renesas-rz/rzv_drp-ai_tvm.git
```

No manual download needed.

---

## Build the Docker Image

### 1. Clone the DRP-AI TVM repo (contains the Dockerfile)

```bash
git clone --recursive https://github.com/renesas-rz/rzv_drp-ai_tvm.git
cd rzv_drp-ai_tvm
```

### 2. Copy Renesas installers into the repo root

The Dockerfile expects these files next to it in the same directory:

```
rzv_drp-ai_tvm/
├── Dockerfile                                                              # Already in the repo
├── rz-vlp-glibc-x86_64-*-cortexa55-rzv2n-*-toolchain-5.0.6.sh            # SDK installer (you downloaded)
└── DRP-AI_Translator_i8-v1.11-Linux-x86_64-Install                        # Translator installer (you downloaded)
```

> **How it works**: The Dockerfile `COPY ./*.sh /opt` picks up the SDK installer, and `COPY ./DRP-AI_Translator*-Install /opt` picks up the Translator. It then clones a fresh copy of the same repo inside the image as `/drp-ai_tvm/`.

### 3. Build the image

The Dockerfile defaults to `PRODUCT=V2H`. For RZ/V2N, override with `--build-arg`:

```bash
docker build \
    -t drp-ai_tvm_v2n_image_$(whoami) \
    --build-arg PRODUCT=V2N \
    .
```

**Build time**: 30–60 minutes depending on internet speed and CPU.
**Image size**: ~27 GB.

### Create a persistent container

```bash
docker run -dit \
    --name drp-ai_tvm_v2n_container_$(whoami) \
    drp-ai_tvm_v2n_image_$(whoami) \
    /bin/bash
```

This creates a long-lived container that you can start/stop without losing state:

```bash
docker start  drp-ai_tvm_v2n_container_$(whoami)   # start
docker stop   drp-ai_tvm_v2n_container_$(whoami)   # stop
docker exec -it drp-ai_tvm_v2n_container_$(whoami) bash  # shell into it
```

---

## Verify Docker Environment

Shell into the container and verify all components are installed:

```bash
docker exec -it drp-ai_tvm_v2n_container_$(whoami) bash
```

### Check SDK

```bash
# IMPORTANT: unset LD_LIBRARY_PATH first (Docker image sets it, SDK refuses to work with it)
unset LD_LIBRARY_PATH

# Source SDK environment
source /opt/rz-vlp/5.0.6/environment-setup-cortexa55-poky-linux

# Verify cross-compiler
aarch64-poky-linux-gcc --version
# Expected: aarch64-poky-linux-gcc (GCC) 13.3.0

echo $SDKTARGETSYSROOT
# Expected: /opt/rz-vlp/5.0.6/sysroots/cortexa55-poky-linux

echo $OECORE_NATIVE_SYSROOT
# Expected: /opt/rz-vlp/5.0.6/sysroots/x86_64-pokysdk-linux
```

### Check DRP-AI Translator + Quantizer

```bash
ls /opt/DRP-AI_Translator_i8/
# Expected: drpAI_Quantizer/  translator/  GettingStarted/  ...

ls /opt/DRP-AI_Translator_i8/drpAI_Quantizer/
# Expected: nchw_datareader.py  ...

ls /opt/DRP-AI_Translator_i8/translator/
# Expected: DRP-AI_Translator/  UserConfig/  run_Translator_v2h.sh  run_Translator_v2n.sh  ...

ls /opt/DRP-AI_Translator_i8/translator/DRP-AI_Translator/python_api/
# python_api lives inside DRP-AI_Translator/
```

### Check DRP-AI TVM

```bash
echo $TVM_ROOT
# Expected: /drp-ai_tvm

ls $TVM_ROOT/tutorials/compile_onnx_model_quant.py
# Should exist

python3 -c "import tvm; print('TVM OK')"
# Expected: TVM OK

ls $TVM_ROOT/obj/build_runtime/v2h/lib/
# Expected: libmera2_runtime.so  libmera2_plan_io.so  libdrp_tvm_rt.so  ...
```

### Check MERA2 Python packages

```bash
pip3 list | grep -i mera
# Expected: mera2-compilation 2.5.1, mera2-runtime 2.5.1

pip3 list | grep -i tvm
# Expected: tvm 0.7.0.dev...
```

---

## Host Python Environment (Training)

The host Python environment is for training models and exporting ONNX. It does NOT need any Renesas packages.

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip

# PyTorch (GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# PyTorch (CPU only)
# pip install torch torchvision torchaudio

# ONNX export + validation
pip install onnx onnxruntime onnx-simplifier

# Training dependencies
pip install numpy scipy opencv-python pillow matplotlib
```

---

## Board Setup (RZ/V2N)

### Hardware

- Renesas RZ/V2N EVK **or** SolidRun HummingBoard with RZ/V2N SOM
- HDMI display (for Wayland/GTK3 GUI output)
- Touch screen or mouse (for drawing input)
- Ethernet or serial console for SSH access

### Software (Yocto BSP)

The board must be flashed with the Renesas RZ/V Verified Linux Package (Yocto BSP). This provides:
- Linux kernel with DRP-AI driver (`/dev/drpai0`)
- Weston (Wayland compositor)
- Basic system libraries (glibc, OpenCV, etc.)

> **Building the BSP from source is outside the scope of this guide.** Refer to the Renesas RZ/V2N Linux Start-Up Guide for flashing instructions.

### Board prerequisites

```bash
# SSH into the board
ssh root@<board-ip>

# Verify DRP-AI device exists
ls /dev/drpai0
# Must exist — if not, DRP-AI driver is not loaded

# Verify Wayland is running (needed for GTK3 GUI)
ls /run/user/*/wayland-* 2>/dev/null || ls /run/wayland-* 2>/dev/null
# Should show a wayland socket
```

### Runtime library installation (first deploy only)

When you deploy the app for the first time, `run.sh` automatically copies MERA2 runtime libraries to `/usr/lib64/`. To do this manually:

```bash
# On the board, after copying deploy/ folder:
cp deploy/lib/*.so /usr/lib64/
ldconfig
```

The required runtime libraries are:
| Library | Purpose |
|---------|---------|
| `libmera2_runtime.so` | MERA2 inference runtime |
| `libmera2_plan_io.so` | MERA2 plan loader |
| `libdrp_tvm_rt.so` | DRP-AI TVM runtime bridge |
| `libdrp_rt.so` | DRP-AI low-level runtime |
| `libarm_compute.so` | ARM Compute Library (CPU fallback ops) |
| `libarm_compute_core.so` | ARM Compute Library core |
| `libarm_compute_graph.so` | ARM Compute Library graph |
| `libacl_rt.so` | ACL runtime |

---

## End-to-End Workflow

Once everything is installed, the full pipeline looks like this:

```
┌─────────────────────────────────────────────────────┐
│  HOST (Python venv)                                 │
│                                                     │
│  1. Prepare dataset (YOLO or COCO format)           │
│  2. Train model        → best.pt                    │
│  3. Export to ONNX      → best.onnx                 │
│  4. Validate ONNX       → passes all DRP-AI checks  │
│  5. Generate calibration images                     │
└────────────────────────┬────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│  DOCKER CONTAINER                                   │
│                                                     │
│  6. Compile ONNX → DRP-AI model (INT8 quantized)    │
│     compile_onnx_model_quant.py                     │
│     Output: drpai_model/ (mera.plan, deploy.so, ..) │
│                                                     │
│  7. Cross-compile C++ app for ARM64                 │
│     docker_build.sh → build.sh → CMake + make       │
│     Output: app binary (aarch64)                    │
│                                                     │
│  8. Package: binary + model + libs → deploy/        │
└────────────────────────┬────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│  RZ/V2N BOARD                                       │
│                                                     │
│  9. scp deploy/ → board                             │
│  10. ./run.sh                                       │
│      - Installs runtime libs (first run)            │
│      - Detects Wayland                              │
│      - Launches app with DRP-AI inference           │
└─────────────────────────────────────────────────────┘
```

### Quick reference commands

```bash
# ── Step 6: Compile model (inside Docker) ──
cd board_app
./compile_model.sh ../model.onnx ../calibration model_name 200

# ── Step 7: Build C++ app (from host, uses Docker) ──
cd board_app
./docker_build.sh            # incremental build
./docker_build.sh --clean    # full rebuild

# ── Step 8+9: Package and deploy ──
cd board_app
./deploy.sh <board-ip>              # package + scp
./deploy.sh <board-ip> --run        # package + scp + launch
./deploy.sh <board-ip> --build      # rebuild + package + scp

# ── Step 10: Run on board ──
ssh root@<board-ip>
cd /home/root/quickdraw
./run.sh
```

---

## Troubleshooting

### Docker build fails

| Error | Fix |
|-------|-----|
| `*.sh: Permission denied` | SDK installer must be executable: `chmod +x *.sh` |
| `cmake-data=3.28.1-*: not found` | Kitware repo not added — check GPG key and apt source |
| `DRP-AI_Translator*-Install: not found` | Installer binary missing from build context |
| `pip install mera2_r*: no match` | TVM repo clone failed — check git clone step |

### Container runtime issues

| Error | Fix |
|-------|-----|
| `environment-setup-*poky-linux: not found` | SDK not installed — rebuild Docker image |
| `SDKTARGETSYSROOT is empty` | `unset LD_LIBRARY_PATH` before sourcing SDK — the Docker image sets it and SDK refuses to work |
| `Your environment is misconfigured, you probably need to 'unset LD_LIBRARY_PATH'` | Run `unset LD_LIBRARY_PATH` then re-source the SDK environment script |
| `aarch64-poky-linux-gcc: not found` | SDK env not sourced, or SDK install failed |
| `ModuleNotFoundError: No module named 'tvm'` | MERA2 wheels not installed — check pip install step |

### Model compilation issues

| Error | Fix |
|-------|-----|
| `dynamic shape` | Re-export ONNX with `dynamic=False`, `dynamic_axes=None` |
| `Bias not expected to be merged` | Run `onnx-simplifier` on the model first |
| Accuracy drop > 5% after INT8 | Wrong mean/std — YOLO uses `[0,0,0]/[1,1,1]`, not ImageNet values |
| `Failed to download tophub` | Network issue in Docker — set `TVM_NUM_THREADS=1` |

### Board issues

| Error | Fix |
|-------|-----|
| `/dev/drpai0: No such file` | DRP-AI driver not loaded — check BSP/kernel config |
| `libmera2_runtime.so: cannot open` | Run `ldconfig` after copying libs to `/usr/lib64/` |
| Display noise/artifacts | DRP-AI + Display DDR bandwidth conflict — apply TF-A QoS patches |
| `WAYLAND_DISPLAY not set` | Weston not running — start it or use `--console` mode |

---

## Directory Reference: What's Inside the Docker Image

```
/opt/
├── rz-vlp/5.0.6/                          RZ/V LP SDK
│   ├── sysroots/
│   │   ├── cortexa55-poky-linux/           ARM64 sysroot (headers + libs)
│   │   └── x86_64-pokysdk-linux/           Host tools (cross-compiler)
│   │       └── usr/bin/aarch64-poky-linux/
│   │           ├── aarch64-poky-linux-gcc
│   │           └── aarch64-poky-linux-g++
│   └── environment-setup-cortexa55-poky-linux   Source this!
│
├── DRP-AI_Translator_i8/
│   ├── translator/
│   │   ├── DRP-AI_Translator/              Translator engine
│   │   │   └── python_api/                 Python bindings
│   │   ├── UserConfig/
│   │   ├── run_Translator_v2h.sh
│   │   └── run_Translator_v2n.sh
│   ├── drpAI_Quantizer/                    INT8 quantizer
│   │   ├── nchw_datareader.py              Calibration data reader (NCHW)
│   │   └── nhwc_datareader.py              Calibration data reader (NHWC)
│   ├── onnx_models/                        Sample ONNX models
│   └── GettingStarted/                     Renesas documentation
│
/drp-ai_tvm/                               DRP-AI TVM root ($TVM_ROOT)
├── tutorials/
│   └── compile_onnx_model_quant.py         Main compilation script
├── tvm/
│   ├── include/tvm/runtime/                TVM + custom runtime headers
│   └── python/                             TVM Python package
├── obj/
│   ├── build_runtime/v2h/lib/              MERA2 runtime .so files
│   │   ├── libmera2_runtime.so
│   │   ├── libmera2_plan_io.so
│   │   └── libdrp_tvm_rt.so
│   └── pip_package/                        Pre-built Python wheels
├── setup/include/                          Custom runtime headers
├── 3rdparty/
│   ├── spdlog/                             Logging library
│   ├── asio/                               Async I/O library
│   ├── dlpack/                             DLPack tensor standard
│   └── dmlc-core/                          DMLC utilities
└── apps/                                   App integration headers
```

---

## Environment Variables Quick Reference

### Inside Docker (model compilation)

```bash
# Auto-detected by scripts:
export SDK=/opt/rz-vlp/5.0.6/                             # NOT /sysroots — compile script appends it
export TRANSLATOR=/opt/DRP-AI_Translator_i8/translator/    # trailing slash required
export QUANTIZER=/opt/DRP-AI_Translator_i8/drpAI_Quantizer/
export TVM_ROOT=/drp-ai_tvm
export PRODUCT=V2N
export PYTHONPATH=$TVM_ROOT/tvm/python:$QUANTIZER:${PYTHONPATH}
```

### Inside Docker (C++ cross-compilation)

```bash
# IMPORTANT: unset LD_LIBRARY_PATH — SDK refuses to work with it set
unset LD_LIBRARY_PATH

# Source SDK (sets CC, CXX, SDKTARGETSYSROOT, OECORE_NATIVE_SYSROOT, etc.)
source /opt/rz-vlp/5.0.6/environment-setup-cortexa55-poky-linux
export TVM_ROOT=/drp-ai_tvm
export PRODUCT=V2N
```

### On Board (runtime)

```bash
export LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH
export XDG_RUNTIME_DIR=/run/user/0        # Wayland
export WAYLAND_DISPLAY=wayland-0           # Wayland socket
```
