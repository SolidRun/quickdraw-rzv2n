# System Installation Guide — Renesas RZ/V2N DRP-AI TVM

End-to-end setup: from a fresh Ubuntu machine to a Docker container ready to compile models for the RZ/V2N board.

This guide is verified against a working install. Every command, filename, and version is what you will actually see on disk.

---

## Table of Contents

1. [Overview](#overview)
2. [Step 1 — Host Machine Requirements](#step-1--host-machine-requirements)
3. [Step 2 — Download the Renesas Packages](#step-2--download-the-renesas-packages)
4. [Step 3 — Extract the Two Installer Files](#step-3--extract-the-two-installer-files)
5. [Step 4 — Clone the DRP-AI TVM Repository](#step-4--clone-the-drp-ai-tvm-repository)
6. [Step 5 — Stage the Installers Next to the Dockerfile](#step-5--stage-the-installers-next-to-the-dockerfile)
7. [Step 6 — Build the Docker Image](#step-6--build-the-docker-image)
8. [Step 7 — Create a Persistent Container](#step-7--create-a-persistent-container)
9. [Step 8 — Verify the Toolchain Works](#step-8--verify-the-toolchain-works)
10. [Step 9 — Host Python Environment for Training](#step-9--host-python-environment-for-training)
11. [Step 10 — RZ/V2N Board Setup](#step-10--rzv2n-board-setup)
12. [Environment Variables Reference](#environment-variables-reference)
13. [Directory Reference](#directory-reference)
14. [Troubleshooting](#troubleshooting)

---

## Overview

The DRP-AI TVM pipeline uses **three environments**:

| Environment | Purpose | Runs on |
|-------------|---------|---------|
| **Host (Python venv)** | Train models, export ONNX, generate calibration images | Your PC |
| **Docker container** | Cross-compile C++ for ARM64, compile ONNX → DRP-AI model (INT8 quantization) | Your PC (x86_64) |
| **RZ/V2N board** | Run inference with the DRP-AI3 hardware accelerator | Renesas RZ/V2N EVK or SolidRun HummingBoard |

The Docker container bundles the entire Renesas toolchain:
- **RZ/V LP SDK 5.0.6** — Yocto cross-compiler and ARM Cortex-A55 sysroot
- **DRP-AI Translator i8 v1.11** — converts ONNX to INT8 DRP-AI format (with the Quantizer)
- **DRP-AI TVM v2.7+ (MERA2)** — TVM-based compiler targeting the DRP-AI3 accelerator

Once the Docker image is built, you reuse it for every project (Quick Draw, YOLO, etc.).

---

## Step 1 — Host Machine Requirements

- Ubuntu 20.04 / 22.04 (or any Linux with Docker support)
- Docker Engine 20.10+
- Python 3.8+
- Git
- `unzip` utility

### Install Docker (skip if already installed)

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg unzip
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io

# Allow running Docker without sudo
sudo usermod -aG docker $USER
newgrp docker
```

Quick check: `docker --version` should print Docker version info.

---

## Step 2 — Download the Renesas Packages

You need **two ZIP archives** from Renesas. Both require a free [MyRenesas](https://www.renesas.com/myrenesas) account.

### Package A — RZ/V2N AI SDK (~4.0 GB)

Contains the cross-compiler SDK plus board flashing images and PDFs.

- **Website**: https://www.renesas.com/software-tool/rzv2n-ai-software-development-kit
- **Filename you will download**: `RTK0EF0189F06000SJ.zip`
- **Save to**: `~/Downloads/` (any location works, you'll reference it later)

### Package B — DRP-AI Translator i8 v1.11 (~258 MB)

The latest standalone Translator + Quantizer.

- **Website**: https://www.renesas.com/software-tool/drp-ai-translator-i8
- **Filename you will download**: `r20ut5460ej0111-drp-ai-translator-i8.zip`
- **Save to**: `~/Downloads/`

### Confirm both downloads completed successfully

```bash
ls -lh ~/Downloads/RTK0EF0189F06000SJ.zip \
       ~/Downloads/r20ut5460ej0111-drp-ai-translator-i8.zip
```

Expected output (sizes must be ~GB and ~MB respectively — a tiny file means a broken download):

```
-rw-rw-r-- 1 user user 4.0G  ... RTK0EF0189F06000SJ.zip
-rw-rw-r-- 1 user user 247M  ... r20ut5460ej0111-drp-ai-translator-i8.zip
```

> **Why don't we use the Translator inside the AI SDK ZIP?** The AI SDK ZIP contains an older Translator (`v1.04`). We use the standalone v1.11 download instead.

---

## Step 3 — Extract the Two Installer Files

Each ZIP contains many files. We only need ONE installer file from each.

### Create a working directory

```bash
mkdir -p ~/renesas/tvm_work
cd ~/renesas/tvm_work
```

### Extract the SDK installer from Package A

The SDK installer is at `ai_sdk_setup/rz-vlp-glibc-x86_64-core-image-weston-cortexa55-rzv2n-evk-toolchain-5.0.6.sh` inside the ZIP. We extract just that file (using `-j` to flatten the directory structure):

```bash
unzip -j ~/Downloads/RTK0EF0189F06000SJ.zip \
    "ai_sdk_setup/rz-vlp-glibc-x86_64-core-image-weston-cortexa55-rzv2n-evk-toolchain-5.0.6.sh" \
    -d ~/renesas/tvm_work/
```

### Extract the Translator installer from Package B

```bash
unzip -j ~/Downloads/r20ut5460ej0111-drp-ai-translator-i8.zip \
    "DRP-AI_Translator_i8-v1.11-Linux-x86_64-Install" \
    -d ~/renesas/tvm_work/
```

### Verify both files are present

```bash
ls -lh ~/renesas/tvm_work/
```

Expected output:

```
-rw-rw-r-- 1 user user 1.9G  ... rz-vlp-glibc-x86_64-core-image-weston-cortexa55-rzv2n-evk-toolchain-5.0.6.sh
-rwxrwxrwx 1 user user 243M  ... DRP-AI_Translator_i8-v1.11-Linux-x86_64-Install
```

If sizes are wrong, the extraction failed — re-download the ZIP and retry.

> **Tip**: If you prefer to extract the entire ZIP, omit the `-j` flag and the specific filename, e.g., `unzip ~/Downloads/RTK0EF0189F06000SJ.zip -d sdk_full/` — but you only need the two installer files.

---

## Step 4 — Clone the DRP-AI TVM Repository

The Dockerfile we'll use is part of Renesas's open-source DRP-AI TVM repo on GitHub. Clone it (with submodules):

```bash
cd ~/renesas
git clone --recursive https://github.com/renesas-rz/rzv_drp-ai_tvm.git
```

> **Note**: This is a large clone (~4 GB with submodules). Be patient.

Verify the Dockerfile is present:

```bash
ls -la ~/renesas/rzv_drp-ai_tvm/Dockerfile
# Should show a ~3 KB file
```

---

## Step 5 — Stage the Installers Next to the Dockerfile

The Dockerfile contains these `COPY` instructions:

```dockerfile
COPY ./*.sh /opt                                  # picks up the SDK installer
COPY ./DRP-AI_Translator*-Linux*-x86_64-Install /opt    # picks up the Translator installer
```

So both installers must be next to the `Dockerfile`. Move them in:

```bash
mv ~/renesas/tvm_work/rz-vlp-glibc-x86_64-core-image-weston-cortexa55-rzv2n-evk-toolchain-5.0.6.sh \
   ~/renesas/rzv_drp-ai_tvm/

mv ~/renesas/tvm_work/DRP-AI_Translator_i8-v1.11-Linux-x86_64-Install \
   ~/renesas/rzv_drp-ai_tvm/
```

Verify the staging:

```bash
ls -lh ~/renesas/rzv_drp-ai_tvm/{Dockerfile,*.sh,DRP-AI_Translator*-Install}
```

Expected:

```
-rw-rw-r-- 1 user user 3.0K   Dockerfile
-rwxrwxrwx 1 user user 243M   DRP-AI_Translator_i8-v1.11-Linux-x86_64-Install
-rw-rw-r-- 1 user user 1.9G   rz-vlp-glibc-x86_64-core-image-weston-cortexa55-rzv2n-evk-toolchain-5.0.6.sh
```

---

## Step 6 — Build the Docker Image

The Dockerfile defaults to `PRODUCT=V2H`. For RZ/V2N, override it:

```bash
cd ~/renesas/rzv_drp-ai_tvm

docker build \
    -t drp-ai_tvm_v2n_image_$(whoami) \
    --build-arg PRODUCT=V2N \
    .
```

| Property | Value |
|----------|-------|
| **Build time** | 30–60 minutes (depends on internet speed and CPU) |
| **Final image size** | ~27 GB |
| **Image tag** | `drp-ai_tvm_v2n_image_<your_username>` |

The build runs 50 steps, in order:
1. **Steps 1–15**: Ubuntu 22.04 base + GCC 13, CMake 3.28.1, Python 3.10, LLVM 14, build tools
2. **Steps 16–19**: COPY and run the SDK installer (auto-accepts EULA via `yes ""`) — installs to `/opt/rz-vlp/5.0.6/`
3. **Step 20**: Symlink `aarch64-poky-linux` → `cortexa55-poky-linux` sysroot
4. **Steps 21–24**: COPY and run the Translator installer (auto-accepts via `yes`) — installs to `/opt/DRP-AI_Translator_i8/`
5. **Steps 25–28**: Install Python deps (psutil, numpy 1.26.4, cython 3.0.11, decorator, attrs, TensorFlow 2.18.1, tflite, tqdm)
6. **Step 30**: `git clone --recursive` the DRP-AI TVM repo into `/drp-ai_tvm/` (~5 min, ~4 GB)
7. **Steps 31–35**: Set env vars (`TVM_ROOT`, `PYTHONPATH`, `LD_LIBRARY_PATH`, `LIBRARY_PATH`)
8. **Steps 37–41**: Purge `python3-yaml`, upgrade pip, install MERA2 wheels via `find ... -name "*.whl" -exec pip3 install`
9. **Steps 43–45**: More env vars (`TRANSLATOR`, `QUANTIZER`, `PRODUCT=V2N`)
10. **Steps 46–48**: Clone spdlog and asio into `/drp-ai_tvm/3rdparty/`
11. **Steps 49–50**: Copy custom TVM runtime headers; set final WORKDIR to `/drp-ai_tvm/tutorials`

> **Note**: Some Dockerfile lines like `RUN find . -name "mera2_r*.whl" -exec pip3 install {} \;` may show as `RUN pip3 install mera2_r*` in older clones. Both work — the newer `find` syntax is more robust. Don't worry if the Dockerfile in your clone differs slightly.

Verify the image exists:

```bash
docker images | grep drp-ai_tvm_v2n
# drp-ai_tvm_v2n_image_<user>   latest   <id>   <time>   ~27GB
```

---

## Step 7 — Create a Persistent Container

A persistent container keeps state across stop/start cycles (useful so you don't re-run setup each time).

```bash
docker run -dit \
    --name drp-ai_tvm_v2n_container_$(whoami) \
    drp-ai_tvm_v2n_image_$(whoami) \
    /bin/bash
```

> Files are exchanged with the container via `docker cp` (used by `docker_build.sh` and the build flow in [BUILD.md](BUILD.md)). No volume mount needed.

Lifecycle commands:

```bash
docker start  drp-ai_tvm_v2n_container_$(whoami)            # start (after a host reboot)
docker stop   drp-ai_tvm_v2n_container_$(whoami)            # stop
docker exec -it drp-ai_tvm_v2n_container_$(whoami) bash     # open a shell
```

> **Note**: The project's `board_app/docker_build.sh` auto-detects this container by name pattern, so the standard naming above is recommended.

---

## Step 8 — Verify the Toolchain Works

Open a shell into the container:

```bash
docker exec -it drp-ai_tvm_v2n_container_$(whoami) bash
```

Then run these checks. **Do NOT source the SDK environment yet** — TVM needs the default Python paths.

### 8.1 Check the DRP-AI TVM Python stack

```bash
python3 -c "import tvm; print('TVM:', tvm.__version__)"
# Expected: TVM: 0.11.1

pip3 list | grep -iE "mera|tvm "
# Expected (note: pip reports a different version string than tvm.__version__ — both are correct):
#   mera2-compilation     2.5.1
#   mera2-runtime         2.5.1
#   tvm                   0.7.0.dev1599+g2af1556b1

ls /drp-ai_tvm/tutorials/compile_onnx_model_quant.py
# Expected: file exists
```

### 8.2 Check the DRP-AI Translator + Quantizer

```bash
ls /opt/DRP-AI_Translator_i8/
# Expected: drpAI_Quantizer/  translator/  GettingStarted/  onnx_models/

ls /opt/DRP-AI_Translator_i8/drpAI_Quantizer/nchw_datareader.py
# Expected: file exists

ls /opt/DRP-AI_Translator_i8/translator/DRP-AI_Translator/python_api/
# Expected: directory exists with Python modules
```

### 8.3 Check the C++ cross-compiler (separate shell — sourcing SDK changes Python)

In a new shell or after the Python checks:

```bash
# IMPORTANT: must unset LD_LIBRARY_PATH first, or the SDK script refuses to set up
unset LD_LIBRARY_PATH

# Source the SDK environment (sets CC, CXX, sysroot, etc.)
source /opt/rz-vlp/5.0.6/environment-setup-cortexa55-poky-linux

# Verify
aarch64-poky-linux-gcc --version
# Expected: aarch64-poky-linux-gcc (GCC) 13.3.0

echo $SDKTARGETSYSROOT
# Expected: /opt/rz-vlp/5.0.6/sysroots/cortexa55-poky-linux

echo $OECORE_NATIVE_SYSROOT
# Expected: /opt/rz-vlp/5.0.6/sysroots/x86_64-pokysdk-linux
```

> **Important — Two-mode environment**: Sourcing the SDK environment overrides Python paths and breaks `import tvm`. So:
> - **Model compilation (Python)** → DON'T source the SDK
> - **C++ cross-compilation (CMake/make)** → DO source the SDK
> The project's `compile_model.sh` and `build.sh` handle this correctly.

If all checks pass, your toolchain is ready.

---

## Step 9 — Host Python Environment for Training

The host Python environment is for training models and exporting ONNX. It does NOT need any Renesas packages — it runs on your normal PC.

```bash
cd /path/to/your/project   # e.g., quickdraw/
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip

# PyTorch (with CUDA if you have an NVIDIA GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# or CPU-only:
# pip install torch torchvision torchaudio

# ONNX export + validation
pip install onnx onnxruntime onnx-simplifier

# Project dependencies (Quick Draw specifics)
pip install numpy scipy opencv-python pillow matplotlib
```

---

## Step 10 — RZ/V2N Board Setup

### Hardware

- Renesas RZ/V2N EVK **or** SolidRun HummingBoard with RZ/V2N SOM
- HDMI display (for Wayland/GTK3 GUI output)
- Touch screen or mouse (for drawing input)
- Ethernet or serial console for SSH access

### Software (Yocto BSP)

The board must be flashed with the Renesas RZ/V Verified Linux Package (Yocto BSP). The BSP provides:
- Linux kernel with DRP-AI driver (`/dev/drpai0`)
- Weston (Wayland compositor)
- Basic system libraries (glibc, OpenCV, etc.)

> Flashing the BSP is outside the scope of this guide. Refer to the **r11an0872ej0600-rzv2n-ai-sdk.pdf** included in the AI SDK ZIP (Package A above) for flashing instructions, or use the bootable images at `board_setup/{eSD,xSPI}.zip` from inside the same ZIP.

### Verify the board is ready

```bash
ssh root@<board-ip>

# DRP-AI driver loaded?
ls /dev/drpai0
# Expected: /dev/drpai0

# Wayland compositor running?
ls /run/user/*/wayland-* 2>/dev/null || ls /run/wayland-* 2>/dev/null
# Expected: a wayland socket
```

### Runtime libraries (auto-installed on first deploy)

When you deploy the app, `run.sh` automatically copies the MERA2 runtime libraries to `/usr/lib64/`. If you ever need to install them manually:

```bash
cp deploy/lib/*.so /usr/lib64/
ldconfig
```

The libraries deployed are:

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

## Environment Variables Reference

### Inside Docker — model compilation (Python, no SDK sourced)

These are set by the Dockerfile and used by `compile_onnx_model_quant.py`:

```bash
TVM_ROOT=/drp-ai_tvm
QUANTIZER=/opt/DRP-AI_Translator_i8/drpAI_Quantizer/
PYTHONPATH=/opt/DRP-AI_Translator_i8/drpAI_Quantizer/
PRODUCT=V2N
```

The compile script also expects:

```bash
SDK=/opt/rz-vlp/5.0.6/                                    # NOT /sysroots — script appends it
TRANSLATOR=/opt/DRP-AI_Translator_i8/translator/          # trailing slash required
```

### Inside Docker — C++ cross-compilation (SDK sourced)

```bash
unset LD_LIBRARY_PATH                                     # MUST do this first
source /opt/rz-vlp/5.0.6/environment-setup-cortexa55-poky-linux
# Sets: CC, CXX, SDKTARGETSYSROOT, OECORE_NATIVE_SYSROOT, CFLAGS, LDFLAGS, etc.

export TVM_ROOT=/drp-ai_tvm
export PRODUCT=V2N
```

### On the Board — runtime

```bash
export LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH
export XDG_RUNTIME_DIR=/run/user/0       # Wayland
export WAYLAND_DISPLAY=wayland-0          # Wayland socket
```

(`run.sh` on the board sets these automatically.)

---

## Directory Reference

What's inside the Docker image:

```
/opt/
├── rz-vlp/5.0.6/                                  RZ/V LP SDK (cross-compiler)
│   ├── sysroots/
│   │   ├── cortexa55-poky-linux/                  ARM64 sysroot (headers + libs)
│   │   └── x86_64-pokysdk-linux/                  Host tools
│   │       └── usr/bin/aarch64-poky-linux/
│   │           ├── aarch64-poky-linux-gcc
│   │           └── aarch64-poky-linux-g++
│   └── environment-setup-cortexa55-poky-linux     Source this for C++ builds
│
├── DRP-AI_Translator_i8/                          DRP-AI Translator + Quantizer
│   ├── translator/
│   │   ├── DRP-AI_Translator/
│   │   │   └── python_api/                        Translator Python API
│   │   ├── UserConfig/
│   │   ├── run_Translator_v2h.sh
│   │   └── run_Translator_v2n.sh
│   ├── drpAI_Quantizer/                           INT8 quantizer
│   │   ├── nchw_datareader.py
│   │   └── nhwc_datareader.py
│   ├── onnx_models/                               Sample ONNX models
│   └── GettingStarted/                            Renesas docs
│
/drp-ai_tvm/                                       DRP-AI TVM root ($TVM_ROOT)
├── tutorials/
│   └── compile_onnx_model_quant.py                Main compilation script
├── tvm/
│   ├── include/tvm/runtime/                       TVM + custom runtime headers
│   └── python/                                    TVM Python package
├── obj/
│   ├── build_runtime/v2h/lib/                     MERA2 runtime .so files
│   │   ├── libmera2_runtime.so
│   │   ├── libmera2_plan_io.so
│   │   └── libdrp_tvm_rt.so
│   └── pip_package/                               Pre-built Python wheels
├── setup/include/                                 Custom runtime headers
├── 3rdparty/
│   ├── spdlog/                                    Logging library
│   ├── asio/                                      Async I/O library
│   ├── dlpack/                                    DLPack tensor standard
│   └── dmlc-core/                                 DMLC utilities
└── apps/                                          App integration headers
```

---

## Troubleshooting

### Step 3 (Extraction)

| Error | Fix |
|-------|-----|
| `unzip: cannot find or open ... .zip` | Check the ZIP path; the file may not have downloaded fully — re-download |
| Extracted file is 0 bytes | Original ZIP was corrupted — re-download |

### Step 6 (Docker build)

| Error | Fix |
|-------|-----|
| `*.sh: Permission denied` | The COPY/chmod inside the Dockerfile handles this — but make sure the file is in the build context root |
| `cmake-data=3.28.1-*: not found` | Kitware repo failed to add — usually a network blip; retry the build |
| `DRP-AI_Translator*-Install: not found` | The Translator installer is not next to the Dockerfile (Step 5) |
| `pip install mera2_r*: no match` | TVM repo clone failed during build — usually network; retry |
| Build fails ~30 min in with 137 exit | Out of memory or disk — check `df -h` and free disk space |

### Step 8 (Verification)

| Error | Fix |
|-------|-----|
| `ModuleNotFoundError: No module named 'tvm'` (in Python) | You sourced the SDK environment — open a fresh shell and don't source it for Python work |
| `Your environment is misconfigured, you probably need to 'unset LD_LIBRARY_PATH'` | Run `unset LD_LIBRARY_PATH` BEFORE sourcing the SDK script |
| `aarch64-poky-linux-gcc: not found` | SDK env not sourced (or SDK install failed during build) |
| `SDKTARGETSYSROOT is empty` | LD_LIBRARY_PATH was set when sourcing — see above |

### Model compilation (later steps)

| Error | Fix |
|-------|-----|
| `dynamic shape` | Re-export ONNX with `dynamic=False`, `dynamic_axes=None` |
| `Bias not expected to be merged` | Run `onnx-simplifier` on the model first |
| Accuracy drop > 5% after INT8 | Wrong mean/std — for sketch/YOLO models use `[0,0,0]/[1,1,1]`, NOT ImageNet values |
| `Failed to download tophub` | Network issue in Docker — set `TVM_NUM_THREADS=1` |

### Board runtime

| Error | Fix |
|-------|-----|
| `/dev/drpai0: No such file` | DRP-AI driver not loaded — check BSP/kernel config |
| `libmera2_runtime.so: cannot open` | Run `ldconfig` after copying libs to `/usr/lib64/` |
| Display noise/artifacts | DRP-AI + Display DDR bandwidth conflict — apply TF-A QoS patches |
| `WAYLAND_DISPLAY not set` | Weston not running — start it or use `--console` mode |
