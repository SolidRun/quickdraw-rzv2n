# Training

End-to-end guide: from a fresh clone to a trained `qd_model.onnx` ready for DRP-AI compilation.

This guide is verified by re-running the entire pipeline from scratch. Every measurement is real, every step has a verification command, and the final accuracy was reproduced within 0.04% of the reference checkpoint.

---

## Pipeline Overview

```
┌────────────────────────────────────────────────────┐
│ Step 1: Set up Python environment                  │
│ Step 2: Download + render dataset (data_128/)      │  ~30-45 min, ~50 GB
│ Step 3: Run training (train.py)                    │  ~6 hours on RTX 5060 Ti
│ Step 4: Verify outputs (best_model.pt, qd_model.onnx)
│ Step 5: Generate calibration images                 │  ~1 min
└────────────────────────────────────────────────────┘
```

---

## Step 1 — Python Environment

The project uses standard PyTorch + ONNX. No Renesas dependencies (those are only needed for the DRP-AI compile step in [BUILD.md](BUILD.md)).

```bash
# From the project root
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip

# PyTorch (with CUDA — for NVIDIA GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Or CPU-only (slow but works for testing):
# pip install torch torchvision torchaudio

# ONNX export and validation
pip install onnx onnxruntime onnx-simplifier

# Training dependencies
pip install numpy scipy opencv-python pillow matplotlib
```

Verify:
```bash
python3 -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

---

## Step 2 — Download and Render the Dataset

The dataset is **Google Quick Draw simplified NDJSON stroke data**, fetched from `https://storage.googleapis.com/quickdraw_dataset/full/simplified/` and rendered to 128×128 grayscale `.npy` files at training time.

### Run the download

```bash
cd train
python3 download_ndjson.py \
    --categories ../categories.txt \
    --output ./data_128 \
    --max-samples 9000
```

| Parameter | Value | Why |
|---|---|---|
| `--max-samples` | **9000** | Matches the existing trained model. The script's *default* is 8000 — we override |
| `--categories` | `../categories.txt` | 345 class names (one per line) |
| `--output` | `./data_128` | Where to save `.npy` files |

### What the script does (confirmed by reading `download_ndjson.py`)

1. Downloads one `.ndjson` per class from Google Cloud Storage (~50–150 MB each)
2. Filters to `recognized=True` drawings only (cleaner data)
3. Shuffles with `np.random.RandomState(42)` — **deterministic** given the same source data
4. Takes the first `max_samples` after shuffle
5. Renders strokes via `cv2.polylines` at 128×128 with line width 2, padding 10, anti-aliased
6. Saves as `.npy` (shape `(N, 128, 128)`, dtype `uint8`, white strokes on black background)
7. Deletes the `.ndjson` to save disk space

### Measured timing and size

| Metric | Value |
|---|---|
| Wall time | ~30–45 min (network-bound, dominated by ndjson downloads) |
| Final disk usage | ~48 GB |
| Total samples | 3,105,000 (345 classes × 9,000) |
| Per-class file | 147.5 MB (`9000 × 128 × 128` bytes) |

### Verify

```bash
ls train/data_128/*.npy | wc -l
# Expected: 345

python3 -c "
import numpy as np
a = np.load('train/data_128/airplane.npy', mmap_mode='r')
print(f'Shape: {a.shape}, dtype: {a.dtype}, total: {a.nbytes/1e6:.1f} MB')
"
# Expected: Shape: (9000, 128, 128), dtype: uint8, total: 147.5 MB
```

### Verified reproducibility

The dataset generation is **fully deterministic**. I re-downloaded all 345 classes from scratch and SHA256-compared every `.npy` to a saved backup:

> All 345 files byte-identical. Same seed → same shuffle → same renderer → same exact bytes.

---

## Step 3 — Run the Training

```bash
cd train
python3 train.py \
    --data ./data_128 \
    --categories ../categories.txt \
    --output-pt ../best_model.pt \
    --output-onnx ../qd_model.onnx
```

All other flags have sensible defaults — see "Command line arguments" below.

### Two-stage transfer learning

#### Stage 1: Frozen backbone (head only learns)

| Parameter | Value |
|-----------|-------|
| Epochs | 25 (max — early stopping likely terminates earlier) |
| Learning rate | 0.001 |
| Scheduler | `OneCycleLR`, max_lr = lr × 10, pct_start = 0.3 |
| Early stopping | patience = 7 epochs |

#### Stage 2: Fine-tune the last 10 inverted residual blocks

| Parameter | Value |
|-----------|-------|
| Epochs | 20 |
| Learning rate | 0.0001 |
| Scheduler | `OneCycleLR`, max_lr = finetune_lr × 3, pct_start = 0.3 |
| Early stopping | patience = 7 epochs |

#### Shared settings

| Parameter | Value |
|-----------|-------|
| Batch size | 128 |
| Optimizer | AdamW |
| Weight decay | 1e-4 |
| Label smoothing | 0.15 |
| EMA decay | 0.999 |
| Gradient clipping | max_norm = 1.0 |
| Mixup | alpha = 0.2 (applied 30% of the time) |
| Validation split | 10% (seed = 42) |
| Workers | 8 |

### Hardware requirements

The training script auto-detects the device:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

No code change needed to switch between CUDA GPU and CPU. CPU-only works but is much slower.

#### GPU memory (measured)

| | Value |
|---|---|
| Tested GPU | NVIDIA RTX 5060 Ti (16 GB) |
| VRAM used at default `--batch 128` | **~3.0 GB** (3,087 MiB) |
| GPU utilization | ~97% |

If your GPU has less memory and you hit `CUDA out of memory`, lower `--batch` (try 64). VRAM scales roughly linearly with batch size, but other batch sizes are not benchmarked.

#### Wall time (measured)

| Phase | Per-epoch | Total |
|---|---|---|
| Stage 1 (frozen backbone) | ~9.5 min | 11 epochs (early-stopped) ≈ 1h 45min |
| Stage 2 (fine-tuning) | ~13.4 min | 20 epochs (full run) ≈ 4h 30min |
| **Total** | | **~6 hours on RTX 5060 Ti** |

Faster GPUs scale proportionally. CPU-only is roughly 10–50× slower.

### What you'll see during training

```
=================================================================
  STAGE 1: Transfer Learning (frozen backbone)
=================================================================
Epoch | Train Loss | Train Acc | Val Loss | Val Acc | Time | LR
    1 |     3.5669 |    33.32% |   3.0350 |  50.31% | 584s | 8.1e-04 *
    2 |     3.4437 |    35.76% |   2.9948 |  51.53% | 564s | 2.0e-03 *
    ...
  Early stopping: no improvement for 7 epochs
Stage 1 best: 52.37%

=================================================================
  STAGE 2: Fine-tuning (last 10 blocks unfrozen)
=================================================================
    1 |     2.8283 |    50.81% |   2.2891 |  70.97% | 805s | 3.1e-05 *
    2 |     2.6016 |    56.56% |   2.1337 |  74.98% | 804s | 8.4e-05 *
    ...
   20 |     2.0382 |    68.84% |   1.8170 |  82.11% | 796s | 1.2e-09 *
```

The `*` marker means "new best validation accuracy — checkpoint saved."

### Verified reproducibility

Re-running the full pipeline from a freshly downloaded dataset:

| Metric | Original | New | Diff |
|---|---|---|---|
| Stage 1 best val acc | 52.37% | 52.37% | exact |
| **Stage 2 best val acc** | **82.07%** | **82.11%** | **+0.04%** |

The 0.04% variance is normal CUDA non-determinism between runs. ONNX file bytes differ (different floating-point weight values) but model structure (input/output shapes, opset, ops) is identical.

### Command line arguments

| Argument | Default | Notes |
|----------|---------|-------|
| `--data` | `./data` | Directory with `.npy` files |
| `--categories` | `../categories.txt` | One class per line |
| `--epochs` | 25 | Stage 1 max |
| `--finetune-epochs` | 20 | Stage 2 max |
| `--batch` | 128 | Lower if OOM |
| `--lr` | 0.001 | Stage 1 base LR |
| `--finetune-lr` | 0.0001 | Stage 2 base LR |
| `--weight-decay` | 1e-4 | |
| `--label-smoothing` | 0.15 | |
| `--patience` | 7 | Early stopping epochs |
| `--max-per-class` | 50000 | Limited by available data |
| `--workers` | 8 | DataLoader workers |
| `--seed` | 42 | RNG seed |
| `--output-pt` | `../best_model.pt` | Saved checkpoint |
| `--output-onnx` | `../qd_model.onnx` | Auto-exported after training |
| `--norm-mode` | `sketch` | `sketch` (0–1) or `imagenet` |
| `--resume` | None | Path to a `.pt` to resume from |

### Data augmentation (training only)

For 128×128 pre-rendered data (the standard path):

| Augmentation | Parameters |
|---|---|
| SimulateBoardPreprocessing | p=0.3, margin_range=(2, 8) |
| StrokeAugmentation | p=0.2 (random dilation or erosion) |
| RandomAffine | degrees=15, translate=(0.08, 0.08), scale=(0.9, 1.1) |
| RandomPerspective | distortion_scale=0.15, p=0.2 |
| RandomErasing | p=0.15, scale=(0.02, 0.1) |

Validation uses no augmentation — only normalization.

---

## Step 4 — Verify the Output

After training, two files appear in the project root:

```bash
ls -lh best_model.pt qd_model.onnx
```

### `best_model.pt` (PyTorch checkpoint)

```
best_model.pt    14 MB
```

State dict only (model weights). Used by `train.py --resume` to continue training, not by the board.

### `qd_model.onnx` (the file used downstream)

```
qd_model.onnx    14 MB
```

This is what [BUILD.md](BUILD.md) Step 1 (`compile_model.sh`) takes as input.

#### Verify it's valid and DRP-AI-compatible

```bash
python3 -c "
import onnx
m = onnx.load('qd_model.onnx')
onnx.checker.check_model(m)
inp = m.graph.input[0]
out = m.graph.output[0]
print(f'Input:  {inp.name}, shape={[d.dim_value for d in inp.type.tensor_type.shape.dim]}')
print(f'Output: {out.name}, shape={[d.dim_value for d in out.type.tensor_type.shape.dim]}')
print(f'Opset:  {m.opset_import[0].version}')
print(f'Nodes:  {len(m.graph.node)}')
"
```

Expected output:
```
Input:  image, shape=[1, 3, 128, 128]
Output: logits, shape=[1, 345]
Opset:  11
Nodes:  103
```

| Required for DRP-AI | Verified? |
|---|---|
| Opset ≤ 17 | ✓ (11) |
| Static input shape, batch=1 | ✓ |
| FP32 (no QDQ pre-quant) | ✓ |
| No NMS in graph | ✓ (classification model) |
| `onnx.checker.check_model` passes | ✓ |

### Model architecture (for reference)

`build_model()` in `train/train.py`:

```
MobileNetV2 backbone (ImageNet pretrained)
    ↓
19 inverted residual blocks → 1280 channels
    ↓
Global average pooling → [1280]
    ↓
Dropout(0.3)
    ↓
Linear(1280, 768) → BatchNorm1d(768) → ReLU
    ↓
Dropout(0.2)
    ↓
Linear(768, 345)
    ↓
Output: [1, 345] logits
```

### Input convention

- Shape: `[1, 3, 128, 128]` (NCHW)
- Grayscale repeated to 3 channels (R = G = B)
- Normalization: `mean=[0.0, 0.0, 0.0]`, `std=[1.0, 1.0, 1.0]` (simple 0–1 pixel scaling)

> **Critical**: This `mean`/`std` setting is sketch-specific, NOT ImageNet. The DRP-AI compile script in [BUILD.md](BUILD.md) Step 1 has matching patches. Mismatched normalization between training and INT8 calibration will destroy accuracy.

---

## Step 5 — Generate Calibration Images

INT8 quantization in [BUILD.md](BUILD.md) Step 1 requires representative input images to determine quantization ranges.

```bash
cd ..   # back to project root
python3 generate_calibration.py --per-class 5
```

| Parameter | Value |
|---|---|
| Images per class | 5 (default) |
| Total images | 1,725 (345 × 5) |
| Source | Last 10% of each category's data (validation-like split) |
| Even-indexed samples | Standard resize to 128×128 |
| Odd-indexed samples | Board-style crop+pad (crop to ink bbox, pad to square, resize) |
| Output format | 128×128 RGB PNG |
| Output directory | `calibration/` |

The 50/50 mix of standard and board-style preprocessing ensures the quantizer sees activation ranges from both deployment scenarios.

### Verify

```bash
ls calibration/*.png | wc -l
# Expected: 1725

du -sh calibration/
# Expected: ~13 MB
```

> **Note**: This repo includes `calibration/` already-generated to make the DRP-AI compile step in BUILD.md fully reproducible.

---

## Next Step

Now that you have `qd_model.onnx` and `calibration/`, continue to [BUILD.md](BUILD.md) Step 1 to compile the model for DRP-AI.

---

## Troubleshooting

| Symptom | Cause / Fix |
|---|---|
| `CUDA out of memory` during training | Lower `--batch` (try 64 or 32) |
| Training extremely slow on CPU | Expected — get a CUDA GPU or use a smaller subset |
| Stage 1 ends much earlier than 25 epochs | Normal — early stopping with patience=7 typically triggers around epoch 11 |
| Stage 2 val acc stalls below 80% | Check `--norm-mode` is `sketch` (not `imagenet`) |
| Final accuracy differs by ~0.1% from this guide | Normal CUDA non-determinism between GPU models |
| Final accuracy differs by >2% from this guide | Check the dataset hash — Google may have updated the source data |
| `download_ndjson.py: HTTP error` | Transient network issue. Re-run — it skips already-downloaded classes |
| Renders look wrong / blank | Check the `recognized=True` filter is applied (default), confirm `cv2.LINE_AA` works |
