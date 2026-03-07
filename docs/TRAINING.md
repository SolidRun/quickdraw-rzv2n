# Training

## Dataset

**Source:** Google Quick Draw simplified NDJSON stroke data.
**URL:** `https://storage.googleapis.com/quickdraw_dataset/full/simplified/`

### Download and render

```bash
cd train
python download_ndjson.py --categories ../categories.txt --output ./data_128 --max-samples 8000
```

| Parameter | Value |
|-----------|-------|
| Categories | 345 (listed in `categories.txt`) |
| Max samples per class | 8,000 (default) |
| Rendering resolution | 128 x 128 |
| Line width | 2 px |
| Padding | 10 px |
| Anti-aliasing | `cv2.LINE_AA` |
| Filtering | `recognized=true` only |
| Shuffle | Yes (before selecting samples) |
| Output format | `.npy` per class, shape `(N, 128, 128)`, dtype uint8 |
| Pixel convention | White strokes on black background |

The script downloads one category at a time, renders strokes using `cv2.polylines`, saves as `.npy`, then deletes the `.ndjson` source to save disk space.

### Verify

```bash
ls train/data_128/*.npy | wc -l
# Expected: 345
```

---

## Model Architecture

MobileNetV2 with ImageNet pretrained backbone and a custom classifier head.

```
MobileNetV2 backbone (ImageNet pretrained)
    |
    19 inverted residual blocks -> 1280 channels
    |
    Global average pooling -> [1280]
    |
    Dropout(0.3)
    |
    Linear(1280, 768) -> BatchNorm1d(768) -> ReLU
    |
    Dropout(0.2)
    |
    Linear(768, 345)
    |
    Output: [1, 345] logits
```

**Source:** `train/train.py`, `build_model()` function.

### Input

- Shape: `[1, 3, 128, 128]` (NCHW)
- Grayscale repeated to 3 channels (R=G=B)
- Normalization: `mean=[0.0, 0.0, 0.0]`, `std=[1.0, 1.0, 1.0]` (0-1 pixel scaling)

---

## Training Procedure

Two-stage transfer learning.

### Stage 1: Frozen backbone

Only the classifier head trains. The MobileNetV2 backbone is frozen.

| Parameter | Value |
|-----------|-------|
| Epochs | 25 |
| Learning rate | 0.001 |
| Scheduler | OneCycleLR, max_lr = lr x 10, pct_start=0.3 |
| Early stopping | patience = 7 |

### Stage 2: Fine-tuning

The last 10 inverted residual blocks are unfrozen. The classifier head continues training.

| Parameter | Value |
|-----------|-------|
| Epochs | 20 |
| Learning rate | 0.0001 |
| Scheduler | OneCycleLR, max_lr = finetune_lr x 3, pct_start=0.3 |
| Early stopping | patience = 7 |

### Shared settings (both stages)

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
| Max per class | 50,000 (default, limited by available data) |
| Workers | 8 |

### Data augmentation (training only)

For 128x128 pre-rendered data (no resize needed):

| Augmentation | Parameters |
|-------------|------------|
| SimulateBoardPreprocessing | p=0.3, margin_range=(2, 8) |
| StrokeAugmentation | p=0.2 (random dilation or erosion) |
| RandomAffine | degrees=15, translate=(0.08, 0.08), scale=(0.9, 1.1) |
| RandomPerspective | distortion_scale=0.15, p=0.2 |
| RandomErasing | p=0.15, scale=(0.02, 0.1) |

For 28x28 legacy data (resize needed):

| Augmentation | Parameters |
|-------------|------------|
| SimulateBoardPreprocessing | p=0.5, margin_range=(1, 6) |
| StrokeAugmentation | p=0.3 |
| Same geometric augmentations as above | |

Validation uses no augmentation — only resize, grayscale-to-3ch, and normalization.

---

## Run Training

```bash
cd train
python train.py
```

### Command line arguments

| Argument | Default |
|----------|---------|
| `--data` | `./data` |
| `--categories` | `../categories.txt` |
| `--epochs` | 25 |
| `--finetune-epochs` | 20 |
| `--batch` | 128 |
| `--lr` | 0.001 |
| `--finetune-lr` | 0.0001 |
| `--weight-decay` | 1e-4 |
| `--label-smoothing` | 0.15 |
| `--patience` | 7 |
| `--max-per-class` | 50000 |
| `--workers` | 8 |
| `--seed` | 42 |
| `--output-pt` | `../best_model.pt` |
| `--output-onnx` | `../qd_model.onnx` |
| `--norm-mode` | `sketch` (choices: sketch, imagenet) |
| `--resume` | None |

---

## ONNX Export

The training script exports ONNX automatically after training.

| Property | Value |
|----------|-------|
| Opset | 11 |
| Input name | `image` |
| Output name | `logits` |
| Input shape | `[1, 3, 128, 128]` (static) |
| Dynamic axes | None |
| Constant folding | Enabled |
| Simplification | onnx-simplifier applied automatically |

Output files:
- `best_model.pt` — 14 MB (PyTorch state dict)
- `qd_model.onnx` — 14 MB (FP32, weights embedded)

---

## Calibration Images

INT8 quantization requires representative images to set quantization ranges.

```bash
cd quickdraw
python generate_calibration.py --per-class 5
```

| Parameter | Value |
|-----------|-------|
| Images per class | 5 (default) |
| Total images | 1,725 |
| Source | Last 10% of each category's data (validation-like split) |
| Even-indexed samples | Standard resize to 128x128 |
| Odd-indexed samples | Board-style crop+pad (crop to ink bbox, pad to square, resize) |
| Output format | 128x128 RGB PNG |
| Output directory | `calibration/` |

The 50/50 mix of standard and board-style preprocessing ensures the quantizer sees activation ranges from both scenarios.
