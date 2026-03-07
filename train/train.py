#!/usr/bin/env python3
"""
Train MobileNetV2 on Quick Draw grayscale sketches (128x128 or 28x28 auto-detected).
Uses ImageNet pretrained backbone with two-stage transfer learning.
Exports DRP-AI compatible ONNX after training.

Architecture: MobileNetV2 (3-channel RGB input, ImageNet pretrained backbone)
Input:        grayscale → repeat to 3ch → resize 128x128 → normalize (sketch or ImageNet)
Output:       [1, NUM_CLASSES] logits

DRP-AI benchmark: ~0.3-0.4ms inference on RZ/V2H/V2N (128x128)

Improvements v4:
  - 128x128 ndjson-rendered data (was 28x28 upscaled bitmaps)
  - Sketch normalization [0,1] (was ImageNet mean/std)
  - Recognized-only filtering (cleaner training data)
  - Hidden layer 768 + BatchNorm1d
  - OneCycleLR scheduler
  - Mixup augmentation (alpha=0.2)
  - EMA model weights (decay=0.999)
  - Stroke dilation/erosion augmentation
  - Unfreeze 10 blocks in Stage 2
  - Label smoothing 0.15
  - Lighter augmentation for pre-rendered 128x128 data

Usage:
    python train.py --data ./data_128 --categories ../categories.txt --norm-mode sketch
"""
import argparse
import copy
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from scipy.ndimage import binary_dilation, binary_erosion

# Input size — 128x128 proven optimal for Quick Draw (Kaggle top-10)
INPUT_SIZE = 128

# Normalization mode: "sketch" uses [0,1] scaling (natural for binary sketches),
# "imagenet" uses ImageNet mean/std (for photo-pretrained backbones).
# Research shows sketch normalization works better for Quick Draw.
NORM_SKETCH_MEAN = [0.0, 0.0, 0.0]
NORM_SKETCH_STD = [1.0, 1.0, 1.0]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class SimulateBoardPreprocessing:
    """Simulate the C++ board preprocessing pipeline during training.

    The board C++ code does: find ink bbox → crop with margin → pad to square → invert.
    This augmentation randomly crops to the ink bounding box (with variable margin)
    so the model learns to handle strokes at different scales relative to the frame.

    Applied BEFORE Resize, on the raw numpy uint8 image (28x28 or 128x128).
    """

    def __init__(self, p=0.5, margin_range=(1, 6)):
        self.p = p
        self.margin_range = margin_range

    def __call__(self, img):
        """img: numpy uint8 array, white-on-black."""
        if np.random.random() > self.p:
            return img

        # Find ink pixels (white strokes on black bg, threshold > 30)
        ys, xs = np.where(img > 30)
        if len(ys) == 0:
            return img

        # Crop to ink bounding box with random margin
        margin = np.random.randint(self.margin_range[0], self.margin_range[1] + 1)
        y_min = max(0, ys.min() - margin)
        y_max = min(img.shape[0] - 1, ys.max() + margin)
        x_min = max(0, xs.min() - margin)
        x_max = min(img.shape[1] - 1, xs.max() + margin)

        cropped = img[y_min:y_max + 1, x_min:x_max + 1]

        # Pad to square (centered, black fill — matches board preprocessing after invert)
        ch, cw = cropped.shape
        side = max(ch, cw)
        square = np.zeros((side, side), dtype=np.uint8)
        pad_top = (side - ch) // 2
        pad_left = (side - cw) // 2
        square[pad_top:pad_top + ch, pad_left:pad_left + cw] = cropped

        return square


class StrokeAugmentation:
    """Randomly dilate or erode strokes to simulate different pen thicknesses."""

    def __init__(self, p=0.3):
        self.p = p

    def __call__(self, img):
        if np.random.random() > self.p:
            return img
        binary = img > 30
        if not binary.any():
            return img
        if np.random.random() < 0.5:
            binary = binary_dilation(binary, iterations=1)
        else:
            binary = binary_erosion(binary, iterations=1)
        return (binary * 255).astype(np.uint8)


class EMA:
    """Exponential Moving Average of model weights for better generalization."""

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model.state_dict())

    def update(self, model):
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if v.is_floating_point():
                    self.shadow[k].mul_(self.decay).add_(v, alpha=1.0 - self.decay)
                else:
                    self.shadow[k].copy_(v)

    def apply(self, model):
        model.load_state_dict(self.shadow)

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state):
        self.shadow = state


class QuickDrawDataset(Dataset):
    """Load Quick Draw numpy bitmaps with memory-mapped files for low RAM usage.

    Auto-detects image resolution from data:
      - 28x28 (legacy bitmap): needs resize to INPUT_SIZE
      - 128x128 (ndjson-rendered): already at target size, skip resize
    """

    def __init__(self, data_dir: str, categories: list,
                 max_per_class: int = 50000, split: str = "train",
                 val_ratio: float = 0.1, seed: int = 42,
                 norm_mode: str = "sketch"):

        # Auto-detect image size from first available .npy file
        self._img_size = 28  # default
        for cat in categories:
            npy_path = os.path.join(data_dir, f"{cat}.npy")
            if os.path.exists(npy_path):
                sample = np.load(npy_path, mmap_mode='r')
                pixels = sample.shape[1] if len(sample.shape) > 1 else sample.shape[0]
                side = int(np.sqrt(pixels)) if pixels > 200 else pixels
                # Check if data is already 128x128 (shape: N, 128, 128) or flat (N, 784)
                if len(sample.shape) == 3:
                    self._img_size = sample.shape[1]
                elif len(sample.shape) == 2 and sample.shape[1] == 784:
                    self._img_size = 28
                elif len(sample.shape) == 2 and sample.shape[1] == 16384:
                    self._img_size = 128
                break

        needs_resize = (self._img_size != INPUT_SIZE)

        # Select normalization
        if norm_mode == "sketch":
            norm_mean, norm_std = NORM_SKETCH_MEAN, NORM_SKETCH_STD
        else:
            norm_mean, norm_std = IMAGENET_MEAN, IMAGENET_STD

        # Build transforms
        # For 128x128 pre-rendered data: lighter augmentation (data already has natural variation)
        # For 28x28 legacy data: heavier augmentation + resize
        # Always include Resize — SimulateBoardPreprocessing produces variable-size
        # output (crops to ink bbox), so we must resize back to INPUT_SIZE.
        if split == "train":
            if needs_resize:
                self.pre_transform = SimulateBoardPreprocessing(p=0.5, margin_range=(1, 6))
                self.stroke_aug = StrokeAugmentation(p=0.3)
            else:
                self.pre_transform = SimulateBoardPreprocessing(p=0.3, margin_range=(2, 8))
                self.stroke_aug = StrokeAugmentation(p=0.2)

            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
                transforms.Grayscale(num_output_channels=3),
                transforms.RandomAffine(degrees=15, translate=(0.08, 0.08),
                                        scale=(0.9, 1.1)),
                transforms.RandomPerspective(distortion_scale=0.15, p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(norm_mean, norm_std),
                transforms.RandomErasing(p=0.15, scale=(0.02, 0.1)),
            ])
        else:
            self.pre_transform = None
            self.stroke_aug = None
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize(norm_mean, norm_std),
            ])

        # Memory-mapped loading
        self._mmaps = []
        self._index = []
        rng = np.random.RandomState(seed)

        for class_id, cat in enumerate(categories):
            npy_path = os.path.join(data_dir, f"{cat}.npy")
            if not os.path.exists(npy_path):
                print(f"  [WARN] Missing: {npy_path}", flush=True)
                continue

            mmap_data = np.load(npy_path, mmap_mode='r')
            n = len(mmap_data)

            indices = rng.permutation(n)
            val_count = max(1, int(n * val_ratio))

            if split == "train":
                selected = indices[val_count:val_count + max_per_class]
            else:
                selected = indices[:val_count]

            mmap_idx = len(self._mmaps)
            self._mmaps.append(mmap_data)
            for row in selected:
                self._index.append((mmap_idx, int(row), class_id))

        if split == "train":
            rng.shuffle(self._index)

        total = len(self._index)
        n_classes = len(categories)
        res_str = f"{self._img_size}x{self._img_size}"
        if needs_resize:
            res_str += f" → {INPUT_SIZE}x{INPUT_SIZE}"
        print(f"  [{split}] {total} samples, {n_classes} classes, {res_str}, "
              f"norm={norm_mode} (memory-mapped)", flush=True)

    def __len__(self):
        return len(self._index)

    def __getitem__(self, idx):
        mmap_idx, row, label = self._index[idx]
        raw = np.array(self._mmaps[mmap_idx][row])
        img = raw.reshape(self._img_size, self._img_size).astype(np.uint8)
        if self.pre_transform is not None:
            img = self.pre_transform(img)
        if self.stroke_aug is not None:
            img = self.stroke_aug(img)
        img_tensor = self.transform(img)
        return img_tensor, label


def build_model(num_classes: int) -> nn.Module:
    """MobileNetV2 with ImageNet pretrained weights, custom classifier head."""
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    # Replace classifier: 1280 → 768 → num_classes
    # Hidden layer with BatchNorm improves discriminability for 345 sketch classes
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(1280, 768),
        nn.BatchNorm1d(768),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.2),
        nn.Linear(768, num_classes),
    )
    return model


def freeze_backbone(model):
    """Freeze all backbone (features) layers — only train classifier."""
    model.features.requires_grad_(False)
    # Keep BatchNorm in eval mode when frozen
    for m in model.features.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
            m.eval()


def unfreeze_last_blocks(model, num_blocks=6):
    """Unfreeze the last N inverted residual blocks for fine-tuning."""
    # MobileNetV2 has 19 blocks in model.features (indices 0-18)
    total = len(model.features)
    for i in range(max(0, total - num_blocks), total):
        model.features[i].requires_grad_(True)
    # Keep BatchNorm in eval mode for fine-tuning stability
    for m in model.features.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
            m.eval()


def train_one_epoch(model, loader, criterion, optimizer, device,
                    freeze_bn=False, scheduler=None, ema=None, mixup_alpha=0.2):
    model.train()
    if freeze_bn:
        # Keep BN in eval mode during fine-tuning
        for m in model.features.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                m.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        # Mixup augmentation
        use_mixup = mixup_alpha > 0 and np.random.random() < 0.3
        if use_mixup:
            lam = np.random.beta(mixup_alpha, mixup_alpha)
            idx = torch.randperm(images.size(0), device=device)
            images = lam * images + (1 - lam) * images[idx]

        outputs = model(images)

        if use_mixup:
            loss = lam * criterion(outputs, labels) + (1 - lam) * criterion(outputs, labels[idx])
        else:
            loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Per-batch scheduler step (OneCycleLR)
        if scheduler is not None:
            scheduler.step()

        # EMA update
        if ema is not None:
            ema.update(model)

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += images.size(0)

    return total_loss / total, 100.0 * correct / total


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += images.size(0)

    return total_loss / total, 100.0 * correct / total


def export_onnx(model, num_classes, output_path, device):
    """Export to DRP-AI compatible ONNX (3-channel RGB input, 128x128)."""
    model.eval()
    dummy = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE, device=device)

    torch.onnx.export(
        model,
        dummy,
        output_path,
        opset_version=11,
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes=None,
        do_constant_folding=True,
        dynamo=False,  # Legacy exporter — supports opset 11 for DRP-AI TVM
    )
    print(f"\nExported ONNX: {output_path}")

    # Simplify
    try:
        import onnx
        from onnxsim import simplify
        onnx_model = onnx.load(output_path)
        simplified, ok = simplify(onnx_model)
        if ok:
            onnx.save(simplified, output_path,
                      save_as_external_data=False)
            print("ONNX simplified successfully")
        else:
            print("WARNING: onnx-simplifier failed, using unsimplified model")
    except ImportError:
        print("WARNING: onnx-simplifier not installed, skipping simplification")

    # Validate
    try:
        import onnx
        model_onnx = onnx.load(output_path)
        onnx.checker.check_model(model_onnx)

        inp = model_onnx.graph.input[0]
        shape = [d.dim_value for d in inp.type.tensor_type.shape.dim]
        print(f"ONNX input shape: {shape}")
        print(f"ONNX opset: {model_onnx.opset_import[0].version}")

        dynamic = any(d == 0 for d in shape)
        if dynamic:
            print("[FAIL] Dynamic dimensions detected!")
        else:
            print("[PASS] All dimensions static")

        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"File size: {size_mb:.2f} MB")
    except Exception as e:
        print(f"Validation error: {e}")


def run_stage(name, model, train_loader, val_loader, criterion,
              optimizer, scheduler, device, num_epochs, patience,
              best_acc, output_pt, freeze_bn=False, ema=None, mixup_alpha=0.2):
    """Run one training stage, return best accuracy."""
    no_improve = 0

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            freeze_bn, scheduler=scheduler, ema=ema, mixup_alpha=mixup_alpha)

        # Validate using EMA weights if available
        if ema is not None:
            orig_state = copy.deepcopy(model.state_dict())
            ema.apply(model)
            val_loss, val_acc = validate(model, val_loader, criterion, device)
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), output_pt)  # Save EMA weights
                marker = " *"
                no_improve = 0
            else:
                no_improve += 1
                marker = ""
            model.load_state_dict(orig_state)  # Restore training weights
        else:
            val_loss, val_acc = validate(model, val_loader, criterion, device)
            marker = ""
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), output_pt)
                marker = " *"
                no_improve = 0
            else:
                no_improve += 1

        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]['lr']
        print(f"{epoch:5d} | {train_loss:10.4f} | {train_acc:8.2f}% | "
              f"{val_loss:8.4f} | {val_acc:6.2f}% | {elapsed:5.1f}s | "
              f"lr={lr:.1e}{marker}", flush=True)

        if no_improve >= patience:
            print(f"  Early stopping: no improvement for {patience} epochs")
            break

    return best_acc


def main():
    ap = argparse.ArgumentParser(description="Train Quick Draw MobileNetV2")
    ap.add_argument("--data", default="./data", help="Directory with .npy files")
    ap.add_argument("--categories", default="../categories.txt")
    ap.add_argument("--epochs", type=int, default=25,
                    help="Stage 1 epochs (frozen backbone)")
    ap.add_argument("--finetune-epochs", type=int, default=20,
                    help="Stage 2 epochs (unfrozen last blocks)")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.001,
                    help="Stage 1 learning rate")
    ap.add_argument("--finetune-lr", type=float, default=0.0001,
                    help="Stage 2 learning rate")
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--label-smoothing", type=float, default=0.15)
    ap.add_argument("--patience", type=int, default=7,
                    help="Early stopping patience")
    ap.add_argument("--max-per-class", type=int, default=50000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output-pt", default="../best_model.pt")
    ap.add_argument("--output-onnx", default="../qd_model.onnx")
    ap.add_argument("--norm-mode", default="sketch", choices=["sketch", "imagenet"],
                    help="Normalization mode: sketch=[0,1] or imagenet=[0.485,...]/[0.229,...]")
    ap.add_argument("--resume", default=None,
                    help="Resume from checkpoint (skip Stage 1, run Stage 2 only)")
    args = ap.parse_args()

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load categories
    with open(args.categories, "r") as f:
        categories = [line.strip() for line in f if line.strip()]
    num_classes = len(categories)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"{'='*65}")
    print(f"  Quick Draw MobileNetV2 — Transfer Learning v4")
    print(f"{'='*65}")
    print(f"Classes:       {num_classes}")
    print(f"Device:        {device}")
    print(f"Input size:    {INPUT_SIZE}x{INPUT_SIZE}")
    print(f"Batch size:    {args.batch}")
    print(f"Max/class:     {args.max_per_class}")
    print(f"Stage 1:       {args.epochs} epochs, lr={args.lr}, frozen backbone")
    print(f"Stage 2:       {args.finetune_epochs} epochs, lr={args.finetune_lr}, fine-tune last 10 blocks")
    print(f"Weight decay:  {args.weight_decay}")
    print(f"Label smooth:  {args.label_smoothing}")
    print(f"Patience:      {args.patience}")
    print(f"Seed:          {args.seed}")
    print(f"Norm mode:     {args.norm_mode}")
    print(f"{'='*65}\n")

    # Datasets
    print("Loading training data...")
    train_ds = QuickDrawDataset(args.data, categories,
                                max_per_class=args.max_per_class, split="train",
                                seed=args.seed, norm_mode=args.norm_mode)
    print("Loading validation data...")
    val_ds = QuickDrawDataset(args.data, categories,
                              max_per_class=args.max_per_class, split="val",
                              seed=args.seed, norm_mode=args.norm_mode)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, pin_memory=True,
                              persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            num_workers=args.workers, pin_memory=True,
                            persistent_workers=True)

    # Model
    print("Loading MobileNetV2 with ImageNet pretrained weights...")
    model = build_model(num_classes).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    best_acc = 0.0

    header = (f"{'Epoch':>5} | {'Train Loss':>10} | {'Train Acc':>9} | "
              f"{'Val Loss':>8} | {'Val Acc':>7} | {'Time':>6} | LR")
    separator = "-" * 80

    # ═══════════════════════════════════════════════════════════════
    # Stage 1: Frozen backbone — train only classifier head
    # ═══════════════════════════════════════════════════════════════
    if args.resume:
        print(f"\n  Resuming from checkpoint: {args.resume}")
        model.load_state_dict(torch.load(args.resume, map_location=device,
                                         weights_only=True))
        model.to(device)
        # Validate to get baseline accuracy
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        best_acc = val_acc
        print(f"  Checkpoint val accuracy: {val_acc:.2f}%")
        print(f"  Skipping Stage 1 — going directly to Stage 2")
    else:
        print(f"\n{'='*65}")
        print(f"  STAGE 1: Transfer Learning (frozen backbone)")
        print(f"{'='*65}")
        print(header)
        print(separator)

        freeze_backbone(model)

        ema = EMA(model, decay=0.999)

        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr, weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=args.lr * 10,
            steps_per_epoch=len(train_loader),
            epochs=args.epochs, pct_start=0.3)

        best_acc = run_stage(
            "Stage1", model, train_loader, val_loader, criterion,
            optimizer, scheduler, device, args.epochs, args.patience,
            best_acc, args.output_pt, freeze_bn=True, ema=ema)

        print(f"\nStage 1 best: {best_acc:.2f}%")

    # ═══════════════════════════════════════════════════════════════
    # Stage 2: Fine-tune last 10 blocks
    # ═══════════════════════════════════════════════════════════════
    if args.finetune_epochs > 0:
        print(f"\n{'='*65}")
        print(f"  STAGE 2: Fine-tuning (last 10 blocks unfrozen)")
        print(f"{'='*65}")
        print(header)
        print(separator)

        # Reload best checkpoint (from Stage 1 or --resume)
        if not args.resume:
            model.load_state_dict(torch.load(args.output_pt, map_location=device,
                                             weights_only=True))
            model.to(device)

        unfreeze_last_blocks(model, num_blocks=10)

        ema = EMA(model, decay=0.999)

        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.finetune_lr, weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=args.finetune_lr * 3,
            steps_per_epoch=len(train_loader),
            epochs=args.finetune_epochs, pct_start=0.3)

        best_acc = run_stage(
            "Stage2", model, train_loader, val_loader, criterion,
            optimizer, scheduler, device, args.finetune_epochs, args.patience,
            best_acc, args.output_pt, freeze_bn=True, ema=ema)

        print(f"\nStage 2 best: {best_acc:.2f}%")

    # ═══════════════════════════════════════════════════════════════
    # Export ONNX
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*65}")
    print(f"  ONNX Export")
    print(f"{'='*65}")
    print(f"\nFinal best accuracy: {best_acc:.2f}%")
    print(f"Saved: {args.output_pt}")

    model.load_state_dict(torch.load(args.output_pt, map_location=device,
                                     weights_only=True))
    model.to(device)
    export_onnx(model, num_classes, args.output_onnx, device)


if __name__ == "__main__":
    main()
