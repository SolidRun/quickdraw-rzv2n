#!/usr/bin/env python3
"""
Generate calibration images for DRP-AI INT8 quantization (MobileNetV2).
Creates multiple 128x128 RGB PNGs per category from the Quick Draw dataset.

Half the images use standard resize (matching training), and half use
crop+pad preprocessing (matching the board's C++ inference pipeline).
This ensures INT8 activation ranges cover both scenarios.

The .npy data is already white strokes on black background — NO inversion needed.
Images are repeated to 3 channels to match training input format.

Smart features:
  - Reads categories.txt to know all classes automatically
  - Auto-detects data directory (train/data_128, train/data, or custom)
  - Reports missing categories and coverage stats

Usage:
    python generate_calibration.py
    python generate_calibration.py --per-class 100 --categories categories.txt
"""
import argparse
import os
import random
import sys
import numpy as np
from PIL import Image


def crop_pad_like_board(img_gray, target_size, ink_threshold=30, margin=2):
    """Simulate the C++ board preprocessing: crop to ink bbox, pad to square, resize.

    Input: HxW uint8, white strokes on black (0=bg, 255=ink).
    Output: target_size x target_size PIL Image (white strokes on black).
    """
    ys, xs = np.where(img_gray > ink_threshold)
    if len(ys) == 0:
        return None

    y_min = max(0, ys.min() - margin)
    y_max = min(img_gray.shape[0] - 1, ys.max() + margin)
    x_min = max(0, xs.min() - margin)
    x_max = min(img_gray.shape[1] - 1, xs.max() + margin)

    cropped = img_gray[y_min:y_max + 1, x_min:x_max + 1]

    ch, cw = cropped.shape
    side = max(ch, cw)
    square = np.zeros((side, side), dtype=np.uint8)
    pad_top = (side - ch) // 2
    pad_left = (side - cw) // 2
    square[pad_top:pad_top + ch, pad_left:pad_left + cw] = cropped

    img = Image.fromarray(square, mode='L')
    if side > target_size:
        img = img.resize((target_size, target_size), Image.BOX)
    else:
        img = img.resize((target_size, target_size), Image.BILINEAR)

    return img


def find_data_dir(base_dir):
    """Auto-detect the data directory containing .npy files."""
    candidates = [
        os.path.join(base_dir, "train", "data_128"),
        os.path.join(base_dir, "train", "data"),
    ]
    for d in candidates:
        if os.path.isdir(d) and any(f.endswith('.npy') for f in os.listdir(d)):
            return d
    return None


def main():
    ap = argparse.ArgumentParser(description="Generate calibration images for DRP-AI INT8 quantization")
    ap.add_argument("--categories", default="categories.txt",
                    help="Path to categories.txt (one class per line)")
    ap.add_argument("--data", default=None,
                    help="Data directory with .npy files (auto-detected if not set)")
    ap.add_argument("--output", default="calibration")
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--per-class", type=int, default=5,
                    help="Number of calibration images per class (default: 5)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # --- Read categories ---
    if not os.path.exists(args.categories):
        print(f"ERROR: Categories file not found: {args.categories}")
        sys.exit(1)

    with open(args.categories) as f:
        categories = [line.strip() for line in f if line.strip()]

    num_classes = len(categories)
    if num_classes == 0:
        print("ERROR: No categories found in categories.txt")
        sys.exit(1)

    # --- Find data directory ---
    if args.data:
        data_dir = args.data
    else:
        base = os.path.dirname(os.path.abspath(args.categories))
        data_dir = find_data_dir(base)
        if data_dir is None:
            print("ERROR: Could not auto-detect data directory. Use --data to specify.")
            sys.exit(1)

    total_images = num_classes * args.per_class

    print(f"{'='*60}")
    print(f"  Calibration Image Generator")
    print(f"{'='*60}")
    print(f"  Categories file: {args.categories}")
    print(f"  Num classes:     {num_classes}")
    print(f"  Data directory:  {data_dir}")
    print(f"  Per class:       {args.per_class}")
    print(f"  Total expected:  {total_images}")
    print(f"  Image size:      {args.size}x{args.size} RGB")
    print(f"  Output:          {args.output}/")
    print(f"{'='*60}\n")

    # Clear old calibration data
    if os.path.exists(args.output):
        old_count = len([f for f in os.listdir(args.output) if f.endswith('.png')])
        if old_count > 0:
            print(f"Clearing {old_count} old calibration images...")
            for f in os.listdir(args.output):
                if f.endswith('.png'):
                    os.remove(os.path.join(args.output, f))
    os.makedirs(args.output, exist_ok=True)

    count = 0
    count_standard = 0
    count_cropped = 0
    missing_cats = []
    class_counts = {}

    for class_id, cat in enumerate(categories):
        npy_path = os.path.join(data_dir, f"{cat}.npy")
        if not os.path.exists(npy_path):
            missing_cats.append(cat)
            continue

        data = np.load(npy_path, mmap_mode='r')
        # Pick samples from the end of the dataset (least likely to be in training split)
        total_available = len(data)
        val_start = int(total_available * 0.9)  # Last 10% = validation-like
        val_count = total_available - val_start
        if val_count < 1:
            val_start = 0
            val_count = total_available

        n_samples = min(args.per_class, val_count)
        indices = random.sample(range(val_start, val_start + val_count), n_samples)

        cat_count = 0
        for sample_idx, idx in enumerate(indices):
            raw = data[idx]
            if len(raw.shape) == 2:
                img_gray = raw.astype(np.uint8)
            elif raw.shape[0] == 784:
                img_gray = raw.reshape(28, 28).astype(np.uint8)
            else:
                side = int(np.sqrt(raw.shape[0]))
                img_gray = raw.reshape(side, side).astype(np.uint8)

            # Alternate: half standard resize, half crop+pad (board-style)
            if sample_idx % 2 == 0:
                pil_gray = Image.fromarray(img_gray, mode='L').resize(
                    (args.size, args.size), Image.BILINEAR)
                count_standard += 1
            else:
                pil_gray = crop_pad_like_board(img_gray, args.size)
                if pil_gray is None:
                    pil_gray = Image.fromarray(img_gray, mode='L').resize(
                        (args.size, args.size), Image.BILINEAR)
                    count_standard += 1
                else:
                    count_cropped += 1

            img_rgb = Image.merge('RGB', [pil_gray, pil_gray, pil_gray])
            img_rgb.save(os.path.join(args.output, f"{cat}_{sample_idx}.png"))
            count += 1
            cat_count += 1

        class_counts[cat] = cat_count

    # --- Report ---
    print(f"\n{'='*60}")
    print(f"  CALIBRATION GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"  Total images:    {count}")
    print(f"  Standard resize: {count_standard}")
    print(f"  Board-style:     {count_cropped}")
    print(f"  Classes covered: {len(class_counts)}/{num_classes}")

    if missing_cats:
        print(f"\n  WARNING: {len(missing_cats)} categories missing .npy data:")
        for cat in missing_cats:
            print(f"    - {cat}")

    # Verify all classes have expected count
    short_classes = [cat for cat, c in class_counts.items() if c < args.per_class]
    if short_classes:
        print(f"\n  WARNING: {len(short_classes)} classes have fewer than {args.per_class} images:")
        for cat in short_classes[:10]:
            print(f"    - {cat}: {class_counts[cat]} images")
        if len(short_classes) > 10:
            print(f"    ... and {len(short_classes) - 10} more")

    if not missing_cats and not short_classes:
        print(f"\n  All {num_classes} classes have {args.per_class} calibration images each.")

    print(f"{'='*60}")


if __name__ == "__main__":
    main()
