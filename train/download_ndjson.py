#!/usr/bin/env python3
"""
Download Quick Draw simplified .ndjson data and render strokes to 128x128 grayscale .npy files.

Instead of using pre-rendered 28x28 bitmaps (blurry when upscaled), this renders
directly from stroke data at 128x128 with controlled line width for sharp images.

Features:
  - Downloads one class at a time to minimize disk usage
  - Filters to recognized=true drawings only (cleaner data)
  - Shuffles before selecting (avoids temporal bias)
  - Renders strokes with cv2.polylines at native 128x128
  - Saves as .npy per class (memory-mappable)

Usage:
    python download_ndjson.py --categories ../categories.txt --output ./data_128 --max-samples 8000
"""
import argparse
import json
import os
import urllib.request
import numpy as np
import cv2

# Google Cloud Storage base URL for simplified .ndjson files
GCS_BASE = "https://storage.googleapis.com/quickdraw_dataset/full/simplified"


def download_ndjson(category: str, temp_dir: str) -> str:
    """Download simplified .ndjson for one category. Returns local file path."""
    url_name = category.replace(" ", "%20")
    url = f"{GCS_BASE}/{url_name}.ndjson"
    os.makedirs(temp_dir, exist_ok=True)
    local_path = os.path.join(temp_dir, f"{category}.ndjson")

    if os.path.exists(local_path):
        return local_path

    print(f"    Downloading {category}.ndjson ...", end="", flush=True)
    urllib.request.urlretrieve(url, local_path)
    size_mb = os.path.getsize(local_path) / (1024 * 1024)
    print(f" {size_mb:.1f} MB", flush=True)
    return local_path


def render_strokes(drawing, size=128, line_width=2, padding=10):
    """Render Quick Draw strokes to a grayscale image.

    Args:
        drawing: list of strokes, each stroke is [x_coords, y_coords]
        size: output image size (square)
        line_width: stroke width in pixels
        padding: padding around the drawing
    Returns:
        numpy array of shape (size, size), dtype uint8, white strokes on black bg
    """
    # Find bounding box of all strokes
    all_x = []
    all_y = []
    for stroke in drawing:
        all_x.extend(stroke[0])
        all_y.extend(stroke[1])

    if not all_x:
        return np.zeros((size, size), dtype=np.uint8)

    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)

    # Scale to fit in (size - 2*padding) maintaining aspect ratio
    draw_w = max_x - min_x
    draw_h = max_y - min_y
    if draw_w == 0 and draw_h == 0:
        return np.zeros((size, size), dtype=np.uint8)

    max_dim = max(draw_w, draw_h)
    if max_dim == 0:
        max_dim = 1
    scale = (size - 2 * padding) / max_dim

    # Center the drawing
    offset_x = padding + (size - 2 * padding - draw_w * scale) / 2 - min_x * scale
    offset_y = padding + (size - 2 * padding - draw_h * scale) / 2 - min_y * scale

    img = np.zeros((size, size), dtype=np.uint8)

    for stroke in drawing:
        xs = stroke[0]
        ys = stroke[1]
        if len(xs) < 2:
            continue
        points = np.array(
            [[int(x * scale + offset_x), int(y * scale + offset_y)]
             for x, y in zip(xs, ys)],
            dtype=np.int32
        )
        cv2.polylines(img, [points], isClosed=False, color=255,
                       thickness=line_width, lineType=cv2.LINE_AA)

    return img


def process_category(category: str, temp_dir: str, output_dir: str,
                     max_samples: int, size: int, recognized_only: bool):
    """Download, render, and save one category."""
    # Download .ndjson
    ndjson_path = download_ndjson(category, temp_dir)

    # Parse and filter
    drawings = []
    with open(ndjson_path, "r") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if recognized_only and not obj.get("recognized", False):
                continue
            drawings.append(obj["drawing"])

    # Shuffle before selecting (avoid temporal/geographic bias)
    rng = np.random.RandomState(42)
    rng.shuffle(drawings)

    # Take max_samples
    selected = drawings[:max_samples]
    actual = len(selected)

    if actual == 0:
        print(f"    WARNING: No drawings for {category}", flush=True)
        return 0

    # Render all selected drawings
    images = np.zeros((actual, size, size), dtype=np.uint8)
    for i, strokes in enumerate(selected):
        images[i] = render_strokes(strokes, size=size)

    # Save as .npy
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, f"{category}.npy"), images)

    # Delete .ndjson to save disk space
    os.remove(ndjson_path)

    return actual


def main():
    ap = argparse.ArgumentParser(description="Download Quick Draw .ndjson and render to 128x128 .npy")
    ap.add_argument("--categories", default="../categories.txt",
                    help="Path to categories.txt (one class per line)")
    ap.add_argument("--output", default="./data_128",
                    help="Output directory for rendered .npy files")
    ap.add_argument("--temp", default="/tmp/quickdraw_ndjson",
                    help="Temp directory for .ndjson downloads")
    ap.add_argument("--max-samples", type=int, default=8000,
                    help="Max samples per class")
    ap.add_argument("--size", type=int, default=128,
                    help="Output image size (default: 128)")
    ap.add_argument("--recognized-only", action="store_true", default=True,
                    help="Only use recognized=true drawings (default: True)")
    ap.add_argument("--no-recognized-filter", action="store_true",
                    help="Use all drawings including unrecognized")
    args = ap.parse_args()

    if args.no_recognized_filter:
        args.recognized_only = False

    with open(args.categories, "r") as f:
        categories = [line.strip() for line in f if line.strip()]

    print(f"{'='*60}")
    print(f"  Quick Draw .ndjson → {args.size}x{args.size} Renderer")
    print(f"{'='*60}")
    print(f"  Classes:         {len(categories)}")
    print(f"  Max samples:     {args.max_samples}/class")
    print(f"  Output size:     {args.size}x{args.size} grayscale")
    print(f"  Recognized only: {args.recognized_only}")
    print(f"  Output dir:      {args.output}")
    est_gb = len(categories) * args.max_samples * args.size * args.size / (1024**3)
    print(f"  Est. disk usage: {est_gb:.1f} GB")
    print(f"{'='*60}\n")

    total_samples = 0
    for i, cat in enumerate(categories):
        # Check if already rendered
        npy_path = os.path.join(args.output, f"{cat}.npy")
        if os.path.exists(npy_path):
            existing = np.load(npy_path, mmap_mode='r')
            if len(existing) >= args.max_samples:
                print(f"  [{i+1:3d}/{len(categories)}] {cat}: already done ({len(existing)} samples)")
                total_samples += len(existing)
                continue

        print(f"  [{i+1:3d}/{len(categories)}] {cat}:", end=" ", flush=True)
        count = process_category(cat, args.temp, args.output,
                                 args.max_samples, args.size,
                                 args.recognized_only)
        print(f"{count} samples rendered", flush=True)
        total_samples += count

    print(f"\n{'='*60}")
    print(f"  DONE: {total_samples:,} total samples in {args.output}")
    actual_gb = sum(
        os.path.getsize(os.path.join(args.output, f))
        for f in os.listdir(args.output) if f.endswith('.npy')
    ) / (1024**3)
    print(f"  Disk usage: {actual_gb:.1f} GB")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
