#!/usr/bin/env python3
"""
Download Google Quick Draw numpy bitmap dataset for all 345 categories.

Each .npy file contains thousands of 28x28 grayscale images (uint8, 0=bg, 255=ink).
Source: https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/

Usage:
    python download_dataset.py --categories ../categories.txt --output ./data [--max-samples 6000]
"""
import argparse
import os
import urllib.request
import urllib.parse
import numpy as np


BASE_URL = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/"


def load_categories(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def download_category(name: str, output_dir: str, max_samples: int = 6000) -> int:
    """Download a single category's numpy bitmap file."""
    encoded = urllib.parse.quote(name) + ".npy"
    url = BASE_URL + encoded
    npy_path = os.path.join(output_dir, f"{name}.npy")

    if os.path.exists(npy_path):
        data = np.load(npy_path)
        print(f"  [SKIP] {name} — already exists ({len(data)} samples)")
        return len(data)

    try:
        print(f"  [GET]  {name} ... ", end="", flush=True)
        urllib.request.urlretrieve(url, npy_path + ".tmp")

        # Load full file, keep only max_samples, save truncated
        data = np.load(npy_path + ".tmp")
        if len(data) > max_samples:
            data = data[:max_samples]
        np.save(npy_path, data)
        os.remove(npy_path + ".tmp")
        print(f"{len(data)} samples")
        return len(data)
    except Exception as e:
        print(f"FAILED: {e}")
        if os.path.exists(npy_path + ".tmp"):
            os.remove(npy_path + ".tmp")
        return 0


def main():
    ap = argparse.ArgumentParser(description="Download Quick Draw numpy bitmaps")
    ap.add_argument("--categories", default="../categories.txt",
                    help="Path to categories.txt")
    ap.add_argument("--output", default="./data",
                    help="Output directory for .npy files")
    ap.add_argument("--max-samples", type=int, default=6000,
                    help="Max samples per category (default: 6000)")
    args = ap.parse_args()

    categories = load_categories(args.categories)
    print(f"Downloading {len(categories)} categories to {args.output}/")
    os.makedirs(args.output, exist_ok=True)

    total = 0
    failed = []
    for i, cat in enumerate(categories, 1):
        print(f"[{i}/{len(categories)}]", end="")
        count = download_category(cat, args.output, max_samples=args.max_samples)
        if count == 0:
            failed.append(cat)
        total += count

    print(f"\nDone: {total} total samples across {len(categories) - len(failed)} categories")
    if failed:
        print(f"Failed ({len(failed)}): {failed}")


if __name__ == "__main__":
    main()
