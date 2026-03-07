#!/usr/bin/env python3
"""
Test preprocessing alignment: compare training vs board (C++) preprocessing.

Loads a Quick Draw .npy sample, applies both pipelines, and saves side-by-side
images so you can visually verify they produce similar inputs to the model.

Also tests ONNX model accuracy on validation data using both preprocessing methods.

Usage:
    python test_preprocessing.py --data train/data --categories categories.txt
    python test_preprocessing.py --data train/data --categories categories.txt --onnx qd_model.onnx
"""
import argparse
import os
import numpy as np
from PIL import Image

INPUT_SIZE = 128
NORM_MEAN = [0.0, 0.0, 0.0]
NORM_STD = [1.0, 1.0, 1.0]


def training_preprocess(img_28x28):
    """Replicate training val preprocessing (no augmentation)."""
    # ToPILImage → Resize(128) → Grayscale(3ch) → ToTensor → Normalize
    img = Image.fromarray(img_28x28, mode='L')
    img = img.resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    # 3-channel + ImageNet normalize
    chw = np.stack([
        (arr - NORM_MEAN[0]) / NORM_STD[0],
        (arr - NORM_MEAN[1]) / NORM_STD[1],
        (arr - NORM_MEAN[2]) / NORM_STD[2],
    ])
    return chw  # (3, 128, 128)


def board_preprocess(img_28x28, ink_threshold=245, crop_margin=12):
    """Replicate C++ board preprocessing on 28x28 data.

    The board gets a large canvas with black strokes on white bg.
    For 28x28 data (white-on-black), we simulate: imagine the 28x28 image
    was drawn on a white canvas, find ink, crop, pad, invert, resize.
    """
    # The .npy data is white strokes on black (0=bg, 255=ink)
    # Board canvas is black strokes on white (0=ink, 255=bg)
    # So first invert to simulate canvas: now 0=ink (was 255=ink), 255=bg (was 0=bg)
    canvas = 255 - img_28x28

    # Find ink bbox (pixels < ink_threshold, i.e., dark pixels = ink)
    ys, xs = np.where(canvas < ink_threshold)
    if len(ys) == 0:
        return None

    y_min = max(0, ys.min() - crop_margin)
    y_max = min(canvas.shape[0] - 1, ys.max() + crop_margin)
    x_min = max(0, xs.min() - crop_margin)
    x_max = min(canvas.shape[1] - 1, xs.max() + crop_margin)

    cropped = canvas[y_min:y_max + 1, x_min:x_max + 1]

    # Pad to square (white fill)
    ch, cw = cropped.shape
    side = max(ch, cw)
    square = np.full((side, side), 255, dtype=np.uint8)
    pad_top = (side - ch) // 2
    pad_left = (side - cw) // 2
    square[pad_top:pad_top + ch, pad_left:pad_left + cw] = cropped

    # Invert back: white strokes on black bg
    inverted = 255 - square

    # Resize to 128x128 (area-based for downsampling, bilinear for upsampling)
    img_pil = Image.fromarray(inverted, mode='L')
    if side > INPUT_SIZE:
        resized = img_pil.resize((INPUT_SIZE, INPUT_SIZE), Image.BOX)
    else:
        resized = img_pil.resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)

    arr = np.array(resized, dtype=np.float32) / 255.0

    # 3-channel + ImageNet normalize (CHW)
    chw = np.stack([
        (arr - NORM_MEAN[0]) / NORM_STD[0],
        (arr - NORM_MEAN[1]) / NORM_STD[1],
        (arr - NORM_MEAN[2]) / NORM_STD[2],
    ])
    return chw  # (3, 128, 128)


def save_comparison(img_28, train_chw, board_chw, path):
    """Save side-by-side comparison: original | training | board."""
    # Denormalize for visualization (use channel 0)
    def denorm(chw):
        arr = chw[0] * NORM_STD[0] + NORM_MEAN[0]
        return np.clip(arr * 255, 0, 255).astype(np.uint8)

    orig = Image.fromarray(img_28, mode='L').resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)
    train_img = Image.fromarray(denorm(train_chw), mode='L')
    board_img = Image.fromarray(denorm(board_chw), mode='L')

    # Side by side with labels
    gap = 4
    total_w = INPUT_SIZE * 3 + gap * 2
    total_h = INPUT_SIZE + 20
    canvas = Image.new('L', (total_w, total_h), 0)
    canvas.paste(orig, (0, 20))
    canvas.paste(train_img, (INPUT_SIZE + gap, 20))
    canvas.paste(board_img, (INPUT_SIZE * 2 + gap * 2, 20))
    canvas.save(path)


def test_onnx_accuracy(onnx_path, data_dir, categories, preprocess_fn, name, max_per_class=500):
    """Test ONNX model accuracy with a given preprocessing function."""
    try:
        import onnxruntime as ort
    except ImportError:
        print(f"  [SKIP] onnxruntime not installed")
        return

    sess = ort.InferenceSession(onnx_path)
    input_name = sess.get_inputs()[0].name

    correct = 0
    total = 0
    rng = np.random.RandomState(42)

    for class_id, cat in enumerate(categories):
        npy_path = os.path.join(data_dir, f"{cat}.npy")
        if not os.path.exists(npy_path):
            continue
        data = np.load(npy_path)
        indices = rng.permutation(len(data))
        val_count = max(1, int(len(data) * 0.1))
        val_data = data[indices[:val_count]]
        samples = val_data[:max_per_class]

        for img_flat in samples:
            img = img_flat.reshape(28, 28).astype(np.uint8)
            chw = preprocess_fn(img)
            if chw is None:
                continue
            inp = chw[np.newaxis, :, :, :].astype(np.float32)
            out = sess.run(None, {input_name: inp})[0]
            pred = np.argmax(out[0])
            if pred == class_id:
                correct += 1
            total += 1

    acc = 100.0 * correct / total if total > 0 else 0
    print(f"  {name}: {correct}/{total} = {acc:.2f}%")
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="train/data")
    ap.add_argument("--categories", default="categories.txt")
    ap.add_argument("--onnx", default="", help="ONNX model path (optional, for accuracy test)")
    ap.add_argument("--output", default="preprocess_comparison")
    ap.add_argument("--num-samples", type=int, default=5, help="Visual samples per class")
    ap.add_argument("--num-classes-visual", type=int, default=10, help="Classes for visual comparison")
    ap.add_argument("--max-per-class-acc", type=int, default=200, help="Samples/class for accuracy test")
    args = ap.parse_args()

    with open(args.categories) as f:
        categories = [line.strip() for line in f if line.strip()]

    os.makedirs(args.output, exist_ok=True)

    print(f"Preprocessing Alignment Test")
    print(f"  Categories: {len(categories)}")
    print(f"  Data: {args.data}")
    print()

    # 1. Visual comparison
    print("=== Visual Comparison ===")
    rng = np.random.RandomState(42)
    saved = 0
    for cat in categories[:args.num_classes_visual]:
        npy_path = os.path.join(args.data, f"{cat}.npy")
        if not os.path.exists(npy_path):
            continue
        data = np.load(npy_path)
        indices = rng.permutation(len(data))
        for i in range(min(args.num_samples, len(data))):
            img = data[indices[i]].reshape(28, 28).astype(np.uint8)
            train_chw = training_preprocess(img)
            board_chw = board_preprocess(img)
            if board_chw is None:
                continue
            save_comparison(img, train_chw, board_chw,
                          os.path.join(args.output, f"{cat}_{i}.png"))
            saved += 1
    print(f"  Saved {saved} comparison images to {args.output}/")

    # 2. Pixel-level difference stats
    print("\n=== Pixel Difference Statistics ===")
    diffs = []
    for cat in categories[:50]:
        npy_path = os.path.join(args.data, f"{cat}.npy")
        if not os.path.exists(npy_path):
            continue
        data = np.load(npy_path)
        for i in range(min(20, len(data))):
            img = data[i].reshape(28, 28).astype(np.uint8)
            train_chw = training_preprocess(img)
            board_chw = board_preprocess(img)
            if board_chw is None:
                continue
            diff = np.abs(train_chw - board_chw).mean()
            diffs.append(diff)

    if diffs:
        print(f"  Mean absolute diff (normalized): {np.mean(diffs):.4f}")
        print(f"  Max absolute diff (normalized):  {np.max(diffs):.4f}")
        print(f"  Samples compared: {len(diffs)}")

    # 3. ONNX accuracy comparison (if model provided)
    if args.onnx and os.path.exists(args.onnx):
        print(f"\n=== ONNX Accuracy Test ({args.onnx}) ===")
        test_onnx_accuracy(args.onnx, args.data, categories,
                          training_preprocess, "Training preprocess",
                          args.max_per_class_acc)
        test_onnx_accuracy(args.onnx, args.data, categories,
                          board_preprocess, "Board preprocess",
                          args.max_per_class_acc)
    elif args.onnx:
        print(f"\n  ONNX model not found: {args.onnx}")

    print("\nDone.")


if __name__ == "__main__":
    main()
