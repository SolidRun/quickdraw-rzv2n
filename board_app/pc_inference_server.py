#!/usr/bin/env python3
"""
Quick Draw — PC Inference Server (ONNX Runtime)

Drop-in replacement for the C++ DRP-AI server. Speaks the same Unix socket
protocol so the GTK3 GUI (quickdraw_gui.py) works unchanged.

Supports both ONNX (.onnx) and PyTorch (.pt) models.

Usage:
    python3 pc_inference_server.py [--model MODEL_PATH] [--labels LABELS_FILE]
                                   [--size N] [--socket PATH] [--smooth N]
"""

import argparse
import json
import math
import os
import signal
import socket
import struct
import sys
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent


# ═══════════════════════════════════════════════════════════════════════
# Model loading — ONNX or PyTorch
# ═══════════════════════════════════════════════════════════════════════

class ONNXModel:
    def __init__(self, path):
        import onnxruntime as ort
        self.session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        out = self.session.get_outputs()[0]
        self.output_name = out.name
        self.num_classes = out.shape[-1] if out.shape and len(out.shape) > 1 else None
        print(f"  ONNX model loaded: input='{self.input_name}', output='{self.output_name}'")

    def infer(self, tensor_nchw):
        """tensor_nchw: numpy float32 [1, 3, H, W]. Returns logits [num_classes]."""
        outputs = self.session.run([self.output_name], {self.input_name: tensor_nchw})
        return outputs[0].flatten()


class PyTorchModel:
    def __init__(self, path):
        import torch
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(checkpoint, dict):
            if "model_state_dict" in checkpoint:
                # Need architecture info — try to reconstruct MobileNetV2
                self.model = self._build_mobilenetv2(checkpoint)
            elif "model" in checkpoint:
                self.model = checkpoint["model"]
            elif "ema" in checkpoint:
                self.model = checkpoint["ema"]
            else:
                raise ValueError(f"Unknown checkpoint format. Keys: {list(checkpoint.keys())}")
        else:
            self.model = checkpoint

        self.model.float().eval()
        self.num_classes = None
        print(f"  PyTorch model loaded")

    def _build_mobilenetv2(self, checkpoint):
        import torch
        import torchvision.models as models
        state = checkpoint["model_state_dict"]
        # Detect num_classes from classifier weight
        for key in state:
            if "classifier" in key and "weight" in key:
                num_classes = state[key].shape[0]
                break
        else:
            num_classes = 345
        model = models.mobilenet_v2(num_classes=num_classes)
        model.load_state_dict(state)
        self.num_classes = num_classes
        print(f"  Reconstructed MobileNetV2 with {num_classes} classes")
        return model

    def infer(self, tensor_nchw):
        """tensor_nchw: numpy float32 [1, 3, H, W]. Returns logits [num_classes]."""
        import torch
        with torch.no_grad():
            t = torch.from_numpy(tensor_nchw)
            out = self.model(t)
            return out.cpu().numpy().flatten()


def load_model(path):
    path = str(path)
    if path.endswith(".onnx"):
        return ONNXModel(path)
    elif path.endswith(".pt") or path.endswith(".pth"):
        return PyTorchModel(path)
    else:
        raise ValueError(f"Unsupported model format: {path} (use .onnx or .pt)")


# ═══════════════════════════════════════════════════════════════════════
# Preprocessing — matches the C++ pipeline exactly
# ═══════════════════════════════════════════════════════════════════════

def preprocess_canvas(gray_bytes, width, height, model_size=128,
                      ink_threshold=245, crop_margin=12):
    """
    Replicate the C++ preprocessing pipeline:
    1. Find ink bounding box (pixels < ink_threshold)
    2. Crop with margin
    3. Pad to square (centered, white fill)
    4. Invert (white strokes on black bg)
    5. Resize to model_size x model_size (area-based for downscale)
    6. Normalize to [0, 1], replicate to 3 channels, CHW layout

    Returns None if blank, else float32 array [1, 3, model_size, model_size].
    """
    gray = np.frombuffer(gray_bytes, dtype=np.uint8).reshape(height, width)

    # 1. Find ink bounding box
    ink_mask = gray < ink_threshold
    if not ink_mask.any():
        return None

    rows = np.any(ink_mask, axis=1)
    cols = np.any(ink_mask, axis=0)
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]

    # 2. Crop with margin
    x_min = max(0, x_min - crop_margin)
    y_min = max(0, y_min - crop_margin)
    x_max = min(width - 1, x_max + crop_margin)
    y_max = min(height - 1, y_max + crop_margin)

    cropped = gray[y_min:y_max + 1, x_min:x_max + 1]
    ch, cw = cropped.shape

    # 3. Pad to square
    side = max(cw, ch)
    pad_left = (side - cw) // 2
    pad_top = (side - ch) // 2
    square = np.full((side, side), 255, dtype=np.uint8)
    square[pad_top:pad_top + ch, pad_left:pad_left + cw] = cropped

    # 4. Invert
    square = 255 - square

    # 5. Resize (area-based for downsampling, bilinear for upsampling)
    from PIL import Image
    img = Image.fromarray(square, mode="L")
    img = img.resize((model_size, model_size), Image.LANCZOS)
    resized = np.array(img, dtype=np.float32)

    # 6. Normalize to [0, 1], 3-channel CHW
    normalized = resized / 255.0
    tensor = np.stack([normalized, normalized, normalized], axis=0)  # [3, H, W]
    return tensor[np.newaxis, ...].astype(np.float32)  # [1, 3, H, W]


# ═══════════════════════════════════════════════════════════════════════
# Softmax + Top-K
# ═══════════════════════════════════════════════════════════════════════

def softmax(logits):
    e = np.exp(logits - np.max(logits))
    return e / e.sum()


def top_k(probs, class_names, k=5):
    indices = np.argsort(probs)[::-1][:k]
    return [{"class": class_names[i], "class_id": int(i), "prob": float(probs[i])}
            for i in indices]


# ═══════════════════════════════════════════════════════════════════════
# Temporal smoothing
# ═══════════════════════════════════════════════════════════════════════

class Smoother:
    def __init__(self, window=3):
        self.window = window
        self.history = deque(maxlen=window)

    def apply(self, probs):
        self.history.append(probs.copy())
        if len(self.history) == 1:
            return probs
        avg = np.mean(list(self.history), axis=0)
        return avg

    def reset(self):
        self.history.clear()


# ═══════════════════════════════════════════════════════════════════════
# Socket server — same wire protocol as C++ DRP-AI server
# ═══════════════════════════════════════════════════════════════════════

def recv_exact(conn, n):
    buf = bytearray(n)
    view = memoryview(buf)
    pos = 0
    while pos < n:
        nbytes = conn.recv_into(view[pos:], n - pos)
        if nbytes == 0:
            raise ConnectionError("client disconnected")
        pos += nbytes
    return bytes(buf)


def send_exact(conn, data):
    conn.sendall(data)


def send_error(conn, msg):
    payload = json.dumps({"error": msg}).encode("utf-8")
    send_exact(conn, struct.pack("<I", len(payload)))
    send_exact(conn, payload)


def handle_request(conn, model, class_names, smoother, model_size):
    # Read header: [uint32 msg_len][uint16 width][uint16 height]
    header = recv_exact(conn, 8)
    msg_len, width, height = struct.unpack("<IHH", header)

    num_pixels = width * height
    expected = 4 + num_pixels
    if msg_len != expected:
        raise ValueError(f"Protocol error: msg_len={msg_len} expected={expected}")

    # Read grayscale pixels
    gray_bytes = recv_exact(conn, num_pixels)

    # Preprocess
    tensor = preprocess_canvas(gray_bytes, width, height, model_size=model_size)
    if tensor is None:
        smoother.reset()
        send_error(conn, "blank")
        return True

    # Inference
    t0 = time.monotonic()
    logits = model.infer(tensor)
    elapsed_ms = (time.monotonic() - t0) * 1000

    # Softmax → smooth → top-K
    num_classes = min(len(logits), len(class_names))
    probs = softmax(logits[:num_classes])
    smoothed = smoother.apply(probs)
    results = top_k(smoothed, class_names, k=5)

    # Build JSON response (same format as C++ server)
    response = {
        "predictions": results,
        "smooth_n": len(smoother.history),
        "min_conf": 0.0,
    }
    payload = json.dumps(response).encode("utf-8")
    send_exact(conn, struct.pack("<I", len(payload)))
    send_exact(conn, payload)

    # Log
    if results:
        top = results[0]
        print(f"  [{width}x{height}] {top['class']} ({top['prob']*100:.1f}% "
              f"smooth={len(smoother.history)}) {elapsed_ms:.1f}ms")

    return True


def run_server(model, class_names, socket_path, model_size, smooth_window):
    # Remove stale socket
    try:
        os.unlink(socket_path)
    except FileNotFoundError:
        pass

    smoother = Smoother(smooth_window)
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(socket_path)
    server.listen(1)
    server.settimeout(1.0)

    print(f"\nListening on {socket_path}")
    print(f"Ready for connections. Press Ctrl+C to stop.\n")

    running = True

    def on_signal(sig, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    try:
        while running:
            try:
                conn, _ = server.accept()
            except socket.timeout:
                continue

            print(f"Client connected")
            conn.settimeout(5.0)

            try:
                while running:
                    if not handle_request(conn, model, class_names, smoother, model_size):
                        break
            except (ConnectionError, ValueError, socket.timeout) as e:
                print(f"Client disconnected: {e}")
            finally:
                conn.close()
    finally:
        server.close()
        try:
            os.unlink(socket_path)
        except FileNotFoundError:
            pass
        print("\nServer stopped.")


# ═══════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Quick Draw PC Inference Server")
    parser.add_argument("--model", default=str(SCRIPT_DIR.parent / "qd_model.onnx"),
                        help="Path to .onnx or .pt model file")
    parser.add_argument("--labels", default=str(SCRIPT_DIR / "labels.txt"),
                        help="Path to labels.txt (one class per line)")
    parser.add_argument("--size", type=int, default=128,
                        help="Model input size NxN (default: 128)")
    parser.add_argument("--socket", default="/tmp/quickdraw.sock",
                        help="Unix socket path (default: /tmp/quickdraw.sock)")
    parser.add_argument("--smooth", type=int, default=3,
                        help="Temporal smoothing window (default: 3)")
    args = parser.parse_args()

    # Load labels
    with open(args.labels) as f:
        class_names = [line.strip() for line in f if line.strip()]
    if not class_names:
        print(f"ERROR: No class names in {args.labels}", file=sys.stderr)
        sys.exit(1)

    print()
    print("=" * 48)
    print("  Quick Draw PC Server (ONNX Runtime)")
    print("=" * 48)
    print(f"Model:   {args.model}")
    print(f"Labels:  {args.labels} ({len(class_names)} classes)")
    print(f"Input:   {args.size}x{args.size}")
    print(f"Socket:  {args.socket}")
    print(f"Smooth:  window={args.smooth}")
    print("-" * 48)

    # Load model
    model = load_model(args.model)

    # Warmup
    dummy = np.random.randn(1, 3, args.size, args.size).astype(np.float32)
    logits = model.infer(dummy)
    print(f"Warmup:  output_size={len(logits)}, labels={len(class_names)}")
    if len(logits) != len(class_names):
        print(f"WARNING: Model outputs {len(logits)} values but labels has {len(class_names)} entries!")
    print("-" * 48)

    run_server(model, class_names, args.socket, args.size, args.smooth)


if __name__ == "__main__":
    main()
