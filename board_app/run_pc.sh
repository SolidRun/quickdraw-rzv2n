#!/bin/bash
# Quick Draw — PC Mode
# Launches the ONNX inference server + GTK3 GUI on PC (no DRP-AI needed)
#
# Usage: ./run_pc.sh [--model PATH] [--pt]
#
# Options:
#   --model PATH   Path to model file (default: ../qd_model.onnx)
#   --pt           Use PyTorch model (../best_model.pt) instead of ONNX

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL="${SCRIPT_DIR}/../qd_model.onnx"
EXTRA_ARGS=""

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --model) MODEL="$2"; shift 2;;
        --pt)    MODEL="${SCRIPT_DIR}/../best_model.pt"; shift;;
        *)       EXTRA_ARGS="$EXTRA_ARGS $1"; shift;;
    esac
done

echo "=================================="
echo "  Quick Draw — PC Mode"
echo "=================================="
echo "Model:  $MODEL"
echo "Config: ${SCRIPT_DIR}/config_pc.json"
echo ""

# Check model exists
if [ ! -f "$MODEL" ]; then
    echo "ERROR: Model not found: $MODEL"
    echo "Available models:"
    ls -la "${SCRIPT_DIR}"/../*.onnx "${SCRIPT_DIR}"/../*.pt 2>/dev/null
    exit 1
fi

# Start inference server in background
echo "Starting inference server..."
python3 "${SCRIPT_DIR}/pc_inference_server.py" \
    --model "$MODEL" \
    --labels "${SCRIPT_DIR}/labels.txt" \
    $EXTRA_ARGS &
SERVER_PID=$!

# Wait for socket to appear
SOCKET="/tmp/quickdraw.sock"
for i in $(seq 1 30); do
    if [ -S "$SOCKET" ]; then
        break
    fi
    sleep 0.2
done

if [ ! -S "$SOCKET" ]; then
    echo "ERROR: Server failed to start (socket not created)"
    kill $SERVER_PID 2>/dev/null
    exit 1
fi

echo "Server ready. Starting GUI..."
echo ""

# Start GUI (blocks until closed)
python3 "${SCRIPT_DIR}/quickdraw_gui.py" --config "${SCRIPT_DIR}/config_pc.json"

# Cleanup: kill server when GUI exits
echo "GUI closed. Stopping server..."
kill $SERVER_PID 2>/dev/null
wait $SERVER_PID 2>/dev/null
echo "Done."
