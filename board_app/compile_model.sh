#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════
# DRP-AI TVM Model Compilation for Quick Draw MobileNetV2
# Run INSIDE the DRP-AI TVM Docker container.
#
# This compiles qd_model.onnx → DRP-AI runtime model for RZ/V2N
#
# Model: MobileNetV2, 3-channel RGB, 128x128, sketch normalization (0-1 scaling)
#
# Usage:
#   docker exec -it <container> bash
#   cd /path/to/board_app
#   ./compile_model.sh /path/to/qd_model.onnx /path/to/calibration/
#
# The script handles:
#   - 3-channel RGB preprocessing with sketch normalization (mean=0, std=1)
#   - Calibration resize patch (direct resize to 128, no center crop)
#   - INT8 quantization with calibration images
#   - MERA2 compilation for V2H/V2N
# ═══════════════════════════════════════════════════════════════════════

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'

# --- Parse arguments ---
ONNX_PATH="${1:-/quickdraw/qd_model.onnx}"
CALIB_DIR="${2:-/quickdraw/calibration}"
OUTPUT_DIR="${3:-qd_mobilenetv2}"
NUM_CALIB="${4:-1725}"

echo -e "${CYAN}═══════════════════════════════════════════════${NC}"
echo -e "${CYAN}  Quick Draw MobileNetV2 — DRP-AI TVM Compilation${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════${NC}"
echo ""
echo "ONNX model:   $ONNX_PATH"
echo "Calibration:  $CALIB_DIR"
echo "Output:       $OUTPUT_DIR"
echo "Calib images: $NUM_CALIB"
echo "Input shape:  1,3,128,128"
echo "Normalization: Sketch (mean=[0,0,0], std=[1,1,1]) — simple 0-1 scaling"
echo "Quantization: INT8 (Percentile 99.99, CPU=float32)"
echo ""

# --- Verify inputs ---
if [ ! -f "$ONNX_PATH" ]; then
    echo -e "${RED}ERROR: ONNX model not found: $ONNX_PATH${NC}"
    exit 1
fi

if [ ! -d "$CALIB_DIR" ]; then
    echo -e "${RED}ERROR: Calibration directory not found: $CALIB_DIR${NC}"
    exit 1
fi

CALIB_COUNT=$(ls -1 "$CALIB_DIR"/*.png 2>/dev/null | wc -l)
if [ "$CALIB_COUNT" -eq 0 ]; then
    echo -e "${RED}ERROR: No PNG files in calibration directory${NC}"
    exit 1
fi
echo "Found $CALIB_COUNT calibration images"

# --- Verify environment ---
export TVM_ROOT="${TVM_ROOT:-/drp-ai_tvm}"
if [ ! -d "$TVM_ROOT" ]; then
    echo -e "${RED}ERROR: TVM_ROOT not found: $TVM_ROOT${NC}"
    exit 1
fi

export SDK="${SDK:-/opt/rz-vlp/5.0.6}"
export TRANSLATOR="${TRANSLATOR:-/opt/DRP-AI_Translator_i8/translator/}"
export QUANTIZER="${QUANTIZER:-/opt/DRP-AI_Translator_i8/drpAI_Quantizer/}"
# V2N uses the same DRP-AI3 silicon as V2H
export PRODUCT="${PRODUCT:-V2N}"

COMPILE_SCRIPT="$TVM_ROOT/tutorials/compile_onnx_model_quant.py"
if [ ! -f "$COMPILE_SCRIPT" ]; then
    echo -e "${RED}ERROR: Compile script not found: $COMPILE_SCRIPT${NC}"
    exit 1
fi
echo "TVM_ROOT:     $TVM_ROOT"
echo "SDK:          $SDK"
echo "PRODUCT:      $PRODUCT"
echo ""

# --- Patch compile script for 128x128 calibration ---
# The compile script's calibration preprocessing does: resize(N) → center_crop(N).
# Our training does: direct resize(128). Patch to match.
# Also patches mean/std from ImageNet to sketch normalization.
echo -e "${CYAN}Patching compile script for 128x128 calibration...${NC}"

COMPILE_WORK="$TVM_ROOT/tutorials/compile_onnx_model_quant_qd.py"
cp "$COMPILE_SCRIPT" "$COMPILE_WORK"

# Patch calibration preprocessing: resize directly to 128, no center crop
# Use regex to match any integer in the resize/crop calls
sed -i 's/F.resize(img, [0-9]*, Image.BILINEAR)/F.resize(img, 128, Image.BILINEAR)/g' "$COMPILE_WORK"
sed -i 's/F.center_crop(img, [0-9]*)/img/g' "$COMPILE_WORK"

# Patch PreRuntime preprocessing input shape: [1, 480, 640, 3] → [1, 128, 128, 3]
# This controls the DRP-AI hardware preprocessing pipeline shape
sed -i 's/config\.shape_in.*=.*\[1,.*480,.*640,.*3\]/config.shape_in     = [1, 128, 128, 3]/' "$COMPILE_WORK"

# Patch PRODUCT check: compile script only accepts V2H, but V2N uses the same DRP-AI3
# silicon. Allow both V2H and V2N.
sed -i "s/PRODUCT != 'V2H'/PRODUCT not in ('V2H', 'V2N')/" "$COMPILE_WORK"

# Patch mean/std from ImageNet defaults to sketch normalization (0-1 scaling).
# The compile script ships with mean=[0.485, 0.456, 0.406] / std=[0.229, 0.224, 0.225].
# Quick Draw uses simple pixel/255 scaling: mean=[0, 0, 0], std=[1, 1, 1].
sed -i 's/mean\s*=\s*\[0\.485,\s*0\.456,\s*0\.406\]/mean   = [0.0, 0.0, 0.0]/' "$COMPILE_WORK"
sed -i 's/stdev\s*=\s*\[0\.229,\s*0\.224,\s*0\.225\]/stdev  = [1.0, 1.0, 1.0]/' "$COMPILE_WORK"

echo -e "  ${GREEN}OK${NC} Calibration: direct resize to 128 (no crop)"
echo -e "  ${GREEN}OK${NC} Preprocessing shape_in: [1, 128, 128, 3]"
echo -e "  ${GREEN}OK${NC} PRODUCT check: patched to accept V2N"
echo -e "  ${GREEN}OK${NC} Normalization: sketch (mean=[0,0,0], std=[1,1,1])"

# --- Run compilation ---
echo ""
echo -e "${CYAN}═══════════════════════════════════════════════${NC}"
echo -e "${CYAN}  Starting DRP-AI TVM compilation...${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════${NC}"
echo ""

cd "$TVM_ROOT/tutorials"

python3 "$COMPILE_WORK" \
    "$ONNX_PATH" \
    -o "$OUTPUT_DIR" \
    -s 1,3,128,128 \
    -i image \
    -t "$SDK" \
    -d "$TRANSLATOR" \
    -c "$QUANTIZER" \
    --images "$CALIB_DIR" \
    -n "$NUM_CALIB" \
    -f float32 \
    -p "--calibrate_method Percentile --opts percentile_value 99.99" \
    --mera2

echo ""

# --- Verify output ---
echo -e "${CYAN}Verifying compiled model...${NC}"

OUTPATH="$TVM_ROOT/tutorials/$OUTPUT_DIR"
if [ ! -d "$OUTPATH" ]; then
    echo -e "${RED}ERROR: Output directory not found: $OUTPATH${NC}"
    exit 1
fi

ALL_OK=1

# Check deploy files (may be in root dir or sub_*__CPU_DRP_TVM/ subdirectory)
DEPLOY_DIR="$OUTPATH"
for tvm_dir in "$OUTPATH"/sub_*__CPU_DRP_TVM; do
    if [ -d "$tvm_dir" ]; then
        DEPLOY_DIR="$tvm_dir"
        break
    fi
done

for f in deploy.json deploy.params deploy.so; do
    if [ -f "$DEPLOY_DIR/$f" ]; then
        SIZE=$(du -h "$DEPLOY_DIR/$f" | cut -f1)
        echo -e "  ${GREEN}OK${NC} $f ($SIZE)"
    else
        echo -e "  ${RED}MISSING${NC} $f"
        ALL_OK=0
    fi
done

# Check preprocess directory
if [ -d "$OUTPATH/preprocess" ]; then
    PREPROC_COUNT=$(ls -1 "$OUTPATH/preprocess/" | wc -l)
    echo -e "  ${GREEN}OK${NC} preprocess/ ($PREPROC_COUNT files)"
else
    echo -e "  ${YELLOW}WARN${NC} preprocess/ not found"
fi

# Check mera.plan (optional — runtime works without it via ImplDrpTvm)
if [ -f "$OUTPATH/mera.plan" ]; then
    SIZE=$(du -h "$OUTPATH/mera.plan" | cut -f1)
    echo -e "  ${GREEN}OK${NC} mera.plan ($SIZE)"
else
    echo -e "  ${YELLOW}INFO${NC} No mera.plan — runtime will use DRP-TVM path (works fine)"
fi

echo ""
if [ $ALL_OK -eq 1 ]; then
    echo -e "${GREEN}═══════════════════════════════════════════════${NC}"
    echo -e "${GREEN}  COMPILATION SUCCESSFUL${NC}"
    echo -e "${GREEN}═══════════════════════════════════════════════${NC}"
    echo ""
    echo "Compiled model: $OUTPATH"
    echo ""
    echo "Next steps:"
    echo "  1. Copy compiled model to board_app/drpai_model/ (or ../drpai_model/)"
    echo "  2. Run ./build.sh to cross-compile the application"
    echo "  3. Run ./package.sh to create the deploy package"
    echo "  4. scp -r deploy/ root@<board-ip>:/home/root/"
    echo "  5. ssh root@<board-ip> 'cd /home/root/deploy && ./run.sh'"
else
    echo -e "${RED}═══════════════════════════════════════════════${NC}"
    echo -e "${RED}  COMPILATION INCOMPLETE — check errors above${NC}"
    echo -e "${RED}═══════════════════════════════════════════════${NC}"
    exit 1
fi
