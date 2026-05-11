#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════
# Quick Draw — Package for Deployment to RZ/V2N Board
#
# Creates a self-contained deploy/ directory with everything needed
# to run the Quick Draw app on the V2N board.
#
# Prerequisites:
#   - build/app_quickdraw must exist (run build.sh first)
#   - Compiled DRP-AI model must exist (run compile_model.sh first)
#
# Usage:
#   ./package.sh                                    # Defaults
#   ./package.sh --model-dir qd_mobilenetv2         # Custom model name
#   ./package.sh --compiled /path/to/compiled       # Custom compiled model path
# ═══════════════════════════════════════════════════════════════════════

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'

# --- Defaults ---
MODEL_NAME="qd_mobilenetv2"
COMPILED_MODEL=""
DEPLOY_DIR="$SCRIPT_DIR/deploy"
BINARY="$SCRIPT_DIR/build/app_quickdraw_gui"
GUI_SCRIPT="$SCRIPT_DIR/quickdraw_gui.py"

# --- Parse arguments ---
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model-dir)   MODEL_NAME="$2"; shift 2 ;;
        --compiled)    COMPILED_MODEL="$2"; shift 2 ;;
        --output)      DEPLOY_DIR="$2"; shift 2 ;;
        --binary)      BINARY="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "  --model-dir NAME    Model subdirectory name (default: qd_mobilenetv2)"
            echo "  --compiled PATH     Path to compiled DRP-AI model directory"
            echo "  --output PATH       Output deploy directory (default: ./deploy)"
            echo "  --binary PATH       Path to compiled binary (default: ./build/app_quickdraw)"
            exit 0
            ;;
        *) echo -e "${RED}Unknown option: $1${NC}"; exit 1 ;;
    esac
done

# --- Auto-detect compiled model location ---
if [ -z "$COMPILED_MODEL" ]; then
    # Check common locations (Docker output, local drpai_model)
    TVM_ROOT="${TVM_ROOT:-/drp-ai_tvm}"
    for candidate in \
        "$TVM_ROOT/tutorials/$MODEL_NAME" \
        "$SCRIPT_DIR/../drpai_model" \
        "$SCRIPT_DIR/drpai_model"; do
        if [ -d "$candidate" ] && [ -f "$candidate/deploy.so" -o -d "$candidate/sub_"*"__CPU_DRP_TVM" 2>/dev/null ]; then
            COMPILED_MODEL="$candidate"
            break
        fi
    done
fi

echo -e "${CYAN}═══════════════════════════════════════════════${NC}"
echo -e "${CYAN}  Quick Draw — Packaging for V2N Deployment${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════${NC}"
echo ""

# --- Verify binaries ---
if [ ! -f "$BINARY" ]; then
    echo -e "${RED}ERROR: GUI binary not found: $BINARY${NC}"
    echo -e "${RED}  Run ./build.sh first${NC}"
    exit 1
fi
echo -e "  ${GREEN}OK${NC} Binary: $BINARY"

# --- Verify compiled model ---
if [ -z "$COMPILED_MODEL" ] || [ ! -d "$COMPILED_MODEL" ]; then
    echo -e "${RED}ERROR: Compiled model not found.${NC}"
    echo -e "${RED}  Run ./compile_model.sh first, or specify --compiled PATH${NC}"
    exit 1
fi

# Find deploy files (may be in root or sub_*__CPU_DRP_TVM/)
MODEL_DEPLOY_DIR="$COMPILED_MODEL"
for tvm_dir in "$COMPILED_MODEL"/sub_*__CPU_DRP_TVM; do
    if [ -d "$tvm_dir" ]; then
        MODEL_DEPLOY_DIR="$tvm_dir"
        break
    fi
done

if [ ! -f "$MODEL_DEPLOY_DIR/deploy.so" ]; then
    echo -e "${RED}ERROR: deploy.so not found in $MODEL_DEPLOY_DIR${NC}"
    exit 1
fi
echo -e "  ${GREEN}OK${NC} Compiled model: $COMPILED_MODEL"

# --- Create deploy directory ---
echo ""
echo -e "${CYAN}Creating deploy package...${NC}"
rm -rf "$DEPLOY_DIR"
mkdir -p "$DEPLOY_DIR/model/$MODEL_NAME"

# --- Copy binary ---
cp "$BINARY" "$DEPLOY_DIR/app_quickdraw_gui"
chmod +x "$DEPLOY_DIR/app_quickdraw_gui"
echo -e "  ${GREEN}+${NC} app_quickdraw_gui ($(du -h "$DEPLOY_DIR/app_quickdraw_gui" | cut -f1))"

# --- Copy compiled model ---
# Copy deploy files preserving sub_*__CPU_DRP_TVM directory structure
# (mera.plan references this path — flattening breaks MERA2 runtime)
SUB_DIR_NAME=$(basename "$MODEL_DEPLOY_DIR")
if [ "$SUB_DIR_NAME" != "$(basename "$COMPILED_MODEL")" ]; then
    # Deploy files are inside a sub directory (e.g. sub_0000__CPU_DRP_TVM)
    mkdir -p "$DEPLOY_DIR/model/$MODEL_NAME/$SUB_DIR_NAME"
    for f in deploy.json deploy.params deploy.so; do
        if [ -f "$MODEL_DEPLOY_DIR/$f" ]; then
            cp "$MODEL_DEPLOY_DIR/$f" "$DEPLOY_DIR/model/$MODEL_NAME/$SUB_DIR_NAME/"
            echo -e "  ${GREEN}+${NC} model/$MODEL_NAME/$SUB_DIR_NAME/$f ($(du -h "$MODEL_DEPLOY_DIR/$f" | cut -f1))"
        fi
    done
else
    # Deploy files are directly in model root
    for f in deploy.json deploy.params deploy.so; do
        if [ -f "$MODEL_DEPLOY_DIR/$f" ]; then
            cp "$MODEL_DEPLOY_DIR/$f" "$DEPLOY_DIR/model/$MODEL_NAME/"
            echo -e "  ${GREEN}+${NC} model/$MODEL_NAME/$f ($(du -h "$MODEL_DEPLOY_DIR/$f" | cut -f1))"
        fi
    done
fi

# Copy preprocess directory
if [ -d "$COMPILED_MODEL/preprocess" ]; then
    cp -r "$COMPILED_MODEL/preprocess" "$DEPLOY_DIR/model/$MODEL_NAME/"
    PREPROC_COUNT=$(ls -1 "$DEPLOY_DIR/model/$MODEL_NAME/preprocess/" | wc -l)
    echo -e "  ${GREEN}+${NC} model/$MODEL_NAME/preprocess/ ($PREPROC_COUNT files)"
fi

# Copy mera.plan if it exists
if [ -f "$COMPILED_MODEL/mera.plan" ]; then
    cp "$COMPILED_MODEL/mera.plan" "$DEPLOY_DIR/model/$MODEL_NAME/"
    echo -e "  ${GREEN}+${NC} model/$MODEL_NAME/mera.plan"
fi

# --- Copy application files ---
for f in config.ini config.json labels.txt run.sh solidrun_logo.png; do
    if [ -f "$SCRIPT_DIR/$f" ]; then
        cp "$SCRIPT_DIR/$f" "$DEPLOY_DIR/"
        echo -e "  ${GREEN}+${NC} $f"
    else
        echo -e "  ${YELLOW}SKIP${NC} $f (not found)"
    fi
done
chmod +x "$DEPLOY_DIR/run.sh" 2>/dev/null || true

# --- Copy runtime libraries (auto-pull from Docker if missing) ---
HOST_LIB_DIR="$SCRIPT_DIR/lib"
if [ ! -d "$HOST_LIB_DIR" ] || [ -z "$(ls -A "$HOST_LIB_DIR"/*.so* 2>/dev/null)" ]; then
    echo -e "  ${YELLOW}lib/ missing — auto-pulling from Docker container...${NC}"
    CONTAINER="${CONTAINER:-drp-ai_tvm_v2n_container_$(whoami)}"
    CONTAINER_LIB_DIR="/drp-ai_tvm/obj/build_runtime/v2h/lib"
    docker start "$CONTAINER" >/dev/null 2>&1 || true
    if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER}$"; then
        mkdir -p "$HOST_LIB_DIR"
        if docker cp "${CONTAINER}:${CONTAINER_LIB_DIR}/." "$HOST_LIB_DIR/" 2>/dev/null; then
            PULLED=$(ls -1 "$HOST_LIB_DIR"/*.so* 2>/dev/null | wc -l)
            echo -e "  ${GREEN}+${NC} Pulled $PULLED runtime libs from $CONTAINER:$CONTAINER_LIB_DIR"
        else
            echo -e "  ${RED}FAIL${NC} docker cp from $CONTAINER:$CONTAINER_LIB_DIR failed"
        fi
    else
        echo -e "  ${RED}FAIL${NC} Container $CONTAINER not found — start it or set CONTAINER=<name>"
    fi
fi

if [ -d "$HOST_LIB_DIR" ] && [ -n "$(ls -A "$HOST_LIB_DIR"/*.so* 2>/dev/null)" ]; then
    mkdir -p "$DEPLOY_DIR/lib"
    cp "$HOST_LIB_DIR"/*.so* "$DEPLOY_DIR/lib/" 2>/dev/null || true
    cp "$HOST_LIB_DIR"/*.bin "$DEPLOY_DIR/lib/" 2>/dev/null || true
    LIB_COUNT=$(ls -1 "$DEPLOY_DIR/lib/" 2>/dev/null | wc -l)
    echo -e "  ${GREEN}+${NC} lib/ ($LIB_COUNT runtime files)"
else
    echo -e "  ${RED}MISSING${NC} lib/ — deploy will fail on board without runtime libraries"
fi

# --- Summary ---
echo ""
TOTAL_SIZE=$(du -sh "$DEPLOY_DIR" | cut -f1)
echo -e "${GREEN}═══════════════════════════════════════════════${NC}"
echo -e "${GREEN}  PACKAGING COMPLETE${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════${NC}"
echo ""
echo "Deploy directory: $DEPLOY_DIR ($TOTAL_SIZE)"
echo ""
echo "Contents:"
echo "  app_quickdraw_gui      — C++ GTK3 + DRP-AI (single binary, ARM64)"
echo "  model/$MODEL_NAME/     — Compiled INT8 DRP-AI model"
echo "  config.ini             — DRP-AI parameters (freq, model path, input size)"
echo "  config.json            — UI layout, colors, AI comments"
echo "  labels.txt             — 345 class names"
echo "  run.sh                 — Board startup script"
echo "  lib/                   — MERA2 runtime libraries"
echo ""
echo "Deploy to board:"
echo "  scp -r $DEPLOY_DIR/ root@<board-ip>:/home/root/quickdraw"
echo "  ssh root@<board-ip> 'cd /home/root/quickdraw && ./run.sh'"
echo ""
