#!/bin/bash
# Build Quick Draw DRP-AI server for RZ/V2N
# Run inside the DRP-AI TVM Docker container.
#
# Only builds the C++ inference server (app_quickdraw).
# The GUI is Python (quickdraw_gui.py) — no compilation needed.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

GREEN='\033[0;32m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}═══════════════════════════════════════════════${NC}"
echo -e "${CYAN}  Quick Draw — Build DRP-AI Server${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════${NC}"
echo ""

# --- Source SDK environment ---
unset LD_LIBRARY_PATH 2>/dev/null
if [ -z "$SDKTARGETSYSROOT" ]; then
    SDK_ENV=$(find /opt/ -maxdepth 3 -name "environment-setup-*poky-linux" 2>/dev/null | head -1)
    if [ -n "$SDK_ENV" ]; then
        echo "Sourcing SDK: $SDK_ENV"
        source "$SDK_ENV"
    else
        echo -e "${RED}ERROR: SDK env script not found under /opt/${NC}"
        exit 1
    fi
fi
echo "Sysroot: $SDKTARGETSYSROOT"

# --- Set SDK path for toolchain cmake ---
if [ -z "$SDK" ] && [ -n "$SDKTARGETSYSROOT" ]; then
    export SDK="$(dirname "$(dirname "$SDKTARGETSYSROOT")")"
fi
export SDK="${SDK:-/opt/rz-vlp/5.0.6}"
echo "SDK: $SDK"

# --- Resolve TVM_ROOT ---
export TVM_ROOT="${TVM_ROOT:-/drp-ai_tvm}"
if [ ! -d "$TVM_ROOT" ]; then
    echo -e "${RED}ERROR: TVM_ROOT not found: $TVM_ROOT${NC}"
    exit 1
fi
echo "TVM_ROOT: $TVM_ROOT"

export PRODUCT="${PRODUCT:-V2N}"
echo "PRODUCT: $PRODUCT"
echo ""

# --- Build ---
echo "Building app_quickdraw + app_quickdraw_gui..."
cd "$SCRIPT_DIR"
rm -rf build
mkdir -p build
cd build
cmake -DCMAKE_TOOLCHAIN_FILE=../toolchain/runtime.cmake ..
make -j$(nproc)

GUI_BIN="$SCRIPT_DIR/build/app_quickdraw_gui"
SERVER_BIN="$SCRIPT_DIR/build/app_quickdraw"

if [ -f "$GUI_BIN" ]; then
    SIZE=$(du -h "$GUI_BIN" | cut -f1)
    ARCH=$(file "$GUI_BIN" | grep -o 'ARM aarch64\|x86-64\|ELF' | head -1)
    echo ""
    echo -e "${GREEN}═══════════════════════════════════════════════${NC}"
    echo -e "${GREEN}  BUILD COMPLETE${NC}"
    echo -e "${GREEN}═══════════════════════════════════════════════${NC}"
    echo ""
    echo "  app_quickdraw_gui  ($SIZE, $ARCH) — C++ GTK3 + DRP-AI"
    [ -f "$SERVER_BIN" ] && echo "  app_quickdraw      ($(du -h "$SERVER_BIN" | cut -f1)) — socket server (legacy)"
    echo ""
    echo "Next steps:"
    echo "  1. Package:  ./package.sh"
    echo "  2. Deploy:   scp -r deploy/ root@<board-ip>:/home/root/quickdraw"
    echo "  3. Run:      ssh root@<board-ip> 'cd quickdraw && ./run.sh'"
    echo ""
else
    echo ""
    echo -e "${RED}═══════════════════════════════════════════════${NC}"
    echo -e "${RED}  BUILD FAILED — check errors above${NC}"
    echo -e "${RED}═══════════════════════════════════════════════${NC}"
    exit 1
fi
