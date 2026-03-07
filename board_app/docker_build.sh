#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════
# Quick Draw — Build inside Docker (run from HOST)
#
# This script handles the full Docker build workflow:
#   1. Copies source files into the running container
#   2. Builds inside the container with correct user ownership
#   3. Copies compiled binaries back to the host
#
# Prerequisites:
#   - Docker container 'drp-ai_tvm_v2n_container_ahmad' must be running
#   - SDK + TVM_ROOT must be available inside the container
#
# Usage:
#   ./docker_build.sh              # Build everything
#   ./docker_build.sh --clean      # Clean build/ first, then build
# ═══════════════════════════════════════════════════════════════════════

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONTAINER="drp-ai_tvm_v2n_container_ahmad"
CONTAINER_WORK="/tmp/board_app"
HOST_UID=$(id -u)
HOST_GID=$(id -g)

GREEN='\033[0;32m'
RED='\033[0;31m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
NC='\033[0m'

CLEAN=0
for arg in "$@"; do
    case "$arg" in
        --clean|-c) CLEAN=1 ;;
    esac
done

echo -e "${CYAN}═══════════════════════════════════════════════${NC}"
echo -e "${CYAN}  Quick Draw — Docker Build${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════${NC}"
echo ""

# ── Check container is running ──
if ! docker inspect -f '{{.State.Running}}' "$CONTAINER" 2>/dev/null | grep -q true; then
    echo -e "${RED}ERROR: Container '$CONTAINER' is not running.${NC}"
    echo "  Start it with: docker start $CONTAINER"
    exit 1
fi
echo -e "  ${GREEN}OK${NC} Container: $CONTAINER"

# ── Copy source files into container ──
echo -e "\n${CYAN}Copying source files into container...${NC}"
docker exec "$CONTAINER" rm -rf "$CONTAINER_WORK"
docker exec "$CONTAINER" mkdir -p "$CONTAINER_WORK"

# Copy only what's needed (not build/, deploy/, etc.)
for item in src toolchain CMakeLists.txt build.sh config.ini config.json labels.txt; do
    if [ -e "$SCRIPT_DIR/$item" ]; then
        docker cp "$SCRIPT_DIR/$item" "$CONTAINER:$CONTAINER_WORK/$item"
    fi
done
echo -e "  ${GREEN}OK${NC} Source files copied"

# ── Clean if requested ──
if [ $CLEAN -eq 1 ]; then
    echo -e "  ${YELLOW}--clean${NC}: removing previous build artifacts"
    docker exec "$CONTAINER" rm -rf "$CONTAINER_WORK/build"
fi

# ── Build inside container ──
echo -e "\n${CYAN}Building inside container...${NC}"
docker exec "$CONTAINER" bash -c "
    set -e
    cd $CONTAINER_WORK

    # Source SDK environment
    unset LD_LIBRARY_PATH 2>/dev/null
    SDK_ENV=\$(find /opt/ -maxdepth 3 -name 'environment-setup-*poky-linux' 2>/dev/null | head -1)
    if [ -n \"\$SDK_ENV\" ]; then
        source \"\$SDK_ENV\"
    else
        echo 'ERROR: SDK env script not found'
        exit 1
    fi

    # Set TVM_ROOT
    export TVM_ROOT=\"\${TVM_ROOT:-/drp-ai_tvm}\"
    export PRODUCT=\"\${PRODUCT:-V2N}\"

    # Build — always re-run cmake to pick up any file changes
    mkdir -p build && cd build
    cmake -DCMAKE_TOOLCHAIN_FILE=../toolchain/runtime.cmake .. 2>&1
    make -j\$(nproc) 2>&1
"

BUILD_OK=$?
if [ $BUILD_OK -ne 0 ]; then
    echo -e "\n${RED}BUILD FAILED${NC}"
    exit 1
fi

# ── Copy binaries back to host ──
echo -e "\n${CYAN}Copying binaries to host...${NC}"

# Remove old build/ if it exists and is root-owned (from previous Docker builds)
if [ -d "$SCRIPT_DIR/build" ] && [ "$(stat -c '%u' "$SCRIPT_DIR/build" 2>/dev/null)" = "0" ]; then
    echo -e "  ${YELLOW}Removing root-owned build/ via container...${NC}"
    docker run --rm -v "$SCRIPT_DIR:/mnt" alpine rm -rf /mnt/build
fi

mkdir -p "$SCRIPT_DIR/build"

for bin in app_quickdraw app_quickdraw_gui; do
    docker cp "$CONTAINER:$CONTAINER_WORK/build/$bin" "$SCRIPT_DIR/build/$bin" 2>/dev/null && \
        echo -e "  ${GREEN}+${NC} build/$bin ($(du -h "$SCRIPT_DIR/build/$bin" | cut -f1))" || \
        echo -e "  ${YELLOW}SKIP${NC} $bin (not built)"
done

# ── Fix ownership (docker cp creates files as current user, but just in case) ──
chown -R "$HOST_UID:$HOST_GID" "$SCRIPT_DIR/build" 2>/dev/null || true

# ── Cleanup container temp files ──
docker exec "$CONTAINER" rm -rf "$CONTAINER_WORK"

# ── Summary ──
echo ""
GUI_BIN="$SCRIPT_DIR/build/app_quickdraw_gui"
if [ -f "$GUI_BIN" ]; then
    ARCH=$(file "$GUI_BIN" | grep -o 'ARM aarch64\|x86-64\|ELF' | head -1)
    echo -e "${GREEN}═══════════════════════════════════════════════${NC}"
    echo -e "${GREEN}  BUILD COMPLETE${NC}"
    echo -e "${GREEN}═══════════════════════════════════════════════${NC}"
    echo ""
    echo "  app_quickdraw_gui  ($(du -h "$GUI_BIN" | cut -f1), $ARCH)"
    [ -f "$SCRIPT_DIR/build/app_quickdraw" ] && \
        echo "  app_quickdraw      ($(du -h "$SCRIPT_DIR/build/app_quickdraw" | cut -f1))"
    echo ""

    # ── Auto-package deploy/ folder ──
    if [ -f "$SCRIPT_DIR/package.sh" ]; then
        echo -e "${CYAN}Packaging deploy/ folder...${NC}"
        bash "$SCRIPT_DIR/package.sh"
    fi

    echo ""
    echo "Next step:"
    echo "  ./deploy.sh <board-ip> # Deploy to board via scp"
    echo ""
else
    echo -e "${RED}BUILD FAILED — binary not found${NC}"
    exit 1
fi
