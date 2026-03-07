#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════
# Quick Draw — Deploy to RZ/V2N Board
#
# Full workflow: package + scp to board + optional run
#
# Usage:
#   ./deploy.sh <board-ip>                    # Package + deploy
#   ./deploy.sh <board-ip> --run              # Package + deploy + run
#   ./deploy.sh <board-ip> --no-package       # Skip packaging, deploy existing deploy/
#   ./deploy.sh <board-ip> --build            # Build + package + deploy
#   ./deploy.sh <board-ip> --build --run      # Full pipeline: build + package + deploy + run
#
# Board path: /home/root/quickdraw (default)
# ═══════════════════════════════════════════════════════════════════════

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEPLOY_DIR="$SCRIPT_DIR/deploy"
BOARD_PATH="/home/root/quickdraw"
BOARD_USER="root"

GREEN='\033[0;32m'
RED='\033[0;31m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
NC='\033[0m'

BOARD_IP=""
DO_BUILD=0
DO_PACKAGE=1
DO_RUN=0

# ── Parse arguments ──
for arg in "$@"; do
    case "$arg" in
        --build|-b)      DO_BUILD=1 ;;
        --run|-r)        DO_RUN=1 ;;
        --no-package)    DO_PACKAGE=0 ;;
        --help|-h)
            echo "Usage: $0 <board-ip> [OPTIONS]"
            echo ""
            echo "  --build, -b       Build inside Docker before packaging"
            echo "  --run, -r         Run the app on the board after deploy"
            echo "  --no-package      Skip packaging, deploy existing deploy/"
            echo "  --help, -h        Show this help"
            echo ""
            echo "Examples:"
            echo "  $0 192.168.1.100               # Package + deploy"
            echo "  $0 192.168.1.100 --build --run # Full pipeline"
            exit 0
            ;;
        -*)
            echo -e "${RED}Unknown option: $arg${NC}"
            exit 1
            ;;
        *)
            # First positional argument is the board IP
            if [ -z "$BOARD_IP" ]; then
                BOARD_IP="$arg"
            fi
            ;;
    esac
done

if [ -z "$BOARD_IP" ]; then
    echo -e "${RED}Usage: $0 <board-ip> [--build] [--run] [--no-package]${NC}"
    echo ""
    echo "  Example: $0 192.168.1.100 --build --run"
    exit 1
fi

echo -e "${CYAN}═══════════════════════════════════════════════${NC}"
echo -e "${CYAN}  Quick Draw — Deploy to RZ/V2N${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════${NC}"
echo -e "  Board: ${BOARD_USER}@${BOARD_IP}:${BOARD_PATH}"
echo ""

# ── Step 1: Build (optional) ──
if [ $DO_BUILD -eq 1 ]; then
    echo -e "${CYAN}[1/3] Building...${NC}"
    "$SCRIPT_DIR/docker_build.sh" --clean
    echo ""
fi

# ── Step 2: Package ──
if [ $DO_PACKAGE -eq 1 ]; then
    echo -e "${CYAN}[2/3] Packaging...${NC}"
    "$SCRIPT_DIR/package.sh"
    echo ""
fi

# Verify deploy/ exists
if [ ! -d "$DEPLOY_DIR" ]; then
    echo -e "${RED}ERROR: deploy/ directory not found. Run ./package.sh first.${NC}"
    exit 1
fi
if [ ! -f "$DEPLOY_DIR/app_quickdraw_gui" ]; then
    echo -e "${RED}ERROR: app_quickdraw_gui not in deploy/. Run ./package.sh first.${NC}"
    exit 1
fi

# ── Step 3: Deploy via scp ──
echo -e "${CYAN}[3/3] Deploying to board...${NC}"
DEPLOY_SIZE=$(du -sh "$DEPLOY_DIR" | cut -f1)
echo -e "  Uploading $DEPLOY_SIZE to ${BOARD_USER}@${BOARD_IP}:${BOARD_PATH}"

# Create target directory and copy files
ssh "${BOARD_USER}@${BOARD_IP}" "mkdir -p ${BOARD_PATH}" 2>/dev/null || true
scp -r "$DEPLOY_DIR/"* "${BOARD_USER}@${BOARD_IP}:${BOARD_PATH}/"

echo ""
echo -e "${GREEN}═══════════════════════════════════════════════${NC}"
echo -e "${GREEN}  DEPLOY COMPLETE${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════${NC}"
echo ""
echo "  Deployed to: ${BOARD_USER}@${BOARD_IP}:${BOARD_PATH}"
echo ""

# ── Step 4: Run (optional) ──
if [ $DO_RUN -eq 1 ]; then
    echo -e "${CYAN}Starting app on board...${NC}"
    echo ""
    ssh -t "${BOARD_USER}@${BOARD_IP}" "cd ${BOARD_PATH} && ./run.sh"
else
    echo "To run on board:"
    echo "  ssh ${BOARD_USER}@${BOARD_IP} 'cd ${BOARD_PATH} && ./run.sh'"
    echo ""
fi
