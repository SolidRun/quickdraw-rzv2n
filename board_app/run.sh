#!/bin/bash
# Quick Draw — Run on RZ/V2N
#
# Single C++ binary: GTK3 GUI + DRP-AI inference (no socket, no Python)
#
# Settings:
#   config.ini  — DRP-AI parameters, model path, frequencies
#   config.json — UI layout, colors, AI comments
#
# Usage:
#   ./run.sh                        # Default config.ini
#   ./run.sh --config alt.ini       # Alternate config

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
APP="$SCRIPT_DIR/app_quickdraw_gui"
CONFIG_INI="$SCRIPT_DIR/config.ini"

# Allow --config override
for arg in "$@"; do
    case "$prev" in
        --config|-c) CONFIG_INI="$arg" ;;
    esac
    prev="$arg"
done

RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
NC='\033[0m'

# ── Check root (DRP-AI requires root) ──
if [ "$(id -u)" -ne 0 ]; then
    echo -e "${RED}DRP-AI requires root. Re-running with su...${NC}"
    exec su -c "\"$0\" $@"
fi

# ── Check binary ──
if [ ! -f "$APP" ]; then
    echo -e "${RED}ERROR: app_quickdraw_gui not found at: $APP${NC}"
    exit 1
fi
if [ ! -f "$CONFIG_INI" ]; then
    echo -e "${RED}ERROR: config.ini not found at: $CONFIG_INI${NC}"
    exit 1
fi

# ── Install runtime libraries ──
if [ -d "$SCRIPT_DIR/lib" ]; then
    LIBS_INSTALLED=0
    for lib in "$SCRIPT_DIR/lib"/*.so*; do
        [ -f "$lib" ] || continue
        LIBNAME="$(basename "$lib")"
        if [ ! -f "/usr/lib64/$LIBNAME" ]; then
            cp "$lib" /usr/lib64/
            LIBS_INSTALLED=$((LIBS_INSTALLED + 1))
        fi
    done
    if [ $LIBS_INSTALLED -gt 0 ]; then
        ldconfig 2>/dev/null || true
        echo -e "${GREEN}Installed $LIBS_INSTALLED runtime libraries${NC}"
    fi
fi
export LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH

# ── Auto-detect Wayland (handle root user) ──
WAYLAND_FOUND=0
for udir in /run/user/* /run /tmp; do
    [ -d "$udir" ] || continue
    for sock in "$udir"/wayland-*; do
        case "$sock" in *.lock) continue;; esac
        if [ -S "$sock" ]; then
            WAYLAND_SOCK="$sock"
            WAYLAND_NAME="$(basename "$sock")"
            WAYLAND_DIR="$udir"
            WAYLAND_FOUND=1
            break 2
        fi
    done
done

if [ $WAYLAND_FOUND -eq 1 ]; then
    SOCK_OWNER=$(stat -c '%u' "$WAYLAND_DIR" 2>/dev/null || echo "unknown")
    if [ "$(id -u)" -eq 0 ] && [ "$SOCK_OWNER" != "0" ]; then
        mkdir -p /run/user/0
        ln -sf "$WAYLAND_SOCK" "/run/user/0/$WAYLAND_NAME"
        [ -f "${WAYLAND_SOCK}.lock" ] && ln -sf "${WAYLAND_SOCK}.lock" "/run/user/0/${WAYLAND_NAME}.lock"
        export XDG_RUNTIME_DIR=/run/user/0
        export WAYLAND_DISPLAY="$WAYLAND_NAME"
        echo -e "${GREEN}Wayland: /run/user/0/$WAYLAND_NAME (symlink from $WAYLAND_SOCK)${NC}"
    else
        export XDG_RUNTIME_DIR="$WAYLAND_DIR"
        export WAYLAND_DISPLAY="$WAYLAND_NAME"
        echo -e "${GREEN}Wayland: $WAYLAND_DIR/$WAYLAND_NAME${NC}"
    fi
fi

# ── Banner ──
echo ""
echo -e "${CYAN}════════════════════════════════════════════${NC}"
echo -e "${CYAN}  Quick Draw — C++ GTK3 + DRP-AI3 (RZ/V2N)${NC}"
echo -e "${CYAN}════════════════════════════════════════════${NC}"
echo -e "  Config: $CONFIG_INI"
echo -e "  Binary: $APP"
echo -e "${CYAN}────────────────────────────────────────────${NC}"
echo ""

# ── Run ──
cd "$SCRIPT_DIR"
echo "Press Ctrl+C or close window to stop."
echo ""
exec "$APP" --config "$CONFIG_INI"
