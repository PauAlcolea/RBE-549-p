#!/usr/bin/env bash
# =============================================================================
# run_blender.sh
# Invokes Blender headlessly to render one scene+camera pair, or all of them.
#
# Usage (from the Code/ directory):
#   bash run_blender.sh --scene scene1 --cam front
#   bash run_blender.sh --scene scene1 --allcam
#   bash run_blender.sh --all --cam front
#   bash run_blender.sh --all --allcam
#   bash run_blender.sh --scene scene1 --cam front --debug
#
# Requirements:
#   - Blender must be on PATH, or set BLENDER_BIN env var:
#       export BLENDER_BIN=/Applications/Blender.app/Contents/MacOS/Blender
#   - Perception JSONs must already exist in Outputs/Detections/
# =============================================================================
set -euo pipefail

BLENDER_BIN="${BLENDER_BIN:-blender}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BLENDER_SCRIPT="$SCRIPT_DIR/blender/scene.py"
CONFIG="$SCRIPT_DIR/config.yaml"

# Keep Blender's JSON sidecar in sync with YAML edits.
# Blender's Python may not have PyYAML and can read config.json instead.
python3 - <<PY
from pathlib import Path
import sys

code_dir = Path("$SCRIPT_DIR")
cfg_path = Path("$CONFIG")
sys.path.insert(0, str(code_dir))

from utils.io_utils import load_config
load_config(str(cfg_path))
print(f"[run_blender] Synced config sidecar: {cfg_path.with_suffix('.json')}")
PY

# Resolve Blender executable if not on PATH.
if [[ "$BLENDER_BIN" == "~/"* ]]; then
    BLENDER_BIN="$HOME/${BLENDER_BIN#~/}"
fi

if [[ -d "$BLENDER_BIN" && "$BLENDER_BIN" == *.app ]]; then
    BLENDER_BIN="$BLENDER_BIN/Contents/MacOS/Blender"
fi

if [[ "$BLENDER_BIN" == "blender" ]] && ! command -v blender >/dev/null 2>&1; then
    if [[ -x "$HOME/Applications/Blender.app/Contents/MacOS/Blender" ]]; then
        BLENDER_BIN="$HOME/Applications/Blender.app/Contents/MacOS/Blender"
    elif [[ -x "/Applications/Blender.app/Contents/MacOS/Blender" ]]; then
        BLENDER_BIN="/Applications/Blender.app/Contents/MacOS/Blender"
    fi
fi

if [[ ! -x "$BLENDER_BIN" ]]; then
    echo "ERROR: Blender executable not found: $BLENDER_BIN" >&2
    echo "Set BLENDER_BIN, e.g.:" >&2
    echo "  export BLENDER_BIN=/Applications/Blender.app/Contents/MacOS/Blender" >&2
    echo "or run with:" >&2
    echo "  BLENDER_BIN=/Applications/Blender.app/Contents/MacOS/Blender bash run_blender.sh --scene scene2 --cam front --debug" >&2
    exit 127
fi

# ── Argument parsing ──────────────────────────────────────────────────────────
SCENE=""
CAM=""
ALL_SCENES=false
ALL_CAMS=false
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --scene)   SCENE="$2";    shift 2 ;;
        --cam)     CAM="$2";      shift 2 ;;
        --all)     ALL_SCENES=true; shift ;;
        --allcam)  ALL_CAMS=true;   shift ;;
        --debug)   EXTRA_ARGS="$EXTRA_ARGS --debug"; shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ -z "$SCENE" && "$ALL_SCENES" == false ]]; then
    echo "Usage: bash run_blender.sh (--scene <name> | --all) (--cam <name> | --allcam) [--debug]"
    exit 1
fi
if [[ -z "$CAM" && "$ALL_CAMS" == false ]]; then
    echo "Usage: bash run_blender.sh (--scene <name> | --all) (--cam <name> | --allcam) [--debug]"
    exit 1
fi

# ── Resolve scene and camera lists from config if --all / --allcam ────────────
if $ALL_SCENES; then
    SCENES=$(python3 -c "
import yaml
cfg = yaml.safe_load(open('$CONFIG'))
print(' '.join(cfg['sequences']))
")
else
    SCENES="$SCENE"
fi

if $ALL_CAMS; then
    CAMERAS=$(python3 -c "
import yaml
cfg = yaml.safe_load(open('$CONFIG'))
print(' '.join(cfg['cameras']))
")
else
    CAMERAS="$CAM"
fi

# ── Runner ────────────────────────────────────────────────────────────────────
run_one() {
    local scene="$1"
    local cam="$2"
    echo "=== Rendering $scene / $cam ==="
    "$BLENDER_BIN" \
        --background \
        --python "$BLENDER_SCRIPT" \
        -- \
        --scene "$scene" \
        --cam   "$cam" \
        --config "$CONFIG" \
        $EXTRA_ARGS
    echo "=== Done: $scene / $cam ==="
}

for s in $SCENES; do
    for c in $CAMERAS; do
        run_one "$s" "$c"
    done
done
