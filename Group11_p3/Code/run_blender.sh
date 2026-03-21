#!/usr/bin/env bash
# =============================================================================
# run_blender.sh
# Invokes Blender headlessly to render one or all sequences.
#
# Usage:
#   bash run_blender.sh Seq1            # render single sequence
#   bash run_blender.sh all             # render all sequences
#   bash run_blender.sh Seq1 --debug    # render with debug overlays
#
# Requirements:
#   - Blender must be on PATH, or set BLENDER_BIN below.
#   - Perception JSONs must already exist in Data/outputs/detections/
# =============================================================================

# set -euo pipefail

# BLENDER_BIN="${BLENDER_BIN:-blender}"
# SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# BLENDER_SCRIPT="$SCRIPT_DIR/blender/scene.py"
# CONFIG="$SCRIPT_DIR/config.yaml"

# SEQ="${1:-}"
# EXTRA_ARGS="${@:2}"   # pass any extra flags through to the Python script

# if [[ -z "$SEQ" ]]; then
#     echo "Usage: bash run_blender.sh <SeqName|all> [--debug]"
#     exit 1
# fi

# run_seq() {
#     local seq="$1"
#     echo "=== Rendering $seq ==="
#     "$BLENDER_BIN" \
#         --background \
#         --python "$BLENDER_SCRIPT" \
#         -- \
#         --seq "$seq" \
#         --config "$CONFIG" \
#         $EXTRA_ARGS
#     echo "=== Done: $seq ==="
# }

# if [[ "$SEQ" == "all" ]]; then
#     # Read sequence list from config.yaml (requires python3 + PyYAML)
#     SEQUENCES=$(python3 -c "
# import yaml, sys
# cfg = yaml.safe_load(open('$CONFIG'))
# print(' '.join(cfg['sequences']))
# ")
#     for s in $SEQUENCES; do
#         run_seq "$s"
#     done
# else
#     run_seq "$SEQ"
# fi
