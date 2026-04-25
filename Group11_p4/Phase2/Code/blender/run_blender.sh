#!/usr/bin/env bash
set -euo pipefail

BLENDER_BIN="${BLENDER_BIN:-/Applications/Blender.app/Contents/MacOS/Blender}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
GEN_SCRIPT="$SCRIPT_DIR/generate.py"
BLEND_FILE="$SCRIPT_DIR/data_gen.blend"

print_usage() {
	cat <<'EOF'
Usage: ./run_blender.sh [--shape SHAPE]

Options:
	-s, --shape SHAPE   Trajectory shape: square | figure8 | circle
	-h, --help          Show this help message

Environment:
	BLENDER_BIN         Blender executable path override
EOF
}

SHAPE_OVERRIDE=""
while [[ $# -gt 0 ]]; do
	case "$1" in
		-s|--shape)
			if [[ $# -lt 2 ]]; then
				echo "[run_blender] ERROR: --shape requires a value." >&2
				print_usage
				exit 2
			fi
			SHAPE_OVERRIDE="$2"
			shift 2
			;;
		-h|--help)
			print_usage
			exit 0
			;;
		*)
			echo "[run_blender] ERROR: Unknown argument: $1" >&2
			print_usage
			exit 2
			;;
	esac
done

if [[ -n "$SHAPE_OVERRIDE" ]]; then
	echo "[run_blender] Using trajectory shape override: $SHAPE_OVERRIDE"
	DRONE_TRAJECTORY_SHAPE="$SHAPE_OVERRIDE" \
		"$BLENDER_BIN" "$BLEND_FILE" --background --python "$GEN_SCRIPT"
else
	"$BLENDER_BIN" "$BLEND_FILE" --background --python "$GEN_SCRIPT"
fi
