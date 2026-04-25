#!/usr/bin/env bash
set -euo pipefail

BLENDER_BIN="${BLENDER_BIN:-/Applications/Blender.app/Contents/MacOS/Blender}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
GEN_SCRIPT="$SCRIPT_DIR/generate.py"
TEXTURE_CHOICE="playrug"

print_usage() {
	cat <<'EOF'
Usage: ./run_blender.sh [--shape SHAPE] [--height METERS] [--texture NAME]

Options:
	-s, --shape SHAPE   Trajectory shape: square | figure8 | circle
	--height METERS     Camera flight height in meters (e.g. 1.5)
	-t, --texture NAME  Blend texture file to use (e.g. playrug, newyork, ispy, or leaves)
	-h, --help          Show this help message

Environment:
	BLENDER_BIN         Blender executable path override
EOF
}

SHAPE_OVERRIDE=""
HEIGHT_OVERRIDE=""
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
		--height)
			if [[ $# -lt 2 ]]; then
				echo "[run_blender] ERROR: --height requires a value." >&2
				print_usage
				exit 2
			fi
			HEIGHT_OVERRIDE="$2"
			shift 2
			;;
		-t|--texture)
			if [[ $# -lt 2 ]]; then
				echo "[run_blender] ERROR: --texture requires a value." >&2
				print_usage
				exit 2
			fi
			TEXTURE_CHOICE="$2"
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

if [[ "$TEXTURE_CHOICE" == *.blend ]]; then
	BLEND_FILE="$SCRIPT_DIR/$TEXTURE_CHOICE"
else
	BLEND_FILE="$SCRIPT_DIR/${TEXTURE_CHOICE}.blend"
fi

if [[ ! -f "$BLEND_FILE" ]]; then
	echo "[run_blender] ERROR: Blend file not found: $BLEND_FILE" >&2
	echo "[run_blender] Available .blend files in $SCRIPT_DIR:" >&2
	ls "$SCRIPT_DIR"/*.blend 2>/dev/null | xargs -n 1 basename >&2 || true
	exit 2
fi

if [[ -n "$SHAPE_OVERRIDE" ]]; then
	echo "[run_blender] Using trajectory shape override: $SHAPE_OVERRIDE"
fi
if [[ -n "$HEIGHT_OVERRIDE" ]]; then
	echo "[run_blender] Using camera height override: $HEIGHT_OVERRIDE m"
fi
echo "[run_blender] Using texture blend file: $(basename "$BLEND_FILE")"

env_cmd=()
if [[ -n "$SHAPE_OVERRIDE" ]]; then
	env_cmd+=("DRONE_TRAJECTORY_SHAPE=$SHAPE_OVERRIDE")
fi
if [[ -n "$HEIGHT_OVERRIDE" ]]; then
	env_cmd+=("DRONE_CAMERA_HEIGHT=$HEIGHT_OVERRIDE")
fi

if [[ ${#env_cmd[@]} -gt 0 ]]; then
	env "${env_cmd[@]}" "$BLENDER_BIN" "$BLEND_FILE" --background --python "$GEN_SCRIPT"
else
	"$BLENDER_BIN" "$BLEND_FILE" --background --python "$GEN_SCRIPT"
fi
