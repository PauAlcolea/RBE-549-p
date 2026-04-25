#!/usr/bin/env bash
set -euo pipefail

BLENDER_BIN="${BLENDER_BIN:-/Applications/Blender.app/Contents/MacOS/Blender}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
GEN_SCRIPT="$SCRIPT_DIR/generate.py"
BASE_BLEND="$SCRIPT_DIR/data_gen.blend"
TEXTURES_DIR="$SCRIPT_DIR/textures"
TEXTURE_CHOICE="playrug"

print_usage() {
	cat <<'EOF'
Usage: ./run_blender.sh [--shape SHAPE] [--height METERS] [--texture NAME]

Options:
	-s, --shape SHAPE   Trajectory shape: square | figure8 | circle
	--height METERS     Camera flight height in meters (e.g. 1.5)
	-t, --texture NAME  Texture image in textures/ (e.g. playrug, newyork, ispy, leaves, or toys)
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

if [[ ! -f "$BASE_BLEND" ]]; then
	echo "[run_blender] ERROR: Base blend file not found: $BASE_BLEND" >&2
	exit 2
fi

if [[ ! -d "$TEXTURES_DIR" ]]; then
	echo "[run_blender] ERROR: Textures directory not found: $TEXTURES_DIR" >&2
	exit 2
fi

TEXTURE_FILE=""
if [[ "$TEXTURE_CHOICE" == *.* ]]; then
	if [[ -f "$TEXTURES_DIR/$TEXTURE_CHOICE" ]]; then
		TEXTURE_FILE="$TEXTURES_DIR/$TEXTURE_CHOICE"
	fi
else
	for ext in png jpg jpeg webp tif tiff; do
		candidate="$TEXTURES_DIR/${TEXTURE_CHOICE}.${ext}"
		if [[ -f "$candidate" ]]; then
			TEXTURE_FILE="$candidate"
			break
		fi
	done
fi

if [[ -z "$TEXTURE_FILE" ]]; then
	echo "[run_blender] ERROR: Texture image not found for: $TEXTURE_CHOICE" >&2
	echo "[run_blender] Available textures in $TEXTURES_DIR:" >&2
	ls "$TEXTURES_DIR"/*.{png,jpg,jpeg,webp,tif,tiff} 2>/dev/null | xargs -n 1 basename >&2 || true
	exit 2
fi

if [[ -n "$SHAPE_OVERRIDE" ]]; then
	echo "[run_blender] Using trajectory shape override: $SHAPE_OVERRIDE"
fi
if [[ -n "$HEIGHT_OVERRIDE" ]]; then
	echo "[run_blender] Using camera height override: $HEIGHT_OVERRIDE m"
fi
echo "[run_blender] Using base blend file: $(basename "$BASE_BLEND")"
echo "[run_blender] Using texture image: $(basename "$TEXTURE_FILE")"

env_cmd=()
if [[ -n "$SHAPE_OVERRIDE" ]]; then
	env_cmd+=("DRONE_TRAJECTORY_SHAPE=$SHAPE_OVERRIDE")
fi
if [[ -n "$HEIGHT_OVERRIDE" ]]; then
	env_cmd+=("DRONE_CAMERA_HEIGHT=$HEIGHT_OVERRIDE")
fi
env_cmd+=("DRONE_TEXTURE_NAME=${TEXTURE_CHOICE%.blend}")
env_cmd+=("DRONE_TEXTURE_FILE=$TEXTURE_FILE")

if [[ ${#env_cmd[@]} -gt 0 ]]; then
	env "${env_cmd[@]}" "$BLENDER_BIN" "$BASE_BLEND" --background --python "$GEN_SCRIPT"
else
	"$BLENDER_BIN" "$BASE_BLEND" --background --python "$GEN_SCRIPT"
fi
