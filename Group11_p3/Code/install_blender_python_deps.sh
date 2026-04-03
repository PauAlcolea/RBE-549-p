#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Install Python packages into Blender's embedded Python.

Usage:
  ./Code/install_blender_python_deps.sh [--blender /path/to/blender] [package ...]

Examples:
  ./Code/install_blender_python_deps.sh
  ./Code/install_blender_python_deps.sh Pillow numpy
  ./Code/install_blender_python_deps.sh --blender /Applications/Blender.app/Contents/MacOS/Blender Pillow

Notes:
- If no package is provided, Pillow is installed.
- You must re-run this after upgrading Blender (new Blender versions use a new embedded Python).
EOF
}

BLENDER_BIN="${BLENDER_BIN:-}"
PACKAGES=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --blender)
      if [[ $# -lt 2 ]]; then
        echo "ERROR: --blender requires a path" >&2
        exit 1
      fi
      BLENDER_BIN="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      PACKAGES+=("$1")
      shift
      ;;
  esac
done

if [[ ${#PACKAGES[@]} -eq 0 ]]; then
  PACKAGES=(Pillow)
fi

if [[ "$BLENDER_BIN" == "~/"* ]]; then
  BLENDER_BIN="$HOME/${BLENDER_BIN#~/}"
fi

if [[ -n "$BLENDER_BIN" && -d "$BLENDER_BIN" && "$BLENDER_BIN" == *.app ]]; then
  BLENDER_BIN="$BLENDER_BIN/Contents/MacOS/Blender"
fi

if [[ -z "$BLENDER_BIN" ]]; then
  if command -v blender >/dev/null 2>&1; then
    BLENDER_BIN="$(command -v blender)"
  elif [[ -x "$HOME/Applications/Blender.app/Contents/MacOS/Blender" ]]; then
    BLENDER_BIN="$HOME/Applications/Blender.app/Contents/MacOS/Blender"
  elif [[ -x "/Applications/Blender.app/Contents/MacOS/Blender" ]]; then
    BLENDER_BIN="/Applications/Blender.app/Contents/MacOS/Blender"
  fi
fi

if [[ -z "$BLENDER_BIN" || ! -x "$BLENDER_BIN" ]]; then
  echo "ERROR: Blender executable not found." >&2
  echo "Set BLENDER_BIN or pass --blender /path/to/Blender" >&2
  exit 1
fi

echo "[setup] Blender executable: $BLENDER_BIN"

BL_OUT="$($BLENDER_BIN --background --factory-startup --python-expr 'import sys; print("COPILOT_PY_EXE=" + sys.executable)' 2>&1 || true)"

PY_EXE="$(printf '%s\n' "$BL_OUT" | tr -d '\r' | sed -n 's/.*COPILOT_PY_EXE=//p' | tail -n 1)"

if [[ ( -z "$PY_EXE" || ! -x "$PY_EXE" ) && "$BLENDER_BIN" == *.app/Contents/MacOS/Blender ]]; then
  APP_ROOT="${BLENDER_BIN%/Contents/MacOS/Blender}"
  PY_EXE="$(find "$APP_ROOT/Contents/Resources" -type f -path '*/python/bin/python3*' 2>/dev/null | sort | tail -n 1 || true)"
fi

if [[ -z "$PY_EXE" || ! -x "$PY_EXE" ]]; then
  echo "ERROR: Could not resolve Blender Python executable." >&2
  echo "[debug] Blender output:" >&2
  printf '%s\n' "$BL_OUT" >&2
  echo "Try: ./Code/install_blender_python_deps.sh --blender /Applications/Blender.app" >&2
  exit 1
fi

echo "[setup] Blender Python: $PY_EXE"

SITE_PACKAGES="$($PY_EXE -c 'import sysconfig; print(sysconfig.get_paths().get("purelib", ""))' 2>/dev/null || true)"
if [[ -z "$SITE_PACKAGES" ]]; then
  echo "ERROR: Could not resolve Blender site-packages path." >&2
  exit 1
fi

echo "[setup] Blender site-packages: $SITE_PACKAGES"

echo "[setup] Bootstrapping pip in Blender Python"
"$PY_EXE" -m ensurepip --upgrade >/dev/null 2>&1 || true

"$PY_EXE" -m pip install --upgrade pip setuptools wheel
"$PY_EXE" -m pip install --upgrade --target "$SITE_PACKAGES" "${PACKAGES[@]}"

echo "[setup] Installed packages in Blender Python: ${PACKAGES[*]}"

if printf '%s\n' "${PACKAGES[@]}" | grep -qi '^pillow$'; then
  "$PY_EXE" -c 'from PIL import Image; print("[setup] Pillow OK:", Image.__version__)'
fi
