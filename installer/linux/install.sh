#!/usr/bin/env bash
# VOXPrint AI Model Forge: Linux install from a source checkout, per user.
# Creates a venv at ~/.local/share/voxprint-ai-suite/venv (no root, nothing in /opt),
# installs the vendored voxprint-common wheel, then this checkout with the [gui] extra
# (the voxprint-model-forge window app), then the MiVOLO age estimator for the LoRA
# safety gate from its pinned commit. MiVOLO's own extra pins are not needed for the
# classifier and stay uninstalled (--no-deps). Writes a GNOME/freedesktop menu entry
# with the app icon, so the app shows up in the desktop menu.
# Re-running reuses the venv and overwrites the menu entry (idempotent).
# Set VOXPRINT_INSTALL_DIR to choose another install root.
# This script does not enable command tracing, does not unset tokens, and does not start the GUI.
set +x
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PREFIX="${VOXPRINT_INSTALL_DIR:-$HOME/.local/share/voxprint-ai-suite}"
VENV="$PREFIX/venv"
PY="$VENV/bin/python"
PIP=("$PY" -m pip)

# The only wheel filename that exists in vendor/ (see pyproject.toml pin voxprint-common==0.7.0).
COMMON_WHEEL="$ROOT/vendor/voxprint_common-0.7.0-py3-none-any.whl"
ICON_SRC="$ROOT/assets/icon/frames/256.png"
ICON_FILE="$HOME/.local/share/icons/hicolor/256x256/apps/voxprint-model-forge.png"
DESKTOP_FILE="$HOME/.local/share/applications/voxprint-model-forge.desktop"
MIVOLO_COMMIT="37475e3f8818b5f22448003feec3e64b01bfb188"

log() { printf 'install: %s\n' "$*"; }
die() { printf 'install: error: %s\n' "$*" >&2; exit 1; }

[[ -f "$COMMON_WHEEL" ]] || die "vendored wheel not found: $COMMON_WHEEL"
[[ -f "$ICON_SRC" ]] || die "app icon not found: $ICON_SRC"
command -v python3 >/dev/null 2>&1 || die "python3 is not on PATH"
python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 14) else 1)' \
  || die "Python 3.14 or newer is required (pyproject requires-python); found: $(python3 --version 2>&1)"

if [[ ! -x "$PY" ]]; then
  log "creating venv at $VENV"
  python3 -m venv "$VENV" || die "could not create the venv (on Debian/Ubuntu: sudo apt install python3-venv)"
fi

log "venv: $VENV"
"${PIP[@]}" install --quiet --upgrade pip

log "installing vendored voxprint-common wheel"
"${PIP[@]}" install "$COMMON_WHEEL"

log "installing this checkout with the [gui] extra (voxprint-model-forge window app)"
"${PIP[@]}" install "$ROOT[gui]"

# MiVOLO (Apache-2.0) is not on PyPI. Its setup.py imports pkg_resources, so setuptools
# stays below 81; --no-deps keeps its own pins (unused by the classifier) uninstalled.
log "installing MiVOLO age estimator (pinned commit, no deps)"
"${PIP[@]}" install "setuptools<81" wheel
"${PIP[@]}" install --no-deps --no-build-isolation \
  "mivolo @ git+https://github.com/WildChlamydia/MiVOLO.git@$MIVOLO_COMMIT"

log "writing menu entry $DESKTOP_FILE"
mkdir -p "$(dirname "$DESKTOP_FILE")" "$(dirname "$ICON_FILE")"
cp "$ICON_SRC" "$ICON_FILE"
chmod 0644 "$ICON_FILE"
cat >"$DESKTOP_FILE" <<EOF
[Desktop Entry]
Type=Application
Name=VOXPrint AI Model Forge
Comment=Train local image models and LoRAs
Exec=$VENV/bin/voxprint-model-forge
Icon=$ICON_FILE
Terminal=false
Categories=Graphics;Photography;
EOF
chmod 0644 "$DESKTOP_FILE"

log "done. Launch 'VOXPrint AI Model Forge' from the desktop menu, or run: $VENV/bin/voxprint-model-forge"
