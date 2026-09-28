#!/usr/bin/env bash
# install-laya-serve.sh - build laya-serve and install its systemd USER unit.
#
# This script is deliberately non-activating: it builds the project, writes
# ~/.config/systemd/user/laya-serve.service and runs `systemctl --user
# daemon-reload`. It does NOT enable or start the unit. Claude (or the operator)
# decides when to do that.
#
# Usage:
#   scripts/install-laya-serve.sh [--print] [--no-build] [--no-reload]
#
#   --print      render the unit to stdout and exit; touch nothing
#   --no-build   install the unit without rebuilding dist/
#   --no-reload  skip `systemctl --user daemon-reload`
#
# Env overrides:
#   LAYA_SERVE_REPO   repository root        (default: parent of this script)
#   LAYA_SERVE_NODE   node binary for the unit (default: /usr/bin/node)
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="${LAYA_SERVE_REPO:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
NODE_BIN="${LAYA_SERVE_NODE:-/usr/bin/node}"
UNIT_SOURCE="$REPO/deploy/laya-serve.service"
UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
UNIT_DEST="$UNIT_DIR/laya-serve.service"
SHADOW_DIR="${LAYA_SERVE_SHADOW_DIR:-$HOME/.auraforge-work/shadow}"

PRINT=0
BUILD=1
RELOAD=1
for arg in "$@"; do
  case "$arg" in
    --print) PRINT=1 ;;
    --no-build) BUILD=0 ;;
    --no-reload) RELOAD=0 ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "install-laya-serve: unknown argument: $arg" >&2; exit 2 ;;
  esac
done

render_unit() {
  local unit
  unit="$(cat -- "$UNIT_SOURCE")"
  # %h/projects/auraforge-semantic-control is the canonical layout; point the
  # installed unit at wherever this checkout actually lives.
  unit="${unit//%h\/projects\/auraforge-semantic-control/$REPO}"
  unit="${unit//\/usr\/bin\/node/$NODE_BIN}"
  printf '%s\n' "$unit"
}

if [[ ! -f "$UNIT_SOURCE" ]]; then
  echo "install-laya-serve: missing unit template: $UNIT_SOURCE" >&2
  exit 1
fi

if [[ "$PRINT" -eq 1 ]]; then
  render_unit
  exit 0
fi

if [[ ! -x "$NODE_BIN" ]]; then
  echo "install-laya-serve: node binary not executable: $NODE_BIN (set LAYA_SERVE_NODE)" >&2
  exit 1
fi

if [[ ! -d "$REPO/node_modules" ]]; then
  echo "install-laya-serve: $REPO/node_modules is missing; run 'yarn install --frozen-lockfile' first" >&2
  exit 1
fi

if [[ "$BUILD" -eq 1 ]]; then
  echo "install-laya-serve: building $REPO/dist"
  (cd -- "$REPO" && "$NODE_BIN" ./node_modules/typescript/bin/tsc -p tsconfig.json)
fi

if [[ ! -f "$REPO/dist/serve/index.js" ]]; then
  echo "install-laya-serve: $REPO/dist/serve/index.js is missing; rerun without --no-build" >&2
  exit 1
fi

mkdir -p -- "$SHADOW_DIR"
mkdir -p -- "$UNIT_DIR"
render_unit > "$UNIT_DEST"
echo "install-laya-serve: wrote $UNIT_DEST"

if [[ "$RELOAD" -eq 1 ]] && command -v systemctl >/dev/null 2>&1; then
  if systemctl --user daemon-reload 2>/dev/null; then
    echo "install-laya-serve: systemctl --user daemon-reload done"
  else
    echo "install-laya-serve: could not reach the user systemd manager; run 'systemctl --user daemon-reload' yourself" >&2
  fi
fi

cat <<EOF

install-laya-serve: unit installed but NOT enabled and NOT started (by design).
The shadow log directory exists at: $SHADOW_DIR

Next steps (operator):
  systemctl --user enable --now laya-serve.service
  systemctl --user status laya-serve.service
  journalctl --user -u laya-serve.service -f
  curl -s http://127.0.0.1:8790/health

For the service to keep running without an active login:
  loginctl enable-linger "$USER"
EOF
