#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_ROOT="$ROOT_DIR/prototype/second-brain-starter"

if [[ ! -d "$APP_ROOT" ]]; then
  echo "second-brain-starter not found at: $APP_ROOT" >&2
  exit 1
fi

export PYTHONPATH="$APP_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
exec python3 -m second_brain.cli --root "$APP_ROOT" "$@"
