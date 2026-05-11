#!/usr/bin/env bash
# Run Bybit public recorder with repo-root-relative imports (src layout).
# Usage (tmux): BYBIT_SYMBOL=ETHUSDT ./scripts/run_bybit_recorder.sh --log-level INFO
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
if [[ -x "${ROOT}/venv/bin/python" ]]; then
  exec "${ROOT}/venv/bin/python" -m tjtb.data.bybit_recorder "$@"
fi
exec python3 -m tjtb.data.bybit_recorder "$@"
