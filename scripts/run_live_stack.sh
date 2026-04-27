#!/usr/bin/env bash
# Paper stack: canonical PROJECT_ROOT, no duplicate bot/dashboard, read-only dashboard.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

if [[ ! -d "${PROJECT_ROOT}/venv" ]]; then
  echo "error: venv not found at ${PROJECT_ROOT}/venv" >&2
  exit 1
fi

# shellcheck source=/dev/null
source "${PROJECT_ROOT}/venv/bin/activate"

export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

mkdir -p "${PROJECT_ROOT}/logs"
"${PROJECT_ROOT}/venv/bin/python3" -c "from tjtb.runtime_paths import ensure_runtime_dirs; ensure_runtime_dirs()"

if pgrep -f "tjtb.live.live_paper_crypto" >/dev/null 2>&1; then
  echo "live bot already running; skip start"
else
  echo "starting live bot -> logs/live_bot.log"
  nohup "${PROJECT_ROOT}/venv/bin/python3" -m tjtb.live.live_paper_crypto \
    >>"${PROJECT_ROOT}/logs/live_bot.log" 2>&1 &
  disown || true
fi

if pgrep -f "streamlit run dashboard/app.py" >/dev/null 2>&1; then
  echo "dashboard already running; skip start"
else
  echo "starting dashboard -> logs/dashboard.log"
  nohup "${PROJECT_ROOT}/venv/bin/streamlit" run dashboard/app.py \
    --server.port 8501 \
    --server.address 0.0.0.0 \
    --server.headless true \
    >>"${PROJECT_ROOT}/logs/dashboard.log" 2>&1 &
  disown || true
fi

echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "done."
