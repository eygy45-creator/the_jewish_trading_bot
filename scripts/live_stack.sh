#!/usr/bin/env bash
# One-command live paper stack (bot + Streamlit). Paper only; no tmux required.
# If invoked as `sh scripts/live_stack.sh`, re-exec under bash (dash: invalid option pipefail).
if [ -z "${BASH_VERSION:-}" ]; then
  exec /usr/bin/env bash "$0" "$@"
fi
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
THIS_SCRIPT="${SCRIPT_DIR}/live_stack.sh"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

TJTB_VPS_HOST="${TJTB_VPS_HOST:-130.61.219.247}"
TJTB_SSH_KEY="${TJTB_SSH_KEY:-${HOME}/Downloads/ssh-key-2026-04-26.key}"
TJTB_SSH_USER="${TJTB_SSH_USER:-ubuntu}"

if [[ ! -d "${PROJECT_ROOT}/venv" ]]; then
  echo "error: venv not found at ${PROJECT_ROOT}/venv" >&2
  exit 1
fi

# shellcheck source=/dev/null
source "${PROJECT_ROOT}/venv/bin/activate"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

mkdir -p "${PROJECT_ROOT}/logs"
"${PROJECT_ROOT}/venv/bin/python3" -c "from tjtb.runtime_paths import ensure_runtime_dirs; ensure_runtime_dirs()"

_live_running() { pgrep -f "tjtb.live.live_paper_crypto" >/dev/null 2>&1; }
_dash_running() { pgrep -f "streamlit run dashboard/app.py" >/dev/null 2>&1; }

_csv_data_rows() {
  local f="$1"
  if [[ ! -f "$f" ]]; then
    echo "n/a (file missing)"
    return
  fi
  local n
  n=$(wc -l <"$f" | tr -d '[:space:]')
  if (( n <= 1 )); then
    echo "0"
  else
    echo "$((n - 1))"
  fi
}

_heartbeat_status() {
  local hb="${PROJECT_ROOT}/logs/heartbeat.txt"
  if [[ ! -f "$hb" ]]; then
    echo "heartbeat: exists=no path=${hb}"
    return
  fi
  local now mt age
  now=$(date +%s)
  if mt=$(stat -c %Y "$hb" 2>/dev/null); then
    :
  elif mt=$(stat -f %m "$hb" 2>/dev/null); then
    :
  else
    echo "heartbeat: exists=yes path=${hb} age_sec=unknown (stat failed)"
    return
  fi
  age=$((now - mt))
  echo "heartbeat: exists=yes path=${hb} age_sec=${age}"
}

_print_urls() {
  echo ""
  echo "=== Dashboard URLs ==="
  echo "  SSH tunnel (on your laptop):  http://localhost:8501"
  echo "  VPS direct:                    http://${TJTB_VPS_HOST}:8501"
  echo ""
  echo "=== SSH tunnel command (run on your laptop) ==="
  echo "  ssh -i ${TJTB_SSH_KEY} -L 8501:localhost:8501 ${TJTB_SSH_USER}@${TJTB_VPS_HOST}"
  echo ""
}

_port_8501() {
  if command -v ss >/dev/null 2>&1; then
    if ss -tlnp 2>/dev/null | grep -qE ':8501\b'; then
      echo "yes"
    else
      echo "no"
    fi
  elif command -v netstat >/dev/null 2>&1; then
    if netstat -tlnp 2>/dev/null | grep -qE ':8501\b'; then
      echo "yes"
    else
      echo "no"
    fi
  else
    echo "unknown (install iproute2/ss)"
  fi
}

_touch_stack_files() {
  mkdir -p "${PROJECT_ROOT}/logs" "${PROJECT_ROOT}/data/live"
  : >>"${PROJECT_ROOT}/logs/live_bot.log"
  TJTB_ROOT="${PROJECT_ROOT}" "${PROJECT_ROOT}/venv/bin/python3" -c \
'from datetime import datetime, timezone
from pathlib import Path
import os
root = Path(os.environ["TJTB_ROOT"])
p = root / "logs" / "heartbeat.txt"
p.parent.mkdir(parents=True, exist_ok=True)
p.write_text(datetime.now(timezone.utc).isoformat() + "\n")'
}

cmd="${1:-status}"

case "${cmd}" in
  status)
    echo "PROJECT_ROOT=${PROJECT_ROOT}"
    echo -n "live bot (tjtb.live.live_paper_crypto): "
    if _live_running; then echo "RUNNING"; else echo "STOPPED"; fi
    echo -n "dashboard (streamlit dashboard/app.py): "
    if _dash_running; then echo "RUNNING"; else echo "STOPPED"; fi
    _heartbeat_status
    echo -n "paper_trades.csv data rows (excl. header): "
    _csv_data_rows "${PROJECT_ROOT}/data/live/paper_trades.csv"
    echo -n "opportunities.csv data rows (excl. header): "
    _csv_data_rows "${PROJECT_ROOT}/data/live/opportunities.csv"
    echo -n "port 8501 listening: "
    _port_8501
    _print_urls
    echo "Logs: ${PROJECT_ROOT}/logs/live_bot.log  ${PROJECT_ROOT}/logs/dashboard.log"
    ;;
  start)
    _touch_stack_files
    if _live_running; then echo "live bot already running"; else
      echo "starting live bot -> logs/live_bot.log"
      nohup "${PROJECT_ROOT}/venv/bin/python3" -m tjtb.live.live_paper_crypto \
        >>"${PROJECT_ROOT}/logs/live_bot.log" 2>&1 &
      disown || true
    fi
    if _dash_running; then echo "dashboard already running"; else
      echo "starting dashboard -> logs/dashboard.log"
      nohup "${PROJECT_ROOT}/venv/bin/streamlit" run dashboard/app.py \
        --server.port 8501 \
        --server.address 0.0.0.0 \
        --server.headless true \
        >>"${PROJECT_ROOT}/logs/dashboard.log" 2>&1 &
      disown || true
    fi
    sleep 1
    "${BASH:-/usr/bin/env bash}" "${THIS_SCRIPT}" status
    ;;
  stop)
    echo "stopping live bot (if running)..."
    pkill -f "tjtb.live.live_paper_crypto" 2>/dev/null || true
    echo "stopping dashboard (if running)..."
    pkill -f "streamlit run dashboard/app.py" 2>/dev/null || true
    sleep 1
    "${BASH:-/usr/bin/env bash}" "${THIS_SCRIPT}" status
    ;;
  restart)
    "${BASH:-/usr/bin/env bash}" "${THIS_SCRIPT}" stop || true
    sleep 2
    "${BASH:-/usr/bin/env bash}" "${THIS_SCRIPT}" start
    ;;
  logs)
    echo "== tail live_bot.log (last 80 lines) =="
    tail -n 80 "${PROJECT_ROOT}/logs/live_bot.log" 2>/dev/null || echo "(file missing)"
    echo ""
    echo "== tail dashboard.log (last 80 lines) =="
    tail -n 80 "${PROJECT_ROOT}/logs/dashboard.log" 2>/dev/null || echo "(file missing)"
    echo ""
    echo "Follow live:   tail -f ${PROJECT_ROOT}/logs/live_bot.log"
    echo "Follow dash:   tail -f ${PROJECT_ROOT}/logs/dashboard.log"
    ;;
  *)
    echo "usage: bash scripts/live_stack.sh {status|start|stop|restart|logs}" >&2
    exit 2
    ;;
esac
