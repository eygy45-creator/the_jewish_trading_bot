#!/usr/bin/env bash
# One-command live paper stack (bot + Streamlit). Paper only; no tmux required.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
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

cmd="${1:-status}"

case "${cmd}" in
  status)
    echo "PROJECT_ROOT=${PROJECT_ROOT}"
    echo -n "live bot (tjtb.live.live_paper_crypto): "
    if _live_running; then echo "RUNNING"; else echo "STOPPED"; fi
    echo -n "dashboard (streamlit dashboard/app.py): "
    if _dash_running; then echo "RUNNING"; else echo "STOPPED"; fi
    echo -n "port 8501 listening: "
    if command -v ss >/dev/null 2>&1; then
      if ss -tlnp 2>/dev/null | grep -q ':8501'; then echo "yes"; else echo "no"; fi
    elif command -v netstat >/dev/null 2>&1; then
      if netstat -tlnp 2>/dev/null | grep -q ':8501'; then echo "yes"; else echo "no"; fi
    else
      echo "unknown (install iproute2/ss)"
    fi
    _print_urls
    echo "Logs: ${PROJECT_ROOT}/logs/live_bot.log  ${PROJECT_ROOT}/logs/dashboard.log"
    ;;
  start)
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
    bash "${PROJECT_ROOT}/scripts/live_stack.sh" status
    ;;
  stop)
    echo "stopping live bot (if running)..."
    pkill -f "tjtb.live.live_paper_crypto" 2>/dev/null || true
    echo "stopping dashboard (if running)..."
    pkill -f "streamlit run dashboard/app.py" 2>/dev/null || true
    sleep 1
    bash "${PROJECT_ROOT}/scripts/live_stack.sh" status
    ;;
  restart)
    bash "${PROJECT_ROOT}/scripts/live_stack.sh" stop || true
    sleep 2
    bash "${PROJECT_ROOT}/scripts/live_stack.sh" start
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
