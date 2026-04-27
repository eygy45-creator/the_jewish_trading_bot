#!/usr/bin/env bash
# Install, enable, and start TJTB systemd units (recorder, live paper bot, dashboard).
# Run on the VPS from the repo root: bash deploy/install_systemd_services.sh
set -euo pipefail

if [ -z "${BASH_VERSION:-}" ]; then
  exec /usr/bin/env bash "$0" "$@"
fi

ROOT="/home/ubuntu/the_jewish_trading_bot/the_jewish_trading_bot"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ "${REPO_ROOT}" != "${ROOT}" ]]; then
  echo "warning: repo root is ${REPO_ROOT}, expected ${ROOT}. Unit files use hardcoded paths for /home/ubuntu/..." >&2
fi

if [[ "$(id -u)" -ne 0 ]]; then
  echo "this installer must be run with sudo: sudo bash deploy/install_systemd_services.sh" >&2
  exit 1
fi

mkdir -p "${ROOT}/logs"
install -o ubuntu -g ubuntu -m 644 "${REPO_ROOT}/deploy/systemd/tjtb-coinbase-recorder.service" /etc/systemd/system/
install -o ubuntu -g ubuntu -m 644 "${REPO_ROOT}/deploy/systemd/tjtb-live-paper.service" /etc/systemd/system/
install -o ubuntu -g ubuntu -m 644 "${REPO_ROOT}/deploy/systemd/tjtb-dashboard.service" /etc/systemd/system/

systemctl daemon-reload

# Disable legacy unit name if present (avoid two live bots).
if systemctl list-unit-files tjtb-live.service 2>/dev/null | grep -q tjtb-live.service; then
  systemctl disable --now tjtb-live.service 2>/dev/null || true
fi

systemctl enable tjtb-coinbase-recorder.service tjtb-live-paper.service tjtb-dashboard.service
systemctl restart tjtb-coinbase-recorder.service
sleep 2
systemctl restart tjtb-live-paper.service
systemctl restart tjtb-dashboard.service

echo ""
echo "Installed and started:"
echo "  tjtb-coinbase-recorder.service"
echo "  tjtb-live-paper.service"
echo "  tjtb-dashboard.service"
echo ""
echo "Status:  sudo systemctl status tjtb-coinbase-recorder tjtb-live-paper tjtb-dashboard --no-pager"
echo "Logs:    tail -f ${ROOT}/logs/*.systemd.log ${ROOT}/logs/live_bot.log"
