#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if [[ ! -d ".venv" ]]; then
  echo "Missing .venv in ${ROOT_DIR}"
  exit 1
fi

source ".venv/bin/activate"
export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"

if [[ -z "${API_TOKEN:-}" ]]; then
  echo "API_TOKEN is not set"
  exit 1
fi

HOST="${TJTB_API_HOST:-127.0.0.1}"
PORT="${TJTB_API_PORT:-8080}"

exec uvicorn tjtb.api.app:app --host "${HOST}" --port "${PORT}"
