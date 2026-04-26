# The Jewish Trading Bot

Milestone 1: research skeleton for MNQ microstructure signals—no live broker.

## Repository layout

Python package `tjtb` lives under `src/tjtb/`. Top-level folders mirror the architecture spec (`config`, `data`, `features`, …).

## Install

```bash
cd /home/itaicohen/the_jewish_trading_bot
python3 -m venv .venv   # requires python3-venv on Debian/Ubuntu
source .venv/bin/activate
pip install -e ".[dev]"
```

## Run tests

```bash
pytest -q
```

## Small replay (synthetic CSV)

```bash
python scripts/run_replay.py --csv data/sample_events.csv --max-events 500
```

## Baseline training (synthetic features + labels)

```bash
python scripts/train_baseline.py --rows 2000 --output-dir runs/baseline_demo
```

## Session research report (scaffold + synthetic metrics)

```bash
python scripts/session_report.py --output runs/session_report_demo.json
```

## Design notes

- **Tradable hours** are not hard-coded as final rules: use `SessionResearchSettings` candidate windows and `reports.session_research` for out-of-sample stability scaffolding.
- **Risk** is independent of signals (`tjtb.risk`).
- **News** is dependency-injected via `NewsCalendarService` (`tjtb.news`).
