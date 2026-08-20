#!/usr/bin/env bash
# Layer 4: checks every 10 minutes whether the loop's champion has earned an
# automatic promotion to main. All judgement lives in auto_promote.py; this only
# provides the heartbeat. Kept separate from the watchdog so a promotion bug can
# never take down the training loop.
set -uo pipefail
cd "$(dirname "$0")/.."
while true; do
    .venv-rocm/bin/python scripts/auto_promote.py >> artifacts/state_loop/promote.log 2>&1
    sleep 600
done
