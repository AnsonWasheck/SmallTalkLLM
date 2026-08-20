#!/usr/bin/env bash
# Layer 3: the watchdog's watchdog. Deliberately dumb, so it cannot itself fail
# in an interesting way.
#
#   bash scripts/keeper.sh
#
# Every 2 minutes: is watchdog.sh alive? If not, start it. That is all it does.
# The chain is:
#
#   keeper.sh  ->  watchdog.sh  ->  state_loop.py  ->  training
#
# Each layer only knows how to restart the one below it. Three independent
# processes must all die simultaneously for work to stop, and none of them
# depends on the agent session being alive -- which matters, because scheduled
# agent wake-ups have failed twice in this project and cannot be relied on.
set -uo pipefail
cd "$(dirname "$0")/.."

LOG="artifacts/state_loop/keeper.log"
mkdir -p artifacts/state_loop

while true; do
    if ! pgrep -f "[w]atchdog.sh" > /dev/null; then
        echo "[$(date -Is)] watchdog missing; starting" >> "$LOG"
        nohup setsid bash scripts/watchdog.sh >> artifacts/state_loop/watchdog.out 2>&1 &
        sleep 5
    fi
    if ! pgrep -f "[p]romoter.sh" > /dev/null; then
        echo "[$(date -Is)] promoter missing; starting" >> "$LOG"
        nohup setsid bash scripts/promoter.sh >> artifacts/state_loop/promoter.out 2>&1 &
        sleep 5
    fi
    sleep 120
done
