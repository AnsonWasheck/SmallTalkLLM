#!/usr/bin/env bash
# Layer 2 of the keep-alive chain: makes sure the loop is RUNNING and PROGRESSING.
#
#   bash scripts/watchdog.sh            # run forever, checks every 60s
#
# Three independent failure modes are covered, because "the process exists" is
# not the same as "work is happening":
#
#   1. DEAD    -- no state_loop.py process at all           -> relaunch
#   2. STALLED -- process alive but heartbeat older than
#                 STALL_SECONDS (default 45 min; a full round
#                 is ~35 min, so this only fires on a genuine
#                 hang)                                     -> kill and relaunch
#   3. IDLE    -- loop alive, heartbeat fresh, but no GPU
#                 training process for IDLE_SECONDS         -> logged loudly;
#                 not killed, because evaluation and corpus
#                 building legitimately occupy this window
#
# Everything is appended to artifacts/state_loop/watchdog.log with timestamps so
# the night's behaviour is reconstructable in the morning.
set -uo pipefail
cd "$(dirname "$0")/.."

LOOPDIR="artifacts/state_loop"
LOG="$LOOPDIR/watchdog.log"
HEARTBEAT="$LOOPDIR/heartbeat"
INTERVAL="${INTERVAL:-60}"
STALL_SECONDS="${STALL_SECONDS:-2700}"
IDLE_SECONDS="${IDLE_SECONDS:-900}"

mkdir -p "$LOOPDIR"

say() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }

launch() {
    say "LAUNCH state_loop.py"
    nohup setsid .venv-rocm/bin/python scripts/state_loop.py \
        >> "$LOOPDIR/loop.log" 2>&1 &
    sleep 10
}

say "watchdog started (interval=${INTERVAL}s stall=${STALL_SECONDS}s idle=${IDLE_SECONDS}s)"
last_train_seen=$(date +%s)

while true; do
    now=$(date +%s)

    if ! pgrep -f "[s]tate_loop.py" > /dev/null; then
        say "DEAD: no state_loop.py process"
        launch
        continue
    fi

    if [ -f "$HEARTBEAT" ]; then
        hb=$(head -1 "$HEARTBEAT" 2>/dev/null | cut -d. -f1)
        hb=${hb:-0}
        age=$(( now - hb ))
        if [ "$age" -gt "$STALL_SECONDS" ]; then
            say "STALLED: heartbeat ${age}s old (> ${STALL_SECONDS}s); restarting loop"
            pkill -f "[s]tate_loop.py"
            pkill -f "state_sft_7m.yaml"
            pkill -f "state_stage1_7m.yaml"
            sleep 15
            launch
            continue
        fi
    fi

    if pgrep -f "[p]ython scripts/(train|sft).py" > /dev/null 2>&1 \
       || pgrep -f "scripts/sft.py" > /dev/null 2>&1 \
       || pgrep -f "scripts/train.py" > /dev/null 2>&1; then
        last_train_seen=$now
    else
        idle=$(( now - last_train_seen ))
        if [ "$idle" -gt "$IDLE_SECONDS" ]; then
            say "IDLE: no training process for ${idle}s (loop alive, heartbeat fresh)"
            last_train_seen=$now      # report once per window, do not spam
        fi
    fi

    sleep "$INTERVAL"
done
