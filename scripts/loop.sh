#!/usr/bin/env bash
# Unattended round-by-round improvement loop for smalltalk-ai.
#
#   bash scripts/loop.sh                 # run forever
#   ROUNDS=10 bash scripts/loop.sh       # run 10 rounds
#   nohup bash scripts/loop.sh > /tmp/loop.log 2>&1 &   # detached
#
# Each round: build a VARIED deck -> leakage gate -> train -> measure over N
# samples -> log -> promote if better. Knobs are nudged from the previous round's
# measurements, so rounds explore rather than repeat (the flaw in autoloop.py v1).
#
# INVARIANTS
#   * Architecture is FROZEN at smalltalk-7m (6,689,024 params). Never touched.
#   * SmallTalkBench-HARD stays held out; build_deck.py fails closed on leakage.
#   * artifacts/loop/ledger.jsonl is append-only, so regressions stay visible.
#   * Best-scoring checkpoint is mirrored to artifacts/loop/best/.
set -uo pipefail
cd "$(dirname "$0")/.."

ROUNDS="${ROUNDS:-0}"                 # 0 = forever
SAMPLES="${SAMPLES:-16}"
STEPS="${STEPS:-1100}"
LR="${LR:-3e-4}"
BATCH="${BATCH:-64}"
SEQ="${SEQ:-512}"
BASE="${BASE:-artifacts/runs/real7m/best}"   # base-language checkpoint
TOKENIZER="${TOKENIZER:-artifacts/tokenizer-4096}"
MODEL_CFG="configs/model/smalltalk-7m.yaml"  # FROZEN
LOOP_DIR="artifacts/loop"
LEDGER="$LOOP_DIR/ledger.jsonl"
STATE="$LOOP_DIR/loop_state.json"
mkdir -p "$LOOP_DIR"

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

# resume round numbering from whatever already exists
round_start=$(ls -d artifacts/runs/auto* 2>/dev/null | sed 's/.*auto//' | sort -n | tail -1)
round_start=$(( ${round_start:-0} + 1 ))

i=$round_start
while :; do
  [ "$ROUNDS" -gt 0 ] && [ $((i - round_start)) -ge "$ROUNDS" ] && { log "done $ROUNDS rounds"; break; }
  RUN="auto$(printf '%03d' "$i")"
  log "=== round $i ($RUN) ==="

  # ---- adaptive knobs from the last measurement --------------------------
  # Defaults; python nudges them from the previous round's weaknesses.
  read -r COMBI BORED HEAVY PMEM PTOPIC PUNK SEED <<<"$(python3 - "$STATE" "$i" <<'PY'
import json, sys, random
state_path, i = sys.argv[1], int(sys.argv[2])
combi, bored, heavy, pmem, ptopic, punk = 10000, 0.20, 0.11, 0.22, 0.18, 0.10
try:
    m = json.load(open(state_path))
    p = m.get("probes", {})
    # Weak boredom handling -> more boredom data.
    if p.get("bored", {}).get("pass_rate", 1) < 0.75: bored = min(0.30, bored + 0.05)
    # Grief failures -> more heavy-valence data.
    if min(p.get("grief", {}).get("pass_rate", 1),
           p.get("grief2", {}).get("pass_rate", 1)) < 0.9: heavy = min(0.20, heavy + 0.04)
    # Memory recall weak -> more memory probes.
    if p.get("memory", {}).get("pass_rate", 1) < 0.6: pmem = min(0.35, pmem + 0.05)
    # Graceful-ignorance was the weakest probe (0.50-0.62) with no lever at all.
    if p.get("unknown", {}).get("pass_rate", 1) < 0.8: punk = min(0.25, punk + 0.04)
    # Low diversity -> bigger, more varied corpus.
    if m.get("mean_unique_rate", 1) < 0.85: combi = min(20000, combi + 3000)
except Exception:
    pass
print(combi, bored, heavy, pmem, ptopic, punk, random.randint(1, 10**6))
PY
)"
  log "knobs: combi=$COMBI bored=$BORED heavy=$HEAVY p_mem=$PMEM p_unk=$PUNK seed=$SEED"

  # ---- deck (fails closed on leakage) ------------------------------------
  if ! python3 scripts/build_deck.py --combi "$COMBI" --seed "$SEED" \
        --bored "$BORED" --heavy "$HEAVY" --p-memory "$PMEM" --p-topic-switch "$PTOPIC" \
        --p-unknown "$PUNK" \
        --out data/processed/sft_train.jsonl \
        --stats-out "$LOOP_DIR/deck_$RUN.json" > "$LOOP_DIR/deck_$RUN.log" 2>&1; then
    log "deck build FAILED; see $LOOP_DIR/deck_$RUN.log"; sleep 30; i=$((i+1)); continue
  fi
  tail -2 "$LOOP_DIR/deck_$RUN.log" | head -1

  # ---- train --------------------------------------------------------------
  if ! bash scripts/rocm.sh scripts/sft.py --config configs/train/sft_7m.yaml \
        --run-name "$RUN" --output-dir artifacts/runs \
        --model-config "$MODEL_CFG" --tokenizer "$TOKENIZER" \
        --train-data data/processed/sft_train.jsonl \
        --val-data data/processed/sft_val.jsonl \
        --init-from "$BASE" \
        --max-steps "$STEPS" --learning-rate "$LR" --seq-len "$SEQ" --batch-size "$BATCH" \
        --eval-every $((STEPS/10)) --save-every "$STEPS" --log-every $((STEPS/5)) \
        --device auto > "$LOOP_DIR/train_$RUN.log" 2>&1; then
    log "training FAILED; see $LOOP_DIR/train_$RUN.log"; sleep 30; i=$((i+1)); continue
  fi
  VAL=$(grep -o '"best_val_loss": [0-9.]*' "$LOOP_DIR/train_$RUN.log" | tail -1 | awk '{print $2}')
  log "trained, best_val_loss=$VAL"

  # ---- measure over N samples --------------------------------------------
  CKPT="artifacts/runs/$RUN/best"
  if ! python3 scripts/measure.py --checkpoint "$CKPT" --tokenizer "$TOKENIZER" \
        --samples "$SAMPLES" --json-out "$LOOP_DIR/measure_$RUN.json" \
        > "$LOOP_DIR/measure_$RUN.log" 2>&1; then
    log "measure FAILED; see $LOOP_DIR/measure_$RUN.log"; sleep 30; i=$((i+1)); continue
  fi
  cp "$LOOP_DIR/measure_$RUN.json" "$STATE"
  SCORE=$(python3 -c "import json;print(json.load(open('$LOOP_DIR/measure_$RUN.json'))['score'])")
  log "score=$SCORE"

  # ---- ledger + promotion -------------------------------------------------
  python3 - "$RUN" "$SCORE" "${VAL:-null}" "$LOOP_DIR" "$i" <<'PY'
import json, sys, shutil, time
from pathlib import Path
run, score, val, loop_dir, i = sys.argv[1], float(sys.argv[2]), sys.argv[3], Path(sys.argv[4]), int(sys.argv[5])
m = json.loads((loop_dir / f"measure_{run}.json").read_text())
deck = json.loads((loop_dir / f"deck_{run}.json").read_text())
ledger = loop_dir / "ledger.jsonl"
rows = [json.loads(l) for l in ledger.read_text().splitlines() if l.strip()] if ledger.exists() else []
best = max((r.get("score", 0) for r in rows if "score" in r), default=-1)
promoted = score > best
entry = {"round": i, "run": run, "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
         "score": score, "val_loss": None if val in ("null", "") else float(val),
         "mean_pass_rate": m["mean_pass_rate"], "mean_unique_rate": m["mean_unique_rate"],
         "mean_words": m["mean_words"],
         "probe_pass": {k: v["pass_rate"] for k, v in m["probes"].items()},
         "train_examples": deck["written"], "promoted": promoted}
with ledger.open("a") as f:
    f.write(json.dumps(entry) + "\n")
if promoted:
    dest = loop_dir / "best"
    if dest.exists(): shutil.rmtree(dest)
    shutil.copytree(f"artifacts/runs/{run}/best", dest)
    print(f"PROMOTED (score {score} > {best})")
else:
    print(f"not promoted (score {score} <= {best})")
PY

  # keep disk in check: retain only the last few non-promoted runs
  ls -dt artifacts/runs/auto* 2>/dev/null | tail -n +6 | xargs -r rm -rf
  i=$((i+1))
done
