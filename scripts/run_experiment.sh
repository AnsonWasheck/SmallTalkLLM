#!/usr/bin/env bash
# Full scaling study, end to end. Edit SIZES / OFFLINE_N and go.
#
#   bash scripts/run_experiment.sh
#
# Stops at the first failure. The pre-flight gate runs first and is not optional:
# it has already caught two silent-corruption bugs in this repo.
set -euo pipefail
cd "$(dirname "$0")/.."

SIZES="${SIZES:-4m 5m 7m 8m 15m}"
OFFLINE_N="${OFFLINE_N:-20000}"
EVAL_DIR="${EVAL_DIR:-artifacts/eval}"
DEVICE="${DEVICE:-auto}"

echo "=== 0. backend ==="
python scripts/check_device.py

echo "=== 1. pre-flight gate ==="
python scripts/smoke_test.py
python scripts/count_params.py

echo "=== 2. corpus ==="
# Add --dailydialog / --empathetic / --jsonl for the real study; --offline alone
# only exercises the pipeline.
python scripts/prepare_data.py \
    ${DAILYDIALOG:+--dailydialog "$DAILYDIALOG"} \
    ${EMPATHETIC:+--empathetic "$EMPATHETIC"} \
    ${TEACHER_JSONL:+--jsonl "$TEACHER_JSONL"} \
    --jsonl data/seed/example_conversations.jsonl \
    --offline "$OFFLINE_N" \
    --out data/processed

# RQ5 control: same data, generic-LM filtering
python scripts/prepare_data.py --offline "$OFFLINE_N" --permissive \
    --out data/processed_generic

echo "=== 3. tokenizers ==="
python scripts/train_tokenizer.py --data data/processed/train.jsonl --vocab-size 4096 6144

echo "=== 4-5. train + sft each size ==="
CKPTS=()
for S in $SIZES; do
    echo "--- stage 1: $S ---"
    python scripts/train.py --config "configs/train/stage1_${S}.yaml" --device "$DEVICE"
    echo "--- stage 2: $S ---"
    python scripts/sft.py --config "configs/train/sft_${S}.yaml" --device "$DEVICE"
    CKPTS+=("artifacts/runs/sft-${S}/best")
done

echo "=== 6. evaluate (per-model decoding sweep, then bench) ==="
python scripts/evaluate.py --checkpoint "${CKPTS[@]}" \
    --val-data data/processed/val.jsonl --out "$EVAL_DIR" \
    --device "$DEVICE" --sweep \
    --judge-out "$EVAL_DIR/judge.jsonl"

# blind pairwise between the smallest and largest model
if [ "${#CKPTS[@]}" -ge 2 ]; then
    python scripts/evaluate.py \
        --checkpoint "${CKPTS[0]}" "${CKPTS[${#CKPTS[@]}-1]}" \
        --out "$EVAL_DIR/pairwise_run" --device "$DEVICE" \
        --pairwise "$EVAL_DIR/pairwise.jsonl"
fi

echo "=== 7. report ==="
python scripts/scaling_report.py --eval-dir "$EVAL_DIR" --out docs/RESULTS.md

cat <<'EOF'

Done. Next steps that require a human or a teacher model:
  * fill in `scores` in artifacts/eval/judge-*.jsonl (LLM or human), then
      python scripts/evaluate.py --checkpoint ... --judge-scores <filled>.jsonl
  * collect blind pairwise votes for artifacts/eval/pairwise.jsonl, then
      python -c "from smalltalk.eval.judge import score_pairwise; \
                 print(score_pairwise('votes.jsonl','artifacts/eval/pairwise.key.json'))"
  * stage 3 distillation: scripts/distill.py prompts -> teacher -> select -> sft.py
Only then are RQ7 and RQ8 answerable.
EOF
