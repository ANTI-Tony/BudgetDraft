#!/usr/bin/env bash
# Ablation training: fixed budget = 1024 (no multi-view sampling).
# Purpose: show that multi-view {256,512,1024,2048} sampling is NECESSARY,
# not just "sparse-cache branch ON".
# Expected behaviour: the resulting drafter does well at B=1024 but degrades
# at B=256 / 512 / 2048 (budget-sensitive curve, opposite of A+0.5C).
#
# Time budget: ~5h training + ~30min eval on 9 (ds, ctx) cells × γ=5.

set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/workspace/tf/hf_cache
export TRANSFORMERS_CACHE=/workspace/tf/hf_cache
export HF_DATASETS_CACHE=/workspace/tf/hf_cache/datasets

CKPT_DIR="${CKPT_DIR:-/workspace/tf/checkpoints/tinydraft_1024only}"
EVAL_OUT="${EVAL_OUT:-results/full/main_1024only}"
TEACHER="NousResearch/Yarn-Llama-2-7b-128k"
STUDENT="JackFram/llama-68m"

mkdir -p "$CKPT_DIR" "$EVAL_OUT"

banner () { echo; echo "============================================================"; echo " $1"; echo "============================================================"; }

# ============ 1/2  Train fixed-budget=1024 ============
banner "1/2  Train tinydraft_1024only (seq_len=16384, lam=0.5, B=1024 fixed, 5000 steps)"
python3 sd_code/hl/train_tinydraft.py \
    --teacher_model "$TEACHER" \
    --student_model "$STUDENT" \
    --seq_len 16384 \
    --cont_len 256 \
    --chunk_size 8 \
    --lam 0.5 \
    --beta 0 \
    --fixed_budget 1024 \
    --total_steps 5000 \
    --warmup_steps 150 \
    --lr 1e-5 \
    --weight_decay 0.01 \
    --grad_clip 1.0 \
    --log_interval 10 \
    --save_interval 500 \
    --output_dir "$CKPT_DIR" \
    --gradient_checkpointing \
    --seed 42

# ============ 2/2  Eval 3 ds × 3 ctx × γ=5 across 4 budgets ============
banner "2/2  Eval 1024-only ckpt on 9 (ds, ctx) cells × γ=5"

DATASETS=(gs longbench_packed_qmsum lwm)
# (ctx_label, max_length_arg)
CTX_LABELS=(4k 8k 16k)
CTX_MODES=(short long long)
CTX_MAXLEN=("" "--max_length 8192" "--max_length 16384")

for i in 0 1 2; do
  LBL="${CTX_LABELS[$i]}"
  MODE="${CTX_MODES[$i]}"
  MAXLEN="${CTX_MAXLEN[$i]}"
  for DS in "${DATASETS[@]}"; do
    OUT="$EVAL_OUT/eval_${LBL}_${DS}_g5.csv"
    if [ -f "$OUT" ] && [ -f "${OUT%.csv}_samples.csv" ]; then
      echo "  [skip] $OUT (both aggregate + samples already exist)"
      continue
    fi
    echo "[1024only] ctx=$LBL ds=$DS γ=5 -> $OUT"
    # shellcheck disable=SC2086
    python3 sd_code/hl/eval_tinydraft.py \
        --target_model "$TEACHER" \
        --original_student "$STUDENT" \
        --trained_student "$CKPT_DIR/final" \
        --dataset "$DS" \
        --context "$MODE" $MAXLEN \
        --gamma 5 \
        --budgets "256,512,1024,2048" \
        --max_samples 10 \
        --warmup 1 \
        --output_csv "$OUT"
  done
done

banner "ALL DONE — checkpoint: $CKPT_DIR, results: $EVAL_OUT"
ls "$EVAL_OUT" | head
