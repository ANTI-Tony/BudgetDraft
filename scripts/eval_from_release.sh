#!/usr/bin/env bash
# Eval-only entrypoint for users who downloaded the released checkpoints
# (instead of re-training them). Auto-detects the 4 ckpt subdirs and
# orchestrates: main + ablation + lambda eval (96 configs)
#                + 1024-only eval (9 configs).
#
# Usage:
#   bash scripts/eval_from_release.sh /path/to/downloaded/checkpoints
#
# Expected layout of the argument directory:
#   <ckpt_root>/
#   ├── main/final/         (A+0.5C — main ckpt)
#   ├── aonly/final/        (A-only ablation)
#   ├── ac/final/           (A+C  lambda=1 ablation)
#   └── budget1024/final/   (fixed-budget ablation, optional)
#
# Total wall-clock: ~6 h on A100 80GB.

set -euo pipefail

if [ $# -lt 1 ]; then
  echo "usage: $0 <path/to/downloaded/checkpoints>" >&2
  exit 2
fi
CKPT_ROOT="$1"
[ -d "$CKPT_ROOT" ] || { echo "ERROR: $CKPT_ROOT not a directory"; exit 1; }

# Resolve absolute path so the rest of the script can chdir freely
CKPT_ROOT="$(cd "$CKPT_ROOT" && pwd)"

# ---- locate the 4 ckpts ------------------------------------------------------
MAIN="$CKPT_ROOT/main/final"
AONLY="$CKPT_ROOT/aonly/final"
AC="$CKPT_ROOT/ac/final"
K1024="$CKPT_ROOT/budget1024/final"

# MAIN / AONLY / AC are required (Table 1 + Table 3 main columns).
# K1024 is optional (extra ablation table only).
have_k1024=1
missing=0
for c in "$MAIN" "$AONLY" "$AC"; do
  if [ ! -d "$c" ] || [ ! -f "$c/config.json" ]; then
    echo "  ✗ missing: $c"
    missing=1
  else
    echo "  ✓ found:   $c"
  fi
done
if [ ! -d "$K1024" ] || [ ! -f "$K1024/config.json" ]; then
  echo "  ⚠ optional 1024-only ckpt missing → phase 2 will be skipped"
  have_k1024=0
else
  echo "  ✓ found:   $K1024"
fi
[ "$missing" -eq 0 ] || {
  echo "ERROR: required ckpts missing (main / aonly / ac)"
  echo "       each <ckpt>/final/ must contain config.json + weights"
  exit 1
}

# ---- env vars consumed by run_full_experiments.sh ---------------------------
export CKPT_MAIN="$MAIN"
export CKPT_AONLY="$AONLY"
export CKPT_AC="$AC"

# Repo root (this script lives in scripts/)
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

echo
echo "============================================================"
echo " Phase 1/2: main (A+0.5C) + ablation (A-only) + lambda (A+C)"
echo "            -> 96 configs via run_full_experiments.sh"
echo "============================================================"
./run_full_experiments.sh

if [ "$have_k1024" -eq 0 ]; then
  echo
  echo "Phase 2/2 (1024-only) skipped — checkpoint not present."
  echo "============================================================"
  echo " ALL DONE. Results: results/full/main/"
  echo "============================================================"
  exit 0
fi

echo
echo "============================================================"
echo " Phase 2/2: 1024-only ablation eval (9 configs)"
echo "============================================================"

OUT_DIR="results/full/main_1024only"
mkdir -p "$OUT_DIR"
TARGET="NousResearch/Yarn-Llama-2-7b-128k"
ORIGINAL="JackFram/llama-68m"

# (ctx, mode, max_length_arg)
CTXS=(4k 8k 16k)
MODES=(short long long)
MAXLEN_ARGS=("" "--max_length 8192" "--max_length 16384")

for i in 0 1 2; do
  CTX="${CTXS[$i]}"; MODE="${MODES[$i]}"; MAXLEN="${MAXLEN_ARGS[$i]}"
  for DS in gs longbench_packed_qmsum lwm; do
    OUT="$OUT_DIR/eval_${CTX}_${DS}_g5.csv"
    if [ -f "$OUT" ] && [ -f "${OUT%.csv}_samples.csv" ]; then
      echo "  [skip] $OUT"; continue
    fi
    echo "[1024only] ctx=$CTX ds=$DS γ=5"
    # shellcheck disable=SC2086
    python3 src/eval.py \
        --target_model "$TARGET" \
        --original_student "$ORIGINAL" \
        --trained_student "$K1024" \
        --dataset "$DS" --context "$MODE" $MAXLEN \
        --gamma 5 --budgets "256,512,1024,2048" \
        --max_samples 10 --warmup 1 \
        --output_csv "$OUT"
  done
done

echo
echo "============================================================"
echo " ALL DONE. Results:"
echo "   results/full/main/                — main + ablation + lambda"
echo "   results/full/main_1024only/       — fixed-budget ablation"
echo
echo " Sanity-check against EXPECTED_RESULTS.md."
echo "============================================================"
