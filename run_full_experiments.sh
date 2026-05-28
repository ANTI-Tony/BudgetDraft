#!/usr/bin/env bash
# run_full_experiments.sh — single entry point for the paper's eval matrix.
#
# Sections:
#   1. MAIN          : A+0.5C ckpt, 3 datasets × 3 contexts × budgets
#                      {256,512,1024,2048}, with per-context γ grid:
#                        4K  : γ ∈ {5,10,15,...,60}   (12 values)
#                        8K  : γ ∈ {5,10,15,...,30}   ( 6 values)
#                        16K : γ ∈ {5,10,15}          ( 3 values)
#                      Each CSV contains both `original` (untrained 68M) and
#                      `budgetdraft` (trained 68M) rows.
#   2. ABLATION      : A-only ckpt, 3 datasets × 3 contexts, γ=5 fixed.
#                      Paired with section 1 at γ=5 for A+0.5C vs A-only.
#   3. λ SENSITIVITY : A+C ckpt,    3 datasets × 3 contexts, γ=5 fixed.
#                      Paired with section 1 at γ=5 for A+0.5C vs A+C.
#
# Run from the repo root on a machine with at least one A100-class GPU.
#
# Checkpoint paths must be provided via env vars (no defaults — point them at
# the directory where you downloaded the released checkpoints):
#   CKPT_MAIN=/path/to/main CKPT_AONLY=/path/to/aonly CKPT_AC=/path/to/ac \
#     ./run_full_experiments.sh
#
# Resume-safe: any CSV that already exists is skipped.

set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---- paths -----------------------------------------------------------------
: "${CKPT_MAIN:?CKPT_MAIN must point at the A+0.5C checkpoint dir}"     # A+0.5C (primary)
: "${CKPT_AONLY:?CKPT_AONLY must point at the A-only checkpoint dir}"    # A-only
: "${CKPT_AC:?CKPT_AC must point at the A+C (lambda=1.0) checkpoint dir}" # A+C

TARGET="NousResearch/Yarn-Llama-2-7b-128k"
ORIGINAL="JackFram/llama-68m"
EVAL_PY="src/eval.py"

RESULTS_ROOT="results/full"
mkdir -p "$RESULTS_ROOT/main" "$RESULTS_ROOT/ablation_a_only" "$RESULTS_ROOT/lambda_ac"

# ---- grid ------------------------------------------------------------------
DATASETS=(gs longbench_packed_qmsum lwm)
BUDGETS="256,512,1024,2048"

# Per-context γ grid (longer contexts memory-bound -> shorter γ sweep)
#   4K : 5,10,15,...,60   (12 values)
#   8K : 5,10,15,...,30   ( 6 values)
#   16K: 5,10,15,...,40   ( 8 values)  — bumped from 3 after observing peak still
#                                        unresolved at γ=15
GAMMAS_4K=();  for g in $(seq 5 5  60); do GAMMAS_4K+=("$g");  done
GAMMAS_8K=();  for g in $(seq 5 5  30); do GAMMAS_8K+=("$g");  done
GAMMAS_16K=(); for g in $(seq 5 5  40); do GAMMAS_16K+=("$g"); done

# context configs: index 0=4K, 1=8K, 2=16K
CTX_LABELS=(4k 8k 16k)
CTX_MODES=(short long long)            # src/eval.py: short ≈ 4K, long needs --max_length
CTX_MAXLEN_ARGS=("" "--max_length 8192" "--max_length 16384")

MAX_SAMPLES="${MAX_SAMPLES:-10}"
WARMUP="${WARMUP:-1}"

# ---- helpers ---------------------------------------------------------------
# run_eval <output_csv> <ckpt> <dataset> <ctx_mode> <maxlen_arg> <gamma>
run_eval () {
  local out="$1" ckpt="$2" ds="$3" mode="$4" maxlen="$5" gamma="$6"
  local samples_out="${out%.csv}_samples.csv"
  # Skip only if BOTH aggregate AND per-sample CSV already exist.
  # Older runs (pre commit 080b464) emitted only the aggregate; for error-bar
  # analysis we need the per-sample companion too — re-run those.
  if [ -f "$out" ] && [ -f "$samples_out" ]; then
    echo "  [skip] $out + $samples_out both exist"
    return 0
  fi
  if [ -f "$out" ] && [ ! -f "$samples_out" ]; then
    echo "  [redo] $out exists but no $samples_out — re-running for per-sample data"
  fi
  # $maxlen is intentionally unquoted: empty string -> no arg, "--max_length N" -> two args.
  # shellcheck disable=SC2086
  python3 "$EVAL_PY" \
      --target_model "$TARGET" \
      --original_student "$ORIGINAL" \
      --trained_student "$ckpt" \
      --dataset "$ds" \
      --context "$mode" $maxlen \
      --gamma "$gamma" \
      --budgets "$BUDGETS" \
      --max_samples "$MAX_SAMPLES" \
      --warmup "$WARMUP" \
      --output_csv "$out"
}

banner () {
  echo ""
  echo "============================================================"
  echo " $1"
  echo "============================================================"
}

# ---- sanity ----------------------------------------------------------------
[ -f "$EVAL_PY" ] || { echo "ERROR: $EVAL_PY not found (run from repo root)"; exit 1; }
for c in "$CKPT_MAIN" "$CKPT_AONLY" "$CKPT_AC"; do
  [ -d "$c" ] || echo "WARN: checkpoint dir missing: $c"
done

START_TS="$(date +%s)"
echo "Start: $(date)"
echo "Main ckpt   : $CKPT_MAIN"
echo "A-only ckpt : $CKPT_AONLY"
echo "A+C ckpt    : $CKPT_AC"
echo "Datasets    : ${DATASETS[*]}"
echo "γ 4K        : ${GAMMAS_4K[*]}"
echo "γ 8K        : ${GAMMAS_8K[*]}"
echo "γ 16K       : ${GAMMAS_16K[*]}"
echo "Budgets     : $BUDGETS"

# ---- 1. MAIN ---------------------------------------------------------------
banner "1. MAIN — A+0.5C, 3 ds × 3 ctx, per-context γ grid × 4 budgets"
for i in 0 1 2; do
  LBL="${CTX_LABELS[$i]}"; MODE="${CTX_MODES[$i]}"; MAXLEN="${CTX_MAXLEN_ARGS[$i]}"
  case "$LBL" in
    4k)  CTX_GAMMAS=("${GAMMAS_4K[@]}")  ;;
    8k)  CTX_GAMMAS=("${GAMMAS_8K[@]}")  ;;
    16k) CTX_GAMMAS=("${GAMMAS_16K[@]}") ;;
  esac
  for DS in "${DATASETS[@]}"; do
    for G in "${CTX_GAMMAS[@]}"; do
      OUT="$RESULTS_ROOT/main/eval_${LBL}_${DS}_g${G}.csv"
      echo "[main] ctx=$LBL ds=$DS γ=$G -> $OUT"
      run_eval "$OUT" "$CKPT_MAIN" "$DS" "$MODE" "$MAXLEN" "$G"
    done
  done
done

# ---- 2. ABLATION (A-only) --------------------------------------------------
G_ABL=5
banner "2. ABLATION — A-only ckpt, 3 ds × 3 ctx, γ=$G_ABL"
for i in 0 1 2; do
  LBL="${CTX_LABELS[$i]}"; MODE="${CTX_MODES[$i]}"; MAXLEN="${CTX_MAXLEN_ARGS[$i]}"
  for DS in "${DATASETS[@]}"; do
    OUT="$RESULTS_ROOT/ablation_a_only/eval_${LBL}_${DS}_g${G_ABL}.csv"
    echo "[ablation_a_only] ctx=$LBL ds=$DS γ=$G_ABL -> $OUT"
    run_eval "$OUT" "$CKPT_AONLY" "$DS" "$MODE" "$MAXLEN" "$G_ABL"
  done
done

# ---- 3. λ SENSITIVITY (A+C) ------------------------------------------------
G_LAM=5
banner "3. λ SENSITIVITY — A+C ckpt, 3 ds × 3 ctx, γ=$G_LAM"
for i in 0 1 2; do
  LBL="${CTX_LABELS[$i]}"; MODE="${CTX_MODES[$i]}"; MAXLEN="${CTX_MAXLEN_ARGS[$i]}"
  for DS in "${DATASETS[@]}"; do
    OUT="$RESULTS_ROOT/lambda_ac/eval_${LBL}_${DS}_g${G_LAM}.csv"
    echo "[lambda_ac] ctx=$LBL ds=$DS γ=$G_LAM -> $OUT"
    run_eval "$OUT" "$CKPT_AC" "$DS" "$MODE" "$MAXLEN" "$G_LAM"
  done
done

END_TS="$(date +%s)"
banner "ALL DONE — elapsed $(( (END_TS - START_TS) / 60 )) min — results under $RESULTS_ROOT/"
