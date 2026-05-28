#!/usr/bin/env bash
# Eval-only entrypoint for users who downloaded the released checkpoints
# (instead of re-training them). Orchestrates main + ablation + lambda eval
# (96 configs total).
#
# Usage:
#   bash scripts/eval_from_release.sh /path/to/downloaded/checkpoints
#
# Expected layout of the argument directory (matches the HF repo
# qwe123wjb/BudgetDraft-checkpoints):
#   <ckpt_root>/
#   ├── main/    (A+0.5C — main ckpt)
#   ├── aonly/   (A-only ablation)
#   └── ac/      (A+C  lambda=1 ablation)
#
# Total wall-clock: ~5.5 h on A100 80GB.

set -euo pipefail

if [ $# -lt 1 ]; then
  echo "usage: $0 <path/to/downloaded/checkpoints>" >&2
  exit 2
fi
CKPT_ROOT="$1"
[ -d "$CKPT_ROOT" ] || { echo "ERROR: $CKPT_ROOT not a directory"; exit 1; }

# Resolve absolute path so the rest of the script can chdir freely
CKPT_ROOT="$(cd "$CKPT_ROOT" && pwd)"

# ---- locate the 3 ckpts ------------------------------------------------------
MAIN="$CKPT_ROOT/main"
AONLY="$CKPT_ROOT/aonly"
AC="$CKPT_ROOT/ac"

missing=0
for c in "$MAIN" "$AONLY" "$AC"; do
  if [ ! -d "$c" ] || [ ! -f "$c/config.json" ]; then
    echo "  ✗ missing: $c"
    missing=1
  else
    echo "  ✓ found:   $c"
  fi
done
[ "$missing" -eq 0 ] || {
  echo "ERROR: required ckpts missing (main / aonly / ac)"
  echo "       each subfolder must contain config.json + weights"
  echo "       download with:"
  echo "         hf download qwe123wjb/BudgetDraft-checkpoints --local-dir <ckpt_root>"
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
echo " main (A+0.5C) + ablation (A-only) + lambda (A+C)"
echo " -> 96 configs via run_full_experiments.sh"
echo "============================================================"
./run_full_experiments.sh

echo
echo "============================================================"
echo " ALL DONE. Results: results/full/main/"
echo
echo " Sanity-check against EXPECTED_RESULTS.md."
echo "============================================================"
