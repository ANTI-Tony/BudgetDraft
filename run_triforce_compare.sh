#!/usr/bin/env bash
# run_triforce_compare.sh — TriForce 7-combo eval against TinyDraft.
#
# Combos (gs 8K/16K already done in prior run; this script covers the remaining 7):
#   4K : gs, longbench (lb), lwm   --prefill 3800   budget=128 draft=128 chunk=1
#   8K : longbench, lwm            --prefill 8064   budget=264 draft=128 chunk=1
#   16K: longbench, lwm            --prefill 16128  budget=512 draft=128 chunk=2
#
# Assumptions:
#   - Fresh pod or willing to (re)create a dedicated conda env: triforce_eval
#   - Clones TriForce upstream into /workspace/tf/triforce-reproduce
#   - Patches its modeling_llama.py:
#       (a) max_position_embeddings 131072 -> 16384
#       (b) squeeze cos/sin in apply_rotary_pos_emb (transformers 4.37.2 quirk)
#   - on_chip.py prints a line `Accept rate: X%, Speedup: Y×` — script greps that
#
# Output:
#   /workspace/tf/triforce-reproduce/results/triforce_compare/
#     <ctx>_<ds>.log              raw run logs
#     summary.csv                 ctx,ds,accept_pct,speedup,tokens_per_sec

set -euo pipefail

# ============== knobs (override via env) =====================================
TF_REPO_DIR="${TF_REPO_DIR:-/workspace/tf/triforce-reproduce}"
VENV_DIR="${VENV_DIR:-/workspace/tf/triforce_venv}"
HF_CACHE="${HF_CACHE:-/workspace/tf/hf_cache}"
RESULTS_DIR="${RESULTS_DIR:-$TF_REPO_DIR/results/triforce_compare}"
TRIFORCE_GIT="${TRIFORCE_GIT:-https://github.com/Infini-AI-Lab/TriForce.git}"
GEN_LEN="${GEN_LEN:-256}"
GAMMA="${GAMMA:-3}"
TOP_P="${TOP_P:-0.9}"
TEMP="${TEMP:-0.6}"

export HF_HOME="$HF_CACHE"
export TRANSFORMERS_CACHE="$HF_CACHE"
export HF_DATASETS_CACHE="$HF_CACHE/datasets"

mkdir -p "$RESULTS_DIR"

# ============== venv setup ===================================================
banner () { echo; echo "============================================================"; echo " $1"; echo "============================================================"; }

banner "1/4  python venv: $VENV_DIR"
PY_BIN="$(command -v python3 || command -v python)"
[ -n "$PY_BIN" ] || { echo "ERROR: no python3 on PATH"; exit 1; }
echo "  using $PY_BIN ($($PY_BIN --version))"

if [ ! -d "$VENV_DIR" ]; then
  # try venv first; fall back to virtualenv if ensurepip is missing
  "$PY_BIN" -m venv "$VENV_DIR" 2>/dev/null || {
    "$PY_BIN" -m pip install --user -q virtualenv
    "$PY_BIN" -m virtualenv "$VENV_DIR"
  }
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
echo "  venv python: $(which python)  ($(python --version))"

pip install -q --upgrade pip
# Install torch only if missing — installing it fresh in a venv can take a while
# and the pod's system python likely has a matching CUDA build available via
# --system-site-packages if you re-create the venv with that flag.
python -c "import torch" 2>/dev/null || pip install -q "torch>=2.1"
pip install -q "transformers==4.37.2" sentencepiece accelerate datasets
# flash-attn: prefer prebuilt wheel; build-from-source is slow.
python -c "import flash_attn" 2>/dev/null || pip install -q flash-attn --no-build-isolation || pip install -q flash-attn
python - <<'PY'
import torch, transformers
print(f"  torch={torch.__version__}, cuda={torch.version.cuda}, transformers={transformers.__version__}")
try:
    import flash_attn
    print(f"  flash_attn={flash_attn.__version__}")
except Exception as e:
    print(f"  flash_attn missing: {e}")
PY

# ============== clone + patch TriForce =======================================
banner "2/4  clone + patch TriForce"
if [ ! -d "$TF_REPO_DIR/.git" ]; then
  git clone "$TRIFORCE_GIT" "$TF_REPO_DIR"
fi
cd "$TF_REPO_DIR"
# install TriForce's own requirements if present
[ -f requirements.txt ] && pip install -q -r requirements.txt || true

MODELING="models/modeling_llama.py"
if [ ! -f "$MODELING" ]; then
  echo "ERROR: $TF_REPO_DIR/$MODELING not found — check repo layout."
  exit 1
fi

SENTINEL="# patched-for-yarn-16k"
if grep -q "$SENTINEL" "$MODELING"; then
  echo "  patches already applied (sentinel found)"
else
  cp "$MODELING" "$MODELING.orig"

  # (a) max_position 131072 -> 16384  (catches `max_position_embeddings=131072`
  #     and `max_position_embeddings = 131072` and JSON-style "max_position_embeddings": 131072)
  sed -i -E 's/(max_position_embeddings[[:space:]]*[:=][[:space:]]*)131072/\116384/g' "$MODELING"

  # (b) squeeze cos/sin in apply_rotary_pos_emb. transformers 4.37.2 ships:
  #         def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
  #             cos = cos.unsqueeze(unsqueeze_dim)
  #             sin = sin.unsqueeze(unsqueeze_dim)
  #     If TriForce calls it with already-unsqueezed cos/sin, the extra unsqueeze
  #     breaks the broadcast. Idempotent fix: collapse any leading length-1 dims
  #     before the unsqueeze.
  python - <<PY
import re, pathlib
p = pathlib.Path("$MODELING")
src = p.read_text()
needle = "def apply_rotary_pos_emb("
i = src.find(needle)
if i == -1:
    raise SystemExit("could not locate apply_rotary_pos_emb in $MODELING")
# find first 'cos = cos.unsqueeze' inside that function
j = src.find("cos = cos.unsqueeze", i)
if j == -1:
    raise SystemExit("expected 'cos = cos.unsqueeze' inside apply_rotary_pos_emb")
# inject squeeze lines right before that
indent = "    "
inject = (f"{indent}while cos.dim() > 2:\n"
          f"{indent}    cos = cos.squeeze(0)\n"
          f"{indent}while sin.dim() > 2:\n"
          f"{indent}    sin = sin.squeeze(0)\n")
src = src[:j] + inject + src[j:]
p.write_text(src)
print("  injected cos/sin squeeze guard")
PY

  echo "$SENTINEL" >> "$MODELING"
  echo "  patches applied; original saved as $MODELING.orig"
fi

# ============== run 7 combos =================================================
banner "3/4  run 7 combos"

# (ctx_label, prefill, budget, draft, chunk, dataset)
COMBOS=(
  "4k  3800  128 128 1 gs"
  "4k  3800  128 128 1 longbench"
  "4k  3800  128 128 1 lwm"
  "8k  8064  264 128 1 longbench"
  "8k  8064  264 128 1 lwm"
  "16k 16128 512 128 2 longbench"
  "16k 16128 512 128 2 lwm"
)

SUMMARY="$RESULTS_DIR/summary.csv"
echo "ctx,ds,accept_pct,speedup,tokens_per_sec,log" > "$SUMMARY"

for combo in "${COMBOS[@]}"; do
  read -r CTX PREFILL BUDGET DRAFT CHUNK DS <<< "$combo"
  LOG="$RESULTS_DIR/${CTX}_${DS}.log"
  banner "  $CTX / $DS  (prefill=$PREFILL budget=$BUDGET draft=$DRAFT chunk=$CHUNK)"

  python test/on_chip.py \
      --prefill "$PREFILL" \
      --gen_len "$GEN_LEN" \
      --budget "$BUDGET" \
      --chunk_size "$CHUNK" \
      --draft_cache_budget "$DRAFT" \
      --gamma "$GAMMA" \
      --top_p "$TOP_P" \
      --temp "$TEMP" \
      --dataset "$DS" \
      2>&1 | tee "$LOG"

  # Parse "Accept rate: X%, Speedup: Y×" and "tokens/sec" if present.
  ACCEPT=$(grep -Eo 'Accept[[:space:]]*rate[[:space:]]*[:=][[:space:]]*[0-9.]+%?' "$LOG" | tail -1 | grep -Eo '[0-9.]+' | head -1 || true)
  SPEED=$(grep -Eo 'Speedup[[:space:]]*[:=][[:space:]]*[0-9.]+[xX×]?' "$LOG" | tail -1 | grep -Eo '[0-9.]+' | head -1 || true)
  TPS=$(grep -Eo '[0-9.]+[[:space:]]*tok(en)?s?/s' "$LOG" | tail -1 | grep -Eo '[0-9.]+' | head -1 || true)

  echo "${CTX},${DS},${ACCEPT:-NA},${SPEED:-NA},${TPS:-NA},${LOG##*/}" >> "$SUMMARY"
  echo "  -> accept=${ACCEPT:-NA}%  speedup=${SPEED:-NA}x  tps=${TPS:-NA}"
done

# ============== summary ======================================================
banner "4/4  summary"
column -s, -t < "$SUMMARY" || cat "$SUMMARY"
echo
echo "Done. Summary: $SUMMARY"
echo "Logs:    $RESULTS_DIR/*.log"
