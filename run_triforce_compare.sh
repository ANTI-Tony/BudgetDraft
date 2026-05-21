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
#   - Patches data/dataset.py:
#       (c) make `lwm` respect args.prefill (upstream hard-codes 127*1024)
#       (d) add `longbench_packed_qmsum` branch mirroring the TinyDraft loader
#   - on_chip.py prints `average acceptance rate (NOT per token): X` (fraction)
#     and `[E2E Speedup]: Y` — script greps those.
#
# Output:
#   /workspace/tf/triforce-reproduce/results/triforce_compare/
#     <ctx>_<ds>.log              raw run logs
#     summary.csv                 ctx,ds,accept_rate (fraction),speedup,log

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
  # --system-site-packages: inherit the pod's pre-built torch + flash-attn
  # (matching the system CUDA toolchain). Avoids multi-GB re-downloads and
  # flash-attn source builds that take 10-30 minutes.
  "$PY_BIN" -m venv --system-site-packages "$VENV_DIR" 2>/dev/null || {
    "$PY_BIN" -m pip install --user -q virtualenv
    "$PY_BIN" -m virtualenv --system-site-packages "$VENV_DIR"
  }
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
echo "  venv python: $(which python)  ($(python --version))"
echo "  upgrading pip..."
pip install --upgrade pip 2>&1 | tail -1

# Pin transformers to 4.37.2 (TriForce requirement). Reuse system torch/flash-attn
# via --system-site-packages above. Re-install only if a version mismatch shows up.
echo "  installing transformers==4.37.2 + small deps..."
pip install "transformers==4.37.2" sentencepiece accelerate datasets 2>&1 | tail -3

# torch: install only if completely missing (rare with --system-site-packages).
python -c "import torch" 2>/dev/null || {
  echo "  torch missing, installing (this is slow)..."
  pip install "torch>=2.1" 2>&1 | tail -3
}
# flash-attn: same — only build if missing and we genuinely need it. TriForce's
# on_chip.py imports it directly, so without it the script will crash anyway.
python -c "import flash_attn" 2>/dev/null || {
  echo "  flash_attn missing — attempting prebuilt wheel..."
  pip install flash-attn --no-build-isolation 2>&1 | tail -3 || \
    echo "  WARN: flash-attn install failed; on_chip.py will likely fail to import"
}

python - <<'PY'
import torch, transformers
print(f"  torch={torch.__version__}, cuda={torch.version.cuda}, transformers={transformers.__version__}")
try:
    import flash_attn
    print(f"  flash_attn={flash_attn.__version__}")
except Exception as e:
    print(f"  flash_attn NOT importable: {e}")
PY

# ============== clone + patch TriForce =======================================
banner "2/4  clone + patch TriForce"

needs_clone=0
if [ ! -d "$TF_REPO_DIR" ]; then
  needs_clone=1
elif [ ! -f "$TF_REPO_DIR/test/on_chip.py" ] || [ ! -f "$TF_REPO_DIR/models/modeling_llama.py" ] || [ ! -f "$TF_REPO_DIR/data/dataset.py" ]; then
  echo "  $TF_REPO_DIR exists but is missing TriForce source files — wiping non-results contents and re-cloning"
  # Preserve any results/ subdir, blow everything else away
  if [ -d "$TF_REPO_DIR/results" ]; then
    mv "$TF_REPO_DIR/results" "${TF_REPO_DIR}.results.bak"
  fi
  # If $TF_REPO_DIR is a mount point, rm -rf the *contents* not the dir itself
  find "$TF_REPO_DIR" -mindepth 1 -maxdepth 1 -exec rm -rf {} +
  needs_clone=1
else
  echo "  reusing existing TriForce checkout at $TF_REPO_DIR"
fi

if [ "$needs_clone" = "1" ]; then
  if [ -d "$TF_REPO_DIR" ] && [ -z "$(ls -A "$TF_REPO_DIR" 2>/dev/null)" ]; then
    # empty dir — clone into it
    git clone "$TRIFORCE_GIT" "$TF_REPO_DIR.tmp"
    mv "$TF_REPO_DIR.tmp"/* "$TF_REPO_DIR.tmp"/.* "$TF_REPO_DIR/" 2>/dev/null || true
    rmdir "$TF_REPO_DIR.tmp"
  else
    git clone "$TRIFORCE_GIT" "$TF_REPO_DIR"
  fi
  # Restore any preserved results
  if [ -d "${TF_REPO_DIR}.results.bak" ]; then
    mv "${TF_REPO_DIR}.results.bak" "$TF_REPO_DIR/results"
  fi
fi

cd "$TF_REPO_DIR"
for f in models/modeling_llama.py data/dataset.py test/on_chip.py; do
  [ -f "$f" ] || { echo "ERROR: $TF_REPO_DIR/$f still missing after clone — bail"; exit 1; }
done
echo "  TriForce source verified"
# NOTE: deliberately NOT installing TriForce's requirements.txt — it pins
# torch==2.2.1 which conflicts with the pod's torch 2.4.1+cu124 (and the
# flash-attn 2.8.3 wheel built against it). The hard deps it actually needs
# (transformers==4.37.2, accelerate, datasets, sentencepiece, flash-attn,
# matching torch) are already installed above. Install any leftovers we know
# on_chip.py needs but the venv might be missing:
echo "  installing TriForce-specific deps (skipping their torch==2.2.1 pin)..."
pip install termcolor tiktoken 2>&1 | tail -3

# Heal a possibly-downgraded torch from a previous broken run.
# The venv has --system-site-packages, so uninstalling a venv-local torch
# re-exposes the system's torch 2.4.1+cu124 without re-downloading.
TORCH_VER=$(python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "missing")
if [[ ! "$TORCH_VER" =~ ^2\.4 ]]; then
  echo "  WARN: venv torch is $TORCH_VER (need 2.4.x). Uninstalling venv-local torch to expose system 2.4.1..."
  pip uninstall -y torch 2>&1 | tail -3
  NEW_VER=$(python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "missing")
  if [[ ! "$NEW_VER" =~ ^2\.4 ]]; then
    echo "  system torch also wrong ($NEW_VER) — installing 2.4.1+cu124 from pytorch index"
    pip install "torch==2.4.1" --index-url https://download.pytorch.org/whl/cu124 2>&1 | tail -3
  fi
fi
echo "  torch=$(python -c 'import torch;print(torch.__version__)')"

MODELING="models/modeling_llama.py"
if [ ! -f "$MODELING" ]; then
  echo "ERROR: $TF_REPO_DIR/$MODELING not found — check repo layout."
  exit 1
fi

SENTINEL="# patched-for-yarn-16k"
# Self-heal: if .orig exists but sentinel is missing, the previous patch run
# bailed mid-way. Restore from .orig and re-apply cleanly.
if [ -f "$MODELING.orig" ] && ! grep -q "$SENTINEL" "$MODELING"; then
  echo "  recovering from half-patched state (restoring .orig)"
  cp "$MODELING.orig" "$MODELING"
fi

if grep -q "$SENTINEL" "$MODELING"; then
  echo "  patches already applied (sentinel found)"
else
  cp "$MODELING" "$MODELING.orig"

  # (a) max_position 131072 -> 16384  (catches `=131072`, `: 131072`, `(131072)`)
  sed -i -E 's/(max_position_embeddings[[:space:]]*[:=(][[:space:]]*)131072/\116384/g' "$MODELING"
  sed -i -E 's/=131072([,)])/=16384\1/g' "$MODELING"

  # (b) squeeze cos/sin BEFORE the apply_rotary_pos_emb call. transformers 4.37.2
  #     internally does `cos = cos.unsqueeze(unsqueeze_dim)`. If TriForce passes
  #     cos/sin with extra leading length-1 dims, the broadcast breaks. TriForce
  #     imports apply_rotary_pos_emb from transformers, so we patch the call
  #     site here instead of the (vendored) function definition.
  # Quoted heredoc (<<'PY') so bash doesn't process backslash sequences inside.
  python - "$MODELING" <<'PY'
import pathlib, sys
p = pathlib.Path(sys.argv[1])
src = p.read_text()
needle = "cos, sin = self.rotary_emb"
sites = []
off = 0
while True:
    j = src.find(needle, off)
    if j == -1:
        break
    sites.append(j)
    off = j + 1
if not sites:
    raise SystemExit(f"could not locate any 'cos, sin = self.rotary_emb' call site in {p}")

for j in reversed(sites):
    eol = src.find("\n", j)
    line_start = src.rfind("\n", 0, j) + 1
    indent = ""
    k = line_start
    while k < len(src) and src[k] in " \t":
        indent += src[k]; k += 1
    inject = ("\n" + indent + "while cos.dim() > 2: cos = cos.squeeze(0)"
              + "\n" + indent + "while sin.dim() > 2: sin = sin.squeeze(0)")
    src = src[:eol] + inject + src[eol:]

p.write_text(src)
print(f"  injected cos/sin squeeze at {len(sites)} call site(s)")
PY

  echo "$SENTINEL" >> "$MODELING"
  echo "  patches applied to $MODELING; original saved as $MODELING.orig"
fi

# (c) data/dataset.py: make `lwm` respect `datalen` and add `longbench_packed_qmsum`
DATASET_PY="data/dataset.py"
# v3 sentinel: v2 fixed the bash heredoc issue but the injected longbench
# branch did `import zipfile, os, json` inside the function, shadowing the
# module-level `os` and causing UnboundLocalError in the older gs branch
# (which calls os.listdir before our inner import line is reached). v3
# removes the inner os/json shadowing.
DS_SENTINEL="# patched-for-tinydraft-eval-v3"
# Self-heal: restore from .orig if previous patch run bailed mid-way
if [ -f "$DATASET_PY.orig" ] && ! grep -q "$DS_SENTINEL" "$DATASET_PY"; then
  echo "  recovering from half-patched $DATASET_PY (restoring .orig)"
  cp "$DATASET_PY.orig" "$DATASET_PY"
fi
if [ -f "$DATASET_PY" ] && ! grep -q "$DS_SENTINEL" "$DATASET_PY"; then
  cp "$DATASET_PY" "$DATASET_PY.orig"
  # Quoted heredoc — bash does NOT process expansions or backslash escapes.
  # Path + sentinel passed via argv.
  python - "$DATASET_PY" "$DS_SENTINEL" <<'PY'
import re, pathlib, sys
p = pathlib.Path(sys.argv[1])
ds_sentinel = sys.argv[2]
src = p.read_text()

# 1) Rewrite the existing `lwm` branch to respect the datalen argument
#    (upstream hard-codes prefill=127*1024 and filters out short prompts).
new_lwm = '''    elif dataset_name == 'lwm':
        # patched: dynamic prefill from datalen (was hard-coded 127*1024)
        prefill = datalen if datalen else 127*1024
        try:
            stream = load_dataset("deepmind/narrativeqa", split="train", streaming=True)
        except Exception:
            stream = load_dataset("narrativeqa", split="train", streaming=True)
        idx_set = {0, 50, 300, 800, 950, 1100, 2150, 2450, 2550, 2750,
                   3350, 3400, 3600, 3900, 4000, 4100, 4200, 4400, 4500, 4550}
        max_idx = max(idx_set)
        collected = {}
        for i, item in enumerate(stream):
            if i in idx_set:
                collected[i] = item
            if i > max_idx:
                break
        tokenized_prompts = []
        for idx in sorted(idx_set):
            item = collected.get(idx)
            if item is None:
                continue
            doc = item.get('document')
            doc_text = doc.get('text', '') if isinstance(doc, dict) else (doc or item.get('text', ''))
            if not doc_text:
                continue
            book_tokens = tokenizer.encode(doc_text)[:max(prefill - 100, 1)]
            prompt = (
                "You are a helpful assistant. USER: Please read a part of the book below, "
                "and then give me the summary.\\n[start of the book]\\n"
                + tokenizer.decode(book_tokens, skip_special_tokens=True)
                + "\\n[end of the book]\\n\\nNow you have read it. Please summarize it for me.\\n\\nASSISTANT: "
            )
            ids = tokenizer.encode(prompt, return_tensors="pt")
            if ids.shape[-1] > prefill:
                ids = ids[:, :prefill]
            tokenized_prompts.append(ids)
        return tokenized_prompts

'''
m = re.search(r"    elif dataset_name == 'lwm':.*?(?=\n    elif |\n    else:)", src, re.S)
assert m, f"could not locate lwm branch in {p}"
src = src[:m.start()] + new_lwm + src[m.end()+1:]

# 2) Add a longbench_packed_qmsum branch right before the final `else:`
new_lb = '''    elif dataset_name == 'longbench_packed_qmsum':
        # patched: TinyDraft-parity QMSum loader
        # NOTE: os/json are imported at module top; do NOT re-import inside
        # this function or Python will treat them as locals across all branches
        # (UnboundLocalError when `gs` calls os.listdir before we reach an
        # inner `import os` line).
        prefill = datalen if datalen else 4096
        try:
            ds = load_dataset("THUDM/LongBench", "qmsum", split="test")
        except Exception:
            from huggingface_hub import hf_hub_download
            import zipfile
            zp = hf_hub_download(repo_id="THUDM/LongBench", filename="data.zip", repo_type="dataset")
            ed = os.path.join(os.path.dirname(zp), "longbench_extracted")
            if not os.path.exists(os.path.join(ed, "qmsum.jsonl")):
                with zipfile.ZipFile(zp, "r") as zf:
                    zf.extractall(ed)
            qp = None
            for r, _, fs in os.walk(ed):
                for fn in fs:
                    if fn == "qmsum.jsonl":
                        qp = os.path.join(r, fn); break
            ds = [json.loads(l) for l in open(qp)]
        tokenized_prompts = []
        for item in ds:
            if len(tokenized_prompts) >= 20:
                break
            ctx = item.get('context', '') or item.get('input', '')
            qry = item.get('input', '') if 'context' in item else ''
            text = ctx + ("\\n" + qry if qry else "")
            ids = tokenizer.encode(text, return_tensors="pt")
            if ids.shape[-1] < prefill // 2:
                continue
            if ids.shape[-1] > prefill:
                ids = ids[:, :prefill]
            tokenized_prompts.append(ids)
        return tokenized_prompts

'''
m2 = re.search(r"\n    else:\s*\n\s*raise Exception\(\"Dataset not found\"\)", src)
assert m2, "could not locate final else clause"
src = src[:m2.start()+1] + new_lb + src[m2.start()+1:]

p.write_text(src + "\n" + ds_sentinel + "\n")
print(f"  patched {p}: dynamic lwm + longbench_packed_qmsum")
PY
  echo "  data/dataset.py patched; original saved as $DATASET_PY.orig"
elif [ -f "$DATASET_PY" ]; then
  echo "  $DATASET_PY already patched (sentinel found)"
fi

# ============== run 7 combos =================================================
banner "3/4  run 7 combos"

# (ctx_label, prefill, budget, draft, chunk, dataset)
# Using TriForce's default budget=4096, draft=256, chunk=8 across all combos.
# This matches the previously-validated gs 8K (73.3%/1.41x) and gs 16K (71.8%/
# 1.38x) runs and avoids the spec-graph-cache overflow that occurs when
# budget < gen_len (256). At 4K, budget=4096 > prefill=3800 → TriForce
# effectively runs with full target KV cache (no on-chip sparsity); this is
# TriForce's degenerate case but still a fair upper-bound baseline.
COMBOS=(
  "4k  3800  4096 256 8 gs"
  "4k  3800  4096 256 8 longbench_packed_qmsum"
  "4k  3800  4096 256 8 lwm"
  "8k  8064  4096 256 8 longbench_packed_qmsum"
  "8k  8064  4096 256 8 lwm"
  "16k 16128 4096 256 8 longbench_packed_qmsum"
  "16k 16128 4096 256 8 lwm"
)

SUMMARY="$RESULTS_DIR/summary.csv"
echo "ctx,ds,accept_rate,speedup,log" > "$SUMMARY"

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

  # on_chip.py prints e.g.:
  #   average acceptance rate (NOT per token): 0.7178
  #   [E2E Speedup]: 1.38
  ACCEPT=$(grep -E 'average acceptance rate' "$LOG" | tail -1 | grep -Eo '[0-9]+\.[0-9]+' | tail -1 || true)
  SPEED=$(grep -E '\[E2E Speedup\]' "$LOG"        | tail -1 | grep -Eo '[0-9]+\.[0-9]+' | tail -1 || true)

  echo "${CTX},${DS},${ACCEPT:-NA},${SPEED:-NA},${LOG##*/}" >> "$SUMMARY"
  echo "  -> accept=${ACCEPT:-NA}  speedup=${SPEED:-NA}x"
done

# ============== summary ======================================================
banner "4/4  summary"
column -s, -t < "$SUMMARY" || cat "$SUMMARY"
echo
echo "Done. Summary: $SUMMARY"
echo "Logs:    $RESULTS_DIR/*.log"
