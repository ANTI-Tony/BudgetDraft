# BudgetDraft — Evaluation Code & Reproduction Guide

Evaluation code for the paper **"BudgetDraft: Acceptance-Aware Multi-View Training for Sparse-KV Speculative Decoding"**.

This repository ships:
- **Evaluation code** that emits both aggregate and per-sample CSV (for error bars)
- **Baseline code** for AR (autoregressive) and SD (sparse / full KV)
- **TriForce baseline** wrapper (auto-clones and patches upstream)
- **All experiment shell scripts** used to produce the paper's tables and figures

Reproduction uses **released checkpoints** (request access via the paper's contact address). Training code is not included in this release.

Designed to run on a single **NVIDIA A100 80GB GPU**.

---

## 1. Hardware & expected wall-clock

| Job | GPU mem | Wall-clock |
|---|---|---|
| Full evaluation (78 main + 9 ablation + 9 lambda configs) | ~22 GB | ~5.5 h |
| TriForce baseline (7 combos) | ~22 GB | ~1.5 h |

A100 80GB is the reference. Smaller GPUs (A100 40GB, H100 80GB) should also work for evaluation.

---

## 2. Environment setup

### 2.1 System

- Linux x86_64 with **CUDA 12.4** + driver matching CUDA 12.4
- Python 3.10 or 3.11
- ~50 GB free disk for HuggingFace cache (YaRN-Llama-2-7B-128K weights ~14 GB + datasets)

### 2.2 BudgetDraft evaluation (transformers 4.44.2)

```bash
pip install -r requirements.txt

# optional, for faster inference
pip install flash-attn --no-build-isolation
```

**Why these pins:**
- `transformers==4.44.2` — supports YaRN-Llama and our DynamicCache usage. 4.50+ requires torch ≥ 2.5 (incompatible with our torch 2.4.1).
- `datasets==2.18.0` — newer versions (≥3.x) dropped support for script-based datasets, breaking PG-19 loading.
- `torch==2.4.1` — matches the CUDA 12.4 prebuilt flash-attn 2.8 wheel.

### 2.3 TriForce baseline (separate env recommended)

TriForce pins `transformers==4.37.2` which is incompatible with the BudgetDraft eval. `run_triforce_compare.sh` creates a dedicated venv (default `$PWD/triforce_venv`) and patches TriForce's `modeling_llama.py` and `data/dataset.py` automatically.

---

## 3. Datasets

Three datasets, auto-downloaded via HuggingFace `datasets` on first use:

| Dataset | HF repo | Local fallback |
|---|---|---|
| GS (PG-19) | `pg19` (script) | `data/pg19_test.jsonl` (included) |
| LongBench QMSum | `THUDM/LongBench` | downloads `data.zip` and extracts `qmsum.jsonl` |
| LWM (NarrativeQA) | `deepmind/narrativeqa` | streamed (20 fixed indices) |

Set `HF_HOME` if you want them in a specific location:

```bash
export HF_HOME=$HOME/.cache/huggingface
```

Models downloaded on first use:
- `NousResearch/Yarn-Llama-2-7b-128k` (~14 GB, fp16, verifier)
- `JackFram/llama-68m` (~270 MB, fp32, untrained drafter — used as the "original" baseline row)

---

## 4. Reproduction steps

### 4.1 Quick smoke test (~5 minutes)

Verifies the eval pipeline works end-to-end on the untrained drafter:

```bash
make smoke
```

Or directly:

```bash
python3 src/eval.py \
  --target_model NousResearch/Yarn-Llama-2-7b-128k \
  --original_student JackFram/llama-68m \
  --trained_student JackFram/llama-68m \
  --dataset gs --context short --gamma 5 \
  --budgets "256" --max_samples 2 --warmup 1 \
  --output_csv /tmp/smoke.csv
```

Expect: two CSV files (`/tmp/smoke.csv` aggregate + `/tmp/smoke_samples.csv` per-sample), no errors.

### 4.2 Full evaluation from released checkpoints

Place the released checkpoints in a directory with this layout:

```
<your_ckpt_dir>/
├── main/final/         (A+0.5C — paper's main checkpoint)
├── aonly/final/        (A-only ablation)
├── ac/final/           (A+C, lambda=1 ablation)
└── budget1024/final/   (fixed-budget B=1024 ablation, optional)
```

Each `final/` directory is a standard HuggingFace checkpoint (`config.json` + weights + tokenizer). Total ~1.1 GB.

**One-command eval:**

```bash
make eval-from-release CHECKPOINTS=/path/to/your_ckpt_dir
# or directly:
bash scripts/eval_from_release.sh /path/to/your_ckpt_dir
```

This runs both phases:
1. `run_full_experiments.sh` — 96 configs (main + A-only + A+C)
2. inline loop — 9 configs (fixed-budget ablation, only if `budget1024/` is present)

Output ends up in `results/full/main/` and `results/full/main_1024only/`. Compare against `EXPECTED_RESULTS.md`.

Resume-safe: any config with both `eval_*.csv` and `eval_*_samples.csv` is skipped.

### 4.3 TriForce baseline

```bash
./run_triforce_compare.sh
```

7 combos at TriForce defaults (budget=2048 for 4K, 4096 for 8K/16K, chunk=8, draft=256).
Output: `$TF_REPO_DIR/results/triforce_compare/summary.csv` (default `$TF_REPO_DIR=$PWD/triforce_baseline`).

### 4.4 Optional: extend γ sweep at 4K (paper Figure 3 main view)

Default `GAMMAS_4K` in `run_full_experiments.sh` covers γ=5..60. To extend to γ=80 like paper Figure 3:

```bash
for G in 65 70 75 80; do
  for DS in gs longbench_packed_qmsum lwm; do
    python3 src/eval.py \
      --target_model NousResearch/Yarn-Llama-2-7b-128k \
      --original_student JackFram/llama-68m \
      --trained_student "$CKPT_MAIN" \
      --dataset "$DS" --context short \
      --gamma "$G" --budgets "256,512,1024,2048" \
      --max_samples 10 --warmup 1 \
      --output_csv "results/full/main/eval_4k_${DS}_g${G}.csv"
  done
done
```

~36 minutes total.

---

## 5. Evaluation hyperparameters

| Setting | Value | Notes |
|---|---|---|
| Verifier | YaRN-Llama-2-7B-128K (fp16, full KV) | from NousResearch |
| Drafter | llama-68m (fp32, sparse KV) | from JackFram |
| Continuation length | 256 tokens | `--max_new_tokens 256` |
| Samples per cell | 10 (1 warmup discarded) | `--max_samples 10 --warmup 1` |
| Chunk size for sparse cache | 8 | TriForce-style chunked retrieval |
| Greedy decoding | top-1 verifier match | required for output-identical SD |

---

## 6. Expected results

Paper Table 1 (Best BudgetDraft speedup, panel A, B=256) headline numbers — reproduce within ±0.05× speedup, ±2pp acceptance:

| Context | GS | LongBench | LWM |
|---|---|---|---|
| 4K  | 5.31×/67.98% | 5.55×/67.44% | **6.54×/79.37%** |
| 8K  | 4.26×/51.76% | 2.13×/20.43% | 4.43×/55.11% |
| 16K | 1.22×/18.81% | 1.53×/27.78% | 2.10×/34.17% |

AR baselines (Table 1 — averaged over multiple runs):

| Context | GS | LongBench | LWM |
|---|---|---|---|
| 4K  | 38.10 | 37.06 | 35.73 |
| 8K  | 30.40 | 30.10 | 29.51 |
| 16K | 19.41 | 19.67 | 19.43 |

After running `run_full_experiments.sh`, compute speedups from `results/full/main/eval_*.csv`. See `EXPECTED_RESULTS.md` for the full sanity-check matrix.

---

## 7. File map

```
.
├── README.md                       # this file
├── EXPECTED_RESULTS.md             # paper headline numbers + sanity checks
├── Makefile                        # 1-command targets: make check / smoke / eval / triforce
├── requirements.txt                # pinned deps for evaluation
│
├── src/
│   ├── eval.py                     # main eval; emits aggregate + _samples.csv pair
│   ├── data_loader.py              # GS / LongBench / LWM dataset loaders
│   ├── AR.py                       # autoregressive baseline
│   ├── SD.py                       # SD (sparse/full) baseline
│   ├── speculative/                # speculative decoding + sparse cache
│   ├── tree/                       # benchmark utilities shared with AR.py
│   └── llama/                      # llama utilities (RoPE handling)
│
├── data/
│   └── pg19_test.jsonl             # local PG-19 test split (used by GS)
│
├── scripts/
│   ├── check_env.sh                # `make check` — verify env versions
│   ├── eval_from_release.sh        # orchestrates eval from downloaded checkpoints
│   ├── prepare_data.py             # offline data prep utilities
│   ├── clone_triforce.sh           # clones TriForce upstream
│   ├── download_models.sh          # pre-fetches HF models
│   └── patch_triforce.sh           # applies TriForce compatibility patches
│
└── run_full_experiments.sh         # main + ablation + lambda evaluation (96 configs)
    run_triforce_compare.sh         # TriForce baseline (own venv, 7 combos)
```

### One-command workflow

```bash
make check              # verify environment (no GPU needed)
make smoke              # ~5-min functional test (needs GPU + HF cache)
make eval-from-release CHECKPOINTS=/path/to/ckpts   # ~6 h — full eval matrix
make triforce           # ~1.5 h — TriForce baseline (separate venv)
```

---

## 8. Troubleshooting

**`ModuleNotFoundError: No module named 'transformers'`**: re-run `pip install -r requirements.txt`. If using virtualenv with `--system-site-packages`, ensure the venv is activated.

**`RuntimeError: Dataset scripts are no longer supported, but found pg19.py`**: your `datasets` is ≥3.0. Downgrade to `datasets==2.18.0`.

**Eval CSV produces 8 rows but no `_samples.csv`**: your `src/eval.py` is out of date. Pull the latest revision to get the per-sample emission feature.

**TriForce `topk k=... out of range`**: at 4K context with the default `--budget 4096`, retrieval needs more chunks than the 4K prefill provides. Use `--budget 2048` for 4K (script already does this).

**TriForce `accept rate: 0` at every sample**: the cos/sin squeeze patch failed; check `models/modeling_llama.py.orig` exists and re-run with the latest `run_triforce_compare.sh`.

---

## 9. Citation

If you use this code, please cite the BudgetDraft paper (citation TBA after acceptance). Baseline implementations cite their original works (TriForce, EAGLE-3, llama-68m, YaRN).

Code derived from / based on:
- [Infini-AI-Lab/TriForce](https://github.com/Infini-AI-Lab/TriForce) (TriForce baseline)
- HuggingFace transformers Llama implementation
