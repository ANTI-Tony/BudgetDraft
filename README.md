# BudgetDraft — Evaluation Code & Reproduction Guide

Evaluation code for the paper **"BudgetDraft: Acceptance-Aware Multi-View Training for Sparse-KV Speculative Decoding"**.

This repository ships:
- Evaluation code that emits both aggregate and per-sample CSV (for error bars)
- AR (autoregressive) reference and SD (sparse / full KV) entry points
- All experiment shell scripts used to produce the paper's tables and figures

**Released checkpoints:** <https://huggingface.co/qwe123wjb/BudgetDraft-checkpoints>

```bash
hf download qwe123wjb/BudgetDraft-checkpoints --local-dir ./ckpts
make eval-from-release CHECKPOINTS=./ckpts
```

Training code is not included in this release. Designed to run on a single **NVIDIA A100 80GB GPU**.

For baseline comparisons referenced in the paper, fetch the upstream implementations directly:
- TriForce: <https://github.com/Infini-AI-Lab/TriForce>
- EAGLE-3: <https://github.com/SafeAILab/EAGLE>

---

## 1. Hardware & expected wall-clock

| Job | GPU mem | Wall-clock |
|---|---|---|
| Full evaluation (78 main + 9 ablation + 9 lambda configs) | ~22 GB | ~5.5 h |

A100 80GB is the reference. Smaller GPUs (A100 40GB, H100 80GB) should also work for evaluation.

---

## 2. Environment setup

### 2.1 System

- Linux x86_64 with **CUDA 12.4** + driver matching CUDA 12.4
- Python 3.10 or 3.11
- ~50 GB free disk for HuggingFace cache (YaRN-Llama-2-7B-128K weights ~14 GB + datasets)

### 2.2 Python packages

```bash
pip install -r requirements.txt

# optional, for faster inference
pip install flash-attn --no-build-isolation
```

**Why these pins:**
- `transformers==4.44.2` — supports YaRN-Llama and our DynamicCache usage. 4.50+ requires torch ≥ 2.5 (incompatible with our torch 2.4.1).
- `datasets==2.18.0` — newer versions (≥3.x) dropped support for script-based datasets, breaking PG-19 loading.
- `torch==2.4.1` — matches the CUDA 12.4 prebuilt flash-attn 2.8 wheel.

---

## 3. Datasets

All three datasets used in the paper are shipped under `data/`. The loader prefers the local jsonl over any HuggingFace download, so evaluation runs offline once the repo is checked out.

| Dataset | Local file | Size | Source |
|---|---|---|---|
| GS (PG-19)       | `data/gs.jsonl`        | ~40 MB | PG-19 test split |
| LongBench QMSum  | `data/longbench.jsonl` | ~12 MB | THUDM/LongBench (`qmsum.jsonl`) |
| LWM (NarrativeQA)| `data/lwm.jsonl`       | ~19 MB | deepmind/narrativeqa, 20 fixed indices |

If any local file is missing the loader falls back to a HuggingFace download. Set `HF_HOME` to control where downloads land:

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

Released checkpoints are on Hugging Face: <https://huggingface.co/qwe123wjb/BudgetDraft-checkpoints>

```bash
# Download all three (~786 MB total) into ./ckpts
hf download qwe123wjb/BudgetDraft-checkpoints --local-dir ./ckpts
```

Repo layout:

```
ckpts/
├── main/    (A+0.5C — paper's main checkpoint)
├── aonly/   (A-only ablation)
└── ac/      (A+C, lambda=1 ablation)
```

Each subfolder is a standard HuggingFace checkpoint (`config.json` + `model.safetensors` + tokenizer files).

**One-command eval:**

```bash
make eval-from-release CHECKPOINTS=./ckpts
# or directly:
bash scripts/eval_from_release.sh ./ckpts
```

This runs `run_full_experiments.sh` over 96 configurations (main + A-only + A+C ablations).

Output ends up in `results/full/main/`. Compare against `EXPECTED_RESULTS.md`.

Resume-safe: any config with both `eval_*.csv` and `eval_*_samples.csv` is skipped.

### 4.3 Optional: extend γ sweep at 4K (paper Figure 3 main view)

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
| Chunk size for sparse cache | 8 | chunked retrieval over the drafter KV cache |
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
├── Makefile                        # 1-command targets: make check / smoke / eval
├── requirements.txt                # pinned deps for evaluation
│
├── src/
│   ├── eval.py                     # main eval; emits aggregate + _samples.csv pair
│   ├── data_loader.py              # GS / LongBench / LWM dataset loaders
│   ├── AR.py                       # autoregressive reference
│   ├── SD.py                       # SD (sparse/full) entry point
│   ├── speculative/                # speculative decoding + sparse cache
│   ├── tree/                       # benchmark utilities shared with AR.py
│   └── llama/                      # llama utilities (RoPE handling)
│
├── data/
│   ├── gs.jsonl                    # PG-19 test split
│   ├── longbench.jsonl             # LongBench QMSum
│   └── lwm.jsonl                   # NarrativeQA, 20 fixed sample indices
│
├── scripts/
│   ├── check_env.sh                # `make check` — verify env versions
│   ├── eval_from_release.sh        # orchestrates eval from downloaded checkpoints
│   └── download_models.sh          # pre-fetches HF models
│
└── run_full_experiments.sh         # main + ablation + lambda evaluation (96 configs)
```

### One-command workflow

```bash
make check              # verify environment (no GPU needed)
make smoke              # ~5-min functional test (needs GPU + HF cache)
make eval-from-release CHECKPOINTS=/path/to/ckpts   # ~6 h — full eval matrix
```

---

## 8. Troubleshooting

**`ModuleNotFoundError: No module named 'transformers'`**: re-run `pip install -r requirements.txt`. If using virtualenv with `--system-site-packages`, ensure the venv is activated.

**`RuntimeError: Dataset scripts are no longer supported, but found pg19.py`**: your `datasets` is ≥3.0. Downgrade to `datasets==2.18.0`.

**Eval CSV produces 8 rows but no `_samples.csv`**: your `src/eval.py` is out of date. Pull the latest revision to get the per-sample emission feature.

---

## 9. Citation

If you use this code, please cite the BudgetDraft paper (citation TBA after acceptance).
