# BudgetDraft — Reproduction Guide

Reproduce the BudgetDraft paper (acceptance-aware multi-view sparse training for speculative decoding).

This repo includes:
- **Training code** for BudgetDraft and its ablations (A-only, A+0.5C, A+C, fixed-budget)
- **Evaluation code** that emits both aggregate and per-sample CSV (for error bars)
- **Baseline code** for AR, SD(sparse/full), TriForce, EAGLE-3
- **All experiment shell scripts** used in the paper

Designed to run on a single **NVIDIA A100 80GB GPU**.

---

## 1. Hardware & expected wall-clock

| Job | GPU mem | Wall-clock |
|---|---|---|
| Main checkpoint training (`run_train_phase_a.sh`, 5000 steps) | ~32 GB | ~5 h |
| Full evaluation (78 main + 9 ablation + 9 lambda configs) | ~22 GB | ~5.5 h |
| TriForce baseline (7 combos) | ~22 GB | ~1.5 h |

A100 80GB is the reference. Smaller GPUs (e.g. A100 40GB, H100 80GB) should work for evaluation; training at `seq_len=16384` needs ≥40 GB.

---

## 2. Environment setup

### 2.1 System

- Linux x86_64 with **CUDA 12.4** + driver matching CUDA 12.4
- Python 3.10 or 3.11
- ~50 GB free disk for HuggingFace cache (YaRN-Llama-2-7B-128K weights ~14 GB + datasets)

### 2.2 BudgetDraft training & evaluation (transformers 4.44.2)

```bash
# Pinned versions — verified working
pip install \
  "torch==2.4.1" \
  "transformers==4.44.2" \
  "datasets==2.18.0" \
  "accelerate>=0.27" \
  "sentencepiece" "protobuf" "huggingface_hub" \
  "termcolor" "tqdm" "numpy<2"

# (optional, for faster training) flash-attn matching the torch wheel
pip install flash-attn --no-build-isolation
```

**Why these pins:**
- `transformers==4.44.2` — supports YaRN-Llama and our DynamicCache usage. 4.50+ requires torch ≥ 2.5 (incompatible with our torch 2.4.1).
- `datasets==2.18.0` — newer versions (≥3.x) dropped support for script-based datasets, breaking PG-19 loading.
- `torch==2.4.1` — matches the CUDA 12.4 prebuilt flash-attn 2.8 wheel.

### 2.3 TriForce baseline (separate env recommended)

TriForce pins `transformers==4.37.2` which is incompatible with the BudgetDraft eval. Use a separate venv:

```bash
python -m venv --system-site-packages /workspace/tf/triforce_venv
source /workspace/tf/triforce_venv/bin/activate
pip install "transformers==4.37.2" "datasets==2.18.0" termcolor protobuf accelerate sentencepiece huggingface_hub tqdm
```

`run_triforce_compare.sh` handles this automatically (creates the venv, patches TriForce's `modeling_llama.py` and `data/dataset.py`).

---

## 3. Datasets

Three datasets, automatically downloaded via HuggingFace `datasets` on first use (cached under `$HF_HOME`):

| Dataset | HF repo | Used as | Local fallback |
|---|---|---|---|
| GS (PG-19) | `pg19` (script) | training + eval | `data/pg19_test.jsonl` (included) |
| LongBench QMSum | `THUDM/LongBench` | eval | downloads `data.zip` and extracts `qmsum.jsonl` |
| LWM (NarrativeQA) | `deepmind/narrativeqa` | eval | streamed |

Set `HF_HOME` if you want them in a specific location:

```bash
export HF_HOME=/workspace/tf/hf_cache
export TRANSFORMERS_CACHE=/workspace/tf/hf_cache
```

Models downloaded on first use:
- `NousResearch/Yarn-Llama-2-7b-128k` (~14 GB, fp16, verifier)
- `JackFram/llama-68m` (~270 MB, fp32, untrained drafter)

---

## 4. Reproduction steps

### 4.1 Quick smoke test (5 minutes, no training)

Verifies the eval pipeline works end-to-end:

```bash
cd sd_code/hl
python3 eval_tinydraft.py \
  --target_model NousResearch/Yarn-Llama-2-7b-128k \
  --original_student JackFram/llama-68m \
  --trained_student JackFram/llama-68m \
  --dataset gs --context short --gamma 5 \
  --budgets "256" --max_samples 2 --warmup 1 \
  --output_csv /tmp/smoke.csv
```

Expect: two CSV files (`/tmp/smoke.csv` aggregate + `/tmp/smoke_samples.csv` per-sample), no errors.

### 4.2 Train BudgetDraft (main checkpoint)

```bash
# A+0.5C with multi-view budget sampling — the headline checkpoint
./run_train_phase_a.sh
```

Output: `/workspace/tf/checkpoints/tinydraft_phase_a_16k/final/`

Details:
- 5000 steps, AdamW lr=1e-5, cosine schedule
- seq_len=16384, prefix=16128, continuation=256
- Random budget sampling from {256, 512, 1024, 2048} with weights {0.4, 0.3, 0.2, 0.1}
- λ=0.5 (L = L_A + 0.5·L_C)

### 4.3 Train ablation checkpoints

```bash
./run_train_phase_a_only.sh   # L_A only (no sparse-cache branch)
./run_train_1024only.sh        # L_A + 0.5·L_C with fixed B=1024 (no multi-view)
./run_train_phase_abc.sh       # L_A + L_C  (λ=1.0)
```

### 4.4 Full evaluation

After training completes:

```bash
# Main + ablation + lambda — 96 configs total
./run_full_experiments.sh
```

Output structure:

```
results/full/
├── main/                       # 78 configs, A+0.5C ckpt
│   ├── eval_4k_gs_g5.csv       # aggregate (one row per student × budget)
│   ├── eval_4k_gs_g5_samples.csv   # per-sample (for error bars)
│   └── ...                     # γ ∈ {5..60 for 4K, 5..30 for 8K, 5..40 for 16K}
├── ablation_a_only/   (9 configs, γ=5, A-only ckpt)
└── lambda_ac/          (9 configs, γ=5, A+C ckpt)
```

Resume-safe: any config with both `eval_*.csv` and `eval_*_samples.csv` is skipped.

### 4.5 TriForce baseline

```bash
./run_triforce_compare.sh
```

7 combos at TriForce defaults (budget=2048 for 4K, 4096 for 8K/16K, chunk=8, draft=256).
Output: `/workspace/tf/triforce-reproduce/results/triforce_compare/summary.csv`

### 4.6 Optional: extend γ sweep at 4K (paper Figure 3 main view)

Default `GAMMAS_4K` covers γ=5..60. To extend to γ=80 like paper Figure 3:

```bash
for G in 65 70 75 80; do
  for DS in gs longbench_packed_qmsum lwm; do
    python3 sd_code/hl/eval_tinydraft.py \
      --target_model NousResearch/Yarn-Llama-2-7b-128k \
      --original_student JackFram/llama-68m \
      --trained_student /workspace/tf/checkpoints/tinydraft_phase_a_16k/final \
      --dataset "$DS" --context short \
      --gamma "$G" --budgets "256,512,1024,2048" \
      --max_samples 10 --warmup 1 \
      --output_csv "results/full/main/eval_4k_${DS}_g${G}.csv"
  done
done
```

~36 minutes total.

---

## 5. Hyperparameters (Table reproducer)

For paper Table 1 / Table 3 / Figure 3, the exact settings are:

| Setting | Value | Notes |
|---|---|---|
| Verifier | YaRN-Llama-2-7B-128K (fp16, full KV) | from NousResearch |
| Drafter | llama-68m (fp32, sparse KV) | from JackFram |
| Continuation length | 256 tokens | `--max_new_tokens 256` |
| Samples per cell | 10 (1 warmup discarded) | `--max_samples 10 --warmup 1` |
| Chunk size for sparse cache | 8 | TriForce-style chunked retrieval |
| Greedy decoding | top-1 verifier match | required for output-identical SD |
| Training steps | 5000 | AdamW, lr=1e-5, weight_decay=0.01 |
| Warmup steps | 150 (linear) → cosine to 0 | |
| Sequence length (training) | 16384 (prefix 16128 + continuation 256) | |
| Multi-view budget weights | {256: 0.4, 512: 0.3, 1024: 0.2, 2048: 0.1} | |
| λ (sparse loss weight) | 0.5 | A+0.5C variant |
| Gradient clip | 1.0 | |

---

## 6. Expected results

Paper Table 1 (Best BudgetDraft speedup, panel A, B=256) headline numbers — reproduce within ±0.05× speedup, ±2pp acceptance:

| Context | GS | LongBench | LWM |
|---|---|---|---|
| 4K  | 5.31×/67.98% | 5.55×/67.44% | **6.54×/79.37%** |
| 8K  | 4.26×/51.76% | 2.13×/20.43% | 4.43×/55.11% |
| 16K | 1.22×/18.81% | 1.53×/27.78% | 2.10×/34.17% |

AR baselines used (Table 1 — averaged over multiple runs):

| Context | GS | LongBench | LWM |
|---|---|---|---|
| 4K  | 38.10 | 37.06 | 35.73 |
| 8K  | 30.40 | 30.10 | 29.51 |
| 16K | 19.41 | 19.67 | 19.43 |

After running `run_full_experiments.sh`, compute these from `results/full/main/eval_*.csv`.

---

## 7. File map

```
sd_code/hl/
├── train_tinydraft.py       # main training script (supports --fixed_budget, --full_cache_only)
├── eval_tinydraft.py        # evaluation; emits aggregate + _samples.csv pair
├── data_loader.py           # GS / LongBench / LWM dataset loaders
├── AR.py                    # standard autoregressive baseline
├── SD.py                    # standard SD (sparse/full) baseline
├── speculative/             # speculative decoding implementation + sparse cache
└── llama/                   # modified llama modeling utilities (RoPE handling)

# Reproduction shell scripts (top-level)
run_train_phase_a.sh              # train A+0.5C (paper main ckpt)
run_train_phase_a_only.sh         # ablation: L_A only
run_train_phase_abc.sh            # ablation: L_A + 1.0·L_C (λ=1)
run_train_1024only.sh             # ablation: L_A + 0.5·L_C, fixed B=1024
run_full_experiments.sh           # main + ablation + lambda evaluation (96 configs)
run_triforce_compare.sh           # TriForce baseline (separate venv, 7 combos)

# Data
data/
├── pg19_test.jsonl                # local copy of PG-19 test split (used by GS)
└── validation-00000-of-00001.jsonl  # Dolly validation (used by short-context tests)
```

---

## 8. Troubleshooting

**`ModuleNotFoundError: No module named 'transformers'`** after switching pods/envs: re-run the pip install in §2.2. If using virtualenv with `--system-site-packages`, ensure the venv is activated.

**`RuntimeError: Dataset scripts are no longer supported, but found pg19.py`**: your `datasets` is ≥3.0. Downgrade to `datasets==2.18.0`.

**Eval CSV produces 8 rows but no `_samples.csv`**: your `sd_code/hl/eval_tinydraft.py` is the old version (pre-commit `080b464`). `git pull` to get the per-sample emission feature.

**Training fails at first step with `RuntimeError: The expanded size of the tensor ... at non-singleton dimension 3` in attention backward**: gradient checkpointing conflicts with `DynamicCache` mutation in the L_C branch. Remove `--gradient_checkpointing` from your train script (paper's main checkpoint did not use it).

**TriForce `topk k=... out of range`**: at 4K context with the default `--budget 4096`, retrieval needs more chunks than the 4K prefill provides. Use `--budget 2048` for 4K (script already does this).

**TriForce `accept rate: 0` at every sample**: the cos/sin squeeze patch failed; check `models/modeling_llama.py.orig` exists and re-run with the latest `run_triforce_compare.sh`.

---

## 9. Citation

If you use this code, please cite the BudgetDraft paper (citation TBA after acceptance). Baseline implementations cite their original works (TriForce, EAGLE-3, llama-68m, YaRN).

Code derived from / based on:
- [Infini-AI-Lab/TriForce](https://github.com/Infini-AI-Lab/TriForce) (TriForce baseline)
- HuggingFace transformers Llama implementation
