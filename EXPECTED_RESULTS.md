# Expected Results

This file lists the headline numbers reported in the BudgetDraft paper.
After running `make eval` (or `./run_full_experiments.sh`), the values you read out of `results/full/main/eval_*.csv` should match these within ±0.05× speedup / ±2pp acceptance (GPU non-determinism + per-sample seed variance).

If your numbers diverge by more than that, see [§ Sanity checks](#sanity-checks) at the bottom.

---

## AR baseline (Table 1)

Throughput (tok/s) of the verifier alone, averaged across multiple runs:

| Context | GS | LongBench | LWM |
|---|---|---|---|
| 4K  | 38.10 | 37.06 | 35.73 |
| 8K  | 30.40 | 30.10 | 29.51 |
| 16K | 19.41 | 19.67 | 19.43 |

Used as the denominator for all speedup numbers below.

---

## Table 1 — Best BudgetDraft per (ctx, ds, budget)

**Panel A: best speedup** (and the corresponding acceptance rate)

| Ctx | B | GS speedup / acc% | LongBench speedup / acc% | LWM speedup / acc% |
|---|---|---|---|---|
| 4K  | 256  | 5.31× / 67.98 | 5.55× / 67.44 | **6.54× / 79.37** |
| 4K  | 512  | 5.31× / 67.98 | 5.56× / 67.44 | 6.55× / 79.37 |
| 4K  | 1024 | 5.32× / 67.98 | 5.56× / 67.44 | 6.54× / 79.37 |
| 4K  | 2048 | 5.28× / 67.98 | 5.56× / 67.44 | **6.55× / 79.37** |
| 8K  | 256  | 4.26× / 51.76 | 2.13× / 20.43 | 4.43× / 55.11 |
| 8K  | 512  | 4.27× / 52.26 | 2.13× / 20.43 | 4.46× / 55.11 |
| 8K  | 1024 | 4.27× / 52.26 | 2.13× / 20.43 | 4.46× / 55.11 |
| 8K  | 2048 | 4.27× / 52.26 | 2.13× / 20.43 | **4.46× / 55.11** |
| 16K | 256  | 1.22× / 18.81 | 1.53× / 27.78 | 2.10× / 34.17 |
| 16K | 512  | 1.21× / 18.35 | 1.53× / 27.98 | 2.10× / 34.17 |
| 16K | 1024 | 1.21× / 18.03 | 1.53× / 27.98 | 2.10× / 34.17 |
| 16K | 2048 | 1.21× / 17.98 | 1.53× / 27.98 | **2.10× / 34.17** |

---

## Table 3 — Ablation at γ=5

L_A + 0.5·L_C (BudgetDraft) reported as **speedup vs AR / accept%**.

| Ctx | B | GS | LongBench | LWM |
|---|---|---|---|---|
| 4K  | 256  | 2.95× / 91.62 | 3.05× / 92.49 | 3.26× / 96.34 |
| 4K  | 512  | 2.97× / 91.62 | 3.07× / 92.49 | 3.26× / 96.34 |
| 4K  | 1024 | 2.98× / 91.62 | 3.07× / 92.49 | 3.26× / 96.34 |
| 4K  | 2048 | 2.97× / 91.62 | 3.06× / 92.49 | 3.27× / 96.34 |
| 8K  | 256  | 2.39× / 74.78 | 1.37× / 20.43 | 2.54× / 77.47 |
| 8K  | 512  | 2.39× / 75.22 | 1.37× / 20.43 | 2.54× / 77.47 |
| 8K  | 1024 | 2.40× / 75.22 | 1.38× / 20.43 | 2.54× / 77.47 |
| 8K  | 2048 | 2.39× / 75.22 | 1.38× / 20.43 | 2.54× / 77.47 |
| 16K | 256  | 1.22× / 18.81 | 1.53× / 27.78 | 1.94× / 66.48 |
| 16K | 512  | 1.22× / 18.35 | 1.53× / 27.98 | 1.89× / 62.15 |
| 16K | 1024 | 1.21× / 18.03 | 1.53× / 27.98 | 1.77× / 55.80 |
| 16K | 2048 | 1.21× / 17.98 | 1.53× / 27.98 | 1.52× / 33.92 |

Key claim: at 4K, acceptance is **budget-invariant** (all four budgets give the same Acc% per dataset). This is the central result.

---

## Table 2 — Comparison vs TriForce / EAGLE-3 on 8K and 16K LWM, γ=5

| Method | 8K speedup | 16K speedup | Drafter |
|---|---|---|---|
| AR | 1.00× | 1.00× | – |
| SD (sparse/full) | 1.19× | 0.78× | 68M |
| TriForce | 1.21× | 1.19× | 68M+7B |
| EAGLE-3 | 1.64× | 1.36× | draft head |
| **BudgetDraft (B=2048)** | **2.54×** | **1.52×** | **68M** |

---

## Figure 3 — Budget-averaged γ sensitivity peaks

Best (γ, speedup) per (ctx, ds) at B=256, averaged across the 4 budgets:

| Ctx | GS | LongBench | LWM |
|---|---|---|---|
| 4K  | γ=50: **5.29×** | γ=60: 5.60× | γ=65: **6.54×** |
| 8K  | γ=15: 2.72× | γ=5: 1.37× (monotone) | γ=15: 2.90× |
| 16K | γ=5: 1.21× (monotone) | γ=5: 1.54× (monotone) | γ=10: 1.87× |

5 of 9 cells have a clear interior γ peak. 4 cells (LongBench 8K, GS/LongBench 16K) are monotonically decreasing in γ — these contexts don't benefit from longer drafts.

---

## Sanity checks

If your numbers diverge from the table values:

1. **Check AR baseline**: `awk -F, '$1=="original" && $4=="b256" && $5==5 {print $0}' results/full/main/eval_8k_gs_g5.csv` should show throughput ≈ AR value above (within GPU noise). If your AR is way off, the verifier model or hardware is different.

2. **Check the trained checkpoint**: did the training loss converge to ~2.5–3.0 by step 5000? If much higher, training was incomplete or the data loader is broken.

3. **Check budget-invariance at 4K**: for any 4K cell, all 4 budgets should give **identical** accept_rate (e.g. 91.62 / 91.62 / 91.62 / 91.62 for 4K-GS γ=5). If they differ, the sparse-cache training (L_C branch) is not working — verify `--lam 0.5` was set, not `0.0`.

4. **Cross-check with `make smoke`**: a 2-sample run at γ=5 B=256 should reproduce the headline numbers within ~5% (small n).
