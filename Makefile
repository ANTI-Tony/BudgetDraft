# BudgetDraft reproduction shortcuts.
# Wraps the 6 paper-active shell scripts behind named targets.
#
# Usage:
#   make check       # verify environment (CUDA, torch, transformers, datasets)
#   make smoke       # 5-min eval smoke test (no training needed)
#   make train       # train A+0.5C main checkpoint (~5 h)
#   make ablation    # train all 3 ablation checkpoints (~15 h sequential)
#   make eval        # full evaluation: 78 main + 9 ablation + 9 lambda (~5.5 h)
#   make triforce    # TriForce baseline, 7 combos (~1.5 h, separate venv)
#   make all         # train -> eval -> triforce, blocking, total ~12 h
#
# Override checkpoint path via env: CKPT_DIR=/path make train

.PHONY: check smoke train ablation eval eval-from-release triforce all clean-results help

CKPT_DIR ?= /workspace/tf/checkpoints/tinydraft_phase_a_16k
RESULTS_DIR ?= results/full

help:
	@echo "Targets: check | smoke | train | ablation | eval | triforce | all"
	@echo "See README §4 for details."

check:
	@bash scripts/check_env.sh

smoke:
	@echo "[smoke] running 2-sample eval at γ=5, B=256 (no training needed)..."
	cd sd_code/hl && python3 eval_tinydraft.py \
	  --target_model NousResearch/Yarn-Llama-2-7b-128k \
	  --original_student JackFram/llama-68m \
	  --trained_student JackFram/llama-68m \
	  --dataset gs --context short --gamma 5 \
	  --budgets "256" --max_samples 2 --warmup 1 \
	  --output_csv /tmp/smoke.csv
	@test -f /tmp/smoke_samples.csv && \
	  echo "✓ smoke OK — both /tmp/smoke.csv and /tmp/smoke_samples.csv produced" || \
	  (echo "✗ _samples.csv missing — your eval_tinydraft.py is the old (pre-080b464) version"; exit 1)

train:
	./run_train_phase_a.sh

ablation:
	./run_train_phase_a_only.sh
	./run_train_phase_abc.sh
	./run_train_1024only.sh

eval:
	./run_full_experiments.sh

# Evaluation-only path for users who downloaded the released checkpoints
# instead of re-training. Pass CHECKPOINTS=/path/to/dir
eval-from-release:
	@test -n "$(CHECKPOINTS)" || (echo "usage: make eval-from-release CHECKPOINTS=/path/to/dir"; exit 1)
	bash scripts/eval_from_release.sh "$(CHECKPOINTS)"

triforce:
	./run_triforce_compare.sh

all: train ablation eval triforce
	@echo "✓ full pipeline complete"

clean-results:
	@echo "WARNING: this removes $(RESULTS_DIR)/ — Ctrl-C in 5s to abort"
	@sleep 5
	rm -rf $(RESULTS_DIR)
