# BudgetDraft evaluation shortcuts.
# Reproduces the paper's evaluation matrix from released checkpoints.
#
# Usage:
#   make check             # verify environment (CUDA, torch, transformers, datasets)
#   make smoke             # ~5-min eval smoke test (2 samples)
#   make eval              # full evaluation: 78 main + 9 ablation + 9 lambda (~5.5 h)
#   make eval-from-release # eval using pre-downloaded checkpoints (CHECKPOINTS=/path)
#   make triforce          # TriForce baseline, 7 combos (~1.5 h, separate venv)

.PHONY: check smoke eval eval-from-release triforce clean-results help

RESULTS_DIR ?= results/full

help:
	@echo "Targets: check | smoke | eval | eval-from-release | triforce"
	@echo "See README for details."

check:
	@bash scripts/check_env.sh

smoke:
	@echo "[smoke] running 2-sample eval at gamma=5, B=256..."
	python3 src/eval.py \
	  --target_model NousResearch/Yarn-Llama-2-7b-128k \
	  --original_student JackFram/llama-68m \
	  --trained_student JackFram/llama-68m \
	  --dataset gs --context short --gamma 5 \
	  --budgets "256" --max_samples 2 --warmup 1 \
	  --output_csv /tmp/smoke.csv
	@test -f /tmp/smoke_samples.csv && \
	  echo "OK smoke — both /tmp/smoke.csv and /tmp/smoke_samples.csv produced" || \
	  (echo "FAIL _samples.csv missing — your src/eval.py is out of date"; exit 1)

eval:
	./run_full_experiments.sh

# Evaluation-only path for users who downloaded the released checkpoints.
# Pass CHECKPOINTS=/path/to/dir (containing main/, aonly/, ac/ subdirs).
eval-from-release:
	@test -n "$(CHECKPOINTS)" || (echo "usage: make eval-from-release CHECKPOINTS=/path/to/dir"; exit 1)
	bash scripts/eval_from_release.sh "$(CHECKPOINTS)"

triforce:
	./run_triforce_compare.sh

clean-results:
	@echo "WARNING: this removes $(RESULTS_DIR)/ — Ctrl-C in 5s to abort"
	@sleep 5
	rm -rf $(RESULTS_DIR)
