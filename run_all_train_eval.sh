#!/usr/bin/env bash
# Train A+C, A-only, then evaluate all 3 models × 3 datasets × 3 contexts
set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

EVAL_CMD="python3 sd_code/hl/eval_tinydraft.py"
TARGET="NousResearch/Yarn-Llama-2-7b-128k"
ORIG="JackFram/llama-68m"

# ═══════════════════════════════════════
# 1. Train A+C (λ=1.0)
# ═══════════════════════════════════════
echo "========================================="
echo "  Training A+C (λ=1.0)"
echo "========================================="
python3 sd_code/hl/train_tinydraft.py \
    --teacher_model $TARGET \
    --student_model $ORIG \
    --seq_len 16384 --cont_len 256 --chunk_size 8 \
    --lam 1.0 --beta 0 \
    --lr 1e-5 --weight_decay 0.01 --warmup_steps 150 \
    --total_steps 5000 --grad_clip 1.0 \
    --log_interval 10 --save_interval 500 \
    --output_dir /workspace/tf/checkpoints/tinydraft_ac \
    --seed 42

# ═══════════════════════════════════════
# 2. Train A-only (full cache)
# ═══════════════════════════════════════
echo "========================================="
echo "  Training A-only"
echo "========================================="
python3 sd_code/hl/train_tinydraft.py \
    --teacher_model $TARGET \
    --student_model $ORIG \
    --seq_len 4056 --cont_len 256 --chunk_size 8 \
    --lam 0.5 --beta 0 \
    --lr 1e-5 --weight_decay 0.01 --warmup_steps 150 \
    --total_steps 5000 --grad_clip 1.0 \
    --log_interval 10 --save_interval 500 \
    --output_dir /workspace/tf/checkpoints/tinydraft_aonly \
    --full_cache_only \
    --seed 42

# ═══════════════════════════════════════
# 3. Evaluate all models
# ═══════════════════════════════════════
echo "========================================="
echo "  Evaluating all models"
echo "========================================="

MODELS=(
    "tinydraft_phase_a_16k:a05c"
    "tinydraft_ac:ac"
    "tinydraft_aonly:aonly"
)

DATASETS=("gs" "longbench_packed_qmsum" "lwm")

mkdir -p results

for MODEL_PAIR in "${MODELS[@]}"; do
    MODEL_DIR="${MODEL_PAIR%%:*}"
    MODEL_TAG="${MODEL_PAIR##*:}"
    CKPT="/workspace/tf/checkpoints/${MODEL_DIR}/final"

    echo ""
    echo "===== Model: ${MODEL_TAG} (${CKPT}) ====="

    for DS in "${DATASETS[@]}"; do
        # Short (4K)
        echo "--- ${MODEL_TAG} / ${DS} / short ---"
        $EVAL_CMD --target_model $TARGET --original_student $ORIG \
            --trained_student $CKPT --dataset $DS --context short \
            --gamma 3 --budgets "256,512,1024,2048,3800" \
            --max_samples 10 --warmup 1 \
            --output_csv "results/eval_${MODEL_TAG}_short_${DS}.csv"

        # 8K
        echo "--- ${MODEL_TAG} / ${DS} / 8K ---"
        $EVAL_CMD --target_model $TARGET --original_student $ORIG \
            --trained_student $CKPT --dataset $DS --context long \
            --max_length 8192 --gamma 3 --budgets "256,512,1024,2048" \
            --max_samples 10 --warmup 1 \
            --output_csv "results/eval_${MODEL_TAG}_8k_${DS}.csv"

        # 16K
        echo "--- ${MODEL_TAG} / ${DS} / 16K ---"
        $EVAL_CMD --target_model $TARGET --original_student $ORIG \
            --trained_student $CKPT --dataset $DS --context long \
            --max_length 16384 --gamma 3 --budgets "256,512,1024,2048" \
            --max_samples 10 --warmup 1 \
            --output_csv "results/eval_${MODEL_TAG}_16k_${DS}.csv"
    done
done

echo ""
echo "========================================="
echo "  All Done!"
echo "========================================="
