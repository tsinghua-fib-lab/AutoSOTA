
UI_GROUNDING_BENCH_BASE_DIR="datasets/UI-Grounding-Benchmarks"

MODEL_TYPE="focusui_3b"
MODEL_PATH="checkpoints/focusui_3b"
SAVE_PATH="eval_results/focusui_3b/ui_vision"

EVAL_VISUAL_REDUCT_RATIOS="0.00 0.50 0.70"

for drop in $EVAL_VISUAL_REDUCT_RATIOS; do
    echo "UI-Vision | drop=${drop} | device=cuda:0"
    python -m evaluation.ui_vision_eval \
        --model_type "${MODEL_TYPE}" \
        --model_name_or_path "${MODEL_PATH}" \
        --save_path "${SAVE_PATH}/drop_${drop}" \
        --data_path "${UI_GROUNDING_BENCH_BASE_DIR}/UI-Vision" \
        --topk 3 \
        --device "cuda:0" \
        --visual_reduct_ratio "${drop}" 
done


