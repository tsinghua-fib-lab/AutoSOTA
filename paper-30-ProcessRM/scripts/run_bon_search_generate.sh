gpu=(0 1 2 3 4)
temperature=(0.6 0.7 0.8 0.9 1.0)
generate_model_dir="../../../../models/deepseek-math-7b-instruct"

for i in "${!gpu[@]}"; do
    CUDA_VISIBLE_DEVICES="${gpu[$i]}" python ../bon_search.py \
    --data_dir ../../../../math_dataset/MATH_test.jsonl \
    --save_dir "../bon_data/deepseek_${temperature[$i]}.json" \
    --generate_model_dir "$generate_model_dir" \
    --test_n 64 \
    --generate \
    --temperature "${temperature[$i]}" \
    --top_p 0.95 &
done