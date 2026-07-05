# RULER-only token-level recovery (Recall@10pct) examples.
#
# Dataset can be:
# - a RULER name (hotpotqa_long / niah_* / vt_*) resolved under data/ruler_multihop/<len>/.../validation.jsonl
# - a raw RULER JSONL path
# - an exp2 cache JSONL path (must contain metadata.needle_spans)

# Example: evaluate on exp2 cache
# CUDA_VISIBLE_DEVICES=0 python3 evaluations/attribution_recovery.py \
#   --model qwen-8B --model_path /opt/share/models/Qwen/Qwen3-8B/ \
#   --cuda 0 --num_examples 50 --attr_func ifr_multi_hop \
#   --dataset exp/exp2/data/hotpotqa.jsonl

# Example: evaluate on raw RULER JSONL
# CUDA_VISIBLE_DEVICES=0 python3 evaluations/attribution_recovery.py \
#   --model qwen-8B --model_path /opt/share/models/Qwen/Qwen3-8B/ \
#   --cuda 0 --num_examples 50 --attr_func ifr_multi_hop \
#   --dataset data/ruler_multihop/4096/hotpotqa_long/validation.jsonl
