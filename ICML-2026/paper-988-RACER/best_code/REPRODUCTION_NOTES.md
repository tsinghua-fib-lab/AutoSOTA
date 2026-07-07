# RACER Reproduction Notes

## Summary
Successfully reproduced RACER-P (inverse probability nonconformity score) on GSM8K with MLP base router.
The reproduced metric (77.57% ± 1.45%) is within the CI bounds [76.7%, 79.1%] of the paper's RouterDC+RACER-P result (77.9% ± 1.2%).

## Key Results
- **Primary Metric**: Accuracy = 77.57% ± 1.45% (Weighted-p_true aggregation, α=0.03, 100 trials)
- **Base Router (MLP)**: 75.07% ± 1.19% (paper MLPR: 75.1% ± 1.2%)
- **Risk Control**: 2.91% ± 0.87% (target α=0.03, controlled below threshold)
- **Avg Set Size**: 2.82 models (excluding null model)
- **Additional α=0.01 run**: 78.30% ± 1.16% (Weighted-binary_confidence)

## Router Substitution
Paper specifies `base_router=RouterDC`, but RouterDC code is not included in this repository (only KNN is implemented). We implemented an MLP router (MLPR) matching the paper's description: hidden_size=256, BCEWithLogitsLoss, lr=1e-4, 100 epochs, AdamW with weight_decay=0.01. The paper reports MLPR+RACER-P = 77.8% ± 1.2% on GSM8K, and our MLP result (77.57%) is within 0.23pp of that.

## Environment
- Base image: docker.1ms.run/pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime
- PyTorch 2.1.0, CUDA 12.1
- transformers 4.36.0 (downgraded from 5.x for PyTorch 2.1 compatibility)
- 2× NVIDIA A100-SXM4-80GB
- mDeBERTa-v3-base encoder (host-downloaded via proxy, copied to /models/mdeberta-v3-base)

## Eval Command
```bash
cd /repo
python3 main.py \
    --router_name mlp \
    --model_path /models/mdeberta-v3-base \
    --data_name gsm8k \
    --train_paths data/mmlu_train.json,data/gsm8k_train.json,data/arc_challenge_train.json,data/cmmlu_train.json \
    --cal_paths data/gsm8k_cal.json \
    --test_paths data/gsm8k_test.json \
    --answer_path data/gsm8k_test.json \
    --save_folder results/repro \
    --alpha 0.03 \
    --racer_nonc_score one_minus_prob \
    --n_splits 100 \
    --test_ratio 0.4 \
    --held_out_ratio 0.1 \
    --data_types multi_attempt \
    --data_format label \
    --hidden_size 256 \
    --lr 1e-4 \
    --epoch 100 \
    --weight_decay 0.01 \
    --train_bs 32 \
    --seed 42
```

## Metric Extraction
The script prints to stdout and saves JSON. Parse from JSON:
```python
import json
d = json.load(open("results/repro/gsm8k_repeated_racer_results.json"))
accuracy = d["summary"]["mean_racer_agg_acc_weighted_p_true"] * 100  # 77.57
```
Or grep stdout for: `RACER Aggregated (Weighted-p_true): 0.7757 ± 0.0145`

## Known Issues
1. RouterDC implementation not in repo; MLP substituted with similar performance
2. mDeBERTa-v3-base downloaded from host via proxy (hf-mirror.com blocked, direct HF works via HTTP proxy)
3. transformers downgraded to 4.36.0 for PyTorch 2.1.0 compatibility
4. SOCKS proxy (ALL_PROXY) must be unset for huggingface_hub downloads
