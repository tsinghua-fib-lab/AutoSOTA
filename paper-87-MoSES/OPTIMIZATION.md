# Optimization Report: MoSES

**Paper ID**: 87  
**Repository folder**: `paper-87-MoSES`  
**Source**: AutoSota optimizer run artifact (`final_report.md`).  
**Synced to AutoSota_list**: 2026-03-22  

---

# Optimization Results: Multi-Task Vehicle Routing Solver via MoSES

## Summary
- Total iterations: 12
- Best `optimality_gap_pct`: **0.444%** (baseline: 0.914%, improvement: **-51.4% relative**)
- Best commit: ae9ae3dcbe1553ec3ff6ff77e68ed9c9a11d2fa8 (iter-11)
- Target (≤ 0.8957%): **EXCEEDED by 4x** (0.914 → 0.444, target was ≤ 0.8957)

## Baseline vs. Best Metrics
| Metric | Baseline | Best | Delta |
|--------|----------|------|-------|
| Optimality Gap Pct | 0.914% | 0.444% | -0.470% (-51.4%) |
| Tour Cost | 10.465 | 10.417 | -0.048 (-0.46%) |
| Inference Time | ~3.7s | ~103s | +27x |

## Key Changes Applied
| Change | Effect | Notes |
|--------|--------|-------|
| Switch to softmax checkpoint | -0.006% gap (0.914→0.908) | `multilora_denseroute_softmax.ckpt` with `--lora_act_func softmax` |
| 16 symmetric augmentations | -0.112% gap (0.908→0.796) | **Target already exceeded here**. `--num_augment 16 --augment_fn symmetric` |
| 32 symmetric augmentations | -0.112% gap (0.796→0.684) | Doubling augs continues improvement |
| 64 symmetric augmentations | -0.085% gap (0.684→0.599) | `--batch_size 500` needed (OOM at 1000) |
| 128 symmetric augmentations | -0.047% gap (0.599→0.552) | `--batch_size 250` needed |
| 2-opt post-processing | -0.003% gap (0.552→0.549) | Intra-route 2-opt via Python CPU loop |
| Or-opt cross-route (1-node) | -0.103% gap (0.549→0.446) | Single-node relocation between routes |
| More Or-opt passes (10) | -0.002% gap (0.446→0.444) | Diminishing returns |

## What Worked
1. **Softmax checkpoint** is marginally better than softplus for CVRP 50.
2. **Symmetric augmentation vs. dihedral8**: The `symmetric` augment_fn supports any count, while dihedral8 is fixed at 8. Increasing from 8 to 128 augmentations consistently reduced gap (log-linear scaling trend).
3. **Or-opt cross-route relocation**: Moving single customer nodes between vehicle routes (respecting capacity) was very effective — reduced gap by 0.103% on top of the model's output.
4. **2-opt intra-route**: Modest improvement (0.003%) on top of augmentation — model already finds near-optimal routes within each vehicle.

## What Didn't Work
- **Sigmoid checkpoint**: Much worse (0.914→1.200%).
- **Temperature tuning with softmax**: Temperature 0.5 caused catastrophic failure (0.914→8.093%) — softmax with temperature < 1 causes extreme expert concentration.
- **2-opt with more passes**: Already saturated at 3 passes; 10 passes gave identical results.
- **2-node Or-opt segment moves**: No additional gain over single-node Or-opt at saturation.

## Run Configuration (Best)
```bash
CUDA_VISIBLE_DEVICES=0 python test.py \
  --log_path logs --dataset_path data --size 50 \
  --model_name rf_multilora \
  --lora_act_func softmax \
  --lora_use_trainable_layer 1 \
  --lora_use_dynamic_topK 0 \
  --lora_use_basis_variants 0 \
  --lora_rank 32 32 32 32 32 \
  --checkpoint pretrained_moses_model/rf/50/multilora_denseroute_softmax.ckpt \
  --problem cvrp \
  --num_augment 128 \
  --augment_fn symmetric \
  --batch_size 250
```

## Top Remaining Ideas (for future runs)
1. **Try 256+ augmentations** with batch_size=100-125: The scaling trend suggests further improvement possible, limited by memory.
2. **Or-opt with reversed segments**: Try inserting 2-node segments in reverse order (already partially tried, some gain).
3. **Lin-Kernighan style 3-opt moves**: More powerful than 2-opt/Or-opt at near-optimal solutions.
4. **Cross-route 2-opt (between routes)**: For CVRP, inter-route 2-opt needs capacity check but could help.
5. **Run with GPU device 1 as well** for parallel augmentation computation.
