# Code Analysis — Paper 3720: LCL (Lineage-aware Contrastive Learning)

## Overview
This repo implements contrastive learning on scRNA-seq data with lineage barcodes as natural augmentations. The LCL model uses a base encoder (MLP: 2000→1024→256→64) + projection head (256→256→32) trained with NT-Xent loss + entropy penalty on unlabeled cells.

## Evaluation Path
1. `base_embed_extraction.py` — loads checkpoint, extracts base-encoder embeddings for train/test cells
2. `eval_both_metrics.py` — runs KNN classifier (K=5) and KL divergence prediction (linear decoder)
3. Primary metric: KNN_Test_Error (lower, baseline 0.4009)
4. Guardrail: KL_Divergence (lower, baseline 0.4558, max 0.595)

## Key Files
| File | Role | Safe to Modify |
|------|------|----------------|
| `LCL_Model_Semi.py` | Model architecture + ContrastiveLoss | Yes (add loss, architecture changes) |
| `LCL_Main_Semi.py` | Training script, config, LightningModule | Yes (hyperparams, training logic) |
| `DataLoader_combination_final.py` | Positive pair construction, batch generation | Yes (augmentation) |
| `General_Dataloader.py` | Thin wrapper around SClineage_DataLoader | Yes (pass-through params) |
| `SCDataset.py` | PyTorch Dataset for batches | Yes (data format) |
| `base_embed_extraction.py` | Embedding extraction for eval | No (eval protocol) |
| `eval_both_metrics.py` | KNN + KL evaluation | No (metric definitions) |

## Config Path
- Hparams class in `LCL_Main_Semi.py` (lines ~155-170)
- CLI args via argparse in `get_args()` function
- Key defaults: batch_size=25 (CLI default), temperature=0.5, lambda_penalty=1, gradient_accumulation_steps=5, lr=3e-4, weight_decay=1e-6

## Training command (from baseline)
```bash
cd /repo/main && python3 LCL_Main_Semi.py --inputFilePath Biddy_train.h5ad --testFilePath Biddy_test.h5ad --batch_size 50 --size_factor 0.04 --unlabeled_per_batch 5 --lambda_penalty 0.05 --temperature 0.5 --max_epoch 220 --output_dir /repo/output/celltag_lcl --train_test 1
```

## Eval command
```bash
cd /repo/main && python3 base_embed_extraction.py --inputFilePath /datasets/celltag/processed/Biddy_train.h5ad --batch_size 50 --output_dir /repo/output/celltag_lcl --resume_from_checkpoint /repo/output/celltag_lcl/saved_models/scContrastiveLearn_last.ckpt --out_file_name train_base_embed.npy && python3 base_embed_extraction.py --inputFilePath /datasets/celltag/processed/Biddy_test.h5ad --batch_size 50 --output_dir /repo/output/celltag_lcl --resume_from_checkpoint /repo/output/celltag_lcl/saved_models/scContrastiveLearn_last.ckpt --out_file_name test_base_embed.npy && python3 /repo/eval_both_metrics.py --train_h5ad /datasets/celltag/processed/Biddy_train.h5ad --test_h5ad /datasets/celltag/processed/Biddy_test.h5ad --train_embed /repo/output/celltag_lcl/train_base_embed.npy --test_embed /repo/output/celltag_lcl/test_base_embed.npy --n_neighbors 5
```

## Metric Parser
- KNN_Test_Error: grep "KNN Test Error (all):" from eval stdout
- KL_Divergence: grep "Test KL Divergence:" from eval stdout
- Both also saved to `evaluation_results.json`

## Container
- `autosota_repro_paper_3720`
- GPU: single A100 available
- Python 3.x with PyTorch 2.13.0+cu130
- Key deps: pytorch-lightning==1.9.5, scanpy, sklearn

## Data
- CellTag dataset at `/datasets/celltag/processed/Biddy_{train,test}.h5ad`
- 5893 train cells, 641 test cells (10% split), 2000 HVGs

## Risky Files (DO NOT MODIFY)
- `eval_both_metrics.py` — metric computation
- `base_embed_extraction.py` — eval embedding extraction
- `/tools/record_score.sh` — scoring
- `/datasets/celltag/processed/` — test data

## Safe Modification Targets
- `LCL_Model_Semi.py`: ContrastiveLoss, BaseEncoder_ProjHead_MLP
- `LCL_Main_Semi.py`: Hparams config, training_step, scContraLearn
- `DataLoader_combination_final.py`: SClineage_DataLoader augmentation methods
- `SCDataset.py`: data return format
