# Code Analysis - Paper 4298 (ConOrd / Contrastive Order Learning)

## Key Files
- **reproduce_bid.py**: Main training + evaluation script. Builds config, runs 3-fold CV, reports SRCC/PLCC.
- **config/basic.py**: ConfigBasic with set_biqa_dataset() for BID transforms.
- **data/get_datasets_BIQA.py**: BIDDataset, get_datasets_BIQA() - 80/20 split per seed.
- **networks/gol.py**: ConOrd model with flexible reference points.
- **networks/base.py**: BaseModel using ViT-B/16 CLIP backbone.
- **networks/util.py**: prepare_model() factory.
- **utils/loss_util.py**: ConOrdLoss (contrastive ordinal loss), LabelDifference, FeatureSimilarity, compute_center_loss.
- **utils/comparison_utils.py**: find_kNN() for inference-time k-NN regression.
- **utils/util.py**: cal_srocc_plcc(), extract_embs(), AverageMeter.

## Evaluation Path
1. reproduce_bid.py runs 3-fold 80/20 split (seeds 42, 43, 44)
2. Training: batch_size=32, AdamW, CosineAnnealingLR, ConOrdLoss
3. Inference: k-NN (k=10) from test embeddings to train embeddings
4. Metrics: SRCC median, PLCC median (Spearman/Pearson correlation)

## Data
- BID dataset at `/tmp/bid_local` (copy from `/datasets/BID/BID`)
- 590 JPG images, MOS scores from DatabaseGrades.xls
- 80/20 split: ~472 train, ~118 test per fold

## Safe Modification Targets
- reproduce_bid.py: LR, WD, epochs, temperature, k, optimizer params
- utils/loss_util.py: label_diff type, center loss, new loss terms
- utils/comparison_utils.py: k-NN weighting
- data/get_datasets_BIQA.py: DataLoader config, augmentations
- networks/gol.py: ref point initialization

## Red-line Boundaries
- NO changes to metric computation (cal_srocc_plcc)
- NO changes to dataset splits or labels
- NO hard-coded predictions
- NO changes to evaluation protocol
