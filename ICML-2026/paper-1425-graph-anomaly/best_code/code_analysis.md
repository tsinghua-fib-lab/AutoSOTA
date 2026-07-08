# Code Analysis for Paper 1425 — REFI-GAD

## Evaluation Path

**Command**: `CUDA_VISIBLE_DEVICES=0,1 python3 main.py`
**Working directory**: `/repo`
**Output**: stdout — final summary block with AUROC/AUPRC per target dataset
**Parsing**: `Target: cora.*AUROC: X.XXXX.*AUPRC: Y.YYYY` (cora mean values from "Final Cross-Domain Results")

### Evaluation flow
1. `main.py` loads 7 training datasets (Group 1): Amazon, citeseer, weibo, ACM, BlogCatalog, cs, photo
2. `main.py` loads 7 test datasets (Group 2): YelpChi, Reddit, questions, Facebook, Flickr, cora, pubmed
3. Runs 5 trials with seeds 1-5
4. Each trial: `Detector.train_mixed()` on all 7 training datasets, then `test_one_dataset()` on each test dataset
5. Reports mean ± std across 5 trials

## Key Source Files

### `main.py` — Entry point and config
- Lines 42-57: `set_defaults()` with key hyperparameters
- NOTE: Some `set_defaults` values override parser defaults (e.g., k=50 not 10, num_layers=4 not 2, batch_size=512 not 256, lr=1e-4 not 1e-3, epoch=100)
- Lines 73-76: Facebook dataset uses k=10, others use k=50
- Lines 82-96: Final cross-domain results printing

### `model.py` — PromptGADModel and GlobalBatchWeighting
- Lines 6-24: GlobalBatchWeighting — computes SNR-based feature weights
- Lines 26-103: PromptGADModel forward pass
  - Line 45: `logit_scale` = learnable temperature (init `log(1/0.07)`)
  - Lines 47-55: `_generate_inductive_mask` — custom attention mask
  - Lines 57-68: Feature projection + transformer encoding
  - Lines 70-77: Batch weighting application
  - Lines 79-96: Similarity scoring, softmax attention, anomaly probability

### `train_test.py` — Training and inference
- Line 32: Hardcoded `pos_weight=4.0` (CODE-01 target)
- Lines 34-82: `get_intra_dataset_batch` — support set sampling
  - Lines 62-63: `n_query_ano=50, n_query_norm=500` — batch composition
- Lines 84-129: `train_mixed` — training loop
  - Line 119: `loss_po` computed
  - Line 119-121: `loss_ne` computed but DISCARDED — `loss = loss_po` (ALGO-01 target)
  - Line 121: No gradient clipping (CODE-02 target)
- Lines 131-162: `test_one_dataset` — inference
  - Line 140: `num_test_runs = 1` (ALGO-02 target)

### `construct_P_features.py` — Feature construction
- 5 fingerprint dimensions: global cosine similarity, neighbor cosine similarity, neighbor distance similarity, degree centrality, clustering coefficient
- All features are rank-normalized to [0, 100]

### `utils.py` — Dataset loading, graph computation, evaluation
- Lines 10-25: `test_eval()` — AUROC and AUPRC computation
- Lines 47-97: `Dataset` class — loads .mat files, normalizes adjacency, computes feature propagation and similarity convolution
- Lines 99-136: `sim_conv()` — similarity-weighted graph convolution

## Safe Modification Targets

| File | Safe Changes | Risky Changes |
|------|-------------|---------------|
| `train_test.py` | Loss function, optimizer config, batch sampling, inference runs | Data loading, label handling |
| `model.py` | Attention mechanism, temperature, feature dimensions | Output format |
| `main.py` | Hyperparameter defaults, LR schedule | Dataset lists, metric reporting format |
| `construct_P_features.py` | Feature computation, additional features | Label usage |
| `utils.py` | Graph construction, feature computation | Metric computation (test_eval), data loading format |

## Red-Line Boundaries
- DO NOT modify: `test_eval()` metric computation, dataset labels/splits, test dataset list
- DO NOT use: test dataset labels during training, hard-coded predictions
- DO NOT change: evaluation output format (parsing depends on it)

## Container Environment
- Container: `autosota_repro_paper_1425`
- GPU: CUDA_VISIBLE_DEVICES=0,1 (devices 4,5 on host)
- Datasets in: `/repo/dataset/` (14 .mat files)
- PyTorch 2.1.0, torch_geometric 2.8.0
