# Code Analysis for Paper 2776 - ChaosNexus SOTA Optimization

## Architecture Summary
- **Model**: `PatchTSTForPrediction` (line 1853, scaleformer.py) - ChaosNexus with U-Net encoder/decoder
- **Config**: `d_model=48`, `patch_length=8`, `patch_stride=8`, 4 stages with depths [2,2,2,2]
- **MoE**: Hardcoded True with 8 experts, top_k=2 (line 881-883 in encoder layer)
- **Freq Analyzer**: kymatio wavelet scattering (WST), J=8, Q=8, output dim=48
- **Koopman Embedding**: `PatchTSTKernelEmbedding` with RFF (frozen by default), polynomial features
- **Prediction Head**: `MultiStagePredictionHead` - concatenates 4 decoder stage outputs + wavelet embedding
- **Loss**: MSE + 0.5×MMD(rational_quadratic) + 0.1×load_balance

## Evaluation Path
- `scripts/scaleformer/evaluate.py` via Hydra config
- Uses `PatchTSTPipeline.from_pretrained()` to load checkpoint
- Metrics saved to CSV in eval_results directory
- Key metrics: sMAPE@128, sMAPE@512, ME-LRw, max_lyap (pred and gt), gd_rmse

## Training Path
- `scripts/scaleformer/train.py` via Hydra config
- `load_patchtst_model()` with `pretrained_pft_path` for fine-tuning
- `CustomTrainer` wraps HuggingFace Trainer with scheduler support

## Config Files
- `/repo/config/model.yaml` - template (d_model=512 differs from actual baseline d_model=48)
- `/repo/config/dataset.yaml` - augmentation config
- Baseline actual config stored in: `/repo/checkpoints/Nexus1.0-base/checkpoint-final/training_info.json`

## Key Modification Targets
1. **RFF trainable**: `PatchTSTKernelEmbedding.__init__()` (modules.py:41-42) - `requires_grad=config.rff_trainable`
2. **Wavelet padding**: `PatchTSTForPrediction.forward()` (scaleformer.py:2031) - `F.pad(..., "constant", 0)` → `"reflect"`
3. **Step-weighted loss**: `PatchTSTForPrediction.forward()` (scaleformer.py:2089) - MSE loss computation
4. **Augmentation**: `config/dataset.yaml` - rate and probabilities
5. **MMD kernel**: `PatchTSTForPrediction.__init__()` (scaleformer.py:1896) - kernel type and params
6. **Load balance coeff**: `PatchTSTForPrediction.__init__()` (scaleformer.py:1886) - λ1=0.1

## Baseline Config (from training_info.json)
- d_model=48, patch_len=8, patch_stride=8
- loss=mse, lr=0.001, batch=1024, max_steps=100000
- lr_scheduler=cosine, warmup=0.1, weight_decay=0.0
- use_dynamics_embedding=true, rff_trainable=false
- aug_rate=0.2, probs=[1,1,1,0,0]

## Risks
- Wavelet padding change affects freq_analyzer input distribution (trained with zero-padding)
- RFF trainable changes parameter count and may need retraining
- Step-weighted loss changes optimization landscape
