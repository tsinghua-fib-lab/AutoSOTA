# Code Analysis -- Paper 866: Streaming Prototype OOD Detection

## Evaluation Path
- Entry: main.py -> parses args -> loads config -> StreamingOODPipeline.run()
- Pipeline: openood/pipelines/streaming_ood_pipeline.py
- Config: configs/streaming_ood_imagenet_repro.yaml
- Eval command: python3 main.py --config configs/streaming_ood_imagenet_repro.yaml --seed 74

## Key Components
1. Text feature extraction: openood/networks/clip_fixed_ood_prompt.py
   - get_id_text_features() -- generates ID class text prototypes via CLIP
   - get_ood_text_features() -- generates OOD negative label prototypes via WordNet
   - get_templates() -- maps config key to template list

2. Streaming adapter: openood/postprocessors/streaming_prototype_adapter.py
   - StreamingPrototypeAdapter -- main class maintaining text + visual prototypes
   - _update_id() -- gradient descent update on ID visual prototypes
   - _update_ood() -- gradient descent update on OOD visual prototypes
   - _compute_scores() -- group-softmax OOD scoring (hardcoded ID temp 0.01)

3. Pipeline: openood/pipelines/streaming_ood_pipeline.py
   - StreamingOODPipeline.run() -- orchestrates per-dataset evaluation
   - resolve_routes() -- binary ground-truth routing

4. Evaluator: openood/evaluators/ood_evaluator.py -- AUROC + FPR95 via sklearn

## Metric Parser
- Output: results/imagenet_ood_results.json
- Parse: Read AVERAGE entry for aggregate AUROC and FPR95

## Baseline
- AUROC: 99.26%, FPR95: 2.51%
- 50K ID images (val_imagenet_paper.txt)

## Risky Files (do not modify)
- openood/evaluators/ood_evaluator.py -- metric computation
- openood/datasets/imglist_dataset.py -- data loading
- Dataset files under /datasets/

## Safe Modification Targets
- configs/streaming_ood_imagenet_repro.yaml -- hyperparameters
- openood/postprocessors/streaming_prototype_adapter.py -- adapter logic
  - Line ~315: hardcoded 0.01 ID temperature
  - _update_id() and _update_ood() -- update mechanisms
  - _compute_scores() -- scoring aggregation
- openood/networks/clip_fixed_ood_prompt.py -- template selection
- openood/pipelines/streaming_ood_pipeline.py -- routing logic

## Hardcoded Values Noted
1. scaled_id = logits_id / 0.01 in _compute_scores() -- should be configurable
2. text_prompt: nice -- single template, full has 80 templates
3. Single blend_factor shared for ID and OOD
4. Global step counter in _update_id/_update_ood -- per-class normalization possible
