# Code Analysis — Paper 3690 SOTA Preparation Repair

## Original Preparation Failure

**Root cause:** The container `autosota_repro_paper_3690` had a full overlay filesystem (200G/200G, 100% used). The `apt-get install git` command failed with "No space left on device" (error 28). This prevented the baseline git commit from being created.

**Secondary issue:** After the container was removed and recreated, the reproduced image `autosota/paper-3690:reproduced` was not available locally, and Docker Hub was unreachable. The `env` image lacked the full reproduction setup (no `gift` conda env, no repo code, no modified transformers).

## Corrected In-Container Evaluation Command

```bash
cd /repo
source /opt/conda/etc/profile.d/conda.sh
conda activate gift
python3 eval_chair.py --config configs/chair_llava_1.5_7b.yaml
```

## Evidence Baseline Matches Reproduction

- **CHAIRs:** 43.6 (manifest: 43.6) ✅
- **CHAIRi:** 25.1 (manifest: 25.1) ✅
- Model: LLaVA-1.5-7B-hf from /models/llava-1.5-7b-hf
- Images: 498 COCO val2014 images from /datasets/coco/val2014
- Annotations: /datasets/coco/annotations/instances_val2014.json
- GIFT config: alpha=5.0, enhancement layers 12-22, saliency layer 11

## Reusable Resources

- `/models/llava-1.5-7b-hf` — LLaVA model weights (14GB)
- `/datasets/coco/val2014` — 498 COCO validation images (83MB)
- `/datasets/coco/annotations/instances_val2014.json` — COCO annotations (154MB)

No `/paper_data` mount was configured.

## Safe Optimization Targets

### Code-level changes (in modeling_llama.py)
The GIFT attention enhancement mechanism is in `eager_attention_forward()` (lines 220-270):
- `visual_saliency_map` multiplication (line 253)
- `query_scale` computation (line 257)
- Renormalization (line 265)

### Config-level changes (in YAML configs)
- `alpha`: Visual attention scaling (default 5.0, best found: 7.0)
- `attention_enhancement_layers`: Which layers apply GIFT (default 12-22)
- `visual_saliency_computation_layers`: Which layer computes saliency (default [11])
- `caption_prompt`: Caption prompt template (default "Please describe this image in detail.")
- `max_new_tokens`: Generation length (default 1024)

## Key Finding

Alpha=7.0 gives CHAIRs=34.7 (vs baseline 43.6 at alpha=5.0), a -20.4% improvement. This beats the paper's reported CHAIRs of 39.8.
