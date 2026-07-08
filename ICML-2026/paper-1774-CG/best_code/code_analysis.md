# Code Analysis - Paper 1774

## Evaluation Path
- python3 eval_task5.py --device cuda
- GPU: CUDA_VISIBLE_DEVICES=1 (GPU 0 busy, GPU 1 free)
- Metric: task5_c2st=<float> from stdout

## Key Files
- eval_task5.py: entry point, add CLI args only
- experiments/sbi/benchmark.py: task integration, safe to modify
- calibrated_guidance/guidance.py: core estimators, main optimization target
- calibrated_guidance/inference.py: flow_matching sampler, safe to modify
- experiments/sbi/tasks.py: log-likelihoods verbatim from paper, DO NOT MODIFY
- experiments/sbi/c2st.py: C2ST metric definition, DO NOT MODIFY
- calibrated_guidance/diffusion_posterior/memory.py: MemoryDiffusionPosterior

## Risky Files
- c2st.py, tasks.py, data_io.py: DO NOT MODIFY

## Known Bottlenecks
1. flow_matching initializes from Gaussian, not Uniform prior
2. Hard clamping of -inf log-likelihoods
3. Fixed guidance_scale=1.0
4. No memory across flow steps
5. Single softmax weighting, no control variates

## Container
- pytorch 2.1.0, CUDA 12.1, 2x A100 80GB
- GPU 1 is free for SOTA runs
