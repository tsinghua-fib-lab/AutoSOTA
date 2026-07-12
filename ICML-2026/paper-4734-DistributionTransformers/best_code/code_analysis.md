# Code Analysis - Paper 4734 Distribution Transformers

## Evaluation Path
- main.py -> sacred experiment -> experiments/sources/static/inverse_gamma_variance_closed_form.py:run()
- Config: experiments/configs/dt2_wide_rep.yaml
- Training: workflows/train.py:train()
- Testing: workflows/test.py:test_conjugate_prior()
- Eval command: python3 main.py with experiments/configs/dt2_wide_rep.yaml -F experiments/runs/dt2_wide --force

## Metric Parsing
- KL-Divergence: stdout "Posterior mean KL divergence: <value>"
  - Computed via kl_divergence(exact_posterior, model_posterior, ...)
  - n_kl_samples=10000, n_test_priors=1000
- Inference Time: sacred run info.json key "model_inference_time"
  - GPU time for one forward pass over 1000 priors (line 107 test.py)

## Key Files
- Config: experiments/configs/dt2_wide_rep.yaml
- Experiment: experiments/sources/static/inverse_gamma_variance_closed_form.py
- Training: workflows/train.py
- Testing: workflows/test.py
- Model: src/model/distribution_transformer.py
- Factory: src/competitor_methods/ace_dt_morphology.py
- Embeddings: src/model/embeddings.py
- Distributions: distributions/distributions.py

## Safe Targets
1. Config YAML - hyperparameters
2. distribution_transformer.py:87-92 - init_weights()
3. train.py:182-185 - total_loss (prior_loss_weight)
4. ace_dt_morphology.py - DistributionTransformerWithEncoder

## Risky Files (do not modify)
- workflows/test.py - metric computation
- distributions/utils.py - kl_divergence
- distributions/distributions.py - distribution defs
- main.py - eval entry point

## Pre-downloaded Data
- No /paper_data mount. All data sampled from meta-prior.
- /datasets, /models, /autosota_cache available but not needed.

## Baseline
- KL-Divergence: 0.005746
- Inference Time: 0.0144s
- Commit: 391b6aa
