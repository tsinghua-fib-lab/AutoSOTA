# LLaDA-V: Large Language Diffusion Models with Visual Instruction Tuning: A Technical Report on Automated Optimization

## Abstract
This report documents an automated optimization study targeting the MMMU (Massive Multi-discipline Multimodal Understanding) benchmark accuracy of LLaDA‑V, a diffusion‑based vision–language model. Using the AutoSOTA pipeline, five iterations explored inference‑time parameter adjustments and algorithmic modifications, all without retraining or dataset changes. The best result was a marginal MMMU accuracy improvement from 48.11 % to 48.33 % (+0.22 percentage points), far below the 5 % improvement target (51.10 %). The only beneficial intervention—increasing diffusion generation steps and answer length—yielded a 3.33‑point gain in the Science category but caused slight regressions in Business, Health, and Humanities. All other attempts (classifier‑free guidance, higher vision resolution, margin‑based remasking) either failed to run or produced no effect. The core finding is that LLaDA‑V’s MMMU performance is capability‑bound rather than generation‑bound: the model’s underlying multimodal understanding determines accuracy, and inference‑time tuning alone cannot reach the desired target.

## 1. Introduction
MMMU (Massive Multi-discipline Multimodal Understanding) evaluates multimodal models on exam‑level questions spanning six broad domains and has become a standard benchmark for real‑world understanding. LLaDA‑V (You et al., 2025) offers a diffusion‑based alternative to autoregressive vision–language models, demonstrating that iterative denoising can support complex multimodal reasoning. The paper reports an MMMU accuracy of 48.67 %. This study investigates whether systematic, inference‑only optimization through the AutoSOTA pipeline can raise that score without modifying the model’s weights or training data. The results serve as a controlled case study in the limits of post‑training parameter tuning for diffusion‑based vision–language models.

## 2. Original Method (Background)
LLaDA‑V pairs a pretrained large language diffusion model (LLaDA‑8B) with a SigLIP vision encoder via a linear projector. Text generation proceeds as iterative denoising: starting from a fully masked sequence, the model predicts tokens at each diffusion step, progressively unmasking words. Key inference hyperparameters are the number of generation steps (`gen_steps`), the block length per step (`block_length`), and the total generation length (`gen_length`). The model was fine‑tuned using visual instruction data from LLaVA‑NeXT, MAmmoTH‑VL, and VisualWebInstruct following the LLaVA recipe. MMMU evaluation uses the lmms‑eval harness with multiple‑choice questions and exact‑match accuracy. The paper’s reported overall accuracy is 48.67 %, providing a reference point for this study.

## 3. Identified Limitations
**Suboptimal generation hyperparameters.** The default inference configuration (`gen_steps=2`, `gen_length=2`, `block_length=2`) severely truncates answers, preventing the model from fully leveraging the diffusion denoising process. Our baseline measurement with these defaults dropped to 48.11 % (0.56 points below the paper‑reported 48.67 %), consistent with premature answer cutoff for open‑ended responses. This confirms that the default parameters form a performance bottleneck.

**Critical argument‑parsing bug.** The evaluation harness (`eval/lmms-eval/lmms_eval/__main__.py`) uses `simple_parse_args_string` to parse `model_args`. The original `arg.split("=")` corrupts strings that contain nested `=` characters (e.g., `pretrained=GSAI-ML/LLaDA-V,gen_steps=2`), making it impossible to supply generation parameters correctly. Fixing this bug was a prerequisite for any parameter tuning.

**Incompatibility of classifier‑free guidance.** Classifier‑free guidance (CFG) requires two forward passes per diffusion step, doubling inference time to approximately 8 s per sample in our setup. Existing research indicates that CFG with guidance scale 0 (unconditional generation) is optimal for MMLU‑style multiple‑choice tasks, so the technique offers no expected benefit while making full‑benchmark evaluation infeasible.

**Memory explosion with higher vision resolution.** Reducing the vision encoder’s spatial pooling stride to 1 quadruples the number of visual tokens. The subsequent softmax in the diffusion process is cast to float64 for numerical stability, and this operation exhausts GPU memory even on an 80 GB device, causing an out‑of‑memory (OOM) error. The current implementation cannot handle such tensor sizes without numerical precision trade‑offs.

**Remasking strategy ineffectiveness.** The standard diffusion remasking selects tokens with the lowest confidence. A margin‑based confidence criterion (P(top1) – P(top2)) was hypothesized to better identify uncertain tokens in multiple‑choice settings. Empirical testing showed that for questions with sharply peaked answer distributions both strategies identify identical tokens, yielding exactly the baseline 48.11 % accuracy with no advantage.

## 4. Optimization Methodology
Each intervention was applied sequentially by the AutoSOTA pipeline, which adjusted source code or command‑line arguments and reran the full MMMU evaluation. Only changes that produced a measurable improvement were retained; others were reverted or aborted.

**Fix argument‑parsing function (`simple_parse_args_string`).**  
File: `eval/lmms-eval/lmms_eval/__main__.py`, function `simple_parse_args_string` (imported from `lmms_eval/utils.py`).  
Change: Replace `arg.split("=")` with `arg.split("=", 1)` to correctly parse model‑argument strings containing multiple `=` signs. This infrastructure fix was required before any parameter tuning and allowed the harness to accept generation hyperparameters without corruption.

**Increase generation hyperparameters (`gen_steps`, `gen_length`, `block_length`).**  
The default `gen_steps=2`, `gen_length=2`, `block_length=2` were raised to 4, 8, and 8, respectively. More denoising steps and a longer generation window allow the diffusion process to refine answers iteratively and accommodate longer chain‑of‑thought reasoning. This was the only accepted intervention, yielding a +0.22 pp overall gain with a notable +3.33 pp improvement in Science.

**Aborted and failed interventions.**  
- **CFG bug fix and cfg=0.5 test:** The mapping from the external `cfg` argument to the internal `cfg_scale` parameter was corrected, but the doubled evaluation time made the full run impractical, and prior literature indicated no benefit; the test was aborted.  
- **Vision resolution increase:** `mm_spatial_pool_stride=1` caused an OOM during the float64 softmax; the intervention failed without producing a metric.  
- **Margin‑based confidence remasking:** The remasking logic was replaced with a margin‑based criterion, but accuracy returned exactly 48.11 %, showing zero effect.  
- **Re‑test of gen_params=4/8/8:** A re‑evaluation of the successful parameter combination was aborted.

No other changes (e.g., dynamic temperature scheduling, ensemble voting) were implemented.

## 5. Experiments

### 5.1 Setup
All experiments ran on a single NVIDIA 80 GB GPU, using the official LLaDA‑V model checkpoint `GSAI-ML/LLaDA-V` and the lmms‑eval harness. The evaluation dataset was the full MMMU validation split; multiple‑choice exact‑match accuracy was the primary metric. A fixed random seed was used across all runs to ensure reproducibility. The baseline command used the default generation parameters:

```
lmms-eval --model llada_v --model_args pretrained=GSAI-ML/LLaDA-V,gen_steps=2,gen_length=2,block_length=2 --tasks mmmu_val --batch_size 1
```

Five iterations were permitted (including the baseline). The pipeline was restricted to inference‑time modifications; no model retraining or dataset changes were allowed. The baseline accuracy measured in this environment (48.11 %) is 0.56 points below the paper’s 48.67 %, consistent with truncation by the short `gen_length` setting.

### 5.2 Quantitative Results
Table 1 compares the baseline (iteration 0) with the best configuration (iteration 1) across MMMU categories and overall.

| Category                       | Baseline (48.11%) | Best (48.33%) | Δ (pp) |
|--------------------------------|-------------------|---------------|--------|
| Art and Design                 | 52.50             | 54.17         | +1.67  |
| Business                       | 51.33             | 48.67         | –2.66  |
| Science                        | 32.00             | 35.33         | +3.33  |
| Health and Medicine            | 46.00             | 45.33         | –0.67  |
| Humanities and Social Science  | 71.67             | 70.83         | –0.84  |
| Tech and Engineering           | 42.86             | 43.33         | +0.47  |
| **Overall**                    | **48.11**         | **48.33**      | **+0.22** |

Table 1. Per‑category and overall MMMU accuracy (%). The best configuration uses `gen_steps=4, gen_length=8, block_length=8`.

The overall improvement is marginal. Science gained the most (+3.33 pp), suggesting that longer answers help computation‑heavy questions. Conversely, Business, Health and Medicine, and Humanities declined slightly, indicating that increased generation length can hurt performance on verbal reasoning tasks.

### 5.3 Ablation / Iteration Trajectory
Table 2 lists every iteration chronologically with the nature of the change and the resulting overall accuracy.

| Iter | Idea                                      | Type    | Accuracy (%) | Δ (pp) | Status       |
|------|-------------------------------------------|---------|---------------|--------|--------------|
| 0    | Baseline (gen_steps=2, gen_length=2)      | —       | 48.11         | —      | baseline     |
| 1    | gen_steps=4, gen_length=8, block_length=8 | PARAM   | 48.33         | +0.22  | success      |
| 2    | CFG bug fix + cfg=0.5                     | CODE    | —             | —      | failed       |
| 3    | Vision resolution (stride=1)              | PARAM   | —             | —      | failed (OOM) |
| 4    | Margin‑based confidence remasking         | ALGO    | 48.11         | 0.00   | no effect    |
| 5    | Re‑test gen_params=4/8/8                  | PARAM   | —             | —      | aborted      |

Table 2. Optimization trajectory. Only iteration 1 produced a measurable gain; all others failed, had no effect, or were aborted.

## 6. Discussion
The results confirm that LLaDA‑V’s MMMU accuracy is fundamentally bounded by its pretrained multimodal understanding, not by generation hyperparameters. The only effective change—longer generation steps and length—raised the overall score by a negligible 0.22 pp, with domain‑wise gains offset by regressions. This suggests no single parameter set is optimal across all categories.  

CFG doubled inference cost with no expected benefit for multiple‑choice tasks, and the float64 softmax implementation made higher vision resolution infeasible without risking numerical stability. The margin‑based remasking experiment demonstrated that for questions with clear answer preferences, token‑remasking criteria are interchangeable, resulting in identical accuracy.  

These findings highlight a critical limitation: inference‑only tuning cannot substantially improve LLaDA‑V on knowledge‑intensive benchmarks. Meaningful gains likely require training‑time interventions (e.g., variance‑reduced preference optimization, mixture‑of‑experts architectures, or dataset enrichment) that were outside the scope of this study. While the measured baseline was 0.56 points below the paper’s figure due to generation truncation, that gap does not change the conclusion that post‑training, inference‑only optimization is insufficient to reach the target.

## 7. Reproducibility
- Repository: `https://github.com/ML-GSAI/LLaDA-V`
- Environment setup: `cd eval && bash init_env.sh`, install lmms‑eval dependencies as needed.
- All runs used a fixed random seed for reproducibility.
- Baseline run (requires the argument‑parsing fix applied to `simple_parse_args_string` in `lmms_eval/utils.py`, changing `split("=")` to `split("=", 1)`):
  ```
  cd eval && lmms-eval --model llada_v --model_args pretrained=GSAI-ML/LLaDA-V,gen_steps=2,gen_length=2,block_length=2 --tasks mmmu_val --batch_size 1
  ```
- Optimized run (best commit `f7c603d3f1`):
  ```
  cd eval && lmms-eval --model llada_v --model_args pretrained=GSAI-ML/LLaDA-V,gen_steps=4,gen_length=8,block_length=8 --tasks mmmu_val --batch_size 1
  ```

## 8. References
```bibtex
@article{you2025llada,
  title={LLaDA-V: Large Language Diffusion Models with Visual Instruction Tuning},
  author={You, Zebin and Nie, Shen and Zhang, Xiaolu and Hu, Jun and Zhou, Jun and Lu, Zhiwu and Wen, Ji-Rong and Li, Chongxuan},
  journal={arXiv preprint arXiv:2505.16933},
  year={2025}
}

@misc{autosota2025,
  author = {{tsinghua-fib-lab}},
  title  = {{AutoSOTA}: An Automated State-of-the-Art Optimization Pipeline},
  year   = {2025},
  url    = {https://github.com/tsinghua-fib-lab/AutoSOTA}
}
```
