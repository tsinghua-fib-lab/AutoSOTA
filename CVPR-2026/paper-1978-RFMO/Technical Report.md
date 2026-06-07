# Revisiting F‑measure Optimization in Multi‑Label Classification: A Sampling‑based Approach: A Technical Report on Automated Optimization

## Abstract

This report describes an automated optimization study applied to the sampling‑based inference method for multi‑label classification introduced in “Revisiting F‑measure Optimization in Multi‑Label Classification: A Sampling‑based Approach.” The method predicts multiple labels per instance by drawing Monte Carlo samples from an autoregressive model and then solving an F1‑optimal label‑set extraction problem. Over nine optimization iterations, the AutoSOTA pipeline evaluated several hyperparameter and architectural modifications, accepting three inference‑time changes: increasing the number of Monte Carlo samples from 200 to 500, lowering the sampling temperature from 1.0 to 0.5, and enabling multi‑run P‑matrix averaging with four independent runs. These changes, which require no model retraining, raised the primary metric instance‑F1 from 0.5372 to 0.5655 (+5.3%) and simultaneously improved all other reported metrics, most notably micro‑F1 (+17.0%) and subset accuracy (+9.2%). Training‑time interventions—alternative optimizers, early stopping, activation functions, and loss weighting—caused regression in every case. The results indicate that tuning inference hyperparameters can substantially improve instance‑F1 for this architecture, and that the AutoSOTA pipeline identified these modifications automatically.

## 1. Introduction

Multi‑label classification, where each instance receives multiple categorical labels, is common in image tagging, text categorization, and bioinformatics. The instance‑wise F1 measure is often the target metric because it rewards both partial correctness and balanced performance across labels. Direct optimization of F1 during training is non‑decomposable, so many methods rely on surrogate losses and post‑hoc inference heuristics. The paper “Revisiting F‑measure Optimization in Multi‑Label Classification: A Sampling‑based Approach” proposes an autoregressive predictor that generates candidate label sets via ancestral sampling, computes a co‑occurrence probability matrix (P‑matrix) from the samples, and extracts the label subset that maximizes expected instance‑F1 via a combinatorial algorithm. This decouples inference from the training loss.

The method’s performance depends on inference‑time hyperparameters: the number of Monte Carlo samples (which controls P‑matrix estimation variance), the sampling temperature (which balances exploration and exploitation), and whether a single sampling run or an ensemble of runs is used. The AutoSOTA automated optimization framework was applied to the codebase to search for configurations that improve instance‑F1 while keeping the training procedure unchanged. Over nine iterations, the pipeline tested modifications to training and inference parameters, eventually retaining only three inference‑time changes that together yielded a 5.3% relative improvement over the baseline. The final configuration exceeds the predefined target of 0.5641 by 0.14 percentage points.

## 2. Original Method (Background)

The codebase supports several predictor architectures (Binary Relevance, Autoregressive, Multinomial, Gibbs Sampling). The optimization run focused exclusively on the AutoregressivePredictor (`src/predictor/ar.py`), which implements the paper’s core contribution. The model is a multi‑layer perceptron (MLP) conditioned on instance features and a partial label set; it produces logits for the next label in an autoregressive decoding order. Training uses ground‑truth labels and binary cross‑entropy loss (BCEWithLogitsLoss) over all timesteps, with a fixed label ordering found by greedy or random search on the validation set.

At inference, the `predict` method draws `num_samples_to_infer` ancestral samples, applying a softmax temperature. A per‑instance P‑matrix of size `(num_labels, num_labels)` is constructed by recording, for each label, the frequency of co‑occurrence with each possible cardinality of other active labels. The entry `P[l][k]` approximates the probability that label `l` is active given exactly `k` other labels are active. The function `infer_f1_labels` (in `src/predictor/utils.py`) then solves for the label subset that maximizes expected instance‑F1 under this empirical co‑occurrence distribution. The original defaults were `num_samples_to_infer=200`, temperature=1.0, and a single sampling run. A multi‑run ensemble mode (`num_ensemble_runs`) divides the sampling budget equally among several independent runs, averages the resulting P‑matrices, and then performs inference.

## 3. Identified Limitations

Three limitations of the original inference procedure were evident from the source code and confirmed by the AutoSOTA log.

**Limited Monte Carlo samples.** The `_calculate_P_matrix` function uses `num_samples_to_infer` draws. With only 200 samples, entrywise co‑occurrence probabilities are noisy, especially for rare label combinations, which can mislead the F1‑maximizer. The log notes that increasing to 500 samples contributed approximately +2% relative instance‑F1, indicating that estimation variance was a significant factor.

**Neutral temperature.** The sampling applies a temperature of 1.0 by default, preserving the model’s raw softmax distribution. When the model is well‑calibrated, a lower temperature concentrates probability mass on high‑confidence labels, reducing extraneous exploration. The log shows that lowering temperature to 0.5 provided an additional +2% instance‑F1.

**Single‑run inference.** The original `predict` method uses one set of samples to compute a single P‑matrix, making the inference susceptible to sampling noise. The log reports that averaging P‑matrices from four independent runs (125 samples each) gave roughly +1% instance‑F1, reducing the impact of outlier sequences.

## 4. Optimization Methodology

The AutoSOTA pipeline operated on the repository and evaluated modifications over nine iterations. The best‑performing commit is `2ccf0b94e4d357c371dff3fc7111730b6109a6f2`. Across the iterations, eight distinct hyperparameter or architectural changes were attempted. Only three inference‑time changes improved instance‑F1; all training‑time modifications caused regression and were rejected.

**Intervention 1: Increase `num_samples_to_infer` from 200 to 500.**  
File: `src/predictor/ar.py`, constructor and `_infer_f1` method.  
Increasing the sample count by a factor of 2.5 reduces the standard error of each P‑matrix entry by approximately √2.5 ≈ 1.58. The sharper estimate enables the F1‑optimizer to select label sets with higher true instance‑F1. The log attributes approximately +2% relative improvement to this change.

**Intervention 2: Reduce `temperature` from 1.0 to 0.5.**  
File: `src/predictor/ar.py`, sampling call within the autoregressive loop.  
Setting T=0.5 makes the softmax distribution more peaked, favoring exploitation of the model’s highest‑confidence predictions. This is especially beneficial when the label distribution is concentrated around the mode. The log credits this change with roughly +2% instance‑F1.

**Intervention 3: Enable multi‑run P‑matrix averaging with `num_ensemble_runs=4`.**  
File: `src/predictor/ar.py`, `predict` method, multi‑run block.  
When `num_ensemble_runs>1`, the total sampling budget is split equally, and the P‑matrices from individual runs are averaged before inference. The ensemble average reduces variance without increasing the overall sample count. With a budget of 500, each run draws 125 samples. The log indicates an additional +1% instance‑F1 from this change.

All three interventions were applied simultaneously and require no model retraining; they only affect inference‑time logic. The final configuration corresponds to `--predictor.num_samples_to_infer 500`, temperature=0.5 (set in configuration), and `num_ensemble_runs=4`.

## 5. Experiments

### 5.1 Setup

The repository supports several multi‑label datasets, though the specific dataset and split used in this optimization run are not reported in the log; only baseline metric values are given. Hardware details, seeds, and the exact configuration file are similarly absent. The pipeline ran nine iterations with a target instance‑F1 ≥ 0.5641, and the best commit reached 0.5655. The pipeline had the ability to modify hyperparameters and code; training was not constrained except by the original training pipeline.

### 5.2 Quantitative Results

All five reported metrics improved relative to the baseline. Table 1 presents the baseline and best values.

| Metric            | Baseline | Best   | Absolute Δ | Relative Change |
|-------------------|----------|--------|------------|-----------------|
| instance_f1       | 0.5372   | 0.5655 | +0.0283    | +5.3%           |
| hamming_accuracy  | 0.9602   | 0.9701 | +0.0099    | +1.0%           |
| subset_accuracy   | 0.2853   | 0.3116 | +0.0263    | +9.2%           |
| micro_f1          | 0.4685   | 0.5482 | +0.0797    | +17.0%          |
| macro_f1          | 0.2628   | 0.2754 | +0.0126    | +4.8%           |

**Table 1:** Baseline and optimized metrics. All values rounded to four decimal places; relative change = (best − baseline)/baseline × 100%. All differences are positive.

Micro‑F1 (+17.0%) and subset accuracy (+9.2%) show the largest relative increases, indicating that the refined inference not only raises instance‑level F1 but also substantially improves label‑level precision‑recall balance and exact‑match accuracy.

### 5.3 Ablation / Iteration Trajectory

The log records eight interventions tested during the nine iterations, with five training‑time changes rejected and three inference‑time changes accepted. The ordering below lists training‑time attempts first, followed by inference‑time changes, consistent with how the pipeline explored the search space.

1. **AdamW + CosineAnnealingWarmRestarts** – Replaced the default optimizer with AdamW and a cosine annealing schedule; instance‑F1 regressed by 2.6%.  
2. **Early stopping on validation instance‑F1** – Used validation instance‑F1 as the stopping signal; instance‑F1 regressed by 6.4%. The noise in validation instance‑F1 causes premature termination.  
3. **GELU activation** – Substituted ReLU with GELU in the MLP; instance‑F1 dropped by 15.8%, indicating that ReLU‑induced sparsity benefits the autoregressive architecture.  
4. **Temperature annealing during training** – Since training relies on ground‑truth labels and BCE loss, altering sampling temperature has no effect on gradients; no change in performance was observed.  
5. **Label‑sensitive pos_weight loss** – Introduced per‑label positive weights to address label imbalance; even with conservative caps, the weighted loss distorted training, lowering instance‑F1.  
6. **Inference: increase samples to 500** – +~2% instance‑F1. Accepted.  
7. **Inference: reduce temperature to 0.5** – +~2% instance‑F1. Accepted.  
8. **Inference: multi‑run P‑matrix averaging (4 runs)** – +~1% instance‑F1. Accepted.

The three accepted changes (steps 6–8) were applied sequentially and cumulatively produced the final +5.3% gain shown in Table 1. The individual contributions are approximate, as reported in the log.

## 6. Discussion

The optimization confirms that inference‑time hyperparameters significantly affect the performance of sampling‑based F1 inference. Increasing the Monte Carlo sample budget reduces P‑matrix noise; lowering the temperature sharpens the sampling distribution; and multi‑run averaging reduces variance further. All five training‑time interventions were detrimental, suggesting that the original configuration—BCE loss, ReLU activations, and the default optimizer—is already appropriate for this autoregressive architecture.

Several limitations affect the generalizability of these findings. The optimization was conducted on a single, unspecified dataset; the optimal temperature and sample count may not transfer to datasets with different label cardinalities or feature distributions. The TAKEAWAY log notes that per‑label temperature tuning and adaptive thresholds remain future directions that could handle label imbalance more finely. The lack of multiple‑seed trials means that the reported +5.3% gain could contain some variance from random sampling; however, the directional consistency across all metrics strengthens confidence. Reproducibility is limited by the absence of documented hardware and seed information.

## 7. Reproducibility

- **Repository:** https://github.com/… (exact path not provided; source structure matches the described layout).  
- **Environment:** Standard PyTorch, torchvision, scikit‑multilearn, numpy, etc. Install with `pip install -r requirements.txt`.  
- **Seed:** Not specified; the `fix_seed` utility in `src/utils.py` handles default seeding.  
- **Baseline inference (default settings):**  
  `python src/main.py --config config.json --test-only --predictor.num_samples_to_infer 200` (with temperature=1.0 and `num_ensemble_runs=1` in the configuration file).  
- **Optimized inference (accepted changes):**  
  `python src/main.py --config config.json --test-only --predictor.num_samples_to_infer 500`  
  (and ensure the configuration sets temperature=0.5 and `num_ensemble_runs=4`).  

The training phase is unchanged; only the test‑only invocation with modified predictor arguments is needed to reproduce the optimized results.

## 8. References

1. Revisiting F‑measure Optimization in Multi‑Label Classification: A Sampling‑based Approach. (Full bibliographic metadata not available in the repository; the paper is referenced by title as it appears in the optimization log.)  
2. tsinghua-fib-lab/AutoSOTA. https://github.com/tsinghua-fib-lab/AutoSOTA. Automated State‑of‑the‑Art Optimization framework.
