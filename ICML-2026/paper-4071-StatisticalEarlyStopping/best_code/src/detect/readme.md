# Construction of Uncertainty Keywords

We explain the semi-supervised construction of uncertainty keywords, which involves the following steps:

#### Step 1: Train a Random Forest Classifier
Train a random forest classifier with TF-IDF $k$-gram features ($k=2,3,4,5$) to discriminate between the two types of traces. Use the following steps:

1. Apply 5-fold stratified cross-validation to obtain five random forest models.
2. Average the feature importance scores across all folds.
3. Select the top features as initial candidates.

To do this, run:
```bash
python -m src.detect.classifier
```

This produces `classification_results/gsm8k_classification_results.json`, which contains the top $k$-gram features.

#### Step 2: Categorizing the Keyword Set

Organize the extracted $k$-gram features into three primary semantic categories:
- **Impossibility**
- **Speculation**
- **Insufficiency**

Assign each feature to a category using a keyword-based filtering approach. For each category, specify a small set of core terms that reflect its characteristic type of uncertainty. The core filtering terms are listed below:

| Category         | Core Filtering Terms                                                                 |
|------------------|---------------------------------------------------------------------------------|
| Impossibility    | not possible, impossible, cannot, hard              |
| Speculation | guess, assum, forgot, intend |
| Insufficiency    | missing, insufficient, incomplete, without, additional, lack, no data, no info, not helpful, not give, not specif, not recall, not provide, not include, not enough, not access, absence, ambiguous, vague |

For ablation studies, additional categories are defined separately:
- **Epistemic Uncertainty**: maybe, perhaps
- **Transition**: alternatively, wait

Keywords are assigned to the first matching category based on whether any core term appears in the $k$-gram feature.

This process yields three primary disjoint sets:
- $K_{\mathrm{imp}}$: Impossibility keywords
- $K_{\mathrm{spec}}$: Speculation keywords
- $K_{\mathrm{ins}}$: Insufficiency keywords

The full uncertainty keyword set is $K = K_{\mathrm{imp}} \cup K_{\mathrm{spec}} \cup K_{\mathrm{ins}}$.

In our experiments, our script `src/simulator.py` calls `get_uncertainty_keywords` in `src/detect/uncertainty_keywords.py` to retrieve the full keyword set $K$ for uncertainty detection.

#### Detect Abstention Using LLM Judge

The `src/detect/detectors.py` module provides the `LLMJudgeAbstentionDetector` class, which is used to determine whether a model's response constitutes an abstention. This is achieved by leveraging a large language model (LLM) to judge the response based on a predefined prompt.

In the `compute_metrics.py` script, the `LLMJudgeAbstentionDetector` is used to perform batched abstention detection. 

# LLM Judge Abstention Detection

`src/detect/detectors.py` implements the `LLMJudgeAbstentionDetector` class, which uses an LLM to determine if a model's response is an abstention. The system prompt is included in `evaluation_judge_prompt.py`.