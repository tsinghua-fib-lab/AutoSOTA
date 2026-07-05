# Fairness and Robustness Review

This document tracks potential critiques and limitations of the benchmarking system that should be addressed or documented before publication.

## Fixed Issues ✅

### 1. Inconsistent skip behavior for "ill-posed" status
**Status**: FIXED
**Location**: `generate_answers.py:82`
**Fix**: Now consistently skips both "failed" and "ill-posed" unless `--rerun-failures` is used.

### 2. Missing "ill-posed" questions in previous attempts
**Status**: FIXED
**Location**: `generate_benchmark.py:98`
**Fix**: Now includes both "failed" and "ill-posed" questions in previous_attempts list, so models learn from ill-posed questions when generating replacements.

---

## Open Issues - Require Documentation or Future Work

### 3. Model-Specific Advantages/Disadvantages

**Issue**: Some models might be better at self-critique than others, giving them an unfair advantage in the self-improvement loop. Similarly, some models might be better at debate than problem-solving.

**Impact**: Could inflate/deflate scores for certain models.

**Mitigation already in place**:
- Cross-model evaluation (different models answer different models' questions)
- All models use the same self-improvement loop with the same max rounds

**Recommendation for paper**:
- Document this as a known limitation
- Report correlation between self-critique accuracy and final benchmark score
- Consider ablation: benchmark score with vs. without self-improvement loop

**Analysis to run**:
```bash
# Check if models with higher self-critique pass rates score better
# Compare: (self-critique pass rate on round 1) vs. (final benchmark score)
```

---

### 4. Temperature Settings Consistency

**Issue**: Different temperatures across models could affect creativity, consistency, and debate performance.

**Check needed**: Verify temperature settings are consistent across roles.

**Current state**: Each model has its own temperature setting in the model registry.

**Questions to answer**:
1. Are temperatures consistent across all roles for a given model?
2. Do some models use different temperatures for question generation vs. answering?
3. Should temperature be standardized across all models for fairness?

**Recommendation**:
- Document the temperature settings used for each model in the paper
- If temperatures vary, report whether temperature correlates with benchmark performance
- Consider running sensitivity analysis: how do results change with temperature?

**Files to check**:
- `configs/models.json` - model temperature settings
- `model_config.py` - how temperature is loaded and used

---

### 5. Lack of Inter-Rater Reliability Metrics

**Issue**: For human evaluation, reviewers will want to know agreement rates.

**Metrics needed**:
1. **Judge-judge agreement**: Do different judge models agree on the same debates?
2. **Judge-human agreement**: Do automated judges agree with human labels?
3. **Confidence calibration**: Are high-confidence judgments more accurate than low-confidence ones?

**Recommendation for paper**:
- Report Fleiss' kappa or Krippendorff's alpha for judge agreement
- Report precision/recall of automated judges against human labels
- Plot calibration curve: predicted confidence vs. actual accuracy

**Analysis to implement**:
```python
# Add to stats.py or new file: judge_agreement.py

def compute_judge_agreement(judgments_dir: Path) -> Dict:
    """
    For each debate, collect all judge verdicts and compute:
    - Pairwise agreement rate
    - Fleiss' kappa
    - Majority verdict
    """
    pass

def compute_confidence_calibration(judgments_dir: Path, labels_dir: Path) -> Dict:
    """
    For each confidence level (1-5), compute accuracy of judgments
    """
    pass
```

---

### 6. Potential Gaming Through Verbosity

**Issue**: Models might learn that longer, more detailed answers are less likely to be critiqued, even if the mathematics isn't better.

**Risk**: Could reward verbosity over correctness.

**Check needed**: Analyze whether answer length correlates with critique verdict.

**Analysis to implement**:
```python
# Add to stats.py or new analysis file

def analyze_answer_length_bias(answers_dir: Path, critiques_dir: Path):
    """
    For each answer:
    - Measure length (words, characters, or lines)
    - Get critique verdict
    - Compute correlation between length and "correct" verdict
    - Control for question difficulty (topic)
    """
    pass

# Expected result: No significant correlation
# If correlation exists: This is a known limitation to document
```

**Recommendation**:
- Report mean answer length by verdict category
- Report correlation coefficient with p-value
- If significant: document as limitation and suggest future work on length-normalized scoring

---

### 7. Verdict Scoring Rules (For Future Implementation)

**Issue**: How do "tie" and "unknown" verdicts map to benchmark scores?

**Current state**: Undefined - needs to be implemented when computing final scores.

**Options**:

**Option A: Strict scoring**
- "defender_wins" = 1 point (answer correct)
- "claimant_wins" = 0 points (answer incorrect)
- "tie" = 0 points (conservative: treat ambiguous as incorrect)
- "unknown" = exclude from score calculation

**Option B: Partial credit**
- "defender_wins" = 1 point
- "claimant_wins" = 0 points
- "tie" = 0.5 points
- "unknown" = exclude from score calculation

**Option C: Multiple metrics**
Report both:
- **Strict score**: Only count clear defender_wins
- **Optimistic score**: Count defender_wins + tie
- **Coverage**: Percentage of non-"unknown" judgments

**Recommendation**: Use Option C to provide a comprehensive view.

**Implementation location**: New file `scoring.py` or add to existing scoring logic.

---

## Summary for Publication

### Strengths to Emphasize:
1. ✅ Cross-model evaluation prevents self-serving bias
2. ✅ Self-improvement loop gives all models equal opportunity to refine
3. ✅ Debate-based verification leverages verification-generation asymmetry
4. ✅ Consistent handling of ill-posed questions across all stages
5. ✅ Raw answers stored for transparency and analysis

### Limitations to Document:
1. Self-critique ability may vary across models
2. Temperature settings may affect results (document settings used)
3. Need inter-rater reliability metrics for judge evaluation
4. Potential length bias needs to be analyzed and reported
5. Verdict scoring rules need to be clearly defined and justified

### Future Work:
1. Ablation studies on self-improvement loop impact
2. Temperature sensitivity analysis
3. Length-normalized scoring variants
4. Human evaluation of a sample to validate automated judges
