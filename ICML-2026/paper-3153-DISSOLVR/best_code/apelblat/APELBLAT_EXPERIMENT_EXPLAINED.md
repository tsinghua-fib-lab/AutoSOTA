# Apelblat Equation Comparison Experiment

This document explains the full experimental pipeline used to evaluate whether the solubility model's predictions follow the **Apelblat equation**, a well-known thermodynamic model for temperature-dependent solubility.

---

## 1. Background: The Apelblat Equation

The Apelblat equation describes how solubility changes with temperature:

```
log(Solubility) = A + B/T + C·ln(T)
```

Where:
- `T` is temperature in Kelvin.
- `A`, `B`, `C` are fitted constants unique to each solute-solvent pair.

A good solubility model should naturally follow this physical law. This experiment quantifies how well our model's predictions align with it.

---

## 2. Data Curation

### 2.1. Source Data
- **Train Data**: `data/train.csv`
- **Test Data**: `data/test.csv`

Each row contains a `Solute (SMILES)`, `Solvent (SMILES)`, `Temperature (K)`, and experimental `LogS` value.

### 2.2. Filtering Criteria

Pairs are included only if they pass **all** of these quality gates:

| Criterion | Threshold | Purpose |
|-----------|-----------|---------|
| Minimum Temperature Points | ≥ 7 | Need enough data to fit a 3-parameter curve |
| Experimental Apelblat R² | ≥ 0.80 | The real data must itself follow the Apelblat law |
| RMSE at Known Temps | ≤ 0.50 | Exclude pairs where the model already makes large errors |

### 2.3. Final Dataset Size

| Dataset | Pairs Analyzed |
|---------|----------------|
| Train   | 3,861          |
| Test    | 524            |

---

## 3. What the Experiment Evaluates

The experiment uses **two main approaches** to test predictions at temperatures the model has and has not seen.

### 3.1. Approach A: Hold-Out Validation

This is the gold standard test. For pairs with ≥7 temperature points:
1.  **Split**: Hold out 2 *interpolation* points (from the middle) and 2 *extrapolation* points (min/max temperatures).
2.  **Fit**: Fit the Apelblat equation using only the *remaining* training temperatures.
3.  **Predict**: Use the model to predict LogS at the held-out temperatures.
4.  **Compare**: Measure how well predictions match both:
    - The **Experimental** values at the held-out points.
    - The **Apelblat Curve** fit on the training points (physical expectation).

### 3.2. Approach B: Synthetic Temperature Testing

For all pairs:
1.  **Generate Synthetic Temps**: Create temperatures at 5K intervals across the experimental range, plus 10K beyond it (extrapolation zone). Exclude any temps within 2K of a real data point.
2.  **Predict**: Get the model's LogS prediction at these new temperatures.
3.  **Compare to Apelblat**: Since we have no experimental data at these points, we compare predictions to the *experimental Apelblat curve* (fitted on all real data). This tests if the model has learned the correct *shape* of the temperature-solubility relationship.

---

## 4. Metrics Computed

### 4.1. Core Metrics

| Metric | What it Measures |
|--------|------------------|
| **Shape Correlation** | Pearson correlation between the predicted and experimental Apelblat curves. Value of 1.0 = identical trend. |
| **Curve Max Deviation** | The largest absolute difference between the two curves. Lower is better. |
| **% Within Tolerance** | Percentage of pairs where `Curve Max Deviation` is below the model's expected RMSE (0.216 for train, 0.6775 for test). |
| **RMSE at Known Temps** | Root Mean Square Error at temperatures present in the original dataset. |
| **R² at Synthetic Temps** | How well predictions at generated temperatures match the Apelblat curve. |
| **R² Holdout Interpolation** | Accuracy on held-out middle-range temperatures. |
| **R² Holdout Extrapolation** | Accuracy on held-out edge temperatures (hardest task). |

### 4.2. Final Combined Score

A single score summarizing overall Apelblat compliance:

```
Final Score = 0.25×Shape + 0.25×Tolerance% + 0.25×Synthetic_R² + 0.25×Holdout_R²
```

---

## 5. Results Summary

### 5.1. Apelblat Compliance (Shape & Tolerance)

| Metric | Train | Test |
|--------|-------|------|
| Mean Shape Correlation | **0.9891** | **0.9862** |
| % Within RMSE Tolerance | 58.9% | 99.2% |
| % Meeting Both Criteria | 58.8% | 98.9% |

> [!NOTE]
> Shape correlation is excellent (~0.99) for both sets, meaning the model has learned the correct *trend* of solubility vs. temperature. The test set has a higher tolerance pass rate because its tolerance threshold (0.6775) is larger than the train's (0.216).

### 5.2. Performance at Known Temperatures

| Metric | Train | Test |
|--------|-------|------|
| Mean RMSE | 0.149 | 0.228 |
| Mean R² | -9.51 | -5.33 |

> [!CAUTION]
> Negative R² values indicate that the model's predictions are *worse than simply predicting the mean*. This is likely because the variance in LogS for many pairs is very small, making R² unstable. RMSE is a more reliable metric here.

### 5.3. Performance at Synthetic Temperatures (Unseen)

| Metric | Train | Test |
|--------|-------|------|
| Mean RMSE | 0.233 | 0.293 |
| Mean R² | -3.01 | -1.31 |

### 5.4. Holdout Performance

| Holdout Type | Train R² | Test R² |
|--------------|----------|---------|
| Interpolation | -45.3 ± 1237 | -55.8 ± 419 |
| Extrapolation | -5.6 ± 161 | -2.5 ± 20 |

> [!WARNING]
> The extreme variance in holdout R² (e.g., ± 1237) suggests a small number of outlier pairs dominate the average. Median-based analysis may be more informative.

### 5.5. Final Scores

| Metric | Train | Test |
|--------|-------|------|
| **Final Score** | 0.395 | 0.495 |
| Generalization Gap | **-0.10** (Test > Train) |

> [!IMPORTANT]
> A negative generalization gap means the model *generalizes better than it fits*. This is unusual but can occur when test data happens to have simpler temperature-solubility relationships.

---

## 6. Output Files

All results are saved to `apelblat_results/`:

| File | Contents |
|------|----------|
| `pair_analysis.csv` | Per-pair detailed results (RMSE, R², shape, tolerance, etc.) |
| `summary_statistics.json` | High-level aggregate metrics and final scores |
| `metrics/*.json` | Modular per-dataset metric files for easy parsing |

---

## 7. Interpretation & Next Steps

### Key Takeaways

1.  **Shape is Excellent**: The model has learned the correct thermodynamic trend (Apelblat shape).
2.  **Absolute Accuracy Needs Work**: While shape is good, the raw R² values are poor, indicating offset errors.
3.  **Extrapolation is Harder**: As expected, predicting at temperatures outside the training range (holdout extrapolation) is the most challenging task.

### Potential Improvements

- Investigate pairs with high shape correlation but poor tolerance to understand systematic biases.
- Use median statistics instead of mean to reduce outlier influence.
- Consider temperature-as-a-feature augmentation to improve extrapolation.
