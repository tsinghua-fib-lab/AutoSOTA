# Evaluating Binary Classifiers Under Label Shift

Reproduction code for "Aligning Evaluation with Clinical Priorities: Calibration, Label Shift, and Error Costs" (NeurIPS 2025).

This repository demonstrates how to evaluate clinical prediction models across demographic subgroups using the APACHE IV mortality predictor on the eICU dataset. The code generates accuracy vs. prevalence curves that visualize how model performance varies with class balance, calibration quality, and cost asymmetries.

## What This Code Does

The paper proposes a new evaluation metric (Bounded DCA Log Score) that addresses limitations of accuracy and AUC-ROC when evaluating clinical models. This code:

1. **Loads pre-computed APACHE IV predictions** from the public eICU database (no model training required)
2. **Analyzes subgroup performance** by race/ethnicity and gender
3. **Generates accuracy vs. prevalence curves** showing how performance changes across different class balances
4. **Compares calibration and discrimination** using both traditional metrics (AUC-ROC, ECE) and the proposed Bounded DCA Log Score

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Generate all paper figures (saves to ~/Desktop by default)
bash generate_figures.sh
```

This reproduces the main empirical results from Section 5 comparing Caucasian vs. African American patients and Male vs. Female patients.

## Key Figures

The paper's empirical analysis (Section 5) includes:

- **Table 1**: Calibration vs. discrimination trade-off for race subgroups
- **Figure (left)**: Gender subgroup accuracy curves with averaged performance
- **Appendix figures**: Detailed calibration decomposition

## Understanding the Output

Each plot shows:
- **Solid curves**: Accuracy at each prevalence level for different subgroups
- **Dashed curves**: Performance after recalibration (isolates discrimination)
- **Horizontal bars**: Bounded DCA Log Score (average across prevalence range)
- **Circles**: Accuracy at the empirical prevalence in the dataset

## Customization

The main script accepts numerous options. Common use cases:

```bash
# Analyze gender subgroups with custom prevalence range
python plot.py --demo --subgroup-field "gender" \
  --subgroups "Male" "Female" --maxlogodds 0.1

# Show calibration curves with AUC and ECE reference lines
python plot.py --demo --calibration --auc --ece

# Generate curves with confidence intervals (slower)
python plot.py --demo --ci --subgroup-field "ethnicity"
```

Key options:
- `--demo`: Use subset of data (faster)
- `--subgroup-field`: Choose `"ethnicity"` or `"gender"`
- `--calibration`: Show calibration curves (dashed lines)
- `--average`: Display Bounded DCA Log Score (horizontal bars)
- `--maxlogodds`: Set prevalence range (e.g., 0.1 restricts to reasonable clinical range)

Full option list: see `python plot.py --help`

## Data

The code uses pre-computed APACHE IV predictions from the [eICU Collaborative Research Database](https://eicu-crd.mit.edu/). No model training or raw data processing is required. The public subset is sufficient for reproduction.

## Code Structure

- [plot.py](plot.py) - Main script
- [generate_figures.sh](generate_figures.sh) - Reproduces all paper figures
- [core/curves.py](core/curves.py) - Net benefit curve plotting
- [stats/ece.py](stats/ece.py) - Expected calibration error
- [etl/eicu.py](etl/eicu.py) - eICU data loader

## Citation

```
@inproceedings{flores25,
  title={Aligning Evaluation with Clinical Priorities: Calibration, Label Shift, and Error Costs},
  author={Flores, Gerardo and Smith, Alyssa H. and Fukuyama, Julia A. and Wilson, Ashia C.},
  booktitle={NeurIPS},
  year={2025}
}
```