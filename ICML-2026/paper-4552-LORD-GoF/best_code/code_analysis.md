# Code Analysis: LORD-GoF Watermark Detection (Paper 4552)

## Evaluation Path
- Main script: `/repo/LORD-GoF-Analysis/TemperatureAnalysis.py`
- Eval command: `cd /repo/LORD-GoF-Analysis && python3 TemperatureAnalysis.py`
- Metrics extracted from stdout: FDR and Power at tau=0.5 for LORD-Anderson (and other methods)

## Key Components
1. **OnlineLORD class** - Sequential FDR controller with gamma sequence
2. **GoF static methods** - 8 GoF statistics (Kolmogorov, Kuiper, Anderson, Cramer, Watson, Chi_squared, Rao, Greenwood)
3. **Calibrator class** - Tail calibration using linear regression on log p-values (top 10%)
4. **load_pool()** - Loads pre-computed Y values from pickle files
5. **main()** - Runs stream experiment, prints results

## Config
- M_STREAM=1000, TOKEN_LEN=400, ALPHA=0.05, W0=0.01, GAMMA_EXP=1.2, N_CALIB=20000, PI=0.05, RHO=0.7
- 5 temperatures: [0.1, 0.3, 0.5, 0.7, 0.9]
- MODEL_PFX="opt1.3b", IS_INV=False

## Data
- Pre-computed pickle files in `raw_data/` for 3 models (opt1.3b, qwen2p5_3b, sheared_llama_2p7b)
- Each contains watermark Y values
- Uses real OPT-1.3B Gumbel-Max watermark data

## Safe Modification Targets
- `Calibrator.__init__()` - tail calibration method (GPD instead of linear regression)
- `Calibrator.pval()` - p-value computation
- `OnlineLORD.__init__()` and `test()` - FDR controller logic
- `GoF` static methods - numerical edge cases
- `N_CALIB`, `W0`, `GAMMA_EXP` - hyperparameters
- Add new GoF combination methods

## Risky Files
- `raw_data/*.pkl` - DO NOT modify (evaluation data)
- Metric extraction logic in main() - DO NOT change how metrics are computed
- Random seed for label generation - could change if justified (multi-seed)

## Red-Line Boundaries
- No changing evaluation protocol
- No modifying test data/labels/splits
- No hard-coding outputs
- Must report all metrics honestly
