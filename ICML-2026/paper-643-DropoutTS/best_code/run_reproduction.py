"""Reproduction: TimeMixer + DropoutTS on ETTh2.
Paper Table 4: input_len=96, H in {96,192,336,720}, averaged.
Hyperparams: p_min=0.05, p_max=0.5, alpha=10.0, gamma=1.0 (Section 5.5).
"""
import os, sys, json, glob

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

import numpy as np
from basicts.models.TimeMixer import TimeMixerForForecasting, TimeMixerConfig
from basicts.configs import BasicTSForecastingConfig
from basicts.runners.callback import EarlyStopping, DropoutTSCallback
from basicts import BasicTSLauncher

DATASET, INPUT_LEN, OUTPUT_LENS = "ETTh2", 96, [96, 192, 336, 720]
GPU = "0"
P_MIN, P_MAX = 0.05, 0.5
ALPHA, SENS = 10.0, 1.0
EPOCHS = 100

def find_metrics(ol):
    pattern = "checkpoints/TimeMixerForForecasting/ETTh2_%d_%d_%d/*/test_metrics.json" % (EPOCHS, INPUT_LEN, ol)
    matches = sorted(glob.glob(pattern))
    return matches[-1] if matches else None

all_results = {}
for output_len in OUTPUT_LENS:
    tag = "I%d_O%d" % (INPUT_LEN, output_len)
    print("\n" + "=" * 60)
    print("  TimeMixer + DropoutTS on %s  %s" % (DATASET, tag))
    print("=" * 60)

    mc = TimeMixerConfig(input_len=INPUT_LEN, output_len=output_len, num_features=7)
    cb = [
        DropoutTSCallback(p_min=P_MIN, p_max=P_MAX, init_alpha=ALPHA,
                          init_sensitivity=SENS,
                          enable_visualization=False, enable_statistics=False),
        EarlyStopping(patience=10),
    ]
    cfg = BasicTSForecastingConfig(
        model=TimeMixerForForecasting, model_config=mc,
        dataset_name=DATASET, input_len=INPUT_LEN, output_len=output_len,
        use_timestamps=False,
        gpus=GPU, num_epochs=EPOCHS, batch_size=64,
        callbacks=cb, seed=42,
        train_data_num_workers=8, val_data_num_workers=8, test_data_num_workers=8,
        train_data_pin_memory=True, val_data_pin_memory=True, test_data_pin_memory=True,
    )
    BasicTSLauncher.launch_training(cfg)

    mp = find_metrics(output_len)
    if mp:
        with open(mp) as f:
            m = json.load(f)
        all_results[output_len] = m["overall"]
        print("  H=%d: MSE=%.6f  MAE=%.6f" % (output_len,
            m["overall"].get("MSE", -1), m["overall"].get("MAE", -1)))
    else:
        print("  H=%d: METRICS NOT FOUND" % output_len)
        all_results[output_len] = None

# Summary
mse_vals = [v['MSE'] for v in all_results.values() if v and 'MSE' in v]
mae_vals = [v['MAE'] for v in all_results.values() if v and 'MAE' in v]

print("\n" + "=" * 60)
print("REPRODUCTION RESULTS: TimeMixer + DropoutTS on ETTh2")
print("=" * 60)
for h in OUTPUT_LENS:
    v = all_results.get(h)
    if v:
        print("  H=%4d: MSE=%.6f  MAE=%.6f" % (h, v.get('MSE', -1), v.get('MAE', -1)))
    else:
        print("  H=%4d: FAILED" % h)
if mse_vals:
    avg_mse = float(np.mean(mse_vals))
    avg_mae = float(np.mean(mae_vals))
    print("  Avg  : MSE=%.6f  MAE=%.6f" % (avg_mse, avg_mae))
    print("  Paper: MSE=0.380   MAE=0.399")

summary = {
    "model": "TimeMixer+DropoutTS", "dataset": DATASET,
    "input_len": INPUT_LEN, "output_lens": OUTPUT_LENS,
    "hyperparams": {"p_min": P_MIN, "p_max": P_MAX, "init_alpha": ALPHA, "init_sensitivity": SENS},
    "per_horizon": {str(k): v for k, v in all_results.items()},
    "avg_MSE": avg_mse if mse_vals else None,
    "avg_MAE": avg_mae if mae_vals else None,
    "paper_MSE": 0.380, "paper_MAE": 0.399,
}
with open("reproduction_results.json", "w") as f:
    json.dump(summary, f, indent=2)
print("\nSaved to reproduction_results.json")
