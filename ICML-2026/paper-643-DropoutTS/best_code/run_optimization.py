"""Parameterized optimization runner for DropoutTS SOTA optimization."""
import os, sys, json, glob
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from basicts.models.TimeMixer import TimeMixerForForecasting, TimeMixerConfig
from basicts.configs import BasicTSForecastingConfig
from basicts.runners.callback import EarlyStopping, DropoutTSCallback
from basicts.runners.callback.clip_grad import GradientClipping
from basicts.runners.optim.lr_schedulers_fixed import CosineWarmupFixed as CosineWarmup
from basicts import BasicTSLauncher

DATASET = os.environ.get("DATASET", "ETTh2")
INPUT_LEN = int(os.environ.get("INPUT_LEN", "96"))
OUTPUT_LENS = json.loads(os.environ.get("OUTPUT_LENS", "[96, 192, 336, 720]"))
GPU = os.environ.get("GPU", "0")
P_MIN = float(os.environ.get("DROPOUTTS_P_MIN", "0.05"))
P_MAX = float(os.environ.get("DROPOUTTS_P_MAX", "0.5"))
ALPHA = float(os.environ.get("DROPOUTTS_ALPHA", "10.0"))
GAMMA = float(os.environ.get("DROPOUTTS_GAMMA", "1.0"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "64"))
LR = float(os.environ.get("LR", "2e-4"))
WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", "5e-4"))
NUM_EPOCHS = int(os.environ.get("NUM_EPOCHS", "100"))
SEED = int(os.environ.get("SEED", "42"))
USE_CLIP_GRAD = os.environ.get("USE_CLIP_GRAD", "0") == "1"
CLIP_GRAD_MAX_NORM = float(os.environ.get("CLIP_GRAD_MAX_NORM", "1.0"))
USE_COSINE_WARMUP = os.environ.get("USE_COSINE_WARMUP", "0") == "1"
WARMUP_EPOCHS = int(os.environ.get("WARMUP_EPOCHS", "5"))

print("=== DropoutTS Optimization Run ===")
print(f"  gamma={GAMMA}, p_min={P_MIN}, p_max={P_MAX}, alpha={ALPHA}")
print(f"  batch_size={BATCH_SIZE}, lr={LR}, weight_decay={WEIGHT_DECAY}")
print(f"  epochs={NUM_EPOCHS}, seed={SEED}")
print(f"  clip_grad={USE_CLIP_GRAD}, clip_max_norm={CLIP_GRAD_MAX_NORM}")
print(f"  cosine_warmup={USE_COSINE_WARMUP}, warmup_epochs={WARMUP_EPOCHS}")
print(f"  gpu={GPU}")

SEP = "=" * 60

def find_metrics(ol):
    pattern = f"checkpoints/TimeMixerForForecasting/ETTh2_{NUM_EPOCHS}_{INPUT_LEN}_{ol}/*/test_metrics.json"
    matches = glob.glob(pattern)
    if not matches:
        return None
    # Pick by modification time, NOT alphabetical (bug fix)
    return max(matches, key=lambda p: os.stat(p).st_mtime)

all_results = {}
for output_len in OUTPUT_LENS:
    tag = f"I{INPUT_LEN}_O{output_len}"
    print(f"\n{SEP}")
    print(f"  TimeMixer + DropoutTS on {DATASET}  {tag}")
    print(SEP)

    mc = TimeMixerConfig(input_len=INPUT_LEN, output_len=output_len, num_features=7)

    cb = [
        DropoutTSCallback(
            p_min=P_MIN, p_max=P_MAX, init_alpha=ALPHA,
            init_sensitivity=GAMMA,
            enable_visualization=False, enable_statistics=False),
        EarlyStopping(patience=10),
    ]
    if USE_CLIP_GRAD:
        cb.insert(0, GradientClipping(max_norm=CLIP_GRAD_MAX_NORM))

    cfg_kwargs = dict(
        model=TimeMixerForForecasting, model_config=mc,
        dataset_name=DATASET, input_len=INPUT_LEN, output_len=output_len,
        use_timestamps=False,
        gpus=GPU, num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE,
        callbacks=cb, seed=SEED,
        train_data_num_workers=4, val_data_num_workers=4, test_data_num_workers=4,
        train_data_pin_memory=True, val_data_pin_memory=True, test_data_pin_memory=True,
        optimizer_params={"lr": LR, "weight_decay": WEIGHT_DECAY},
    )

    if USE_COSINE_WARMUP:
        cfg_kwargs["lr_scheduler"] = CosineWarmup
        cfg_kwargs["lr_scheduler_params"] = {
            "num_warmup_steps": WARMUP_EPOCHS,
            "num_training_steps": NUM_EPOCHS,
        }

    cfg = BasicTSForecastingConfig(**cfg_kwargs)
    BasicTSLauncher.launch_training(cfg)

    mp = find_metrics(output_len)
    if mp:
        with open(mp) as f:
            m = json.load(f)
        all_results[output_len] = m["overall"]
        mse_v = m["overall"].get("MSE", -1)
        mae_v = m["overall"].get("MAE", -1)
        print(f"  H={output_len}: MSE={mse_v:.6f}  MAE={mae_v:.6f}")
    else:
        print(f"  H={output_len}: METRICS NOT FOUND")
        all_results[output_len] = None

# Summary
mse_vals = [v["MSE"] for v in all_results.values() if v and "MSE" in v]
mae_vals = [v["MAE"] for v in all_results.values() if v and "MAE" in v]

print(f"\n{SEP}")
print("OPTIMIZATION RESULTS: TimeMixer + DropoutTS on ETTh2")
print(SEP)
for h in OUTPUT_LENS:
    v = all_results.get(h)
    if v:
        mse_v = v.get("MSE", -1)
        mae_v = v.get("MAE", -1)
        print(f"  H={h:4d}: MSE={mse_v:.6f}  MAE={mae_v:.6f}")
    else:
        print(f"  H={h:4d}: FAILED")

if mse_vals:
    avg_mse = float(np.mean(mse_vals))
    avg_mae = float(np.mean(mae_vals))
    print(f"  Avg  : MSE={avg_mse:.6f}  MAE={avg_mae:.6f}")
    print(f"  Paper: MSE=0.380   MAE=0.399")

summary = {
    "model": "TimeMixer+DropoutTS", "dataset": DATASET,
    "input_len": INPUT_LEN, "output_lens": OUTPUT_LENS,
    "hyperparams": {
        "p_min": P_MIN, "p_max": P_MAX, "init_alpha": ALPHA,
        "init_sensitivity": GAMMA, "batch_size": BATCH_SIZE,
        "lr": LR, "weight_decay": WEIGHT_DECAY, "seed": SEED,
        "clip_grad": USE_CLIP_GRAD, "clip_max_norm": CLIP_GRAD_MAX_NORM,
        "cosine_warmup": USE_COSINE_WARMUP, "warmup_epochs": WARMUP_EPOCHS,
    },
    "per_horizon": {str(k): v for k, v in all_results.items()},
    "avg_MSE": avg_mse if mse_vals else None,
    "avg_MAE": avg_mae if mae_vals else None,
    "paper_MSE": 0.380, "paper_MAE": 0.399,
}
with open("optimization_results.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nSaved to optimization_results.json")
