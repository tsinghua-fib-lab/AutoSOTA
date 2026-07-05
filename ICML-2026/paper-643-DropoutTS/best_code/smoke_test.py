"""Quick smoke test: 5 epochs, TimeMixer+DropoutTS on ETTh2, H=96."""
import os, sys, json, glob
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from basicts.models.TimeMixer import TimeMixerForForecasting, TimeMixerConfig
from basicts.configs import BasicTSForecastingConfig
from basicts.runners.callback import EarlyStopping, DropoutTSCallback
from basicts import BasicTSLauncher

mc = TimeMixerConfig(input_len=96, output_len=96, num_features=7)
cb = [
    DropoutTSCallback(p_min=0.05, p_max=0.5, init_alpha=10.0, init_sensitivity=1.0,
                      enable_visualization=False, enable_statistics=False),
    EarlyStopping(patience=10),
]
cfg = BasicTSForecastingConfig(
    model=TimeMixerForForecasting, model_config=mc,
    dataset_name="ETTh2", input_len=96, output_len=96,
    use_timestamps=False,
    gpus="0", num_epochs=5, batch_size=64,
    callbacks=cb, seed=42,
    train_data_num_workers=8, val_data_num_workers=8, test_data_num_workers=8,
    train_data_pin_memory=True, val_data_pin_memory=True, test_data_pin_memory=True,
)
print("Starting smoke test (5 epochs)...")
BasicTSLauncher.launch_training(cfg)

# Find metrics
mp = "checkpoints/TimeMixerForForecasting/ETTh2_5_96_96/test_metrics.json"
if os.path.exists(mp):
    with open(mp) as f:
        m = json.load(f)
    print("SMOKE TEST PASSED: MSE=%s, MAE=%s" % (
        m["overall"].get("MSE", "?"), m["overall"].get("MAE", "?")))
else:
    print("Metrics not found at " + mp)
    for f in sorted(glob.glob("checkpoints/**/test_metrics.json", recursive=True)):
        print("Found: " + f)
