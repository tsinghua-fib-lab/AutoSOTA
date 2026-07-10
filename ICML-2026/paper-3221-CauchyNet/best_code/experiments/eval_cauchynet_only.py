"""Fast evaluation: CauchyNet only, 10 seeds, 3000 epochs."""
import json, math, time, os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

sys.path.insert(0, os.path.dirname(__file__))
# Import everything from best_config_gap_filling.py except main()
with open(os.path.join(os.path.dirname(__file__), "best_config_gap_filling.py")) as f:
    code = f.read()
main_idx = code.find("\ndef main()")
exec(code[:main_idx])

torch.manual_seed(10)
np.random.seed(10)

tX, tY, vX, vY, teX, teY = build_data(seed=10)
train_loader = DataLoader(TensorDataset(tX, tY), batch_size=32, shuffle=True)
val_loader = DataLoader(TensorDataset(vX, vY), batch_size=32, shuffle=False)
test_data = (teX, teY)

n_seeds = 10
epochs = 3000
LR = 5e-2
H = 64

print("=" * 60)
print("CauchyNet Evaluation (ellipse (2.5, 0.4), fixed, h=%d, lr=%.3f)" % (H, LR))
print("=" * 60)

summary = run_model(
    "CauchyNet",
    lambda s: CauchyNet(hidden_size=H, r_re=2.5, r_im=0.4, seed=s),
    train_loader, val_loader, test_data,
    n_seeds=n_seeds, epochs=epochs, lr=LR, imag_pen=0.0,
                         use_cosine=True, warmup_epochs=150)

print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)
print("Mean MAE:  %.5f" % summary["mae_mean"])
print("Median:    %.5f" % summary["mae_median"])
print("Max:       %.5f" % summary["mae_max"])
print("Std:       %.5f" % summary["mae_std"])
print("Params:    %d" % summary["params"])
print("Train time: %.1fs (mean over %d seeds)" % (summary["train_time_s_mean"], n_seeds))

# Write results
out_dir = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "cauchynet_eval.json")
save_summary = {k: v for k, v in summary.items() if k != "per_seed"}
save_summary["per_seed"] = [
    {k3: v3 for k3, v3 in r.items() if k3 not in ("errs", "train_curve", "val_curve")}
    for r in summary["per_seed"]
]
with open(out_path, "w") as fp:
    json.dump(save_summary, fp, indent=2)
print("Saved: %s" % out_path)
