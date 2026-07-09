import sys, os, time
sys.path.insert(1, os.path.join(sys.path[0], "../"))
import numpy as np
import yaml
from core.methods import UniversalPortfolio
from core.model_scores import generate_forecasts
from datasets import load_dataset
from core.runner import run_conformal_inference
from plotting_utils import longest_true_sequence

args = yaml.safe_load(open("./configs/AXP.yaml"))
config_name = "AXP"
print("Loading data...", flush=True)
data = load_dataset(args["sequences"][0]["dataset"])
print(f"Data shape: {data.shape}", flush=True)

print("Generating forecasts...", flush=True)
t0 = time.time()
model_name = "prophet"
seq_args = dict(args["sequences"][0])
seq_args["savename"] = "./datasets/processed/" + config_name + "/" + model_name + ".npz"
seq_args["T_burnin"] = args["T_burnin"]
seq_args["ahead"] = 1
seq_args["model_name"] = model_name
fc = generate_forecasts(data, **seq_args)
print(f"  Forecasts done in {time.time()-t0:.1f}s, shape={fc.shape}", flush=True)
data["forecasts"] = fc

print("Computing scores...", flush=True)
scores_list = [np.abs(y - f) for y, f in zip(data["y"], data["forecasts"])]
data["scores"] = np.array(scores_list)
print("  Scores done", flush=True)

print("Running UP method...", flush=True)
t0 = time.time()
predictor = UniversalPortfolio(alpha=args["alpha"])
r = run_conformal_inference(data["scores"], predictor, T_burnin=args["T_burnin"])
print(f"  UP done in {time.time()-t0:.1f}s", flush=True)

q = r["predicted_scores"]
c = r["coverages"]
T_burnin = args["T_burnin"]

q_burnin = q[T_burnin:]
c_burnin = c[T_burnin:]

print(f"Marginal Coverage: {np.mean(c_burnin):.4f}", flush=True)

set_sizes = 2 * q_burnin
print(f"Avg Set Size: {np.mean(set_sizes):.2f}", flush=True)
print(f"Median Set Size: {np.median(set_sizes):.2f}", flush=True)
for pct in [75, 90, 95]:
    print(f"{pct}% Quantile Size: {np.percentile(set_sizes, pct):.2f}", flush=True)

errors = 1 - c_burnin
print(f"Longest Err. Seq.: {longest_true_sequence(errors.astype(bool))}", flush=True)
