import numpy as np
import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.insert(0, "/repo")
from ocmb import OCMB_CaPS, calculate_metrics
from reproduce_experiment import generate_scale_free_graph, generate_nonlinear_gaussian_data

d = 100; n = 1000; seed = 42
true_adj = generate_scale_free_graph(d, 3, seed)
X = generate_nonlinear_gaussian_data(true_adj, n, seed)

ocmb = OCMB_CaPS(max_parents=5, k_mb=5, alpha_mb=0.01,
    score_threshold_quantile=0.95, use_spouse_closure=True,
    eta_G=0.001, eta_H=0.001, dispersion="mean", device="cuda:0", verbose=False)
ocmb.fit(X, true_adj=true_adj)
graph = ocmb.get_adjacency_matrix()
metrics = calculate_metrics(true_adj, graph)
timings = ocmb.get_timings()
n_cmi = ocmb.get_n_cmi_calls()
print("SHD=%d F1=%.3f Time=%.1fs CI=%d" % (metrics["SHD"], metrics["F1"], timings["total"], n_cmi))
