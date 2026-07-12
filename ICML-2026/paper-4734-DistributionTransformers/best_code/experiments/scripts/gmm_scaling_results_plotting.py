import matplotlib.pyplot as plt
import numpy as np
from json import load
import os

result_indexes = list(range(70, 78))

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

state_size = []
model_expected_kl = []
model_conf_kl = []
vi_expected_kl = []
vi_conf_kl = []

for idx in result_indexes:
    with open(ROOT_DIR + f"/../experiments/runs/gmm_closed_form/{idx}/config.json") as f:
        state_size.append(load(f)["state_size"])
    with open(ROOT_DIR + f"/../experiments/runs/gmm_closed_form/{idx}/info.json") as f:
        d = load(f)
        model_expected_kl.append(d["model_expected_posterior_kl_divergence"])
        model_conf_kl.append(d["model_conf_posterior_kl_divergence"])
        vi_expected_kl.append(d["vi_expected_kl_divergence"])
        vi_conf_kl.append(d["vi_conf_kl_divergence"])

model_expected_kl = np.array(model_expected_kl).flatten()
model_conf_kl = np.array(model_conf_kl).flatten()
vi_expected_kl = np.array(vi_expected_kl).flatten()
vi_conf_kl = np.array(vi_conf_kl).flatten()


plt.style.use(['seaborn-v0_8-paper'])
fig, ax = plt.subplots()
ax.set_yscale('log')

ax.plot(state_size, vi_expected_kl)
ax.plot(state_size, model_expected_kl)
ax.legend(["SVI", "8-Component DT"])

ax.fill_between(state_size, vi_expected_kl-vi_conf_kl, vi_expected_kl+vi_conf_kl, alpha=0.5)
ax.fill_between(state_size, model_expected_kl-model_conf_kl, model_expected_kl+model_conf_kl, alpha=0.5)

ax.set_xlabel("Latent Variable Dimensionality")
ax.set_ylabel("Expected KL-Divergence")

fig.savefig(ROOT_DIR + "/../experiments/GMM_scaling.pdf")
