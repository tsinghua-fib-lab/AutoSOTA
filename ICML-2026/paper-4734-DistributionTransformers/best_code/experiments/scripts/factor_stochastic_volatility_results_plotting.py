import matplotlib.pyplot as plt
import numpy as np
from json import load
import os


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_IDX = 46

results = load(open(os.path.join(ROOT_DIR, f'../runs/factor_stochastic_volatility/{RESULTS_IDX}/info.json')))

pf_nll = np.array(results["particle_filter_nll_series"])
pf_nll_conf = np.array(results["particle_filter_nll_conf_series"])
pf_upper = pf_nll + pf_nll_conf
pf_lower = pf_nll - pf_nll_conf
pf_inference_time = np.array(results["particle_filter_single_inference_time_series"])
pf_frequency = 1 / pf_inference_time

model_nll = results['model_nll'] * np.ones(2)
model_nll_conf = results['model_nll_conf'] * np.ones(2)
model_upper = (model_nll + model_nll_conf)
model_lower = (model_nll - model_nll_conf)
model_inference_time = results['model_single_inference_time']
model_frequency_scalar = 1 / model_inference_time
model_frequency = np.array([min(pf_frequency), model_frequency_scalar])

plt.style.use(['seaborn-v0_8-paper'])
fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.plot(pf_frequency, pf_nll, color="tab:blue", label="PF")
ax.plot(model_frequency, model_nll, color="tab:orange", label="DT")
plt.legend()
ax.fill_between(pf_frequency, pf_lower, pf_upper, color="tab:blue", alpha=0.5)
ax.fill_between(model_frequency, model_lower, model_upper, color="tab:orange", alpha=0.5)
ax.scatter(model_frequency[-1], model_nll[-1], color="black", marker="*")
ax.set_xlabel("Minimum Required Processing Frequency / Hz")
ax.set_ylabel("Minimum Achievable Expected NLL")
ax.set_xlim(right=500)
ax.set_ylim(top=100)
fig.savefig(ROOT_DIR + "/plots/factor_stochastic_volatility_results.pdf")
plt.show()
