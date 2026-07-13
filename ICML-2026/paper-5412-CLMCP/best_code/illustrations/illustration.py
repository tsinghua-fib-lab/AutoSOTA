import math
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

# -----------------------------
# Environment: piecewise mean
# -----------------------------
def simulate_env(T, taus, mus, sigma=1.0, seed=1):
    """
    taus: [tau0=1, tau1, ..., tauS, tau_{S+1}=T+1]
    mus:  [mu0, mu1, ..., muS]  (S+1 means, one per segment)
    """
    rng = np.random.default_rng(seed)
    mu_t = np.zeros(T + 1)  # index 1..T used
    for j in range(len(mus)):
        start, end = taus[j], taus[j + 1]
        mu_t[start:end] = mus[j]
    X = mu_t[1:] + sigma * rng.standard_normal(T)
    return X, mu_t[1:]

# -----------------------------
# ATC algorithm (brute-force scan)
# -----------------------------
def atc_run(X, sigma=1.0, alpha=0.05):
    r"""
    Implements:
      \hat D_{k,t}^r = sqrt((k-r)(t-k)/(t-r)) * |mean(r+1:k) - mean(k+1:t)|
      C_t^r = max_{r<k<t} \hat D_{k,t}^r
      gamma_t^r = sigma * sqrt(6 log(t-r) + 2 log(1/alpha) + 2 log(pi^2/3))
      alarm when C_t^r >= gamma_t^r, then restart r <- t
    Indexing:
      time t = 1..T
      restart r is in {0,1,...}, with r=0 meaning "start from t=1".
    """
    T = len(X)
    cs = np.concatenate([[0.0], np.cumsum(X)])  # cs[t] = sum_{i=1}^t X_i

    r = 0
    hatmu = np.zeros(T + 1)   # store for t=1..T
    C = np.zeros(T + 1)
    gamma = np.full(T + 1, np.nan)
    alarms = []

    for t in range(1, T + 1):
        # estimator: running average since last restart
        hatmu[t] = (cs[t] - cs[r]) / (t - r)

        # compute statistic only if there exists a split point k: need t >= r+2
        if t >= r + 2:
            best = -np.inf
            for k in range(r + 1, t):
                mean1 = (cs[k] - cs[r]) / (k - r)
                mean2 = (cs[t] - cs[k]) / (t - k)
                val = math.sqrt((k - r) * (t - k) / (t - r)) * abs(mean1 - mean2)
                if val > best:
                    best = val
            C[t] = best

            gamma[t] = sigma * math.sqrt(
                6 * math.log(t - r)
                + 2 * math.log(1 / alpha)
                + 2 * math.log(math.pi**2 / 3)
            )

            if C[t] >= gamma[t]:
                alarms.append(t)
                r = t  # restart

        else:
            C[t] = 0.0

    return {
        "hatmu": hatmu[1:],
        "C": C[1:],
        "gamma": gamma[1:],
        "alarms": alarms
    }

# -----------------------------
# Make the requested instance
# -----------------------------
T = 600
### No conatmination
taus = [1, 120, 240, 245, 380, 480, 601]  # tau0=1, ..., tau6=T+1
mus  = [2.0, 3.0, 1.8, 1.3, 1.1, 2.5]   # mu0..mu5 (mu2 slightly lower to widen gap to mu1)
sigma = 1.0
alpha = 0.05
seed = 1
out_dir = "figures_illustration"
os.makedirs(out_dir, exist_ok=True)

X, mu_t = simulate_env(T, taus, mus, sigma=sigma, seed=seed)
res = atc_run(X, sigma=sigma, alpha=alpha)

hatmu = res["hatmu"]
C = res["C"]
gamma = res["gamma"]
alarms = res["alarms"]

t = np.arange(1, T + 1)

print("Change points:", taus[1:-1])
print("Alarms:", alarms)

# -----------------------------
# Plot both panels in one figure (shared x-axis)
# -----------------------------
tau_list = taus[1:-1]

fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)

ax = axes[0]
ax.scatter(t, X, s=8, alpha=0.4, label=r"observations $X_t$")
ax.step(t, mu_t, where="post", linewidth=2.0, label=r"true mean $\mu_t$")
ax.plot(t, hatmu, linewidth=2.0, label=r"estimate $\hat\mu_t$")

for tau in tau_list:
    ax.axvline(
        tau,
        linestyle=":",
        linewidth=1.4,
        color="green",
        label=None,
    )

ax.set_xlim(1, T)
ax.set_ylim(bottom=0)
ax.set_ylabel("")
ax.grid(True, alpha=0.25)
ax.legend(loc="upper left", ncols=1, frameon=True, fontsize=18)

ax = axes[1]
ax.plot(t, C, linewidth=2.0, label=r"ATC statistic $C_t^r$")
ax.plot(t, gamma, linewidth=2.0, label=r"threshold $\gamma_t^r$")

for a in alarms:
    ax.plot(a, C[a - 1], marker="o", linestyle="None", label=None)

first_tau = True
for tau in tau_list:
    ax.axvline(
        tau,
        linestyle=":",
        linewidth=1.4,
        color="green",
        label="change points" if first_tau else None,
    )
    first_tau = False

first_alarm = True
for a in alarms:
    ax.axvline(a, linestyle="-", linewidth=1.2, color="black",
               label="alarms" if first_alarm else None)
    first_alarm = False

ax.set_xlim(1, T)
ax.set_ylabel("")
ax.grid(True, alpha=0.25)

h1, l1 = ax.get_legend_handles_labels()
if h1:
    ax.legend(h1, l1, loc="lower right", ncols=1, frameon=True, fontsize=16)

fig.tight_layout()
fig.savefig(os.path.join(out_dir, "atc_illustration.png"), dpi=240)
plt.close(fig)

print(f"Wrote images to: {out_dir}/")
