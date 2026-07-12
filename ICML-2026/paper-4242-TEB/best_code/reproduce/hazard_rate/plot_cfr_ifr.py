"""
Plot CFR vs IFR hazard rate comparison for §3 (CFR_IFR.pdf in the paper).

Constant Failure Rate (CFR) = geometric distribution of output lengths
  → hazard rate h(t) = constant (memorylessness)
Increasing Failure Rate (IFR) = e.g., gamma with shape > 1
  → hazard rate h(t) is monotonically increasing in t

This script renders three panels:
  (left)   PDFs of CFR (geometric) and IFR (gamma) with matched means
  (middle) Survival functions S(t) = P(X > t)
  (right)  Hazard rates h(t) = f(t) / S(t)

Optionally accepts a JSON of empirical output-length samples (e.g., from
ShareGPT) and overlays the empirical hazard rate.

Inputs (optional):
  --empirical-json  path to a JSON file like {"output_lengths": [...]}

Output:
  CFR_IFR.pdf
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def empirical_hazard(samples: np.ndarray, num_bins: int = 80):
    samples = samples[samples <= np.percentile(samples, 99)]
    counts, edges = np.histogram(samples, bins=num_bins, density=True)
    centers = (edges[:-1] + edges[1:]) / 2
    width = edges[1] - edges[0]
    pdf = counts
    cdf = np.cumsum(pdf) * width
    survival = np.clip(1 - cdf, 1e-10, None)
    hazard = pdf / survival
    return centers, pdf, survival, hazard


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mean", type=float, default=512.0, help="Matched mean of both distributions")
    p.add_argument("--gamma-shape", type=float, default=4.0, help="IFR gamma shape parameter (>1)")
    p.add_argument("--empirical-json", default=None, help="Optional empirical samples JSON")
    p.add_argument("--output", default="CFR_IFR.pdf")
    args = p.parse_args()

    mu = args.mean
    # Geometric (CFR) parametrised so E[X] = mu (treat as continuous exponential for plotting)
    rate_cfr = 1.0 / mu
    # Gamma (IFR): shape k, scale theta=mu/k → mean = mu
    k = args.gamma_shape
    theta = mu / k

    t = np.linspace(1, 3 * mu, 800)
    pdf_cfr = rate_cfr * np.exp(-rate_cfr * t)
    surv_cfr = np.exp(-rate_cfr * t)
    haz_cfr = pdf_cfr / np.clip(surv_cfr, 1e-12, None)  # constant = rate_cfr

    # Gamma PDF: t^(k-1) e^{-t/theta} / (Gamma(k) theta^k)
    from math import gamma as gammaf
    pdf_ifr = (t ** (k - 1)) * np.exp(-t / theta) / (gammaf(k) * theta ** k)
    # Survival via 1 - regularised lower incomplete gamma — approximate by numerical CDF
    cdf_ifr = np.cumsum(pdf_ifr) * (t[1] - t[0])
    cdf_ifr = np.clip(cdf_ifr, 0, 1)
    surv_ifr = np.clip(1 - cdf_ifr, 1e-12, None)
    haz_ifr = pdf_ifr / surv_ifr

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))

    axes[0].plot(t, pdf_cfr, "C0-", label=f"CFR (geometric, μ={mu:.0f})")
    axes[0].plot(t, pdf_ifr, "C3--", label=f"IFR (gamma k={k}, μ={mu:.0f})")
    axes[0].set_xlabel("Output length t")
    axes[0].set_ylabel("PDF f(t)")
    axes[0].set_title("Output length distribution")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, surv_cfr, "C0-", label="CFR")
    axes[1].plot(t, surv_ifr, "C3--", label="IFR")
    axes[1].set_xlabel("Output length t")
    axes[1].set_ylabel("S(t) = P(X > t)")
    axes[1].set_title("Survival function")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(t, haz_cfr, "C0-", label="CFR (constant)")
    axes[2].plot(t, haz_ifr, "C3--", label="IFR (increasing)")
    if args.empirical_json:
        path = Path(args.empirical_json)
        if path.exists():
            data = json.load(open(path))
            samples = np.array(data.get("output_lengths", data))
            ec, _epdf, _esurv, ehaz = empirical_hazard(samples)
            axes[2].plot(ec, ehaz, "k.", markersize=3, label="Empirical")
    axes[2].set_xlabel("Output length t")
    axes[2].set_ylabel("h(t) = f(t) / S(t)")
    axes[2].set_title("Hazard rate")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    fig.suptitle("CFR vs IFR — distributional assumptions for output length",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(args.output, bbox_inches="tight")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
