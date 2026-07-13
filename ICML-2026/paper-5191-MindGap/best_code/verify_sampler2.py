import numpy as np

np.random.seed(42)
n = 200000
noise_scale = 0.035
eps = 0.3242659560307008

# Calibrated sigmas
sig_ag_cal = 21.105645147662777
sig_mg_cal = 19.83505759494502
Delta_cal = 2.0
K = 10

# Scaled
sig_ag = sig_ag_cal * noise_scale
sig_mg = sig_mg_cal * noise_scale
Delta_scaled = Delta_cal * noise_scale

print(f"AG sigma_eff = {sig_ag:.6f}")
print(f"MG sigma_eff = {sig_mg:.6f}")
print(f"Delta_eff = {Delta_scaled:.6f}")

# MG sampler
ks = np.arange(-K, K+1)
w = np.exp(-np.abs(ks)*eps); w /= w.sum()
centers = ks * Delta_scaled
comp = np.random.choice(len(ks), size=n, p=w)
mg = np.random.normal(centers[comp], sig_mg)
ag = np.random.normal(0, sig_ag, n)

print(f"\nAG: L1={np.mean(np.abs(ag)):.6f}, L2={np.sqrt(np.mean(ag**2)):.6f}")
print(f"MG: L1={np.mean(np.abs(mg)):.6f}, L2={np.sqrt(np.mean(mg**2)):.6f}")
print(f"MG/AG: L1={np.mean(np.abs(mg))/np.mean(np.abs(ag)):.6f}, L2={np.sqrt(np.mean(mg**2))/np.sqrt(np.mean(ag**2)):.6f}")

# Expected from Julia calibration ratio
# Julia: MG L1 / AG L1 = 16.827 / 16.840 = 0.9992
print(f"\nExpected MG/AG L1 ratio: {16.827/16.840:.6f}")
print(f"Actual MG/AG L1 ratio: {np.mean(np.abs(mg))/np.mean(np.abs(ag)):.6f}")
