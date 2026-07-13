import numpy as np

# Verify MG sampler empirically
np.random.seed(42)
n_samples = 100000

# Parameters
sigma_calib = 19.83505759494502
Delta_calib = 2.0
epsilon = 0.3242659560307008
K = 10
noise_scale = 0.035

sigma_eff = sigma_calib * noise_scale
Delta_eff = Delta_calib * noise_scale

print(f"sigma_eff={sigma_eff:.6f}, Delta_eff={Delta_eff:.6f}")

# MG sampler
ks = np.arange(-K, K + 1)
w = np.exp(-np.abs(ks) * epsilon)
w /= w.sum()
centers = ks * Delta_eff

components = np.random.choice(len(ks), size=n_samples, p=w)
samples_mg = np.random.normal(centers[components], sigma_eff)

# AG sampler (for comparison)
samples_ag = np.random.normal(0, sigma_eff, n_samples)

# Compute stats
l1_mg = np.mean(np.abs(samples_mg))
l1_ag = np.mean(np.abs(samples_ag))
l2_mg = np.sqrt(np.mean(samples_mg**2))
l2_ag = np.sqrt(np.mean(samples_ag**2))

print(f"AG: L1={l1_ag:.6f}, L2={l2_ag:.6f}")
print(f"MG: L1={l1_mg:.6f}, L2={l2_mg:.6f}")
print(f"MG/AG ratio: L1={l1_mg/l1_ag:.6f}, L2={l2_mg/l2_ag:.6f}")

# Theoretical values
# AG: L1 = sigma * sqrt(2/pi)
l1_ag_theory = sigma_eff * np.sqrt(2/np.pi)
print(f"\nAG theory: L1={l1_ag_theory:.6f}")
print(f"AG actual matches theory: {abs(l1_ag - l1_ag_theory) < 3*l1_ag_theory/np.sqrt(n_samples)}")

# MG: compute theoretical L1 from Julia calibration ratio
# Julia: MG L1/AG L1 for calibrated values = 16.827/16.840 = 0.99922
# So scaled MG L1 = scaled AG L1 * 0.99922 ≈ same
l1_mg_expected = l1_ag_theory * 0.99922
print(f"MG expected L1 (from Julia ratio): {l1_mg_expected:.6f}")
print(f"MG actual vs expected: ratio = {l1_mg/l1_mg_expected:.6f}")
