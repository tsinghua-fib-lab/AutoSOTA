import numpy as np
from scipy.stats import ks_2samp

# Load data
data = np.load("data/rough_volatility/simple_rbergomi_data.npz")
log_price = data["log_price"]  # (15000, 129)
print(f"Data shape: {log_price.shape}")
print(f"Range: [{log_price.min():.4f}, {log_price.max():.4f}]")

# Get test split (last 10%)
test_start = int(0.9 * len(log_price))
test_data = log_price[test_start:]
print(f"Test split: {test_data.shape[0]} paths, {test_data.shape[1]} steps")

# KS on test data at t=128 (last timestep)
test_terminal = test_data[:, 128]
print(f"Test terminal stats: mean={test_terminal.mean():.4f}, std={test_terminal.std():.4f}")

# Compare with train terminal
train_terminal = log_price[:test_start, 128]
print(f"Train terminal stats: mean={train_terminal.mean():.4f}, std={train_terminal.std():.4f}")
ks_stat, _ = ks_2samp(train_terminal, test_terminal)
print(f"KS(train_terminal, test_terminal) at t=128: {ks_stat:.6f} (x100 = {ks_stat*100:.2f})")
