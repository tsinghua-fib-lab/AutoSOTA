import os, torch, numpy as np, pandas as pd

# Load training data
train_data = torch.load('/autosota_cache/tmp/darcy_64/darcy_train_64.pt', map_location='cpu')
test_data = torch.load('/autosota_cache/tmp/darcy_64/darcy_test_64.pt', map_location='cpu')

# Zenodo format: x=permeability K, y=pressure p
# Our format: p_data.csv (pressure), K_data.csv (permeability)

train_K = train_data['x']  # (N, 64, 64)
train_p = train_data['y']  # (N, 64, 64)
test_K = test_data['x']
test_p = test_data['y']

print(f"Train: K={train_K.shape}, p={train_p.shape}")
print(f"Test: K={test_K.shape}, p={test_p.shape}")

# Flatten to (N, 4096)
train_K_flat = train_K.reshape(train_K.shape[0], -1).numpy()
train_p_flat = train_p.reshape(train_p.shape[0], -1).numpy()
test_K_flat = test_K.reshape(test_K.shape[0], -1).numpy()
test_p_flat = test_p.reshape(test_p.shape[0], -1).numpy()

# Split test into valid/test (or use all as train/valid)
# Use 4000 train, 1000 valid from training set
# And use test set as additional validation
n_train = 4000

os.makedirs('/repo/data/darcy/train', exist_ok=True)
os.makedirs('/repo/data/darcy/valid', exist_ok=True)

# Save train (from training set)
pd.DataFrame(train_p_flat[:n_train]).to_csv('/repo/data/darcy/train/p_data.csv', index=False, header=False)
pd.DataFrame(train_K_flat[:n_train]).to_csv('/repo/data/darcy/train/K_data.csv', index=False, header=False)

# Save valid (rest of training set + test set)
valid_p = np.concatenate([train_p_flat[n_train:], test_p_flat], axis=0)
valid_K = np.concatenate([train_K_flat[n_train:], test_K_flat], axis=0)
pd.DataFrame(valid_p).to_csv('/repo/data/darcy/valid/p_data.csv', index=False, header=False)
pd.DataFrame(valid_K).to_csv('/repo/data/darcy/valid/K_data.csv', index=False, header=False)

print(f"Train: {n_train} samples")
print(f"Valid: {valid_p.shape[0]} samples")
print("Data saved successfully!")
