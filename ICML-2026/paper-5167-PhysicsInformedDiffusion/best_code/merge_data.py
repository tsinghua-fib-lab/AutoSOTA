import numpy as np
import os
import scipy.io as sio
from scipy.fft import dstn


# =============================================================
# Border padding
# =============================================================
def add_soft_zero_border(a, n_layers=1):
    result = a.copy()
    for k in range(1, n_layers + 1):
        alpha = k / (n_layers + 1)
        top_row    = (1 - alpha) * result[0, :]
        bottom_row = (1 - alpha) * result[-1, :]
        result = np.vstack([top_row[None, :], result, bottom_row[None, :]])
        left_col  = (1 - alpha) * result[:, 0]
        right_col = (1 - alpha) * result[:, -1]
        result = np.hstack([left_col[:, None], result, right_col[:, None]])
    return result


def add_soft_mean_border(a, n_layers=1):
    result = a.copy()
    
    for k in range(1, n_layers + 1):
        H, W = result.shape  # aggiornati ad ogni layer
    
        # Compute row and column means from current state
        row_mean = (result[0, :] + result[-1, :]) / 2
        col_mean = (result[:, 0] + result[:, -1]) / 2
    
        alpha = k / (n_layers + 1)
    
        # Top and bottom rows
        top_row = (1 - alpha) * result[0, :] + alpha * row_mean
        bottom_row = (1 - alpha) * result[-1, :] + alpha * row_mean
    
        result = np.vstack([
            top_row[np.newaxis, :],
            result,
            bottom_row[np.newaxis, :]
        ])
    
        # Update column mean after adding rows
        col_mean_ext = (result[:, 0] + result[:, -1]) / 2
    
        # Left and right columns
        left_col = (1 - alpha) * result[:, 0] + alpha * col_mean_ext
        right_col = (1 - alpha) * result[:, -1] + alpha * col_mean_ext
    
        result = np.hstack([
            left_col[:, np.newaxis],
            result,
            right_col[:, np.newaxis]
        ])
    
    return result

# =============================================================
# Strategy presets
# =============================================================
# mode: 'hyperbolic' (low-freq block + axis strips) or 'square' (just top-left crop)
# For 'hyperbolic': keep == base + sum(strips). 'strips' is ignored for 'square'.
STRATEGY_PRESETS = {
    "hyperbolic_32": dict(mode="hyperbolic", keep=32, base=24, strips=[6, 2],  pad_a=4, pad_u=0),
    "hyperbolic_44": dict(mode="hyperbolic", keep=44, base=38, strips=[4, 2],  pad_a=4, pad_u=0),
    "hyperbolic_64": dict(mode="hyperbolic", keep=64, base=54, strips=[10, ], pad_a=4, pad_u=0),
    "square_38":     dict(mode="square",     keep=38, pad_a=4, pad_u=0),
    "square_44":     dict(mode="square",     keep=44, pad_a=4, pad_u=0),
}


def get_strategy(name):
    if name not in STRATEGY_PRESETS:
        raise ValueError(f"Unknown strategy '{name}'. Available: {list(STRATEGY_PRESETS)}")
    p = STRATEGY_PRESETS[name]
    if p["mode"] == "hyperbolic":
        assert p["keep"] == p["base"] + sum(p["strips"]), (
            f"Invalid preset '{name}': keep != base + sum(strips)"
        )
    return p


# =============================================================
# Spectral packing (full DST grid -> keep x keep buffer)
# =============================================================
def pack_spectrum(spec, params):
    """Pack a 2D DST spectrum into the keep x keep buffer defined by params."""
    mode = params["mode"]
    keep = params["keep"]
    out = np.zeros((keep, keep), dtype=spec.dtype)

    if mode == "square":
        out[:, :] = spec[:keep, :keep]
        return out

    # hyperbolic
    base   = params["base"]
    strips = params["strips"]
    size   = spec.shape[0]

    out[:base, :base] = spec[:base, :base]

    off = base
    for i, n in enumerate(strips):
        band_start = (i + 1) * base
        band_end   = min((i + 2) * base, size)
        band_len   = max(band_end - band_start, 0)
        if band_len > 0:
            for f in range(n):
                out[:band_len, off + f] = spec[f, band_start:band_end]
                out[off + f, :band_len] = spec[band_start:band_end, f]
        off += n
    return out


def compute_components(f1, f2, params):
    """Return stacked (2, keep, keep) array: [a_hat_packed, u_hat_packed]."""
    f1_pad = add_soft_zero_border(f1, params["pad_a"])
    f2_pad = add_soft_zero_border(f2, params["pad_u"])

    a_hat = dstn(f1_pad[1:-1, 1:-1], type=1, norm="ortho")
    u_hat = dstn(f2_pad[1:-1, 1:-1], type=1, norm="ortho")

    return np.stack([pack_spectrum(a_hat, params),
                     pack_spectrum(u_hat, params)], axis=0)


# =============================================================
# Configuration
# =============================================================
NAME             = "poisson"          # 'poisson' or 'helmholtz'
STRATEGY_NAME    = "hyperbolic_44"    # any key from STRATEGY_PRESETS
N_FILES          = 5
SAMPLES_PER_FILE = 10000

params  = get_strategy(STRATEGY_NAME)
keep    = params["keep"]
in_dim  = 2   # (a_hat, u_hat)

# Output folders
data_dir  = f"data/{NAME}-merged-spectral/{STRATEGY_NAME}"
stats_dir = "processed"   # single flat folder for all stats
os.makedirs(data_dir,  exist_ok=True)
os.makedirs(stats_dir, exist_ok=True)
output_path_tmpl = os.path.join(data_dir, "merge_new_{}.npy")

# File-name prefix for stats: e.g. "poisson_hyperbolic_44"
stats_prefix = f"{NAME}_{STRATEGY_NAME}"

print(f"Strategy: {STRATEGY_NAME}")
print(f"  data  -> {data_dir}")
print(f"  stats -> {stats_dir}/{stats_prefix}_{{mean,std,min,max}}.npy")

if NAME == "poisson":
    key_a, key_u = "f_data", "phi_data"
elif NAME == "helmholtz":
    key_a, key_u = "f_data", "psi_data"
else:
    raise ValueError(NAME)


# =============================================================
# First pass: global stats + per-file partial stats
# =============================================================
print("\nFirst pass: computing global statistics")

sum_vals   = np.zeros((in_dim, keep, keep))
sum_sqs    = np.zeros((in_dim, keep, keep))
global_min = np.full((in_dim, keep, keep),  np.inf)
global_max = np.full((in_dim, keep, keep), -np.inf)
count = 0

for j in range(1, N_FILES + 1):
    print(f"\nProcessing file {j} (first pass)...")
    raw = sio.loadmat(f"data/training/{NAME}/{NAME}_{j}.mat")
    a, u = raw[key_a], raw[key_u]

    sum_vals_j = np.zeros((in_dim, keep, keep))
    sum_sqs_j  = np.zeros((in_dim, keep, keep))
    min_vals_j = np.full((in_dim, keep, keep),  np.inf)
    max_vals_j = np.full((in_dim, keep, keep), -np.inf)

    for i in range(SAMPLES_PER_FILE):
        comp = compute_components(a[i], u[i], params)

        # per-file accumulators
        sum_vals_j += comp
        sum_sqs_j  += comp ** 2
        min_vals_j  = np.minimum(min_vals_j, comp)
        max_vals_j  = np.maximum(max_vals_j, comp)

        # global accumulators
        sum_vals   += comp
        sum_sqs    += comp ** 2
        global_min  = np.minimum(global_min, comp)
        global_max  = np.maximum(global_max, comp)
        count += 1

    mean_j = sum_vals_j / SAMPLES_PER_FILE
    std_j  = np.sqrt(np.maximum(sum_sqs_j / SAMPLES_PER_FILE - mean_j ** 2, 0.0))

    print(f"File {j} stats:")
    print(f"  Mean: {mean_j}")
    print(f"  Std : {std_j}")
    print(f"  Min : {min_vals_j}")
    print(f"  Max : {max_vals_j}")

# Global stats
mean = sum_vals / count
std  = np.sqrt(np.maximum(sum_sqs / count - mean ** 2, 0.0))

print("\n=== Global Stats ===")
print("Global mean:", mean)
print("Global std :", std)
print("Global min :", global_min)
print("Global max :", global_max)

np.save(os.path.join(stats_dir, f"{stats_prefix}_mean.npy"), mean)
np.save(os.path.join(stats_dir, f"{stats_prefix}_std.npy"),  std)
np.save(os.path.join(stats_dir, f"{stats_prefix}_min.npy"),  global_min)
np.save(os.path.join(stats_dir, f"{stats_prefix}_max.npy"),  global_max)


# =============================================================
# Second pass: normalize & save each sample
# =============================================================
print("\nSecond pass: saving normalized samples")

mean = np.load(os.path.join(stats_dir, f"{stats_prefix}_mean.npy"))
std  = np.load(os.path.join(stats_dir, f"{stats_prefix}_std.npy"))

for j in range(1, N_FILES + 1):
    print(f"\nProcessing file {j}...")
    raw = sio.loadmat(f"data/training/{NAME}/{NAME}_{j}.mat")
    a, u = raw[key_a], raw[key_u]

    mean_acc = np.zeros(in_dim)
    std_acc  = np.zeros(in_dim)
    min_vals = np.full(in_dim,  np.inf)
    max_vals = np.full(in_dim, -np.inf)

    for i in range(SAMPLES_PER_FILE):
        comp       = compute_components(a[i], u[i], params)
        normalized = (comp - mean) / (std + 1e-8)
        combined   = np.moveaxis(normalized, 0, -1)   # (keep, keep, 2)

        out_path = output_path_tmpl.format(i + (j - 1) * SAMPLES_PER_FILE)
        np.save(out_path, combined)

        flat = combined.reshape(-1, in_dim)
        mean_acc += flat.mean(axis=0)
        std_acc  += flat.std(axis=0)
        min_vals  = np.minimum(min_vals, flat.min(axis=0))
        max_vals  = np.maximum(max_vals, flat.max(axis=0))

        if i % 500 == 0:
            print(f"  saved sample {i} -> {out_path}")

    print(f"Normalized file {j} stats:")
    print(f"  Mean: {mean_acc / SAMPLES_PER_FILE}")
    print(f"  Std : {std_acc  / SAMPLES_PER_FILE}")
    print(f"  Min : {min_vals}")
    print(f"  Max : {max_vals}")

print("\nFinished processing all files.")
