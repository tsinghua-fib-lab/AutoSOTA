from sklearn.datasets import fetch_openml
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import jax
import jax.numpy as jnp

def load_boston(batch_size, test_size=0.2, seed=42, standardize_X=True, standardize_y=False):
    """
    Returns JAX arrays and (optionally) standardization stats.
    Shapes:
      X_*: (N, 13), y_*: (N, 1)  # regression target
    """
    # Load from OpenML (since sklearn.load_boston is deprecated)
    ds = fetch_openml(name="boston", version=1, as_frame=True)
    X = ds.data.to_numpy(dtype=np.float32)
    y = ds.target.to_numpy().astype(np.float32).reshape(-1, 1)

    # Train/test split
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )

    # Standardize using train stats
    x_mean = X_tr.mean(axis=0, keepdims=True)
    x_std  = X_tr.std(axis=0, keepdims=True) + 1e-8
    if standardize_X:
        X_tr = (X_tr - x_mean) / x_std
        X_te = (X_te - x_mean) / x_std

    y_min = y_tr.min(axis=0, keepdims=True)
    y_max = y_tr.max(axis=0, keepdims=True)
    if standardize_y:
        y_tr = 2 * (y_tr - y_min) / (y_max - y_min + 1e-8) - 1
        y_te = 2 * (y_te - y_min) / (y_max - y_min + 1e-8) - 1

    # Shuffle the training data before batching
    key = jax.random.PRNGKey(seed)
    perm = jax.random.permutation(key, X_tr.shape[0])
    X_tr = X_tr[perm]
    y_tr = y_tr[perm]

    # Batch them
    N_tr = y_tr.shape[0]
    num_batches_tr = (N_tr + batch_size - 1) // batch_size
    pad = num_batches_tr * batch_size - N_tr
    if pad > 0:
        y_tr = jnp.pad(y_tr, ((0, pad), (0,0)), mode='edge')
        X_tr = jnp.pad(X_tr, ((0, pad), (0,0)), mode='edge')
    y_tr = y_tr.reshape(num_batches_tr, batch_size, *y_tr.shape[1:])
    X_tr = X_tr.reshape(num_batches_tr, batch_size, *X_tr.shape[1:])

    N_te = y_te.shape[0]
    num_batches_te = (N_te + batch_size - 1) // batch_size
    pad = num_batches_te * batch_size - N_te
    if pad > 0:
        y_te = jnp.pad(y_te, ((0, pad), (0,0)), mode='edge')
        X_te = jnp.pad(X_te, ((0, pad), (0,0)), mode='edge')
    y_te = y_te.reshape(num_batches_te, batch_size, *y_te.shape[1:])
    X_te = X_te.reshape(num_batches_te, batch_size, *X_te.shape[1:])

    # Convert to JAX arrays (targets flattened to shape (N,))
    out = {
        "Z": jnp.asarray(X_tr),
        "y": jnp.asarray(y_tr),
        "Z_test":  jnp.asarray(X_te),
        "y_test":  jnp.asarray(y_te),
        "z_stats": (jnp.asarray(x_mean.squeeze(0)), jnp.asarray(x_std.squeeze(0))),
        "y_stats": (jnp.asarray(y_min.item()),     jnp.asarray(y_max.item())),
        "batch_size": batch_size,
        "num_batches_tr": num_batches_tr,
        "num_batches_te": num_batches_te,
    }
    return out

def load_covertype(batch_size, test_size=0.2, seed=42,
                   standardize_X=True, one_hot_y=False):
    """
    Returns batched JAX arrays for UCI Covertype classification.
    Shapes:
      X: (num_batches, batch_size, 54)
      y: (num_batches, batch_size,) or one-hot
      X_test: (N_test, 54)
      y_test: same
    """
    
    # ---- Load dataset ----
    ds = fetch_covtype(as_frame=False)
    X = ds.data.astype(np.float32)
    y = (ds.target - 1).astype(np.int32)  # convert to {0,...,6}

    # ---- Train/test split ----
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    # ---- Standardization ----
    if standardize_X:
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr).astype(np.float32)
        X_te = scaler.transform(X_te).astype(np.float32)
        x_mean = scaler.mean_.astype(np.float32)
        x_std  = scaler.scale_.astype(np.float32)
    else:
        x_mean = X_tr.mean(axis=0).astype(np.float32)
        x_std  = X_tr.std(axis=0).astype(np.float32) + 1e-8

    # ---- One-hot encoding (optional) ----
    if one_hot_y:
        num_classes = 7
        y_tr_oh = np.eye(num_classes)[y_tr]
        y_te_oh = np.eye(num_classes)[y_te]

        y_tr = y_tr_oh.astype(np.float32)
        y_te = y_te_oh.astype(np.float32)

    # ---- Shuffle training set ----
    key = jax.random.PRNGKey(seed)
    perm = jax.random.permutation(key, X_tr.shape[0])
    X_tr = X_tr[perm]
    y_tr = y_tr[perm]

    # ---- Batch + pad ----
    N_train = X_tr.shape[0]
    N_test = X_te.shape[0]
    num_batches_tr = (N_train + batch_size - 1) // batch_size
    num_batches_te = (N_test + batch_size - 1) // batch_size
    pad_tr = num_batches_tr * batch_size - N_train
    pad_te = num_batches_te * batch_size - N_test
    if pad_tr > 0:
        X_tr = jnp.pad(X_tr, ((0, pad_tr), (0, 0)), mode='edge')
        if one_hot_y:
            y_tr = jnp.pad(y_tr, ((0, pad_tr), (0, 0)), mode='edge')
        else:
            y_tr = jnp.pad(y_tr, ((0, pad_tr),), mode='edge')

    X_tr = X_tr.reshape(num_batches_tr, batch_size, -1)
    if one_hot_y:
        y_tr = y_tr.reshape(num_batches_tr, batch_size, -1)
    else:
        y_tr = y_tr.reshape(num_batches_tr, batch_size)

    if pad_te > 0:
        X_te = jnp.pad(X_te, ((0, pad_te), (0, 0)), mode='edge')
        if one_hot_y:
            y_te = jnp.pad(y_te, ((0, pad_te), (0, 0)), mode='edge')
        else:
            y_te = jnp.pad(y_te, ((0, pad_te),), mode='edge')
    X_te = X_te.reshape(num_batches_te, batch_size, -1)
    if one_hot_y:
        y_te = y_te.reshape(num_batches_te, batch_size, -1)
    else:
        y_te = y_te.reshape(num_batches_te, batch_size)

    # ---- Convert to JAX arrays ----
    out = {
        "Z": jnp.asarray(X_tr),
        "y": jnp.asarray(y_tr),
        "Z_test": jnp.asarray(X_te),
        "y_test": jnp.asarray(y_te),
        "z_stats": (jnp.asarray(x_mean), jnp.asarray(x_std)),
        "num_classes": 7,
        "batch_size": batch_size,
        "num_batches_tr": num_batches_tr,
        "num_batches_te": num_batches_te,
    }

    return out


def make_teacher_fn(q1_nn_apply):
    """
    Returns a function teacher_fn(z) that maps
    z: (..., d) -> y: (..., 1)
    """
    teacher_params = sample_teacher_params_multimodal_per_neuron(
        key_params,
        d_input,
        M,
        num_modes=(M // 10)+1,
        mode_separation=2.0,
        within_mode_std=0.2,
        base_scale=0.8,
    )
    def teacher_fn(z):
        # z can be shape (d,) or (N, d)
        contribs = jax.vmap(
            lambda p: q1_nn_apply(z, p),
            in_axes=0
        )(teacher_params)
        return contribs.mean(axis=0, keepdims=True)

    return teacher_fn


def load_student_teacher(batch_size, train_size, q1_nn_apply, d, M, standardize_Z=True, standardize_y=False):
    """
    Synthetic student-teacher regression dataset, batched like your load_boston().

    Returns dict with:
      Z, y:       (num_batches_tr, batch_size, d_in), (num_batches_tr, batch_size, 1)
      Z_test,y_test similarly
      z_stats: (mean, std) computed from train split (like your code)
      y_stats: (min, max) from train split (for optional [-1,1] scaling)

    Teacher:
      z ~ N(0, I)
      y = q1_nn_apply(z, (W1,b1,W2)) + noise,  noise ~ N(0, noise_std^2)
    """
    # RNG
    key = jax.random.PRNGKey(42)
    key_z, key_perm, key_noise, key_params = jax.random.split(key, 4)

    # Sample inputs
    d_input = d
    total_size = train_size + 1024
    Z = jax.random.normal(key_z, shape=(total_size, d_input), dtype=jnp.float32) * 0.01
    # Z = Z / jnp.linalg.norm(Z, axis=1, keepdims=True)

    teacher_params = sample_teacher_params_multimodal_per_neuron(
        key_params,
        d_input,
        M,
        num_modes=(M // 10)+1,
        mode_separation=2.0,
        within_mode_std=0.2,
        base_scale=0.8,
    )
    
    # Generate targets
    y_clean = jax.vmap(jax.vmap(q1_nn_apply, in_axes=(0, None)),
                    in_axes=(None, 0)
                )(Z, teacher_params).mean(0)
    eps = 0.0 * jax.random.normal(key_noise, shape=y_clean.shape, dtype=jnp.float32)
    y = y_clean + eps

    # Train/test split (numpy-style, deterministic)
    N_tr = train_size
    N_te = total_size - N_tr
    perm = jax.random.permutation(key_perm, total_size)
    Z = Z[perm]
    y = y[perm]
    Z_tr, Z_te = Z[:N_tr], Z[N_tr:]
    y_tr, y_te = y[:N_tr], y[N_tr:]

    # Standardize using train stats
    z_mean = jnp.mean(Z_tr, axis=0, keepdims=True)
    z_std = jnp.std(Z_tr, axis=0, keepdims=True) + 1e-8
    if standardize_Z:
        Z_tr = (Z_tr - z_mean) / z_std
        Z_te = (Z_te - z_mean) / z_std

    y_min = jnp.min(y_tr, axis=0, keepdims=True)
    y_max = jnp.max(y_tr, axis=0, keepdims=True)
    if standardize_y:
        y_tr = 2 * (y_tr - y_min) / (y_max - y_min + 1e-8) - 1
        y_te = 2 * (y_te - y_min) / (y_max - y_min + 1e-8) - 1

    # Batch + pad (edge) like your load_boston
    def batchify(Z_, y_):
        N_ = y_.shape[0]
        num_batches = (N_ + batch_size - 1) // batch_size
        pad = num_batches * batch_size - N_
        if pad > 0:
            Z_ = jnp.pad(Z_, ((0, pad), (0, 0)), mode="edge")
            y_ = jnp.pad(y_, ((0, pad), (0, 0)), mode="edge")
        Z_ = Z_.reshape(num_batches, batch_size, Z_.shape[-1])
        y_ = y_.reshape(num_batches, batch_size, 1)
        return Z_, y_, num_batches

    Z_tr_b, y_tr_b, num_batches_tr = batchify(Z_tr, y_tr)
    Z_te_b, y_te_b, num_batches_te = batchify(Z_te, y_te)

    out = {
        "Z": Z_tr_b,
        "y": y_tr_b,
        "Z_test": Z_te_b,
        "y_test": y_te_b,
        "z_stats": (jnp.asarray(z_mean.squeeze(0)), jnp.asarray(z_std.squeeze(0))),
        "y_stats": (jnp.asarray(y_min.item()), jnp.asarray(y_max.item())),
        "batch_size": batch_size,
        "num_batches_tr": num_batches_tr,
        "num_batches_te": num_batches_te,
        "d_in": d_input,
        "noise_std": 0.3,
        "teacher_params": teacher_params,
    }

    return out


def sample_teacher_params_multimodal_per_neuron(
    key,
    d_input: int,
    M: int,
    num_modes: int = 8,
    mode_separation: float = 2.0,
    within_mode_std: float = 0.2,
    base_scale: float = 0.8,
):
    key_centers, key_assign, key_eps = jax.random.split(key, 3)
    Drow = d_input + 2

    centers = mode_separation * jax.random.normal(key_centers, (num_modes, Drow), dtype=jnp.float32)
    # assign a mode to each hidden unit
    idx = jax.random.randint(key_assign, (M,), 0, num_modes)
    mu_rows = centers[idx]  # (M, Drow)

    eps = within_mode_std * jax.random.normal(key_eps, (M, Drow), dtype=jnp.float32)
    teacher_params = mu_rows + eps  # (M, d_input+2)

    # scale W1/W2 similarly
    W1 = teacher_params[:, :d_input] * (base_scale)
    b1 = teacher_params[:, d_input:d_input+1]
    W2 = teacher_params[:, d_input+1:d_input+2] * (base_scale)
    return jnp.concatenate([W1, b1, W2], axis=1)
