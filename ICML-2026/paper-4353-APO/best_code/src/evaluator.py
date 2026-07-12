import numpy as np
import torch
from src.utils import optimize_t_for_x_batch_torch  # (Core: importing the new Torch function)

def evaluate_metrics(model, dataset, test_x, t_grid_size: int = 101, eval_chunk_size: int = 256):
    """
    Parameters
    ----------
    model : GPModel
        Trained GP model wrapper.
    dataset : BaseSyntheticDataset
        Object providing ground truth functions like get_t_star, get_f.
    test_x : np.ndarray
        Test set covariates (N_test, dim_x).
    t_grid_size : int, default 101
        Number of discrete points in the t grid for evaluation.
    eval_chunk_size : int, default 256
        Chunk size for splitting test_x during evaluation to manage VRAM usage.
    """
    metrics = {}
    n_test = len(test_x)

    if n_test == 0:
        return {'E2_error': 0.0, 'policy_suboptimality': 0.0, 'dose_error': 0.0}

    # Get ground truth values
    t_star_true = dataset.get_t_star(test_x)
    f_at_t_star_true = dataset.get_f(test_x, t_star_true)

    # --- 1. Batch find t_hat*(x) = argmax mu(x,t) ---
    try:
        test_x_torch = torch.tensor(test_x, dtype=model.dtype, device=model.device)

        t_pred_chunks = []
        val_dummy_chunks = []

        for test_chunk in torch.split(test_x_torch, eval_chunk_size, dim=0):
            if test_chunk.shape[0] == 0:
                continue
            t_chunk, val_chunk = optimize_t_for_x_batch_torch(
                model,
                test_chunk,
                'mean',
                t_grid_size=t_grid_size
            )
            t_pred_chunks.append(t_chunk.cpu())
            val_dummy_chunks.append(val_chunk.cpu())

        if not t_pred_chunks:
            raise RuntimeError("evaluate_metrics: empty prediction chunks produced.")

        t_star_pred = torch.cat(t_pred_chunks, dim=0).numpy()

    except Exception as e:
        print(f"Warning: evaluate_metrics argmax 'mean' failed: {e}. Defaulting t_pred=0.5")
        t_star_pred = np.full_like(t_star_true, 0.5)

    # --- 2. Batch evaluate f_hat(x, t*(x)) ---
    try:
        x_t_true_batch = np.hstack([test_x, t_star_true.reshape(-1, 1)])
        f_pred_at_t_star_true, _ = model.predict(x_t_true_batch)
    except Exception as e:
        print(f"Warning: evaluate_metrics predict at t_true failed: {e}. Defaulting f_pred=0")
        f_pred_at_t_star_true = np.zeros_like(f_at_t_star_true)

    # --- 3. Value of the true function under the predicted policy ---
    f_true_at_t_star_pred = dataset.get_f(test_x, t_star_pred)

    # --- 4. Metric calculation ---
    metrics['E2_error'] = np.mean((f_pred_at_t_star_true - f_at_t_star_true) ** 2)

    v_star = np.mean(f_at_t_star_true)
    v_hat = np.mean(f_true_at_t_star_pred)
    metrics['policy_suboptimality'] = v_star - v_hat

    metrics['dose_error'] = np.mean(np.abs(t_star_pred - t_star_true))

    return metrics