import os
import itertools
import sys
import copy
import traceback
from datetime import datetime

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.cls_performance import cls_metrics
from utils.setups import build_model, setup_penalties, setup_importance, set_seed


MAX_EPOCHS = 2000
EARLY_STOPPING_PATIENCE = 50
MIN_VAL_IMPROVEMENT = 1e-5


def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg, worker=None):
    prefix = "[{}]".format(now_str())
    if worker is not None:
        prefix += " [Worker {}]".format(worker)
    print(prefix, msg)
    sys.stdout.flush()


def train_model(model, train_dataset, val_dataset, penalty, batch_size,
                ind_lambda, int_lambda, weight_decay, lr, penalty_type, cudnn_flag=True):
    """Fit one model configuration and restore the best validation checkpoint."""

    mse_loss = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_loader = DataLoader(
        train_dataset,
        batch_size=(len(train_dataset) if batch_size == -1 else batch_size)
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=(len(val_dataset) if batch_size == -1 else batch_size)
    )

    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0

    for epoch in range(MAX_EPOCHS):
        # ----------------- Training -----------------
        model.train()
        total_train_loss = 0.0

        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()

            # LSTM-style models need explicit cuDNN control for reproducibility.
            with torch.backends.cudnn.flags(enabled=cudnn_flag):
                y_pred = model(x_batch)
                pred_loss = mse_loss(y_pred, y_batch)

                if penalty_type.startswith('Fast_Shap'):
                    ind, inter = penalty(model, x_batch)
                    struct_loss = ind_lambda * ind + int_lambda * inter
                elif penalty_type == 'Shapley':
                    ind, inter = penalty(model, x_batch)
                    struct_loss = ind_lambda * (ind + inter)
                elif penalty_type in ['Jacob_F', 'Jacob_L1']:
                    ind, _ = penalty(model, x_batch)
                    struct_loss = ind_lambda * ind
                elif penalty_type == 'Layer_Weight':
                    struct_loss = ind_lambda * penalty(model)
                else:
                    raise ValueError(f"Unknown penalty_type: {penalty_type}")

                loss = pred_loss + struct_loss

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item() * x_batch.size(0)

        train_loss = total_train_loss / len(train_dataset)

        # ----------------- Validation (MSE only) -----------------
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                y_pred = model(x_batch)
                pred_loss = mse_loss(y_pred, y_batch)
                total_val_loss += pred_loss.item() * x_batch.size(0)

        val_loss = total_val_loss / len(val_dataset)

        # ----------------- Early stopping -----------------
        if val_loss < (best_val_loss - MIN_VAL_IMPROVEMENT):
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                log(f"Epoch {epoch:04d} | Train={train_loss:.6f} | Val={val_loss:.6f}")
                log(f"Early stopping at epoch {epoch+1}")
                break

        if epoch % 100 == 0:
            log(f"Epoch {epoch:04d} | Train={train_loss:.6f} | Val={val_loss:.6f}")

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, best_val_loss

def evaluate_model(trained_model, network, dataset, importance,
                   importance_type, ignore_diagonal, cudnn_flag):
    """Compute the graph-importance matrix and AUROC/AUPRC metrics."""

    results = {}
    trained_model.eval()

    if importance_type == 'Layer_Weight':
        importance_matrix = importance.cal_gc(trained_model)
    elif importance_type in ['Shapley', 'Jacobian']:
        dataloader = DataLoader(dataset, batch_size=len(dataset))
        x_all, _ = next(iter(dataloader))
        with torch.backends.cudnn.flags(enabled=cudnn_flag):
            y_pred = trained_model(x_all)
            importance_matrix = importance.cal_shapley_value(y_pred, x_all)
    else:
        raise ValueError(f"Unknown importance_type: {importance_type}")

    metrics = cls_metrics(network, importance_matrix,
                          ignore_diagonal=ignore_diagonal, threshold=0.1)

    results.update({"AUROC": metrics['auroc'], "AUPRC": metrics['auprc']})
    suffix = " (ignr)" if ignore_diagonal else " (diag)"
    log(f"{importance_type+suffix:<16} | AUROC={metrics['auroc']:.4f} | "
        f"AUPRC={metrics['auprc']:.4f}")

    return results

def _safe_append_csv_row(row_dict, csv_path):
    """Append a row immediately so interrupted grid searches can resume."""

    try:
        dirpath = os.path.dirname(csv_path)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        write_header = (not os.path.exists(csv_path)) or (os.path.getsize(csv_path) == 0)
        df = pd.DataFrame([row_dict])
        df.to_csv(csv_path, mode='a', index=False, header=write_header)
    except Exception as e:
        log(f"[WARNING] Failed to write row to {csv_path}: {e}")
        traceback.print_exc()

def _load_existing_results(worker_save_path, keys, worker=None):
    """Load successful worker rows and reconstruct completed grid combinations."""

    if (not os.path.exists(worker_save_path)) or os.path.getsize(worker_save_path) == 0:
        return [], set()

    try:
        df = pd.read_csv(worker_save_path)
    except pd.errors.ParserError as e:
        log(f"[WARNING] ParserError while reading {worker_save_path}: {e}. "
            f"Existing results for resuming will be ignored.", worker=worker)
        return [], set()
    except Exception as e:
        log(f"[WARNING] Failed to read {worker_save_path}: {e}. "
            f"Existing results for resuming will be ignored.", worker=worker)
        return [], set()

    # Keep *all* rows as existing_rows so results include them
    existing_rows = df.to_dict(orient="records")

    # --- Only successful rows count as "completed" ---
    df_completed = df.copy()
    if "error" in df_completed.columns:
        df_completed = df_completed[df_completed["error"].isna()]
    if "val_loss" in df_completed.columns:
        df_completed = df_completed[~df_completed["val_loss"].isna()]
    # ------------------------------------------------

    missing = [k for k in keys if k not in df_completed.columns]
    if missing:
        log(f"[WARNING] In {worker_save_path} the following param columns are "
            f"missing: {missing}. Cannot safely use it for skipping configs.",
            worker=worker)
        return existing_rows, set()

    completed_combos = set()
    for _, row in df_completed.iterrows():
        try:
            combo = tuple(row[k] for k in keys)
            completed_combos.add(combo)
        except Exception:
            continue

    return existing_rows, completed_combos

def grid_search(dataset, network, model_type, penalty_type,
                importance_type, ignore_diagonal, param_grid,
                batch_size, save_path, num_workers,
                exec_idx, device, seed):
    """Run a deterministic, sharded hyperparameter grid search.

    The random seed is reset before each configuration. This mirrors the
    original ICML experiments and is required for reproducible comparisons
    across hyperparameter settings.
    """

    # ----------------- basic setup -----------------
    try:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
    except Exception as e:
        log(f"[WARNING] Could not create directory for save_path={save_path}: {e}. "
            f"Continuing anyway.", worker=exec_idx)

    worker_save_path = save_path.replace(".csv", f"_worker{exec_idx}.csv")

    # cuDNN flag: only useful for LSTM-type models
    cudnn_flag = False if model_type in ['cLSTM', 'LSTM'] else True

    # penalty & importance objects (shared across configs)
    penalty_obj = setup_penalties(model_type, penalty_type, device)
    importance_obj = setup_importance(model_type, importance_type, device)

    train_dataset, val_dataset = dataset.split_series(split_ratio=0.7)

    # ----------------- build list of hyperparam combinations -----------------
    keys = list(param_grid.keys())
    values = [param_grid[k] for k in keys]

    all_combinations = list(itertools.product(*values))
    total_configs = len(all_combinations)

    # assign combinations to workers (shard by index)
    assigned_combinations = [
        combo
        for idx, combo in enumerate(all_combinations)
        if idx % num_workers == (exec_idx - 1)
    ]
    assigned_configs = len(assigned_combinations)

    log(f"Total configs={total_configs} | assigned to this worker={assigned_configs}",
        worker=exec_idx)

    # ----------------- load existing results & completed combos -----------------
    existing_rows, completed_combos = _load_existing_results(
        worker_save_path, keys, worker=exec_idx
    )

    if completed_combos:
        before = len(assigned_combinations)
        assigned_combinations = [
            combo for combo in assigned_combinations if combo not in completed_combos
        ]
        after = len(assigned_combinations)
        skipped = before - after
        log(f"Resuming from {worker_save_path}: skipped {skipped} already-finished "
            f"configs, remaining={after}", worker=exec_idx)
    else:
        if existing_rows:
            # we loaded rows but could not safely build combos
            log("Existing rows found but cannot use them to skip configs; "
                "running full assigned grid again.", worker=exec_idx)
        else:
            log("No previous results found for this worker; starting fresh.",
                worker=exec_idx)

    # Start results with already finished rows, so returned DataFrame contains them
    results = list(existing_rows)

    # ----------------- iterate over configurations -----------------
    for combo_idx, combination in enumerate(assigned_combinations, start=1):
        params = dict(zip(keys, combination))
        set_seed(seed)

        log(f"Config {combo_idx}/{len(assigned_combinations)} | params={params}",
            worker=exec_idx)

        try:
            model = build_model(
                model_type=model_type,
                dim=dataset.output_dim,
                lag=dataset.lag,
                params=params,
                device=device,
            )

            trained_model, val_loss = train_model(
                model=model,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                penalty=penalty_obj,
                batch_size=batch_size,
                ind_lambda=params['ind_lambda'],
                int_lambda=params['int_lambda'],
                weight_decay=params['weight_decay'],
                lr=params['lr'],
                penalty_type=penalty_type,
                cudnn_flag=cudnn_flag,
            )

            log(f"Validation loss: {val_loss:.6f}", worker=exec_idx)

            metrics_dict = evaluate_model(
                trained_model=trained_model,
                network=network,
                dataset=dataset,
                importance=importance_obj,
                importance_type=importance_type,
                ignore_diagonal=ignore_diagonal,
                cudnn_flag=cudnn_flag,
            )

            result_entry = {
                **params,
                "val_loss": val_loss,
                "model_type": model_type,
                "penalty_type": penalty_type,
                "importance_type": importance_type,
                **metrics_dict,
                "worker": exec_idx,
            }
            _safe_append_csv_row(result_entry, worker_save_path)
            results.append(result_entry)

            log(
                f"DONE config {combo_idx}/{len(assigned_combinations)} | "
                f"val_loss={val_loss:.6f} | AUROC={metrics_dict['AUROC']:.4f} | "
                f"AUPRC={metrics_dict['AUPRC']:.4f}",
                worker=exec_idx,
            )

        except Exception as e:
            log(f"[WARNING] Error with params {params}: {e}", worker=exec_idx)
            traceback.print_exc()

            # error_entry = {
            #     **params,
            #     "val_loss": float("nan"),
            #     "model_type": model_type,
            #     "penalty_type": penalty_type,
            #     "importance_type": importance_type,
            #     "AUROC": float("nan"),
            #     "AUPRC": float("nan"),
            #     "worker": exec_idx,
            #     "error": repr(e),
            # }
            # results.append(error_entry)
            # _safe_append_csv_row(error_entry, worker_save_path)

    results_df = pd.DataFrame(results)
    return results_df
