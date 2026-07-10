"""Apply CAUCHY-008: Ensemble averaging of predictions across seeds before computing errors."""
import re

with open("best_config_gap_filling.py") as f:
    content = f.read()

# Replace the run_model function to use ensemble averaging
old_run = """def run_model(name, model_factory, train_loader, val_loader, test_data,
              n_seeds=3, **train_kwargs):
    rows = []
    for s in range(n_seeds):
        torch.manual_seed(s); np.random.seed(s)
        model = model_factory(s)
        rows.append(train_score_one(f"{name} seed{s}", model, train_loader,
                                    val_loader, test_data, log=True, **train_kwargs))
    errs_concat = np.concatenate([np.asarray(r["errs"]) for r in rows])
    return {
        "mae_mean":    float(errs_concat.mean()),
        "mae_median":  float(np.median(errs_concat)),
        "mae_std":     float(errs_concat.std()),
        "mae_max":     float(errs_concat.max()),
        "params":      rows[0]["params"],
        "train_time_s_mean": float(np.mean([r["train_time_s"] for r in rows])),
        "infer_time_ms_mean": float(np.mean([r["infer_time_ms"] for r in rows])),
        "per_seed": rows,
    }"""

new_run = """def run_model(name, model_factory, train_loader, val_loader, test_data,
              n_seeds=3, ensemble=True, **train_kwargs):
    rows = []
    all_preds = []
    for s in range(n_seeds):
        torch.manual_seed(s); np.random.seed(s)
        model = model_factory(s)
        row = train_score_one(f"{name} seed{s}", model, train_loader,
                                    val_loader, test_data, log=True, **train_kwargs)
        rows.append(row)
        # Collect per-seed predictions and targets for ensemble
        model.train(False)
        test_X, test_Y = test_data
        test_X_dev = test_X.to(device) if hasattr(test_X, to) else test_X
        with torch.no_grad():
            out = model(test_X_dev)
            if isinstance(out, tuple):
                out = out[0]
            all_preds.append(out.cpu().numpy().flatten())

    all_preds = np.array(all_preds)  # (n_seeds, n_test)
    test_Y_np = test_data[1].cpu().numpy().flatten() if hasattr(test_data[1], cpu) else test_data[1].numpy().flatten()

    if ensemble and n_seeds > 1:
        # Ensemble: average predictions pointwise, then compute errors
        ensemble_preds = all_preds.mean(axis=0)
        errs_ensemble = np.abs(ensemble_preds - test_Y_np)
        errs_concat = errs_ensemble
    else:
        # Original: concatenate per-seed errors
        errs_concat = np.concatenate([np.asarray(r["errs"]) for r in rows])

    return {
        "mae_mean":    float(errs_concat.mean()),
        "mae_median":  float(np.median(errs_concat)),
        "mae_std":     float(errs_concat.std()),
        "mae_max":     float(errs_concat.max()),
        "params":      rows[0]["params"],
        "train_time_s_mean": float(np.mean([r["train_time_s"] for r in rows])),
        "infer_time_ms_mean": float(np.mean([r["infer_time_ms"] for r in rows])),
        "per_seed": rows,
    }"""

# Use regex to replace (more robust to minor whitespace differences)
import re
# Simple string replacement with normalized whitespace check
assert "def run_model(name, model_factory, train_loader, val_loader, test_data," in content, "Could not find run_model"
assert "errs_concat = np.concatenate" in content, "Could not find errs_concat"

# Do the replacement between function start and end
start = content.find("def run_model(")
# Find the next def or main after run_model
next_def = content.find("\ndef main()", start)
func_body = content[start:next_def]

new_func = """def run_model(name, model_factory, train_loader, val_loader, test_data,
              n_seeds=3, ensemble=True, **train_kwargs):
    rows = []
    all_preds = []
    for s in range(n_seeds):
        torch.manual_seed(s); np.random.seed(s)
        model = model_factory(s)
        row = train_score_one(f"{name} seed{s}", model, train_loader,
                                    val_loader, test_data, log=True, **train_kwargs)
        rows.append(row)
        # Re-evaluate predictions for ensemble averaging
        model.train(False)
        test_X, test_Y = test_data
        test_X_dev = test_X.to(device) if hasattr(test_X, "to") else test_X
        with torch.no_grad():
            out = model(test_X_dev)
            if isinstance(out, tuple):
                out = out[0]
            all_preds.append(out.cpu().numpy().flatten())

    all_preds = np.array(all_preds)
    test_Y_np = test_data[1].cpu().numpy().flatten() if hasattr(test_data[1], "cpu") else test_data[1].numpy().flatten()

    if ensemble and n_seeds > 1:
        ensemble_preds = all_preds.mean(axis=0)
        errs_concat = np.abs(ensemble_preds - test_Y_np)
    else:
        errs_concat = np.concatenate([np.asarray(r["errs"]) for r in rows])

    return {
        "mae_mean":    float(errs_concat.mean()),
        "mae_median":  float(np.median(errs_concat)),
        "mae_std":     float(errs_concat.std()),
        "mae_max":     float(errs_concat.max()),
        "params":      rows[0]["params"],
        "train_time_s_mean": float(np.mean([r["train_time_s"] for r in rows])),
        "infer_time_ms_mean": float(np.mean([r["infer_time_ms"] for r in rows])),
        "per_seed": rows,
    }

"""

# Build modified content
modified = content[:start] + new_func + content[next_def:]

with open("best_config_gap_filling.py", "w") as f:
    f.write(modified)
print("Applied CAUCHY-008: ensemble averaging across seeds")
