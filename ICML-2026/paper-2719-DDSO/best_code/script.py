import os
import torch
from torch.optim import Adam
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from utils import evaluate
from model_all import * 
from datasets import ClassificationDataModule, classification_shapes
import pandas as pd
from typing import Dict, Tuple, Any, List


def suggest_n_hidden(input_dim: int) -> int:
    """Heuristic to choose hidden width based on input dimension."""
    if input_dim <= 16:
        return int(2 * input_dim)
    elif input_dim <= 64:
        return int(1.6 * input_dim)
    else:
        return min(2.5 * input_dim, 256)


def train_and_evaluate(
    dataset_str: str,
    model_str: str = "ALD_Classifier",
    epochs: int = 200,
    batch_size: int = 256,
    lr: float = 1e-3,          
    num_runs: int = 5,
    lambda_reg: float = 0.9,   
    t: float = 1.0,
    # MMD/KCE hyperparams (only used for *_MMD models)
    lambda_kce: float = 0.1,   # weight of KCE(MMD) regularizer
    sigma2_z: float = None,    # RBF bandwidth for k_Z on probs; None -> median heuristic
    early_stop: bool = True,
    patience: int = 50,        
    min_epochs: int = 10,     
) -> Dict[str, Tuple[float, float]]:
    """
    Train model for `epochs`, evaluate, and return dict of metric -> (mean, std) over runs.
    """
    all_metrics: List[Dict[str, float]] = []

    all_run_train_hist: List[List[float]] = []
    all_run_test_hist: List[List[float]] = []

    for run in range(num_runs):
        print(f"=== {model_str} | {dataset_str} | Epochs {epochs} | Run {run + 1}/{num_runs} ===")
        data_module = ClassificationDataModule(
            dataset_str,
            data_dir="datasets",
            normalize=True,
            batch_size=batch_size,
            test_batch_size=batch_size,
            random_seed=42 + run,
        )
        data_module.setup()
        input_dim, num_classes = classification_shapes[dataset_str]
        n_hidden = suggest_n_hidden(input_dim)

        if model_str not in globals():
            raise ValueError(f"Unknown model_str: {model_str} (no such class in model_all.py)")

        ModelClass = globals()[model_str]

        ic_kwargs = {}
        if model_str.startswith("IC") and "MMD" not in model_str:
            ic_kwargs["q_dim"] = 2

        model = ModelClass(
            input_dim=input_dim,
            n_hidden=n_hidden,
            output_dim=num_classes,
            **ic_kwargs,
        )

        model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-4)


        def compute_batch_loss(x_batch: torch.Tensor, y_batch: torch.Tensor) -> torch.Tensor:
            x_batch = x_batch.to(model.device if hasattr(model, 'device') else next(model.parameters()).device)
            y_batch = y_batch.to(model.device if hasattr(model, 'device') else next(model.parameters()).device)

            if hasattr(model, "loss_with_mmd"):
                try:
                    loss_val, logs = model.loss_with_mmd(
                        x_batch,
                        y_batch.view(-1),
                        lambda_kce=lambda_kce,
                        t=t,
                        sigma2_z=sigma2_z, 
                    )
                except TypeError:
                    loss_val, logs = model.loss_with_mmd(
                        x_batch,
                        y_batch.view(-1),
                        lambda_kce=lambda_kce,
                        sigma2_z=sigma2_z,
                    )
                return loss_val


            loss_methods = [
                name
                for name in dir(model)
                if name.startswith("loss_") and "with_mmd" not in name
            ]
            if len(loss_methods) != 1:
                pass 
            

            if len(loss_methods) > 0:
                loss_fn_name = loss_methods[0]
                loss_fn = getattr(model, loss_fn_name)
            else:
                raise RuntimeError(f"No loss function found for {model_str}")

            q_local = torch.rand(x_batch.shape[0], 1, device=x_batch.device)
            loss_val = None

            arg_candidates = [
                (x_batch, y_batch.view(-1), q_local, lambda_reg, t),
                (x_batch, y_batch.view(-1), q_local, lambda_reg),
                (x_batch, y_batch.view(-1), t),
                (x_batch, y_batch.view(-1),),
            ]

            for args in arg_candidates:
                try:
                    loss_val = loss_fn(*args)
                    break
                except TypeError:
                    continue

            if loss_val is None:
                raise RuntimeError(
                    f"Could not call loss function {loss_fn.__name__} "
                    f"for model {model_str} with any of the tried signatures."
                )
            return loss_val

        train_loss_hist: List[float] = []
        test_loss_hist: List[float] = []

        best_score = -1e9          
        best_epoch = -1
        best_state_dict = None
        epochs_no_improve = 0

        for epoch in range(epochs):
            model.train()
            epoch_train_losses = []


            for x, y in data_module.train_dataloader():
                optimizer.zero_grad()
                loss = compute_batch_loss(x, y)
                loss.backward()
                optimizer.step()
                epoch_train_losses.append(loss.item())

            mean_train_loss = float(np.mean(epoch_train_losses))

            model.eval()
            epoch_test_losses = []
            

            x_test_all = []
            y_test_all = []
            
            with torch.no_grad():
                for x_test, y_test in data_module.test_dataloader():
                    dev = next(model.parameters()).device
                    x_test = x_test.to(dev)
                    y_test = y_test.to(dev)

                    l_test = compute_batch_loss(x_test, y_test)
                    epoch_test_losses.append(l_test.item())
                    x_test_all.append(x_test)
                    y_test_all.append(y_test)

                x_test_cat = torch.cat(x_test_all, dim=0)
                y_test_cat = torch.cat(y_test_all, dim=0)
                
                val_metrics = evaluate(model_str, model, x_test_cat, y_test_cat.view(-1), t=t)

            mean_test_loss = float(np.mean(epoch_test_losses))
            train_loss_hist.append(mean_train_loss)
            test_loss_hist.append(mean_test_loss)

            acc = val_metrics.get("Accuracy", float("nan"))
            ece = val_metrics.get("ECE", float("nan"))
            kce = val_metrics.get("KCE", float("nan")) 


            if not np.isnan(acc) and not np.isnan(ece):
                score = acc - ece # Reference metric. Replace with your specific early stopping condition.
            elif not np.isnan(acc):
                score = acc
            else:
                score = -1e9

            improved = score > best_score + 1e-6

            if improved:
                best_score = score
                best_epoch = epoch

                best_state_dict = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1


            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(
                    f"Run {run+1} | Ep {epoch+1}/{epochs} "
                    f"| TrLoss: {mean_train_loss:.4f} "
                    f"| TeLoss: {mean_test_loss:.4f} "
                    f"| Acc: {acc:.4f} | ECE: {ece:.4f} | KCE: {kce:.4f}" 
                )

            if (
                early_stop
                and epoch + 1 >= min_epochs
                and epochs_no_improve >= patience
            ):
                print(
                    f"[EarlyStop] Stop at epoch {epoch+1}, "
                    f"best epoch = {best_epoch+1}, best score (Acc-ECE) = {best_score:.4f}"
                )
                break

        all_run_train_hist.append(train_loss_hist)
        all_run_test_hist.append(test_loss_hist)

        if best_state_dict is not None:
            model.load_state_dict(best_state_dict)

        model.eval()
        with torch.no_grad():
            x_test_all = []
            y_test_all = []
            dev = next(model.parameters()).device
            for x, y in data_module.test_dataloader():
                x_test_all.append(x.to(dev))
                y_test_all.append(y.to(dev))
            
            x_test = torch.cat(x_test_all, dim=0)
            y_test = torch.cat(y_test_all, dim=0)
            

            test_metrics = evaluate(model_str, model, x_test, y_test.view(-1), t=t)
        
        all_metrics.append(test_metrics)
        print(f"Test Metrics (final, best epoch {best_epoch+1}): {test_metrics}")


    figs_dir = "Figs"
    os.makedirs(figs_dir, exist_ok=True)

    for i, (train_hist, test_hist) in enumerate(zip(all_run_train_hist, all_run_test_hist)):
        epochs_arr = np.arange(1, len(train_hist) + 1)
        plt.figure(figsize=(6, 4))
        plt.plot(epochs_arr, train_hist, label=f"Train Loss (run {i+1})")
        plt.plot(epochs_arr, test_hist, label=f"Test Loss (run {i+1})")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"{model_str} on {dataset_str} (run {i+1})")
        plt.legend()
        plt.tight_layout()
        fig_path = os.path.join(figs_dir, f"{model_str}_{dataset_str}_run{i+1}.png")
        plt.savefig(fig_path, dpi=150)
        plt.close()

    metric_keys = sorted(set().union(*[m.keys() for m in all_metrics]))
    mean_std_metrics: Dict[str, Tuple[float, float]] = {}
    for k in metric_keys:
        vals = [m.get(k, float("nan")) for m in all_metrics]
        vals = np.array(vals, dtype=float)
        mean_std_metrics[k] = (np.nanmean(vals), np.nanstd(vals))

    return mean_std_metrics


def fmt_mean_std(mean: float, std: float, is_percent: bool = False) -> str:
    if mean is None or np.isnan(mean):
        return "-"
    if is_percent:
        return f"{mean * 100:.3f} ± {std * 100:.3f}"
    else:
        return f"{mean:.3f} ± {std:.3f}"



def build_row(dataset_str: str, model_str: str, epochs: int,
              metric_dict: Dict[str, Tuple[float, float]]) -> Dict[str, Any]:
    def get(k: str) -> Tuple[float, float]:
        return metric_dict.get(k, (float("nan"), float("nan")))

    return {
        "Datasets": dataset_str,
        "Model": model_str,
        "Epochs": epochs,
        "Accuracy ↑": fmt_mean_std(*get("Accuracy"), is_percent=True),
        "ECE ↓": fmt_mean_std(*get("ECE")),
        "GroupX ECE ↓": fmt_mean_std(*get("GroupX_ECE")),
        "GroupY ECE ↓": fmt_mean_std(*get("GroupY_ECE")),
        "Entropy ↓": fmt_mean_std(*get("Entropy")),
        "KCE ↓": fmt_mean_std(*get("KCE")),
        "AP ↑": fmt_mean_std(*get("AP"), is_percent=True),
        "AUC ↑": fmt_mean_std(*get("AUC"), is_percent=True),
    }


if __name__ == "__main__":
    datasets = [
        "wdbc",
        "heart-disease",
        "online-shoppers",
        "dry-bean",
        "adult"
    ]

    epoch_grid = {
        "wdbc": [400],
        "heart-disease": [600],
        "online-shoppers": [200],
        "dry-bean": [500],
        "adult": [100],
    }

    model_list = [
        # "Softmax_Classifier",
        "ALD_Classifier",
        # "Norm_Classifier", 
        # "Logistic_Classifier",
        # "Exp_Classifier",
        # "LogNorm_Classifier",
        # "Weibull_Classifier",
        "ICALD_Classifier",
        # "ICSoftmax_Classifier",
        # "ALDMMD_Classifier",  
        # "SoftmaxMMD_Classifier",
    ]

    results: Dict[Tuple[str, str, int], Dict[str, Tuple[float, float]]] = {}

    for model_name in model_list:
        print(f"\n======= Model: {model_name} =======")
        for dataset_str in datasets:
            for epochs in epoch_grid.get(dataset_str, [100]):
                print(f"--- Training {model_name} on {dataset_str} with {epochs} epochs ---")
                
                metrics = train_and_evaluate(
                    dataset_str=dataset_str,
                    model_str=model_name,
                    epochs=epochs,
                    batch_size=128, 
                    num_runs=5,          
                    lambda_kce=0.1, 
                    sigma2_z=None,  
                    t=2,
                    early_stop=True,
                    patience=50,
                    lr=1e-3,
                    min_epochs=20,
                )
                results[(model_name, dataset_str, epochs)] = metrics


    print("\n=== Final Results (raw) ===")
    for (model_name, dataset_str, epochs), metric_dict in results.items():
        print(f"\nModel: {model_name} | Dataset: {dataset_str} | Epochs: {epochs}")
        for k, (mean, std) in metric_dict.items():
            if k in ["Accuracy", "AP", "AUC"]:
                print(f"{k:12s}: Mean = {mean * 100:.3f}, Std = {std * 100:.3f}")
            else:
                print(f"{k:12s}: Mean = {mean:.6f}, Std = {std:.6f}")


    ordered_cols = [
        "Datasets", "Model", "Epochs",
        "Accuracy ↑", "ECE ↓", "GroupX ECE ↓", "GroupY ECE ↓",
        "Entropy ↓", "KCE ↓", "AP ↑", "AUC ↑",
    ]

    rows = []
    for (model_name, dataset_str, epochs), metric_dict in results.items():
        rows.append(build_row(dataset_str, model_name, epochs, metric_dict))

    df = pd.DataFrame(rows)[ordered_cols]
    csv_path = "results_summary.csv"
    df.to_csv(csv_path, index=False)

    print("\n=== Saved results to results_summary.csv ===")
    print(df.to_string(index=False))