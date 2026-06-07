import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)


@torch.no_grad()
def eval_source(model, loader, device):
    model.eval()
    y_true, y_prob, y_pred = [], [], []
    for batch in loader:
        images = batch["img"].to(device)
        monet = batch["monet"].to(device)
        logits, _, _, _ = model.forward_eval(images, monet)
        prob = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        y_true.extend(batch["label"].numpy().tolist())
        y_prob.extend(prob.tolist())
        y_pred.extend((prob >= 0.5).astype(int).tolist())
    try:
        auroc = roc_auc_score(y_true, y_prob)
    except Exception:
        auroc = float("nan")
    return {
        "acc": accuracy_score(y_true, y_pred),
        "bacc": balanced_accuracy_score(y_true, y_pred),
        "auroc": auroc,
    }


@torch.no_grad()
def eval_target(model, loader, device, use_recall_floor: bool, mel_recall_floor: float, report_opt: bool = True):
    model.eval()
    y_true, y_prob = [], []
    for batch in loader:
        images = batch["img"].to(device)
        monet = batch["monet"].to(device)
        logits, _, _, _ = model.forward_eval(images, monet)
        prob = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        y_true.extend(batch["label"].numpy().tolist())
        y_prob.extend(prob.tolist())
    y_true_np = np.array(y_true)
    y_prob_np = np.array(y_prob)
    y_pred_np = (y_prob_np >= 0.5).astype(int)
    try:
        auroc = roc_auc_score(y_true_np, y_prob_np)
    except Exception:
        auroc = float("nan")
    output = {
        "acc": accuracy_score(y_true_np, y_pred_np),
        "bacc": balanced_accuracy_score(y_true_np, y_pred_np),
        "auroc": auroc,
    }
    if report_opt and len(np.unique(y_true_np)) == 2:
        fpr, tpr, thresholds = roc_curve(y_true_np, y_prob_np)
        best_idx = int(np.argmax(tpr - fpr))
        best_threshold = float(thresholds[best_idx])
        if use_recall_floor:
            best_bacc = -1.0
            floor_threshold = 0.5
            for threshold in np.linspace(0, 1, 1001):
                y_pred = (y_prob_np >= threshold).astype(int)
                true_pos = ((y_true_np == 1) & (y_pred == 1)).sum()
                false_neg = ((y_true_np == 1) & (y_pred == 0)).sum()
                recall_mel = true_pos / max(true_pos + false_neg, 1)
                if recall_mel >= mel_recall_floor:
                    bacc = balanced_accuracy_score(y_true_np, y_pred)
                    if bacc > best_bacc:
                        best_bacc = bacc
                        floor_threshold = float(threshold)
            best_threshold = floor_threshold
            best_bacc_value = best_bacc
        else:
            best_bacc_value = balanced_accuracy_score(y_true_np, (y_prob_np >= best_threshold).astype(int))
        output["thr_opt"] = best_threshold
        output["bacc_opt"] = best_bacc_value
    return output


def print_eval_report(title: str, y_true, y_prob, threshold: float, out_csv: str, paths: list[str]) -> None:
    y_true_np = np.array(y_true)
    y_prob_np = np.array(y_prob)
    y_pred_np = (y_prob_np >= threshold).astype(int)
    try:
        auroc = roc_auc_score(y_true_np, y_prob_np)
    except Exception:
        auroc = float("nan")
    print(f"\n{title} (threshold={threshold:.2f})")
    print(f"Mel prevalence   : {y_true_np.mean():.3f}")
    print(f"Accuracy         : {accuracy_score(y_true_np, y_pred_np):.4f}")
    print(f"Balanced Acc     : {balanced_accuracy_score(y_true_np, y_pred_np):.4f}")
    print(f"AUROC            : {auroc:.4f}")
    print("\nClassification report:")
    print(classification_report(y_true_np, y_pred_np, target_names=["other", "mel"], digits=4))
    print("Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_true_np, y_pred_np, labels=[0, 1]))
    pd.DataFrame(
        {
            "path": paths,
            "label_true": y_true_np,
            "prob_mel": y_prob_np,
            "pred_mel": y_pred_np,
        }
    ).to_csv(out_csv, index=False)
    print(f"\nSaved per-image predictions to: {out_csv}")
