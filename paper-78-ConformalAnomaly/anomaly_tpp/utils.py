from typing import List

import numpy as np
import torch

from sklearn.metrics import roc_auc_score
from tqdm.auto import trange

import anomaly_tpp.models.recurrent_tpp
from anomaly_tpp.data import Batch


def extract_poisson_arrival_times(model, batch: Batch) -> List[List[np.ndarray]]:
    r"""Get the compensated arrivals times of each mark for each sequence in batch.

    Returns:
        poisson_times: poisson_times[idx][k] = compensated arrival times of events of
            type k for sequence idx. poisson_times[idx][k][-1] = \Lambda_k^*(t_max).
    """
    with torch.no_grad():
        compensator = model.get_compensator_per_mark(batch).cpu().detach().numpy()
        lengths = batch.mask.cpu().sum(-1).long().numpy()
        marks = batch.marks
        if marks is not None:
            marks = batch.marks.cpu().numpy()
        batch_size, _, num_marks = compensator.shape
        poisson_times = []
        for idx in range(batch_size):
            num_events = lengths[idx]
            comp = compensator[idx, : num_events + 1]
            if marks is None:
                arrivals = [comp[..., 0]]
            else:
                m = marks[idx, :num_events]
                arrivals = [
                    np.concatenate(
                        [comp[:num_events][m == k, k], [comp[num_events, k]]]
                    )
                    for k in range(num_marks)
                ]
            poisson_times.append(arrivals)
        return poisson_times


def roc_auc_from_pvals(id_pvals: np.ndarray, ood_pvals: np.ndarray):
    """Compute ROC AUC score given p-values for ID and OOD instances."""
    y_true = np.concatenate([np.zeros_like(id_pvals), np.ones_like(ood_pvals)])
    y_pred = np.concatenate([-id_pvals, -ood_pvals])
    return roc_auc_score(y_true, y_pred)


def fit_ntpp_model(
    dataloader,
    num_marks,
    context_size=64,
    mark_embedding_size=32,
    num_components=24,
    learning_rate=1e-4,
    max_epochs=100,
    grad_clip_norm=5.0,
    patience=5,
    seed=None,
):
    """Fit a neural TPP model to the sequences in the given dataloader"""
    if seed is not None:
        torch.manual_seed(seed)
    model = anomaly_tpp.models.recurrent_tpp.RecurrentTPP(
        num_marks=num_marks,
        context_size=context_size,
        mark_embedding_size=mark_embedding_size,
        num_components=num_components,
    )
    model.cuda()
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_loss = np.inf
    impatient = 0

    with trange(max_epochs) as t:
        for epoch in t:
            batch_losses = []
            for batch in dataloader:
                batch.cuda()
                loss = model.nll_loss(batch).mean()
                opt.zero_grad()
                loss.backward()
                if grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                opt.step()
                batch_losses.append(loss.item())
            epoch_loss = np.mean(batch_losses)

            t.set_postfix({"train_loss": epoch_loss})

            if epoch_loss >= best_loss:
                impatient += 1
            else:
                best_state_dict = model.state_dict()
                best_loss = epoch_loss
                impatient = 0

            if impatient > patience:
                print(f"Early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state_dict)
    return model.cpu()


def tpr_fpr_from_pvals(id_pvals: np.ndarray, ood_pvals: np.ndarray, alpha: float):
    """
    Calculate TPR and FPR using p-values.
    We treat OOD sequences as positives and ID sequences as negatives.
    """
    
    true_labels = np.concatenate([np.zeros_like(id_pvals), np.ones_like(ood_pvals)])
    p_values = np.concatenate([id_pvals, ood_pvals])

    rejected = p_values <= alpha

    true_positives = np.sum(rejected & (true_labels == 1))
    false_positives = np.sum(rejected & (true_labels == 0))
    false_negatives = np.sum(~rejected & (true_labels == 1))
    true_negatives = np.sum(~rejected & (true_labels == 0))
    
    tpr = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    fpr = false_positives / (false_positives + true_negatives) if (false_positives + true_negatives) > 0 else 0
        
    return tpr, fpr