import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score


def compute_supervised_metrics(logits, labels, suffix):
    result = {}

    if logits.dtype == torch.long:
        probs = logits.cpu().numpy()
        preds = logits.cpu().numpy()
        np_labels = labels.cpu().numpy()
    else:
        if logits.dim() == 1:
            probs = F.sigmoid(logits)
            preds = (probs > 0.5).long()
        else:
            probs = F.softmax(logits, dim=1)
            _, preds = torch.max(logits, 1)

        probs = probs.cpu().numpy()
        preds = preds.cpu().numpy()
        np_labels = labels.cpu().numpy()

    result[f"accuracy_{suffix}"] = accuracy_score(np_labels, preds)

    return result
