import time
import numpy as np
import torch
import torch.nn.functional as F


def get_weight_matrix(n_classes):
    i = torch.arange(1, n_classes + 1).unsqueeze(1)
    j = torch.arange(1, n_classes + 1).unsqueeze(0)
    W = 1.0 / (i + j)
    return W

def get_weight_kernel(n_classes):
    w = 1 / torch.arange(2, 2*n_classes+1).view(1, 1, -1)
    return w

def infer_f1_labels(P: torch.Tensor, p0: float, matrix_mul=False, return_time=False) -> torch.Tensor:
    n_classes = P.size(-1)
    start = time.time()
    
    if matrix_mul:
        W = get_weight_matrix(n_classes).to(P.device)
        D = torch.matmul(P, W)
    else:
        w = get_weight_kernel(n_classes).to(P.device)
        D = F.conv1d(w, P.view(n_classes, 1, n_classes)).squeeze(0)

    end = time.time()
    # print(f'MatMul/Conv time: {end - start:.8f}s')

    max_score, max_labels = p0, torch.tensor([], dtype=torch.long)
    for num_positives in range(1, n_classes+1):
        values, labels = torch.topk(D[:, num_positives-1], num_positives)
        score = values.sum().item()
        if score > max_score:
            max_score = score
            max_labels = labels

    if return_time:
        return max_labels, end - start
    else:
        return max_labels


def infer_f1_labels_numpy(P: np.ndarray) -> np.ndarray:
    n_classes = P.shape[-1]
    W = np.zeros((n_classes, n_classes))
    for i in range(1, n_classes + 1):
        for j in range(1, n_classes + 1):
            W[i-1, j-1] = 2 / (i + j)

    D = np.matmul(P, W)

    max_score, max_labels = 0, []
    for num_positives in range(1, n_classes+1):
        # values, labels = np.argsort(D[:, num_positives-1])[:num_positives], np.argsort(D[:, num_positives-1])[:num_positives]
        labels = np.argsort(D[:, num_positives-1])[-num_positives:]
        values = [D[i, num_positives-1] for i in labels]
        score = np.sum(values)
        if score > max_score:
            max_score = score
            max_labels = labels
    return max_labels
