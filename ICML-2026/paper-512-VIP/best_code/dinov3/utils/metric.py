import numpy as np
import torch


def IoU(mask1, mask2):
    mask1, mask2 = (mask1>0.5).to(torch.bool), (mask2>0.5).to(torch.bool)
    intersection = torch.sum(mask1 * (mask1 == mask2), dim=[-1, -2]).squeeze()
    union = torch.sum(mask1 + mask2, dim=[-1, -2]).squeeze()
    return (intersection.to(torch.float) / union).mean().item()


def accuracy(mask1, mask2):
    mask1, mask2 = (mask1>0.5).to(torch.bool), (mask2>0.5).to(torch.bool)
    return torch.mean((mask1 == mask2).to(torch.float)).item()


def precision_recall(mask_gt, mask):
    mask_gt, mask = mask_gt.to(torch.bool), mask.to(torch.bool)
    true_positive = torch.sum(mask_gt * (mask_gt == mask), dim=[-1, -2]).squeeze()
    mask_area = torch.sum(mask, dim=[-1, -2]).to(torch.float)
    mask_gt_area = torch.sum(mask_gt, dim=[-1, -2]).to(torch.float)

    precision = true_positive / mask_area
    precision[mask_area == 0.0] = 1.0

    recall = true_positive / mask_gt_area
    recall[mask_gt_area == 0.0] = 1.0

    return precision.item(), recall.item()


def F_score(p, r, betta_sq=0.3):
    f_scores = ((1 + betta_sq) * p * r) / (betta_sq * p + r)
    f_scores[f_scores != f_scores] = 0.0  # handle nans
    return f_scores


def F_max(precisions, recalls, betta_sq=0.3):
    F = F_score(precisions, recalls, betta_sq)
    return F.mean(dim=0).max().item()

@torch.no_grad()
def metrics(pred, gt, stats=(IoU, accuracy, F_max), prob_bins=255):
    avg_values = {}
    precisions = []
    recalls = []
    out_dict = {}
    
    nb_sample = len(gt)
    for step in range(nb_sample):
        prediction, mask = torch.from_numpy(pred[step]), torch.from_numpy(gt[step])

        for metric in stats:
            method = metric.__name__
            if method not in avg_values and metric != F_max:
                avg_values[method] = 0.0

            if metric != F_max:
                avg_values[method] += metric(mask, prediction)
            else:
                p, r = [], []
                splits = 2.0 * prediction.mean(dim=0) if prob_bins is None else \
                    np.arange(0.0, 1.0, 1.0 / prob_bins)

                for split in splits:
                    pr = precision_recall(mask, prediction > split)
                    p.append(pr[0])
                    r.append(pr[1])
                precisions.append(p)
                recalls.append(r)

    for metric in stats:
        method = metric.__name__
        if metric == F_max:
            out_dict[method] = F_max(torch.tensor(precisions), torch.tensor(recalls))
        else:
            out_dict[method] = avg_values[method] / nb_sample

    return out_dict