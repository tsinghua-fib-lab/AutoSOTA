import torch
import numpy as np




def compute_correlations(labels, preds, return_detail=False):
    device = labels.device

    labels = labels.detach().cpu().numpy()
    preds = preds.detach().cpu().numpy()
    corr = np.nan_to_num([np.corrcoef(labels[:, i], preds[:, i])[0, 1] for i in range(labels.shape[1])],
                         nan=-1).tolist()
    if return_detail:
        return corr
    corr = np.mean(corr)
    return torch.FloatTensor([corr]).to(device)


def pearsonr(x, y):

    mean_x = torch.mean(x)
    mean_y = torch.mean(y)
    xm = x.sub(mean_x)
    ym = y.sub(mean_y)
    r_num = xm.dot(ym)
    r_den = torch.norm(xm, 2) * torch.norm(ym, 2)
    r_val = r_num / (r_den + 1e-8)
    r_val = torch.nan_to_num(r_val,nan=-1)
    return r_val