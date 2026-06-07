import torch
import torch.nn.functional as F

def sequence_loss(init_disp, disp_pred, disp_gt, valid, max_disp, loss_gamma=0.9):
    disp_loss = 0
    if max_disp:
        valid = (valid >= 0.5) & (disp_gt < max_disp)
    else:
        valid = (valid >= 0.5)
    disp_loss += F.smooth_l1_loss(init_disp[valid], disp_gt[valid], reduce='mean')
    
    n_prediction = len(disp_pred)
    for i in range(n_prediction):
        adjusted_loss_gamma = loss_gamma ** (15 / (n_prediction - 1))
        i_weight = adjusted_loss_gamma ** (n_prediction - i - 1)
        i_loss = torch.abs(disp_pred[i] - disp_gt)
        disp_loss += i_weight * i_loss[valid].mean()
    
    epe = torch.abs(disp_pred[-1] - disp_gt)
    epe = epe.view(-1)[valid.view(-1)]

    metric = {
        'train/EPE': epe.mean(),
        'train/1px': (epe < 1).float().mean(),
        'train/3px': (epe < 3).float().mean(),
        'train/5px': (epe < 5).float().mean()
    }

    return disp_loss, metric