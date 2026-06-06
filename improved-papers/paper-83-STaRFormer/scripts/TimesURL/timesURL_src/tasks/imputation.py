import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader


def metrics(true, pred, mask):
    mask = 1. - mask
    mse = np.power((true - pred) * mask, 2).sum() / mask.sum()
    mae = np.abs((true - pred) * mask).sum() / mask.sum()
    rmse = np.sqrt(mse)
    return {'mse': mse, 'mae': mae, 'rmse': rmse}


def split(data, test_slice, seq_len):
    x, m = data['x'][:, test_slice], data['mask'][:, test_slice]
    value, mask = np.zeros((x.shape[1] // seq_len, seq_len, x.shape[2])), np.zeros((x.shape[1] // seq_len, seq_len, m.shape[2]))
    for i in range(x.shape[1] // seq_len):
        if (i+1) * seq_len > x.shape[1]:
            break
        value[i] = x[0, i*seq_len:(i+1)*seq_len, :]
        mask[i] = m[0, i*seq_len:(i+1)*seq_len, :]
    return torch.from_numpy(value), torch.from_numpy(mask)


def eval_imputation(model, data, test_slice, missing_rate, n_covariate_cols, device):
    value, mask = split(data, test_slice, 96)
    test_loader = DataLoader(TensorDataset(value, mask), batch_size=128, shuffle=False, num_workers=8)
    with torch.no_grad():
        true, pred, m = [], [], []
        for batch in test_loader:
            x = batch[0].float().to(device)

            mask = torch.randn_like(x[..., :-1])
            mask[mask > missing_rate] = 1.
            mask[mask <= missing_rate] = 0.

            # val = torch.cat([x[..., :-1].masked_fill(mask == 0., 0.), x[..., -1:]], dim=-1)
            # out = model._net.imputation(x[..., :-1].masked_fill(mask == 0., 0.), mask)
            out = model.net(x[..., :-1].masked_fill(mask == 0., 0.), mask, imputation=True)

            true.append(x[..., :-1].cpu().detach().numpy())
            pred.append(out.cpu().detach().numpy())
            # true.append(out[0].cpu().detach().numpy())
            # pred.append(out[1].cpu().detach().numpy())
            m.append(mask.cpu().detach().numpy())

        true = np.concatenate(true, axis=0)[..., n_covariate_cols:]
        pred = np.concatenate(pred, axis=0)[..., n_covariate_cols:]
        # pred = np.concatenate(pred, axis=0)
        m = np.concatenate(m, axis=0)
    return None, metrics(true, pred, m)
