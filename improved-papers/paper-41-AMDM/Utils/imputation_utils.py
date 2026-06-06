import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from torch import nn


def get_quantile(samples,q,dim=1):
    return torch.quantile(samples,q,dim=dim).cpu().numpy()

def plot_sample(ori_data, gen_data, masks, sample_idx=0):
    plt.rcParams["font.size"] = 12
    fig, axes = plt.subplots(nrows=7, ncols=4, figsize=(12, 15))
    sample_num, seq_len, feat_dim = ori_data.shape
    observed = ori_data * masks

    quantiles = []
    quantiles.append(get_quantile(torch.from_numpy(gen_data), 0.5, dim=0) * (1 - masks) + observed)
    quantiles.append(get_quantile(torch.from_numpy(gen_data), 0.05, dim=0) * (1 - masks) + observed)
    quantiles.append(get_quantile(torch.from_numpy(gen_data), 0.95, dim=0) * (1 - masks) + observed)

    for feat_idx in range(feat_dim):
        row = feat_idx // 4
        col = feat_idx % 4

        df_x = pd.DataFrame({"x": np.arange(0, seq_len), "val": ori_data[sample_idx, :, feat_idx],
                             "y": masks[sample_idx, :, feat_idx]})
        df_x = df_x[df_x.y!=0]

        df_o = pd.DataFrame({"x": np.arange(0, seq_len), "val": ori_data[sample_idx, :, feat_idx],
                             "y": (1 - masks)[sample_idx, :, feat_idx]})
        df_o = df_o[df_o.y!=0]

        axes[row][col].plot(range(0, seq_len), quantiles[0][sample_idx, :, feat_idx], color='g', linestyle='solid', label='Diffusion-TS')
        axes[row][col].fill_between(range(0, seq_len), quantiles[1][sample_idx, :, feat_idx],
                         quantiles[2][sample_idx, :, feat_idx], color='g', alpha=0.3)
    
        axes[row][col].plot(df_o.x, df_o.val, color='b', marker='o', linestyle='None')
        axes[row][col].plot(df_x.x, df_x.val, color='r', marker='x', linestyle='None')

        if col == 0:
            plt.setp(axes[row, 0], ylabel='value')
        if row == -1:
            plt.setp(axes[-1, col], xlabel='time')
    plt.tight_layout()
    plt.show()


class MaskedLoss(nn.Module):
    """ Masked MSE Loss
    """

    def __init__(self, reduction: str = 'mean', mode='mse'):

        super().__init__()

        self.reduction = reduction
        if mode == 'mse':
            self.loss = nn.MSELoss(reduction=self.reduction)
        else:
            self.loss = nn.L1Loss(reduction=self.reduction)

    def forward(self,
                y_pred: torch.Tensor, y_true: torch.Tensor, mask: torch.BoolTensor) -> torch.Tensor:
        """Compute the loss between a target value and a prediction.

        Args:
            y_pred: Estimated values
            y_true: Target values
            mask: boolean tensor with 0s at places where values should be ignored and 1s where they should be considered

        Returns
        -------
        if reduction == 'none':
            (num_active,) Loss for each active batch element as a tensor with gradient attached.
        if reduction == 'mean':
            scalar mean loss over batch as a tensor with gradient attached.
        """

        # for this particular loss, one may also elementwise multiply y_pred and y_true with the inverted mask
        masked_pred = torch.masked_select(y_pred, mask)
        masked_true = torch.masked_select(y_true, mask)

        return self.loss(masked_pred, masked_true)
    
def random_mask(observed_values, missing_ratio=0.1, seed=1984):
    observed_masks = ~np.isnan(observed_values)

    # randomly set some percentage as ground-truth
    masks = observed_masks.reshape(-1).copy()
    obs_indices = np.where(masks)[0].tolist()

    # Store the state of the RNG to restore later.
    st0 = np.random.get_state()
    np.random.seed(seed)

    miss_indices = np.random.choice(
        obs_indices, (int)(len(obs_indices) * missing_ratio), replace=False
    )

    # Restore RNG.
    np.random.set_state(st0)
    
    masks[miss_indices] = False
    gt_masks = masks.reshape(observed_masks.shape)

    observed_values = np.nan_to_num(observed_values)
    return torch.from_numpy(observed_values).float(), torch.from_numpy(observed_masks).float(),\
           torch.from_numpy(gt_masks).float()