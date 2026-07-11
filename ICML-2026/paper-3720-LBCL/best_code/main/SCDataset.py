# deep learning package
import torch
from torch.utils.data import Dataset

class SCDataset(Dataset):
    """
    A PyTorch Dataset that returns, for each batch index:
      - x1: tensor of shape (N_pos, G)  positive first elements
      - x2: tensor of shape (N_pos, G)  positive second elements
      - x_unl: tensor of shape (N_unlabeled, G)  unlabeled samples
    """

    def __init__(self, batch_all_label, batch_all_unlabel):
        """
        Args:
            batch_all_label (dict): batch_idx -> list of (tensor_i, tensor_j) positive pairs.
            batch_all_unlabel (dict): batch_idx -> list of tensor_k unlabeled samples.
        """
        self.batch_keys = list(batch_all_label.keys())
        self.batch_all_label = batch_all_label
        self.batch_all_unlabel = batch_all_unlabel

    def __len__(self):
        return len(self.batch_keys)

    def __getitem__(self, idx):
        """
        Returns:
            x1 (Tensor): shape (N_pos, G)
            x2 (Tensor): shape (N_pos, G)
            x_unl (Tensor): shape (N_unlabeled, G)
        """
        b = self.batch_keys[idx]
        pos_pairs = self.batch_all_label[b]      # list of (tensor_i, tensor_j)
        unlab_list = self.batch_all_unlabel[b]   # list of tensor_k

        # stack the positive pairs into two tensors
        x1 = torch.stack([p[0] for p in pos_pairs], dim=0)
        x2 = torch.stack([p[1] for p in pos_pairs], dim=0)

        # stack the unlabeled samples
        x_unl = torch.stack(unlab_list, dim=0) if len(unlab_list) > 0 else torch.empty((0, x1.shape[1]))

        return x1, x2, x_unl