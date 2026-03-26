# current implementation: only support numerical values
import numpy as np
import torch, os
from torch import nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import math
import argparse

class MaskEmbed(nn.Module):
    """ record to mask embedding
    """
    def __init__(self, rec_len=25, embed_dim=64, norm_layer=None):
        
        super().__init__()
        self.rec_len = rec_len
        self.proj = nn.Conv1d(1, embed_dim, kernel_size=1, stride=1)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, _, L = x.shape
        # assert(L == self.rec_len, f"Input data width ({L}) doesn't match model ({self.rec_len}).")
        x = self.proj(x)
        x = x.transpose(1, 2)
        x = self.norm(x)
        return x


class ActiveEmbed(nn.Module):
    """ record to mask embedding
    """
    def __init__(self, rec_len=25, embed_dim=64, norm_layer=None):
        
        super().__init__()
        self.rec_len = rec_len
        self.proj = nn.Conv1d(1, embed_dim, kernel_size=1, stride=1)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, _, L = x.shape
        # assert(L == self.rec_len, f"Input data width ({L}) doesn't match model ({self.rec_len}).")
        x = self.proj(x)
        x = torch.sin(x)
        x = x.transpose(1, 2)
        #   x = torch.cat((torch.sin(x), torch.cos(x + math.pi/2)), -1)
        x = self.norm(x)
        return x



def get_1d_sincos_pos_embed(embed_dim, pos, cls_token=False):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """

    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = np.arange(pos)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    pos_embed = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)

    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)

    return pos_embed


def adjust_learning_rate(optimizer, epoch, lr, min_lr, max_epochs, warmup_epochs):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < warmup_epochs:
        tmp_lr = lr * epoch / warmup_epochs 
    else:
        tmp_lr = min_lr + (lr - min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - warmup_epochs) / (max_epochs - warmup_epochs)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = tmp_lr * param_group["lr_scale"]
        else:
            param_group["lr"] = tmp_lr
    return tmp_lr


def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == np.inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm


class NativeScaler:

    state_dict_key = "amp_scaler"
    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)



class MAEDataset(Dataset):

    def __init__(self, X, M):        
         self.X = X
         self.M = M

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.M[idx]



def get_dataset(dataset : str, path : str):

    if dataset in ['climate', 'compression', 'wine', 'yacht', 'spam', 'letter', 'credit', 'raisin', 'bike', 'obesity', 'airfoil', 'blood', 'yeast', 'health', 'review', 'travel']:
        df = pd.read_csv(os.path.join(path, 'data', dataset + '.csv'))
        last_col = df.columns[-1]
        y = df[last_col]
        X = df.drop(columns=[last_col])
    elif dataset == 'california':
        from sklearn.datasets import fetch_california_housing
        X, y = fetch_california_housing(as_frame=True, return_X_y=True)
    elif dataset == 'diabetes':
        from sklearn.datasets import load_diabetes
        X, y = load_diabetes(as_frame=True, return_X_y=True)
    elif dataset == 'iris':
        # only for testing
        from sklearn.datasets import load_iris
        X, y = load_iris(as_frame=True, return_X_y=True)
    

    return X, y


def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--dataset', default='california', type=str)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--max_epochs', default=600, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--mask_ratio', default=0.5, type=float, help='Masking ratio (percentage of removed patches).')
    parser.add_argument('--embed_dim', default=32, type=int, help='embedding dimensions')
    parser.add_argument('--depth', default=6, type=int, help='encoder depth')
    parser.add_argument('--decoder_depth', default=4, type=int, help='decoder depth')
    parser.add_argument('--num_heads', default=4, type=int, help='number of heads')
    parser.add_argument('--mlp_ratio', default=4., type=float, help='mlp ratio')
    parser.add_argument('--encode_func', default='linear', type=str, help='encoding function')

    parser.add_argument('--norm_field_loss', default=False,
                        help='Use (per-patch) normalized field as targets for computing loss')
    parser.set_defaults(norm_field_loss=False)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05, help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=None, metavar='LR', help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N', help='epochs to warmup LR')

    ###### change this path
    parser.add_argument('--path', default='/data/tianyu/remasker/', type=str, help='dataset path')
    parser.add_argument('--exp_name', default='test', type=str, help='experiment name')

    # Dataset parameters
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--seed', default=666, type=int)

    parser.add_argument('--overwrite', default=True, help='whether to overwrite default config')
    parser.add_argument('--pin_mem', action='store_false')

    # distributed training parameters
    return parser

if __name__ == '__main__':
    
    X = torch.tensor([[1, 2, 3, 4]], dtype=torch.float32)
    X = X.unsqueeze(1)
    mask_embed = ActiveEmbed(4, 6)
    print(mask_embed(X).shape)
