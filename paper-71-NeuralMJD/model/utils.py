import torch
import torch.nn as nn
import numpy as np

def get_emb(sin_inp):
    """
    Gets a base embedding for one dimension with sin and cos intertwined
    """
    emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
    return torch.flatten(emb, -2, -1)


class PositionalEmbedding(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        num_channels = int(np.ceil(num_channels / 2) * 2)  # Ensures even number
        self.register_buffer("inv_freq", 1.0 / (10000 ** (torch.arange(0, num_channels, 2).float() / num_channels)))
        self.channels = num_channels

    def forward(self, tensor):
        values = tensor.unsqueeze(-1).to(torch.float32)
        pos_enc = torch.zeros(*tensor.shape, self.channels, device=tensor.device, dtype=torch.float32)
        pos_enc[..., 0::2] = torch.sin(values * self.inv_freq)  # Even indices
        pos_enc[..., 1::2] = torch.cos(values * self.inv_freq)  # Odd indices
        return pos_enc
    

def mask_nodes(nodes, node_flags, value=0.0, in_place=True, along_dim=None):
    """
    Masking out node embeddings according to node flags.
    @param nodes:        [B, N] or [B, N, D] by default, [B, *, N, *] if along_dim is specified
    @param node_flags:   [B, N] or [B, N, N]
    @param value:        scalar
    @param in_place:     flag of in place operation
    @param along_dim:    along certain specified dimension
    @return NODES:       [B, N] or [B, N, D]
    """
    if len(node_flags.shape) == 3:
        # raise ValueError("node_flags should be [B, N] or [B, N, D]")
        # if node_flags is [B, N, N], then we don't apply any mask
        return nodes
    elif len(node_flags.shape) == 2:
        if along_dim is None:
            # mask along the second dimension by default
            if len(nodes.shape) == 2:
                pass
            elif len(nodes.shape) == 3:
                node_flags = node_flags.unsqueeze(-1)  # [B, N, 1]
            else:
                raise NotImplementedError
        else:
            assert nodes.size(along_dim) == len(node_flags)
            shape_ls = list(node_flags.shape)
            assert len(shape_ls) == 2
            for i, dim in enumerate(nodes.shape):
                if i == 0:
                    pass
                else:
                    if i < along_dim:
                        shape_ls.insert(1, 1)  # insert 1 at the second dim
                    elif i == along_dim:
                        assert shape_ls[i] == dim  # check the length consistency
                    elif i > along_dim:
                        shape_ls.insert(len(shape_ls), 1)  # insert 1 at the end
            node_flags = node_flags.view(*shape_ls)  # [B, *, N, *]

        if in_place:
            nodes.masked_fill_(torch.logical_not(node_flags), value)
        else:
            nodes = nodes.masked_fill(torch.logical_not(node_flags), value)
    else:
        raise NotImplementedError
    return nodes


def mask_adjs(adjs, node_flags, value=0.0, in_place=True, col_only=False):
    """
    Masking out adjs according to node flags.
    @param adjs:        [B, N, N] or [B, C, N, N]
    @param node_flags:  [B, N] or [B, N, N]
    @param value:       scalar
    @param in_place:    flag of in place operation
    @param col_only:    masking in the column direction only
    @return adjs:       [B, N, N] or [B, C, N, N]
    """
    # assert node_flags.sum(-1).gt(2-1e-5).all(), f"{node_flags.sum(-1).cpu().numpy()}, {adjs.cpu().numpy()}"
    if len(node_flags.shape) == 2:
        # mask adjs by columns and/or by rows, [B, N] shape
        if len(adjs.shape) == 4:
            node_flags = node_flags.unsqueeze(1)  # [B, 1, N]
        if in_place:
            if not col_only:
                adjs.masked_fill_(torch.logical_not(node_flags).unsqueeze(-1), value)
            adjs.masked_fill_(torch.logical_not(node_flags).unsqueeze(-2), value)
        else:
            if not col_only:
                adjs = adjs.masked_fill(torch.logical_not(node_flags).unsqueeze(-1), value)
            adjs = adjs.masked_fill(torch.logical_not(node_flags).unsqueeze(-2), value)
    elif len(node_flags.shape) == 3:
        # mask adjs element-wisely, [B, N, N] shape
        assert node_flags.size(1) == node_flags.size(2) and node_flags.size(1) == adjs.size(2)
        assert not col_only
        if len(adjs.shape) == 4:
            node_flags = node_flags.unsqueeze(1)  # [B, 1, N, N]
        if in_place:
            adjs.masked_fill_(torch.logical_not(node_flags), value)  # [B, N, N] or [B, C, N, N]
        else:
            adjs = adjs.masked_fill(torch.logical_not(node_flags), value)  # [B, N, N] or [B, C, N, N]
    return adjs


class FlexIdentity(nn.Module):
    """
    Flexibly applies an identity function to the input.
    borrowed from DiT
    """
    def __init__(self, constant_output = None):
        super().__init__()
        self.constant_output = constant_output

    def forward(self, x, *args, **kwargs):
        if self.constant_output is None:
            return x
        else:
            return torch.empty_like(x).fill_(self.constant_output)

