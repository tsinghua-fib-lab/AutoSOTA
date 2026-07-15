import os
import time
import math
import logging
from typing import Optional

import torch
import numpy as np
import torch.nn as nn
from torch import Tensor
from torch.nn import Module
from torch.nn.functional import pad


__all__ = [
    "safe_cholesky_inv",
    "AdaOBCHandle",
    "CAPHandle",
    "CAPNMHandle"
]


_LOGGER = logging.getLogger(__name__)


def safe_cholesky_inv(X: Tensor, damp: float = 1e-2):
    try:
        return torch.cholesky_inverse(torch.linalg.cholesky(X))
    except RuntimeError:
        reg = (damp * torch.diag(X).mean()) * torch.eye(X.shape[0], device=X.device)
        return torch.cholesky_inverse(torch.linalg.cholesky(X + reg))


class AdaOBCHandle:

    def __init__(
        self, 
        layer: Module,
        num_calibration_samples: int,
        blocks_in_parallel: Optional[int] = -1,
        damp: float = 0.0,
        verbose: bool = False
    ) -> None:
        self.layer = layer
        # set params
        self.num_calibration_samples = num_calibration_samples
        self.blocks_in_parallel = blocks_in_parallel
        self.damp = damp
        self.verbose = verbose
        # set weight
        self.weight = layer.weight
        # get weight
        self.device  = self.weight.device
        # convert self.weight to the matrix form (d_out, d_in)
        self.dim_out = self.weight.shape[0]
        self.dim_in  = np.prod(self.weight.shape[1:])
        # init hessian
        self.H = None
        # init the loss evolution
        self.losses = None
        # init weight traces
        self.traces = None


    def update_H(self, inp: Tensor) -> None:
        # allocate memory (if not initialized)
        if self.H is None:
            self.H = torch.zeros((self.dim_in, self.dim_in), device=self.device)
        # unfold inp if needed
        if isinstance(self.layer, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            unfold = nn.Unfold(
                self.layer.kernel_size,
                dilation=self.layer.dilation,
                padding=self.layer.padding,
                stride=self.layer.stride
            )
            inp = unfold(inp)
            inp = inp.permute([1, 0, 2])
            inp = inp.flatten(1)
        else:
            inp = inp.view(-1, inp.shape[-1])
        self.H += (2 / self.num_calibration_samples) * inp.T @ inp


    def prepare(self) -> None:
        self.weight = self.layer.weight.data.clone()
        if isinstance(self.layer, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            self.weight = self.weight.flatten(1)
        # if the entire input is 0 -> channel is dead and doesn't contribute
        dead = torch.diag(self.H) == 0
        self.H[dead, dead] = 1
        self.weight[:, dead] = 0
        # prepare losses
        self.losses = torch.zeros((self.dim_out, self.dim_in + 1), device=self.device)
        # prepare traces
        self.traces = torch.zeros((self.dim_in + 1, self.dim_out, self.dim_in), device='cpu')


    def prepare_batch(self, i_start, i_end) -> tuple[Tensor]:
        W_batch = self.weight[i_start:i_end, :]
        mask_batch = torch.zeros_like(W_batch).bool()
        return W_batch, mask_batch


    def prepare_batch_sparse(self, W_batch, mask_batch): 
        min_zeros = torch.sum((W_batch == 0), dim=1).min().item()
        # temporary hessian
        H_inv_batch = torch.empty((W_batch.shape[0], *self.H.shape), device=self.device)
        for i in range(W_batch.shape[0]):
            zero_ids = (W_batch[i] == 0)
            H_cur = self.H.clone()
            H_cur[zero_ids, :] = 0
            H_cur[:, zero_ids] = 0
            H_cur[zero_ids, zero_ids] = 1
            # invert
            H_inv_batch[i] = safe_cholesky_inv(H_cur)
            mask_batch[i, torch.nonzero(zero_ids, as_tuple=True)[0][:min_zeros]] = True

        return H_inv_batch, min_zeros


    def run(self) -> None:
        # prepare all
        self.prepare()

        _start = time.perf_counter()

        if self.blocks_in_parallel < 0:
            self.blocks_in_parallel = self.dim_out

        for i_start in range(0, self.dim_out, self.blocks_in_parallel):
            i_end = min(i_start + self.blocks_in_parallel, self.dim_out)
            blocks_in_parallel = i_end - i_start
            block_ids = torch.arange(blocks_in_parallel, device=self.device)
            # prepare batch 
            W_batch, mask_batch = self.prepare_batch(i_start, i_end)
            H_inv_batch, min_nnz = self.prepare_batch_sparse(W_batch, mask_batch) 
            # init weight traces
            trace = torch.zeros((self.dim_in + 1, i_end - i_start, self.dim_in), device=self.device)
            trace[:(min_nnz + 1), :, :] = W_batch      

            for zeros in range(min_nnz + 1, self.dim_in + 1):
                H_inv_batch_diag = torch.diagonal(H_inv_batch, dim1=1, dim2=2)
                scores = (W_batch ** 2) / H_inv_batch_diag
                scores[mask_batch] = float('inf')
                pruned_id = torch.argmin(scores, 1)
                self.losses[i_start: i_end, zeros] = scores[block_ids, pruned_id]
                row = H_inv_batch[block_ids, pruned_id, :]
                d = H_inv_batch_diag[block_ids, pruned_id]
                W_batch -= row * (W_batch[block_ids, pruned_id] / d).unsqueeze(1)
                mask_batch[block_ids, pruned_id] = True
                W_batch[mask_batch] = 0
                trace[zeros, :, :] = W_batch
                # do not update on the last iteration
                if zeros == self.dim_in:
                    break
                row /= torch.sqrt(d).unsqueeze(1)
                H_inv_batch -= torch.bmm(row.unsqueeze(2), row.unsqueeze(1))

            self.losses[i_start: i_end, :] /= 2
            self.traces[:, i_start: i_end, :] = trace.cpu()

            torch.cuda.synchronize()

        _end = time.perf_counter()

        if self.verbose:
            _LOGGER.info(f'Preparation of losses and traces took {(_end - _start):.2f} s')


    def get_pruning_database(self, sparsities: np.ndarray) -> Tensor:
        losses = self.losses[:, 1:].reshape(-1)
        order = torch.argsort(losses)
        Ws = torch.zeros((len(sparsities), self.dim_out, self.dim_in), device=self.device)
        cum_losses = [0] * len(sparsities)

        for i in range(self.dim_out):
            for j, sparsity in enumerate(sparsities):
                count = int(math.ceil(self.dim_out * self.dim_in * sparsity))
                num_zeros_in_row = torch.sum(
                    torch.div(order[:count], self.dim_in, rounding_mode='trunc') == i
                ).item()
                cum_losses[j] += torch.sum(self.losses[i, :(num_zeros_in_row + 1)]).item()
                Ws[j, i, :] = self.traces[num_zeros_in_row, i, :].to(self.device)
        
        if self.verbose:
            for sparsity, cum_loss in zip(sparsities, cum_losses):
                _LOGGER.info(f'Sparsity: {sparsity:.3f} / Loss: {cum_loss:.4f}')

        # free memory
        self.free()

        return Ws


    def free(self) -> None:
        self.H = None
        self.losses = None
        self.traces = None
        torch.cuda.empty_cache()


class CAPHandle:

    def __init__(
        self, 
        weight: Tensor,
        blocks_in_parallel: int = -1,
        backup_path: Optional[str] = None,
        device: Optional[str] = None,
        verbose: bool = False,
    ) -> None:
        self.verbose = verbose
        # by default model device is the weight device
        self.device = (weight.device, device)[device is not None]
        self.blocks_in_parallel = blocks_in_parallel
        # one can store weight traces on drive instead of RAM an load when needed
        self.backup_path = backup_path
        # convert weight to the matrix form (d_out, d_in)
        self.dim_out = weight.shape[0]
        self.dim_in  = np.prod(weight.shape[1:])
        # backup original shape
        self.shape_orig = weight.shape
        # init Finv
        self.Finv = None
        # init weight
        self.weight = weight
        # init the loss evolution
        self.losses = None
        # init weight traces
        self.traces = None

    
    def set_Finv(self, Finv: Tensor):
        assert len(Finv.shape) == 3 and Finv.shape[1] == Finv.shape[2], \
            "Finv has to be of shape (num_blocks, block_size, block_size)"
        self.Finv = Finv.to(self.device)
        self.dim_in  = self.Finv.shape[1]
        # reshape weight to match Finv -> (-1, block_size)
        self.weight = self.weight.reshape(-1, self.dim_in).to(self.device)
        self.dim_out = self.weight.shape[0]


    def prepare(self) -> None:
        # prepare losses
        self.losses = torch.zeros((self.dim_out, self.dim_in), device=self.device)
        # prepare traces
        self.traces = torch.zeros((self.dim_in + 1, self.dim_out, self.dim_in), device='cpu')


    def prepare_batch(self, i_start, i_end) -> tuple:
        W_batch = self.weight[i_start: i_end, :]
        M_batch = torch.zeros_like(W_batch).bool()
        Hinv_batch = self.Finv[i_start: i_end, :]
        # get minimum number of zeros in a row
        min_zeros = torch.sum((W_batch == 0), dim=1).min().item()
        for i in range(W_batch.shape[0]):
            zero_ids = (W_batch[i] == 0)
            M_batch[i, torch.nonzero(zero_ids, as_tuple=True)[0][:min_zeros]] = True
        return W_batch, M_batch, Hinv_batch, min_zeros


    def run(self) -> None:
        # prepare all
        self.prepare()

        _start = time.perf_counter()

        if self.blocks_in_parallel < 0:
            self.blocks_in_parallel = self.dim_out

        for i_start in range(0, self.dim_out, self.blocks_in_parallel):
            i_end = min(i_start + self.blocks_in_parallel, self.dim_out)
            # get number of blocks in parallel
            blocks_in_parallel = i_end - i_start
            block_ids = torch.arange(blocks_in_parallel, device=self.device)
            cur_block_ids = block_ids + i_start
            # prepare batch 
            W_batch, M_batch, H_inv_batch, min_nnz = self.prepare_batch(i_start, i_end)
            # init weight traces
            trace = torch.zeros((self.dim_in + 1, i_end - i_start, self.dim_in), device=self.device)
            trace[:(min_nnz + 1), :, :] = W_batch
            # get list of current losses
            cur_losses = torch.zeros(blocks_in_parallel, device=self.device)  
            for zeros in range(min_nnz + 1, self.dim_in + 1):
                H_inv_batch_diag = torch.diagonal(H_inv_batch, dim1=1, dim2=2)
                scores = (W_batch ** 2) / H_inv_batch_diag
                scores[M_batch] = float('inf')
                min_scores, pruned_id = torch.min(scores, 1)
                cur_losses += min_scores
                self.losses[cur_block_ids, pruned_id] = cur_losses
                row = H_inv_batch[block_ids, pruned_id, :]
                d   = H_inv_batch_diag[block_ids, pruned_id]
                W_batch -= row * (W_batch[block_ids, pruned_id] / d).unsqueeze(1)
                M_batch[block_ids, pruned_id] = True
                W_batch[M_batch] = 0
                trace[zeros, :, :] = W_batch
                # do not update on the last iteration
                if zeros == self.dim_in:
                    break
                row /= torch.sqrt(d).unsqueeze(1)
                H_inv_batch -= torch.bmm(row.unsqueeze(2), row.unsqueeze(1))

            self.losses[i_start: i_end, :] /= 2
            self.traces[:, i_start: i_end, :] = trace.cpu()

            torch.cuda.synchronize()

        _end = time.perf_counter()

        if self.verbose:
            _LOGGER.info(f'Preparation of losses and traces took {(_end - _start):.2f} s')

        # dump weights to drive
        if self.backup_path:
            torch.save(self.traces, self.backup_path)
            self.traces = None


    def get_pruning_database(self, sparsities: np.ndarray) -> list[Tensor]:
        # load traces from drive
        if self.backup_path:
            self.traces = torch.load(self.backup_path)

        sorted_losses, _ = torch.sort(self.losses.view(-1))
        # prepare list of weight for every sparsity level of interest
        Ws = [torch.zeros((self.dim_out, self.dim_in), device=self.device) for _ in sparsities]
        for i, sparsity in enumerate(sparsities):
            num_zeros = int(math.ceil(self.dim_out * self.dim_in * sparsity))
            # loss threshold
            loss_thr = sorted_losses[num_zeros]
            for row in range(self.dim_out):
                num_zeros_in_row = torch.count_nonzero(self.losses[row, :] <= loss_thr)
                Ws[i][row, :] = self.traces[num_zeros_in_row, row, :]

            Ws[i] = Ws[i].reshape(self.shape_orig)

        # free memory
        self.free()

        return Ws

    def free(self) -> None:
        self.Finv = None
        self.losses = None
        self.traces = None
        if self.backup_path:
            os.remove(self.backup_path)
        torch.cuda.empty_cache()

    
class CAPNMHandle(CAPHandle):

    def __init__(
        self, 
        weight: Tensor,
        n: int,
        m: int, 
        blocks_in_parallel: int = -1,
        backup_path: Optional[str] = None,
        verbose: bool = False,
        device: str = 'cpu'
    ) -> None:
        super().__init__(
            weight, 
            blocks_in_parallel, 
            backup_path,
            device,
            verbose
        )
        assert self.dim_in % self.m == 0, "Block size in N:M has to divide the inner dim."
        self.n = n
        self.m = m


    def prepare(self) -> None:
        # prepare losses
        self.losses = torch.full((self.dim_out, self.dim_in), fill_value=torch.inf, device=self.device)


    def run(self) -> None:
        # prepare all
        self.prepare()

        _start = time.perf_counter()

        if self.blocks_in_parallel < 0:
            self.blocks_in_parallel = self.dim_out

        for i_start in range(0, self.dim_out, self.blocks_in_parallel):
            i_end = min(i_start + self.blocks_in_parallel, self.dim_out)
            # get current batch size
            blocks_in_parallel = i_end - i_start
            block_ids = torch.arange(blocks_in_parallel, device=self.device)
            cur_block_ids = block_ids + i_start
            # prepare batch 
            W_batch, M_batch, H_inv_batch, min_nnz = self.prepare_batch(i_start, i_end)
            # init buckets
            buckets = torch.zeros((blocks_in_parallel, self.dim_in // self.m, 1), device=self.device)
            # get list of current losses
            cur_losses = torch.zeros(blocks_in_parallel, device=self.device)  
            for zeros in range(min_nnz + 1, self.dim_in + 1):
                H_inv_batch_diag = torch.diagonal(H_inv_batch, dim1=1, dim2=2)
                scores = (W_batch ** 2) / H_inv_batch_diag
                # mask of filled buckets
                M_bucket = (buckets >= self.n).repeat(1, 1, self.m).view(blocks_in_parallel, -1)
                scores[M_batch | M_bucket] = float('inf')
                min_scores, pruned_id = torch.min(scores, 1)
                cur_losses += min_scores
                self.losses[cur_block_ids, pruned_id] = cur_losses
                row = H_inv_batch[block_ids, pruned_id, :]
                d   = H_inv_batch_diag[block_ids, pruned_id]
                W_batch -= row * (W_batch[block_ids, pruned_id] / d).unsqueeze(1)
                M_batch[block_ids, pruned_id] = True
                # fill buckets
                buckets[block_ids, torch.div(pruned_id, self.m, rounding_mode='floor')] += 1 
                # do not update on the last iteration
                if zeros == int(self.dim_in * (self.n / self.m)):
                    break
                row /= torch.sqrt(d).unsqueeze(1)
                H_inv_batch -= torch.bmm(row.unsqueeze(2), row.unsqueeze(1))

            W_batch[M_batch] = 0
            self.losses[i_start: i_end, :] /= 2

            torch.cuda.synchronize()

        _end = time.perf_counter()

        if self.verbose:
            _LOGGER.info(f'Preparation of losses took {(_end - _start):.2f} s')

        # reshape to original shape
        self.weight = self.weight.reshape(self.shape_orig)

    def free(self) -> None:
        self.Finv = None
        self.losses = None
        self.traces = None
        if self.backup_path:
            os.remove(self.backup_path)
        torch.cuda.empty_cache()


class FastCAPHandle:

    def __init__(
        self,
        weight: Tensor,
        block_size: Optional[int] = None,
        damp: float = 0.0,
    ) -> None:
        self.block_size = block_size
        self.damp = damp
        # backup dtype
        self.dtype = weight.dtype
        # backup device
        self.device= weight.device
        # backup shape
        self.shape = weight.shape
        # set default padding value
        self.padding = 0
        self.weight = weight
        # flag if pre step was performed
        self.pre_step_completed = False
        # set default sparsity
        self.sparsity = None

    def set_F(self, F: Tensor):
        assert len(F.shape) == 2 and F.shape[0] == F.shape[1], \
            "F has to be of shape (block_size, block_size)"
        self.F = F.to(self.device)
        self.fisher_block_size  = self.F.shape[1]
        # reshape weight to match Finv -> (-1, block_size)
        self.weight = self.weight.reshape(-1, self.fisher_block_size).to(self.device)
        self.num_fisher_blocks = self.weight.shape[0]
        # set block size to fisher_block_size if not set
        self.block_size = self.block_size or self.fisher_block_size

    def _prepare_weight(self, weight: Tensor) -> Tensor:
        if weight.numel() % self.fisher_block_size != 0:
            self.padding = self.fisher_block_size - weight.numel() % self.fisher_block_size
            weight = pad(weight.view(-1), (0, self.padding))
        return weight.reshape(-1, self.fisher_block_size)

    def _reshape_to_orig_shape(self, weight: Tensor) -> Tensor:
        # remove padding
        if self.padding != 0:
            weight = weight.view(-1)[:-self.padding]
        return weight.reshape(self.shape).to(self.dtype)

    def set_sparsity(self, sparsity: float):
        self.sparsity = sparsity

    @torch.no_grad()
    def prepare_data(self):
        w = self.weight.clone()
        # get columns with all zeros
        zero_cols = torch.nonzero(w.eq(0).all(dim=0))
        # get number of zero structures
        num_zeros = len(zero_cols)
        F = self.F
        # mask rows with zero input channels
        F[zero_cols, :] = 0
        F[:, zero_cols] = 0
        F[zero_cols, zero_cols] = 1
        # invert
        F = safe_cholesky_inv(F)
        F_inv_cho = torch.linalg.cholesky(F, upper=True)
        return w, F_inv_cho, num_zeros

    @torch.no_grad()
    def pruning_pre_step(self) -> None:
        """
        Preparatory step with hessian regularization and weight reshaping.
        """
        assert self.F is not None, \
            "One has to process at least one sample of calibration data to run pruning"
        self.weight = self._prepare_weight(self.weight)
        # get ids of pruned channels
        pruned_ids = torch.diag(self.F) == 0
        self.F[pruned_ids, pruned_ids] = 1
        # Hessian regularization
        damp = self.damp * torch.diag(self.F).mean()
        self.F.add_(torch.eye(self.F.shape[0], device=self.F.device), alpha=damp)
        self.weight[:, pruned_ids] = 0
        # flag pre step as completed
        self.pre_step_completed = True

    @torch.no_grad()
    def run(self) -> None:
        # run preparatory step
        self.pruning_pre_step()
        assert self.sparsity is not None
        fisher_block_size, block_size = self.fisher_block_size, self.block_size
        # prepare weight and Cholesky of F^{-1}
        w, F_inv_cho, nzeros = self.prepare_data()
        # iterate over columns
        for c1 in range(nzeros, fisher_block_size, block_size):
            c2 = min(c1 + block_size, fisher_block_size)
            ncols = c2 - c1 # number of columns
            w_blk = w[:, c1:c2].clone() # column-wise weight slice
            res = torch.zeros_like(w_blk)
            errs = torch.zeros_like(w_blk)
            losses_blk = torch.zeros_like(w_blk)
            F_inv_cho_blk = F_inv_cho[c1:c2, c1:c2]
            # 1) score computation
            scores = w_blk ** 2 / F_inv_cho_blk.diag().reshape(1, -1) ** 2
            thr, _ = torch.kthvalue(scores.view(-1), round(w_blk.numel() * self.sparsity))
            mask = scores > thr
            # 2) iterate over block
            for i in range(ncols):
                w_ci = w_blk[:, i]
                d = F_inv_cho_blk[i, i]

                q = w_ci.clone()
                q[~mask[:, i]] = 0

                res[:, i] = q
                err = (w_ci - q) / d
                losses_blk[:, i] = err ** 2
                
                w_blk[:, i:].addr_(err, F_inv_cho_blk[i, i:], alpha=-1)
                errs[:, i] = err
            # 3) update the weights after block
            w[:, c1:c2] = res
            w[:, c2:].addmm_(errs, F_inv_cho[c1:c2, c2:], alpha=-1)

        self.weight = self._reshape_to_orig_shape(self.weight)

    def free(self):
        """
        Free all allocated data.
        """
        self.F = None
        self.padding = 0
        torch.cuda.empty_cache()
