import time
import math
import torch
import torch.nn as nn

from quant_utils import *

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


class Aespa:
    def __init__(self, layer):
        self.layer = layer        
        self.quantizer = None
        self.H = None
        self.cov_G = None
        self.dXXT = None
        self.dXXT_per_qhead = None
        self.fp_inps = [] 

    def cache_fp_block_input(self, _, inp, out): 
        inp = inp[0].data
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        if len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1]))  # shape = [BL, d]
        inp = inp.t()  # shape = [d, BL]
        self.fp_inps.append(inp.cpu())

    def compute_cov_in_batch(self, _, inp, out):
        if self.H is None:
            self.H = 0
            self.n_data_in = 0
            if len(self.fp_inps) > 0: 
                self.dXXT = 0

        inp = inp[0].data
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        if len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1]))  # shape = [BL, d]
        inp = inp.t()  # shape = [d, BL]

        n_current = inp.shape[-1]
        self.H *= self.n_data_in / (self.n_data_in + n_current)
        if len(self.fp_inps) > 0: 
            self.dXXT *= self.n_data_in / (self.n_data_in + n_current)
        self.n_data_in += n_current
        inp = math.sqrt(2 / self.n_data_in) * inp.float()
        self.H += inp.matmul(inp.t())

        if len(self.fp_inps) > 0: 
            fp_inp = self.fp_inps.pop(0).to(inp.device).float() 
            fp_inp = math.sqrt(2 / self.n_data_in) * fp_inp
            dX = fp_inp - inp
            self.dXXT += dX.matmul(inp.t())

    def compute_cov_out_batch(self, _, inps, outs, n_heads):
        if not hasattr(self, "cov_out"):
            self.cov_out = 0
            self.n_data_out = 0
        
        head_dim = outs.shape[-1] // n_heads
        outs = outs.view(outs.shape[0], outs.shape[1], n_heads, head_dim).transpose(1, 2).contiguous()  # [B, H, L, d_h]
        outs = outs.transpose(0, 1).view(n_heads, -1, head_dim).transpose(-1, -2).contiguous()  # [H, d_h, BL]

        n_current = outs.shape[-1]
        self.cov_out *= self.n_data_out / (self.n_data_out + n_current)
        self.n_data_out += n_current
        outs = math.sqrt(2 / self.n_data_out) * outs.float()
        self.cov_out += outs @ outs.transpose(-1, -2)

    def refine_quant_params(self, use_zfold: bool, hyperparams: dict):
        assert self.quantizer is not None, "Quantizer should be defined first."
        assert self.H is not None, "Hessian should be computed first."

        W = self.layer.weight.data.clone()
        if not self.quantizer.ready():
            self.quantizer.find_params(W)
        
        W, H = W.float(), self.H.clone()
        W, H = filter_dead_neuron(W, H, replace=hyperparams['replace'], percdamp=hyperparams['percdamp'], apply_damping=True)
        
        tick = time.time()
        if use_zfold:
            return refine_qparams_zfold({self.name: self}, [self.name], hyperparams)
        else:
            scale, zero, zeta = self.quantizer.scale.view([-1, 1]), self.quantizer.zero.view([-1, 1]), self.quantizer.zeta.view([1, -1])
            n_bits = self.quantizer.nbits

            # compute initial loss perturbation incurred by quantization
            loss_perturb_before = compute_loss_perturb(W, scale, zero, zeta, n_bits, H)

            # update scale and zero-point
            scale, zero = find_quant_params(W, zeta, n_bits, self.quantizer.sym, H)

            # compute loss perturbation after update
            loss_perturb_after = compute_loss_perturb(W, scale, zero, zeta, n_bits, H)

            delta_loss_improvement = loss_perturb_before - loss_perturb_after
            
            self.quantizer.scale.data = scale.view(self.quantizer.scale.shape)
            self.quantizer.zero.data = zero.view(self.quantizer.zero.shape)

            return delta_loss_improvement, 1, time.time() - tick

    def quant(self, opts:dict, hyperparams: dict):
        assert self.quantizer is not None, "Quantizer should be defined first."
        assert self.H is not None, "Hessian should be computed first."

        W = self.layer.weight.data.clone()
        if not self.quantizer.ready():
            self.quantizer.find_params(W, weight=True)
        orig_w_shape, orig_w_dtype = W.shape, W.dtype
        W = W.float()

        # Quant. Params.
        scale, zero, zeta = self.quantizer.scale.view([-1, 1]), self.quantizer.zero.view([-1, 1]), self.quantizer.zeta.view([1, -1])
        n_bits = self.quantizer.nbits

        H, cov_G = self.H.clone(), self.cov_G.clone() if self.cov_G is not None else None

        # pre-processing: detect dead neurons BEFORE filter_dead_neuron replaces zeros
        if opts['comp_method'] == 'GPTAQ': 
            dXXT = self.dXXT.clone()
            if len(H.shape) == 2:
                dead = torch.diag(H) == 0
                dXXT[:, dead] = 0
                dXXT[dead, :] = 0
            elif len(H.shape) == 3:
                for h_idx in range(H.shape[0]):
                    dead_h = torch.diag(H[h_idx]) == 0
                    dXXT[h_idx, :, dead_h] = 0
                    dXXT[h_idx, dead_h, :] = 0
                    if hasattr(self, 'dXXT_per_qhead') and self.dXXT_per_qhead is not None:
                        n_qheads = self.dXXT_per_qhead.shape[0]
                        kv_group_size = n_qheads // H.shape[0]
                        for q in range(kv_group_size):
                            q_idx = h_idx * kv_group_size + q
                            self.dXXT_per_qhead[q_idx, :, dead_h] = 0
                            self.dXXT_per_qhead[q_idx, dead_h, :] = 0

        W, H = filter_dead_neuron(W, H, replace=hyperparams['replace'], percdamp=hyperparams['percdamp'], apply_damping=True)

        if len(H.shape) == 2:  # common Hessian for all heads
            H = H.unsqueeze(0)
        if opts['comp_method'] == 'GPTAQ' and len(dXXT.shape) == 2: 
            dXXT = dXXT.unsqueeze(0)

        num_heads = H.shape[0] if cov_G is None else cov_G.shape[0]
        hidden_size = W.shape[-1]
        head_dim = W.shape[0] // num_heads
        W = W.view(num_heads, head_dim, hidden_size)
        scale, zero = scale.view(num_heads, head_dim, 1), zero.view(num_heads, head_dim, 1)

        # initialize weight-rounding policy
        groupsize = opts['groupsize']
        clustersize = opts['clustersize']
        blocksize = opts['blocksize']
        comp_method = opts['comp_method']
        # Check group, cluster, and block size
        if clustersize == -1:
            clustersize = hidden_size
        if blocksize == -1:
            blocksize = hidden_size
        if groupsize == -1:
            groupsize = hidden_size
        if groupsize > 0 and hidden_size % groupsize != 0:
            raise ValueError(f'input dimension {hidden_size} must be divisible by groupsize {groupsize}')
        if groupsize > clustersize and groupsize % clustersize != 0:
            raise ValueError(f'groupsize {groupsize} must be greater than or equal to clustersize {clustersize} and divisible by clustersize')
        if groupsize < clustersize and clustersize % groupsize != 0:
            raise ValueError(f'clustersize {clustersize} must be greater than or equal to groupsize {groupsize} and divisible by groupsize')
        if (blocksize < clustersize) or (blocksize % clustersize != 0):
            raise ValueError(f'blocksize {blocksize} must be greater than or equal to clustersize {clustersize} and divisible by clustersize')
        
        # zeta reshape
        if groupsize > 0:
            zeta = zeta.view(1, hidden_size // groupsize, -1)
        else:
            zeta = zeta.view([1, 1, -1])

        if comp_method == 'GPTAQ':
            dXXT_per_qhead = self.dXXT_per_qhead.clone() if hasattr(self, 'dXXT_per_qhead') and self.dXXT_per_qhead is not None else None
            W_update, Q = self.GPTAQ(W, H, dXXT, cov_G, scale, zero, zeta, n_bits, opts, dXXT_per_qhead=dXXT_per_qhead)
        elif comp_method == 'GPTQ':
            W_update, Q = self.GPTQ(W, H, cov_G, scale, zero, zeta, n_bits, opts)
        else:
            raise ValueError(f'Invalid comp method: {comp_method}')
        print(f'|{self.i}: {self.name : <24}|GPU memory: {torch.cuda.max_memory_allocated("cuda") / 1024**3:.3f}\t|')

        # assign quantized (fake-quant) weights
        self.layer.weight.data = Q.reshape(orig_w_shape).to(orig_w_dtype)

    def GPTQ(self, W, H, cov_G, scale, zero, zeta, n_bits, opts):
        order_option = opts['order_option']
        learn_rounding = opts['learn_rounding']
        groupsize = opts['groupsize']
        blocksize = opts['blocksize']
        clustersize = opts['clustersize']

        # Handling negative cluster, and block size
        if clustersize == -1:
            clustersize = W.shape[-1]
        if blocksize == -1:
            blocksize = W.shape[-1]
        if groupsize == -1:
            groupsize = W.shape[-1]

        W, H = W.clone(), H.clone()
        n_columns = W.shape[-1]

        if order_option != 'none':
            num_heads, hidden_size = W.shape[0], H.shape[-1]
            if H.shape[0] == 1:  # Common Hessian for all heads
                W = W.view(1, -1, hidden_size)
            perm_multi_head = torch.zeros((H.shape[0], hidden_size), dtype=torch.int64, device=H.device)
            invperm_multi_head = torch.zeros_like(perm_multi_head)
            zeta_multi_head = torch.zeros((H.shape[0], *zeta.shape[1:]), device=zeta.device)
            zeta_flat = zeta.view(1,-1)
            for idx_head in range(H.shape[0]):
                if order_option == 'spin':
                    csz = clustersize if clustersize > 0 else H.shape[-1]
                    perm = self.spin_greedy_optimized(W[idx_head], H[idx_head], clustersize=csz)
                elif order_option == 'act':
                    perm = torch.argsort(torch.diag(H[idx_head]), descending=True)
                else:
                    raise ValueError(f'Invalid order option: {order_option}')
                invperm_multi_head[idx_head] = torch.argsort(perm)
                W[idx_head] = W[idx_head][:, perm]
                H[idx_head] = H[idx_head][perm][:, perm]
                zeta_perm = zeta_flat[:,perm].view(*zeta.shape)
                zeta_multi_head[idx_head] = zeta_perm[0]
                perm_multi_head[idx_head] = perm

            W = W.view(num_heads, -1, hidden_size)
            zeta = zeta_multi_head
        
        W_org = W.clone() if learn_rounding else None

        # Cholesky Decomposition
        Hinv = torch.zeros_like(H)
        for idx_head in range(H.shape[0]):
            Hinv[idx_head] = torch.linalg.cholesky(
                torch.cholesky_inverse(torch.linalg.cholesky(H[idx_head])), upper=True
            )

        num_heads = W.shape[0]
        W_update = torch.zeros_like(W)
        Q = torch.zeros_like(W)

        scale = torch.zeros_like(W)
        zero = torch.zeros_like(W)

        num_groups = H.shape[-1] // groupsize
        zeta_full = torch.zeros_like(W)
        for g in range(num_groups):
            zeta_full[..., g*groupsize : (g+1)*groupsize] = zeta[:, g:g+1, :]
        zeta = zeta_full
        
        for i1 in range(0, n_columns, blocksize):
            i2 = min(i1 + blocksize, n_columns)
            count = i2 - i1

            W1 = W[..., i1:i2].clone()
            Err1 = torch.zeros_like(W1)
            Hinv1 = Hinv[..., i1:i2, i1:i2]

            for start in range(0, count, clustersize):
                end = min(start + clustersize, count)

                W_cluster = W1[..., start:end]
                col_start = i1 + start
                col_end = i1 + end

                if col_start % groupsize == 0:
                    z_head = zeta[0:1]

                    if groupsize >= clustersize:
                        h_group = H[..., col_start:col_start+groupsize, col_start:col_start+groupsize]
                        hinv_group = Hinv[..., col_start:col_start+groupsize, col_start:col_start+groupsize]
                        z_group = z_head[..., (col_start//groupsize):((col_start)//groupsize+1), (col_start % groupsize):(col_start % groupsize + groupsize)]  # [1,1,groupsize]

                        W_group = W[..., col_start:(col_start+groupsize)]

                        if opts['loss_option'] == 'local':
                            current_scale, current_zero = find_quant_params(
                                W_group,
                                z_group,
                                n_bits,
                                self.quantizer.sym,
                                h_group,
                                cov_G=None,
                                Hinv=None,
                            )
                        elif opts['loss_option'] == 'global':
                            current_scale, current_zero = find_quant_params(
                                W_group,
                                z_group,
                                n_bits,
                                self.quantizer.sym,
                                h_group,
                                cov_G=None,
                                Hinv=hinv_group,
                            )
                        else:
                            raise ValueError(f'Invalid loss option: {opts["loss_option"]}')

                        current_scale = current_scale.view(W.shape[0], W.shape[1], -1)
                        current_zero = current_zero.view(W.shape[0], W.shape[1], -1)
                        scale[..., col_start:(col_start+groupsize)] = current_scale
                        zero[..., col_start:(col_start+groupsize)] = current_zero

                    elif groupsize < clustersize:
                        for i in range(start, end, groupsize):
                            end_group = min(i + groupsize, end)
                            h_group = H[..., col_start + i:col_start + end_group, col_start + i:col_start + end_group]
                            hinv_group = Hinv[..., col_start + i:col_start + end_group, col_start + i:col_start + end_group]
                            z_group = z_head[..., (i//groupsize):((i)//groupsize+1), (i % groupsize):(i % groupsize + groupsize)]

                            W_group = W[..., col_start + i:col_start + end_group]  # [num_heads, head_dim, group]
                            if opts['loss_option'] == 'local':
                                current_scale, current_zero = find_quant_params(
                                    W_group,
                                    z_group,
                                    n_bits,
                                    self.quantizer.sym,
                                    h_group,
                                    cov_G=None,
                                    Hinv=None,
                                )
                            elif opts['loss_option'] == 'global':
                                current_scale, current_zero = find_quant_params(
                                    W_group,
                                    z_group,
                                    n_bits,
                                    self.quantizer.sym,
                                    h_group,
                                    cov_G=None,
                                    Hinv=hinv_group,
                                )
                            else:
                                raise ValueError(f'Invalid loss option: {opts["loss_option"]}')

                            current_scale = current_scale.view(W.shape[0], W.shape[1], -1)
                            current_zero = current_zero.view(W.shape[0], W.shape[1], -1)
                            scale[..., col_start + i:col_start + end_group] = current_scale
                            zero[..., col_start + i:col_start + end_group] = current_zero

                if clustersize == 1:
                    if learn_rounding:
                        raise ValueError("learn_rounding is not supported for clustersize = 1")
                    
                    q_cluster = quantize_zfold(W_cluster, scale[..., col_start:col_end], zero[..., col_start:col_end], zeta[..., col_start:col_end], n_bits)
                    err_cluster = (W_cluster - q_cluster) / Hinv1[..., start, start].unsqueeze(-1).unsqueeze(-1)
                else:
                    Hinv_cluster = Hinv1[..., start:end, start:end]

                    if learn_rounding:
                        Hinv_cluster = Hinv1[..., start:end, start:end]
                        H_cluster = H[..., col_start:col_end, col_start:col_end]
                        W_org_cluster = W_org[..., col_start:col_end]
                        if opts['loss_option'] == 'local':
                            q_cluster = self.adaround(W_org_cluster, 
                                                    W_cluster, 
                                                    H_cluster, 
                                                    cov_G, 
                                                    scale[..., col_start:col_end], 
                                                    zero[..., col_start:col_end], 
                                                    zeta[..., col_start:col_end], 
                                                    n_bits, 
                                                    opts, 
                                                    Hinv=None,
                                                    correction=None)
                        elif opts['loss_option'] == 'global':
                            q_cluster = self.adaround(W_org_cluster, 
                                                    W_cluster, 
                                                    H_cluster, 
                                                    cov_G, 
                                                    scale[..., col_start:col_end], 
                                                    zero[..., col_start:col_end], 
                                                    zeta[..., col_start:col_end], 
                                                    n_bits, 
                                                    opts, 
                                                    Hinv=Hinv_cluster,
                                                    correction=None)
                        else:
                            raise ValueError(f'Invalid loss option: {opts["loss_option"]}')
                    else:
                        q_cluster = quantize_zfold(W_cluster, 
                                                   scale[..., col_start:col_end], 
                                                   zero[..., col_start:col_end], 
                                                   zeta[..., col_start:col_end], 
                                                   n_bits)

                    delta_cluster = W_cluster - q_cluster
                    err_cluster_T = torch.linalg.solve_triangular(
                        Hinv_cluster.transpose(-1, -2),
                        delta_cluster.transpose(-1, -2),
                        upper=False,
                    )
                    err_cluster = err_cluster_T.transpose(-1,-2)
                
                W_update[..., col_start:col_end] = W_cluster
                Q[..., col_start:col_end] = q_cluster
                Err1[..., start:end] = err_cluster

                # Intra-block compensation
                W1[..., end:] -= torch.matmul(err_cluster, Hinv1[..., start:end, end:])
            
            # Inter-block compensation
            W[..., i2:] -= torch.matmul(Err1, Hinv[..., i1:i2, i2:])
        
        if order_option != 'none':
            if H.shape[0] == 1:
                W_update = W_update.view(1, -1, hidden_size)
                Q = Q.view(1, -1, hidden_size)
            for idx_head in range(H.shape[0]):
                W_update[idx_head] = W_update[idx_head][:, invperm_multi_head[idx_head]]
                Q[idx_head] = Q[idx_head][:, invperm_multi_head[idx_head]]
            W_update = W_update.view(num_heads, -1, hidden_size)
            Q = Q.view(num_heads, -1, hidden_size)

        return W_update, Q

    def GPTAQ(self, W, H, dXXT, cov_G, scale, zero, zeta, n_bits, opts, dXXT_per_qhead=None):
        order_option = opts['order_option']
        learn_rounding = opts['learn_rounding']
        groupsize = opts['groupsize']
        blocksize = opts['blocksize']
        clustersize = opts['clustersize']
        alpha = 0.25
        # Handling negative cluster, and block size
        if clustersize == -1:
            clustersize = W.shape[-1]
        if blocksize == -1:
            blocksize = W.shape[-1]
        if groupsize == -1:
            groupsize = W.shape[-1]

        W, H = W.clone(), H.clone()
        dXXT = dXXT.clone()
        n_columns = W.shape[-1]
        num_heads, head_dim = W.shape[0], W.shape[1]

        if order_option != 'none':
            num_heads, hidden_size = W.shape[0], H.shape[-1]
            if H.shape[0] == 1:  # Common Hessian for all heads
                W = W.view(1, -1, hidden_size)
            perm_multi_head = torch.zeros((H.shape[0], hidden_size), dtype=torch.int64, device=H.device)
            invperm_multi_head = torch.zeros_like(perm_multi_head)
            zeta_flat = zeta.view(1, -1)
            zeta_multi_head = torch.zeros((H.shape[0], *zeta.shape[1:]), device=zeta.device)
            for idx_head in range(H.shape[0]):
                if order_option == 'spin':
                    csz = clustersize if clustersize > 0 else H.shape[-1]
                    perm = self.spin_greedy_optimized(W[idx_head], H[idx_head], clustersize=csz)
                elif order_option == 'act':
                    perm = torch.argsort(torch.diag(H[idx_head]), descending=True)
                else:
                    raise ValueError(f'Invalid order option: {order_option}')
                invperm_multi_head[idx_head] = torch.argsort(perm)
                W[idx_head] = W[idx_head][:, perm]
                H[idx_head] = H[idx_head][perm][:, perm]
                dXXT[idx_head] = dXXT[idx_head][perm][:, perm]
                if dXXT_per_qhead is not None:
                    kv_group_size = dXXT_per_qhead.shape[0] // dXXT.shape[0]
                    for q in range(kv_group_size):
                        q_idx = idx_head * kv_group_size + q
                        dXXT_per_qhead[q_idx] = dXXT_per_qhead[q_idx][perm][:, perm]
                zeta_perm = zeta_flat[:, perm].view(*zeta.shape)
                zeta_multi_head[idx_head] = zeta_perm[0]
                perm_multi_head[idx_head] = perm

            W = W.view(num_heads, -1, hidden_size)
            zeta = zeta_multi_head

            del perm, perm_multi_head, zeta_multi_head, zeta_flat, zeta_perm
            torch.cuda.empty_cache()

        W_org = W.clone() if learn_rounding else None

        # Cholesky Decomposition
        Hinv = torch.zeros_like(H)
        for idx_head in range(H.shape[0]):
            Hinv[idx_head] = torch.linalg.cholesky(
                torch.cholesky_inverse(torch.linalg.cholesky(H[idx_head])), upper=True
            )

        if dXXT_per_qhead is not None:
            n_qheads = dXXT_per_qhead.shape[0]
            n_kv_heads = H.shape[0]
            kv_group_size = n_qheads // n_kv_heads
            P = torch.zeros_like(Hinv)
            for kv_idx in range(n_kv_heads):
                P_group = torch.zeros_like(Hinv[kv_idx])
                for q in range(kv_group_size):
                    q_idx = kv_idx * kv_group_size + q
                    tmp_q = torch.matmul(dXXT_per_qhead[q_idx], Hinv[kv_idx].transpose(-1, -2))
                    tmp_q = torch.triu(tmp_q, diagonal=1)
                    P_group += torch.matmul(tmp_q, Hinv[kv_idx])
                P[kv_idx] = P_group / kv_group_size
            del dXXT_per_qhead
            P.mul_(alpha)
        else:
            tmp = torch.matmul(dXXT, Hinv.transpose(-1, -2))
            tmp = torch.triu(tmp, diagonal=1)
            P = torch.matmul(tmp, Hinv)
            P.mul_(alpha)
            del tmp
        del dXXT
        torch.cuda.empty_cache()
        
        W_update = torch.zeros_like(W)
        Q = torch.zeros_like(W)
        scale = torch.zeros_like(W)
        zero = torch.zeros_like(W)

        num_groups = H.shape[-1] // groupsize
        zeta_full = torch.zeros_like(W)
        for g in range(num_groups):
            zeta_full[..., g*groupsize : (g+1)*groupsize] = zeta[:, g:g+1, :]
        zeta = zeta_full
        del zeta_full
        torch.cuda.empty_cache()

        device = W.device
        H = H.cpu()
        Hinv = Hinv.cpu()
        P = P.cpu()
        torch.cuda.empty_cache()

        for i1 in range(0, n_columns, blocksize):
            i2 = min(i1 + blocksize, n_columns)
            count = i2 - i1

            W1 = W[..., i1:i2].clone()
            Err1 = torch.zeros_like(W1)
            Hinv1 = Hinv[..., i1:i2, i1:i2].to(device=device)
            P1 = P[..., i1:i2, i1:i2].to(device=device)

            for start in range(0, count, clustersize):
                end = min(start + clustersize, count)

                W_cluster = W1[..., start:end]
                col_start = i1 + start
                col_end = i1 + end               

                if col_start % groupsize == 0:
                    z_head = zeta[0:1]

                    if groupsize >= clustersize:
                        h_group = H[..., col_start:col_start+groupsize, col_start:col_start+groupsize].to(device=device)
                        H = H.cpu()
                        hinv_group = Hinv[..., col_start:col_start+groupsize, col_start:col_start+groupsize].to(device=device)
                        Hinv = Hinv.cpu()
                        z_group = z_head[..., (col_start//groupsize):((col_start)//groupsize+1), (col_start % groupsize):(col_start % groupsize + groupsize)]
                        
                        W_group = W[..., col_start:(col_start+groupsize)] 
                        correction = W_group @ P[..., col_start:col_start+groupsize, col_start:col_start+groupsize].to(device=device)
                        P = P.cpu()
                        torch.cuda.empty_cache()

                        if opts['loss_option'] == 'local':
                            current_scale, current_zero = find_quant_params(
                                W_group,
                                z_group,
                                n_bits,
                                self.quantizer.sym,
                                h_group,  
                                cov_G=None,
                                Hinv=None,
                                correction=None,
                            )
                        elif opts['loss_option'] == 'global':
                            current_scale, current_zero = find_quant_params(
                                W_group,
                                z_group,
                                n_bits,
                                self.quantizer.sym,
                                h_group,
                                cov_G=None,
                                Hinv=hinv_group,
                                correction=correction,
                            )
                        else:
                            raise ValueError(f'Invalid loss option: {opts["loss_option"]}')

                        current_scale = current_scale.view(W.shape[0], W.shape[1], -1)
                        current_zero = current_zero.view(W.shape[0], W.shape[1], -1)
                        scale[..., col_start:(col_start+groupsize)] = current_scale
                        zero[..., col_start:(col_start+groupsize)] = current_zero

                    elif groupsize < clustersize:
                        for i in range(start, end, groupsize):
                            end_group = min(i + groupsize, end)
                            h_group = H[..., col_start + i:col_start + end_group, col_start + i:col_start + end_group].to(device=device)
                            H = H.cpu()
                            hinv_group = Hinv[..., col_start + i:col_start + end_group, col_start + i:col_start + end_group].to(device=device)
                            Hinv = Hinv.cpu()
                            z_group = z_head[..., (i//groupsize):((i)//groupsize+1), (i % groupsize):(i % groupsize + groupsize)]

                            W_group = W[..., col_start + i:col_start + end_group]  # [num_heads, head_dim, group]
                            correction = W_group @ P[..., col_start + i:col_start + end_group, col_start + i:col_start + end_group].to(device=device)
                            P = P.cpu()
                            torch.cuda.empty_cache()

                            if opts['loss_option'] == 'local':
                                current_scale, current_zero = find_quant_params(
                                    W_group,
                                    z_group,
                                    n_bits,
                                    self.quantizer.sym,
                                    h_group,
                                    cov_G=None,
                                    Hinv=None,
                                    correction=None,
                                )
                            elif opts['loss_option'] == 'global':
                                current_scale, current_zero = find_quant_params(
                                    W_group,
                                    z_group,
                                    n_bits,
                                    self.quantizer.sym,
                                    h_group,
                                    cov_G=None,
                                    Hinv=hinv_group,
                                    correction=correction,
                                )
                            else:
                                raise ValueError(f'Invalid loss option: {opts["loss_option"]}')
                                
                            current_scale = current_scale.view(W.shape[0], W.shape[1], -1)
                            current_zero = current_zero.view(W.shape[0], W.shape[1], -1)
                            scale[..., col_start + i:col_start + end_group] = current_scale
                            zero[..., col_start + i:col_start + end_group] = current_zero

                if clustersize == 1:
                    if learn_rounding:
                        raise ValueError("learn_rounding is not supported for clustersize = 1")
                    
                    q_cluster = quantize_zfold(W_cluster, scale[..., col_start:col_end], zero[..., col_start:col_end], zeta[..., col_start:col_end], n_bits)
                    err_cluster = (W_cluster - q_cluster) / Hinv1[..., start, start].unsqueeze(-1).unsqueeze(-1)
                else:
                    Hinv_cluster = Hinv1[..., start:end, start:end]

                    if learn_rounding:
                        Hinv_cluster = Hinv1[..., start:end, start:end]
                        H_cluster = H[..., col_start:col_end, col_start:col_end].to(device=device)
                        H = H.cpu()
                        W_org_cluster = W_org[..., col_start:col_end]
                        correction = W_cluster @ P[..., col_start:col_end, col_start:col_end].to(device=device)
                        P = P.cpu()
                        torch.cuda.empty_cache()
                        if opts['loss_option'] == 'local':
                            q_cluster = self.adaround(W_org_cluster, 
                                                    W_cluster, 
                                                    H_cluster, 
                                                    cov_G, 
                                                    scale[..., col_start:col_end], 
                                                    zero[..., col_start:col_end], 
                                                    zeta[..., col_start:col_end], 
                                                    n_bits, 
                                                    opts, 
                                                    Hinv=None,
                                                    correction=None)
                        elif opts['loss_option'] == 'global':
                            q_cluster = self.adaround(W_org_cluster, 
                                                    W_cluster, 
                                                    H_cluster, 
                                                    cov_G, 
                                                    scale[..., col_start:col_end], 
                                                    zero[..., col_start:col_end], 
                                                    zeta[..., col_start:col_end], 
                                                    n_bits, 
                                                    opts, 
                                                    Hinv=Hinv_cluster,
                                                    correction=correction)
                        else:
                            raise ValueError(f'Invalid loss option: {opts["loss_option"]}')
                    else:
                        q_cluster = quantize_zfold(W_cluster, 
                                                   scale[..., col_start:col_end], 
                                                   zero[..., col_start:col_end], 
                                                   zeta[..., col_start:col_end], 
                                                   n_bits)

                    delta_cluster = W_cluster - q_cluster
                    err_cluster_T = torch.linalg.solve_triangular(
                        Hinv_cluster.transpose(-1, -2),
                        delta_cluster.transpose(-1, -2),
                        upper=False,
                    )
                    err_cluster = err_cluster_T.transpose(-1,-2)
                
                W_update[..., col_start:col_end] = W_cluster
                Q[..., col_start:col_end] = q_cluster
                Err1[..., start:end] = err_cluster

                # Intra-block compensation
                delta_W1 = torch.matmul(err_cluster, Hinv1[..., start:end, end:])
                delta_W1 -= torch.matmul(W_cluster, P1[..., start:end, end:])
                W1[..., end:] -= delta_W1
            
            # Inter-block compensation
            delta_W = torch.matmul(Err1, Hinv[..., i1:i2, i2:].to(device=device))
            delta_W -= torch.matmul(W1, P[..., i1:i2, i2:].to(device=device))            
            W[..., i2:] -= delta_W
        
        if order_option != 'none':
            if H.shape[0] == 1:
                W_update = W_update.view(1, -1, hidden_size)
                Q = Q.view(1, -1, hidden_size)
            for idx_head in range(H.shape[0]):
                W_update[idx_head] = W_update[idx_head][:, invperm_multi_head[idx_head]]
                Q[idx_head] = Q[idx_head][:, invperm_multi_head[idx_head]]
            W_update = W_update.view(num_heads, -1, hidden_size)
            Q = Q.view(num_heads, -1, hidden_size)

        return W_update, Q

    def adaround(self, W_org, W_update, H, cov_G, scale, zero, zeta, n_bits, opts: dict, Hinv=None, correction=None):
        lr, num_iters = opts['lr'], opts['num_iters']
        round_weight = opts['round_weight_qkv'] if self.name in ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"] else opts['round_weight']
        # Layer-dependent scaling: reduce rounding pressure for deeper layers
        if self.i is not None:
            if self.i < 4: layer_scale = 1.0
            elif self.i < 8: layer_scale = 0.7
            else: layer_scale = 0.5
            round_weight = round_weight * layer_scale

        scale = scale * zeta

        print_period = int(num_iters * 0.2)

        H_from_Hinv = None
        if Hinv is not None:
            eye = torch.eye(Hinv.shape[-1], device=Hinv.device, dtype=Hinv.dtype)
            H_from_Hinv = torch.cholesky_solve(eye, Hinv, upper=True)

        with torch.enable_grad():
            sigm = RectifiedSigmoid(-0.1, 1.1)
            sb = nn.Parameter(sigm.inverse(W_update / scale - torch.floor(W_update / scale)))
            optimizer = torch.optim.Adam([sb], lr=lr)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_iters, eta_min=lr*0.3)

            round_loss_func = RoundLoss(max_count=num_iters, b_range=(20, 2), decay_start=0.0, warmup=0.2, p_norm=2.0)

            for i in range(num_iters):
                q = torch.clamp(torch.floor(W_update / scale) + sigm(sb) + zero, 0, 2**n_bits-1)
                q = scale * (q - zero)
                e = q - W_org
                ue = q - W_update

                if correction is not None:
                    ue = ue - correction

                if Hinv is not None:
                    if cov_G is None:
                        recon_loss = ((ue @ H_from_Hinv) * ue).sum()
                    else:
                        recon_loss = (cov_G * (ue @ H_from_Hinv @ ue.transpose(-1, -2))).sum()
                else:
                    if cov_G is None:
                        recon_loss = ((e @ H) * e).sum()
                    else:
                        recon_loss = (cov_G * (e @ H @ e.transpose(-1, -2))).sum()

                round_loss = round_loss_func(i, sigm(sb))
                total_loss = recon_loss + round_weight * round_loss
                
                # Print initial loss
                if i == 0:
                    if self.i is None:
                        print(f'|{self.name : <27}| {i+1: <2}\t| {float(recon_loss):.3f}\t| {float(round_loss):.3f}\t| {torch.cuda.max_memory_allocated("cuda") / 1024**3: .3f}\t|')
                    else:
                        print(f'|{self.i}: {self.name : <24}| {i+1: <2}\t| {float(recon_loss):.3f}\t| {float(round_loss):.3f}\t|{torch.cuda.max_memory_allocated("cuda") / 1024**3: .3f}\t|')
                
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                scheduler.step()
                
                if (i + 1) % print_period == 0:
                    if self.i is None:
                        print(f'|{self.name : <27}| {i+1: <2}\t| {float(recon_loss):.3f}\t| {float(round_loss):.3f}\t| {torch.cuda.max_memory_allocated("cuda") / 1024**3: .3f}\t|')
                    else:
                        print(f'|{self.i}: {self.name : <24}| {i+1: <2}\t| {float(recon_loss):.3f}\t| {float(round_loss):.3f}\t|{torch.cuda.max_memory_allocated("cuda") / 1024**3: .3f}\t|')
            print('+===========================+================+=================+=================+')

        Q = torch.clamp(torch.floor(W_update / scale) + (sb >= 0).float() + zero, 0, 2**n_bits-1)
        Q = scale * (Q - zero)

        return Q

    @staticmethod
    def spin_greedy_optimized(W: torch.Tensor, H_head: torch.Tensor, clustersize: int = 128) -> torch.Tensor:
        device = W.device
        num_columns = W.shape[1]

        diag = torch.diag(H_head)
        inv_sqrt_diag = 1.0 / (torch.sqrt(diag) + 1e-6)
        H_norm = H_head * inv_sqrt_diag.unsqueeze(0) * inv_sqrt_diag.unsqueeze(1)
        H_norm.fill_diagonal_(0)
        energy = torch.sqrt(diag + 1e-8)

        visited = torch.zeros(num_columns, dtype=torch.bool, device=device)
        perm = []

        num_groups = math.ceil(num_columns / clustersize)
        scores = torch.empty(num_columns, device=device)

        for _ in range(num_groups):
            masked_diag = diag.masked_fill(visited, -float('inf'))
            if torch.all(visited):
                break
            seed_idx = torch.argmax(masked_diag)
            current_group = [seed_idx]
            visited[seed_idx] = True
            current_potential = H_norm[seed_idx].clone()

            for _ in range(clustersize - 1):
                gain = torch.abs(current_potential) * energy
                scores.copy_(gain)
                scores[visited] = -float('inf')
                best_idx = torch.argmax(scores)
                if visited[best_idx]:
                    break
                best_potential = current_potential[best_idx]
                best_spin = -1.0 if best_potential > 0 else 1.0
                current_group.append(best_idx)
                visited[best_idx] = True
                current_potential += H_norm[best_idx] * best_spin

            perm.append(torch.stack(current_group))

        perm = torch.cat(perm)
        return perm.long()

    def free(self):
        self.H = None
        self.cov_G = None
        self.dXXT = None
        self.dXXT_per_qhead = None

        torch.cuda.empty_cache()


class LinearTempDecay:
    def __init__(self, t_max: int, rel_start_decay, start_b, end_b):
        self.t_max = t_max
        self.start_decay = rel_start_decay * t_max
        self.start_b = start_b
        self.end_b = end_b

    def __call__(self, t):
        """
        annealing scheduler for temperature b.
        :param t: the current time step
        :return: scheduled temperature
        """
        if t < self.start_decay:
            return self.start_b
        else:
            rel_t = (t-self.start_decay) / (self.t_max-self.start_decay)
            return self.end_b + (self.start_b-self.end_b)*max(0.0, 1 - rel_t)
        
class RoundLoss(nn.Module):
    def __init__(self, max_count, b_range, decay_start, warmup, p_norm):
        super(RoundLoss, self).__init__()
        self.loss_start = max_count * warmup
        # NOTE: cosine temp decay does not improve accuracy.
        self.temp_decay = LinearTempDecay(max_count, rel_start_decay=warmup + (1-warmup)*decay_start, start_b=b_range[0], end_b=b_range[1])
        self.p_norm = p_norm
        self.b = 0

    def forward(self, iter_count, sb):
        """Compute regularization term to optimize the rounding policy"""
        if iter_count < self.loss_start:
            return 0
        else:
            self.b = self.temp_decay(iter_count)
            return (1 - (2*sb - 1).abs().pow(self.b)).sum()
        
class RectifiedSigmoid(nn.Module):
    """
    Implementation of Rectified Sigmoid Function
    Based on https://arxiv.org/pdf/1712.01312
    """

    def __init__(self, gamma, zeta):
        super(RectifiedSigmoid, self).__init__()
        self.gamma = gamma
        self.zeta = zeta

    def forward(self, x):
        return torch.clamp(torch.sigmoid(x)*(self.zeta-self.gamma) + self.gamma, 0, 1)

    def inverse(self, y):
        """return x that satisfies y = RectifiedSigmoid(x)"""
        return -torch.log((self.zeta-self.gamma)/(y-self.gamma) - 1)