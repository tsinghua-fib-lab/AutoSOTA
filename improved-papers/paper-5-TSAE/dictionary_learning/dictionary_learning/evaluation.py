"""
Utilities for evaluating dictionaries on a model and dataset.
"""

import torch as t
import numpy as np
from collections import defaultdict

from .buffer import ActivationBuffer, EvalActivationBuffer #, NNsightActivationBuffer
from nnsight import LanguageModel
from .config import DEBUG
from .trainers import TemporalBatchTopKSAE, TemporalMatryoshkaBatchTopKSAE, MatryoshkaBatchTopKSAE
from tqdm import tqdm
import pdb

def loss_recovered(
    text,  # a batch of text
    model: LanguageModel,  # an nnsight LanguageModel
    submodule,  # submodules of model
    dictionary,  # dictionaries for submodules
    max_len=None,  # max context length for loss recovered
    normalize_batch=False,  # normalize batch before passing through dictionary
    io="out",  # can be 'in', 'out', or 'in_and_out'
    tracer_args = {'use_cache': False, 'output_attentions': False}, # minimize cache during model trace.
):
    """
    How much of the model's loss is recovered by replacing the component output
    with the reconstruction by the autoencoder?
    """
    
    if max_len is None:
        invoker_args = {}
    else:
        invoker_args = {"truncation": True, "max_length": max_len }

    with model.trace("_"):
        temp_output = submodule.output.save()

    output_is_tuple = False
    # Note: isinstance() won't work here as torch.Size is a subclass of tuple,
    # so isinstance(temp_output.shape, tuple) would return True even for torch.Size.
    if type(temp_output.shape) == tuple:
        output_is_tuple = True

    # unmodified logits
    with model.trace(text, invoker_args=invoker_args):
        logits_original = model.output.save()
    logits_original = logits_original.value
    
    # logits when replacing component activations with reconstruction by autoencoder
    with model.trace(text, **tracer_args, invoker_args=invoker_args):
        if io == 'in':
            x = submodule.input
            if normalize_batch:
                scale = (dictionary.activation_dim ** 0.5) / x.norm(dim=-1).mean()
                x = x * scale
        elif io == 'out':
            x = submodule.output
            if output_is_tuple: x = x[0]
            if normalize_batch:
                scale = (dictionary.activation_dim ** 0.5) / x.norm(dim=-1).mean()
                x = x * scale
        elif io == 'in_and_out':
            x = submodule.input
            if normalize_batch:
                scale = (dictionary.activation_dim ** 0.5) / x.norm(dim=-1).mean()
                x = x * scale
        else:
            raise ValueError(f"Invalid value for io: {io}")
        x = x.save()

    # If we incorrectly handle output_is_tuple, such as with some mlp submodules, we will get an error here.
    assert len(x.shape) == 3, f"Expected x to have shape (B, L, D), got {x.shape}, output_is_tuple: {output_is_tuple}"

    x_hat = dictionary(x).to(model.dtype)

    # intervene with `x_hat`
    with model.trace(text, **tracer_args, invoker_args=invoker_args):
        if io == 'in':
            x = submodule.input
            if normalize_batch:
                scale = (dictionary.activation_dim ** 0.5) / x.norm(dim=-1).mean()
                x_hat = x_hat / scale
            submodule.input[:] = x_hat
        elif io == 'out':
            x = submodule.output
            if output_is_tuple: x = x[0]
            if normalize_batch:
                scale = (dictionary.activation_dim ** 0.5) / x.norm(dim=-1).mean()
                x_hat = x_hat / scale
            if output_is_tuple:
                submodule.output[0][:] = x_hat
            else:
                submodule.output[:] = x_hat
        elif io == 'in_and_out':
            x = submodule.input
            if normalize_batch:
                scale = (dictionary.activation_dim ** 0.5) / x.norm(dim=-1).mean()
                x_hat = x_hat / scale
            if output_is_tuple:
                submodule.output[0][:] = x_hat
            else:
                submodule.output[:] = x_hat
        else:
            raise ValueError(f"Invalid value for io: {io}")

        logits_reconstructed = model.output.save()
    logits_reconstructed = logits_reconstructed.value

    # logits when replacing component activations with zeros
    with model.trace(text, **tracer_args, invoker_args=invoker_args):
        if io == 'in':
            x = submodule.input
            submodule.input[:] = t.zeros_like(x)
        elif io in ['out', 'in_and_out']:
            x = submodule.output
            if output_is_tuple:
                submodule.output[0][:] = t.zeros_like(x[0])
            else:
                submodule.output[:] = t.zeros_like(x)
        else:
            raise ValueError(f"Invalid value for io: {io}")
        
        input = model.inputs.save()
        logits_zero = model.output.save()

    logits_zero = logits_zero.value

    # get everything into the right format
    try:
        logits_original = logits_original.logits
        logits_reconstructed = logits_reconstructed.logits
        logits_zero = logits_zero.logits
    except:
        pass

    if isinstance(text, t.Tensor):
        tokens = text
    else:
        try:
            tokens = input[1]['input_ids']
        except:
            tokens = input[1]['input']

    # compute losses
    losses = []
    if hasattr(model, 'tokenizer') and model.tokenizer is not None:
        loss_kwargs = {'ignore_index': model.tokenizer.pad_token_id}
    else:
        loss_kwargs = {}
    for logits in [logits_original, logits_reconstructed, logits_zero]:
        loss = t.nn.CrossEntropyLoss(**loss_kwargs)(
            logits[:, :-1, :].reshape(-1, logits.shape[-1]), tokens[:, 1:].reshape(-1)
        )
        losses.append(loss)

    return tuple(losses)

def smoothness_tv(f, dictionary):

    if type(dictionary) is TemporalMatryoshkaBatchTopKSAE or type(dictionary) is MatryoshkaBatchTopKSAE:
        f_chunks = t.split(f, dictionary.group_sizes.tolist(), dim=1)
        chunk_tvs = {}
        for time in range(1, f.shape[0]):
            for c, f_c in enumerate(f_chunks):
                i = 0 if c < 1 else 1
                if i in chunk_tvs:
                    chunk_tvs[i] += t.abs(f_c[time] - f_c[time-1]).sum()
                else:
                    chunk_tvs[i] = t.abs(f_c[time] - f_c[time-1]).sum()
        return chunk_tvs[0], chunk_tvs[1]
    else:
        tv = 0
        for time in range(1, f.shape[0]):
            tv += t.abs(f[time] - f[time-1]).sum()
        return tv, t.tensor(0)
    
def lipschitz_cont(x, f, dictionary):
    min_l = {}
    # f = f.clone().detach().cpu()
    if type(dictionary) is TemporalMatryoshkaBatchTopKSAE or type(dictionary) is MatryoshkaBatchTopKSAE:
        f_chunks = t.split(f, dictionary.group_sizes.tolist(), dim=1)
        
        for c, f_c in enumerate(f_chunks):
            active_indices = (f_c.sum(0) != 0).nonzero(as_tuple=True)[0]

            # x = x[active_indices]
            f_c = f_c[:, active_indices]
            min_l[c] = t.zeros((f_c.shape[0]-1, f_c.shape[1]))
            for time in range(1, f_c.shape[0]):
                f_change = t.abs(f_c[time] - f_c[time-1])
                x_change = t.linalg.vector_norm(x[time] - x[time-1], ord=2)

                l = f_change / x_change

                min_l[c][time-1, :] = l

        active_indices = (f.sum(0) != 0).nonzero(as_tuple=True)[0]
        # x = x[active_indices]
        f = f[:, active_indices]
        min_l[2] = t.zeros((f.shape[0]-1, f.shape[1]))

        for time in range(1, f.shape[0]):
            f_change = t.abs(f[time] - f[time-1])
            x_change = t.linalg.vector_norm(x[time] - x[time-1], ord=2)
            l = f_change / x_change
            min_l[2][time-1, :] = l
        return min_l[2].max(dim=0)[0].mean(), min_l[0].max(dim=0)[0].mean(), min_l[1].max(dim=0)[0].mean()
    else:
        active_indices = (f.sum(0) != 0).nonzero(as_tuple=True)[0]
        # x = x[active_indices]
        f = f[:, active_indices]
        min_l[0] = t.zeros((f.shape[0]-1, f.shape[1]))

        for time in range(1, f.shape[0]):
            f_change = t.abs(f[time] - f[time-1])
            x_change = t.linalg.vector_norm(x[time] - x[time-1], ord=2)
            l = f_change / x_change
            min_l[0][time-1, :] = l
        return min_l[0].max(dim=0)[0].mean(), t.tensor(0), t.tensor(0)

def fft_smoothness(f, dictionary, cutoff_ratio=0.5):
    smoothnesses = {}
    f = f.clone().detach().cpu()
    f = f.to(t.float32)

    if type(dictionary) is TemporalMatryoshkaBatchTopKSAE or type(dictionary) is MatryoshkaBatchTopKSAE:
        f_chunks = t.split(f, dictionary.group_sizes.tolist(), dim=1)
        
        for c, f_c in enumerate(f_chunks):
            active_indices = (f_c.sum(0) != 0).nonzero(as_tuple=True)[0]
            f_c = f_c[:, active_indices]

            f_c_fft = t.fft.rfft(f_c, dim=0)
            power = t.abs(f_c_fft) ** 2

            # Define cutoff frequency index
            cutoff_idx = int(cutoff_ratio * power.shape[0])
            
            # Split into low and high frequency components
            low_freq_energy = power[:cutoff_idx].sum(dim=0)
            high_freq_energy = power[cutoff_idx:].sum(dim=0)
            
            # Compute ratio for each feature (add small epsilon to avoid division by zero)
            eps = 1e-10
            ratio = high_freq_energy / (low_freq_energy + eps)
            
            # Average across features
            smoothnesses[c] = ratio.mean().item()

        active_indices = (f.sum(0) != 0).nonzero(as_tuple=True)[0]

        f = f[:, active_indices]
        f_fft = t.fft.rfft(f, dim=0)
        power = t.abs(f_fft) ** 2
        cutoff_idx = int(cutoff_ratio * power.shape[0])
        low_freq_energy = power[:cutoff_idx].sum(dim=0)
        high_freq_energy = power[cutoff_idx:].sum(dim=0)
        eps = 1e-10
        ratio = high_freq_energy / (low_freq_energy + eps)
        smoothnesses[2] = ratio.mean().item()
        
        return smoothnesses[2], smoothnesses[0], smoothnesses[1]
    else:
        active_indices = (f.sum(0) != 0).nonzero(as_tuple=True)[0]
        # x = x[active_indices]
        f = f[:, active_indices]
        f_fft = t.fft.rfft(f, dim=0)
        power = t.abs(f_fft) ** 2
        cutoff_idx = int(cutoff_ratio * power.shape[0])
        low_freq_energy = power[:cutoff_idx].sum(dim=0)
        high_freq_energy = power[cutoff_idx:].sum(dim=0)
        eps = 1e-10
        ratio = high_freq_energy / (low_freq_energy + eps)
        smoothnesses[0] = ratio.mean().item()
        
        return smoothnesses[0], 0, 0

def multiscale_smoothness(f, dictionary, scales=None, method='variance'):
    """
    Compute multi-scale smoothness by analyzing differences at multiple scales.
    
    Args:
        f: torch.Tensor of shape [t, d] where t is time steps, d is features
        dictionary: dictionary object (checks if Matryoshka type for partitioning)
        scales: list of int, scales to analyze (default: [1, 2, 4, 8])
        method: str, either 'variance' or 'gradient'
    
    Returns:
        tuple of (overall_smoothness, chunk0_smoothness, chunk1_smoothness)
    """
    if scales is None:
        scales = [1, 2, 4, 8]
    
    smoothnesses = {}
    f = f.clone().detach().cpu()
    f = f.to(t.float32)
    
    def compute_scale_ratio(matrix):
        """Helper to compute multiscale ratio for a matrix."""
        time_steps = matrix.shape[0]
        scale_measures = {}
        
        for scale in scales:
            if scale >= time_steps:
                continue
            
            if method == 'variance':
                diffs = matrix[scale:] - matrix[:-scale]
                measure = diffs.var(dim=0).mean().item()
            elif method == 'gradient':
                diffs = t.abs(matrix[scale:] - matrix[:-scale]) / scale
                measure = diffs.mean().item()
            
            scale_measures[scale] = measure
        
        # Compute ratio: fine-scale / coarse-scale variation
        fine_scale = min(scales)
        valid_scales = [s for s in scales if s < time_steps]
        if not valid_scales:
            return 0.0
        coarse_scale = max(valid_scales)
        
        eps = 1e-10
        ratio = scale_measures[fine_scale] / (scale_measures[coarse_scale] + eps)
        return ratio

    if type(dictionary) is TemporalMatryoshkaBatchTopKSAE or type(dictionary) is MatryoshkaBatchTopKSAE:
        f_chunks = t.split(f, dictionary.group_sizes.tolist(), dim=1)
        
        for c, f_c in enumerate(f_chunks):
            active_indices = (f_c.sum(0) != 0).nonzero(as_tuple=True)[0]
            f_c = f_c[:, active_indices]
            
            smoothnesses[c] = compute_scale_ratio(f_c)

        active_indices = (f.sum(0) != 0).nonzero(as_tuple=True)[0]
        f = f[:, active_indices]
        
        smoothnesses[2] = compute_scale_ratio(f)
        
        return smoothnesses[2], smoothnesses[0], smoothnesses[1]
    else:
        active_indices = (f.sum(0) != 0).nonzero(as_tuple=True)[0]
        f = f[:, active_indices]

        smoothnesses[0] = compute_scale_ratio(f)
        
        return smoothnesses[0], 0, 0


def wavelet_smoothness(f, dictionary, levels=3):
    """
    Compute smoothness using wavelet-inspired multi-resolution analysis.
    
    Args:
        f: torch.Tensor of shape [t, d]
        dictionary: dictionary object (checks if Matryoshka type for partitioning)
        levels: int, number of decomposition levels
    
    Returns:
        tuple of (overall_smoothness, chunk0_smoothness, chunk1_smoothness)
    """
    smoothnesses = {}
    f = f.clone().detach().cpu()
    f = f.to(t.float32)
    
    def compute_wavelet_ratio(matrix):
        """Helper to compute wavelet-based smoothness for a matrix."""
        signal = matrix.clone()
        detail_energy = 0
        approx_energy = 0
        
        for level in range(levels):
            if signal.shape[0] < 2:
                break
            
            if signal.shape[0] % 2 == 1:
                signal = signal[:-1]
            
            even = signal[::2]
            odd = signal[1::2]
            
            approx = (even + odd) / 2
            detail = (even - odd) / 2
            
            detail_eng = (detail ** 2).sum().item()
            detail_energy += detail_eng
            
            signal = approx
        
        approx_energy = (signal ** 2).sum().item()
        
        eps = 1e-10
        ratio = detail_energy / (approx_energy + eps)
        return ratio

    if type(dictionary) is TemporalMatryoshkaBatchTopKSAE or type(dictionary) is MatryoshkaBatchTopKSAE:
        f_chunks = t.split(f, dictionary.group_sizes.tolist(), dim=1)
        
        for c, f_c in enumerate(f_chunks):
            active_indices = (f_c.sum(0) != 0).nonzero(as_tuple=True)[0]
            f_c = f_c[:, active_indices]
            
            smoothnesses[c] = compute_wavelet_ratio(f_c)

        active_indices = (f.sum(0) != 0).nonzero(as_tuple=True)[0]
        f = f[:, active_indices]        

        smoothnesses[2] = compute_wavelet_ratio(f)
        
        return smoothnesses[2], smoothnesses[0], smoothnesses[1]
    else:
        active_indices = (f.sum(0) != 0).nonzero(as_tuple=True)[0]
        f = f[:, active_indices]
        
        smoothnesses[0] = compute_wavelet_ratio(f)

        return smoothnesses[0], 0, 0

def recon_splits(x,f,dictionary):
    if type(dictionary) is TemporalMatryoshkaBatchTopKSAE or type(dictionary) is MatryoshkaBatchTopKSAE:
        f_chunks = t.split(f, dictionary.group_sizes.tolist(), dim=1)
        W_chunks = t.split(dictionary.W_dec, dictionary.group_sizes.tolist(), dim=0)
        x_hats = []
        for i, f_c in enumerate(f_chunks):
            x_recon = f_c@W_chunks[i]
            x_hats.append(x_recon)
        return x_hats[0], x_hats[1]
    else:
        return 0,0

@t.no_grad()
def evaluate(
    dictionary,  # a dictionary
    activations, # a generator of activations; if an ActivationBuffer, also compute loss recovered
    max_len=128,  # max context length for loss recovered
    batch_size=128,  # batch size for loss recovered
    io="out",  # can be 'in', 'out', or 'in_and_out'
    normalize_batch=False, # normalize batch before passing through dictionary
    tracer_args={'use_cache': False, 'output_attentions': False}, # minimize cache during model trace.
    device="cpu",
    n_batches: int = 1,
):
    assert n_batches > 0
    out = defaultdict(float)
    active_features = t.zeros(dictionary.dict_size, dtype=t.float32, device=device)

    for i in tqdm(range(n_batches)):
        try:
            x = next(activations).to(device)
            if normalize_batch:
                x = x / x.norm(dim=-1).mean() * (dictionary.activation_dim ** 0.5)
        except StopIteration:
            raise StopIteration(
                "Not enough activations in buffer. Pass a buffer with a smaller batch size or more data."
            )
        x_hat, f = dictionary(x, output_features=True)
        x_hat_high, x_hat_low = recon_splits(x, f, dictionary)

        l2_loss = t.linalg.norm(x - x_hat, dim=-1).mean()
        l1_loss = f.norm(p=1, dim=-1).mean()
        l0 = (f != 0).float().sum(dim=-1).mean()
        sequence_l0 = (f.sum(0) != 0).float().sum(dim=-1).mean() 
        sequence_smoothness_h,  sequence_smoothness_l = smoothness_tv(f, dictionary)
        sequence_lips_tot, sequence_lips_cont_h, sequence_lips_cont_l = lipschitz_cont(x, f, dictionary)
        fft_tot, fft_h, fft_l = fft_smoothness(f, dictionary)
        wavelet_tot, wavelet_h, wavelet_l = wavelet_smoothness(f, dictionary)
        multiscale_tot, multiscale_h, multiscale_l = multiscale_smoothness(f, dictionary)
        features_BF = t.flatten(f, start_dim=0, end_dim=-2).to(dtype=t.float32) # If f is shape (B, L, D), flatten to (B*L, D)
        assert features_BF.shape[-1] == dictionary.dict_size
        assert len(features_BF.shape) == 2

        active_features += features_BF.sum(dim=0)

        # cosine similarity between x and x_hat
        x_normed = x / t.linalg.norm(x, dim=-1, keepdim=True)
        x_hat_normed = x_hat / t.linalg.norm(x_hat, dim=-1, keepdim=True)
        cossim = (x_normed * x_hat_normed).sum(dim=-1).mean()

        # l2 ratio
        l2_ratio = (t.linalg.norm(x_hat, dim=-1) / t.linalg.norm(x, dim=-1)).mean()

        #compute variance explained
        total_variance = t.var(x, dim=0).sum()
        residual_variance = t.var(x - x_hat, dim=0).sum()
        frac_variance_explained = (1 - residual_variance / total_variance)

        #
        residual_variance_high = t.var(x - x_hat_high, dim=0).sum()
        residual_variance_low = t.var(x-x_hat_low,dim=0).sum()
        fve_high = (1-residual_variance_high/total_variance)
        fve_low = (1-residual_variance_low/total_variance)

        # Equation 10 from https://arxiv.org/abs/2404.16014
        x_hat_norm_squared = t.linalg.norm(x_hat, dim=-1, ord=2)**2
        x_dot_x_hat = (x * x_hat).sum(dim=-1)
        relative_reconstruction_bias = x_hat_norm_squared.mean() / x_dot_x_hat.mean()

        out["l2_loss"] += l2_loss.item()
        out["l1_loss"] += l1_loss.item()
        out["l0"] += l0.item()
        out["sequence_l0"] += sequence_l0.item()
        out["smoothness_tv_h"] += sequence_smoothness_h.item()
        out["smoothness_tv_l"] += sequence_smoothness_l.item()
        out["lipschitz_cont_tot"] += sequence_lips_tot.item()
        out["lipschitz_cont_h"] += sequence_lips_cont_h.item()
        out["lipschitz_cont_l"] += sequence_lips_cont_l.item()
        out["fft_tot"] += fft_tot
        out["fft_h"] += fft_h
        out["fft_l"] += fft_l
        out["wavelet_tot"] += wavelet_tot
        out["wavelet_h"] += wavelet_h
        out["wavelet_l"] += wavelet_l
        out["multiscale_tot"] += multiscale_tot
        out["multiscale_h"] += multiscale_h
        out["multiscale_l"] += multiscale_l
        out["frac_variance_explained"] += frac_variance_explained.item()
        out["frac_variance_explained_high"] += fve_high.item()
        out["frac_variance_explained_low"] += fve_low.item()
        out["cossim"] += cossim.item()
        out["l2_ratio"] += l2_ratio.item()
        out['relative_reconstruction_bias'] += relative_reconstruction_bias.item()

        if not isinstance(activations, (EvalActivationBuffer, ActivationBuffer)): # NNsightActivationBuffer
            continue

        # compute loss recovered
        loss_original, loss_reconstructed, loss_zero = loss_recovered(
            activations.text_batch(batch_size=batch_size),
            activations.model,
            activations.submodule,
            dictionary,
            max_len=max_len,
            normalize_batch=normalize_batch,
            io=io,
            tracer_args=tracer_args
        )
        frac_recovered = (loss_reconstructed - loss_zero) / (loss_original - loss_zero)
        
        out["loss_original"] += loss_original.item()
        out["loss_reconstructed"] += loss_reconstructed.item()
        out["loss_zero"] += loss_zero.item()
        out["frac_recovered"] += frac_recovered.item()

    out = {key: value / n_batches for key, value in out.items()}
    frac_alive = (active_features != 0).float().sum() / dictionary.dict_size
    out["frac_alive"] = frac_alive.item()

    return out