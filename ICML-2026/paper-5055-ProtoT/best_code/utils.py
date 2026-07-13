"""
utils.py: FLOP counting and model analysis using torch.profiler.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import (
    profile,
    ProfilerActivity,
    schedule,
    record_function,
    tensorboard_trace_handler,
)
from transformers import PreTrainedModel, PretrainedConfig, GenerationMixin, GenerationConfig
from transformers.modeling_outputs import CausalLMOutput


def fmt(x):  # get float from tensor for printing
    return float(x.detach().cpu()) if torch.is_tensor(x) else float(x)


def _to_scalar_float(x):
    if torch.is_tensor(x):
        if x.numel() != 1:
            return None
        return float(x.detach().cpu().item())
    if isinstance(x, (float, int)):
        return float(x)
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _accumulate_scalar_dict(bucket, values):
    sums = bucket.setdefault("sums", {})
    counts = bucket.setdefault("counts", {})
    for key, value in values.items():
        if key == "per_layer":
            continue
        scalar = _to_scalar_float(value)
        if scalar is None:
            continue
        sums[key] = sums.get(key, 0.0) + scalar
        counts[key] = counts.get(key, 0) + 1


def _finalize_scalar_dict(bucket):
    sums = bucket.get("sums", {})
    counts = bucket.get("counts", {})
    out = {}
    for key, total in sums.items():
        n = counts.get(key, 0)
        if n > 0:
            out[key] = total / n
    return out


def accumulate_aux_for_epoch(acc, aux):
    if not isinstance(aux, dict):
        return acc
    if acc is None:
        acc = {"main": {"sums": {}, "counts": {}}, "per_layer": []}
    _accumulate_scalar_dict(acc["main"], aux)
    per_layer = aux.get("per_layer", None)
    if isinstance(per_layer, (list, tuple)):
        for i, layer_aux in enumerate(per_layer):
            if not isinstance(layer_aux, dict):
                continue
            while len(acc["per_layer"]) <= i:
                acc["per_layer"].append({"sums": {}, "counts": {}})
            _accumulate_scalar_dict(acc["per_layer"][i], layer_aux)
    return acc


def finalize_aux_for_epoch(acc):
    if not acc:
        return None
    out = _finalize_scalar_dict(acc["main"])
    if acc.get("per_layer"):
        out["per_layer"] = [_finalize_scalar_dict(b) for b in acc["per_layer"]]
    return out


def print_stats(a, grad_stats: dict):
    # Head-averaged scalars per layer for readability
    alpha_mean = None
    tau_w_mean = None
    tau_r_mean = None
    if isinstance(a.get('alpha_per_head', None), torch.Tensor) and a['alpha_per_head'].numel() > 0:
        alpha_mean = a['alpha_per_head'].mean()
    if isinstance(a.get('tau_per_head', None), torch.Tensor) and a['tau_per_head'].numel() > 0:
        tau_w_mean = a['tau_per_head'].mean()
    if isinstance(a.get('tau_read_per_head', None), torch.Tensor) and a['tau_read_per_head'].numel() > 0:
        tau_r_mean = a['tau_read_per_head'].mean()
    alpha_str = f"  α={fmt(alpha_mean):.3f}" if alpha_mean is not None else ""
    tau_w_str = f"  τw={fmt(tau_w_mean):.3f}" if tau_w_mean is not None else ""
    tau_r_str = f"  τr={fmt(tau_r_mean):.3f}" if tau_r_mean is not None else ""
    print(
        f"top1={fmt(a['router_top1_mean']):.3f}  "
        f"margin={fmt(a['router_margin_mean']):.3f}  "
        f"active@1%={fmt(a['active_hubs_per_token@1%']):.2f}  "
        f"gini={fmt(a['hub_gini']):.3f}  maxShare={fmt(a['hub_max_share']):.3f}  dead={fmt(a['hub_dead_frac']):.2f}  "
        f"JS={fmt(a.get('read_write_JS', torch.tensor(0.0))):.3f}  cos={fmt(a.get('read_write_cosine', torch.tensor(0.0))):.3f}  "
        f"βmean={fmt(a.get('beta_mean', torch.tensor(0.0))):.3f}  halfLife={fmt(a.get('memory_half_life_mean', torch.tensor(0.0))):.1f}  "
        f"prefixMin={fmt(a.get('prefix_mass_min', torch.tensor(0.0))):.2e}  <1e-4={fmt(a.get('prefix_mass_lt1e4_frac', torch.tensor(0.0))):.2%}  "
        f"logitStd={fmt(a['router_logit_std']):.3f}  |Vx|={fmt(a['Vx_mean_norm']):.3f}  |Pn|={fmt(a['Pn_mean_norm']):.3f}  "
        f"|∇proto|={grad_stats.get('proto_grad_norm', 0.0):.3e}"
        f"{alpha_str}{tau_w_str}{tau_r_str}"
    )

def count_params(m): 
    return sum(p.numel() for p in m.parameters())

def maybe_compile(model, use_compile=False, **kw):
    """Wrap torch.compile but honour USE_COMPILE."""
    # Skip compilation for models that have _disable_compile flag (e.g., DeltaNet)
    if hasattr(model, '_disable_compile') and model._disable_compile:
        print("⚠️  Skipping torch.compile for this model (incompatible)")
        return model
    
    if use_compile:
        try:
            # First try In​ductor
            compiled = torch.compile(model, **kw, backend="inductor")
            print("✅ Using Inductor backend")
            return compiled
        except Exception as e:
            # Log why Inductor failed
            print(f"⚠️ Inductor failed with error: {e!r}")
            # Fall back to eager
            compiled = torch.compile(model, **kw, backend="eager")
            print("✅ Fell back to eager backend")
            return compiled
    else:
        kw.pop("disable", None)        # keep signature identical
        return model                   # leave it eager

# Daignostics for ProtoBroadcastLM
def collect_router_grad_stats(model: nn.Module) -> dict:
    d = {}
    with torch.no_grad():
        # reach into the first mixer block (or iterate all)
        mix = model.backbone.blocks[0].mixer
        # prototypes (if learnable)
        if not mix.use_memory:
            g = torch.stack([p.grad.detach().flatten() for p in mix.P_tables if p.grad is not None]) \
                    if any(p.grad is not None for p in mix.P_tables) else None
            if g is not None:
                d["proto_grad_norm"] = g.norm().item()
        # temperature / alpha
        if any(t.grad is not None for t in mix.tau):
            d["tau_grad_norm"] = torch.stack([t.grad.detach().flatten() for t in mix.tau if t.grad is not None]).norm().item()
        if any(a.grad is not None for a in mix.alpha):
            d["alpha_grad_norm"] = torch.stack([a.grad.detach().flatten() for a in mix.alpha if a.grad is not None]).norm().item()
    return d

def count_flops_and_params_direct(model, inputs, device='cuda', verbose=True):
    """
    Count total FLOPs and parameters for a model using torch.profiler.
    Returns a dict with:
      - flops: total FLOPs over the batch
      - flops_per_sample: flops / batch_size
      - params: total number of parameters
      - trainable_params: number of trainable parameters
      - flops_by_module: dict of {op_name: flops}
      - params_by_module: dict of {module_name: param_count}
    """
    # 1) Move model + inputs to device
    model = model.to(device).eval()
    if isinstance(inputs, torch.Tensor):
        inputs = (inputs.to(device),)
    else:
        inputs = tuple(inp.to(device) for inp in inputs)
    batch_size = inputs[0].shape[0] if inputs else 1

    # 2) Profile one forward pass (CPU + CUDA) with FLOPs
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_flops=True
    ) as prof:
        with torch.no_grad():
            model(*inputs)

    # 3) Aggregate FLOPs from key_averages()
    ka = prof.key_averages()
    total_flops = sum(evt.flops or 0 for evt in ka)
    flops_by_op = {evt.key: int(evt.flops or 0) for evt in ka if evt.flops}
    sum_by_op = sum(flops_by_op.values())

    if verbose:
        print("\n🔍 FLOP summary:")
        print(f"  Total FLOPs (events sum): {total_flops:,}")
        print(f"  Total FLOPs (ops   sum): {sum_by_op:,}")
        if total_flops != sum_by_op:
            diff = abs(total_flops - sum_by_op)
            print(f"  ⚠️ Mismatch between events vs ops sums: {diff:,}")

    flops_per_sample = total_flops / batch_size if batch_size > 0 else 0

    # 4) Parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # 5) Per-module (deepest owner) param breakdown
    params_by_module = {}
    for name, param in model.named_parameters():
        module_name = ".".join(name.split(".")[:-1]) or "root"
        params_by_module[module_name] = params_by_module.get(module_name, 0) + param.numel()

    if verbose:
        print(f"  FLOPs per sample:     {flops_per_sample:,.0f}")
        print(f"  Total params:         {total_params:,}")
        print(f"  Trainable params:     {trainable_params:,}")

        print("\nTop ops by FLOPs:")
        for op, fl in sorted(flops_by_op.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {op}: {fl:,}")

        print("\nTop modules by params:")
        for mod, sz in sorted(params_by_module.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {mod}: {sz:,}")

    return {
        'flops': total_flops,
        'flops_per_sample': flops_per_sample,
        'params': total_params,
        'trainable_params': trainable_params,
        'flops_by_module': flops_by_op,
        'params_by_module': params_by_module
    }


def analyze_model_first_batch(model, loader_func, device='cuda', verbose=True):
    """
    Grab the first batch from loader_func (callable) or dataloader and run count_flops_and_params_direct.
    Profiles on the first tensor of the batch.
    
    Args:
        model: The model to analyze
        loader_func: Either a callable that returns an iterator, or a dataloader directly
        device: Device to run analysis on
        verbose: Whether to print verbose output
    """
    model = model.to(device).eval()
    
    # Handle both callable loader functions and dataloaders directly
    if callable(loader_func):
        # If it's a callable, call it to get the dataloader
        dataloader = loader_func()
    else:
        # If it's already a dataloader, use it directly
        dataloader = loader_func
    
    batch = next(iter(dataloader))

    inp = batch[0] if isinstance(batch, (tuple, list)) else batch
    inputs = (inp.to(device),)
    return count_flops_and_params_direct(model, inputs, device=device, verbose=verbose)

def install_torch_profiler_if_needed():
    """
    Return True if torch.profiler is available (PyTorch >= 1.8); else print a warning.
    """
    try:
        import torch.profiler  # noqa
        return True
    except ImportError:
        print("❌ torch.profiler not available; please upgrade to PyTorch >=1.8.")
        return False

def profile_model_gpu(
    model: nn.Module,
    loader_func,
    *,
    log_dir: str = "./gpu_profiler_logs",
    wait: int = 1,
    warmup: int = 0,
    active: int = 3,
    repeat: int = 1,
    row_limit: int = 20
) -> None:
    """
    GPU-only profiling of the model forward pass over a few steps.

    Args:
      model        – nn.Module on CUDA (just the backbone, no head)
      loader_func  – either a zero-arg callable returning an iterator, or a dataloader directly
      log_dir      – where to write TensorBoard traces
      wait, warmup,
      active, repeat– torch.profiler.schedule params
      row_limit    – how many lines in the printed summary tables

    Outputs two tables:
      1) Top CUDA kernels by total time
      2) Top ops by peak CUDA memory usage

    And writes trace files under `log_dir` (for `tensorboard --logdir=...`).
    """
    prof_sched = schedule(wait=wait, warmup=warmup, active=active, repeat=repeat)

    model = model.to("cuda").train()
    
    # Handle both callable loader functions and dataloaders directly
    if callable(loader_func):
        # If it's a callable, call it to get the dataloader
        dataloader = loader_func()
    else:
        # If it's already a dataloader, use it directly
        dataloader = loader_func
    with profile(
        activities=[ProfilerActivity.CUDA],
        schedule=prof_sched,
        on_trace_ready=tensorboard_trace_handler(log_dir),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        for step, batch in enumerate(dataloader):
            if step >= (wait + warmup + active):
                break

            with record_function("model_forward"):
                # Handle different batch formats and move to GPU
                if len(batch) == 3:
                    ctx, _, _ = batch
                elif len(batch) == 2:
                    ctx, _ = batch
                else:
                    ctx = batch[0] if isinstance(batch, (tuple, list)) else batch
                
                # Move context to GPU
                ctx = ctx.to("cuda")
                _ = model(ctx)

            prof.step()

    print("\n=== Top CUDA kernels by total time ===")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=row_limit))

    print("\n=== Top ops by peak CUDA memory usage ===")
    print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=row_limit))

    print(f"\n✔︎ GPU trace saved under {log_dir}")


class HFCompatConfig(PretrainedConfig):
    def __init__(self, vocab_size: int = 0,
                 pad_token_id: int | None = None,
                 eos_token_id: int | None = None,
                 bos_token_id: int | None = None,
                 num_hidden_layers: int = 1,
                 is_encoder_decoder: bool = False, **kwargs):
        super().__init__(pad_token_id=pad_token_id, eos_token_id=eos_token_id,
                         bos_token_id=bos_token_id, is_encoder_decoder=is_encoder_decoder, **kwargs)
        self.vocab_size = vocab_size
        self.num_hidden_layers = int(num_hidden_layers)


class HFCompatLM(PreTrainedModel, GenerationMixin):
    config_class = HFCompatConfig
    base_model_prefix = "wrapped"  # .base_model points to the wrapped inner model
    def __init__(self, inner_model, config: HFCompatConfig):
        super().__init__(config)
        self.wrapped = inner_model
        setattr(self, self.base_model_prefix, self.wrapped)
        self.generation_config = GenerationConfig.from_model_config(config)

    def forward(self, input_ids=None, attention_mask=None, use_cache=False, return_dict=True, **_):
        pad_mask = None
        if attention_mask is not None:
            pad_mask = ~attention_mask.bool()
        logits = self.wrapped(input_ids, pad_mask)
        return CausalLMOutput(logits=logits) if return_dict else (logits,)

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **_):
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def _reorder_cache(self, past_key_values, beam_idx):
        return past_key_values


def check_for_nan_inf_gradients(model, verbose=True):
    """
    Check for NaN or Inf values in gradients of the model parameters.
    Returns True if any NaN/Inf found, False otherwise.
    """
    has_issues = False
    for name, param in model.named_parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                has_issues = True
                if verbose:
                    print(f"⚠️ NaN/Inf in gradient of {name}")
    # Only print success message when issues are found and fixed
    return has_issues

def check_gradient_norms(model, threshold_low=1e-7, threshold_high=1e3, verbose=True):
    """
    Check gradient norms for exploding or vanishing gradients.
    Returns dict with counts of parameters with low/high norms.
    """
    low_count = 0
    high_count = 0
    total_count = 0
    total_norm_sq = 0.0  # For computing global gradient norm
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            norm = param.grad.norm().item()
            total_count += 1
            total_norm_sq += norm ** 2
            
            if norm < threshold_low:
                low_count += 1
                if verbose:
                    print(f"⚠️ Low gradient norm in {name}: {norm:.2e}")
            elif norm > threshold_high:
                high_count += 1
                if verbose:
                    print(f"⚠️ High gradient norm in {name}: {norm:.2e}")
    
    global_norm = (total_norm_sq ** 0.5) if total_count > 0 else 0.0
    
    # Only print summary when there are actual issues
    if verbose and (low_count > 0 or high_count > 0):
        print(f"Gradient norm issues: {low_count}/{total_count} low, {high_count}/{total_count} high (global norm: {global_norm:.2e})")
    
    return {
        'low': low_count, 
        'high': high_count, 
        'total': total_count,
        'global_norm': global_norm
    }

def check_zero_gradients(model, verbose=True, tolerance=1e-12):
    """
    Check for zero gradients in parameters.
    Returns count of parameters with zero gradients.
    
    Args:
        tolerance: Consider gradients below this as effectively zero
    """
    zero_count = 0
    total_count = 0
    for name, param in model.named_parameters():
        if param.grad is not None:
            total_count += 1
            # Use L2 norm instead of abs().sum() for numerical stability.
            grad_norm = param.grad.norm().item()
            if grad_norm < tolerance:
                zero_count += 1
                if verbose:
                    print(f"⚠️ Zero gradient in {name} (norm: {grad_norm:.2e})")
    # Only print summary when there are zero gradients
    if verbose and zero_count > 0:
        print(f"Zero gradient warning: {zero_count}/{total_count} parameters have zero gradients")
    return zero_count

def log_gradient_stats(model, tag="", verbose=True, warn_thresholds=None):
    """
    Log basic statistics of gradients: mean, std, min, max norms.
    Only prints when gradient stats are concerning.
    
    Args:
        warn_thresholds: dict with 'mean_low', 'mean_high', 'max_high' thresholds
    """
    if warn_thresholds is None:
        warn_thresholds = {'mean_low': 1e-6, 'mean_high': 1e2, 'max_high': 1e4}
    
    norms = []
    for param in model.parameters():
        if param.grad is not None:
            norms.append(param.grad.norm().item())
    
    if norms:
        norms = torch.tensor(norms)
        mean_norm = norms.mean().item()
        std_norm = norms.std().item()
        min_norm = norms.min().item()
        max_norm = norms.max().item()
        
        # Only print if concerning values detected
        should_warn = (
            mean_norm < warn_thresholds['mean_low'] or 
            mean_norm > warn_thresholds['mean_high'] or
            max_norm > warn_thresholds['max_high']
        )
        
        if verbose and should_warn:
            print(f"⚠️ {tag} Gradient stats: mean={mean_norm:.2e}, std={std_norm:.2e}, min={min_norm:.2e}, max={max_norm:.2e}")
        return {'mean': mean_norm, 'std': std_norm, 'min': min_norm, 'max': max_norm}
    else:
        if verbose:
            print(f"⚠️ {tag} No gradients found")
        return None

def comprehensive_gradient_check(model, step=None, verbose=True):
    """
    Run all gradient checks and return a summary of gradient health.
    This is the main function to call for gradient debugging.
    
    Returns:
        dict with gradient health summary and issue flags
    """
    if step is not None:
        tag = f"Step {step}"
    else:
        tag = ""
    
    # Run all checks
    has_nan_inf = check_for_nan_inf_gradients(model, verbose=verbose)
    norm_stats = check_gradient_norms(model, verbose=verbose)
    zero_count = check_zero_gradients(model, verbose=verbose)
    grad_stats = log_gradient_stats(model, tag=tag, verbose=verbose)
    
    # Determine overall health
    has_issues = (
        has_nan_inf or 
        norm_stats['low'] > 0 or 
        norm_stats['high'] > 0 or 
        zero_count > norm_stats['total'] * 0.1  # More than 10% zero gradients is concerning
    )
    
    health_summary = {
        'has_issues': has_issues,
        'has_nan_inf': has_nan_inf,
        'vanishing_count': norm_stats['low'],
        'exploding_count': norm_stats['high'],
        'zero_count': zero_count,
        'total_params': norm_stats['total'],
        'global_norm': norm_stats['global_norm'],
        'grad_stats': grad_stats
    }
    
    return health_summary

def is_flash_attn_2_available():
    """Check if flash attention 2 is available."""
    try:
        from transformers import is_flash_attn_2_available
        return is_flash_attn_2_available()
    except ImportError:
        return False


# --- SwiGLU Helper Class ---
class SwiGLU(nn.Module):
    """ Swish-Gated Linear Unit """
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, hidden_dim, bias=False)
        self.gate = nn.Linear(in_dim, hidden_dim, bias=False)

    def forward(self, x):
        # F.silu is the Swish activation function
        return F.silu(self.gate(x)) * self.linear(x)


class HFDecoderAdapter(nn.Module):
    """
    Thin wrapper so the training loop can call `model(x, pad_mask)` and
    get logits back, while using a HuggingFace CausalLM under the hood.
    """
    def __init__(self, hf_model, pad_token_id):
        super().__init__()
        self.hf = hf_model
        self.pad_token_id = pad_token_id

    def forward(self, x, pad_mask=None):
        # x: [B, T] token IDs
        if pad_mask is None:
            attention_mask = (x != self.pad_token_id).long()
        else:
            attention_mask = (~pad_mask).long()  # pad_mask is True on padding positions
        out = self.hf(input_ids=x, attention_mask=attention_mask, use_cache=False)
        return out.logits  # [B, T, vocab]


# --- Small wrappers that keep original weights intact ---

class EmbeddingWithDropout(nn.Module):
    def __init__(self, emb: nn.Embedding, p: float):
        super().__init__()
        self.emb = emb
        self.drop = nn.Dropout(p)
    def forward(self, input_ids: torch.LongTensor):
        return self.drop(self.emb(input_ids))
    @property
    def weight(self):  # expose .weight for HF tying and compatibility checks
        return self.emb.weight
    def num_embeddings(self):  # optional: helps some utilities introspect
        return self.emb.num_embeddings
    def embedding_dim(self):   # optional
        return self.emb.embedding_dim


class LinearWithDropout(nn.Module):
    """Wrap a Linear and add dropout to its *output* (useful for residual-branch dropout)."""
    def __init__(self, lin: nn.Linear, p: float):
        super().__init__()
        self.lin = lin
        self.drop = nn.Dropout(p)
    def forward(self, x):
        return self.drop(self.lin(x))
    # Proxy common attributes so optimizers/state_dict behave normally
    @property
    def weight(self): return self.lin.weight
    @property
    def bias(self):   return self.lin.bias
