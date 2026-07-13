import os, json, math, random, argparse, time
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # prevents possible deadlock risks and suppresses a warning
import torch

# PyTorch < 2.3 compatibility shims
if not hasattr(torch.compiler, 'cudagraph_mark_step_begin'):
    torch.compiler.cudagraph_mark_step_begin = lambda: None
if not hasattr(torch.compiler, 'set_stance'):
    torch.compiler.set_stance = lambda _stance: None
import torch.nn.functional as F
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR, ConstantLR
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from tokenizers import Tokenizer

# Import local modules
from llama_baseline import create_llama31_from_args
from utils import (
    HFCompatConfig, HFCompatLM, comprehensive_gradient_check, collect_router_grad_stats,
    count_params, maybe_compile, print_stats,
    accumulate_aux_for_epoch, finalize_aux_for_epoch)
from data_utils import FineWebBPETokenizer, create_causal_collate_fn, NPZDataset
from prototype_attn import ProtoBroadcastLM
from mamba import create_mamba_from_args

# Import utilities for FLOP counting
try:
    from utils import analyze_model_first_batch, install_torch_profiler_if_needed, profile_model_gpu
except ImportError:
    def install_torch_profiler_if_needed():
        return False


# ── 0) Config ───────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Configuration for model training and architecture.")

    model_group = parser.add_argument_group("Architecture, tokenizer, and dataset")
    model_group.add_argument('--MODEL', type=str, choices=['PrototypeAttn', 'llama', 'mamba', 'deltanet'])
    model_group.add_argument('--DATASET', type=str, default='FineWeb', choices=['FineWeb'])
    model_group.add_argument('--TRAIN_SIZE', type=int, default=18000, help='Number of training examples.')  # default from Matteo's code
    model_group.add_argument('--DEV_SIZE', type=int, default=4000, help='Number of validation examples.')  # default increased for stat significance
    model_group.add_argument('--TOKENIZER', type=str, default='bpe', choices=['bpe'])
    model_group.add_argument('--TOKENIZER_PATH', type=str, default=None, help='Path to tokenizer files (for BPE).')
    model_group.add_argument('--VOCAB_SIZE', type=int, default=10000, help='Vocabulary size; overridden by tokenizer config.')  # small to keep classification head small
    model_group.add_argument('--SEQ_LEN', type=int, default=256, help='Context size.')
    model_group.add_argument('--BOTTLENECK', type=int, default=256,
                        help='Bottleneck dimension (hidden size) in the architecture.')
    model_group.add_argument('--LAYERS', type=int, default=6)
    model_group.add_argument('--HEADS', type=int, default=4, help='Number of attention heads.')
    model_group.add_argument('--TF_FFN_RATIO', type=float, default=2.7,  #  2/3*4 ~ 2.7 for LLaMA-2; set to 3.5 for LLaMA-3
                        help='Feedforward layer size ratio (FFN) of the FFN blocks, for all models.')
    model_group.add_argument('--TIE_HEAD', action='store_true',
                        help='Whether to tie head to the embedding layer; TIE_HEAD=True needs much higher LR.')
    model_group.add_argument('--R', type=int, default=32, help='Number of prototypes (per mixer layer).')
    model_group.add_argument('--PROTO_ALPHA_INIT', type=float, default=1.0,
                        help='Initial value for ProtoAttn ReZero alpha gate.')
    model_group.add_argument('--PROTO_DISABLE_MASS_NORM', action='store_true',
                        help='Disable ProtoAttn mass normalization in prefix aggregation.')
    model_group.add_argument('--PROTO_DISABLE_VALUE_LOWRANK', action='store_true',
                        help='Disable ProtoAttn low-rank value projection bottleneck (use full value width).')
    model_group.add_argument('--PROTO_KERNEL_SIZE', type=int, default=5,
                        help='Causal depthwise-conv kernel size in PrototypeAttn local mixer path (layers < 2).')
    model_group.add_argument('--HUB_DROPOUT', type=float, default=0.0,
                        help='Hub dropout probability (drops routing entries during training).')
    model_group.add_argument('--TAU_INIT', type=float, default=1.0,
                        help='Initial tau (temperature) for routing in layers 1-N (L0 always uses 3.0).')
    model_group.add_argument('--DROPOUT', type=float, default=0.1,
                        help='Base dropout probability used by model components.')
    model_group.add_argument('--ATTN_DROPOUT', type=float, default=0.1,
                        help='Inside-attention dropout probability (used only where supported).')

    train_group = parser.add_argument_group("Training hyperparameters")
    train_group.add_argument('--DEVICE', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    train_group.add_argument('--BATCH', type=int, default=32)
    train_group.add_argument('--EPOCHS1', type=int, default=10, help='Training epochs', )
    train_group.add_argument('--LR1', type=float, default=6e-4, help='Learning rate')
    train_group.add_argument('--USE_LR_SCHEDULER', action='store_true', help='Use learning rate scheduler.')
    train_group.add_argument('--LR_SCHEDULE', type=str, default='cosine', choices=['cosine', 'wsd'],
                        help='LR schedule type: cosine or wsd (warmup-stable-decay).')
    train_group.add_argument('--W_ENTROPY', type=float, default=0.0,
                        help='Weight for routing entropy regularizer (PrototypeAttn only). Use +weight to encourage lower entropy; negative to increase it')
    train_group.add_argument('--W_BALANCE', type=float, default=0.0,
                        help='Weight for hub utilization balance regularizer (PrototypeAttn only).')
    train_group.add_argument('--USE_COMPILE', action='store_true',
                        help='Whether to use torch.compile() for optimization.')
    train_group.add_argument('--PROFILE_FLOPS', action='store_true',
                        help='Profile FLOPs and parameters using fvcore (disabled by default).')
    train_group.add_argument('--PROFILE_GPU', action='store_true',
                        help='Run a short GPU‐only torch.profiler session')
    train_group.add_argument('--CHECK_GRADIENTS', action='store_true',
                        help='Enable gradient checks for debugging (may slow down training).')
    train_group.add_argument('--DIAGNOSE', action='store_true',
                        help='Whether to return diagnostics for PrototypeAttn.')
    train_group.add_argument('--DISABLE_TQDM', action='store_true',
                        help='Disable tqdm progress bars during training.')
    train_group.add_argument('--DISABLE_GENERATION', action='store_true',
                        help='Disable text generation after training.')
    train_group.add_argument('--DELETE_SAVED_MODEL', action='store_true',
                        help='Delete model_state_dict.pth after final evaluation to keep sweep artifacts small.')
    train_group.add_argument('--EVAL_NO_COMPILE', action='store_true',
                        help='Run evaluation on the uncompiled model (avoid TorchDynamo/Inductor).')
    train_group.add_argument('--SEED', type=int, default=4257,
                        help='Random seed for reproducibility.')
    train_group.add_argument('--SAVE_DIR', type=str, default=None,
                        help='Directory to save the trained model and tokenizer.')
    train_group.add_argument('--LOAD_DIR', type=str, default=None,
                        help='Directory containing trained model and config.')
    train_group.add_argument('--TEST_EVALUATE', action='store_true',
                        help='Whether to evaluate on the test set after training.')

    return parser.parse_args(), train_group

args, train_args = parse_args()

# Load the config file into args from LOAD_DIR if provided; ignore training-specific keys for flexibility
if args.LOAD_DIR is not None:
    config_path = os.path.join(args.LOAD_DIR, "args.json")
    assert os.path.isfile(config_path), f"Config file not found in LOAD_DIR: {config_path}"
    with open(config_path, "r") as f:
        loaded_config = json.load(f)
    ignore_keys = [a.dest for a in train_args._group_actions]
    for key, value in loaded_config.items():
        if key not in ignore_keys:
            if hasattr(args, key):
                print(f"Overriding arg '{key}': {getattr(args, key)} -> {value}")
                setattr(args, key, value)
            else:
                print(f"Ignoring unknown arg '{key}' from config with value: {value}")
    print(f"✅ Loaded configuration from {config_path}, ignoring keys: {ignore_keys}")

LOG_DIR     = os.environ.get('LOG_DIR')  # injected by EasyTuna for per-run storage
if LOG_DIR is not None:  # make sure both log_dir and save_dir are populated
    print('Log dir:', LOG_DIR)
    args.SAVE_DIR = LOG_DIR if args.SAVE_DIR is None else args.SAVE_DIR
else:
    if args.SAVE_DIR is None:
        args.SAVE_DIR = 'saved_models/test_run'
    else:
        LOG_DIR = args.SAVE_DIR
os.makedirs(args.SAVE_DIR, exist_ok=True)

# FFN size for Transformer blocks
ffn_size = int(args.TF_FFN_RATIO * args.BOTTLENECK)
TF_FFN_SIZE = (ffn_size // 16) * 16  # make it divisible by 16 for fast tensor cores

assert args.BOTTLENECK % args.HEADS == 0, "--BOTTLENECK must be divisible by --HEADS"

# Set random seeds for reproducibility
torch.manual_seed(args.SEED)
torch.cuda.manual_seed(args.SEED)
torch.cuda.manual_seed_all(args.SEED)  # also covers multi-GPU runs
random.seed(args.SEED)
np.random.seed(args.SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Mixed precision settings optimized for A100/H100+
# BF16 > FP16 on Ampere/Hopper: more stable and doesn't require GradScaler
AUTOCAST_DTYPE = torch.bfloat16
USE_SCALER = False
print("✅ Using BF16 autocast (no scaler) optimized for A100/H100+ GPUs")

# Save args
if LOG_DIR:
    out_path = os.path.join(LOG_DIR, "args.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)  # make sure the parent dir exists
    with open(out_path, "w") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)

# ── 1) Data Prep ------------------------------
if args.DATASET != 'FineWeb':
    raise ValueError("--DATASET must be FineWeb.")
if args.TOKENIZER != 'bpe':
    raise ValueError("--TOKENIZER must be bpe.")
assert args.TOKENIZER_PATH is not None, "TOKENIZER_PATH must be provided for BPE tokenizer."

train_ds = NPZDataset('data/FineWeb/train.npz', seq_length=args.SEQ_LEN+1, max_num_datapoints=args.TRAIN_SIZE)  # 1 extra token due to the causal shift
dev_ds = NPZDataset('data/FineWeb/val.npz', seq_length=args.SEQ_LEN+1, max_num_datapoints=args.DEV_SIZE)
dev2_ds = NPZDataset('data/FineWeb/val.npz', seq_length=args.SEQ_LEN+1, max_num_datapoints=20_000) # larger dev set for final eval
test_ds = NPZDataset('data/FineWeb/test.npz', seq_length=args.SEQ_LEN+1)  # full test set for final eval
print('Test size:', len(test_ds))

tok = FineWebBPETokenizer(vocab_size=args.VOCAB_SIZE)
tok.tokenizer = Tokenizer.from_file(args.TOKENIZER_PATH)
tok._setup_special_tokens()
PAD_IDX = tok.get_pad_id()

causal_collate_fn = create_causal_collate_fn(PAD_IDX, args.SEQ_LEN)  # simple collate for causal LM: targets are inputs shifted by 1
train_loader = DataLoader(train_ds, batch_size=args.BATCH, shuffle=True,
                          num_workers=8, pin_memory=True, prefetch_factor=2,
                          collate_fn=causal_collate_fn)
dev_loader  = DataLoader(dev_ds,  batch_size=args.BATCH, collate_fn=causal_collate_fn, shuffle=False,  num_workers=0)
dev2_loader = DataLoader(dev2_ds, batch_size=args.BATCH, collate_fn=causal_collate_fn, shuffle=False,  num_workers=0) if dev2_ds is not None else None
test_loader = DataLoader(test_ds, batch_size=args.BATCH, collate_fn=causal_collate_fn, shuffle=False, num_workers=0) if test_ds is not None else None

# Added autocast for mixed precision to reduce memory usage and speed-up training
def train_one(model, opt, loader, lr_scheduler=None, epoch=None, max_epoch=None):
    model.train()
    tot = 0
    n_batches = 0
    update_freq = max(1, len(loader) // 10)  # Update every N batches for ~10 total updates
    pbar = tqdm(
        loader, desc=f"Training", leave=False, miniters=update_freq, mininterval=0,
        disable=True if args.DISABLE_TQDM else False)
    aux = None
    diag_aux_acc = None

    for ctx, ids in pbar:
        n_batches += 1
        torch.compiler.cudagraph_mark_step_begin()  # fixing an issue with torch.compile()

        opt.zero_grad()
        ctx  = ctx.to(args.DEVICE, non_blocking=True)   # just in case
        ids  = ids.to(args.DEVICE, non_blocking=True)

        pad_mask = (ctx == PAD_IDX)
        aux = None
        with torch.amp.autocast(device_type='cuda', dtype=AUTOCAST_DTYPE):
            if (args.W_ENTROPY > 0 or args.W_BALANCE > 0 or args.DIAGNOSE) and args.MODEL == 'PrototypeAttn':
                logits, aux = model(ctx, pad_mask, return_aux=True)  # PrototypeAttn path
                ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)), ids.view(-1), ignore_index=PAD_IDX)
                aux_loss = 0.0
                if isinstance(aux, dict):
                    aux_loss += aux.get('entropy_loss', 0.0)
                    aux_loss += aux.get('balance_loss', 0.0)
                loss = ce + aux_loss
            else:
                logits = model(ctx, pad_mask)  # HF adapter / others
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), ids.view(-1), ignore_index=PAD_IDX)

        if args.DIAGNOSE and isinstance(aux, dict):
            diag_aux_acc = accumulate_aux_for_epoch(diag_aux_acc, aux)

        if USE_SCALER:
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            # BF16 doesn't need scaling
            loss.backward()
            opt.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        # Gradient checks for debugging (after backward, before zero_grad)
        if args.CHECK_GRADIENTS:
            health = comprehensive_gradient_check(model, step=n_batches, verbose=True)
            # For detecting serious issues
            if health['has_nan_inf']:
                print(f"🚨 CRITICAL: NaN/Inf gradients detected at batch {n_batches}!")
                # Optional early exit on NaN/Inf:
                # break

        tot += loss.item()
        cur_lr = opt.param_groups[0]['lr']  # track current lr for lr scheduler
        pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{cur_lr:.2e}")

    aux = finalize_aux_for_epoch(diag_aux_acc) if args.DIAGNOSE else aux

    if args.DIAGNOSE and args.MODEL == 'PrototypeAttn':
        grad_stats = collect_router_grad_stats(model)
        if epoch in [1, 3, 5, max_epoch]:
            print(f"\n📊 Router stats after epoch {epoch}/{max_epoch}:")
            if isinstance(aux, dict) and isinstance(aux.get('per_layer', None), (list, tuple)) and len(aux['per_layer']) > 0:
                for i, layer_aux in enumerate(aux['per_layer']):
                    print(f"[L{i}] ", end="")
                    print_stats(layer_aux, grad_stats=grad_stats)
            elif isinstance(aux, dict):
                print_stats(aux, grad_stats=grad_stats)                
    return tot / n_batches if n_batches > 0 else None

@torch.no_grad()
def evaluate(model, loader):
    """
    Evaluate model on loader and return loss and perplexity
    """
    model.eval()
    if args.EVAL_NO_COMPILE and hasattr(model, "_orig_mod"):
        model = model._orig_mod
    total_loss = 0.0
    for inputs, targets in loader:
        torch.compiler.cudagraph_mark_step_begin()  # fixing issue with torch.compile()
        inputs  = inputs.to(args.DEVICE, non_blocking=True)
        targets  = targets.to(args.DEVICE, non_blocking=True)
        pad_mask = (inputs == PAD_IDX)
        with torch.amp.autocast(device_type='cuda', dtype=AUTOCAST_DTYPE):
            logits = model(inputs, pad_mask)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.view(-1), ignore_index=PAD_IDX)
            total_loss += loss.item()
    avg_loss = total_loss / len(loader) if len(loader) > 0 else 0
    return avg_loss, math.exp(avg_loss) if avg_loss > 0 else float('inf')

# ── Common Training Functions ───────────────────────────────────────
def create_model(model_type):
    """Create and return the appropriate model based on model_type"""
    proto_use_mass_norm = not args.PROTO_DISABLE_MASS_NORM
    proto_use_value_low_rank = not args.PROTO_DISABLE_VALUE_LOWRANK
    if model_type == 'PrototypeAttn':
        return ProtoBroadcastLM(
            args.VOCAB_SIZE, args.BOTTLENECK, args.LAYERS, args.R, args.SEQ_LEN, TF_FFN_SIZE,
            dropout=args.DROPOUT, pad_id=PAD_IDX, tie_weights=args.TIE_HEAD,
            w_entropy=args.W_ENTROPY, w_balance=args.W_BALANCE,
            alpha_init=args.PROTO_ALPHA_INIT,
            use_mass_norm=proto_use_mass_norm,
            use_value_low_rank=proto_use_value_low_rank,
            local_conv_kernel_size=args.PROTO_KERNEL_SIZE,
            hub_dropout=args.HUB_DROPOUT,
            tau_init=args.TAU_INIT,
        ).to(args.DEVICE)
    elif model_type == 'llama':
        return create_llama31_from_args(args, tok, TF_FFN_SIZE)
    elif model_type == 'mamba':
        return create_mamba_from_args(args, pad_idx=PAD_IDX, dropout=args.DROPOUT)
    elif model_type == 'deltanet':
        from deltanet import create_deltanet_from_args
        return create_deltanet_from_args(args, PAD_IDX=PAD_IDX, TF_FFN_SIZE=TF_FFN_SIZE, tok=tok)
    else:
        raise ValueError(f'Unknown model type: {model_type}')

def profile_model_if_needed(model, model_name):
    """Profile model FLOPs if PROFILE_FLOPS is enabled"""
    global flops_per_example
    if args.PROFILE_FLOPS and install_torch_profiler_if_needed():
        # Skip profiling for DeltaNet models as they can hang during profiling
        if model_name == 'deltanet':
            print(f"⚠️  Skipping FLOP profiling for {model_name} (known profiling issues)")
            flops_per_example = 1  # placeholder to indicate skipped profiling
            return
        try:
            print(f"\n🔍 Profiling {model_name}")
            with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
                profile_result = analyze_model_first_batch(
                    model,
                    train_loader,
                    device=args.DEVICE,
                    verbose=True
                )
                # Store the FLOPs per sample in the global variable
                flops_per_example = profile_result.get('flops_per_sample', None)
        except Exception as e:
            print(f"Profiling failed: {e}")
            flops_per_example = None
        finally:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

def profile_gpu_if_needed(model, model_name):
    """Profile GPU usage if PROFILE_GPU is enabled"""
    if args.PROFILE_GPU and install_torch_profiler_if_needed():
        try:
            print(f"🔍 GPU profiling {model_name} (short run)…")
            profile_model_gpu(
                model=model,
                loader_func=train_loader,
                log_dir="./gpu_profiler_logs",
                wait=1,
                warmup=1,
                active=3,
                repeat=1,
                row_limit=20
            )
        except Exception as e:
            print(f"GPU profiling failed: {e}")

# ── Model Training ──────────────────────────────────────────────────
# Create the initial model
model = create_model(args.MODEL)
print('Params:', count_params(model))

# Load model state if a directory is provided
if args.LOAD_DIR is not None:
    model_path = os.path.join(args.LOAD_DIR, "model_state_dict.pth")
    if os.path.isfile(model_path):
        state_dict = torch.load(model_path, map_location=args.DEVICE)
        # Remove prefix "_orig_mod." obtained when using torch.compile()
        clean_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("_orig_mod."):
                clean_state_dict[k[len("_orig_mod."):]] = v
            else:
                clean_state_dict[k] = v
        model.load_state_dict(clean_state_dict)
        print(f"✅ Loaded model state from {model_path}")
    else:
        print(f"⚠️ Warning: No model_state_dict.pth found in {args.LOAD_DIR}. Starting from scratch.")

# Profile FLOPs on the eager model (before compiling) if requested
profile_model_if_needed(model, args.MODEL)

# Compile model if enabled (after FLOPs profiling to avoid interference)
model = maybe_compile(model, use_compile=args.USE_COMPILE, mode="reduce-overhead")

# Optional GPU profiling can observe the compiled model
profile_gpu_if_needed(model, args.MODEL)

# ── Training ───────────────────────────────────────────
print(f"\n🚀 Starting training for {args.MODEL}")

# Setup optimizer and training
def no_wd(name, p):
    """
    Return True for params that should NOT get weight decay.
    - norms & biases
    - scalars/vectors (e.g., LayerNorm weights, ReZero gates)
    - ProtoAttn-specific state: prototypes, gates, temperatures, EMA logits
    - Mamba-specific: dt_proj (time constants), conv1d biases, layer norms
    """
    n = name.lower()
    # standard no-decay: biases, norms, any 0D/1D tensors
    if (p.ndim <= 1) or n.endswith("bias") or (".bias" in n) or ("norm" in n):
        return True
    # proto-specific: memory/prototypes & routing/gating scalars
    if any(k in n for k in ("p_tables", "alpha", "tau", "tau_read", "logit_beta")):
        return True
    # mamba-specific: time constants and specific params that shouldn't decay
    if any(k in n for k in ("dt_proj", "dt_bias", "conv1d.bias", "norm_f")):
        return True
    return False
param_groups = [
    {'params':[p for n,p in model.named_parameters() if p.requires_grad and not no_wd(n,p)], 'weight_decay':0.1},
    {'params':[p for n,p in model.named_parameters() if p.requires_grad and     no_wd(n,p)], 'weight_decay':0.0},
]
opt1 = torch.optim.AdamW(param_groups, lr=args.LR1, fused=(args.DEVICE=='cuda'))
scaler = torch.amp.GradScaler() if USE_SCALER else None
lr_scheduler = None  # placeholder
if args.USE_LR_SCHEDULER:  # scheduler with warmup and cosine decay for LM
    num_training_steps = args.EPOCHS1 * len(train_loader)
    warmup_steps = max(1, int(0.02 * num_training_steps))  # 2% warmup
    base_lr = float(args.LR1)
    lr_min = max(0.1 * base_lr, 1e-6)
    # Warmup: ramp from ~0 to 1.0 of base LR
    warmup = LinearLR(opt1, start_factor=1e-8, end_factor=1.0, total_iters=warmup_steps)
    if args.LR_SCHEDULE == "wsd":
        # WSD: 80% stable at peak LR, then linear decay to 10% over remaining steps
        remaining = num_training_steps - warmup_steps
        stable_steps = int(0.80 * remaining)
        decay_steps = remaining - stable_steps
        stable = ConstantLR(opt1, factor=1.0, total_iters=stable_steps)
        decay = LinearLR(opt1, start_factor=1.0, end_factor=0.1, total_iters=max(1, decay_steps))
        lr_scheduler = SequentialLR(opt1, schedulers=[warmup, stable, decay], milestones=[warmup_steps, warmup_steps + stable_steps])
    else:
        cosine = CosineAnnealingLR(opt1, T_max=max(1, num_training_steps - warmup_steps), eta_min=lr_min)
        lr_scheduler = SequentialLR(opt1, schedulers=[warmup, cosine], milestones=[warmup_steps])

# Training loop
best_ppl = 1.0e100
epoch_iter_rates = []
for ep in range(args.EPOCHS1):
    print(f'Epoch {ep+1}')
    t0 = time.perf_counter()
    train_loss = train_one(model, opt1, train_loader, lr_scheduler=lr_scheduler, epoch=ep+1, max_epoch=args.EPOCHS1)
    val_loss, val_ppl = evaluate(model, dev_loader)
    dt = time.perf_counter() - t0
    n_batches = len(train_loader)
    its = n_batches / dt
    epoch_iter_rates.append((ep + 1, its))
    print(f"[{args.MODEL} P1 ep{ep+1}] train={train_loss:.4f} val_loss={val_loss:.4f} val_ppl={val_ppl:.2f}  —  {its:.1f} it/s  —  {dt:.1f} s")
    if val_ppl < best_ppl:  # Save the best model so far
        best_ppl = val_ppl
        if args.SAVE_DIR is not None:
            model_path = os.path.join(args.SAVE_DIR, "model_state_dict.pth")
            torch.save(model.state_dict(), model_path)
            print(f"✅ New best model saved to {model_path} with val_ppl={best_ppl:.2f}")
            with open(os.path.join(args.SAVE_DIR, "current_best_val_ppl.txt"), "w") as f:
                f.write(f"{best_ppl:.4f}\n")  # save the best ppl thus far to file
    elif val_ppl > best_ppl:
        print(f"⚠️ Validation perplexity increased from {best_ppl:.2f} to {val_ppl:.2f}. Early stopping.")
        break

it_s_window_start_epoch = math.floor(0.30 * args.EPOCHS1) + 1 if args.EPOCHS1 > 0 else None
it_s_window_rates = [
    rate for epoch_idx, rate in epoch_iter_rates
    if it_s_window_start_epoch is not None and epoch_idx >= it_s_window_start_epoch
]
mean_it_s_last70pct = (
    sum(it_s_window_rates) / len(it_s_window_rates)
    if it_s_window_rates else None
)
if mean_it_s_last70pct is not None:
    print(
        f"Mean it/s over epochs {it_s_window_start_epoch}-{epoch_iter_rates[-1][0]} "
        f"(last 70% window): {mean_it_s_last70pct:.2f}"
    )
else:
    print("Mean it/s over last 70% window: n/a")

# Test-evaluate on the test set if provided
if test_loader is not None and args.TEST_EVALUATE:
    test_loss, test_ppl = evaluate(model, test_loader)
    print(f"[{args.MODEL} TEST] test_loss={test_loss:.4f} test_ppl={test_ppl:.2f}")

# Evaluate on the larger dev set after the final epoch
if 'dev2_loader' in globals() and dev2_loader is not None:
    # Load the model with the best val ppl thus far (if exists) and evaluate it on the larger dev set
    if args.EPOCHS1 > 0:
        model.load_state_dict(torch.load(os.path.join(args.SAVE_DIR, "model_state_dict.pth")))
    val2_loss, final_val_ppl = evaluate(model, dev2_loader)
    print(f"[{args.MODEL} FINAL large-dev] val_loss={val2_loss:.4f} val_ppl={final_val_ppl:.2f}")
    if LOG_DIR:  # Save ppl to file
        out_path = os.path.join(LOG_DIR, "final_val_ppl.txt")
        with open(out_path, "w") as f:
            f.write(f"{final_val_ppl:.4f}\n")
        print(f"✅ Final dev ppl saved to {out_path}")
        if mean_it_s_last70pct is not None:
            it_s_path = os.path.join(LOG_DIR, "mean_it_s_last70pct.txt")
            with open(it_s_path, "w") as f:
                f.write(f"{mean_it_s_last70pct:.4f}\n")
            print(f"✅ Mean it/s saved to {it_s_path}")

# Final results
n_params = count_params(model)
print(f"\n✅ {args.MODEL} training complete!")
print(f"{args.MODEL} params: {n_params}")
torch.cuda.empty_cache()

if args.DELETE_SAVED_MODEL and args.SAVE_DIR is not None:
    model_path = os.path.join(args.SAVE_DIR, "model_state_dict.pth")
    if os.path.exists(model_path):
        os.remove(model_path)
        print(f"Deleted saved model checkpoint: {model_path}")

print(f"🎉 All artifacts saved in directory: '{args.SAVE_DIR}'")

# ── 6) Text Generation ───────────────────────────────────────
@torch.no_grad()
def generate_text_beam(model, tokenizer, prompt, max_new_tokens=120, num_beams=6,
                        top_p=1.0, top_k=50,
                        no_repeat_ngram_size=3, length_penalty=1.05, repetition_penalty=1.05,
                        early_stopping=True, num_beam_groups=1, diversity_penalty=0.0,
                        do_sample=False, temperature=1.0, min_new_tokens=60):

    cfg = HFCompatConfig(
        vocab_size=args.VOCAB_SIZE,
        pad_token_id=tok.specials['<pad>'],
        eos_token_id=tok.specials['<eos>'],
        bos_token_id=tok.specials['<bos>'],
        num_hidden_layers=args.LAYERS,
    )
    # If the trained model is an HF model wrapped by the adapter, use it directly.
    if hasattr(model, "hf"):
        hf_model = model.hf.to(args.DEVICE).eval()
        # DeltaNet requires bfloat16 for generation
        if hasattr(model, '_disable_compile') and model._disable_compile:
            hf_model = hf_model.to(dtype=torch.bfloat16)
    else:
        hf_model = HFCompatLM(model, cfg).to(args.DEVICE).eval()

    toks = tokenizer.encode(prompt)
    if toks and toks[-1] == tokenizer.specials['<eos>']:
        toks = toks[:-1]
    input_ids = torch.tensor([toks], device=args.DEVICE, dtype=torch.long)
    attention_mask = (input_ids != tok.specials['<pad>']).long()

    out = hf_model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        top_p=top_p,
        top_k=top_k,
        length_penalty=length_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
        repetition_penalty=repetition_penalty,
        early_stopping=early_stopping,
        do_sample=do_sample,
        temperature=temperature,
        num_beam_groups=num_beam_groups,
        diversity_penalty=diversity_penalty,
        return_dict_in_generate=True,
        output_scores=True,
        min_new_tokens=min_new_tokens
    )
    gen = out.sequences[0].tolist()
    new_tokens = gen[len(toks):]
    completion = tokenizer.decode(new_tokens)
    print(f"## Prompt: '{prompt}'")
    print("-" * 20)
    print(f"## Completion: '{completion}'")
    print("-" * 20)
    return completion

print("\n── Generating Sample Text ──────────────────────────")
if len(dev_ds) > 1 and not args.DISABLE_GENERATION:
    val_samples = dev_ds
    num_prompt_toks1 = len(val_samples[0]) // 2
    num_prompt_toks2 = len(val_samples[1]) // 2
    prompt1 = tok.decode(val_samples[0][:num_prompt_toks1].tolist())
    prompt2 = tok.decode(val_samples[1][:num_prompt_toks2].tolist())
    max_new_tokens1 = args.SEQ_LEN - num_prompt_toks1
    max_new_tokens2 = args.SEQ_LEN - num_prompt_toks2
    torch.compiler.set_stance("force_eager")  # disable torch.compile() to skip the overhead for quick generations
    generate_text_beam(
        model, tok, prompt=prompt1, max_new_tokens=max_new_tokens1, do_sample=True, temperature=0.8)
    generate_text_beam(
        model, tok, prompt=prompt2, max_new_tokens=max_new_tokens2, do_sample=True, temperature=0.7,
        top_p=0.92, top_k=50, no_repeat_ngram_size=4, repetition_penalty=1.15, min_new_tokens=60, num_beams=6)
    torch.compiler.set_stance("reduce_overhead")
elif args.DISABLE_GENERATION:
    print("Text generation is disabled by command-line argument.")
else:
    print("Not enough samples in the test set to generate prompts.")
