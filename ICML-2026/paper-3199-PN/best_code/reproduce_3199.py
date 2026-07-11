"""Reproduction script for paper 3199: N-bit Parity Validation Accuracy.
Configurable via environment variables for SOTA optimization iterations.
"""
import torch, numpy as np, sys, time, json, os
from datetime import timedelta

sys.path.insert(0, '/repo')
from models.layers.product import MultiBinaryProductLayer

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
print(f'Using device: {device}')

# --- Configuration via env vars ---
SEED = int(os.environ.get('SOTA_SEED', '42'))
N = int(os.environ.get('SOTA_N', '16'))
n_outputs = int(os.environ.get('SOTA_N_OUTPUTS', '250'))
p_e = float(os.environ.get('SOTA_P_E', str(1.0 / N)))
p_w = float(os.environ.get('SOTA_P_W', '0.5'))
batch_size = int(os.environ.get('SOTA_BATCH_SIZE', '100'))
max_steps = int(os.environ.get('SOTA_MAX_STEPS', '10000'))
lr = float(os.environ.get('SOTA_LR', '0.02'))
chunk = int(os.environ.get('SOTA_CHUNK', '4096'))
init_type = os.environ.get('SOTA_INIT_TYPE', 'gaussian')
gaussian_mean = float(os.environ.get('SOTA_GAUSSIAN_MEAN', '0.5'))
gaussian_std = float(os.environ.get('SOTA_GAUSSIAN_STD', '0.25'))
use_momentum = os.environ.get('SOTA_MOMENTUM', '0')
use_nesterov = os.environ.get('SOTA_NESTEROV', '0')
use_curriculum = os.environ.get('SOTA_CURRICULUM', '0')
curriculum_start_factor = float(os.environ.get('SOTA_CURRICULUM_START_FACTOR', '0.25'))
curriculum_end_step = int(os.environ.get('SOTA_CURRICULUM_END_STEP', '500'))
use_lr_schedule = os.environ.get('SOTA_LR_SCHEDULE', 'none')
lr_warmup_steps = int(os.environ.get('SOTA_LR_WARMUP_STEPS', '100'))
lr_min_factor = float(os.environ.get('SOTA_LR_MIN_FACTOR', '0.25'))
use_grad_clip = os.environ.get('SOTA_GRAD_CLIP', '0')
use_convergence_detect = os.environ.get('SOTA_CONVERGENCE_DETECT', '0')
streaming_eval = os.environ.get('SOTA_STREAMING_EVAL', '0')

tt_size = 2 ** N
torch.manual_seed(SEED); np.random.seed(SEED)

sep = '=' * 80
print(f'\n{sep}')
print(f'REPRODUCTION: {N}-bit Parity, Sparse Product Node')
print(sep)
print(f'N={N} n_outputs={n_outputs} p_e={p_e:.6f} p_w={p_w} batch={batch_size} max_steps={max_steps} lr={lr} seed={SEED}')
print(f'init_type={init_type} curriculum={use_curriculum} lr_schedule={use_lr_schedule} momentum={use_momentum} nesterov={use_nesterov} grad_clip={use_grad_clip} streaming={streaming_eval}')
print(f'{sep}\n')

# Oracle setup
print('Setting up oracle...')
oracle = MultiBinaryProductLayer(n_outputs=n_outputs, hard_step=True).to(device)
_ = oracle(torch.zeros(1, N, device=device))
oracle.product_weights.data.copy_((torch.rand(N, n_outputs, device=device) < p_w).float())

# Generate truth table
indices = torch.arange(tt_size, dtype=torch.int64)
tt = ((indices.unsqueeze(1) >> torch.arange(N-1,-1,-1)) & 1).float().to(device)

if streaming_eval == '1':
    # For large N: compute tt_labels on-the-fly during eval to save GPU memory
    print(f'Using streaming eval (tt_labels computed per chunk)')
    tt_labels = None
    eval_chunk = min(chunk, 2048)
else:
    tt_labels = torch.cat([oracle(tt[i:i+chunk]) for i in range(0, tt_size, chunk)], dim=0)
    eval_chunk = chunk
    print(f'Oracle ready. TT labels shape: {tt_labels.shape}')

# Model with custom init
if init_type == 'rademacher':
    model = MultiBinaryProductLayer(n_outputs=n_outputs, hard_step=False, use_gaussian_init=False).to(device)
    _ = model(torch.zeros(1, N, device=device))
    rad = (torch.randint(0, 2, (N, n_outputs), device=device).float() * 2 - 1)
    rad_scaled = rad * 0.5 + 0.5
    model.product_weights.data.copy_(rad_scaled)
    print(f'Initialized with Rademacher (binary +-1 -> {{0,1}})')
elif init_type == 'rademacher_noisy':
    model = MultiBinaryProductLayer(n_outputs=n_outputs, hard_step=False, use_gaussian_init=False).to(device)
    _ = model(torch.zeros(1, N, device=device))
    rad = (torch.randint(0, 2, (N, n_outputs), device=device).float() * 2 - 1)
    rad_scaled = rad * 0.5 + 0.5
    noise = torch.randn(N, n_outputs, device=device) * 0.01
    model.product_weights.data.copy_(torch.clamp(rad_scaled + noise, 0.0, 1.0))
    print(f'Initialized with Rademacher + small Gaussian noise')
else:
    model = MultiBinaryProductLayer(n_outputs=n_outputs, hard_step=False,
                                     use_gaussian_init=True,
                                     gaussian_mean=gaussian_mean,
                                     gaussian_std=gaussian_std).to(device)
    _ = model(torch.zeros(1, N, device=device))
    print(f'Initialized with Gaussian(mean={gaussian_mean}, std={gaussian_std})')

# Optimizer
momentum_val = float(use_momentum)
nesterov_bool = bool(int(use_nesterov))
if momentum_val > 0:
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum_val, nesterov=nesterov_bool)
    print(f'Optimizer: SGD(lr={lr}, momentum={momentum_val}, nesterov={nesterov_bool})')
else:
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    print(f'Optimizer: SGD(lr={lr})')

# LR scheduler
scheduler = None
if use_lr_schedule == 'cosine':
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_steps, eta_min=lr * lr_min_factor)
    print(f'LR Scheduler: CosineAnnealing(T_max={max_steps}, eta_min={lr*lr_min_factor:.6f})')
elif use_lr_schedule == 'warmup_cosine':
    warmup = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.05, total_iters=lr_warmup_steps)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_steps - lr_warmup_steps, eta_min=lr * lr_min_factor)
    scheduler = torch.optim.lr_scheduler.SequentialLR(opt, schedulers=[warmup, cosine], milestones=[lr_warmup_steps])
    print(f'LR Scheduler: Warmup({lr_warmup_steps} steps) + CosineAnnealing(eta_min={lr*lr_min_factor:.6f})')

loss_fn = lambda p, t: ((p - t) ** 2).sum(dim=1).mean()

# Eval steps
n_pts = int(np.ceil(np.log10(max_steps) * 5)) + 1
log_steps = np.unique(np.round(np.logspace(0, np.log10(max_steps), n_pts)).astype(int))
log_steps = log_steps[(log_steps >= 1) & (log_steps <= max_steps)]
extra = [100, 200, 500, 1000, 2000, 5000, 10000]
extra = [m for m in extra if m <= max_steps]
log_set = set(log_steps.tolist()); log_set.update(extra)
eval_steps = np.unique(sorted(log_set)); eval_set = set(eval_steps.tolist())

powers = (2 ** np.arange(N-1, -1, -1)).astype(np.int64)
seen = np.zeros(tt_size, dtype=bool)

convergence_detected = False
convergence_step = None
convergence_coverage = None
prev_weights = None
stable_count = 0

@torch.no_grad()
def eval_full(model):
    if streaming_eval == '1':
        total_err = 0.0
        for i in range(0, tt_size, eval_chunk):
            end = min(i + eval_chunk, tt_size)
            tt_chunk = tt[i:end]
            tt_lbl = oracle(tt_chunk)
            preds = model(tt_chunk).reshape(end - i, n_outputs)
            total_err += ((preds > 0.5).float() != tt_lbl).float().sum().item()
        return 100.0 * (1.0 - total_err / (tt_size * n_outputs))
    else:
        preds = torch.cat([model(tt[i:i+eval_chunk]) for i in range(0, tt_size, eval_chunk)])
        preds = preds.reshape(tt_size, n_outputs)
        err = ((preds > 0.5).float() != tt_labels).float().sum().item() / tt_labels.shape[1]
        return 100.0 * (1.0 - err / tt_size)

t0 = time.time(); val_accs = {}; final_val = None
print(f'Training... ({len(eval_steps)} eval points)')

for step in range(1, max_steps + 1):
    if use_curriculum == '1':
        if step <= curriculum_end_step:
            progress = step / curriculum_end_step
            curr_p_e = p_e * (curriculum_start_factor + (1.0 - curriculum_start_factor) * progress)
        else:
            curr_p_e = p_e
    else:
        curr_p_e = p_e

    x = (torch.rand(batch_size, N, device=device) < curr_p_e).float()
    y = oracle(x).detach()
    seen[(x.cpu().numpy() @ powers).astype(int)] = True
    opt.zero_grad()
    p = model(x).reshape(batch_size, n_outputs)
    loss = loss_fn(p, y)
    loss.backward()

    grad_clip_val = float(use_grad_clip)
    if grad_clip_val > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_val)

    opt.step()
    if scheduler is not None:
        scheduler.step()

    if use_convergence_detect == '1' and not convergence_detected and step % 50 == 0:
        with torch.no_grad():
            curr_w = model.product_weights.data.flatten()
            if prev_weights is not None:
                corr = torch.corrcoef(torch.stack([prev_weights, curr_w]))[0,1].item()
                if corr > 0.999:
                    stable_count += 1
                    if stable_count >= 3:
                        convergence_detected = True
                        va = eval_full(model)
                        val_accs[step] = va; final_val = va
                        cov = seen.sum() / tt_size
                        convergence_step = step
                        convergence_coverage = cov
                        print(f'  CONVERGENCE DETECTED at step {step}: val_acc={va:.2f}% cov={cov:.4f}')
            prev_weights = curr_w

    if step in eval_set:
        va = eval_full(model)
        val_accs[step] = va; final_val = va
        cov = seen.sum() / tt_size
        if convergence_step is None and va >= 100.0:
            convergence_step = step
            convergence_coverage = cov
        if step in extra:
            eta = timedelta(seconds=int((time.time()-t0)/step*(max_steps-step))) if step > 0 else '?'
            print(f'  step {step:>6d} | loss={loss.item():.6f} | val_acc={va:.2f}% | cov={cov:.4f} | elap={timedelta(seconds=int(time.time()-t0))} | ETA={eta}')

elapsed = time.time() - t0
print(f'\n{sep}')
print('RESULTS')
print(f'{sep}')
print(f'Final val_acc at step {max_steps}: {final_val:.2f}%')
if convergence_step is not None:
    print(f'First 100% accuracy at step {convergence_step}')
    print(f'Coverage at convergence: {convergence_coverage:.4f}')
print(f'Total time: {timedelta(seconds=int(elapsed))}')

for m in extra:
    if m in val_accs:
        print(f'  step {m:>6d}: val_acc={val_accs[m]:.2f}%')

results = {'paper_id': 3199, 'metric': 'Validation Accuracy', 'value': float(final_val),
    'N': N, 'n_outputs': n_outputs, 'p_e': p_e, 'p_w': p_w,
    'batch_size': batch_size, 'max_steps': max_steps, 'lr': lr, 'seed': SEED,
    'init_type': init_type,
    'val_acc_by_step': {str(k): float(v) for k, v in val_accs.items()},
    'elapsed_seconds': elapsed,
    'convergence_step': convergence_step,
    'convergence_coverage': float(convergence_coverage) if convergence_coverage is not None else None}
with open('/repo/reproduce_3199_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print('\nResults saved to /repo/reproduce_3199_results.json')
