"""Reproduce DisCoVR Parametric Model experiment (Paper Table 1)."""
import sys, os, warnings
sys.path.append('/repo/src/')
os.environ['PYRO_LOG_ENABLED'] = '0'
warnings.filterwarnings('ignore')

# Force unbuffered output and tee to log file
LOG_FILE = '/repo/reproduction_output.log'
class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()
log_f = open(LOG_FILE, 'w')
sys.stdout = Tee(sys.stdout, log_f)
sys.stderr = Tee(sys.stderr, log_f)

import numpy as np
import torch
import torch.utils.data as utils
import pyro
import pyro.distributions as dist
import pyro.optim as opt
from scipy.stats import norm, truncnorm
from sklearn.naive_bayes import GaussianNB
from VAE_variants import DLVAE
from VAE_trainers import AdversarialThresholdPyroTrainer

# ─── Paper hyperparameters (Supplementary Table 16 + H.1.1) ───
N_SAMPLES   = 30000
BATCH_SIZE  = 256
LATENT_DIM  = 1
W_DIM       = 1
HIDDEN_DIM  = 8
NUM_LAYERS  = 2        # n_hidden=2
LR          = 1e-3
REC_W       = 0.75     # recon_weight
REC_Z       = 0.25     # recon_weight_z
Z_KL_W      = 0.9      # z_kl_weight
W_KL_W      = 0.2      # w_kl_weight
ADV_W       = 0.8      # adversarial_weight
PATIENCE    = 30
CONV_THRESH = 5e-3
N_REPS      = 3
MC_SAMPLES  = 2000     # MC samples for w-KL estimation

METRICS = {'nll': [], 'kl_z': [], 'kl_w': [], 'delta_bayes': []}
EPOCHS_RUN = []

def make_data(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    pyro.util.set_rng_seed(seed)
    pyro.clear_param_store()

    z_true = np.random.randn(N_SAMPLES)
    w_true = np.random.randn(N_SAMPLES)
    x_arr = z_true + w_true
    y_arr = (w_true > 0).astype(np.int64)
    y_t = torch.tensor(y_arr).float().reshape(-1, 1)
    x_t = torch.tensor(x_arr, dtype=torch.float32).unsqueeze(1)
    gt = torch.hstack((torch.FloatTensor(z_true.reshape(-1, 1)),
                        torch.FloatTensor(w_true.reshape(-1, 1))))
    ds = utils.TensorDataset(x_t, y_t, gt)
    tr, te = utils.random_split(ds, [0.5, 0.5])
    tr = utils.TensorDataset(*tr[:])
    te = utils.TensorDataset(*te[:])
    tr_ldr = utils.DataLoader(tr, shuffle=True, batch_size=BATCH_SIZE)
    te_ldr = utils.DataLoader(te, shuffle=False, batch_size=BATCH_SIZE)
    return tr_ldr, te_ldr, tr, te

def gaussian_kl(mu_q, sigma_q, mu_p, sigma_p):
    """KL(N(mu_q,sigma_q^2) || N(mu_p,sigma_p^2)), element-wise, then mean."""
    var_q = sigma_q ** 2
    var_p = sigma_p ** 2
    kl = torch.log(sigma_p / sigma_q) + (var_q + (mu_q - mu_p)**2) / (2 * var_p) - 0.5
    return kl.mean().item()

for rep in range(N_REPS):
    seed = rep
    print(f"\n{'='*60}")
    print(f"Repetition {rep+1}/{N_REPS} (seed={seed})")
    print(f"{'='*60}")

    tr_ldr, te_ldr, tr_set, te_set = make_data(seed)

    model = DLVAE(
        1, [1],
        latent_dim=LATENT_DIM, w_dim=W_DIM,
        hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS,
        recon_weight=REC_W, recon_weight_z=REC_Z,
        w_kl_weight=W_KL_W, z_kl_weight=Z_KL_W,
        adversarial_weight=ADV_W,
    )

    trainer = AdversarialThresholdPyroTrainer(
        CONV_THRESH, PATIENCE,
        1, 1,
        model, tr_ldr, te_ldr,
        opt.AdamW({"lr": LR}),
        True,  # verbose
    )
    trainer.train()
    EPOCHS_RUN.append(trainer.epochs)

    device = trainer.device
    best_model = trainer.best_model
    best_model.eval()

    # ── Get test data on device ──
    xs = te_set[:][0].to(device)
    ys = te_set[:][1].to(device)
    xs_flat = xs.squeeze()
    ys_flat = ys.squeeze()

    # ── Get encoder parameters (direct model calls) ──
    with torch.no_grad():
        z_loc, z_scale = best_model.encoder(xs)
        w_loc, w_scale = best_model.encoder_w(torch.cat((xs, ys), dim=-1))

    # ── Get reconstruction rec_w via Predictive ──
    trainer._predictive_setup(s=1)
    preds = trainer.get_variables('test')
    rec_w = preds['rec_w'][0, 0].cpu()  # combined reconstruction

    # ── NLL ──
    nll_dist = dist.Normal(rec_w.mean(), rec_w.std())
    nll = -nll_dist.log_prob(xs_flat.cpu()).mean().item()

    # ── D_KL(q_z|x || p_z|x) ──
    # True posterior: p(z|x) = N(x/2, 1/2)
    mu_p_z = xs_flat / 2.0
    sigma_p_z = torch.full_like(z_scale.squeeze(), np.sqrt(0.5))
    kl_z = gaussian_kl(z_loc.squeeze(), z_scale.squeeze(), mu_p_z, sigma_p_z)

    # ── D_KL(q_w|x,y || p_w|x,y) ──
    # True: truncated N(x/2, 1/2), truncation based on y sign
    mu_p_w = xs_flat.cpu().numpy() / 2.0
    sigma_p_w_val = np.sqrt(0.5)
    w_loc_np = w_loc.squeeze().cpu().numpy()
    w_scale_np = w_scale.squeeze().cpu().numpy()
    ys_np = ys_flat.cpu().numpy()
    n_test = xs.shape[0]

    kl_w_total = 0.0
    n_inf = 0
    for i in range(n_test):
        mu_qi = w_loc_np[i]
        sigma_qi = max(w_scale_np[i], 1e-8)
        mu_pi = mu_p_w[i]
        yi = ys_np[i]

        if yi == 1:
            a, b = (0 - mu_pi) / sigma_p_w_val, np.inf
        else:
            a, b = -np.inf, (0 - mu_pi) / sigma_p_w_val

        # Sample from q and estimate KL
        w_s = np.random.normal(mu_qi, sigma_qi, MC_SAMPLES)
        log_q = norm.logpdf(w_s, mu_qi, sigma_qi)
        log_p = truncnorm.logpdf(w_s, a, b, loc=mu_pi, scale=sigma_p_w_val)
        # Clamp -inf log_p values (samples outside truncation bounds)
        finite_mask = np.isfinite(log_p)
        if not finite_mask.all():
            n_inf += 1
            # Use a large penalty for samples outside bounds
            log_p = np.where(finite_mask, log_p, -50.0)
        kl_i = np.mean(log_q - log_p)
        kl_w_total += kl_i

    kl_w = kl_w_total / n_test
    if n_inf > 0:
        print(f"  WARNING: {n_inf}/{n_test} samples had mass outside truncation bounds")

    # ── Delta-Bayes ──
    xs_np = xs_flat.cpu().numpy().reshape(-1, 1)
    ys_np_2d = ys_flat.cpu().numpy().ravel()
    orig_score = GaussianNB().fit(xs_np, ys_np_2d).score(xs_np, ys_np_2d)
    z_np = z_loc.squeeze().cpu().numpy().reshape(-1, 1)
    w_np = w_loc.squeeze().cpu().numpy().reshape(-1, 1)
    combined = np.concatenate((z_np, w_np), axis=1)
    latent_score = GaussianNB().fit(combined, ys_np_2d).score(combined, ys_np_2d)
    delta_bayes = abs(orig_score - latent_score)

    METRICS['nll'].append(nll)
    METRICS['kl_z'].append(kl_z)
    METRICS['kl_w'].append(kl_w)
    METRICS['delta_bayes'].append(delta_bayes)

    print(f"  NLL={nll:.4f}, KL_z={kl_z:.4f}, KL_w={kl_w:.4f}, dBayes={delta_bayes:.4f}")

# ─── Final report ───
print(f"\n{'='*60}")
print(f"FINAL RESULTS (over {N_REPS} repetitions)")
print(f"{'='*60}")
targets = {'nll': 1.769, 'kl_z': 0.17, 'kl_w': 10.10, 'delta_bayes': 0.1}
print(f"{'Metric':20s} {'Mean':>10s} {'Std':>10s}  {'Target':>10s}")
print(f"{'-'*20} {'-'*10} {'-'*10}  {'-'*10}")
for k, v in METRICS.items():
    arr = np.array(v)
    print(f"  {k:20s} {arr.mean():10.4f} {arr.std():10.4f}  {targets[k]:10.3f}")
print(f"\n  Epochs: {np.mean(EPOCHS_RUN):.1f} +/- {np.std(EPOCHS_RUN):.1f} (range: {np.min(EPOCHS_RUN)}-{np.max(EPOCHS_RUN)})")
