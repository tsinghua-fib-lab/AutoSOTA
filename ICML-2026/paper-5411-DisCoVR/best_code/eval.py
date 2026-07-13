"""Official evaluation script for DisCoVR Parametric Model reproduction."""
import sys, os, warnings
sys.path.append('/repo/src/')
os.environ['PYRO_LOG_ENABLED'] = '0'
warnings.filterwarnings('ignore')

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

# Paper hyperparameters (Supplementary Table 16 + Appendix H.1.1)
N_SAMPLES, BATCH_SIZE = 30000, 256
LATENT_DIM, W_DIM = 1, 1
HIDDEN_DIM, NUM_LAYERS = 8, 2     # n_hidden=2, d_hidden=8
LR = 1e-3
REC_W, REC_Z = 0.75, 0.25          # recon_weight, recon_weight_z
Z_KL_W, W_KL_W = 0.9, 0.2          # z_kl_weight, w_kl_weight
ADV_W = 0.8                         # adversarial_weight
PATIENCE, CONV_THRESH = 30, 5e-3
N_REPS = 3

def make_data(seed):
    np.random.seed(seed); torch.manual_seed(seed); pyro.util.set_rng_seed(seed)
    pyro.clear_param_store()
    z_true = np.random.randn(N_SAMPLES); w_true = np.random.randn(N_SAMPLES)
    x_arr = z_true + w_true; y_arr = (w_true > 0).astype(np.int64)
    y_t = torch.tensor(y_arr).float().reshape(-1,1)
    x_t = torch.tensor(x_arr, dtype=torch.float32).unsqueeze(1)
    gt = torch.hstack((torch.FloatTensor(z_true.reshape(-1,1)), torch.FloatTensor(w_true.reshape(-1,1))))
    ds = utils.TensorDataset(x_t, y_t, gt)
    tr, te = utils.random_split(ds, [0.5, 0.5])
    tr, te = utils.TensorDataset(*tr[:]), utils.TensorDataset(*te[:])
    tr_ldr = utils.DataLoader(tr, shuffle=True, batch_size=BATCH_SIZE)
    te_ldr = utils.DataLoader(te, shuffle=False, batch_size=BATCH_SIZE)
    return tr_ldr, te_ldr, tr, te

results = {'nll': [], 'delta_bayes': []}
for rep in range(N_REPS):
    seed = rep
    print(f"Rep {rep+1}/{N_REPS} (seed={seed})")
    tr_ldr, te_ldr, tr_set, te_set = make_data(seed)
    model = DLVAE(1, [1], latent_dim=LATENT_DIM, w_dim=W_DIM, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS,
        recon_weight=REC_W, recon_weight_z=REC_Z, w_kl_weight=W_KL_W, z_kl_weight=Z_KL_W, adversarial_weight=ADV_W)
    trainer = AdversarialThresholdPyroTrainer(CONV_THRESH, PATIENCE, 1, 1, model, tr_ldr, te_ldr, opt.AdamW({'lr': LR}), True, cycle_kl=True, cycle_length=10, z_kl_range=(0.5, 0.9), w_kl_range=(0.1, 0.2))
    trainer.contrastive_weight = 0.05
    trainer.contrastive_tau = 0.1
    trainer.train()
    print(f"  Epochs: {trainer.epochs}")
    device = trainer.device; trainer.best_model.eval()
    xs = te_set[:][0].to(device)
    trainer._predictive_setup(s=1); preds = trainer.get_variables('test')
    rec_w = preds['rec_w'][0, 0].cpu()
    nll = -dist.Normal(rec_w.mean(), rec_w.std()).log_prob(xs.squeeze().cpu()).mean().item()
    xs_np = xs.squeeze().cpu().numpy().reshape(-1,1); ynr = te_set[:][1].squeeze().cpu().numpy().ravel()
    orig_score = GaussianNB().fit(xs_np, ynr).score(xs_np, ynr)
    with torch.no_grad():
        z_loc, _ = trainer.best_model.encoder(xs); w_loc, _ = trainer.best_model.encoder_w(torch.cat((xs, te_set[:][1].to(device)), dim=-1))
    combined = np.concatenate((z_loc.squeeze().cpu().numpy().reshape(-1,1), w_loc.squeeze().cpu().numpy().reshape(-1,1)), axis=1)
    latent_score = GaussianNB().fit(combined, ynr).score(combined, ynr)
    delta_bayes = abs(orig_score - latent_score)
    results['nll'].append(nll); results['delta_bayes'].append(delta_bayes)
    print(f"  NLL={nll:.4f}, Delta-Bayes={delta_bayes:.4f}")

print(f"\n{'='*50}")
print(f"FINAL: NLL={np.mean(results['nll']):.4f}+/-{np.std(results['nll']):.4f}")
print(f"FINAL: Delta-Bayes={np.mean(results['delta_bayes']):.4f}+/-{np.std(results['delta_bayes']):.4f}")
print(f"{'='*50}")
