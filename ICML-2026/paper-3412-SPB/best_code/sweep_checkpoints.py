"""Post-hoc sigma sweep for top checkpoints."""
import sys, os, torch, numpy as np, math
from torch.utils.data import DataLoader
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.params import get_flat_params, set_flat_params
from utils.pacbayes_utils import kl_diag_gaussians, mcallester_bound
from create_data.dataset import TransformedDataset
from models.models_mnist import EquivariantCNN

DATA_DIR = "create_data/rot_mnist"
DEVICE = "cpu"
DELTA = 0.05
S = 50

def egibbs(model, mu, sq, loader, dev, S):
    torch.manual_seed(0)
    model.eval()
    mu = mu.to(dev)
    losses = []
    for s in range(S):
        w = mu + torch.randn_like(mu) * sq
        inc, tot = 0, 0
        for b in loader:
            im, lb = b["x"].to(dev), b["y"].to(dev)
            set_flat_params(model, w)
            with torch.no_grad():
                inc += (model(im).argmax(dim=1) != lb).sum().item()
            tot += lb.size(0)
        losses.append(inc / tot)
    l = np.array(losses)
    return float(l.mean()), float(l.std(ddof=1) / math.sqrt(len(l)))

def main():
    tl = DataLoader(TransformedDataset(f"{DATA_DIR}/train.pt"), 256)
    tsl = DataLoader(TransformedDataset(f"{DATA_DIR}/test.pt"), 256)
    pd = torch.load(f"{DATA_DIR}/prior_mu_equivariant.pt", map_location=DEVICE)
    n_train = len(tl.dataset)

    checkpoints = [
        ("cosine_lr1", "equivariant_cosine.pt"),
        ("klreg_l01", "equivariant_kl01.pt"),
    ]

    sigmas = [0.03, 0.04, 0.045, 0.05, 0.055, 0.06, 0.07, 0.08]

    for ckpt_name, ckpt_path in checkpoints:
        print(f"\n=== Sigma sweep for {ckpt_name} ===")
        m = EquivariantCNN().to(DEVICE)
        m.load_state_dict(torch.load(f"{DATA_DIR}/{ckpt_path}", map_location=DEVICE))
        mu_q = get_flat_params(m)
        mu_p = pd["mu"]

        header = f"{'sigma':>8s}  {'KL':>8s}  {'train_r':>8s}  {'complx':>8s}  {'bound':>8s}  {'test_r':>8s}"
        print(header)
        best_b, best_s = float("inf"), None
        for sigma in sigmas:
            kl = kl_diag_gaussians(mu_q, sigma, mu_p, sigma)
            tr, _ = egibbs(m, mu_q, sigma, tl, DEVICE, S)
            bd, cx = mcallester_bound(tr, kl, n_train, DELTA)
            ter, _ = egibbs(m, mu_q, sigma, tsl, DEVICE, S)
            mk = " <--" if bd < best_b else ""
            if bd < best_b:
                best_b, best_s = bd, sigma
            print(f"{sigma:8.4f}  {kl:8.1f}  {tr:8.4f}  {cx:8.4f}  {bd:8.4f}  {ter:8.4f}{mk}")
        print(f"Best: sigma={best_s} bound={best_b:.4f}")

if __name__ == "__main__":
    main()
