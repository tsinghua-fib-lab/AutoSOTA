#!/usr/bin/env python3
"""
Ablation 5: Hodge Spectral Component Decomposition

Clean 3-channel design from de Rham complex theory:

  c0 slot = c0_harmonic  (harmonic Φ₀ modes, DC/topology)     → Harmonic channel
  c1 slot = grad_c0      (c0 @ Md0^T, gradient via d₀)        → Exact channel
  c2 slot = c2_enriched  (curl + curvature 2-form via δ₁)     → Coexact channel

Key design choices:
  - Raw c1 removed (100% correlated with grad_c0, causes capacity imbalance)
  - Each channel per-mode standardized to zero mean unit variance
  - Model Md0=0 internally (gradient comes from pre-computed c0 slot)
  - Spatial branch disabled (x_raw=0)

This gives balanced exact:coexact capacity and clean Hodge separation.

Usage:
    python -m ablations.hodge_component_ablation
"""
import os, sys, json, argparse, pickle, copy
import numpy as np
import torch
from scipy.sparse import eye as speye
from scipy.sparse.linalg import spsolve
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'hodge-spectral-operator'))

from hodge_spectral.operators import HighOrderSpectralOperators
from hodge_spectral.models.unified import UnifiedHSD, train_unified_hsd
from hodge_spectral.models.dataset import VectorFluxMapper
from hodge_spectral.utils import LpLoss
from configs.experiment_config import DATASETS, HSD_CONFIG, DEVICE


class SimpleDS(Dataset):
    def __init__(s, c0, c1, c2, yr, n_nodes):
        s.c0, s.c1, s.c2, s.yr, s.n = c0, c1, c2, yr, n_nodes
    def __len__(s): return len(s.c0)
    def __getitem__(s, i):
        xr = np.zeros(s.n, dtype=np.float32)
        return (torch.from_numpy(s.c0[i]), torch.from_numpy(s.c1[i]),
                torch.from_numpy(s.c2[i]), torch.zeros(1),
                torch.from_numpy(xr), torch.from_numpy(s.yr[i]))


def load_data(name):
    cfg = DATASETS[name]
    with open(os.path.join(cfg['data_dir'],
              [f for f in os.listdir(cfg['data_dir']) if f.endswith('.pkl')][0]), 'rb') as f:
        return pickle.load(f)


class HodgeProjector:
    def __init__(self, host):
        self.B1, self.B2 = host.B1, host.B2
        eps = 1e-8
        self.L0r = (self.B1.T @ self.B1 + eps * speye(self.B1.shape[1])).tocsr()
        self.L2r = (self.B2 @ self.B2.T + eps * speye(self.B2.shape[0])).tocsr()
    def project(self, w):
        we = np.array(self.B1 @ spsolve(self.L0r, self.B1.T @ w)).flatten()
        wc = np.array(self.B2.T @ spsolve(self.L2r, self.B2 @ w)).flatten()
        return we, wc, w - we - wc


def comp_errors(proj, mapper, pred, gt, n=50):
    errs = {'exact':[], 'coexact':[], 'harmonic':[], 'total':[]}
    for i in range(min(n, len(pred))):
        fp, fg = mapper.node_vector_to_edge_flux(pred[i]), mapper.node_vector_to_edge_flux(gt[i])
        pe,pc,ph = proj.project(fp); ge,gc,gh = proj.project(fg)
        for k,a,b in [('exact',pe,ge),('coexact',pc,gc),('harmonic',ph,gh)]:
            nb = np.linalg.norm(b)
            if nb > 1e-10: errs[k].append(np.linalg.norm(a-b)/nb)
        nt = np.linalg.norm(fg)
        if nt > 1e-10: errs['total'].append(np.linalg.norm(fp-fg)/nt)
    return {k: round(float(np.mean(v)),4) if v else float('nan') for k,v in errs.items()}


def per_mode_standardize(train, *others):
    """Standardize each mode (column) to zero mean, unit variance using train stats."""
    mu = np.mean(train, axis=0, keepdims=True)
    std = np.std(train, axis=0, keepdims=True) + 1e-10
    result = [(train - mu) / std]
    for arr in others:
        result.append((arr - mu) / std)
    return result


# Variants: (name, use_harmonic, use_exact, use_coexact)
VARIANTS = [
    ("Full Hodge",                True,  True,  True),
    ("Harmonic + Exact",          True,  True,  False),
    ("Harmonic + Coexact",        True,  False, True),
    ("Exact + Coexact",           False, True,  True),
    ("Harmonic only",             True,  False, False),
    ("Exact only",                False, True,  False),
    ("Coexact only",              False, False, True),
]


def run(dataset_name):
    print(f"\n{'='*60}")
    print(f"Hodge 3-Channel Ablation ({dataset_name})")
    print(f"{'='*60}")

    data = load_data(dataset_name)
    pts = data['points'].astype(np.float64)
    fcs = data['faces'].astype(np.int64)
    n = len(pts); k = 64

    host = HighOrderSpectralOperators(pts, fcs, k_list=(k, k, k))
    mapper = VectorFluxMapper(pts, fcs)
    proj = HodgeProjector(host)
    Phi0 = host.Phi0[:, :k]
    Md0 = host.Md0.astype(np.float32)

    # Identify harmonic Phi0 modes
    L0_Phi0 = host.L0.dot(Phi0)
    eigs0 = np.sum(Phi0 * L0_Phi0, axis=0)
    harm_mask = eigs0 < 1e-6
    n_harm = int(harm_mask.sum())
    print(f"  Phi0 harmonic modes: {n_harm} (b₀ topology)")

    # Data splits
    X, Y = data['X_data'], data['Y_data']
    Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_size=0.2, random_state=42)
    Xtr, Xva, Ytr, Yva = train_test_split(Xtr, Ytr, test_size=0.15, random_state=42)
    xs = np.max(np.abs(Xtr)) + 1e-9
    ys = np.max(np.abs(Ytr)) + 1e-9

    # --- Compute 3 Hodge channels ---
    def compute_channels(Xn, Yn):
        Xn2 = (Xn / xs).astype(np.float64)
        Yn2 = (Yn / ys).astype(np.float32)

        c0_full = (Xn2 @ Phi0).astype(np.float32)

        # HARMONIC channel: only harmonic Phi0 modes (DC/global topology)
        c0_harm = np.zeros_like(c0_full)
        c0_harm[:, harm_mask] = c0_full[:, harm_mask]

        # EXACT channel: grad_c0 = c0 @ Md0^T (gradient via d₀)
        # Uses full c0 (including non-harmonic) through the gradient operator
        grad_c0 = c0_full @ Md0.T  # (B, k)

        # COEXACT channel: enriched 2-form (curl + curvature via δ₁)
        _, _, c2e = host.compute_enriched_input(Xn2)
        c2e = c2e[:, :k].astype(np.float32)

        return c0_harm, grad_c0, c2e, Yn2

    h_tr, e_tr, c_tr, y_tr = compute_channels(Xtr, Ytr)
    h_va, e_va, c_va, y_va = compute_channels(Xva, Yva)
    h_te, e_te, c_te, y_te = compute_channels(Xte, Yte)

    # Per-mode standardization (fit on train)
    h_tr, h_va, h_te = per_mode_standardize(h_tr, h_va, h_te)
    e_tr, e_va, e_te = per_mode_standardize(e_tr, e_va, e_te)
    c_tr, c_va, c_te = per_mode_standardize(c_tr, c_va, c_te)

    # Effective ranks after standardization
    def eff_rank(X):
        s = np.linalg.svd(X[:200], compute_uv=False)
        s = s / (s.sum() + 1e-20)
        return np.exp(-np.sum(s * np.log(s + 1e-20)))

    print(f"  Effective ranks: harmonic={eff_rank(h_tr):.1f}, "
          f"exact={eff_rank(e_tr):.1f}, coexact={eff_rank(c_tr):.1f}")

    # GT Hodge energy
    energies_e, energies_c = [], []
    for i in range(min(30, len(y_te))):
        fl = mapper.node_vector_to_edge_flux(y_te[i])
        fe, fc, _ = proj.project(fl)
        tot = np.linalg.norm(fl)**2 + 1e-20
        energies_e.append(np.linalg.norm(fe)**2/tot)
        energies_c.append(np.linalg.norm(fc)**2/tot)
    print(f"  GT energy: Exact={np.mean(energies_e):.1%} Coexact={np.mean(energies_c):.1%}")

    # --- Model config ---
    # Compact model to amplify channel differences
    # Md0/Mdelta1 active for de Rham cross-term coupling
    ablation_hidden = (96, 64)
    ablation_res = 48
    ablation_epochs = 150

    results = {}
    crit = LpLoss()

    for name, use_h, use_e, use_c in VARIANTS:
        print(f"\n--- {name} ---")

        def apply(h, e, c):
            return (h.copy() if use_h else np.zeros_like(h),
                    e.copy() if use_e else np.zeros_like(e),
                    c.copy() if use_c else np.zeros_like(c))

        htr, etr, ctr = apply(h_tr, e_tr, c_tr)
        hva, eva, cva = apply(h_va, e_va, c_va)
        hte, ete, cte = apply(h_te, e_te, c_te)

        trl = DataLoader(SimpleDS(htr, etr, ctr, y_tr, n),
                         batch_size=HSD_CONFIG['batch_size'], shuffle=True)
        val = DataLoader(SimpleDS(hva, eva, cva, y_va, n),
                         batch_size=HSD_CONFIG['batch_size'])
        tel = DataLoader(SimpleDS(hte, ete, cte, y_te, n),
                         batch_size=HSD_CONFIG['batch_size'])

        Phi_out = torch.from_numpy(Phi0.astype(np.float32)).to(DEVICE)
        model = UnifiedHSD(
            host, 1, k, k, k, n, k, Phi_out,
            hidden_dims=ablation_hidden,
            res_hidden=ablation_res,
        ).to(DEVICE)

        model = train_unified_hsd(
            model, trl, val,
            epochs=ablation_epochs, device=DEVICE,
            lr=HSD_CONFIG['lr'], patience=HSD_CONFIG['patience'])

        model.eval()
        ap, ag = [], []
        with torch.no_grad():
            for c0,c1,c2,_,xr,yr in tel:
                p,_,_ = model(c0.to(DEVICE),c1.to(DEVICE),c2.to(DEVICE),xr.to(DEVICE))
                ap.append(p.cpu()); ag.append(yr)
        pt, gt = torch.cat(ap), torch.cat(ag)
        rl = crit(pt, gt).item()
        ce = comp_errors(proj, mapper, pt.numpy(), gt.numpy())

        results[name] = {
            'rel_l2': round(rl, 4),
            'exact_err': ce['exact'], 'coexact_err': ce['coexact'],
            'harmonic_err': ce['harmonic'],
        }
        print(f"  Rel L2: {rl:.4f} | E={ce['exact']:.4f} C={ce['coexact']:.4f} H={ce['harmonic']:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='all', choices=['all','ellipsoid','torus'])
    args = parser.parse_args()
    os.makedirs('./output/results', exist_ok=True)

    datasets = {'all': ['ellipsoid_aero','torus_helmholtz'],
                'ellipsoid': ['ellipsoid_aero'], 'torus': ['torus_helmholtz']}[args.dataset]

    final = {}
    for ds in datasets:
        short = 'Ellipsoid' if 'ellipsoid' in ds else 'Torus'
        final[short] = run(ds)

    print(f"\n{'='*75}")
    print("SUMMARY: Hodge Spectral Component Ablation")
    print(f"{'='*75}")
    for ds, res in final.items():
        print(f"\n  {ds}:")
        print(f"  {'Basis Components':<24s} {'Rel L2':>8s} {'Exact':>8s} {'Coexact':>8s} {'Harmon':>8s}")
        print(f"  {'-'*56}")
        for nm, r in sorted(res.items(), key=lambda x: x[1]['rel_l2']):
            print(f"  {nm:<24s} {r['rel_l2']:>8.4f} {r['exact_err']:>8.4f} "
                  f"{r['coexact_err']:>8.4f} {r['harmonic_err']:>8.4f}")

    with open('./output/results/hodge_component_ablation.json','w') as f:
        json.dump(final, f, indent=2)
    print(f"\nSaved.")


if __name__ == "__main__":
    main()
