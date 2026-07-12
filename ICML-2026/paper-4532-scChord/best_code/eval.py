#!/usr/bin/env python3
"""
scChord Evaluation Script for Paper Reproduction.
Loads trained models and evaluates on GSE100866 test set.

Usage:
    python3 eval.py --data_path /repo/data/GSE100866_CBMC.h5ad \
        --vae_path /repo/outputs_stage1_gauss/vae_best.pt \
        --flow_path /repo/outputs_stage2_gauss/flow_best.pt \
        --data_info_path /repo/outputs_stage1_gauss/data_info.pt \
        --device cuda:0
"""
import argparse, numpy as np, torch
from pathlib import Path
from data import load_data, get_dataloader
from models import ProteinVAE, RNAEncoder, FlowNet
from metrics import evaluate_predictions
from torchdiffeq import odeint
from tqdm import tqdm

class ODEFunc(torch.nn.Module):
    def __init__(self, flow_net, c, batch_id, cfg_scale=1.0, use_cfg=True):
        super().__init__()
        self.flow_net = flow_net
        self.c = c
        self.batch_id = batch_id
        self.cfg_scale = cfg_scale
        self.use_cfg = use_cfg
        B = c.shape[0]
        self.cond_null = flow_net.get_cond_null(B, c.device)

    def forward(self, t, x):
        B = x.shape[0]
        t_batch = torch.full((B,), t.item(), device=x.device)
        if self.use_cfg and self.cfg_scale != 1.0:
            v_cond = self.flow_net(x, t_batch, self.c, self.batch_id)
            v_uncond = self.flow_net(x, t_batch, self.cond_null, self.batch_id)
            v = v_uncond + self.cfg_scale * (v_cond - v_uncond)
        else:
            v = self.flow_net(x, t_batch, self.c, self.batch_id)
        return v

@torch.no_grad()
def evaluate(vae, rna_encoder, flow_net, dataloader, device, n_steps=50,
             cfg_scale=3.0, ode_method='dopri5', rtol=1e-5, atol=1e-5, ensemble_k=1):
    vae.eval(); rna_encoder.eval(); flow_net.eval()
    all_preds, all_truth = [], []
    t_span = torch.tensor([0.0, 1.0], device=device)

    for batch in tqdm(dataloader, desc='Eval'):
        rna_norm = batch['rna_norm'].to(device)
        prot_norm = batch['prot_norm'].to(device)
        batch_id = batch['batch_id'].to(device)
        B = rna_norm.shape[0]

        c = rna_encoder(rna_norm, batch_id)

        # Ensemble: average predictions from K independent x0 samples
        y_hat_ensemble = []
        for _ in range(ensemble_k):
            x0 = torch.randn(B, vae.dz, device=device)
            ode_func = ODEFunc(flow_net, c, batch_id, cfg_scale=cfg_scale, use_cfg=True)
            x_traj = odeint(ode_func, x0, t_span, method=ode_method, rtol=rtol, atol=atol)
            x = x_traj[-1]
            y_hat_k = vae.decode(x, batch_id)
            y_hat_ensemble.append(y_hat_k)
        y_hat = torch.stack(y_hat_ensemble, dim=0).mean(dim=0)
        all_preds.append(y_hat.cpu().numpy())
        all_truth.append(prot_norm.cpu().numpy())

    pred = np.concatenate(all_preds, axis=0)
    truth = np.concatenate(all_truth, axis=0)
    return pred, truth

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--vae_path', type=str, required=True)
    parser.add_argument('--flow_path', type=str, required=True)
    parser.add_argument('--data_info_path', type=str, required=True)
    parser.add_argument('--n_top_genes', type=int, default=1000)
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--split_seed', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--n_steps', type=int, default=50)
    parser.add_argument('--cfg_scale', type=float, default=3.0)
    parser.add_argument('--ode_method', type=str, default='dopri5')
    parser.add_argument('--rtol', type=float, default=1e-5)
    parser.add_argument('--atol', type=float, default=1e-5)
    parser.add_argument('--ensemble_k', type=int, default=1)
    parser.add_argument('--output_dir', type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load VAE
    vae_ckpt = torch.load(args.vae_path, map_location=device)
    vc = vae_ckpt['config']
    vae = ProteinVAE(n_proteins=vc['n_proteins'], dz=vc['dz'],
                     hidden_dims=vc['hidden_dims'], batch_emb_dim=vc['batch_emb_dim'],
                     n_batches=vc['n_batches'], beta_kl=vc['beta_kl'],
                     learnable_dispersion=True, dist_type=vc.get('dist_type', 'Gaussian')).to(device)
    vae.load_state_dict(vae_ckpt['model_state_dict'])
    vae.eval()
    print(f"VAE loaded (dist={vc.get('dist_type','Gaussian')})")

    # Load Flow
    flow_ckpt = torch.load(args.flow_path, map_location=device)
    fc = flow_ckpt['config']
    rna_encoder = RNAEncoder(n_genes=fc['n_genes'], dc=fc['dc'],
                             hidden_dims=fc['rna_hidden_dims'],
                             batch_emb_dim=fc['batch_emb_dim'],
                             n_batches=fc['n_batches'], dropout=0.0).to(device)
    rna_encoder.load_state_dict(flow_ckpt['rna_encoder_state_dict'])
    flow_net = FlowNet(dz=fc['dz'], dc=fc['dc'], hidden_dim=fc['flow_hidden_dim'],
                       n_blocks=fc['flow_n_blocks'], time_emb_dim=64,
                       batch_emb_dim=fc['batch_emb_dim'],
                       n_batches=fc['n_batches'], dropout=0.0).to(device)
    flow_net.load_state_dict(flow_ckpt['flow_net_state_dict'])
    rna_encoder.eval(); flow_net.eval()
    print(f"CFM loaded (dc={fc['dc']}, dz={fc['dz']})")

    # Load data
    _, test_ds, data_info = load_data(args.data_path, n_top_genes=args.n_top_genes,
                                       train_ratio=args.train_ratio, random_state=args.split_seed)
    test_loader = get_dataloader(test_ds, batch_size=512, shuffle=False, num_workers=4)

    # Evaluate
    print(f"\nRunning evaluation on {data_info['n_test']} test cells...")
    pred, truth = evaluate(vae, rna_encoder, flow_net, test_loader, device,
                           n_steps=args.n_steps, cfg_scale=args.cfg_scale,
                           ode_method=args.ode_method, rtol=args.rtol, atol=args.atol, ensemble_k=args.ensemble_k)

    results = evaluate_predictions(pred, truth, protein_names=data_info['protein_names'], verbose=True)

    if args.output_dir:
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / 'predictions.npy', pred)
        np.save(out / 'ground_truth.npy', truth)
        print(f"Saved predictions to {out}")

    # Return metrics for manifest
    return results

if __name__ == '__main__':
    main()
