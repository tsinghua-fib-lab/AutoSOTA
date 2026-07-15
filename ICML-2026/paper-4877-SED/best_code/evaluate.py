import os, sys, time, torch, numpy as np
import argparse
sys.path.insert(0, '/repo')
from sed.data.scrna import SparseCellDataModule
from sed.models.vae.svae import SVAE
from sed.models.diffusion.diffusion import Diffusion
from sed.utils import num_to_groups
import scanpy as sc
import anndata as ad
from scipy import stats
import json

def get_scc_mmd(real_data, gen_data):
    """Compute SCC and MMD following the paper's evaluation protocol."""
    # Normalize like the paper
    real_adata = ad.AnnData(real_data)
    gen_adata = ad.AnnData(gen_data)
    sc.pp.normalize_total(real_adata, target_sum=1e4)
    sc.pp.log1p(real_adata)
    sc.pp.normalize_total(gen_adata, target_sum=1e4)
    sc.pp.log1p(gen_adata)
    
    real_arr = real_adata.X
    gen_arr = gen_adata.X
    
    # SCC
    scc, _ = stats.spearmanr(real_arr.mean(axis=0), gen_arr.mean(axis=0))
    
    # MMD (simplified)
    # Use PCA and RBF MMD
    combined = ad.AnnData(np.concatenate([real_arr, gen_arr], axis=0))
    sc.tl.pca(combined, svd_solver='arpack', n_comps=20)
    real_pca = combined[combined.obs_names.str.startswith('0') if hasattr(combined.obs_names, 'str') else range(len(real_arr))].obsm['X_pca'][::2][:5000]
    gen_pca_real = combined.obsm['X_pca'][len(real_arr):][::2][:5000]
    
    # Simple RBF MMD
    X = torch.Tensor(real_pca)
    Y = torch.Tensor(gen_pca_real)
    
    def gaussian_kernel(source, target, kernel_mul=2.0, kernel_num=5):
        n_samples = source.size(0) + target.size(0)
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(n_samples, n_samples, total.size(1))
        total1 = total.unsqueeze(1).expand(n_samples, n_samples, total.size(1))
        L2 = ((total0 - total1) ** 2).sum(2)
        bandwidth = torch.sum(L2.data) / (n_samples**2 - n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2 / bw) for bw in bandwidth_list]
        return sum(kernel_val)
    
    kernels = gaussian_kernel(X, Y)
    XX = kernels[:X.size(0), :X.size(0)]
    YY = kernels[X.size(0):, X.size(0):]
    XY = kernels[:X.size(0), X.size(0):]
    YX = kernels[X.size(0):, :X.size(0)]
    mmd = torch.mean(XX + YY - XY - YX).item()
    
    return scc, mmd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--svae_ckpt', type=str, default='/repo/svae_output/svae_20k.pth')
    parser.add_argument('--sed_ckpt', type=str, default='/repo/sed_output/sed_100k.pth')
    parser.add_argument('--n_samples', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--output', type=str, default='/repo/eval_results.json')
    args = parser.parse_args()
    
    device = torch.device('cuda:0')
    
    # Load data for evaluation (real data reference)
    dm = SparseCellDataModule(
        train_data_dir='/tmp/habermann_human_lung_pf.h5ad',
        batch_size=args.batch_size, data_dimensions=1000, input_mode='scrna'
    )
    dm.setup()
    
    # Get real data for metrics
    print('Loading real data for evaluation...')
    real_cells = []
    for batch in dm.val_dataloader():
        # For CellDataModule (non-sparse), batch is just the array
        if isinstance(batch, (list, tuple)):
            real_cells.append(batch[0].numpy() if torch.is_tensor(batch[0]) else batch[0])
        else:
            real_cells.append(batch.numpy() if torch.is_tensor(batch) else batch)
    real_data = np.concatenate(real_cells, axis=0)[:10000]
    print(f'Real data: {real_data.shape}')
    
    # Load SVAE
    print('Loading SVAE...')
    svae = SVAE(data_dimensions=1000, num_layers=3, d_model=256, d_ff=1024,
                h=4, dropout=0.1, beta=1e-6, input_mode='scrna', lr=None).to(device)
    svae.load_state_dict(torch.load(args.svae_ckpt, map_location=device))
    svae.eval()
    
    # Load diffusion model
    print('Loading Diffusion...')
    diffusion = Diffusion(
        unet_config={'hidden_dim': [512, 512, 256, 128], 'dropout': 0.1},
        image_size=256, timesteps=1000, use_ddim=False, noise_schedule='cosine'
    ).to(device)
    sed_state = torch.load(args.sed_ckpt, map_location=device)
    diffusion.unet_model.load_state_dict(
        {k.replace('unet_model.', ''): v for k, v in sed_state['diffusion_state_dict'].items() if k.startswith('unet_model.')}
    )
    diffusion.eval()
    
    # Generate samples
    print(f'Generating {args.n_samples} samples...')
    start_pos = dm.start_position
    end_pos = dm.end_position
    pad_pos = dm.pad_position
    
    all_cells = []
    batches = num_to_groups(args.n_samples, args.batch_size)
    with torch.no_grad():
        for batch_size in batches:
            # Sample latent from diffusion
            sampled_z = diffusion.sample(batch_size)
            # Decode to sparse representation
            out_positions, out_values = svae.sample(batch_size, sampled_z)
            # Convert to dense cells
            cells = torch.zeros(batch_size, 1000, device=device)
            for b in range(batch_size):
                valid = (out_positions[b] != pad_pos) & (out_positions[b] != start_pos) & (out_positions[b] != end_pos)
                pos = out_positions[b][valid].long()
                val = out_values[b][valid]
                pos = pos.clamp(0, 999)
                cells[b, pos] = val
            all_cells.append(cells.cpu().numpy())
    
    gen_data = np.concatenate(all_cells, axis=0)[:args.n_samples]
    gen_data = np.clip(gen_data, 0, 1)
    print(f'Generated data: {gen_data.shape}')
    
    # Compute metrics
    print('Computing metrics...')
    scc, mmd = get_scc_mmd(real_data, gen_data)
    
    gen_sparsity = (np.size(gen_data) - np.count_nonzero(gen_data)) / np.size(gen_data)
    real_sparsity = (np.size(real_data) - np.count_nonzero(real_data)) / np.size(real_data)
    
    results = {
        'SCC': float(scc),
        'MMD': float(mmd),
        'gen_sparsity': float(gen_sparsity),
        'real_sparsity': float(real_sparsity),
        'n_generated': int(args.n_samples),
        'svae_ckpt': args.svae_ckpt,
        'sed_ckpt': args.sed_ckpt,
    }
    
    print(f'SCC: {scc:.4f}')
    print(f'MMD: {mmd:.4f}')
    print(f'Gen sparsity: {gen_sparsity:.4f}')
    print(f'Real sparsity: {real_sparsity:.4f}')
    
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Results saved to {args.output}')

if __name__ == '__main__':
    main()
