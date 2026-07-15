import os, sys, time, torch, numpy as np
sys.path.insert(0, '/repo')
from sed.data.scrna import SparseCellDataModule, CellDataset
from sed.models.vae.svae import SVAE
from sed.models.diffusion.diffusion import Diffusion
from sed.utils import num_to_groups
import scanpy as sc
import anndata as ad
from scipy import stats
import json

def sparse_to_dense(positions, values, data_dimensions, start_pos, end_pos, pad_pos):
    """Convert sparse (positions, values) to dense array."""
    batch_size = positions.shape[0]
    dense = np.zeros((batch_size, data_dimensions), dtype=np.float32)
    for b in range(batch_size):
        valid_mask = (positions[b] != pad_pos) & (positions[b] != start_pos) & (positions[b] != end_pos)
        pos = positions[b][valid_mask].cpu().numpy().astype(int)
        pos = pos.clip(0, data_dimensions - 1)
        val = values[b][valid_mask].cpu().numpy()
        dense[b, pos] = val
    return dense

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--svae_ckpt', type=str, default='/repo/svae_output/svae_20k.pth')
    parser.add_argument('--sed_ckpt', type=str, default='/repo/sed_output/sed_100k.pth')
    parser.add_argument('--n_samples', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--output', type=str, default='/repo/eval_results.json')
    parser.add_argument('--use_ddim', action='store_true', default=False,
                        help='Use DDIM deterministic sampling instead of DDPM')
    parser.add_argument('--ddim_steps', type=int, default=1000,
                        help='Number of DDIM sampling steps (only with --use_ddim)')
    parser.add_argument('--time_difference', type=float, default=0.,
                        help='Time difference for sampling (0=standard)')
    parser.add_argument('--spacing_power', type=float, default=0.5,
                        help='Power for custom spacing (only with --ddim_spacing power)')
    parser.add_argument('--ddim_spacing', type=str, default='linear',
                        choices=['linear', 'quadratic', 'sqrt', 'power'],
                        help='Step spacing for DDIM: linear (uniform), quadratic (more at low t), sqrt (more at high t), power (t^p with --spacing_power)')
    args = parser.parse_args()
    
    device = torch.device('cuda:0')
    data_dim = 1000
    
    # Load real data (dense, preprocessed)
    print('Loading real data...')
    dm = SparseCellDataModule(
        train_data_dir='/tmp/habermann_human_lung_pf.h5ad',
        batch_size=args.batch_size, data_dimensions=data_dim, input_mode='scrna'
    )
    dm.setup()
    
    start_pos = dm.start_position
    end_pos = dm.end_position
    pad_pos = dm.pad_position
    
    # Get real data from validation set as dense
    real_cells = []
    for batch in dm.val_dataloader():
        positions, values = batch
        dense = sparse_to_dense(positions, values, data_dim, start_pos, end_pos, pad_pos)
        real_cells.append(dense)
    real_data = np.concatenate(real_cells, axis=0)[:args.n_samples]
    print(f'Real data: {real_data.shape}')
    
    # Load models
    print('Loading SVAE...')
    svae = SVAE(data_dimensions=data_dim, num_layers=3, d_model=256, d_ff=1024,
                h=4, dropout=0.1, beta=1e-6, input_mode='scrna', lr=None).to(device)
    svae.load_state_dict(torch.load(args.svae_ckpt, map_location=device))
    svae.eval()
    
    print('Loading Diffusion...')
    diffusion = Diffusion(
        unet_config={'hidden_dim': [512, 512, 256, 128], 'dropout': 0.1},
        image_size=256,
        timesteps=args.ddim_steps if args.use_ddim else 1000,
        use_ddim=args.use_ddim,
        noise_schedule='cosine',
        time_difference=args.time_difference
    ).to(device)
    
    # Custom step spacing for DDIM
    if args.use_ddim and args.ddim_spacing != 'linear':
        import types
        orig_get_steps = diffusion.get_sampling_timesteps
        spacing_type = args.ddim_spacing
        def custom_get_sampling_timesteps(self, batch, *, device):
            if spacing_type == 'quadratic':
                times = torch.linspace(1., 0., self.timesteps + 1, device=device) ** 2
            elif spacing_type == 'sqrt':
                times = torch.linspace(1., 0., self.timesteps + 1, device=device) ** 0.5
            elif spacing_type == 'power':
                times = torch.linspace(1., 0., self.timesteps + 1, device=device) ** args.spacing_power
            else:
                times = torch.linspace(1., 0., self.timesteps + 1, device=device)
            from einops import repeat
            times = repeat(times, 't -> b t', b=batch)
            times = torch.stack((times[:, :-1], times[:, 1:]), dim=0)
            times = times.unbind(dim=-1)
            return times
        diffusion.get_sampling_timesteps = types.MethodType(custom_get_sampling_timesteps, diffusion)
    sed_state = torch.load(args.sed_ckpt, map_location=device)
    diffusion.unet_model.load_state_dict(
        {k.replace('unet_model.', ''): v for k, v in sed_state['diffusion_state_dict'].items() 
         if k.startswith('unet_model.')}
    )
    diffusion.eval()
    
    # Generate samples
    print(f'Generating {args.n_samples} samples...')
    all_cells = []
    batches = num_to_groups(args.n_samples, args.batch_size)
    with torch.no_grad():
        for bs in batches:
            sampled_z = diffusion.sample(bs)
            out_positions, out_values = svae.sample(bs, sampled_z)
            dense = sparse_to_dense(out_positions, out_values, data_dim, start_pos, end_pos, pad_pos)
            all_cells.append(dense)
    
    gen_data = np.concatenate(all_cells, axis=0)[:args.n_samples]
    gen_data = np.clip(gen_data, 0, 1)
    print(f'Generated data: {gen_data.shape}')
    
    # Compute metrics following paper protocol
    print('Computing metrics...')
    gen_sparsity = (np.size(gen_data) - np.count_nonzero(gen_data)) / np.size(gen_data)
    real_sparsity = (np.size(real_data) - np.count_nonzero(real_data)) / np.size(real_data)
    
    # Normalize for SCC/MMD (log1p after normalize_total)
    real_adata = ad.AnnData(real_data)
    gen_adata = ad.AnnData(gen_data)
    sc.pp.normalize_total(real_adata, target_sum=1e4)
    sc.pp.log1p(real_adata)
    sc.pp.normalize_total(gen_adata, target_sum=1e4)
    sc.pp.log1p(gen_adata)
    
    real_arr = real_adata.X
    gen_arr = gen_adata.X
    
    # SCC
    scc_result = stats.spearmanr(real_arr.mean(axis=0), gen_arr.mean(axis=0))
    scc = scc_result.correlation if hasattr(scc_result, 'correlation') else scc_result[0]
    
    # MMD via PCA + RBF kernel
    combined = ad.AnnData(np.concatenate([real_arr, gen_arr], axis=0), dtype=np.float32)
    sc.tl.pca(combined, svd_solver='arpack', n_comps=20)
    
    n_real = real_arr.shape[0]
    real_pca = combined.obsm['X_pca'][:n_real][::2][:5000]
    gen_pca = combined.obsm['X_pca'][n_real:][::2][:5000]
    
    X = torch.Tensor(real_pca)
    Y = torch.Tensor(gen_pca)
    
    # RBF MMD
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
    
    results = {
        'SCC': float(scc),
        'MMD': float(mmd),
        'gen_sparsity': float(gen_sparsity),
        'real_sparsity': float(real_sparsity),
        'n_generated': int(args.n_samples),
        'svae_steps': 20000,
        'sed_steps': 100000,
        'sampling_method': 'ddim' if args.use_ddim else 'ddpm',
        'ddim_steps': args.ddim_steps if args.use_ddim else 1000,
        'time_difference': args.time_difference,
        'ddim_spacing': args.ddim_spacing,
        'spacing_power': args.spacing_power if args.ddim_spacing == 'power' else None,
    }
    
    print(f'\n=== RESULTS ===')
    print(f'SCC: {scc:.4f} (paper: 0.82 for SEDP 4M, 500K steps)')
    print(f'MMD: {mmd:.4f} (paper: 0.54 for SEDP 4M, 500K steps)')
    print(f'Gen sparsity: {gen_sparsity:.4f} (real: {real_sparsity:.4f})')
    if args.use_ddim:
        print(f'Sampling: DDIM with {args.ddim_steps} steps')
    else:
        print(f'Sampling: DDPM with 1000 steps')
    
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Results saved to {args.output}')

if __name__ == '__main__':
    main()
