"""
Visualization functions for SENDAI.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def plot_reconstruction(results, sensor_locs, H, W, output_dir, scaler_state,
                        sim_states=None, Full_Sendai = True, sim_label="Simulation", gt_label="Ground Truth",
                        suffix=''):
    """Plot reconstruction comparison."""
    output_dir = Path(output_dir)
    
    n_samples = len(results['targets'])
    t_indices = [0, n_samples // 4, n_samples // 2, 3 * n_samples // 4]
    
    has_hf = 'hf' in results and not np.allclose(results['da'], results['total'])
    has_sim = sim_states is not None
    
    n_cols = 2 + int(has_sim) + int(Full_Sendai) + (int(has_hf) if Full_Sendai else 0)
    fig, axes = plt.subplots(4, n_cols, figsize=(4 * n_cols, 16))
    
    all_data = np.concatenate([results['targets'], results['total']])
    vmin, vmax = np.percentile(all_data, [2, 98])
    
    if has_hf:
        hf_abs_max = np.percentile(np.abs(results['hf']), 98)
    
    for row, t in enumerate(t_indices):
        col = 0
        
        if has_sim:
            sim_orig = scaler_state.inverse_transform(sim_states)
            ax = axes[row, col]
            t_sim = min(t, len(sim_orig) - 1)
            ax.imshow(sim_orig[t_sim].reshape(H, W), cmap='RdYlGn', vmin=vmin, vmax=vmax)
            if row == 0:
                ax.set_title(sim_label, fontsize=14)
            ax.set_ylabel(f't={t}', fontsize=12)
            ax.axis('off')
            col += 1
        
        ax = axes[row, col]
        ax.imshow(results['da'][t].reshape(H, W), cmap='RdYlGn', vmin=vmin, vmax=vmax)
        if row == 0:
            ax.set_title('SENDAI Jr.', fontsize=14)
        if not has_sim:
            ax.set_ylabel(f't={t}', fontsize=12)
        ax.axis('off')
        col += 1

        if Full_Sendai:
            ax = axes[row, col]
            ax.imshow(results['total'][t].reshape(H, W), cmap='RdYlGn', vmin=vmin, vmax=vmax)
            if row == 0:
                ax.set_title('SENDAI', fontsize=14)
            ax.axis('off')
            col += 1

        ax = axes[row, col]
        im = ax.imshow(results['targets'][t].reshape(H, W), cmap='RdYlGn', vmin=vmin, vmax=vmax)
        ax.scatter(sensor_locs[:, 1], sensor_locs[:, 0], c='red', s=8, marker='^', alpha=0.6)
        if row == 0:
            ax.set_title(gt_label, fontsize=14)
        ax.axis('off')
        col += 1

        if Full_Sendai:
            if has_hf:
                ax = axes[row, col]
                ax.imshow(results['hf'][t].reshape(H, W), cmap='RdBu', vmin=-hf_abs_max, vmax=hf_abs_max)
                if row == 0:
                    ax.set_title('HF Component', fontsize=14)
                ax.axis('off')
    
    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.93, 0.25, 0.02, 0.5])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('NDVI', fontsize=12)
    
    plt.savefig(output_dir / f'reconstruction{suffix}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / f'reconstruction{suffix}.png'}")


def plot_temporal(results, output_dir, suffix=''):
    """Plot temporal analysis."""
    output_dir = Path(output_dir)
    
    has_hf = 'hf' in results and not np.allclose(results['da'], results['total'])
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    ax = axes[0]
    ax.plot(results['targets'].mean(axis=1), 'g-', label='Ground Truth', linewidth=2)
    ax.plot(results['da'].mean(axis=1), 'b--', label='LF only')
    if has_hf:
        ax.plot(results['total'].mean(axis=1), 'r-', label='LF+HF', linewidth=1.5)
    ax.set_xlabel('Time step')
    ax.set_ylabel('Mean NDVI')
    ax.set_title('Spatial Mean Over Time')
    ax.legend()
    ax.grid(alpha=0.3)
    
    ax = axes[1]
    rmse_da = np.sqrt(np.mean((results['da'] - results['targets'])**2, axis=1))
    ax.plot(rmse_da, 'b--', label=f'LF (mean: {rmse_da.mean():.4f})')
    if has_hf:
        rmse_total = np.sqrt(np.mean((results['total'] - results['targets'])**2, axis=1))
        ax.plot(rmse_total, 'r-', label=f'LF+HF (mean: {rmse_total.mean():.4f})')
    ax.set_xlabel('Time step')
    ax.set_ylabel('RMSE')
    ax.set_title('Reconstruction Error')
    ax.legend()
    ax.grid(alpha=0.3)
    
    ax = axes[2]
    ax.axis('off')
    improvement = (rmse_da.mean() - rmse_total.mean()) / rmse_da.mean() * 100 if has_hf else 0
    summary = f"""
    Summary
    =======
    
    LF RMSE:     {rmse_da.mean():.4f}
    LF+HF RMSE:  {f"{rmse_total.mean():.4f}" if has_hf else "N/A"}
    
    Improvement: {improvement:.1f}%
    """
    ax.text(0.1, 0.9, summary, transform=ax.transAxes, fontfamily='monospace',
            fontsize=11, verticalalignment='top')
    
    plt.tight_layout()
    plt.savefig(output_dir / f'temporal{suffix}.png', dpi=150)
    plt.close()
    print(f"Saved: {output_dir / f'temporal{suffix}.png'}")


def plot_hf_analysis(results, H, W, output_dir, suffix=''):
    """Plot HF component analysis."""
    output_dir = Path(output_dir)
    
    if 'hf' not in results or np.allclose(results['da'], results['total']):
        print("No HF component to analyze")
        return
    
    hf = results['hf']
    
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    
    indices = [0, len(hf) // 2, -1]
    for i, idx in enumerate(indices):
        ax = axes[0, i]
        hf_2d = hf[idx].reshape(H, W)
        vmax = np.abs(hf_2d).max()
        im = ax.imshow(hf_2d, cmap='RdBu', vmin=-vmax, vmax=vmax)
        ax.set_title(f'HF t={idx}')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)
    
    ax = axes[1, 0]
    avg_hf = hf.mean(axis=0).reshape(H, W)
    fft_mag = np.abs(np.fft.fftshift(np.fft.fft2(avg_hf)))
    ax.imshow(np.log1p(fft_mag), cmap='viridis')
    ax.set_title('Avg HF Spectrum (log)')
    ax.axis('off')
    
    ax = axes[1, 1]
    ax.hist(hf.ravel(), bins=50, density=True, alpha=0.7)
    ax.set_xlabel('HF value')
    ax.set_ylabel('Density')
    ax.set_title(f'HF Distribution (std={hf.std():.4f})')
    
    ax = axes[1, 2]
    hf_mag = np.sqrt(np.mean(hf**2, axis=1))
    ax.plot(hf_mag)
    ax.set_xlabel('Time step')
    ax.set_ylabel('HF RMS')
    ax.set_title('HF Magnitude Over Time')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'hf_analysis{suffix}.png', dpi=150)
    plt.close()
    print(f"Saved: {output_dir / f'hf_analysis{suffix}.png'}")
