import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.cm import ScalarMappable
from scipy.interpolate import NearestNDInterpolator
from mpl_toolkits.mplot3d import Axes3D


def visualize_flux_field_3d(pts, mapper, GT_Flux, predictions_dict, 
                            output_path, idx=0, sphere_center=(0,0,0), 
                            sphere_radius=0.4, logger=None):
    if logger:
        logger.log(f"\n[Viz] Rendering 3D Flux Field to {output_path}...")
    
    def flux_to_mag(flux):
        vec = mapper.edge_flux_to_node_vector(flux)
        return np.linalg.norm(vec, axis=1)
    
    gt_mag = flux_to_mag(GT_Flux[idx])
    pred_mags = {name: flux_to_mag(pred[idx]) for name, pred in predictions_dict.items()}
    
    all_mags = [gt_mag] + list(pred_mags.values())
    global_vmax = max(m.max() for m in all_mags)
    global_vmin = 0
    
    sphere_center = np.array(sphere_center)
    r_from_center = np.linalg.norm(pts - sphere_center, axis=1)
    valid_mask = r_from_center > sphere_radius * 1.1
    
    n_plots = 1 + len(predictions_dict)
    fig = plt.figure(figsize=(6 * n_plots, 6))
    
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 20)
    sx = sphere_center[0] + sphere_radius * np.outer(np.cos(u), np.sin(v))
    sy = sphere_center[1] + sphere_radius * np.outer(np.sin(u), np.sin(v))
    sz = sphere_center[2] + sphere_radius * np.outer(np.ones(np.size(u)), np.cos(v))
    
    ax = fig.add_subplot(1, n_plots, 1, projection='3d')
    sc = ax.scatter(pts[valid_mask, 0], pts[valid_mask, 1], pts[valid_mask, 2],
                    c=gt_mag[valid_mask], cmap='coolwarm', s=8, alpha=0.8,
                    vmin=global_vmin, vmax=global_vmax)
    ax.plot_surface(sx, sy, sz, color='gray', alpha=0.5)
    ax.set_title('Ground Truth |v|', fontsize=14, fontweight='bold')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.colorbar(sc, ax=ax, shrink=0.6, label='|velocity|')
    
    for i, (name, pred_mag) in enumerate(pred_mags.items()):
        ax = fig.add_subplot(1, n_plots, i + 2, projection='3d')
        sc = ax.scatter(pts[valid_mask, 0], pts[valid_mask, 1], pts[valid_mask, 2],
                        c=pred_mag[valid_mask], cmap='coolwarm', s=8, alpha=0.8,
                        vmin=global_vmin, vmax=global_vmax)
        ax.plot_surface(sx, sy, sz, color='gray', alpha=0.5)
        ax.set_title(f'{name} |v|', fontsize=14, fontweight='bold')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.colorbar(sc, ax=ax, shrink=0.6, label='|velocity|')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    if logger:
        logger.log(f"[Viz] Saved to {output_path}")


def visualize_input_output_3d(pts, X_input, mapper, GT_Flux, predictions_dict,
                              output_path, idx=0, sphere_center=(0,0,0),
                              sphere_radius=0.4, domain_radius=2.0, logger=None):
    if logger:
        logger.log(f"\n[Viz] Rendering 3D Input/Output to {output_path}...")
    
    sphere_center = np.array(sphere_center)
    
    def flux_to_vec(flux):
        return mapper.edge_flux_to_node_vector(flux)
    
    X = X_input[idx]
    gt_vec = flux_to_vec(GT_Flux[idx])
    pred_vecs = {name: flux_to_vec(pred[idx]) for name, pred in predictions_dict.items()}
    
    r_from_center = np.linalg.norm(pts - sphere_center, axis=1)
    outer_limit = domain_radius * 0.85
    inner_limit = sphere_radius * 1.2
    valid_mask = (r_from_center < outer_limit) & (r_from_center > inner_limit)
    valid_indices = np.where(valid_mask)[0]
    
    r_valid = r_from_center[valid_mask]
    dist_sort_idx = valid_indices[np.argsort(r_valid)]
    points_plot = pts[dist_sort_idx]
    
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 20)
    sx = sphere_center[0] + sphere_radius * np.outer(np.cos(u), np.sin(v))
    sy = sphere_center[1] + sphere_radius * np.outer(np.sin(u), np.sin(v))
    sz = sphere_center[2] + sphere_radius * np.outer(np.ones(np.size(u)), np.cos(v))
    
    n_cols = 2 + len(predictions_dict)
    fig = plt.figure(figsize=(6 * n_cols, 6))
    
    X_plot = X[dist_sort_idx]
    ax1 = fig.add_subplot(1, n_cols, 1, projection='3d')
    sc1 = ax1.scatter(points_plot[:, 0], points_plot[:, 1], points_plot[:, 2],
                      c=X_plot, cmap='RdBu_r', s=15, alpha=0.9,
                      vmin=-1, vmax=1, edgecolors='none')
    ax1.plot_surface(sx, sy, sz, color='gray', alpha=0.3)
    ax1.set_title(f'Sample {idx}: Input X (ρ)\nRange: [{X.min():.2f}, {X.max():.2f}]',
                  fontweight='bold')
    plt.colorbar(sc1, ax=ax1, shrink=0.6, label='Charge density')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    
    def compute_radial_component(vec):
        r_vec = pts - sphere_center
        r_norm = np.linalg.norm(r_vec, axis=1, keepdims=True)
        r_hat = r_vec / (r_norm + 1e-6)
        return np.sum(vec * r_hat, axis=1)
    
    gt_radial = compute_radial_component(gt_vec)
    gt_radial_max = np.abs(gt_radial).max()
    if gt_radial_max > 1e-10:
        gt_radial_normalized = gt_radial / gt_radial_max
    else:
        gt_radial_normalized = gt_radial
    
    gt_radial_plot = gt_radial_normalized[dist_sort_idx]
    ax2 = fig.add_subplot(1, n_cols, 2, projection='3d')
    sc2 = ax2.scatter(points_plot[:, 0], points_plot[:, 1], points_plot[:, 2],
                      c=gt_radial_plot, cmap='RdBu_r', s=15, alpha=0.9,
                      vmin=-1, vmax=1, edgecolors='none')
    
    stride = max(1, len(pts) // 200)
    quiver_idx = np.arange(0, len(pts), stride)
    ax2.quiver(pts[quiver_idx, 0], pts[quiver_idx, 1], pts[quiver_idx, 2],
               gt_vec[quiver_idx, 0], gt_vec[quiver_idx, 1], gt_vec[quiver_idx, 2],
               length=0.12, normalize=True, alpha=0.6, color='black')
    ax2.plot_surface(sx, sy, sz, color='purple', alpha=0.5)
    ax2.set_title(f'GT Output Y (v field)\nRadial: [{gt_radial.min():.2f}, {gt_radial.max():.2f}]',
                  fontweight='bold')
    plt.colorbar(sc2, ax=ax2, shrink=0.6, label='Radial component (norm)')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    
    for i, (name, pred_vec) in enumerate(pred_vecs.items()):
        pred_radial = compute_radial_component(pred_vec)
        pred_radial_max = np.abs(pred_radial).max()
        if pred_radial_max > 1e-10:
            pred_radial_normalized = pred_radial / pred_radial_max
        else:
            pred_radial_normalized = pred_radial
        
        pred_radial_plot = pred_radial_normalized[dist_sort_idx]
        ax = fig.add_subplot(1, n_cols, 3 + i, projection='3d')
        sc = ax.scatter(points_plot[:, 0], points_plot[:, 1], points_plot[:, 2],
                        c=pred_radial_plot, cmap='RdBu_r', s=15, alpha=0.9,
                        vmin=-1, vmax=1, edgecolors='none')
        ax.quiver(pts[quiver_idx, 0], pts[quiver_idx, 1], pts[quiver_idx, 2],
                  pred_vec[quiver_idx, 0], pred_vec[quiver_idx, 1], pred_vec[quiver_idx, 2],
                  length=0.12, normalize=True, alpha=0.6, color='black')
        ax.plot_surface(sx, sy, sz, color='purple', alpha=0.5)
        ax.set_title(f'{name} Output Y\nRadial: [{pred_radial.min():.2f}, {pred_radial.max():.2f}]',
                     fontweight='bold')
        plt.colorbar(sc, ax=ax, shrink=0.6, label='Radial component (norm)')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    if logger:
        logger.log(f"[Viz] Saved to {output_path}")


def visualize_sample_3d_full(pts, X_input, mapper, GT_Flux, predictions_dict,
                             output_path, idx=0, sphere_center=(0,0,0),
                             sphere_radius=0.4, domain_radius=2.0, logger=None):
    if logger:
        logger.log(f"\n[Viz] Rendering Full 3D Sample to {output_path}...")
    
    sphere_center = np.array(sphere_center)
    
    def flux_to_vec(flux):
        return mapper.edge_flux_to_node_vector(flux)
    
    X = X_input[idx]
    gt_vec = flux_to_vec(GT_Flux[idx])
    pred_vecs = {name: flux_to_vec(pred[idx]) for name, pred in predictions_dict.items()}
    
    r_from_center = np.linalg.norm(pts - sphere_center, axis=1)
    outer_limit = domain_radius * 0.85
    inner_limit = sphere_radius * 1.2
    valid_mask = (r_from_center < outer_limit) & (r_from_center > inner_limit)
    valid_indices = np.where(valid_mask)[0]
    
    r_valid = r_from_center[valid_mask]
    dist_sort_idx = valid_indices[np.argsort(r_valid)]
    points_plot = pts[dist_sort_idx]
    
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 20)
    sx = sphere_center[0] + sphere_radius * np.outer(np.cos(u), np.sin(v))
    sy = sphere_center[1] + sphere_radius * np.outer(np.sin(u), np.sin(v))
    sz = sphere_center[2] + sphere_radius * np.outer(np.ones(np.size(u)), np.cos(v))
    
    n_models = len(predictions_dict)
    n_rows = 2
    n_cols = max(2, 1 + n_models)
    fig = plt.figure(figsize=(6 * n_cols, 6 * n_rows))
    
    X_plot = X[dist_sort_idx]
    ax1 = fig.add_subplot(n_rows, n_cols, 1, projection='3d')
    sc1 = ax1.scatter(points_plot[:, 0], points_plot[:, 1], points_plot[:, 2],
                      c=X_plot, cmap='RdBu_r', s=12, alpha=0.9,
                      vmin=-1, vmax=1, edgecolors='none')
    ax1.plot_surface(sx, sy, sz, color='gray', alpha=0.3)
    ax1.set_title(f'Input X (ρ)\n[{X.min():.2f}, {X.max():.2f}]', fontweight='bold')
    plt.colorbar(sc1, ax=ax1, shrink=0.6, label='ρ')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    
    def compute_radial_component(vec):
        r_vec = pts - sphere_center
        r_norm = np.linalg.norm(r_vec, axis=1, keepdims=True)
        r_hat = r_vec / (r_norm + 1e-6)
        return np.sum(vec * r_hat, axis=1)
    
    gt_radial = compute_radial_component(gt_vec)
    gt_radial_max = np.abs(gt_radial).max()
    if gt_radial_max > 1e-10:
        gt_radial_normalized = gt_radial / gt_radial_max
    else:
        gt_radial_normalized = gt_radial
    
    gt_radial_plot = gt_radial_normalized[dist_sort_idx]
    ax2 = fig.add_subplot(n_rows, n_cols, 2, projection='3d')
    sc2 = ax2.scatter(points_plot[:, 0], points_plot[:, 1], points_plot[:, 2],
                      c=gt_radial_plot, cmap='RdBu_r', s=12, alpha=0.9,
                      vmin=-1, vmax=1, edgecolors='none')
    
    stride = max(1, len(pts) // 150)
    quiver_idx = np.arange(0, len(pts), stride)
    ax2.quiver(pts[quiver_idx, 0], pts[quiver_idx, 1], pts[quiver_idx, 2],
               gt_vec[quiver_idx, 0], gt_vec[quiver_idx, 1], gt_vec[quiver_idx, 2],
               length=0.1, normalize=True, alpha=0.5, color='black')
    ax2.plot_surface(sx, sy, sz, color='purple', alpha=0.5)
    ax2.set_title('GT Output Y (radial)', fontweight='bold')
    plt.colorbar(sc2, ax=ax2, shrink=0.6, label='v_r (norm)')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    
    gt_mag = np.linalg.norm(gt_vec, axis=1)
    gt_mag_max = gt_mag.max()
    if gt_mag_max > 1e-10:
        gt_mag_normalized = gt_mag / gt_mag_max
    else:
        gt_mag_normalized = gt_mag
    
    gt_mag_plot = gt_mag_normalized[dist_sort_idx]
    ax3 = fig.add_subplot(n_rows, n_cols, n_cols + 1, projection='3d')
    sc3 = ax3.scatter(points_plot[:, 0], points_plot[:, 1], points_plot[:, 2],
                      c=gt_mag_plot, cmap='viridis', s=12, alpha=0.9,
                      vmin=0, vmax=1, edgecolors='none')
    ax3.plot_surface(sx, sy, sz, color='purple', alpha=0.5)
    ax3.set_title('GT |v| (magnitude)', fontweight='bold')
    plt.colorbar(sc3, ax=ax3, shrink=0.6, label='|v| (norm)')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    
    for i, (name, pred_vec) in enumerate(pred_vecs.items()):
        if i + 3 <= n_cols:
            pred_radial = compute_radial_component(pred_vec)
            pred_radial_max = np.abs(pred_radial).max()
            if pred_radial_max > 1e-10:
                pred_radial_normalized = pred_radial / pred_radial_max
            else:
                pred_radial_normalized = pred_radial
            
            pred_radial_plot = pred_radial_normalized[dist_sort_idx]
            ax = fig.add_subplot(n_rows, n_cols, 3 + i, projection='3d')
            sc = ax.scatter(points_plot[:, 0], points_plot[:, 1], points_plot[:, 2],
                            c=pred_radial_plot, cmap='RdBu_r', s=12, alpha=0.9,
                            vmin=-1, vmax=1, edgecolors='none')
            ax.quiver(pts[quiver_idx, 0], pts[quiver_idx, 1], pts[quiver_idx, 2],
                      pred_vec[quiver_idx, 0], pred_vec[quiver_idx, 1], pred_vec[quiver_idx, 2],
                      length=0.1, normalize=True, alpha=0.5, color='black')
            ax.plot_surface(sx, sy, sz, color='purple', alpha=0.5)
            ax.set_title(f'{name} (radial)', fontweight='bold')
            plt.colorbar(sc, ax=ax, shrink=0.6, label='v_r (norm)')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
        
        pred_mag = np.linalg.norm(pred_vec, axis=1)
        pred_mag_max = pred_mag.max()
        if pred_mag_max > 1e-10:
            pred_mag_normalized = pred_mag / pred_mag_max
        else:
            pred_mag_normalized = pred_mag
        
        pred_mag_plot = pred_mag_normalized[dist_sort_idx]
        ax = fig.add_subplot(n_rows, n_cols, n_cols + 2 + i, projection='3d')
        sc = ax.scatter(points_plot[:, 0], points_plot[:, 1], points_plot[:, 2],
                        c=pred_mag_plot, cmap='viridis', s=12, alpha=0.9,
                        vmin=0, vmax=1, edgecolors='none')
        ax.plot_surface(sx, sy, sz, color='purple', alpha=0.5)
        ax.set_title(f'{name} |v|', fontweight='bold')
        plt.colorbar(sc, ax=ax, shrink=0.6, label='|v| (norm)')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
    
    plt.suptitle(f'Sample #{idx}: 3D Visualization', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    if logger:
        logger.log(f"[Viz] Saved to {output_path}")


def visualize_velocity_slices(pts, mapper, GT_Flux, predictions_dict, 
                              output_path, idx=0, sphere_center=(0,0,0),
                              sphere_radius=0.4, domain_radius=2.0, 
                              grid_size=100, logger=None):
    if logger:
        logger.log(f"\n[Viz] Rendering Velocity Slices to {output_path}...")
    
    sphere_center = np.array(sphere_center)
    
    def flux_to_mag(flux):
        vec = mapper.edge_flux_to_node_vector(flux)
        return np.linalg.norm(vec, axis=1)
    
    gt_mag = flux_to_mag(GT_Flux[idx])
    pred_mags = {name: flux_to_mag(pred[idx]) for name, pred in predictions_dict.items()}
    
    all_mags = [gt_mag] + list(pred_mags.values())
    global_vmax = max(m.max() for m in all_mags)
    
    x_range = np.linspace(-domain_radius * 0.95, domain_radius * 0.95, grid_size)
    y_range = np.linspace(-domain_radius * 0.95, domain_radius * 0.95, grid_size)
    X_grid, Y_grid = np.meshgrid(x_range, y_range)
    
    n_models = len(predictions_dict)
    fig, axes = plt.subplots(2, 1 + n_models, figsize=(5 * (1 + n_models), 10))
    
    def plot_slice(ax, data, title, slice_type='z'):
        interp = NearestNDInterpolator(pts, data)
        
        if slice_type == 'z':
            Z_slice = np.zeros_like(X_grid)
            query_points = np.column_stack([X_grid.ravel(), Y_grid.ravel(), Z_slice.ravel()])
            r_grid = np.sqrt((X_grid - sphere_center[0])**2 + (Y_grid - sphere_center[1])**2)
            xlabel, ylabel = 'X', 'Y'
        else:
            Z_slice = np.zeros_like(X_grid)
            query_points = np.column_stack([X_grid.ravel(), Z_slice.ravel(), Y_grid.ravel()])
            r_grid = np.sqrt((X_grid - sphere_center[0])**2 + (Y_grid - sphere_center[2])**2)
            xlabel, ylabel = 'X', 'Z'
        
        slice_data = interp(query_points).reshape(grid_size, grid_size)
        mask = r_grid < sphere_radius * 1.1
        slice_data[mask] = np.nan
        
        im = ax.pcolormesh(X_grid, Y_grid, slice_data, cmap='viridis', 
                           vmin=0, vmax=global_vmax, shading='auto')
        circle = plt.Circle((sphere_center[0], sphere_center[1] if slice_type == 'z' else sphere_center[2]), 
                             sphere_radius, fill=True, color='gray', alpha=0.8)
        ax.add_patch(circle)
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        return im
    
    im = plot_slice(axes[0, 0], gt_mag, 'GT |v| (z=0)', 'z')
    plt.colorbar(im, ax=axes[0, 0], shrink=0.8)
    
    im = plot_slice(axes[1, 0], gt_mag, 'GT |v| (y=0)', 'y')
    plt.colorbar(im, ax=axes[1, 0], shrink=0.8)
    
    for i, (name, pred_mag) in enumerate(pred_mags.items()):
        im = plot_slice(axes[0, i+1], pred_mag, f'{name} |v| (z=0)', 'z')
        plt.colorbar(im, ax=axes[0, i+1], shrink=0.8)
        
        im = plot_slice(axes[1, i+1], pred_mag, f'{name} |v| (y=0)', 'y')
        plt.colorbar(im, ax=axes[1, i+1], shrink=0.8)
    
    plt.suptitle(f'Sample #{idx}: Velocity Magnitude Slices', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    if logger:
        logger.log(f"[Viz] Saved to {output_path}")


# def visualize_velocity_with_error_slices(pts, mapper, GT_Flux, predictions_dict, 
#                                          output_path, idx=0, sphere_center=(0,0,0),
#                                          sphere_radius=0.4, domain_radius=2.0, 
#                                          grid_size=100, logger=None):
#     if logger:
#         logger.log(f"\n[Viz] Rendering Velocity + Error Slices to {output_path}...")
    
#     sphere_center = np.array(sphere_center)
    
#     def flux_to_mag(flux):
#         vec = mapper.edge_flux_to_node_vector(flux)
#         return np.linalg.norm(vec, axis=1)
    
#     gt_mag = flux_to_mag(GT_Flux[idx])
#     pred_mags = {name: flux_to_mag(pred[idx]) for name, pred in predictions_dict.items()}
#     errors = {name: np.abs(pred_mags[name] - gt_mag) for name in predictions_dict.keys()}
    
#     all_mags = [gt_mag] + list(pred_mags.values())
#     global_vmax = max(m.max() for m in all_mags)
#     global_err_max = max(e.max() for e in errors.values())
    
#     x_range = np.linspace(-domain_radius * 0.95, domain_radius * 0.95, grid_size)
#     y_range = np.linspace(-domain_radius * 0.95, domain_radius * 0.95, grid_size)
#     X_grid, Y_grid = np.meshgrid(x_range, y_range)
    
#     n_models = len(predictions_dict)
#     fig, axes = plt.subplots(2, 1 + n_models, figsize=(5 * (1 + n_models), 10))
    
#     def plot_slice(ax, data, title, cmap, vmin, vmax):
#         interp = NearestNDInterpolator(pts, data)
#         Z_slice = np.zeros_like(X_grid)
#         query_points = np.column_stack([X_grid.ravel(), Y_grid.ravel(), Z_slice.ravel()])
#         slice_data = interp(query_points).reshape(grid_size, grid_size)
        
#         r_grid = np.sqrt((X_grid - sphere_center[0])**2 + (Y_grid - sphere_center[1])**2)
#         mask = r_grid < sphere_radius * 1.1
#         slice_data[mask] = np.nan
        
#         im = ax.pcolormesh(X_grid, Y_grid, slice_data, cmap=cmap, 
#                            vmin=vmin, vmax=vmax, shading='auto')
#         circle = plt.Circle((sphere_center[0], sphere_center[1]), 
#                              sphere_radius, fill=True, color='gray', alpha=0.8)
#         ax.add_patch(circle)
#         ax.set_aspect('equal')
#         ax.set_title(title, fontsize=11, fontweight='bold')
#         ax.set_xlabel('X')
#         ax.set_ylabel('Y')
#         return im
    
#     im = plot_slice(axes[0, 0], gt_mag, 'GT |v|', 'viridis', 0, global_vmax)
#     plt.colorbar(im, ax=axes[0, 0], shrink=0.8)
#     axes[1, 0].axis('off')
#     axes[1, 0].text(0.5, 0.5, '', ha='center', va='center', fontsize=12)
    
#     for i, name in enumerate(predictions_dict.keys()):
#         im = plot_slice(axes[0, i+1], pred_mags[name], f'{name} |v|', 'viridis', 0, global_vmax)
#         plt.colorbar(im, ax=axes[0, i+1], shrink=0.8)
        
#         im = plot_slice(axes[1, i+1], errors[name], f'{name} Error', 'hot', 0, global_err_max)
#         plt.colorbar(im, ax=axes[1, i+1], shrink=0.8)
    
#     plt.suptitle(f'Sample #{idx}: Velocity Magnitude & Error (z=0 slice)', fontsize=16, fontweight='bold')
#     plt.tight_layout()
#     plt.savefig(output_path, dpi=200, bbox_inches='tight')
#     plt.close()
    
#     if logger:
#         logger.log(f"[Viz] Saved to {output_path}")


def visualize_velocity_with_error_slices(pts, mapper, GT_Flux, predictions_dict, 
                                         output_path, idx=0, sphere_center=(0,0,0),
                                         sphere_radius=0.4, domain_radius=2.0, 
                                         grid_size=100, error_gamma=0.4, logger=None):
    """
    Visualize velocity magnitude and prediction errors with enhanced low-error visibility.
    
    Parameters:
    -----------
    error_gamma : float
        Gamma value for PowerNorm on error colormap. 
        Lower values (< 1) emphasize differences in low-error regions.
        Default 0.4 makes 0-0.5 range occupy most of the colorbar.
    """
    if logger:
        logger.log(f"\n[Viz] Rendering Velocity + Error Slices to {output_path}...")
    
    sphere_center = np.array(sphere_center)
    
    def flux_to_mag(flux):
        vec = mapper.edge_flux_to_node_vector(flux)
        return np.linalg.norm(vec, axis=1)
    
    gt_mag = flux_to_mag(GT_Flux[idx])
    pred_mags = {name: flux_to_mag(pred[idx]) for name, pred in predictions_dict.items()}
    errors = {name: np.abs(pred_mags[name] - gt_mag) for name in predictions_dict.keys()}
    
    all_mags = [gt_mag] + list(pred_mags.values())
    global_vmax = max(m.max() for m in all_mags)
    global_err_max = max(e.max() for e in errors.values())
    
    # 避免除以零
    if global_err_max < 1e-10:
        global_err_max = 1.0
    
    x_range = np.linspace(-domain_radius * 0.95, domain_radius * 0.95, grid_size)
    y_range = np.linspace(-domain_radius * 0.95, domain_radius * 0.95, grid_size)
    X_grid, Y_grid = np.meshgrid(x_range, y_range)
    
    n_models = len(predictions_dict)
    fig, axes = plt.subplots(2, 1 + n_models, figsize=(5 * (1 + n_models), 10))
    
    def plot_slice(ax, data, title, cmap, vmin, vmax, use_power_norm=False, gamma=1.0):
        interp = NearestNDInterpolator(pts, data)
        Z_slice = np.zeros_like(X_grid)
        query_points = np.column_stack([X_grid.ravel(), Y_grid.ravel(), Z_slice.ravel()])
        slice_data = interp(query_points).reshape(grid_size, grid_size)
        
        r_grid = np.sqrt((X_grid - sphere_center[0])**2 + (Y_grid - sphere_center[1])**2)
        mask = r_grid < sphere_radius * 1.1
        slice_data[mask] = np.nan
        
        # 使用 PowerNorm 实现非线性颜色映射
        if use_power_norm and gamma != 1.0:
            from matplotlib.colors import PowerNorm
            norm = PowerNorm(gamma=gamma, vmin=vmin, vmax=vmax)
            im = ax.pcolormesh(X_grid, Y_grid, slice_data, cmap=cmap, 
                               norm=norm, shading='auto')
        else:
            im = ax.pcolormesh(X_grid, Y_grid, slice_data, cmap=cmap, 
                               vmin=vmin, vmax=vmax, shading='auto')
        
        circle = plt.Circle((sphere_center[0], sphere_center[1]), 
                             sphere_radius, fill=True, color='gray', alpha=0.8)
        ax.add_patch(circle)
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        return im
    
    # GT magnitude plot (linear colormap)
    im = plot_slice(axes[0, 0], gt_mag, 'GT |v|', 'viridis', 0, global_vmax)
    plt.colorbar(im, ax=axes[0, 0], shrink=0.8)
    axes[1, 0].axis('off')
    axes[1, 0].text(0.5, 0.5, '', ha='center', va='center', fontsize=12)
    
    for i, name in enumerate(predictions_dict.keys()):
        # Prediction magnitude plot (linear colormap)
        im = plot_slice(axes[0, i+1], pred_mags[name], f'{name} |v|', 'viridis', 0, global_vmax)
        plt.colorbar(im, ax=axes[0, i+1], shrink=0.8)
        
        # Error plot with PowerNorm for enhanced low-error visibility
        im = plot_slice(axes[1, i+1], errors[name], f'{name} Error', 'hot', 
                        0, global_err_max, use_power_norm=True, gamma=error_gamma)
        cbar = plt.colorbar(im, ax=axes[1, i+1], shrink=0.8)
        cbar.set_label('Error (nonlinear scale)')
    
    plt.suptitle(f'Sample #{idx}: Velocity Magnitude & Error (z=0 slice)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    if logger:
        logger.log(f"[Viz] Saved to {output_path}")


def visualize_vector_field_slices(pts, mapper, GT_Flux, predictions_dict, 
                                  output_path, idx=0, sphere_center=(0,0,0),
                                  sphere_radius=0.4, domain_radius=2.0, 
                                  grid_size=50, logger=None):
    if logger:
        logger.log(f"\n[Viz] Rendering Vector Field Slices to {output_path}...")
    
    sphere_center = np.array(sphere_center)
    
    def flux_to_vec(flux):
        return mapper.edge_flux_to_node_vector(flux)
    
    gt_vec = flux_to_vec(GT_Flux[idx])
    pred_vecs = {name: flux_to_vec(pred[idx]) for name, pred in predictions_dict.items()}
    
    x_range = np.linspace(-domain_radius * 0.9, domain_radius * 0.9, grid_size)
    y_range = np.linspace(-domain_radius * 0.9, domain_radius * 0.9, grid_size)
    X_grid, Y_grid = np.meshgrid(x_range, y_range)
    
    n_models = len(predictions_dict)
    fig, axes = plt.subplots(1, 1 + n_models, figsize=(6 * (1 + n_models), 6))
    if n_models == 0:
        axes = [axes]
    
    def plot_quiver(ax, vec_data, title):
        interp_x = NearestNDInterpolator(pts, vec_data[:, 0])
        interp_y = NearestNDInterpolator(pts, vec_data[:, 1])
        mag = np.linalg.norm(vec_data, axis=1)
        interp_mag = NearestNDInterpolator(pts, mag)
        
        Z_slice = np.zeros_like(X_grid)
        query_points = np.column_stack([X_grid.ravel(), Y_grid.ravel(), Z_slice.ravel()])
        
        U = interp_x(query_points).reshape(grid_size, grid_size)
        V = interp_y(query_points).reshape(grid_size, grid_size)
        M = interp_mag(query_points).reshape(grid_size, grid_size)
        
        r_grid = np.sqrt((X_grid - sphere_center[0])**2 + (Y_grid - sphere_center[1])**2)
        mask = r_grid < sphere_radius * 1.2
        U[mask] = np.nan
        V[mask] = np.nan
        M[mask] = np.nan
        
        im = ax.pcolormesh(X_grid, Y_grid, M, cmap='viridis', shading='auto', alpha=0.6)
        ax.quiver(X_grid, Y_grid, U, V, color='black', alpha=0.7, scale=None)
        
        circle = plt.Circle((sphere_center[0], sphere_center[1]), 
                             sphere_radius, fill=True, color='gray', alpha=0.8)
        ax.add_patch(circle)
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_xlim(-domain_radius, domain_radius)
        ax.set_ylim(-domain_radius, domain_radius)
        return im
    
    im = plot_quiver(axes[0], gt_vec, 'GT Vector Field (z=0)')
    plt.colorbar(im, ax=axes[0], shrink=0.8, label='|v|')
    
    for i, (name, pred_vec) in enumerate(pred_vecs.items()):
        im = plot_quiver(axes[i+1], pred_vec, f'{name} (z=0)')
        plt.colorbar(im, ax=axes[i+1], shrink=0.8, label='|v|')
    
    plt.suptitle(f'Sample #{idx}: Vector Field', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    if logger:
        logger.log(f"[Viz] Saved to {output_path}")


def visualize_input_output_slices(pts, X_input, mapper, GT_Flux, predictions_dict, 
                                  output_path, idx=0, sphere_center=(0,0,0),
                                  sphere_radius=0.4, domain_radius=2.0, 
                                  grid_size=100, logger=None):
    if logger:
        logger.log(f"\n[Viz] Rendering Input/Output Slices to {output_path}...")
    
    sphere_center = np.array(sphere_center)
    
    def flux_to_mag(flux):
        vec = mapper.edge_flux_to_node_vector(flux)
        return np.linalg.norm(vec, axis=1)
    
    X = X_input[idx]
    gt_mag = flux_to_mag(GT_Flux[idx])
    pred_mags = {name: flux_to_mag(pred[idx]) for name, pred in predictions_dict.items()}
    
    x_range = np.linspace(-domain_radius * 0.95, domain_radius * 0.95, grid_size)
    y_range = np.linspace(-domain_radius * 0.95, domain_radius * 0.95, grid_size)
    X_grid, Y_grid = np.meshgrid(x_range, y_range)
    
    n_cols = 2 + len(predictions_dict)
    fig, axes = plt.subplots(2, n_cols, figsize=(5 * n_cols, 10))
    
    def plot_scalar_slice(ax, data, title, cmap, vmin=None, vmax=None, symmetric=False):
        interp = NearestNDInterpolator(pts, data)
        Z_slice = np.zeros_like(X_grid)
        query_points = np.column_stack([X_grid.ravel(), Y_grid.ravel(), Z_slice.ravel()])
        slice_data = interp(query_points).reshape(grid_size, grid_size)
        
        r_grid = np.sqrt((X_grid - sphere_center[0])**2 + (Y_grid - sphere_center[1])**2)
        mask = r_grid < sphere_radius * 1.1
        slice_data[mask] = np.nan
        
        if symmetric:
            abs_max = max(abs(np.nanmin(slice_data)), abs(np.nanmax(slice_data)))
            vmin, vmax = -abs_max, abs_max
        
        im = ax.pcolormesh(X_grid, Y_grid, slice_data, cmap=cmap, 
                           vmin=vmin, vmax=vmax, shading='auto')
        circle = plt.Circle((sphere_center[0], sphere_center[1]), 
                             sphere_radius, fill=True, color='gray', alpha=0.8)
        ax.add_patch(circle)
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        return im
    
    im = plot_scalar_slice(axes[0, 0], X, 'Input ρ (z=0)', 'RdBu_r', symmetric=True)
    plt.colorbar(im, ax=axes[0, 0], shrink=0.8)
    
    im = plot_scalar_slice(axes[0, 1], gt_mag, 'GT |v| (z=0)', 'viridis', vmin=0)
    plt.colorbar(im, ax=axes[0, 1], shrink=0.8)
    
    for i, (name, pred_mag) in enumerate(pred_mags.items()):
        im = plot_scalar_slice(axes[0, i+2], pred_mag, f'{name} |v| (z=0)', 'viridis', vmin=0)
        plt.colorbar(im, ax=axes[0, i+2], shrink=0.8)
    
    def plot_xz_slice(ax, data, title, cmap, vmin=None, vmax=None, symmetric=False):
        interp = NearestNDInterpolator(pts, data)
        Z_grid_xz, X_grid_xz = np.meshgrid(x_range, y_range)
        Y_slice = np.zeros_like(X_grid_xz)
        query_points = np.column_stack([X_grid_xz.ravel(), Y_slice.ravel(), Z_grid_xz.ravel()])
        slice_data = interp(query_points).reshape(grid_size, grid_size)
        
        r_grid = np.sqrt((X_grid_xz - sphere_center[0])**2 + (Z_grid_xz - sphere_center[2])**2)
        mask = r_grid < sphere_radius * 1.1
        slice_data[mask] = np.nan
        
        if symmetric:
            abs_max = max(abs(np.nanmin(slice_data)), abs(np.nanmax(slice_data)))
            vmin, vmax = -abs_max, abs_max
        
        im = ax.pcolormesh(X_grid_xz, Z_grid_xz, slice_data, cmap=cmap, 
                           vmin=vmin, vmax=vmax, shading='auto')
        circle = plt.Circle((sphere_center[0], sphere_center[2]), 
                             sphere_radius, fill=True, color='gray', alpha=0.8)
        ax.add_patch(circle)
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel('X')
        ax.set_ylabel('Z')
        return im
    
    im = plot_xz_slice(axes[1, 0], X, 'Input ρ (y=0)', 'RdBu_r', symmetric=True)
    plt.colorbar(im, ax=axes[1, 0], shrink=0.8)
    
    im = plot_xz_slice(axes[1, 1], gt_mag, 'GT |v| (y=0)', 'viridis', vmin=0)
    plt.colorbar(im, ax=axes[1, 1], shrink=0.8)
    
    for i, (name, pred_mag) in enumerate(pred_mags.items()):
        im = plot_xz_slice(axes[1, i+2], pred_mag, f'{name} |v| (y=0)', 'viridis', vmin=0)
        plt.colorbar(im, ax=axes[1, i+2], shrink=0.8)
    
    plt.suptitle(f'Sample #{idx}: Input → Output', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    if logger:
        logger.log(f"[Viz] Saved to {output_path}")


def visualize_streamlines(pts, mapper, GT_Flux, predictions_dict, 
                          output_path, idx=0, sphere_center=(0,0,0),
                          sphere_radius=0.4, domain_radius=2.0, 
                          n_seeds=50, logger=None):
    if logger:
        logger.log(f"\n[Viz] Rendering Streamlines to {output_path}...")
    
    sphere_center = np.array(sphere_center)
    
    def flux_to_vec(flux):
        return mapper.edge_flux_to_node_vector(flux)
    
    gt_vec = flux_to_vec(GT_Flux[idx])
    pred_vecs = {name: flux_to_vec(pred[idx]) for name, pred in predictions_dict.items()}
    
    n_plots = 1 + len(predictions_dict)
    fig = plt.figure(figsize=(7 * n_plots, 7))
    
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 20)
    sx = sphere_center[0] + sphere_radius * np.outer(np.cos(u), np.sin(v))
    sy = sphere_center[1] + sphere_radius * np.outer(np.sin(u), np.sin(v))
    sz = sphere_center[2] + sphere_radius * np.outer(np.ones(np.size(u)), np.cos(v))
    
    def plot_streamlines(ax, vec_data, title):
        interp = [NearestNDInterpolator(pts, vec_data[:, j]) for j in range(3)]
        
        bounds_min, bounds_max = pts.min(axis=0), pts.max(axis=0)
        
        r_from_center = np.linalg.norm(pts - sphere_center, axis=1)
        outer_mask = (r_from_center > sphere_radius * 1.3) & (r_from_center < domain_radius * 0.9)
        outer_pts = pts[outer_mask]
        
        if len(outer_pts) > n_seeds:
            seed_idx = np.random.choice(len(outer_pts), n_seeds, replace=False)
            seeds = outer_pts[seed_idx]
        else:
            seeds = outer_pts
        
        colors = plt.cm.coolwarm(np.linspace(0, 1, len(seeds)))
        
        for seed, color in zip(seeds, colors):
            traj = [seed.copy()]
            pos = seed.copy()
            for _ in range(80):
                if np.any(pos < bounds_min * 0.95) or np.any(pos > bounds_max * 0.95):
                    break
                if np.linalg.norm(pos - sphere_center) < sphere_radius * 1.05:
                    break
                vel = np.array([interp[j](pos)[0] for j in range(3)])
                vel_norm = np.linalg.norm(vel)
                if vel_norm < 1e-8:
                    break
                pos = pos + 0.03 * vel / vel_norm
                traj.append(pos.copy())
            
            if len(traj) > 3:
                traj = np.array(traj)
                ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], 
                        color=color, alpha=0.6, lw=0.8)
        
        ax.plot_surface(sx, sy, sz, color='purple', alpha=0.5)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim(-domain_radius, domain_radius)
        ax.set_ylim(-domain_radius, domain_radius)
        ax.set_zlim(-domain_radius, domain_radius)
    
    ax = fig.add_subplot(1, n_plots, 1, projection='3d')
    plot_streamlines(ax, gt_vec, 'GT Field Lines')
    
    for i, (name, pred_vec) in enumerate(pred_vecs.items()):
        ax = fig.add_subplot(1, n_plots, i + 2, projection='3d')
        plot_streamlines(ax, pred_vec, f'{name} Field Lines')
    
    plt.suptitle(f'Sample #{idx}: Field Lines', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    if logger:
        logger.log(f"[Viz] Saved to {output_path}")


def visualize_multiple_samples(pts, mapper, GT_Flux, predictions_dict, 
                               output_dir, n_samples=10, sample_indices=None,
                               sphere_center=(0,0,0), sphere_radius=0.4,
                               domain_radius=2.0, X_input=None, logger=None):
    if logger:
        logger.log(f"\n[Viz] Generating visualizations for {n_samples} samples...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    slices_dir = os.path.join(output_dir, "slices")
    scatter_dir = os.path.join(output_dir, "3d_scatter")
    os.makedirs(slices_dir, exist_ok=True)
    os.makedirs(scatter_dir, exist_ok=True)
    
    total_samples = len(GT_Flux)
    if sample_indices is not None:
        indices = [i for i in sample_indices if i < total_samples]
    else:
        if n_samples >= total_samples:
            indices = list(range(total_samples))
        else:
            indices = np.linspace(0, total_samples - 1, n_samples, dtype=int).tolist()
    
    saved_paths = {'slices': [], '3d_scatter': []}
    
    for i, idx in enumerate(indices):
        if logger:
            logger.log(f"         [Progress] Rendering sample {idx} ({i+1}/{len(indices)})...")
        
        slice_path = os.path.join(slices_dir, f"sample_{idx:04d}_slices.png")
        visualize_velocity_with_error_slices(
            pts, mapper, GT_Flux, predictions_dict,
            slice_path, idx=idx, sphere_center=sphere_center,
            sphere_radius=sphere_radius, domain_radius=domain_radius,
            logger=None
        )
        saved_paths['slices'].append(slice_path)
        
        if X_input is not None:
            scatter_path = os.path.join(scatter_dir, f"sample_{idx:04d}_3d.png")
            visualize_input_output_3d(
                pts, X_input, mapper, GT_Flux, predictions_dict,
                scatter_path, idx=idx, sphere_center=sphere_center,
                sphere_radius=sphere_radius, domain_radius=domain_radius,
                logger=None
            )
            saved_paths['3d_scatter'].append(scatter_path)
    
    if logger:
        logger.log(f"[Viz] Saved {len(saved_paths['slices'])} slice visualizations to {slices_dir}")
        if X_input is not None:
            logger.log(f"[Viz] Saved {len(saved_paths['3d_scatter'])} 3D scatter visualizations to {scatter_dir}")
    
    return saved_paths


def visualize_summary_grid(pts, X_input, mapper, GT_Flux, predictions_dict,
                           output_path, n_samples=5, sample_indices=None,
                           sphere_center=(0,0,0), sphere_radius=0.4,
                           domain_radius=2.0, grid_size=80, logger=None):
    if logger:
        logger.log(f"\n[Viz] Generating summary grid with {n_samples} samples...")
    
    sphere_center = np.array(sphere_center)
    
    total_samples = len(GT_Flux)
    if sample_indices is not None:
        indices = [i for i in sample_indices if i < total_samples][:n_samples]
    else:
        if n_samples >= total_samples:
            indices = list(range(total_samples))
        else:
            indices = np.linspace(0, total_samples - 1, n_samples, dtype=int).tolist()
    
    def flux_to_mag(flux):
        vec = mapper.edge_flux_to_node_vector(flux)
        return np.linalg.norm(vec, axis=1)
    
    model_names = list(predictions_dict.keys())
    n_cols = 2 + len(model_names)
    n_rows = len(indices)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 4 * n_rows))
    if n_rows == 1:
        axes = [axes]
    
    x_range = np.linspace(-domain_radius * 0.95, domain_radius * 0.95, grid_size)
    y_range = np.linspace(-domain_radius * 0.95, domain_radius * 0.95, grid_size)
    X_grid, Y_grid = np.meshgrid(x_range, y_range)
    
    r_grid = np.sqrt((X_grid - sphere_center[0])**2 + (Y_grid - sphere_center[1])**2)
    mask = r_grid < sphere_radius * 1.1
    
    for row, idx in enumerate(indices):
        X = X_input[idx]
        gt_mag = flux_to_mag(GT_Flux[idx])
        
        def plot_slice(ax, data, cmap, vmin=None, vmax=None, symmetric=False):
            interp = NearestNDInterpolator(pts, data)
            Z_slice = np.zeros_like(X_grid)
            query_points = np.column_stack([X_grid.ravel(), Y_grid.ravel(), Z_slice.ravel()])
            slice_data = interp(query_points).reshape(grid_size, grid_size)
            slice_data[mask] = np.nan
            
            if symmetric:
                abs_max = max(abs(np.nanmin(slice_data)), abs(np.nanmax(slice_data)), 1e-6)
                vmin, vmax = -abs_max, abs_max
            
            im = ax.pcolormesh(X_grid, Y_grid, slice_data, cmap=cmap, 
                               vmin=vmin, vmax=vmax, shading='auto')
            circle = plt.Circle((sphere_center[0], sphere_center[1]), 
                                 sphere_radius, fill=True, color='gray', alpha=0.8)
            ax.add_patch(circle)
            ax.set_aspect('equal')
            ax.set_xticks([])
            ax.set_yticks([])
            return im
        
        ax = axes[row][0] if n_rows > 1 else axes[0]
        im = plot_slice(ax, X, 'RdBu_r', symmetric=True)
        if row == 0:
            ax.set_title('Input ρ', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'#{idx}', fontsize=11)
        
        ax = axes[row][1] if n_rows > 1 else axes[1]
        im = plot_slice(ax, gt_mag, 'viridis', vmin=0)
        if row == 0:
            ax.set_title('GT |v|', fontsize=12, fontweight='bold')
        
        for col, name in enumerate(model_names):
            pred_mag = flux_to_mag(predictions_dict[name][idx])
            ax = axes[row][col + 2] if n_rows > 1 else axes[col + 2]
            im = plot_slice(ax, pred_mag, 'viridis', vmin=0)
            if row == 0:
                ax.set_title(name, fontsize=12, fontweight='bold')
    
    plt.suptitle('Multi-Sample Summary (z=0 slice)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    if logger:
        logger.log(f"[Viz] Saved summary grid to {output_path}")


def visualize_summary_grid_3d(pts, X_input, mapper, GT_Flux, predictions_dict,
                              output_path, n_samples=3, sample_indices=None,
                              sphere_center=(0,0,0), sphere_radius=0.4,
                              domain_radius=2.0, logger=None):
    if logger:
        logger.log(f"\n[Viz] Generating 3D summary grid with {n_samples} samples...")
    
    sphere_center = np.array(sphere_center)
    
    total_samples = len(GT_Flux)
    if sample_indices is not None:
        indices = [i for i in sample_indices if i < total_samples][:n_samples]
    else:
        if n_samples >= total_samples:
            indices = list(range(min(n_samples, total_samples)))
        else:
            indices = np.linspace(0, total_samples - 1, n_samples, dtype=int).tolist()
    
    def flux_to_vec(flux):
        return mapper.edge_flux_to_node_vector(flux)
    
    r_from_center = np.linalg.norm(pts - sphere_center, axis=1)
    outer_limit = domain_radius * 0.85
    inner_limit = sphere_radius * 1.2
    valid_mask = (r_from_center < outer_limit) & (r_from_center > inner_limit)
    valid_indices_pts = np.where(valid_mask)[0]
    r_valid = r_from_center[valid_mask]
    dist_sort_idx = valid_indices_pts[np.argsort(r_valid)]
    points_plot = pts[dist_sort_idx]
    
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 20)
    sx = sphere_center[0] + sphere_radius * np.outer(np.cos(u), np.sin(v))
    sy = sphere_center[1] + sphere_radius * np.outer(np.sin(u), np.sin(v))
    sz = sphere_center[2] + sphere_radius * np.outer(np.ones(np.size(u)), np.cos(v))
    
    model_names = list(predictions_dict.keys())
    n_cols = 2 + len(model_names)
    n_rows = len(indices)
    
    fig = plt.figure(figsize=(6 * n_cols, 6 * n_rows))
    
    def compute_radial_component(vec):
        r_vec = pts - sphere_center
        r_norm = np.linalg.norm(r_vec, axis=1, keepdims=True)
        r_hat = r_vec / (r_norm + 1e-6)
        return np.sum(vec * r_hat, axis=1)
    
    stride = max(1, len(pts) // 200)
    quiver_idx = np.arange(0, len(pts), stride)
    
    for row, idx in enumerate(indices):
        X = X_input[idx]
        gt_vec = flux_to_vec(GT_Flux[idx])
        
        X_plot = X[dist_sort_idx]
        ax = fig.add_subplot(n_rows, n_cols, row * n_cols + 1, projection='3d')
        sc = ax.scatter(points_plot[:, 0], points_plot[:, 1], points_plot[:, 2],
                        c=X_plot, cmap='RdBu_r', s=10, alpha=0.9,
                        vmin=-1, vmax=1, edgecolors='none')
        ax.plot_surface(sx, sy, sz, color='gray', alpha=0.3)
        if row == 0:
            ax.set_title('Input X (ρ)', fontweight='bold')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.text2D(-0.1, 0.5, f'#{idx}', transform=ax.transAxes, fontsize=11, 
                  fontweight='bold', va='center')
        
        gt_radial = compute_radial_component(gt_vec)
        gt_radial_max = np.abs(gt_radial).max()
        if gt_radial_max > 1e-10:
            gt_radial_normalized = gt_radial / gt_radial_max
        else:
            gt_radial_normalized = gt_radial
        gt_radial_plot = gt_radial_normalized[dist_sort_idx]
        
        ax = fig.add_subplot(n_rows, n_cols, row * n_cols + 2, projection='3d')
        sc = ax.scatter(points_plot[:, 0], points_plot[:, 1], points_plot[:, 2],
                        c=gt_radial_plot, cmap='RdBu_r', s=10, alpha=0.9,
                        vmin=-1, vmax=1, edgecolors='none')
        ax.quiver(pts[quiver_idx, 0], pts[quiver_idx, 1], pts[quiver_idx, 2],
                  gt_vec[quiver_idx, 0], gt_vec[quiver_idx, 1], gt_vec[quiver_idx, 2],
                  length=0.1, normalize=True, alpha=0.5, color='black')
        ax.plot_surface(sx, sy, sz, color='purple', alpha=0.5)
        if row == 0:
            ax.set_title('GT Output Y', fontweight='bold')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        for col, name in enumerate(model_names):
            pred_vec = flux_to_vec(predictions_dict[name][idx])
            pred_radial = compute_radial_component(pred_vec)
            pred_radial_max = np.abs(pred_radial).max()
            if pred_radial_max > 1e-10:
                pred_radial_normalized = pred_radial / pred_radial_max
            else:
                pred_radial_normalized = pred_radial
            pred_radial_plot = pred_radial_normalized[dist_sort_idx]
            
            ax = fig.add_subplot(n_rows, n_cols, row * n_cols + 3 + col, projection='3d')
            sc = ax.scatter(points_plot[:, 0], points_plot[:, 1], points_plot[:, 2],
                            c=pred_radial_plot, cmap='RdBu_r', s=10, alpha=0.9,
                            vmin=-1, vmax=1, edgecolors='none')
            ax.quiver(pts[quiver_idx, 0], pts[quiver_idx, 1], pts[quiver_idx, 2],
                      pred_vec[quiver_idx, 0], pred_vec[quiver_idx, 1], pred_vec[quiver_idx, 2],
                      length=0.1, normalize=True, alpha=0.5, color='black')
            ax.plot_surface(sx, sy, sz, color='purple', alpha=0.5)
            if row == 0:
                ax.set_title(f'{name}', fontweight='bold')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
    
    plt.suptitle('Multi-Sample 3D Summary', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    if logger:
        logger.log(f"[Viz] Saved 3D summary grid to {output_path}")


def plot_training_curves(all_losses, output_path, logger=None):
    model_names = list(all_losses.keys())
    n_models = len(model_names)
    
    # n_cols = min(4, n_models)
    n_cols = 3
    n_rows = (n_models + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4.5 * n_rows))
    axes = np.array(axes).flatten() if n_models > 1 else [axes]
    
    for i, name in enumerate(model_names):
        train_l, val_l = all_losses[name]
        axes[i].plot(train_l, label='Train', alpha=0.8, linewidth=2)
        axes[i].plot(val_l, label='Val', alpha=0.8, linewidth=2)
        axes[i].set_title(f'{name} Training', fontsize=14)
        axes[i].set_xlabel('Epoch', fontsize=12)
        axes[i].set_ylabel('Loss', fontsize=12)
        axes[i].legend(fontsize=10)
        axes[i].set_yscale('log')
        axes[i].grid(True, alpha=0.3)
    
    for i in range(len(model_names), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    if logger:
        logger.log(f"[Viz] Training curves saved to {output_path}")
    
    base_path, ext = os.path.splitext(output_path)
    for name in model_names:
        train_l, val_l = all_losses[name]
        
        plt.figure(figsize=(10, 7))
        plt.plot(train_l, label='Train Loss', linewidth=2.5, color='tab:blue', alpha=0.8)
        plt.plot(val_l, label='Val Loss', linewidth=2.5, color='tab:orange', alpha=0.8)
        
        plt.title(f'{name} Training Process', fontsize=18, fontweight='bold')
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('MSE Loss (Log Scale)', fontsize=14)
        plt.legend(fontsize=12)
        plt.yscale('log')
        plt.grid(True, which="both", ls="--", alpha=0.4)
        
        plt.tight_layout()
        indiv_path = f"{base_path}_{name}{ext}"
        plt.savefig(indiv_path, dpi=200, bbox_inches='tight')
        plt.close()


def plot_training_curves_combined(all_losses, output_path, logger=None):
    model_names = list(all_losses.keys())
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(model_names)))
    
    for i, name in enumerate(model_names):
        train_l, val_l = all_losses[name]
        ax1.plot(train_l, label=name, color=colors[i], alpha=0.8, linewidth=2)
    
    ax1.set_title('Training Loss', fontsize=16, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=14)
    ax1.set_ylabel('Loss', fontsize=14)
    ax1.set_yscale('log')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    for i, name in enumerate(model_names):
        train_l, val_l = all_losses[name]
        ax2.plot(val_l, label=name, color=colors[i], alpha=0.8, linewidth=2)
    
    ax2.set_title('Validation Loss', fontsize=16, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=14)
    ax2.set_ylabel('Loss', fontsize=14)
    ax2.set_yscale('log')
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    if logger:
        logger.log(f"[Viz] Combined training curves saved to {output_path}")


def plot_metrics_comparison(all_results, output_path, logger=None):
    model_names = list(all_results.keys())
    
    metrics_config = [
        ('divergence_fidelity', 'Div Fidelity'),
        ('vorticity_fidelity', 'Vort Fidelity'),
        ('spectral_fidelity', 'Spectral Fidelity'),
        ('gradient_fidelity', 'Grad Fidelity'),
        ('energy_fidelity', 'Energy Fidelity'),
        ('enstrophy_fidelity', 'Enstrophy Fidelity'),
        ('betti0_score', 'Connectivity'),
        ('level_set_iou', 'Level Set IoU'),
        ('vortex_count_accuracy', 'Vortex Accuracy'),
    ]
    
    available_metrics = []
    for key, label in metrics_config:
        for name in model_names:
            if key in all_results[name]:
                available_metrics.append((key, label))
                break
    
    if not available_metrics:
        if logger:
            logger.log("[Viz] No metrics available for plotting")
        return
    
    metrics_keys = [m[0] for m in available_metrics]
    metric_labels = [m[1] for m in available_metrics]
    
    values = {name: [] for name in model_names}
    for name in model_names:
        for key in metrics_keys:
            val = all_results[name].get(key, 0.0)
            if np.isnan(val):
                val = 0.0
            values[name].append(val)
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(model_names)))
    x = np.arange(len(metrics_keys))
    
    total_width = 0.8
    bar_width = total_width / len(model_names)
    
    fig, ax = plt.subplots(figsize=(max(12, 1.5 * len(metrics_keys)), 7))
    
    for i, name in enumerate(model_names):
        offset = (i - len(model_names)/2 + 0.5) * bar_width
        ax.bar(x + offset, values[name], bar_width, label=name, 
               color=colors[i], edgecolor='white', linewidth=0.5, alpha=0.9)
    
    ax.set_ylim(0, 1.15)
    ax.set_ylabel('Score (Higher is Better)', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=10, rotation=30, ha='right')
    ax.set_title('Model Comparison: Topology & Physics Metrics', fontsize=16, fontweight='bold')
    ax.grid(True, axis='y', linestyle='--', alpha=0.4)
    ax.legend(loc='upper right', fontsize=10)
    ax.axhline(y=0, color='black', linewidth=1)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    if logger:
        logger.log(f"[Viz] Metrics comparison saved to {output_path}")


def generate_all_flux_visualizations(pts, mapper, X_test, GT_Flux, predictions_dict,
                                     all_losses, all_results, output_dir, 
                                     n_samples=10, sphere_center=(0,0,0),
                                     sphere_radius=0.4, domain_radius=2.0,
                                     logger=None):
    os.makedirs(output_dir, exist_ok=True)
    
    if logger:
        logger.log(f"\n{'='*60}")
        logger.log("GENERATING ALL FLUX VISUALIZATIONS")
        logger.log(f"{'='*60}")
    
    paths = {}
    
    try:
        main_path = os.path.join(output_dir, "velocity_comparison.png")
        visualize_velocity_with_error_slices(
            pts, mapper, GT_Flux, predictions_dict, main_path, idx=0,
            sphere_center=sphere_center, sphere_radius=sphere_radius,
            domain_radius=domain_radius, logger=logger
        )
        paths['main'] = main_path
    except Exception as e:
        if logger:
            logger.log(f"[Viz] Error in velocity comparison: {e}")
    
    if X_test is not None:
        try:
            io_3d_path = os.path.join(output_dir, "input_output_3d.png")
            visualize_input_output_3d(
                pts, X_test, mapper, GT_Flux, predictions_dict, io_3d_path, idx=0,
                sphere_center=sphere_center, sphere_radius=sphere_radius,
                domain_radius=domain_radius, logger=logger
            )
            paths['input_output_3d'] = io_3d_path
        except Exception as e:
            if logger:
                logger.log(f"[Viz] Error in 3D input/output: {e}")
        
        try:
            sample_3d_path = os.path.join(output_dir, "sample_3d_full.png")
            visualize_sample_3d_full(
                pts, X_test, mapper, GT_Flux, predictions_dict, sample_3d_path, idx=0,
                sphere_center=sphere_center, sphere_radius=sphere_radius,
                domain_radius=domain_radius, logger=logger
            )
            paths['sample_3d_full'] = sample_3d_path
        except Exception as e:
            if logger:
                logger.log(f"[Viz] Error in full 3D sample: {e}")
    
    try:
        vec_path = os.path.join(output_dir, "vector_field.png")
        visualize_vector_field_slices(
            pts, mapper, GT_Flux, predictions_dict, vec_path, idx=0,
            sphere_center=sphere_center, sphere_radius=sphere_radius,
            domain_radius=domain_radius, logger=logger
        )
        paths['vector'] = vec_path
    except Exception as e:
        if logger:
            logger.log(f"[Viz] Error in vector field: {e}")
    
    if X_test is not None:
        try:
            io_path = os.path.join(output_dir, "input_output.png")
            visualize_input_output_slices(
                pts, X_test, mapper, GT_Flux, predictions_dict, io_path, idx=0,
                sphere_center=sphere_center, sphere_radius=sphere_radius,
                domain_radius=domain_radius, logger=logger
            )
            paths['input_output'] = io_path
        except Exception as e:
            if logger:
                logger.log(f"[Viz] Error in input/output: {e}")
    
    try:
        stream_path = os.path.join(output_dir, "streamlines.png")
        visualize_streamlines(
            pts, mapper, GT_Flux, predictions_dict, stream_path, idx=0,
            sphere_center=sphere_center, sphere_radius=sphere_radius,
            domain_radius=domain_radius, logger=logger
        )
        paths['streamlines'] = stream_path
    except Exception as e:
        if logger:
            logger.log(f"[Viz] Error in streamlines: {e}")
    
    try:
        samples_dir = os.path.join(output_dir, "samples")
        paths['samples'] = visualize_multiple_samples(
            pts, mapper, GT_Flux, predictions_dict, samples_dir, 
            n_samples=n_samples, sphere_center=sphere_center,
            sphere_radius=sphere_radius, domain_radius=domain_radius,
            X_input=X_test, logger=logger
        )
    except Exception as e:
        if logger:
            logger.log(f"[Viz] Error in multiple samples: {e}")
    
    if X_test is not None:
        try:
            summary_path = os.path.join(output_dir, "summary_grid.png")
            visualize_summary_grid(
                pts, X_test, mapper, GT_Flux, predictions_dict, summary_path,
                n_samples=min(5, n_samples), sphere_center=sphere_center,
                sphere_radius=sphere_radius, domain_radius=domain_radius,
                logger=logger
            )
            paths['summary'] = summary_path
        except Exception as e:
            if logger:
                logger.log(f"[Viz] Error in summary grid: {e}")
        
        try:
            summary_3d_path = os.path.join(output_dir, "summary_grid_3d.png")
            visualize_summary_grid_3d(
                pts, X_test, mapper, GT_Flux, predictions_dict, summary_3d_path,
                n_samples=min(3, n_samples), sphere_center=sphere_center,
                sphere_radius=sphere_radius, domain_radius=domain_radius,
                logger=logger
            )
            paths['summary_3d'] = summary_3d_path
        except Exception as e:
            if logger:
                logger.log(f"[Viz] Error in 3D summary grid: {e}")
    
    if all_losses:
        try:
            curves_path = os.path.join(output_dir, "training_curves.png")
            plot_training_curves(all_losses, curves_path, logger=logger)
            paths['training_curves'] = curves_path
            
            combined_path = os.path.join(output_dir, "training_curves_combined.png")
            plot_training_curves_combined(all_losses, combined_path, logger=logger)
            paths['training_curves_combined'] = combined_path
        except Exception as e:
            if logger:
                logger.log(f"[Viz] Error in training curves: {e}")
    
    if all_results:
        try:
            metrics_path = os.path.join(output_dir, "metrics_comparison.png")
            plot_metrics_comparison(all_results, metrics_path, logger=logger)
            paths['metrics'] = metrics_path
        except Exception as e:
            if logger:
                logger.log(f"[Viz] Error in metrics comparison: {e}")
    
    if logger:
        logger.log(f"\n[Viz] All visualizations saved to {output_dir}")
    
    return paths