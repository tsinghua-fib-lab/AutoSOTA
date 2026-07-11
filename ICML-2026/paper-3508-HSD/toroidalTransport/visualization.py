"""
Visualization utilities for scalar field prediction

This module provides comprehensive visualization tools for comparing
scalar field predictions across multiple models.

Features:
- Multi-sample visualization (10 groups by default)
- Initial condition comparison
- Shared colorbar across all views
- Publication-ready figure formatting
- Training curve plotting
"""
import os
import uuid
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.gridspec import GridSpec


def visualize_scalar_results(pts, faces, GT_U, predictions_dict, mse_dict, 
                             output_path, idx=0, logger=None):
    """
    Visualize scalar field on mesh surface with shared colorbar.
    (Original function for backward compatibility)
    """
    if logger:
        logger.log(f"\n[Viz] Rendering Scalar Field to {output_path}...")
    
    # Create mesh
    padded_faces = np.hstack([np.full((faces.shape[0], 1), 3), faces])
    mesh = pv.PolyData(pts, padded_faces.flatten().astype(np.int32))
    
    # Calculate global vmin/vmax
    all_data = [GT_U[idx]] + [pred[idx] for pred in predictions_dict.values()]
    global_vmin = min(d.min() for d in all_data)
    global_vmax = max(d.max() for d in all_data)
    
    if logger:
        logger.log(f"         [Config] Shared Range: [{global_vmin:.4e}, {global_vmax:.4e}]")

    def render_scalar(scalar_data, name):
        unique_id = str(uuid.uuid4())[:8]
        fname = f"tmp_viz_{unique_id}.png"
        
        try:
            scalar_data = scalar_data.flatten() 
            d_min, d_max = scalar_data.min(), scalar_data.max()
            title = f"{name}\n[{d_min:.2e}, {d_max:.2e}]"

            mesh_copy = mesh.copy()
            mesh_copy.point_data["scalar"] = scalar_data
            
            # Window size increased by 1.5x: (540, 420) -> (810, 630)
            pl = pv.Plotter(off_screen=True, window_size=(810, 630))
            pl.set_background("white")
            
            pl.add_mesh(
                mesh_copy, 
                scalars="scalar", 
                cmap="coolwarm",
                show_scalar_bar=False,
                clim=[global_vmin, global_vmax]
            )
            
            pl.view_isometric()
            pl.camera.zoom(1.3)  # Zoom in to fill the frame
            # Font size: 8 + 11 = 19
            pl.add_text(title, color="black", font_size=19, position='upper_left')
            
            pl.screenshot(fname)
            pl.close()
            return fname
        except Exception as e:
            if logger:
                logger.log(f"    Render error: {e}")
            return None

    # Render all models
    imgs = [render_scalar(GT_U[idx], "Ground Truth")]
    for name, pred in predictions_dict.items():
        imgs.append(render_scalar(pred[idx], f"{name}"))
    
    imgs = [img for img in imgs if img is not None]

    try:
        n_cols = len(imgs)
        if n_cols == 0: 
            return

        # Figure size increased by 1.5x: (3.5*n_cols, 3.5) -> (5.25*n_cols, 5.25)
        fig, axes = plt.subplots(1, n_cols, figsize=(5.25*n_cols, 5.25))
        
        if n_cols == 1: 
            axes = [axes]
            
        for ax, fname in zip(axes, imgs):
            if fname and os.path.exists(fname):
                img = mpimg.imread(fname)
                ax.imshow(img)
                ax.axis("off")
                os.remove(fname)
            else:
                ax.axis("off")
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        if logger:
            logger.log(f"[Viz] Saved visualization to {output_path}")
    except Exception as e:
        if logger:
            logger.log(f"Viz Error: {e}")


def _render_mesh_scalar(mesh, scalar_data, title, global_vmin, global_vmax, 
                        cmap="coolwarm", window_size=(900, 720)):
    """
    Helper function to render a single scalar field on mesh.
    
    Returns:
        str: Path to temporary image file, or None if failed
    """
    unique_id = str(uuid.uuid4())[:8]
    fname = f"tmp_viz_{unique_id}.png"
    
    try:
        scalar_data = scalar_data.flatten()
        mesh_copy = mesh.copy()
        mesh_copy.point_data["scalar"] = scalar_data
        
        # Window size default increased by 1.5x: (600, 480) -> (900, 720)
        pl = pv.Plotter(off_screen=True, window_size=window_size)
        pl.set_background("white")
        
        pl.add_mesh(
            mesh_copy, 
            scalars="scalar", 
            cmap=cmap,
            show_scalar_bar=False,
            clim=[global_vmin, global_vmax]
        )
        
        pl.view_isometric()
        pl.camera.zoom(1.3)  # Zoom in to fill the frame
        pl.add_text(title, color="black", font_size=20, position='upper_left')
        
        pl.screenshot(fname)
        pl.close()
        return fname
    except Exception as e:
        print(f"    Render error: {e}")
        return None



def visualize_with_initial_condition(pts, faces, X_input, Y_gt, predictions_dict, 
                                     mse_dict, output_path, idx=0, logger=None):
    """
    Visualize scalar field comparison including initial condition.
    
    Layout: Initial → Ground Truth → Model1 → Model2 → ...
    
    Args:
        pts: (N, 3) node coordinates
        faces: (M, 3) triangle faces
        X_input: (B, N) initial conditions
        Y_gt: (B, N) ground truth outputs
        predictions_dict: {'ModelName': predictions_array, ...}
        mse_dict: {'ModelName': mse_value, ...}
        output_path: path to save the figure
        idx: sample index to visualize
        logger: optional logger instance
    """
    if logger:
        logger.log(f"\n[Viz] Rendering with Initial Condition to {output_path}...")
    
    # Create mesh
    padded_faces = np.hstack([np.full((faces.shape[0], 1), 3), faces])
    mesh = pv.PolyData(pts, padded_faces.flatten().astype(np.int32))
    
    # Calculate global vmin/vmax (including initial condition)
    all_data = [X_input[idx], Y_gt[idx]] + [pred[idx] for pred in predictions_dict.values()]
    global_vmin = min(d.min() for d in all_data)
    global_vmax = max(d.max() for d in all_data)
    
    if logger:
        logger.log(f"         [Config] Shared Range: [{global_vmin:.4e}, {global_vmax:.4e}]")

    # Render all fields
    temp_files = []
    
    # Initial condition
    init_min, init_max = X_input[idx].min(), X_input[idx].max()
    title = f"Initial (t=0)\n[{init_min:.2e}, {init_max:.2e}]"
    temp_files.append(_render_mesh_scalar(mesh, X_input[idx], title, global_vmin, global_vmax))
    
    # Ground truth
    gt_min, gt_max = Y_gt[idx].min(), Y_gt[idx].max()
    title = f"Ground Truth\n[{gt_min:.2e}, {gt_max:.2e}]"
    temp_files.append(_render_mesh_scalar(mesh, Y_gt[idx], title, global_vmin, global_vmax))
    
    # Predictions
    for name, pred in predictions_dict.items():
        pred_min, pred_max = pred[idx].min(), pred[idx].max()
        mse = mse_dict.get(name, 0)
        title = f"{name}"
        temp_files.append(_render_mesh_scalar(mesh, pred[idx], title, global_vmin, global_vmax))
    
    # Filter out failed renders
    temp_files = [f for f in temp_files if f is not None]

    # Compose figure
    try:
        n_cols = len(temp_files)
        if n_cols == 0: 
            return

        # Figure size increased by 1.5x: (3.5*n_cols, 4) -> (5.25*n_cols, 6)
        fig, axes = plt.subplots(1, n_cols, figsize=(5.25*n_cols, 6))
        
        if n_cols == 1: 
            axes = [axes]
            
        for ax, fname in zip(axes, temp_files):
            if fname and os.path.exists(fname):
                img = mpimg.imread(fname)
                ax.imshow(img)
                ax.axis("off")
                os.remove(fname)
            else:
                ax.axis("off")
        
        # Font size: 12 + 11 = 23
        plt.suptitle(f"Sample #{idx}: Initial Condition → Prediction Comparison", 
                     fontsize=23, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        if logger:
            logger.log(f"[Viz] Saved visualization to {output_path}")
    except Exception as e:
        if logger:
            logger.log(f"Viz Error: {e}")
        # Cleanup temp files
        for f in temp_files:
            if f and os.path.exists(f):
                os.remove(f)


def visualize_multiple_samples(pts, faces, X_input, Y_gt, predictions_dict, 
                               mse_dict, output_dir, n_samples=10, 
                               sample_indices=None, logger=None):
    """
    Visualize multiple samples (10 groups by default) with initial conditions.
    
    Creates individual figures for each sample showing:
    Initial → Ground Truth → Model1 → Model2 → ...
    
    Args:
        pts: (N, 3) node coordinates
        faces: (M, 3) triangle faces
        X_input: (B, N) initial conditions
        Y_gt: (B, N) ground truth outputs
        predictions_dict: {'ModelName': predictions_array, ...}
        mse_dict: {'ModelName': mse_array_or_value, ...} - can be per-sample MSE
        output_dir: directory to save figures
        n_samples: number of samples to visualize (default: 10)
        sample_indices: specific indices to visualize (overrides n_samples)
        logger: optional logger instance
    
    Returns:
        list: paths to saved figures
    """
    if logger:
        logger.log(f"\n[Viz] Generating visualizations for {n_samples} samples...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine sample indices
    total_samples = len(Y_gt)
    if sample_indices is not None:
        indices = [i for i in sample_indices if i < total_samples]
    else:
        # Evenly spaced samples
        if n_samples >= total_samples:
            indices = list(range(total_samples))
        else:
            indices = np.linspace(0, total_samples - 1, n_samples, dtype=int).tolist()
    
    if logger:
        logger.log(f"         [Config] Visualizing samples: {indices}")
    
    # Create mesh once
    padded_faces = np.hstack([np.full((faces.shape[0], 1), 3), faces])
    mesh = pv.PolyData(pts, padded_faces.flatten().astype(np.int32))
    
    saved_paths = []
    
    for i, idx in enumerate(indices):
        if logger:
            logger.log(f"         [Progress] Rendering sample {idx} ({i+1}/{len(indices)})...")
        
        # Calculate global vmin/vmax for this sample
        all_data = [X_input[idx], Y_gt[idx]] + [pred[idx] for pred in predictions_dict.values()]
        global_vmin = min(d.min() for d in all_data)
        global_vmax = max(d.max() for d in all_data)
        
        # Render all fields
        temp_files = []
        labels = []
        
        # Initial condition
        init_min, init_max = X_input[idx].min(), X_input[idx].max()
        title = f"Initial (t=0)\n[{init_min:.2e}, {init_max:.2e}]"
        temp_files.append(_render_mesh_scalar(mesh, X_input[idx], title, global_vmin, global_vmax))
        labels.append("Initial")
        
        # Ground truth
        gt_min, gt_max = Y_gt[idx].min(), Y_gt[idx].max()
        title = f"Ground Truth\n[{gt_min:.2e}, {gt_max:.2e}]"
        temp_files.append(_render_mesh_scalar(mesh, Y_gt[idx], title, global_vmin, global_vmax))
        labels.append("GT")
        
        # Predictions
        for name, pred in predictions_dict.items():
            pred_min, pred_max = pred[idx].min(), pred[idx].max()
            
            # Get MSE for this sample
            mse_val = mse_dict.get(name, 0)
            if hasattr(mse_val, '__len__') and len(mse_val) > idx:
                mse = mse_val[idx]
            else:
                mse = mse_val
            
            title = f"{name}"
            temp_files.append(_render_mesh_scalar(mesh, pred[idx], title, global_vmin, global_vmax))
            labels.append(name)
        
        # Filter out failed renders
        valid_files = [(f, l) for f, l in zip(temp_files, labels) if f is not None]
        temp_files = [f for f, l in valid_files]
        
        # Compose figure
        try:
            n_cols = len(temp_files)
            if n_cols == 0: 
                continue

            # Figure size increased by 1.5x: (3.2*n_cols, 3.8) -> (4.8*n_cols, 5.7)
            fig, axes = plt.subplots(1, n_cols, figsize=(4.8*n_cols, 5.7))
            
            if n_cols == 1: 
                axes = [axes]
                
            for ax, fname in zip(axes, temp_files):
                if fname and os.path.exists(fname):
                    img = mpimg.imread(fname)
                    ax.imshow(img)
                    ax.axis("off")
                    os.remove(fname)
                else:
                    ax.axis("off")
            
            # Font size: 11 + 11 = 22
            plt.suptitle(f"Sample #{idx}: Evolution Comparison", 
                         fontsize=22, fontweight='bold', y=0.98)
            plt.tight_layout()
            
            output_path = os.path.join(output_dir, f"sample_{idx:04d}.png")
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            saved_paths.append(output_path)
            
        except Exception as e:
            if logger:
                logger.log(f"    Error rendering sample {idx}: {e}")
            # Cleanup temp files
            for f in temp_files:
                if f and os.path.exists(f):
                    os.remove(f)
    
    if logger:
        logger.log(f"[Viz] Saved {len(saved_paths)} visualizations to {output_dir}")
    
    return saved_paths




def _render_error_map(mesh, error_data, title, window_size=(900, 720), clim=None):
    """
    Helper function to render error map (absolute difference).
    
    Args:
        mesh: PyVista mesh object
        error_data: error values to visualize
        title: title for the plot
        window_size: rendering window size
        clim: [vmin, vmax] color limits for unified colorbar. 
              If None, auto-scale to data range (original behavior).
    
    Returns:
        str: Path to temporary image file, or None if failed
    """
    unique_id = str(uuid.uuid4())[:8]
    fname = f"tmp_err_{unique_id}.png"
    
    try:
        error_data = error_data.flatten()
        mesh_copy = mesh.copy()
        mesh_copy.point_data["error"] = error_data
        
        # Window size default increased by 1.5x: (600, 480) -> (900, 720)
        pl = pv.Plotter(off_screen=True, window_size=window_size)
        pl.set_background("white")
        
        # Build add_mesh kwargs - only include clim if provided
        add_mesh_kwargs = {
            "scalars": "error",
            "cmap": "hot",
            "show_scalar_bar": False,
        }
        if clim is not None:
            add_mesh_kwargs["clim"] = clim
        
        pl.add_mesh(mesh_copy, **add_mesh_kwargs)
        
        pl.view_isometric()
        pl.camera.zoom(1.3)  # Zoom in to fill the frame
        # Font size: 9 + 11 = 20
        pl.add_text(title, color="black", font_size=20, position='upper_left')
        
        pl.screenshot(fname)
        pl.close()
        return fname
    except Exception as e:
        print(f"    Render error: {e}")
        return None


def visualize_multiple_samples_with_error(pts, faces, X_input, Y_gt, predictions_dict, 
                                          output_dir, n_samples=10, 
                                          sample_indices=None, logger=None):
    """
    Visualize multiple samples with error maps.
    
    Creates figures showing:
    Row 1: Initial → GT → Model1 → Model2 → ...
    Row 2: (blank) → (blank) → Error1 → Error2 → ...
    
    Note: Error maps now use a unified colorbar across all models within each sample,
          enabling direct visual comparison of error magnitudes.
    
    Args:
        pts: (N, 3) node coordinates
        faces: (M, 3) triangle faces
        X_input: (B, N) initial conditions
        Y_gt: (B, N) ground truth outputs
        predictions_dict: {'ModelName': predictions_array, ...}
        output_dir: directory to save figures
        n_samples: number of samples to visualize
        sample_indices: specific indices to visualize
        logger: optional logger instance
    
    Returns:
        list: paths to saved figures
    """
    if logger:
        logger.log(f"\n[Viz] Generating visualizations with error maps for {n_samples} samples...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine sample indices
    total_samples = len(Y_gt)
    if sample_indices is not None:
        indices = [i for i in sample_indices if i < total_samples]
    else:
        if n_samples >= total_samples:
            indices = list(range(total_samples))
        else:
            indices = np.linspace(0, total_samples - 1, n_samples, dtype=int).tolist()
    
    # Create mesh once
    padded_faces = np.hstack([np.full((faces.shape[0], 1), 3), faces])
    mesh = pv.PolyData(pts, padded_faces.flatten().astype(np.int32))
    
    saved_paths = []
    model_names = list(predictions_dict.keys())
    n_models = len(model_names)
    
    for i, idx in enumerate(indices):
        if logger:
            logger.log(f"         [Progress] Rendering sample {idx} ({i+1}/{len(indices)})...")
        
        # Calculate global vmin/vmax for this sample (for field values)
        all_data = [X_input[idx], Y_gt[idx]] + [pred[idx] for pred in predictions_dict.values()]
        global_vmin = min(d.min() for d in all_data)
        global_vmax = max(d.max() for d in all_data)
        
        # Number of columns: Initial + GT + Models
        n_cols = 2 + n_models
        
        # Render fields (Row 1)
        row1_files = []
        
        # Initial
        init_min, init_max = X_input[idx].min(), X_input[idx].max()
        title = f"Initial (t=0)\n[{init_min:.2e}, {init_max:.2e}]"
        row1_files.append(_render_mesh_scalar(mesh, X_input[idx], title, global_vmin, global_vmax))
        
        # Ground truth
        gt_min, gt_max = Y_gt[idx].min(), Y_gt[idx].max()
        title = f"Ground Truth\n[{gt_min:.2e}, {gt_max:.2e}]"
        row1_files.append(_render_mesh_scalar(mesh, Y_gt[idx], title, global_vmin, global_vmax))
        
        # Predictions
        for name in model_names:
            pred = predictions_dict[name]
            pred_min, pred_max = pred[idx].min(), pred[idx].max()
            mse = float(np.mean((pred[idx] - Y_gt[idx])**2))
            title = f"{name}"
            row1_files.append(_render_mesh_scalar(mesh, pred[idx], title, global_vmin, global_vmax))
        
        # ============================================================
        # Pre-compute all errors and find global max for unified clim
        # ============================================================
        all_errors = []
        for name in model_names:
            pred = predictions_dict[name]
            error = np.abs(pred[idx] - Y_gt[idx])
            all_errors.append(error)
        
        # Global error range for consistent colormap across models
        global_max_err = max([e.max() for e in all_errors]) if all_errors else 1.0
        error_clim = [0, global_max_err]
        
        # Render error maps (Row 2) with unified colorbar
        row2_files = [None, None]  # Blanks for Initial and GT columns
        
        for k, name in enumerate(model_names):
            error = all_errors[k]
            max_err = error.max()
            title = f"|Error|\nMax:{max_err:.2e}"
            # Pass unified clim to ensure consistent color mapping
            row2_files.append(_render_error_map(mesh, error, title, clim=error_clim))
        
        # Compose figure
        try:
            # Figure size increased by 1.5x: (3.0*n_cols, 6.5) -> (4.5*n_cols, 9.75)
            fig, axes = plt.subplots(2, n_cols, figsize=(4.5*n_cols, 9.75))
            
            # Row 1: Fields
            for j, fname in enumerate(row1_files):
                if fname and os.path.exists(fname):
                    img = mpimg.imread(fname)
                    axes[0, j].imshow(img)
                    os.remove(fname)
                axes[0, j].axis("off")
            
            # Row 2: Error maps
            for j, fname in enumerate(row2_files):
                if fname and os.path.exists(fname):
                    img = mpimg.imread(fname)
                    axes[1, j].imshow(img)
                    os.remove(fname)
                else:
                    # Font size: 10 + 11 = 21
                    axes[1, j].text(0.5, 0.5, '', ha='center', va='center', fontsize=21)
                axes[1, j].axis("off")
            
            # Add row labels - font size: 10 + 11 = 21
            axes[0, 0].set_ylabel("Field Values", fontsize=21, labelpad=10)
            axes[1, 0].set_ylabel("Error Maps", fontsize=21, labelpad=10)
            
            # Font size: 12 + 11 = 23
            plt.suptitle(f"Sample #{idx}: Prediction vs Ground Truth", 
                         fontsize=23, fontweight='bold', y=0.98)
            plt.tight_layout()
            
            output_path = os.path.join(output_dir, f"sample_{idx:04d}_with_error.png")
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            saved_paths.append(output_path)
            
        except Exception as e:
            if logger:
                logger.log(f"    Error rendering sample {idx}: {e}")
            # Cleanup
            for f in row1_files + row2_files:
                if f and os.path.exists(f):
                    os.remove(f)
    
    if logger:
        logger.log(f"[Viz] Saved {len(saved_paths)} visualizations with error maps to {output_dir}")
    
    return saved_paths




    

def visualize_summary_grid(pts, faces, X_input, Y_gt, predictions_dict,
                           output_path, n_samples=5, sample_indices=None, logger=None):
    """
    Create a summary grid visualization showing multiple samples in one figure.
    
    Layout:
    - Rows: Different samples
    - Columns: Initial → GT → Model1 → Model2 → ...
    
    Args:
        pts: (N, 3) node coordinates
        faces: (M, 3) triangle faces
        X_input: (B, N) initial conditions
        Y_gt: (B, N) ground truth outputs
        predictions_dict: {'ModelName': predictions_array, ...}
        output_path: path to save the figure
        n_samples: number of samples to show (rows)
        sample_indices: specific indices to visualize
        logger: optional logger instance
    """
    if logger:
        logger.log(f"\n[Viz] Generating summary grid with {n_samples} samples...")
    
    # Determine sample indices
    total_samples = len(Y_gt)
    if sample_indices is not None:
        indices = [i for i in sample_indices if i < total_samples][:n_samples]
    else:
        if n_samples >= total_samples:
            indices = list(range(total_samples))
        else:
            indices = np.linspace(0, total_samples - 1, n_samples, dtype=int).tolist()
    
    # Create mesh
    padded_faces = np.hstack([np.full((faces.shape[0], 1), 3), faces])
    mesh = pv.PolyData(pts, padded_faces.flatten().astype(np.int32))
    
    model_names = list(predictions_dict.keys())
    n_cols = 2 + len(model_names)  # Initial + GT + models
    n_rows = len(indices)
    
    # Calculate global colormap range across all samples
    all_data = []
    for idx in indices:
        all_data.extend([X_input[idx], Y_gt[idx]])
        all_data.extend([pred[idx] for pred in predictions_dict.values()])
    global_vmin = min(d.min() for d in all_data)
    global_vmax = max(d.max() for d in all_data)
    
    if logger:
        logger.log(f"         [Config] Global Range: [{global_vmin:.4e}, {global_vmax:.4e}]")
    
    # Render all images
    all_images = []  # [row][col]
    
    for row, idx in enumerate(indices):
        row_images = []
        
        # Initial - window size increased by 1.5x: (300, 250) -> (450, 375)
        title = f"Sample {idx}\nInitial"
        row_images.append(_render_mesh_scalar(mesh, X_input[idx], title, 
                                               global_vmin, global_vmax, window_size=(450, 375)))
        
        # GT
        title = f"GT"
        row_images.append(_render_mesh_scalar(mesh, Y_gt[idx], title, 
                                               global_vmin, global_vmax, window_size=(450, 375)))
        
        # Models
        for name in model_names:
            pred = predictions_dict[name]
            mse = float(np.mean((pred[idx] - Y_gt[idx])**2))
            title = f"{name}"
            row_images.append(_render_mesh_scalar(mesh, pred[idx], title, 
                                                   global_vmin, global_vmax, window_size=(450, 375)))
        
        all_images.append(row_images)
    
    # Compose figure
    try:
        # Figure size increased by 1.5x: (2.8*n_cols, 2.5*n_rows) -> (4.2*n_cols, 3.75*n_rows)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2*n_cols, 3.75*n_rows))
        
        if n_rows == 1:
            axes = [axes]
        
        for row in range(n_rows):
            for col in range(n_cols):
                ax = axes[row][col] if n_rows > 1 else axes[col]
                fname = all_images[row][col]
                
                if fname and os.path.exists(fname):
                    img = mpimg.imread(fname)
                    ax.imshow(img)
                    os.remove(fname)
                ax.axis("off")
        
        # Column headers - font size: 10 + 11 = 21
        col_labels = ["Initial", "Ground Truth"] + model_names
        for col, label in enumerate(col_labels):
            ax = axes[0][col] if n_rows > 1 else axes[col]
            ax.set_title(label, fontsize=21, fontweight='bold', pad=5)
        
        # Font size: 14 + 11 = 25
        plt.suptitle("Multi-Sample Prediction Comparison", fontsize=25, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        if logger:
            logger.log(f"[Viz] Saved summary grid to {output_path}")
            
    except Exception as e:
        if logger:
            logger.log(f"Viz Error: {e}")
        # Cleanup
        for row_imgs in all_images:
            for f in row_imgs:
                if f and os.path.exists(f):
                    os.remove(f)


def plot_training_curves(all_losses, output_path, logger=None):
    """
    Plot training curves.
    
    Outputs:
    1. A summary grid plot saved to `output_path` (e.g., training_curves.png)
    2. Individual plots for each model saved with suffix (e.g., training_curves_GNO.png)
    """
    
    model_names = list(all_losses.keys())
    n_models = len(model_names)
    
    # ==========================================
    # 1. Draw Summary Grid (Original Logic)
    # ==========================================
    # n_cols = min(4, n_models)
    n_cols = 3
    n_rows = (n_models + n_cols - 1) // n_cols
    
    # Figure size increased by 1.5x: (4*n_cols, 3*n_rows) -> (6*n_cols, 4.5*n_rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 4.5*n_rows))
    axes = np.array(axes).flatten() if n_models > 1 else [axes]
    
    for i, name in enumerate(model_names):
        train_l, val_l = all_losses[name]
        axes[i].plot(train_l, label='Train', alpha=0.8, linewidth=2)
        axes[i].plot(val_l, label='Val', alpha=0.8, linewidth=2)
        # Font sizes: +11
        axes[i].set_title(f'{name} Training', fontsize=21)
        axes[i].set_xlabel('Epoch', fontsize=19)
        axes[i].set_ylabel('Loss', fontsize=19)
        axes[i].legend(fontsize=17)
        axes[i].set_yscale('log')
        axes[i].grid(True, alpha=0.3)
        axes[i].tick_params(axis='both', labelsize=17)
    
    # Hide unused subplots
    for i in range(len(model_names), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    if logger:
        logger.log(f"[Viz] Summary training curves saved to {output_path}")

    # ==========================================
    # 2. Draw Individual Plots 
    # ==========================================

    base_path, ext = os.path.splitext(output_path)
    
    for name in model_names:
        train_l, val_l = all_losses[name]
        
        # Figure size increased by 1.5x: (8, 6) -> (12, 9)
        plt.figure(figsize=(12, 9)) 
        plt.plot(train_l, label='Train Loss', linewidth=3, color='tab:blue', alpha=0.8)
        plt.plot(val_l, label='Val Loss', linewidth=3, color='tab:orange', alpha=0.8)
        
        # Font sizes: +11
        plt.title(f'{name} Training Process', fontsize=25, fontweight='bold')
        plt.xlabel('Epoch', fontsize=23)
        plt.ylabel('MSE Loss (Log Scale)', fontsize=23)
        plt.legend(fontsize=21)
        plt.yscale('log')
        plt.grid(True, which="both", ls="--", alpha=0.4)
        plt.tick_params(axis='both', labelsize=19)
        
        plt.tight_layout()
        
        indiv_path = f"{base_path}_{name}{ext}"
        plt.savefig(indiv_path, dpi=300)
        plt.close() 
        
    if logger:
        logger.log(f"[Viz] Individual training curves saved with prefix {base_path}_*.png")


def plot_training_curves_combined(all_losses, output_path, logger=None):
    """Plot all training curves in a single combined figure."""
    
    model_names = list(all_losses.keys())
    
    # Figure size increased by 1.5x: (12, 5) -> (18, 7.5)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7.5))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(model_names)))
    
    # Training loss
    for i, name in enumerate(model_names):
        train_l, val_l = all_losses[name]
        ax1.plot(train_l, label=name, color=colors[i], alpha=0.8, linewidth=2)
    
    # Font sizes: +11
    ax1.set_title('Training Loss', fontsize=23, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=21)
    ax1.set_ylabel('Loss', fontsize=21)
    ax1.set_yscale('log')
    ax1.legend(loc='upper right', fontsize=17)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='both', labelsize=17)
    
    # Validation loss
    for i, name in enumerate(model_names):
        train_l, val_l = all_losses[name]
        ax2.plot(val_l, label=name, color=colors[i], alpha=0.8, linewidth=2)
    
    # Font sizes: +11
    ax2.set_title('Validation Loss', fontsize=23, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=21)
    ax2.set_ylabel('Loss', fontsize=21)
    ax2.set_yscale('log')
    ax2.legend(loc='upper right', fontsize=17)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='both', labelsize=17)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    if logger:
        logger.log(f"[Viz] Combined training curves saved to {output_path}")


def plot_metrics_comparison(all_results, output_path, logger=None):
    model_names = list(all_results.keys())

    metrics_config = [
        ('gradient_fidelity', r'Grad Fidelity ($\nabla$)'),   
        ('spectral_fidelity', 'Spectral Fidelity'),
        ('energy_fidelity', 'Energy Fidelity'),
        ('betti0_score', r'Connectivity ($S_{\beta_0}$)'),
        ('level_set_iou', 'Level Set IoU'),
        ('net_flux_score', 'Flux Conservation')
    ]

    metrics_keys = [m[0] for m in metrics_config]
    metric_labels = [m[1] for m in metrics_config]

    values = {name: [] for name in model_names}
    for name in model_names:
        for key in metrics_keys:
            val = all_results[name].get(key, 0.0)
            if np.isnan(val):
                val = 0.0
            values[name].append(val)

    colors = plt.cm.Set2(np.linspace(0, 1, len(model_names)))
    x = np.arange(len(metrics_keys))

    total_width = 0.85
    bar_width = total_width / len(model_names)

    fig, ax = plt.subplots(figsize=(14, 7))

    for i, name in enumerate(model_names):
        offset = (i - len(model_names)/2 + 0.5) * bar_width
        
        bars = ax.bar(
            x + offset, 
            values[name], 
            bar_width,
            label=name, 
            color=colors[i], 
            edgecolor='white', 
            linewidth=0.5, 
            alpha=0.9
        )

        if len(model_names) <= 5:
            for j, v in enumerate(values[name]):
                if v > 0.05:
                    ax.text(
                        x[j] + offset, 
                        v + 0.015, 
                        f'{v:.2f}',
                        ha='center', 
                        va='bottom', 
                        fontsize=8, 
                        rotation=90, 
                        color='#333333'
                    )

    ax.set_ylim(0, 1.1)
    ax.set_ylabel('Normalized Score (Higher is Better)', fontsize=12, labelpad=10)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=11, fontweight='medium')

    ax.set_title(
        'Comparative Assessment of Topological and Physical Fidelity',
        fontsize=16, 
        fontweight='bold', 
        pad=20
    )

    ax.grid(True, axis='y', linestyle='--', alpha=0.4, zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.legend(
        loc='lower center', 
        bbox_to_anchor=(0.5, -0.15),
        ncol=min(len(model_names), 6), 
        frameon=False, 
        fontsize=11
    )

    ax.axhline(y=0, color='black', linewidth=1)

    plt.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    if logger:
        logger.log(f"[Viz] Metrics comparison chart saved to {output_path}")



def generate_all_visualizations(pts, faces, X_test, Y_test, predictions_dict, 
                                all_losses, all_results, output_dir, 
                                n_samples=10, logger=None):
    """
    Generate all visualizations for the experiment.
    
    This is a convenience function that generates:
    1. Individual sample visualizations (10 samples)
    2. Summary grid
    3. Training curves
    4. Metrics comparison
    
    Args:
        pts: node coordinates
        faces: triangle faces
        X_test: test inputs
        Y_test: test ground truth
        predictions_dict: model predictions
        all_losses: training losses
        all_results: evaluation results
        output_dir: output directory
        n_samples: number of samples to visualize
        logger: optional logger
    
    Returns:
        dict: paths to all generated figures
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if logger:
        logger.log(f"\n{'='*60}")
        logger.log("GENERATING ALL VISUALIZATIONS")
        logger.log(f"{'='*60}")
    
    paths = {}
    
    # 1. Individual sample visualizations
    sample_dir = os.path.join(output_dir, "samples")
    paths['samples'] = visualize_multiple_samples(
        pts, faces, X_test, Y_test, predictions_dict, {},
        sample_dir, n_samples=n_samples, logger=logger
    )
    
    # 2. Samples with error maps
    error_dir = os.path.join(output_dir, "samples_with_error")
    paths['samples_with_error'] = visualize_multiple_samples_with_error(
        pts, faces, X_test, Y_test, predictions_dict,
        error_dir, n_samples=n_samples, logger=logger
    )
    
    # 3. Summary grid
    summary_path = os.path.join(output_dir, "summary_grid.png")
    visualize_summary_grid(
        pts, faces, X_test, Y_test, predictions_dict,
        summary_path, n_samples=min(5, n_samples), logger=logger
    )
    paths['summary_grid'] = summary_path
    
    # 4. Training curves
    if all_losses:
        curves_path = os.path.join(output_dir, "training_curves.png")
        plot_training_curves(all_losses, curves_path, logger=logger)
        paths['training_curves'] = curves_path
        
        combined_path = os.path.join(output_dir, "training_curves_combined.png")
        plot_training_curves_combined(all_losses, combined_path, logger=logger)
        paths['training_curves_combined'] = combined_path
    
    # 5. Metrics comparison
    if all_results:
        metrics_path = os.path.join(output_dir, "metrics_comparison.png")
        plot_metrics_comparison(all_results, metrics_path, logger=logger)
        paths['metrics_comparison'] = metrics_path
    
    if logger:
        logger.log(f"\n[Viz] All visualizations generated in {output_dir}")
    
    return paths