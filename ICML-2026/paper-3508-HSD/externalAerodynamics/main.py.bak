import os
import pickle
import warnings
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader


warnings.filterwarnings("ignore")
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


import pyvista as pv
pv.OFF_SCREEN = True
try:
    pv.start_xvfb()
except Exception:
    pass


from config import Config, DEVICE
from utils import Logger, count_parameters
from spectral_operators import HighOrderSpectralOperators

from dataset import FluxFieldDataset, DataManager, VectorFluxMapper

from models import (
    GNO, FNO3d, LightweightMGN, DeepONet, GeoFNO,
    SpectralVectorOperator, HSDSpectralVectorFNO
)

from training import (
    train_gno, train_fno, train_mgn, train_deeponet, train_geofno,
    train_HSD
)

from topo_metrics import FluxTopologyEvaluator, evaluate_all_models, save_results_to_csv, save_results_to_json

from visualization import (
    visualize_flux_results, visualize_velocity_magnitude,
    plot_training_curves, generate_all_flux_visualizations
)


def plot_spectral_decay_comparison(evaluator, predictions_dict, Y_test_vectors, output_path, logger=None):
    if logger:
        logger.log("Generating Spectral Decay Analysis plot (Vector-based)...")
    
    n_modes = min(64, evaluator.Phi0.shape[1])
    
    gt_magnitude = np.linalg.norm(Y_test_vectors, axis=-1)
    _, gt_energies, eigenvalues = evaluator.get_spectral_coefficients_from_scalar(gt_magnitude, n_modes)
    gt_energy_mean = gt_energies.mean(axis=0)
    
    frequencies = np.sqrt(np.maximum(eigenvalues, 0))
    
    def smooth_curve(y, window=3):
        if len(y) < window:
            return y
        box = np.ones(window) / window
        y_smooth = np.convolve(y, box, mode='same')
        y_smooth[:window] = y[:window]
        y_smooth[-window:] = y[-window:]
        return y_smooth

    gt_curve = smooth_curve(gt_energy_mean)
    
    colors = {
        'GT': 'black',
        'HSD': '#2ecc71',
        'GNO': '#3498db',
        'DeepONet': '#9b59b6',
        'FNO': '#e74c3c',
        'GeoFNO': '#e67e22',
        'MGN': '#1abc9c',
    }
    
    linestyles = {
        'GT': '-', 'HSD': '-',
        'GNO': '-.', 'DeepONet': ':', 'FNO': '-.',
        'GeoFNO': '--', 'MGN': ':',
    }
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    ax.semilogy(frequencies, gt_curve, color=colors['GT'], 
                linewidth=2.5, label='Ground Truth', linestyle=linestyles['GT'])
    
    model_order = ['HSD', 'GNO', 'DeepONet', 'FNO', 'GeoFNO', 'MGN']
    
    for name in model_order:
        if name not in predictions_dict:
            continue
        pred = predictions_dict[name]
        pred_magnitude = np.linalg.norm(pred, axis=-1)
        _, pred_energies, _ = evaluator.get_spectral_coefficients_from_scalar(pred_magnitude, n_modes)
        pred_energy_mean = pred_energies.mean(axis=0)
        pred_curve = smooth_curve(pred_energy_mean)
        
        ax.semilogy(frequencies, pred_curve, 
                    color=colors.get(name, 'gray'),
                    linewidth=1.5, 
                    label=name,
                    linestyle=linestyles.get(name, '-'),
                    alpha=0.85)
    
    ax.set_xlabel('Intrinsic Frequency $\sqrt{\lambda}$', fontsize=12)
    ax.set_ylabel('Energy Spectrum $|c_k|^2$', fontsize=12)
    ax.set_title('Spectral Regularity Analysis (Velocity Magnitude)', fontsize=14)
    ax.set_yscale('log')
    ax.legend(loc='lower left', fontsize=10, ncol=2)
    ax.grid(True, which='both', linestyle='--', alpha=0.4)
    ax.set_xlim(left=0, right=frequencies.max() * 1.05)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    if logger:
        logger.log(f"  Saved to {output_path}")


def visualize_vector_field_comparison(pts, faces, GT_Vectors, vector_predictions_dict, 
                                       mse_dict, output_path, idx=0, logger=None):
    if logger:
        logger.log("Generating Vector Field Comparison plot...")
    
    gt_magnitude = np.linalg.norm(GT_Vectors[idx], axis=-1)
    
    model_names = list(vector_predictions_dict.keys())
    n_models = len(model_names)
    
    n_cols = (n_models + 2) // 2
    fig, axes = plt.subplots(2, n_cols, figsize=(4 * n_cols, 8))
    axes = axes.flatten()
    
    ax = axes[0]
    sc = ax.tripcolor(pts[:, 0], pts[:, 1], faces, gt_magnitude, shading='gouraud', cmap='viridis')
    ax.set_title('GT Velocity Magnitude', fontsize=10)
    ax.set_aspect('equal')
    ax.axis('off')
    plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    
    for i, name in enumerate(model_names):
        ax = axes[i + 1]
        pred_vec = vector_predictions_dict[name][idx]
        pred_magnitude = np.linalg.norm(pred_vec, axis=-1)
        
        sc = ax.tripcolor(pts[:, 0], pts[:, 1], faces, pred_magnitude, shading='gouraud', cmap='viridis')
        ax.set_title(f'{name}\nVec MSE: {mse_dict[name]:.2e}', fontsize=9)
        ax.set_aspect('equal')
        ax.axis('off')
        plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    
    for j in range(n_models + 1, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    if logger:
        logger.log(f"  Saved to {output_path}")


def check_existing_models(output_dir, model_names):
    existing_models = {}
    for name in model_names:
        model_path = os.path.join(output_dir, f'model_{name.lower()}.pt')
        if os.path.exists(model_path):
            existing_models[name] = model_path
    return existing_models


def ask_user_retrain(existing_models, logger):
    if not existing_models:
        return 'retrain_all'
    
    logger.log("\n" + "=" * 60)
    logger.log("EXISTING TRAINED MODELS DETECTED")
    logger.log("=" * 60)
    logger.log(f"Found {len(existing_models)} pre-trained model(s):")
    for name, path in existing_models.items():
        logger.log(f"  - {name}: {path}")
    
    logger.log("\nOptions:")
    logger.log("  [1] Load existing models and skip training")
    logger.log("  [2] Retrain all models from scratch")
    logger.log("  [3] Train only missing models")
    
    try:
        choice = input("\nEnter your choice (1/2/3) [default=1]: ").strip()
        if choice == '':
            choice = '1'
    except (EOFError, KeyboardInterrupt):
        logger.log("No interactive input available. Defaulting to load existing models.")
        choice = '1'
    
    logger.log(f"User selected: {choice}")
    
    if choice == '1':
        return 'load'
    elif choice == '2':
        return 'retrain_all'
    elif choice == '3':
        return 'train_missing'
    else:
        logger.log("Invalid choice. Defaulting to load existing models.")
        return 'load'


def load_model_weights(model, model_path, logger):
    try:
        state_dict = torch.load(model_path, map_location=DEVICE)
        model.load_state_dict(state_dict)
        logger.log(f"  Loaded weights from {model_path}")
        return True
    except Exception as e:
        logger.log(f"  Failed to load weights: {e}")
        return False


def main():
    cfg = Config()
    
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    log_path = os.path.join(cfg.OUTPUT_DIR, cfg.LOG_FILE)
    logger = Logger(log_path)
    
    logger.section("VECTOR FIELD PREDICTION ON MANIFOLDS")
    logger.log(f"Device: {DEVICE}")
    logger.log("Evaluation Mode: VECTOR FIELD (No Flux Supervision)")
    
    logger.section("[Step 1] Loading Dataset")
    
    data_path = os.path.join(cfg.DATA_DIR, cfg.PICKLE_FILE)
    
    if not os.path.exists(data_path):
        logger.log(f"ERROR: Dataset not found at {data_path}")
        logger.log("Please run the data generation script first.")
        return
    
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    X_data = data['X_data'].astype(np.float32)
    Y_data = data['Y_data'].astype(np.float32)
    pts = data['points'].astype(np.float32)
    faces = data['faces'].astype(np.int32)
    normals = data['normals'].astype(np.float32)
    
    n_nodes = len(pts)
    n_samples = len(X_data)
    
    logger.log(f"Loaded {n_samples} samples from {data_path}")
    logger.log(f"Mesh: {n_nodes} nodes, {len(faces)} faces")
    logger.log(f"X (Forcing):  shape={X_data.shape}, range=[{X_data.min():.4f}, {X_data.max():.4f}]")
    logger.log(f"Y (Velocity): shape={Y_data.shape}, range=[{Y_data.min():.4f}, {Y_data.max():.4f}]")
    
    logger.section("[Step 2] Creating Vector-Flux Mapper")
    
    mapper = VectorFluxMapper(pts, faces)
    logger.log(f"Mapper initialized: {mapper.n_edges} edges (for topology evaluation)")
    
    logger.section("[Step 3] Splitting Data")
    
    indices = np.arange(n_samples)
    idx_train_val, idx_test = train_test_split(indices, test_size=cfg.TEST_RATIO, random_state=42)
    idx_train, idx_val = train_test_split(idx_train_val, test_size=cfg.VAL_RATIO, random_state=42)
    
    logger.log(f"Train: {len(idx_train)}, Val: {len(idx_val)}, Test: {len(idx_test)}")
    
    x_scale_global = np.max(np.abs(X_data[idx_train_val])) + 1e-9
    y_scale_global = np.max(np.abs(Y_data[idx_train_val])) + 1e-9
    logger.log(f"Global Scaling: x_scale={x_scale_global:.6f}, y_scale={y_scale_global:.6f}")
    
    X_train_norm = X_data[idx_train] / x_scale_global
    Y_train_norm = Y_data[idx_train] / y_scale_global
    X_val_norm = X_data[idx_val] / x_scale_global
    Y_val_norm = Y_data[idx_val] / y_scale_global
    X_test_norm = X_data[idx_test] / x_scale_global
    Y_test_norm = Y_data[idx_test] / y_scale_global
    
    X_train = torch.from_numpy(X_train_norm).float().to(DEVICE)
    Y_train = torch.from_numpy(Y_train_norm).float().to(DEVICE)
    X_val = torch.from_numpy(X_val_norm).float().to(DEVICE)
    Y_val = torch.from_numpy(Y_val_norm).float().to(DEVICE)
    X_test = torch.from_numpy(X_test_norm).float().to(DEVICE)
    Y_test = torch.from_numpy(Y_test_norm).float().to(DEVICE)
    
    logger.section("[Step 4] Building Spectral Operators (HOST)")
    
    eigen_basis_start_time = time.time()
    
    host_ops = HighOrderSpectralOperators(
        pts, faces, 
        k_list=(cfg.K_EIGENS, cfg.K_EIGENS, cfg.K_EIGENS),
        logger=logger
    )
    
    eigen_basis_end_time = time.time()
    eigen_basis_elapsed_time = eigen_basis_end_time - eigen_basis_start_time
    
    logger.log(f"\nEigen Basis Preprocessing Time: {eigen_basis_elapsed_time:.4f} seconds\n")
    
    Phi0_torch = torch.from_numpy(host_ops.Phi0[:, :cfg.K_EIGENS].astype(np.float32)).to(DEVICE)
    Phi1_torch = torch.from_numpy(host_ops.Phi1[:, :cfg.K_EIGENS].astype(np.float32)).to(DEVICE)
    logger.log(f"Phi0 shape: {Phi0_torch.shape}")
    logger.log(f"Phi1 shape: {Phi1_torch.shape}")
    
    logger.section("[Step 5] Preparing Spectral Datasets")
    
    train_idx_combined = np.concatenate([idx_train, idx_val])
    
    train_dataset = FluxFieldDataset(
        host_ops, mapper,
        X_data[train_idx_combined], Y_data[train_idx_combined],
        (cfg.K_EIGENS, cfg.K_EIGENS, cfg.K_EIGENS),
        x_scale=x_scale_global, y_scale=y_scale_global, logger=logger
    )
    
    test_dataset = FluxFieldDataset(
        host_ops, mapper,
        X_data[idx_test], Y_data[idx_test],
        (cfg.K_EIGENS, cfg.K_EIGENS, cfg.K_EIGENS),
        x_scale=x_scale_global, y_scale=y_scale_global, logger=logger
    )
    
    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False)
    
    logger.section("[Step 6] Creating Data Manager")
    
    data_mgr = DataManager(n_nodes, pts, DEVICE, faces=faces, grid_res=cfg.FNO_GRID_RES)
    logger.log("DataManager initialized")
    
    logger.section("[Step 7] Initializing Models")
    
    model_names = ['GNO', 'FNO', 'MGN', 'DeepONet', 'GeoFNO', 'HSD']
    
    model_gno = GNO(
        in_channels=4, out_channels=3,
        hidden_channels=cfg.GNO_HIDDEN_CHANNELS,
        projection_channels=cfg.GNO_PROJECTION_CHANNELS,
        n_layers=cfg.GNO_N_LAYERS, radius=cfg.GNO_RADIUS
    ).to(DEVICE)
    
    model_fno = FNO3d(
        in_channels=2, out_channels=3,
        hidden_channels=cfg.FNO_HIDDEN_CHANNELS,
        n_modes=cfg.FNO_MODES, n_layers=cfg.FNO_N_LAYERS
    ).to(DEVICE)
    
    model_mgn = LightweightMGN(
        node_in_dim=4, edge_in_dim=3, out_dim=3,
        hidden_dim=cfg.MGN_HIDDEN_DIM,
        num_layers=cfg.MGN_NUM_LAYERS
    ).to(DEVICE)
    
    model_deeponet = DeepONet(
        branch_input_dim=n_nodes,
        trunk_input_dim=3,
        output_dim=3,
        branch_layers=cfg.DEEPONET_BRANCH_LAYERS,
        trunk_layers=cfg.DEEPONET_TRUNK_LAYERS,
        basis_dim=cfg.DEEPONET_BASIS_DIM
    ).to(DEVICE)
    
    model_geofno = GeoFNO(
        modes=cfg.GEOFNO_MODES,
        width=cfg.GEOFNO_WIDTH,
        layers=cfg.GEOFNO_LAYERS,
        grid_res=cfg.GEOFNO_GRID_RES,
        out_channels=3
    ).to(DEVICE)
    
    model_HSD = HSDSpectralVectorFNO(
        HSD_base_model=SpectralVectorOperator(
            Md0=host_ops.Md0,
            Md1=host_ops.Md1,
            k0=cfg.K_EIGENS, k1=cfg.K_EIGENS, k2=cfg.K_EIGENS,
            hidden_dims=cfg.SPECTRAL_HIDDEN_DIMS
        ),
        data_mgr=data_mgr,
        Phi0=Phi0_torch,
        fno_modes=cfg.HSD_FNO_MODES,
        fno_hidden=cfg.HSD_FNO_HIDDEN,
        fno_layers=cfg.HSD_FNO_LAYERS,
    ).to(DEVICE)
    
    models_dict = {
        'GNO': model_gno,
        'FNO': model_fno,
        'MGN': model_mgn,
        'DeepONet': model_deeponet,
        'GeoFNO': model_geofno,
        'HSD': model_HSD
    }
    
    params_dict = {name: count_parameters(model) for name, model in models_dict.items()}
    
    for name in model_names:
        logger.log(f"[{name}] Parameters: {params_dict[name]:,}")
    
    logger.section("[Step 8] Training Models")
    
    existing_models = check_existing_models(cfg.OUTPUT_DIR, model_names)
    train_decision = ask_user_retrain(existing_models, logger)
    
    all_losses = {}
    all_vram = {}
    all_time = {}
    
    if train_decision == 'load':
        logger.log("\nLoading pre-trained models...")
        for name, model in models_dict.items():
            model_path = os.path.join(cfg.OUTPUT_DIR, f'model_{name.lower()}.pt')
            if os.path.exists(model_path):
                load_model_weights(model, model_path, logger)
        log_pkl_path = os.path.join(cfg.OUTPUT_DIR, 'experiment_log.pkl')
        if os.path.exists(log_pkl_path):
            with open(log_pkl_path, 'rb') as f:
                log_data = pickle.load(f)
                all_losses = log_data.get('training_losses', {})
                all_vram = log_data.get('training_vram', {})
                all_time = log_data.get('training_time', {})
                logger.log("  Loaded training history")
        
    elif train_decision == 'retrain_all':
        logger.log("\nRetraining all models from scratch...")
        
        model_gno, gno_train, gno_val, gno_vram, gno_time = train_gno(
            model_gno, data_mgr, None, X_train, Y_train, X_val, Y_val, 
            cfg.EPOCHS, cfg, logger)
        all_losses['GNO'] = (gno_train, gno_val)
        all_vram['GNO'] = gno_vram
        all_time['GNO'] = gno_time
        
        model_fno, fno_train, fno_val, fno_vram, fno_time = train_fno(
            model_fno, data_mgr, None, X_train, Y_train, X_val, Y_val, 
            cfg.EPOCHS, cfg, logger)
        all_losses['FNO'] = (fno_train, fno_val)
        all_vram['FNO'] = fno_vram
        all_time['FNO'] = fno_time
        
        model_mgn, mgn_train, mgn_val, mgn_vram, mgn_time = train_mgn(
            model_mgn, data_mgr, None, X_train, Y_train, X_val, Y_val, 
            cfg.EPOCHS, cfg, DEVICE, logger)
        all_losses['MGN'] = (mgn_train, mgn_val)
        all_vram['MGN'] = mgn_vram
        all_time['MGN'] = mgn_time
        
        model_deeponet, don_train, don_val, don_vram, don_time = train_deeponet(
            model_deeponet, data_mgr, None, X_train, Y_train, X_val, Y_val, 
            cfg.EPOCHS, cfg, logger)
        all_losses['DeepONet'] = (don_train, don_val)
        all_vram['DeepONet'] = don_vram
        all_time['DeepONet'] = don_time
        
        model_geofno, geo_train, geo_val, geo_vram, geo_time = train_geofno(
            model_geofno, data_mgr, None, X_train, Y_train, X_val, Y_val, 
            cfg.EPOCHS, cfg, logger)
        all_losses['GeoFNO'] = (geo_train, geo_val)
        all_vram['GeoFNO'] = geo_vram
        all_time['GeoFNO'] = geo_time
        
        model_HSD, HSD_train, HSD_val, HSD_vram, HSD_time = train_HSD(
            model_HSD, train_loader, test_loader, 
            cfg.EPOCHS, cfg, DEVICE, logger, host_ops)
        all_losses['HSD'] = (HSD_train, HSD_val)
        all_vram['HSD'] = HSD_vram
        all_time['HSD'] = HSD_time
        
    else:
        logger.log("\nTraining missing models and loading existing ones...")
        for name in existing_models:
            load_model_weights(models_dict[name], existing_models[name], logger)
        
        log_pkl_path = os.path.join(cfg.OUTPUT_DIR, 'experiment_log.pkl')
        if os.path.exists(log_pkl_path):
            with open(log_pkl_path, 'rb') as f:
                log_data = pickle.load(f)
                all_losses = log_data.get('training_losses', {})
                all_vram = log_data.get('training_vram', {})
                all_time = log_data.get('training_time', {})
        
        missing_models = set(model_names) - set(existing_models.keys())
        
        if 'GNO' in missing_models:
            model_gno, gno_train, gno_val, gno_vram, gno_time = train_gno(
                model_gno, data_mgr, None, X_train, Y_train, X_val, Y_val, 
                cfg.EPOCHS, cfg, logger)
            all_losses['GNO'] = (gno_train, gno_val)
            all_vram['GNO'] = gno_vram
            all_time['GNO'] = gno_time
        
        if 'FNO' in missing_models:
            model_fno, fno_train, fno_val, fno_vram, fno_time = train_fno(
                model_fno, data_mgr, None, X_train, Y_train, X_val, Y_val, 
                cfg.EPOCHS, cfg, logger)
            all_losses['FNO'] = (fno_train, fno_val)
            all_vram['FNO'] = fno_vram
            all_time['FNO'] = fno_time
        
        if 'MGN' in missing_models:
            model_mgn, mgn_train, mgn_val, mgn_vram, mgn_time = train_mgn(
                model_mgn, data_mgr, None, X_train, Y_train, X_val, Y_val, 
                cfg.EPOCHS, cfg, DEVICE, logger)
            all_losses['MGN'] = (mgn_train, mgn_val)
            all_vram['MGN'] = mgn_vram
            all_time['MGN'] = mgn_time
        
        if 'DeepONet' in missing_models:
            model_deeponet, don_train, don_val, don_vram, don_time = train_deeponet(
                model_deeponet, data_mgr, None, X_train, Y_train, X_val, Y_val, 
                cfg.EPOCHS, cfg, logger)
            all_losses['DeepONet'] = (don_train, don_val)
            all_vram['DeepONet'] = don_vram
            all_time['DeepONet'] = don_time
        
        if 'GeoFNO' in missing_models:
            model_geofno, geo_train, geo_val, geo_vram, geo_time = train_geofno(
                model_geofno, data_mgr, None, X_train, Y_train, X_val, Y_val, 
                cfg.EPOCHS, cfg, logger)
            all_losses['GeoFNO'] = (geo_train, geo_val)
            all_vram['GeoFNO'] = geo_vram
            all_time['GeoFNO'] = geo_time
        
        if 'HSD' in missing_models:
            model_HSD, HSD_train, HSD_val, HSD_vram, HSD_time = train_HSD(
                model_HSD, train_loader, test_loader, 
                cfg.EPOCHS, cfg, DEVICE, logger, host_ops)
            all_losses['HSD'] = (HSD_train, HSD_val)
            all_vram['HSD'] = HSD_vram
            all_time['HSD'] = HSD_time

    for name, model in models_dict.items():
        save_path = os.path.join(cfg.OUTPUT_DIR, f'model_{name.lower()}.pt')
        torch.save(model.state_dict(), save_path)
    logger.log("All models saved")
    
    logger.section("[Step 9] EVALUATION - VECTOR FIELD MODE")
    
    GT_Vectors = test_dataset.Y_norm
    GT_Flux = test_dataset.gt_flux
    
    logger.log(f"GT Vectors shape: {GT_Vectors.shape}")
    logger.log(f"GT Flux shape (for topology evaluation only): {GT_Flux.shape}")
    
    vector_predictions_dict = {}
    flux_predictions_dict = {}
    mse_dict = {}
    
    def vectors_to_flux_batch(vectors_np):
        flux_list = []
        for i in range(len(vectors_np)):
            flux = mapper.node_vector_to_edge_flux(vectors_np[i])
            flux_list.append(flux)
        return np.array(flux_list)
    
    logger.log("\n--- Baseline Models (output Vectors directly) ---")
    
    model_gno.eval()
    with torch.no_grad():
        gno_in, _ = data_mgr.prepare_gno_batch(X_test, None)
        pred_vec_gno = model_gno(gno_in, data_mgr.pts).permute(0, 2, 1).cpu().numpy()
    vector_predictions_dict['GNO'] = pred_vec_gno
    flux_predictions_dict['GNO'] = vectors_to_flux_batch(pred_vec_gno)
    mse_dict['GNO'] = mean_squared_error(GT_Vectors.flatten(), pred_vec_gno.flatten())
    logger.log(f"  GNO Vector MSE: {mse_dict['GNO']:.6e}")
    
    model_fno.eval()
    with torch.no_grad():
        fno_preds = []
        for i in range(0, X_test.shape[0], 32):
            x_b = X_test[i:i + 32]
            fno_in, _ = data_mgr.prepare_fno_batch(x_b, None)
            pred_grid = model_fno(fno_in)
            pred_mesh = data_mgr.decode_fno_output(pred_grid)
            fno_preds.append(pred_mesh.cpu().numpy())
    pred_vec_fno = np.concatenate(fno_preds, axis=0)
    vector_predictions_dict['FNO'] = pred_vec_fno
    flux_predictions_dict['FNO'] = vectors_to_flux_batch(pred_vec_fno)
    mse_dict['FNO'] = mean_squared_error(GT_Vectors.flatten(), pred_vec_fno.flatten())
    logger.log(f"  FNO Vector MSE: {mse_dict['FNO']:.6e}")
    
    model_mgn.eval()
    with torch.no_grad():
        mgn_preds = []
        for b in range(X_test.shape[0]):
            node_feat = torch.cat([X_test[b:b + 1].T, data_mgr.pts], dim=1)
            pred = model_mgn(node_feat, data_mgr.edge_index, data_mgr.edge_attr)
            mgn_preds.append(pred.cpu().numpy())
    pred_vec_mgn = np.array(mgn_preds)
    vector_predictions_dict['MGN'] = pred_vec_mgn
    flux_predictions_dict['MGN'] = vectors_to_flux_batch(pred_vec_mgn)
    mse_dict['MGN'] = mean_squared_error(GT_Vectors.flatten(), pred_vec_mgn.flatten())
    logger.log(f"  MGN Vector MSE: {mse_dict['MGN']:.6e}")
    
    model_deeponet.eval()
    with torch.no_grad():
        pred_vec_deeponet = model_deeponet(X_test, data_mgr.pts).cpu().numpy()
    vector_predictions_dict['DeepONet'] = pred_vec_deeponet
    flux_predictions_dict['DeepONet'] = vectors_to_flux_batch(pred_vec_deeponet)
    mse_dict['DeepONet'] = mean_squared_error(GT_Vectors.flatten(), pred_vec_deeponet.flatten())
    logger.log(f"  DeepONet Vector MSE: {mse_dict['DeepONet']:.6e}")
    
    model_geofno.eval()
    with torch.no_grad():
        geo_preds = []
        for i in range(0, X_test.shape[0], 32):
            x_b = X_test[i:i + 32]
            x_coords, features = data_mgr.prepare_geofno_batch(x_b)
            pred = model_geofno(x_coords, features)
            geo_preds.append(pred.cpu().numpy())
    pred_vec_geofno = np.concatenate(geo_preds, axis=0)
    vector_predictions_dict['GeoFNO'] = pred_vec_geofno
    flux_predictions_dict['GeoFNO'] = vectors_to_flux_batch(pred_vec_geofno)
    mse_dict['GeoFNO'] = mean_squared_error(GT_Vectors.flatten(), pred_vec_geofno.flatten())
    logger.log(f"  GeoFNO Vector MSE: {mse_dict['GeoFNO']:.6e}")
    
    logger.log("\n--- Spectral Models (output Vectors directly) ---")
    
    model_HSD.eval()
    HSD_vec_preds = []
    with torch.no_grad():
        for c0, c1, c2, c1_tgt, gt_flux, x_spatial, y_vec in test_loader:
            c0 = c0.to(DEVICE)
            c1 = c1.to(DEVICE)
            c2 = c2.to(DEVICE)
            x_spatial = x_spatial.to(DEVICE)
            pred_vec, _, _ = model_HSD(c0, c1, c2, x_spatial)
            HSD_vec_preds.append(pred_vec.cpu().numpy())
    
    pred_vec_HSD = np.concatenate(HSD_vec_preds, axis=0)
    vector_predictions_dict['HSD'] = pred_vec_HSD
    flux_predictions_dict['HSD'] = vectors_to_flux_batch(pred_vec_HSD)
    mse_dict['HSD'] = mean_squared_error(GT_Vectors.flatten(), pred_vec_HSD.flatten())
    logger.log(f"  HSD Vector MSE: {mse_dict['HSD']:.6e}")
    
    logger.log("\n" + "=" * 50)
    logger.log("=== VECTOR FIELD MSE RESULTS (PRIMARY METRIC) ===")
    logger.log("=" * 50)
    for name in model_names:
        if name in mse_dict:
            logger.log(f"  {name:<12} Vector MSE: {mse_dict[name]:.6e}")
    
    logger.section("[Step 10] Topology & Physics Metrics Evaluation")
    
    logger.log("Initializing FluxTopologyEvaluator...")
    logger.log("Note: Flux for topology is computed from predicted vectors (not supervised)")
    
    topo_evaluator = FluxTopologyEvaluator(host_ops, pts, faces, mapper, device=DEVICE)
    
    all_topo_results = evaluate_all_models(
        evaluator=topo_evaluator,
        predictions_dict=flux_predictions_dict,
        Y_test_flux=GT_Flux,
        model_params=params_dict,
        logger=logger
    )
    
    save_results_to_csv(all_topo_results, os.path.join(cfg.OUTPUT_DIR, 'topo_metrics.csv'))
    save_results_to_json(all_topo_results, os.path.join(cfg.OUTPUT_DIR, 'topo_metrics.json'))
    
    with open(os.path.join(cfg.OUTPUT_DIR, 'topo_metrics.pkl'), 'wb') as f:
        pickle.dump(all_topo_results, f)
    logger.log("Topology metrics saved")
    
    logger.section("EVALUATION RESULTS SUMMARY")
    
    logger.log("All MSE values are computed on VECTOR FIELD\n")
    logger.log("Note: HSD models trained WITHOUT flux supervision\n")
    
    logger.log(f"\n{'Model':<12} {'Params':>12} {'Vector_MSE':>14} {'Div_Fid':>10} {'Grad_Fid':>10} {'S_b0':>10} {'VRAM(MB)':>10} {'Time(s)':>10}")
    logger.log("-" * 110)
    
    for name in model_names:
        r = all_topo_results.get(name, {})
        vram_val = all_vram.get(name, 0.0)
        time_val = all_time.get(name, 0.0)
        logger.log(f"{name:<12} {params_dict[name]:>12,} {mse_dict[name]:>14.2e} "
                   f"{r.get('divergence_fidelity', 0):>10.4f} "
                   f"{r.get('gradient_fidelity', 0):>10.4f} "
                   f"{r.get('betti0_score', 0):>10.4f} "
                   f"{vram_val:>10.1f} "
                   f"{time_val:>10.2f}")
    
    best_model = min(mse_dict.items(), key=lambda x: x[1])
    logger.log(f"\nBest Model (Vector MSE): {best_model[0]} with MSE = {best_model[1]:.6e}")
    
    total_training_time = sum(all_time.values())
    logger.log(f"Total Training Time: {total_training_time:.2f}s")

    logger.section("[Step 12] Visualization")
    
    vector_viz_path = os.path.join(cfg.OUTPUT_DIR, "vector_field_comparison.png")
    visualize_vector_field_comparison(
        pts, faces, GT_Vectors, 
        vector_predictions_dict, mse_dict, 
        vector_viz_path, idx=0, logger=logger
    )
    
    flux_viz_path = os.path.join(cfg.OUTPUT_DIR, "flux_comparison.png")
    try:
        visualize_flux_results(pts, faces, host_ops, mapper, GT_Flux,
                               flux_predictions_dict, mse_dict, flux_viz_path, idx=0, logger=logger)
    except Exception as e:
        logger.log(f"  Warning: Flux visualization skipped: {e}")
    
    vel_viz_path = os.path.join(cfg.OUTPUT_DIR, "velocity_magnitude.png")
    try:
        visualize_velocity_magnitude(pts, faces, mapper, GT_Flux,
                                      flux_predictions_dict, mse_dict, vel_viz_path, idx=0, logger=logger)
    except Exception as e:
        logger.log(f"  Warning: Velocity magnitude visualization skipped: {e}")
    
    if all_losses:
        curves_path = os.path.join(cfg.OUTPUT_DIR, "training_curves.png")
        plot_training_curves(all_losses, curves_path, logger=logger)
    
    spectral_decay_path = os.path.join(cfg.OUTPUT_DIR, "spectral_decay_analysis.png")
    try:
        plot_spectral_decay_comparison(topo_evaluator, vector_predictions_dict, GT_Vectors, 
                                       spectral_decay_path, logger=logger)
    except Exception as e:
        logger.log(f"  Warning: Spectral decay plot skipped: {e}")
    
    try:
        full_viz_dir = os.path.join(cfg.OUTPUT_DIR, "full_visualizations")
        generate_all_flux_visualizations(
            pts, faces, host_ops, mapper, GT_Flux, flux_predictions_dict,
            all_losses, all_topo_results,
            output_dir=full_viz_dir, n_samples=20, logger=logger
        )
    except Exception as e:
        logger.log(f"Warning: Extended visualization failed: {e}")
    
    logger.section("[Step 13] Saving Results")
    
    log_data = {
        'mse_results': mse_dict,
        'mse_type': 'vector_field',
        'supervision_mode': 'vector_only_no_flux',
        'params': params_dict,
        'training_losses': all_losses,
        'training_vram': all_vram,
        'training_time': all_time,
        'scaling': {'x_scale': x_scale_global, 'y_scale': y_scale_global},
        'eigen_basis_preprocessing_time_seconds': eigen_basis_elapsed_time,
        'topology_metrics': all_topo_results,
        'geometry_info': topo_evaluator.get_geometry_info(),
        'vector_predictions': vector_predictions_dict,
        'flux_predictions': flux_predictions_dict,
    }
    with open(os.path.join(cfg.OUTPUT_DIR, 'experiment_log.pkl'), 'wb') as f:
        pickle.dump(log_data, f)
    
    logger.section("COMPLETE")
    
    return mse_dict, params_dict, eigen_basis_elapsed_time, all_topo_results, all_vram, all_time


if __name__ == "__main__":
    results = main()