"""
Fast HSD-only training and evaluation for paper 3508 reproduction.
Skips slow GNO/MGN/DeepONet/GeoFNO baselines.
"""
import os
import pickle
import warnings
import time
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore")
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Headless pyvista mock
import pyvista as pv
pv.OFF_SCREEN = True
class _HeadlessMock:
    def __init__(self, *args, **kwargs): pass
    def __getattr__(self, name): return _headless_sentinel
    def __call__(self, *args, **kwargs): return _headless_sentinel
    def __bool__(self): return True
_headless_sentinel = _HeadlessMock()
pv.Plotter = _HeadlessMock
pv.start_xvfb = lambda *a, **kw: None

from config import Config, DEVICE
from utils import Logger, count_parameters
from spectral_operators import HighOrderSpectralOperators
from dataset import FluxFieldDataset, DataManager, VectorFluxMapper
from models import HSDSpectralVectorFNO, SpectralVectorOperator
from training import train_HSD
from topo_metrics import FluxTopologyEvaluator, evaluate_all_models

# Monkey-patch GNO/FNO evaluation in topo_metrics to handle HSD only
def evaluate_hsd_only(evaluator, predictions_dict, Y_test_flux, model_params, logger=None):
    """Evaluate only HSD model with all topology metrics."""
    return evaluate_all_models(evaluator, predictions_dict, Y_test_flux, model_params, logger)


def main():
    cfg = Config()

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    log_path = os.path.join(cfg.OUTPUT_DIR, cfg.LOG_FILE)
    logger = Logger(log_path)

    logger.section("HSD-ONLY TRAINING - PAPER 3508 REPRODUCTION")
    logger.log(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        logger.log(f"GPU: {torch.cuda.get_device_name()}")
        logger.log(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Step 1: Load data
    logger.section("[Step 1] Loading Dataset")
    data_path = os.path.join(cfg.DATA_DIR, cfg.PICKLE_FILE)
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    X_data = data['X_data'].astype(np.float32)
    Y_data = data['Y_data'].astype(np.float32)
    pts = data['points'].astype(np.float32)
    faces = data['faces'].astype(np.int32)
    normals = data['normals'].astype(np.float32)

    n_nodes = len(pts)
    n_samples = len(X_data)
    logger.log(f"Loaded {n_samples} samples, mesh: {n_nodes} nodes, {len(faces)} faces")

    # Step 2: Vector-Flux Mapper
    mapper = VectorFluxMapper(pts, faces)

    # Step 3: Split data (68/12/20 train/val/test)
    indices = np.arange(n_samples)
    idx_train_val, idx_test = train_test_split(indices, test_size=cfg.TEST_RATIO, random_state=42)
    idx_train, idx_val = train_test_split(idx_train_val, test_size=cfg.VAL_RATIO, random_state=42)

    logger.log(f"Train: {len(idx_train)}, Val: {len(idx_val)}, Test: {len(idx_test)}")

    x_scale_global = np.max(np.abs(X_data[idx_train_val])) + 1e-9
    y_scale_global = np.max(np.abs(Y_data[idx_train_val])) + 1e-9

    X_test = torch.from_numpy(X_data[idx_test] / x_scale_global).float().to(DEVICE)
    Y_test = torch.from_numpy(Y_data[idx_test] / y_scale_global).float().to(DEVICE)

    # Step 4: Build Spectral Operators
    logger.section("[Step 4] Building Spectral Operators")
    t0 = time.time()
    host_ops = HighOrderSpectralOperators(
        pts, faces,
        k_list=(cfg.K_EIGENS, cfg.K_EIGENS, cfg.K_EIGENS),
        logger=logger
    )
    eigen_time = time.time() - t0
    logger.log(f"Eigenbasis time: {eigen_time:.2f}s")

    Phi0_torch = torch.from_numpy(host_ops.Phi0[:, :cfg.K_EIGENS].astype(np.float32)).to(DEVICE)
    Phi1_torch = torch.from_numpy(host_ops.Phi1[:, :cfg.K_EIGENS].astype(np.float32)).to(DEVICE)

    # Step 5: Prepare datasets
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

    # Step 6: Data Manager
    data_mgr = DataManager(n_nodes, pts, DEVICE, faces=faces, grid_res=cfg.FNO_GRID_RES)

    # Step 7: Initialize HSD Model
    logger.section("[Step 7] Initializing HSD Model")
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

    n_params = count_parameters(model_HSD)
    logger.log(f"HSD Parameters: {n_params:,}")

    # Step 8: Train HSD
    logger.section("[Step 8] Training HSD")
    model_HSD, HSD_train_losses, HSD_val_losses, HSD_vram, HSD_time = train_HSD(
        model_HSD, train_loader, test_loader,
        cfg.EPOCHS, cfg, DEVICE, logger, host_ops
    )

    # Save model
    torch.save(model_HSD.state_dict(), os.path.join(cfg.OUTPUT_DIR, 'model_hsd.pt'))
    logger.log("HSD model saved")

    # Step 9: Evaluate HSD
    logger.section("[Step 9] Evaluating HSD")

    GT_Vectors = test_dataset.Y_norm
    GT_Flux = test_dataset.gt_flux

    # Predict
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

    # Compute vector MSE (on normalized data)
    mse_HSD = mean_squared_error(GT_Vectors.flatten(), pred_vec_HSD.flatten())
    logger.log(f"HSD Vector MSE (normalized): {mse_HSD:.6e}")

    # Convert to flux for topology evaluation
    flux_preds_HSD = []
    for i in range(len(pred_vec_HSD)):
        flux = mapper.node_vector_to_edge_flux(pred_vec_HSD[i])
        flux_preds_HSD.append(flux)
    flux_preds_HSD = np.array(flux_preds_HSD)

    # Step 10: Topology & Physics Metrics
    logger.section("[Step 10] Topology & Physics Metrics Evaluation")

    topo_evaluator = FluxTopologyEvaluator(host_ops, pts, faces, mapper, device=DEVICE)

    all_topo_results = evaluate_all_models(
        evaluator=topo_evaluator,
        predictions_dict={'HSD': flux_preds_HSD},
        Y_test_flux=GT_Flux,
        model_params={'HSD': n_params},
        logger=logger
    )

    # Print results
    logger.section("=== RESULTS SUMMARY ===")
    r = all_topo_results.get('HSD', {})
    logger.log(f"\n{'='*60}")
    logger.log(f"PAPER 3508 REPRODUCTION RESULTS - External Aerodynamics")
    logger.log(f"{'='*60}")
    logger.log(f"Model: HSD ({n_params:,} params)")
    logger.log(f"Mesh nodes: {n_nodes}")
    logger.log(f"Test samples: {len(idx_test)}")
    logger.log(f"Spectral truncation (k): {cfg.K_EIGENS}")
    logger.log(f"Training epochs: {cfg.EPOCHS}")
    logger.log(f"Batch size: {cfg.BATCH_SIZE}")
    logger.log(f"Learning rate: {cfg.LR_OURS}")
    logger.log(f"Weight decay: {cfg.WEIGHT_DECAY}")
    logger.log(f"{'='*60}")
    logger.log(f"\nVector Field MSE (normalized): {mse_HSD:.6e}")
    logger.log(f"\nTopology Metrics:")
    logger.log(f"  Gradient Fidelity:    {r.get('gradient_fidelity', 'N/A'):.6f}")
    logger.log(f"  Divergence Fidelity:  {r.get('divergence_fidelity', 'N/A'):.6f}")
    logger.log(f"  Enstrophy Fidelity:   {r.get('enstrophy_fidelity', 'N/A'):.6f}")
    logger.log(f"  Energy Fidelity:      {r.get('energy_fidelity', 'N/A'):.6f}")
    logger.log(f"  Spectral Fidelity:    {r.get('spectral_fidelity', 'N/A'):.6f}")
    logger.log(f"  Betti-0 Score:        {r.get('betti0_score', 'N/A'):.6f}")
    logger.log(f"  IoU:                   {r.get('iou', 'N/A')}")
    logger.log(f"  Curl MSE:              {r.get('curl_mse', 'N/A')}")
    logger.log(f"\nTraining Time: {HSD_time:.2f}s, VRAM: {HSD_vram:.1f} MB")
    logger.log(f"Eigenbasis Time: {eigen_time:.2f}s")

    # Save detailed results
    results = {
        'HSD_params': n_params,
        'mse_normalized': float(mse_HSD),
        'topology_metrics': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                            for k, v in r.items()},
        'training_time': HSD_time,
        'eigenbasis_time': eigen_time,
        'training_vram_mb': HSD_vram,
        'config': {
            'n_samples': n_samples,
            'n_nodes': n_nodes,
            'test_samples': len(idx_test),
            'k_eigens': cfg.K_EIGENS,
            'epochs': cfg.EPOCHS,
            'batch_size': cfg.BATCH_SIZE,
            'lr': cfg.LR_OURS,
            'weight_decay': cfg.WEIGHT_DECAY,
            'spectral_hidden_dims': cfg.SPECTRAL_HIDDEN_DIMS,
            'hsd_fno_modes': cfg.HSD_FNO_MODES,
            'hsd_fno_hidden': cfg.HSD_FNO_HIDDEN,
            'hsd_fno_layers': cfg.HSD_FNO_LAYERS,
        },
        'training_losses': {
            'train': [float(x) for x in HSD_train_losses],
            'val': [float(x) for x in HSD_val_losses],
        }
    }

    import json
    with open(os.path.join(cfg.OUTPUT_DIR, 'hsd_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    logger.log(f"\nResults saved to {cfg.OUTPUT_DIR}/hsd_results.json")

    logger.section("REPRODUCTION COMPLETE")
    return results


if __name__ == "__main__":
    results = main()
