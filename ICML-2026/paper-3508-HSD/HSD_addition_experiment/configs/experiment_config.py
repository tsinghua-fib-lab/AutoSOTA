"""
Experiment configurations for ICML rebuttal additional experiments.
Each config is independent and self-contained.
"""
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================
# Dataset configs
# ==============================
DATASETS = {
    'ellipsoid_aero': {
        'data_dir': './data/ellipsoid_aero',
        'generator': 'hodge_spectral.data.generate_ellipsoid_aero',
        'generator_fn': 'generate_ellipsoid_aero',
        'n_samples': 3000,
        'topology': 'genus-0',
        'task': '0to1',
        'physics': 'Poisson stream + global coupling',
    },
    'torus_helmholtz': {
        'data_dir': './data/torus_helmholtz',
        'generator_script': '../hodge-spectral-operator/examples/example_torus_helmholtz.py',
        'n_samples': 3000,
        'topology': 'genus-1',
        'task': '0to1',
        'physics': 'Helmholtz(κ=0.5) vortex flow',
    },
}

# ==============================
# HSD model configs
# ==============================
HSD_CONFIG = {
    'k': 64,
    'hidden_dims': (256, 192),
    'res_hidden': 128,
    'dropout': 0.05,
    'res_dropout': 0.1,
    'lr': 3e-3,
    'epochs': 120,
    'patience': 25,
    'batch_size': 64,
    'weight_decay': 1e-5,
}

# ==============================
# Baseline configs
# ==============================
BASELINES = {
    'FNO': {
        'in_channels': 2, 'out_channels': 3,
        'hidden_channels': 21, 'n_modes': (4, 4, 4), 'n_layers': 2,
        'grid_res': 24, 'lr': 1e-3, 'epochs': 100, 'batch_size': 32,
    },
    'DeepONet': {
        'branch_layers': [64, 64, 64], 'trunk_layers': [64, 64, 64],
        'basis_dim': 74, 'lr': 1e-3, 'epochs': 100, 'batch_size': 64,
    },
    'GeoFNO': {
        'modes': 6, 'width': 12, 'layers': 4, 'grid_res': 24,
        'lr': 1e-3, 'epochs': 100, 'batch_size': 32,
    },
    'MGN': {
        'hidden_dim': 58, 'num_layers': 10,
        'lr': 1e-3, 'epochs': 100, 'batch_size': 64,
    },
    # Recent baselines (2022-2024)
    'GNOT': {  # General Neural Operator Transformer (2023)
        'n_heads': 4, 'hidden': 128, 'layers': 4,
        'lr': 1e-3, 'epochs': 100, 'batch_size': 64,
    },
    'ONO': {  # Operator learning with Neural Operators (2023, orthogonal attention)
        'hidden': 128, 'layers': 4, 'heads': 4,
        'lr': 1e-3, 'epochs': 100, 'batch_size': 64,
    },
    'HAMLET': {  # Hierarchical Attention MeshLEarning Transformer (2024)
        'hidden': 96, 'layers': 3, 'heads': 4,
        'lr': 1e-3, 'epochs': 100, 'batch_size': 64,
    },
}

# ==============================
# Ablation configs
# ==============================
ABLATIONS = {
    # Ablation 1: Input modality
    'input_modality': {
        'modes': ['mesh', 'pointcloud', 'graph'],
        'description': 'Cross-input consistency: strong topology (mesh) vs weak (pointcloud/graph)',
    },
    # Ablation 2: Layer type
    'layer_type': {
        'variants': {
            'plain_mlp': {'layer': 'plain', 'description': 'Standard Linear+GELU+Dropout'},
            'pseudo_spectral_bilinear': {'layer': 'bilinear', 'description': 'Our gMLP with de Rham bilinear interaction'},
        },
    },
    # Ablation 3: Additional baselines (see BASELINES above)
    'extended_baselines': {
        'models': ['FNO', 'DeepONet', 'GeoFNO', 'MGN', 'GNOT', 'ONO', 'HAMLET'],
    },
    # Ablation 4: Spatial encoding
    'spatial_encoding': {
        'variants': {
            'whitney_kde': {'description': 'Whitney form interpolation + KDE smoothing (our method, paper name)'},
            'raw_euclidean': {'description': 'Direct 3D coordinate embedding without smoothing'},
        },
    },
}

# ==============================
# Metrics to compute
# ==============================
METRICS = {
    'operator': ['relative_l2', 'mse', 'riemannian_ip_fidelity'],
    'physics': ['divergence_fidelity', 'curl_mse', 'vorticity_fidelity',
                'energy_fidelity', 'enstrophy_fidelity'],
    'spectral': ['spectral_fidelity', 'gradient_fidelity'],
    'topology': ['betti0_score', 'level_set_iou', 'vortex_count_accuracy'],
}

# ==============================
# Visualization configs
# ==============================
VIZ_CONFIG = {
    'figures_dir': './output/figures',
    'dpi': 300,
    'format': ['png', 'pdf'],
    'n_viz_samples': 3,
    'plots': [
        'training_curves',       # loss vs epoch for all methods
        'vector_field_comparison', # GT vs HSD vs FNO vs DeepONet on mesh
        'topological_contours',   # level-set contours comparison
    ],
}
