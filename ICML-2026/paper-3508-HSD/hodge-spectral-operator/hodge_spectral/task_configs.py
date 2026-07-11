"""
Per-task hyperparameter configurations.
Each task has independent settings to prevent cross-contamination.
"""


TASK_CONFIGS = {
    'ellipsoid_0to1': {
        'name': 'Ellipsoid Darcy Flow',
        'topology': 'genus-0 (sphere-like)',
        'input_form': 0,
        'output_form': 1,
        'data_path': './data/unified_dataset.pkl',

        # Spectral
        'k_eigens': 64,
        'n_heat_scales': 6,

        # Model
        'hidden_dims': (384, 256),
        'res_hidden': 192,

        # Training
        'epochs': 150,
        'lr': 3e-3,
        'weight_decay': 1e-5,
        'batch_size': 64,
        'patience': 30,
        'res_reg': 1e-4,
        'base_loss_weight': 0.1,
    },

    'torus_0to0': {
        'name': 'Torus Advection-Diffusion',
        'topology': 'genus-1 (torus)',
        'input_form': 0,
        'output_form': 0,
        'data_path': './data_torus/torus_advdiff_dataset.pkl',

        # Spectral
        'k_eigens': 128,
        'n_heat_scales': 6,

        # Model — bigger spatial branch for high-Pe advection
        'hidden_dims': (384, 256),
        'res_hidden': 512,  # large spatial branch needed: spectral only captures 77%

        # Training — needs more epochs for time-evolution task
        'epochs': 200,
        'lr': 3e-3,
        'weight_decay': 1e-5,
        'batch_size': 64,
        'patience': 50,
        'res_reg': 0.0,        # no residual reg (let spatial branch learn freely)
        'base_loss_weight': 0.05,  # weaker base constraint
    },
}


# DeepONet baseline configs (independent)
BASELINE_CONFIGS = {
    'deeponet': {
        'branch_layers': [64, 64, 64],
        'trunk_layers': [64, 64, 64],
        'basis_dim': 74,
        'lr': 1e-3,
        'epochs': 100,
    },
    'fno': {
        'modes': (4, 4, 4),
        'hidden': 21,
        'layers': 2,
        'grid_res': 24,
        'lr': 1e-3,
        'epochs': 100,
    },
}
