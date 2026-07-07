from algs.base import base_job
from algs.cox_tree import cox_tree_job
from algs.ddgroup import ddgroup_job, c_ind_ddgroup_job, pl_ddgroup_job, no_exp_ddgroup_job
from algs.prim import prim_job
from algs.random import random_job
from algs.survival_tree import survival_tree_job
from algs.ost import ost_job
from sksurv.datasets import load_veterans_lung_cancer, load_gbsg2, load_aids, load_whas500
import numpy as np


METHOD_DICT = {
    'base': base_job,
    'ddgroup': ddgroup_job,
    'c_ind_ddgroup': c_ind_ddgroup_job,
    'pl_ddgroup': pl_ddgroup_job,
    'no_exp_ddgroup': no_exp_ddgroup_job,
    'cox_tree': cox_tree_job,
    'prim': prim_job,
    'random': random_job,
    'survival_tree': survival_tree_job,
    'ost': ost_job,
}

METRIC_COLS = [
    'train_epe', 'train_c_ind', 'train_pll', 'train_size', 'train_c_p', 'train_epe_p',
    'test_epe', 'test_c_ind', 'test_pll', 'test_size', 'test_c_p', 'test_epe_p', 'f1',
    'train_precision', 'train_recall', 'train_f1', 'train_iou',
    'test_precision', 'test_recall', 'test_f1', 'test_iou'
]

SKSURV_DATASETS = {
    "veterans_lung_cancer": load_veterans_lung_cancer,
    "gbsg2":   load_gbsg2,
    "aids":    load_aids, 
    "whas500": load_whas500
}

SKSURV_COLS = {
    "veterans_lung_cancer": ['Age_in_years', 'Karnofsky_score', 'Months_from_Diagnosis'],
    "gbsg2": ['age', 'estrec', 'pnodes', 'progrec', 'tsize'],
    "aids": ['age', 'cd4', 'karnof', 'priorzdv'],
    "whas500": ['age', 'bmi', 'diasbp', 'hr', 'sysbp', 'los']
}

METABRIC_COLS = ['MKI67',
                'EGFR',
                'PGR',
                'ERBB2',
                'age at diagnosis']




NASA_TURBOFAN_COLS = ['op_setting_1', 
                      'P2', 'P15', 'P30', 'Ps30',  # Pressure sensors
                      'Nf', 'Nc', 'NRf', 'NRc'    # Speed sensors
]

RANDOM_EPE = np.log(2) # EPE of random model

COL_NAMES = SKSURV_COLS.copy()
COL_NAMES['metabric'] = METABRIC_COLS.copy()
COL_NAMES['nasa'] = NASA_TURBOFAN_COLS.copy()
COL_NAMES['nonlinear'] = [f'x{i}' for i in range(10)]  # supports up to d=10



###########################
# Method hyperparameters
###########################

core_sizes = [0.05, 0.1]
rejection_thresholds = np.linspace(0.00, 0.20, 20)

no_exp_core_sizes = np.linspace(0.01, 1., 100).tolist()

ct_min_leaf_sizes = [5 * i for i in range(1, 11)]
ct_max_depth = 10
max_splits_per_feature = 100

st_min_leaf_sizes = [5 * i for i in range(1, 11)]
st_max_depth = 10

ost_regularization_values = np.linspace(0.01, 1., 100).tolist()
ost_depth_budget = 5 # Used 6 for previous runs
ost_min_captured_points = 7 # Used 10 for previous runs

peeling_fracs = np.linspace(0.01, 0.25, 25).tolist()
min_support_sizes = [0.01, 0.02, 0.04, 0.08]

random_seeds = [i+100 for i in range(100)]

METHOD_HYPERS = {
    'base': [{}],
    'ddgroup': [{
        'core_sizes': core_sizes,
        'rejection_thresholds': rejection_thresholds
    }],
    'c_ind_ddgroup': [{
        'core_sizes': core_sizes,
        'rejection_thresholds': rejection_thresholds
    }],
    'pl_ddgroup': [{
        'core_sizes': core_sizes,
        'rejection_thresholds': rejection_thresholds
    }],
    'no_exp_ddgroup': [{
        'core_size': core_size,
    } for core_size in no_exp_core_sizes],
    'cox_tree': [{
        'max_depth': ct_max_depth,
        'min_samples_leaf': k,
        'max_splits_per_feature': max_splits_per_feature,
        'num_subgroups': 1
    } for k in ct_min_leaf_sizes],
    'prim': [{
        'peeling_frac': p,
        'min_support_size': s,
        'num_subgroups': 1
    } for p in peeling_fracs for s in min_support_sizes],
    'random': [{
        'seeds': random_seeds,
        'num_subgroups': 1
    }],
    'survival_tree': [{
        'max_depth': st_max_depth,
        'min_samples_leaf': k,
        'num_subgroups': 1
    } for k in st_min_leaf_sizes],
    'ost': [{
        'regularization': r,
        'minimum_captured_points': ost_min_captured_points,
        'depth_budget': ost_depth_budget,
        'num_subgroups': 1
    } for r in ost_regularization_values],
}