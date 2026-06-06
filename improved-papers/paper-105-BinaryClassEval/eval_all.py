import sys
import types

sys.path.insert(0, '/repo')

# Pre-populate sys.modules to break circular deps between stats and core
sys.modules['stats'] = types.ModuleType('stats')
sys.modules['core'] = types.ModuleType('core')

import importlib.util

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

nb_mod = load_module('core.net_benefit', '/repo/core/net_benefit.py')
prev_mod = load_module('stats.prevalence', '/repo/stats/prevalence.py')
iso_mod = load_module('stats.isotonic', '/repo/stats/isotonic.py')

import numpy as np
import warnings
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')

from etl.eicu import EICU

net_benefit_for_prevalences = nb_mod.net_benefit_for_prevalences
default_prevalence_grid = prev_mod.default_prevalence_grid
log_odds_grid = prev_mod.log_odds_grid
get_calibration_model = iso_mod.get_calibration_model

# Load data
model = EICU(demo=True)
data = model.data

apache_col = 'predicted_hospital_mortality'
target_col = 'mortality_binary'

y_pred = data[apache_col].values
y_true = data[target_col].values

prev_grid = default_prevalence_grid()
logodds_grid_arr = log_odds_grid(prev_grid)

# Ethnicity metrics (maxlogodds=0.1 range defined by subgroup prevalences)
eth_subgroups = ['Caucasian', 'African American']
eth_prevs = []
eth_nb_curves = []
eth_cal_curves = []

for eth in eth_subgroups:
    mask = data['ethnicity'] == eth
    p = y_pred[mask.values]
    l = y_true[mask.values]
    prev = np.mean(l)
    eth_prevs.append(prev)
    nb = net_benefit_for_prevalences(l, p, prevalence_grid=prev_grid, cost_ratio=0.5, train_prevalence=prev)
    eth_nb_curves.append(nb)
    cal_model = get_calibration_model(l, p)
    cal_p = cal_model.transform(p)
    cal_nb = net_benefit_for_prevalences(l, cal_p, prevalence_grid=prev_grid, cost_ratio=0.5, train_prevalence=prev)
    eth_cal_curves.append(cal_nb)

min_prev_eth = min(eth_prevs)
max_prev_eth = max(eth_prevs)
min_logodds_eth = np.log(min_prev_eth / (1 - min_prev_eth))
max_logodds_eth = np.log(max_prev_eth / (1 - max_prev_eth))

for i, eth in enumerate(eth_subgroups):
    mask_range = (logodds_grid_arr >= min_logodds_eth) & (logodds_grid_arr <= max_logodds_eth)
    overall = np.mean(eth_nb_curves[i][mask_range])
    disc_only = np.mean(eth_cal_curves[i][mask_range])
    calib_only = overall / disc_only if disc_only > 0 else 0.0
    eth_key = eth.replace(' ', '_').lower()
    print(f'dca_overall_{eth_key}: {overall:.3f}')
    print(f'dca_calibration_only_{eth_key}: {calib_only:.3f}')
    print(f'dca_discrimination_only_{eth_key}: {disc_only:.3f}')

# Gender metrics
genders = ['Male', 'Female']
g_prevs = []
g_nb_curves = []

for g in genders:
    mask = data['gender'] == g
    p = y_pred[mask.values]
    l = y_true[mask.values]
    prev = np.mean(l)
    g_prevs.append(prev)
    nb = net_benefit_for_prevalences(l, p, prevalence_grid=prev_grid, cost_ratio=0.5, train_prevalence=prev)
    g_nb_curves.append(nb)
    brier = np.mean((p - l)**2)
    nll = -np.mean(l * np.log(p + 1e-10) + (1 - l) * np.log(1 - p + 1e-10))
    g_key = g.lower()
    print(f'brier_{g_key}: {brier:.3f}')
    print(f'nll_{g_key}: {nll:.3f}')

min_prev_g = min(g_prevs)
max_prev_g = max(g_prevs)
min_logodds_g = np.log(min_prev_g / (1 - min_prev_g))
max_logodds_g = np.log(max_prev_g / (1 - max_prev_g))

for i, g in enumerate(genders):
    mask_range = (logodds_grid_arr >= min_logodds_g) & (logodds_grid_arr <= max_logodds_g)
    dca = np.mean(g_nb_curves[i][mask_range])
    g_key = g.lower()
    print(f'dca_{g_key}: {dca:.3f}')
