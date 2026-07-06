from pathlib import Path
import random

import numpy as np
import pandas as pd

# Reproducibility
SEED = 2025
random.seed(SEED)
np.random.seed(SEED)


# Add project paths
root = Path(__file__).resolve().parent


# Import CDE methods
from experiments_gaussian_scm import PC_CDE, ldecc_CDE, MBbyMB_CDE, CMB_CDE, locpc_CDE


# Optional dependencies (pyciphod internals + baselines).
# Do not raise on import; instead set a flag and raise with instruction only when trying to run the script.
REPRO_AVAILABLE = True
REPRO_ERROR = None
try:
    from reproducibility.clear2026.dags_generator import random_DAG_identifiable_CDE, random_DAG_nonidentifiable_CDE
    from reproducibility.clear2026.baselines.Gupta_codes.ldecc import LDECCAlgorithm
    from reproducibility.clear2026.baselines.pyCausalFS.LSL.MBs.CMB.CMB import CMB
    from reproducibility.clear2026.baselines.pyCausalFS.LSL.MBs.MBbyMB import MBbyMB
except Exception as e:
    REPRO_AVAILABLE = False
    REPRO_ERROR = e


# Load dataset
df = pd.read_csv('../../datasets/SNDS_agreg.csv', sep=';')

# Filter out sex = 9
df_filtered = df[df['sexe'] == 9]

# Pivot to get departments vs. pathologies
data = df_filtered.pivot_table(
    index='dept',
    columns='patho_niv1',
    values='prev',
    fill_value=0
).reset_index()

# Drop non-pathology column and 'dept'
drop_cols = ["dept",
             'Pas de pathologie repérée, traitement, maternité, hospitalisation ou traitement antalgique ou anti-inflammatoire']
data = data.drop(columns=drop_cols)

# Convert to numeric
data = data.apply(pd.to_numeric, errors='coerce')

# Dataset overview
print("Dataset shape:", data.shape)
print("\nColumn types:")
print(data.dtypes)
print("\nBasic statistics:")
print(data.describe())

# Optional transformation
data_arcsin = np.arcsin(np.sqrt(data / 100))

# Define target and predictor
x = 'Insuffisance rénale chronique terminale'
y = 'Diabète'

if not REPRO_AVAILABLE:
    # Do not raise: run only core package methods so the script remains usable without extras
    print("Optional reproducibility dependencies are missing. Running only core pyciphod methods (no baselines).")
    methods = ['locpc', 'pc']
else:
    # All dependencies available: run baselines as well
    methods = ['locpc', 'ldecc', 'pc', 'CMB', 'MBbyMB']  # all baselines

# Run CDE methods
# results = {
#     "LocPC": locpc_CDE(data, x, y),
#     "PC": PC_CDE(data, x, y),
#     "LDECC": ldecc_CDE(data, x, y),
#     "CMB": CMB_CDE(data, x, y),
#     "MBbyMB": MBbyMB_CDE(data, x, y),
# }

for method in methods:
    print(f"Running method : {method}")
    res = {
        'locpc': locpc_CDE,
        'ldecc': ldecc_CDE,
        'pc': PC_CDE,
        'CMB': CMB_CDE,
        'MBbyMB': MBbyMB_CDE
    }[method](data, x, y)

    # Display results
    print("\nCDE estimation results:")
    print(res)