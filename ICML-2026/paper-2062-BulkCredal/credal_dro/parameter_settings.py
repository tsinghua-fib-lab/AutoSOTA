"""
Experiment settings.
"""
# ---------------------------------------------------------------------
# Synthetic newsvendor settings
# ---------------------------------------------------------------------

# Number of independent training replications per (algorithm, epsilon) config.
# Paper setting: 100 replications.
NEWSVENDOR_NUM_REPLICATIONS: int = 100

# Contamination level in the testing dataset
# Paper uses 0.0, 0.1, 0.2
NEWSVENDOR_CONTAMINATION_LEVEL: float = 0.2

# ---------------------------------------------------------------------
# California housing settings
# ---------------------------------------------------------------------

# Gap ratio g used in the East/West geographic split.
# Paper setting: g = 0.30 (East train+select+val = 0.50, gap = 0.30, West test = 0.20).
CALIFORNIA_HOUSING_GAP_RATIO: float = 0.30

# Number of independent replications (seeds) per hyperparameter setting.
# Paper setting: 100.
CALIFORNIA_HOUSING_NUM_REPLICATIONS: int = 100

# ---------------------------------------------------------------------
# CivilComments experiment settings
# ---------------------------------------------------------------------

# Number of independent training replications per (dataset, algorithm, epsilon, gamma) config.
# Paper setting (CivilComments): 20 replications.
RW_NUM_REPLICATIONS: int = 20

# Hashed n-gram feature dimension for CivilComments.
# Paper main setting: 4096.
# For the 8192-feature ablation in the paper appendix, set this to 8192.
RW_TEXT_N_FEATURES: int = 4096