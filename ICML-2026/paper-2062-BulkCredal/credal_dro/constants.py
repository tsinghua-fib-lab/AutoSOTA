"""Constants for e.g. number of observations or samples"""

# constants for DGPs
CONTAMINATION_LEVEL = 0.05  # ratio for contamination dataset
CONTAMINATION_LEVEL_SET = [0.1, 0.2]  # ratio for contamination dataset
PRIOR_SIGMA_SCALE = 0.02  # scale for prior covariance matrix
CONTAMINATION_TYPE ="shift"  # "scale" or "shift" or "adversarial" 
DGP_NORMAL_KNOWN_VARIANCE_STD = 5.0 # standard deviation of 1D normal with known variance
IN_SAMPLE_TIME_WINDOW = 52  # number of weeks in training period for portfolio problem
OUT_OF_SAMPLE_TIME_WINDOW = 12  # number of weeks in out-of-sample period for portfolio

# experiment constants
NUM_OBSERVATIONS = 20  # in-sample 'training' observations
NUM_POSTERIOR_SAMPLES = 100  # theta samples from posterior
NUM_TEST_OBSERVATIONS = 50  # out-of-sample 'test' observations
NUM_LIKELIHOOD_SAMPLES = 100  # xi samples from likelihood
NUM_REPLICATIONS = 200  # num times to repeat for loop
BAS_NUM_REPLICATIONS = 500
NUM_CERTIFY = 200 # num certufying points for discretisation of KDRO problem constraints
MAX_PARAMS_OOM = 1000   # if the number of params of a cvxpy exceeds this number, we might go out-of-memory
BAS_TOTAL_MODEL_SAMPLES = [25, 100, 900]

# experiment constants for RoBAS
ROBAS_NEWSVENDOR_NUM_REPLICATIONS = 100
NPL_ETA = 0.1

# synthetic portfolio experiment (LV/KL/BDRO) -------------------------------
PORTFOLIO_SYN_DIM = 5
PORTFOLIO_SYN_NUM_OBSERVATIONS = 2000
PORTFOLIO_SYN_NUM_OBSERVATIONS_SET = [2500]
#PORTFOLIO_SYN_NUM_OBSERVATIONS_SET = [500,1000,2000,3000,4000,5000,6000,7000,8000,9000,10000]
PORTFOLIO_SYN_NUM_TEST_OBSERVATIONS = 500
PORTFOLIO_SYN_NUM_REPLICATIONS = 200
# --------------------------------------------------------------------------
#LV_USE_PC_BULK_GEOMETRY = True
LV_USE_PC_BULK_GEOMETRY = False

# synthetic newsvendor experiment (LV/KL/BDRO) -------------------------------
NEWSVENDOR_NUM_OBSERVATIONS = 2000
NEWSVENDOR_NUM_TEST_OBSERVATIONS = 500
#NEWSVENDOR_NUM_REPLICATIONS = 100
#NEWSVENDOR_CONTAMINATION_LEVEL = 0.2  # ratio for contamination dataset
NEWSVENDOR_CONTAMINATION_TYPE ="spike"  
NESWVENDOR_VARY_RHO = False
# --------------------------------------------------------------------------
# Student-t settings (fixed df)
STUDENT_T_DF = 3.0

# Gibbs defaults for student_t_niw
STUDENT_T_NIW_GIBBS_BURN_IN = 200
STUDENT_T_NIW_GIBBS_THIN = 1
STUDENT_T_NIW_GIBBS_JITTER = 1e-6


# --- California Housing LV-BAS experiment defaults ---
# The California Housing dataset has a fixed number of covariates (d=8). We work
# in xi=(x,y) in R^{d+1}, but only standardise X.
CALIFORNIA_HOUSING_SPLIT_SEED = 0
CALIFORNIA_HOUSING_TRAIN_FRAC = 0.5
CALIFORNIA_HOUSING_SELECT_FRAC = 0.2
CALIFORNIA_HOUSING_VAL_FRAC = 0.1
CALIFORNIA_HOUSING_D = 8
CALIFORNIA_HOUSING_SPLIT_FRACS = (0.40, 0.10, 0.00, 0.50)  # train/cal/val/test
CALIFORNIA_HOUSING_EPSILON_SET = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.10, 0.125, 0.15, 0.175, 0.20, 0.5, 0.75, 0.90,0.99,0.999,1.0]
CALIFORNIA_HOUSING_WASS_SET = [0.0, 0.20, 0.40, 0.60, 0.80, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0,7.0, 8.0, 9.0, 10.0, 40.0, 80.0, 100.0]
CALIFORNIA_HOUSING_CHI2_SET = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05,0.075, 0.10, 0.125, 0.15, 0.175, 0.20, 0.50, 0.75, 1.0, 2.0, 5.0, 10.0]
CALIFORNIA_HOUSING_KL_SET = [0.0, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.50, 0.75, 1.0, 2.0, 5.0, 10.0]
CALIFORNIA_HOUSING_OR_WDRO_ETA = 0.1 # outlier fraction for OR-WDRO baseline (set 0 to recover standard WDRO)
CALIFORNIA_HOUSING_OR_WDRO_DUAL_NORM = 2  # 2 => L2 transportation cost (matches ||w||_2-style baselines)
CALIFORNIA_HOUSING_RIDGE_LAMBDA_SET = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 0.5, 1.0, 5.0, 10.0, 50, 100.0, 500.0, 1000.0]
# CALIFORNIA_HOUSING_EPSILON_SET = [0.02, 0.05,]
# CALIFORNIA_HOUSING_WASS_SET = [0.1,0.5]
# CALIFORNIA_HOUSING_RIDGE_LAMBDA_SET = [5.0, 10.0]
CALIFORNIA_HOUSING_EPS_TRUE_SET = [0.0]
CALIFORNIA_HOUSING_DELTA_SHIFT_SET = [0.0]
CALIFORNIA_HOUSING_GAMMA_SET = [0.075]
CALIFORNIA_HOUSING_DKW_DELTA = 0.05
#CALIFORNIA_HOUSING_GAP_RATIO = 0.30  # gap between train/select/val and test
CALIFORNIA_HOUSING_LV_MC_SAMPLES = 10000
#CALIFORNIA_HOUSING_RIDGE_ALPHA = 1.0
CALIFORNIA_HOUSING_STANDARDISE_Y = False
#CALIFORNIA_HOUSING_NUM_REPLICATIONS = 2
CALIFORNIA_HOUSING_CONTAMINATION_TYPE = "geo_split" 
BAS_DRO_EPSILON_SET = [
    0.001,
    0.002,
    0.005,
    0.01,
    0.02,
    0.03,
    0.04,
    0.05,
    0.06,
    0.07,
    0.08,
    0.09,
    0.1,
    0.15,
    0.2,
    0.25,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
    1,
    1.5,
    2,
    2.5,
    3,
]

SMALL_BAS_DRO_EPSILON_SET = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2]


PORTFOLIO_EPSILON_SET = [
    0.00001,
    0.00002,
    0.00005,
    0.0001,
    0.0002,
    0.0005,
    0.001,
    0.002,
    0.005,
    0.01,
    0.02,
    0.05,
    0.1,
    0.12,
    0.14,
    0.16,
    0.18,
    0.2,
    0.5,
    1.0,
]

NEWSVENDOR_EPSILON_SET = [
    0.00001,
    0.00002,
    0.00005,
    0.0001,
    0.0002,
    0.0005,
    0.001,
    0.002,
    0.005,
    0.01,
    0.02,
    0.05,
    0.1,
    0.125,
    0.15,
    0.2,
    0.25,
    0.3,
    0.35,
    0.4,
    0.45,
    0.5,
    0.75,
    1.0,
]

NEWSVENDOR_KL_EPSILON_SET = [
    0.0001,
    0.0002,
    0.0005,
    0.001,
    0.002,
    0.005,
    0.01,
    0.02,
    0.05,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.9,
    1.0,
    1.2,
    1.5,
    2.0,
    3.0,
    4.0,
    5.0,
    10.0,
    25.0,
]

LV_PORTFOLIO_T_SCALE = 3.0

LV_NEWSVENDOR_T_SCALE = 4.4721    


# Use the existing portfolio ε-grid as the LV-BAS grid.
LV_PORTFOLIO_EPSILON_SET = PORTFOLIO_EPSILON_SET

# Corresponding KL ε via  ε_KL = 0.5 * (ε_LV * t)^2
KL_FROM_LV_PORTFOLIO_EPSILON_SET = [
    0.5 * (eps * LV_PORTFOLIO_T_SCALE) ** 2 for eps in LV_PORTFOLIO_EPSILON_SET
]


ROBAS_DRO_EPSILON_SET = [
    0.0001,
    # 0.0005,
    0.001,
    # 0.003,
    0.005,
    0.01,
    0.05,
    0.1,
    0.15,
    0.2,
    # 0.25,
    0.3,
    0.5,
    1.0
]

def upper_triangular_size(dim: int) -> int:
    """Includes the diagonal!"""
    return int(dim * (dim-1) / 2 + dim)

# -----------------------------------------------------------------------------
# Real-world experiment grids
# -----------------------------------------------------------------------------
# Headline knobs from the blueprint
RW_GAMMA_GRID = [0.05]
#RW_EPSILON_GRID = [0.0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2]
RW_EPSILON_GRID = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.70, 0.75, 0.8, 0.85, 0.90, 0.95, 1.0]

# -----------------------------------------------------------------------------
# Sanity dataset defaults (smoke test for the real-world pipeline)
# -----------------------------------------------------------------------------
# Split fractions for in-sample data (train/cal/val). Must sum to 1.0.
RW_SANITY_SPLIT_FRACS = {"train": 0.7, "cal": 0.15, "val": 0.15}

# Spurious-correlation data generator parameters:
# - x_core correlates with y across all groups (signal we *want*)
# - x_spur correlates with a spurious attribute a that is correlated with y in the majority
#   group and anti-correlated in the minority group (signal we *don't want*)
RW_SANITY_M_CORE = 1.0
RW_SANITY_M_SPUR = 3.0
RW_SANITY_P_MATCH_CLEAN = 0.95   # P(a == y) in clean majority environment
RW_SANITY_P_MATCH_SHIFT = 0.05   # P(a == y) in shifted environment (roughly flips)

# A simple mean-shift to make the shifted component more "tail-like" under Mahalanobis scoring
RW_SANITY_SHIFT_SCALE = 3.0

# -----------------------------------------------------------------------------
# Training defaults for the sanity pipeline (paper configs will override later)
# -----------------------------------------------------------------------------
RW_DEFAULT_LR = 1e-2
RW_DEFAULT_EPOCHS = 50
RW_DEFAULT_BATCH_SIZE = 256
RW_DEFAULT_WEIGHT_DECAY = 1e-4

# LV-BAS smooth max: fixed temperature across runs (keeps ε as the robustness knob)
RW_USE_SMOOTHMAX = True
RW_SMOOTHMAX_TEMPERATURE = 0.2

# -----------------------------------------------------------------------------
# Real-world (embedding) experiment defaults
# -----------------------------------------------------------------------------
# RW_GAMMA_SET = [0.01, 0.05]
# RW_EPSILON_SET = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]

# Tuning protocol: chosen ONCE via ERM on validation, then frozen.
RW_TUNE_WEIGHT_DECAY_GRID = [0.0, 1e-5, 1e-4, 1e-3]
RW_TUNE_HEAD_CONFIGS = [
    {"kind": "linear"},
    {"kind": "mlp", "hidden_dim": 256, "dropout": 0.0},
]

# GroupDRO default
RW_GROUPDRO_ETA = 0.1

# Chi2-DRO knob mapping:
#   epsilon -> rho = epsilon * rho_max, where rho_max = (n-1)/2 for minibatch size n.
RW_CHI2_NORMALISATION = "max"

# --- Real-world LV-BAS (binary classification) defaults ---
RW_DEFAULT_W_NORM_BOUND = 10.0
RW_DEFAULT_BULK_RIDGE = 0.25
RW_DEFAULT_CENTER_N_PER_CLASS = 1024
RW_DEFAULT_CENTER_MAX_REJECT_FACTOR = 50.0
DO_THRESHOLD_CALIBRATION = False