"""
Configuration file for Scalar Field Prediction on Manifolds
"""
import torch

class Config:
    # --- Physics Parameters ---
    DIFFUSIVITY = 0.01
    ADVECTION_SPEED = 1.0
    DT = 0.02
    N_TIMESTEPS = 35  # Predict state after N timesteps
    
    # --- Data Generation ---
    N_SAMPLES = 3000
    N_BLOBS_MIN = 2
    N_BLOBS_MAX = 5
    BLOB_SIGMA = 0.15
    
    # --- Train/Test Split ---
    TEST_RATIO = 0.2
    VAL_RATIO = 0.15
    
    # --- Spectral Parameters ---
    K_EIGENS = 64  # Number of eigenfunctions per form
    
    # --- Training Hyper-params ---
    EPOCHS = 50
    BATCH_SIZE = 64
    LR = 1e-3
    LR_OURS = 1e-3
    WEIGHT_DECAY = 1e-5
    PATIENCE = 20
    
    # --- GNO Parameters ---
    GNO_HIDDEN_CHANNELS = 120
    GNO_PROJECTION_CHANNELS = 68
    GNO_N_LAYERS = 3
    GNO_RADIUS = 0.15
    
    # --- FNO Parameters ---
    FNO_MODES = (4, 4, 4)
    FNO_HIDDEN_CHANNELS = 20
    FNO_N_LAYERS = 3
    FNO_GRID_RES = 16
    
    # --- HSD FNO Parameters ---
    HSD_FNO_MODES = (4, 4, 4)
    HSD_FNO_HIDDEN = 12
    HSD_FNO_LAYERS = 6
    
    # --- MGN Parameters ---
    MGN_HIDDEN_DIM = 72
    MGN_NUM_LAYERS = 8
    
    # --- DeepONet Parameters ---
    DEEPONET_BRANCH_LAYERS = [96, 96, 64]
    DEEPONET_TRUNK_LAYERS = [68, 68, 64]
    DEEPONET_BASIS_DIM = 64
    
    # --- GeoFNO Parameters ---
    GEOFNO_MODES = 6
    GEOFNO_WIDTH = 8
    GEOFNO_LAYERS = 4
    GEOFNO_GRID_RES = 16
    

    # --- Spectral Model Parameters ---
    SPECTRAL_HIDDEN_DIMS = (32, 32)
    
    # --- Paths ---
    DATA_DIR = "./advection_diffusion_data"
    PICKLE_FILE = "advdiff_torus_dataset.pkl"
    OUTPUT_DIR = "./output_scalar"
    LOG_FILE = "training_log.txt"
    VIZ_OUTPUT = "scalar_field_comparison.png"


# Device setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")