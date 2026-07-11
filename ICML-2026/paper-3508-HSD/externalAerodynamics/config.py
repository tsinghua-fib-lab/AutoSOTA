"""
Configuration file for Flux Field Prediction on Manifolds

Task: Predict tangential velocity field (vector) from forcing (scalar)
      using edge flux as the primary representation.
"""
import torch


class Config:
 
    DATA_DIR = "./flux_field_data"
    PICKLE_FILE = "flux_field_dataset.pkl"

    OUTPUT_DIR = "./output_flux"
    LOG_FILE = "training_log.txt"
    VIZ_OUTPUT = "flux_field_comparison.png"
    
    # =========================================
    # Train/Test Split
    # =========================================
    TEST_RATIO = 0.2
    VAL_RATIO = 0.15
    
    # =========================================
    # Spectral Parameters
    # =========================================
    K_EIGENS = 128  # Number of eigenfunctions per form (0, 1, 2)
    
    # =========================================
    # Training Hyper-parameters
    # =========================================
    EPOCHS = 150
    WARMUP_EPOCHS = 5  # Linear warmup epochs
    BATCH_SIZE = 64
    LR = 1e-3           # Learning rate for baseline models
    LR_OURS = 1e-3      # Learning rate for HSD models
    WEIGHT_DECAY = 1e-4
    PATIENCE = 20       # Early stopping patience
    LAMBDA_COEFF_REG = 1e-4  
    LAMBDA_L1 = 1e-3
    LAMBDA_MODE_REG = 1e-5  # L2 regularization for mode_weights         
    
    # =========================================
    # GNO Parameters (Graph Neural Operator)
    # Output: 3D velocity vectors at nodes
    # =========================================
    GNO_HIDDEN_CHANNELS = 84
    GNO_PROJECTION_CHANNELS = 96
    GNO_N_LAYERS = 5
    GNO_RADIUS = 0.2
    
    # =========================================
    # FNO Parameters (Fourier Neural Operator)
    # Output: 3D velocity vectors at nodes
    # =========================================
    FNO_MODES = (4, 4, 4)
    FNO_HIDDEN_CHANNELS = 21
    FNO_N_LAYERS = 2
    FNO_GRID_RES = 16   # Resolution for 3D grid embedding
    
    # =========================================
    # MGN Parameters (MeshGraphNets)
    # Output: 3D velocity vectors at nodes
    # =========================================
    MGN_HIDDEN_DIM = 58
    MGN_NUM_LAYERS = 10
    
    # =========================================
    # DeepONet Parameters
    # Output: 3D velocity vectors at nodes
    # =========================================
    DEEPONET_BRANCH_LAYERS = [64, 64, 64]
    DEEPONET_TRUNK_LAYERS = [64, 64, 64]
    DEEPONET_BASIS_DIM = 74
    
    # =========================================
    # GeoFNO Parameters (Geometry-Aware FNO)
    # Output: 3D velocity vectors at nodes
    # =========================================
    GEOFNO_MODES = 6
    GEOFNO_WIDTH = 12
    GEOFNO_LAYERS = 2
    GEOFNO_GRID_RES = 16
    
    # =========================================
    # Spectral Model Parameters (HSD_base)
    # Output: 1-form spectral coefficients (flux)
    # =========================================
    SPECTRAL_HIDDEN_DIMS = (32, 32)  # Increased for flux prediction
    
    # =========================================
    # HSD Model Parameters (Spectral + FNO Hybrid)
    # Output: Edge flux directly
    # =========================================
    HSD_FNO_MODES = (8, 8, 8)
    HSD_FNO_HIDDEN = 32
    HSD_FNO_LAYERS = 3
    
    # =========================================
    # Loss Weights 
    # =========================================
    FLUX_LOSS_WEIGHT = 1.0
    DIVERGENCE_LOSS_WEIGHT = 0.5
    
    # =========================================
    # Visualization
    # =========================================
    VIZ_N_SAMPLES = 10  # Number of samples to visualize


# =========================================
# Device Setup
# =========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")