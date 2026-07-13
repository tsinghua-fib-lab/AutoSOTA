"""
SENDAI: A Hierarchical Sparse-measurement, EfficieNt Data AssImilation Framework
"""

from .models import (
    SHRED, DASHRED, LatentGAN, HF_SHRED_INR, HF_SHRED_MLP, FullDASHRED,
    FourierPositionalEncoding, SirenLayer,
)
from .training import (
    train_shred, train_dashred, train_hf_layer, train_hf_finetune,
    train_hierarchical_hf, evaluate, get_top_frequencies,
)
from .data import (
    load_data, detect_bad_frames, fix_bad_frames, select_sensors,
    create_time_delay_dataset, SHREDDataset,
)
from .metrics import (
    compute_ssim, compute_gradient_error, compute_morans_i,
    compute_morans_i_batch, compute_all_metrics,
    save_metrics_to_csv, save_comprehensive_metrics,
)
from .visualization import plot_reconstruction, plot_temporal, plot_hf_analysis
from .utils import (
    get_device, TimingLogger, count_parameters,
    count_hf_layer_parameters, print_parameter_summary, Tee,
)

__version__ = '1.0.0'
