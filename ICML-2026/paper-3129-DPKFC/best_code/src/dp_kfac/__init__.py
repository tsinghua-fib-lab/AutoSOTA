from .types import KFACConfig, DPConfig, CovariancePair
from .recorder import KFACRecorder
from .covariance import (
    compute_covariances,
    compute_inverse_sqrt,
    accumulate_covariances,
)
from .precondition import precondition_per_sample_gradients
from .privacy import clip_and_noise_gradients
from .optimizer import DPKFACOptimizer, generate_pink_noise
from .config import Config
from .experiment import ExperimentRun
from .models import (
    MLP,
    SimpleCNN,
    TinyViT,
    CrossViTClassifier,
    CrossViTLoRAClassifier,
    ConvNeXtClassifier,
    RoBERTaClassifier,
    BERTClassifier,
    DistilBERTClassifier,
    LinearClassifier,
    LogisticRegression,
    create_model,
    get_model_img_size,
)
from .data import (
    get_dataset_loaders,
    get_dataset_info,
    get_imdb_data,
    get_sst2_data,
    get_tfidf_features,
    get_text_loaders,
    get_agnews_data,
    get_stackoverflow_data,
    is_text_dataset,
)
from .baselines import DPAdamBC, DiSKFilter
from .trainer import Trainer, train_dp_sgd, train_plain_sgd, evaluate, set_seed
from .methods import (
    METHODS,
    PrecondMethod,
    get_method,
    is_kfac_method,
    list_methods,
    compute_preconditioner,
    estimate_adadps_preconditioner,
    precondition_per_sample_gradients_adadps,
    generate_clustered_pink_noise,
    generate_white_noise,
)
from .results import (
    save_results_csv,
    load_results_csv,
    aggregate_results,
    print_summary_table,
)
from .analysis import (
    compute_kfac_eigenvalues,
    compute_condition_number,
    compute_covariance_similarity,
    track_covariances_epoch,
)

__all__ = [
    # Types
    "KFACConfig", "DPConfig", "CovariancePair",
    # KFAC core
    "KFACRecorder", "compute_covariances", "compute_inverse_sqrt",
    "accumulate_covariances", "precondition_per_sample_gradients",
    # Privacy
    "clip_and_noise_gradients",
    # Optimizer
    "DPKFACOptimizer", "generate_pink_noise",
    # Config & experiment
    "Config", "ExperimentRun",
    # Models
    "MLP", "SimpleCNN", "CrossViTClassifier", "CrossViTLoRAClassifier", "ConvNeXtClassifier",
    "RoBERTaClassifier", "BERTClassifier", "DistilBERTClassifier",
    "LinearClassifier", "LogisticRegression",
    "create_model", "get_model_img_size",
    # Data
    "get_dataset_loaders", "get_dataset_info",
    "get_imdb_data", "get_sst2_data", "get_tfidf_features",
    "get_text_loaders", "get_agnews_data", "get_stackoverflow_data",
    "is_text_dataset",
    # Baselines
    "DPAdamBC", "DiSKFilter",
    # Training
    "Trainer", "train_dp_sgd", "train_plain_sgd", "evaluate", "set_seed",
    # Methods
    "METHODS", "PrecondMethod", "get_method", "is_kfac_method", "list_methods",
    "compute_preconditioner", "estimate_adadps_preconditioner",
    "precondition_per_sample_gradients_adadps", "generate_clustered_pink_noise",
    "generate_white_noise",
    # Results
    "save_results_csv", "load_results_csv", "aggregate_results", "print_summary_table",
    # Analysis
    "compute_kfac_eigenvalues", "compute_condition_number",
    "compute_covariance_similarity", "track_covariances_epoch",
]

__version__ = "0.1.0"
