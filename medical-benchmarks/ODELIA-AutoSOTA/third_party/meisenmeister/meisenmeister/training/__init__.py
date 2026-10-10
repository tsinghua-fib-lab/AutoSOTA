from meisenmeister.architectures import (
    BaseArchitecture,
    PrimusMClsNetwork,
    ResidualEncoderClsNetwork,
    ResNet3D18,
    get_architecture_class,
    get_architecture_registry,
    get_available_architecture_names,
)

from .base_trainer import BaseTrainer
from .predict import predict, predict_case_from_files, predict_from_modelfolder
from .preview_da import preview_da
from .registry import (
    get_available_trainer_names,
    get_trainer_class,
    get_trainer_registry,
)
from .splits import create_five_fold_splits, get_fold_sample_ids, load_splits
from .train import train
from .trainers.mm_trainer import mmTrainer
from .trainers.networks.nnunet_encoder import (
    mmTrainer_NNUNetEncoder,
    mmTrainer_NNUNetEncoder_Finetune,
)
from .trainers.networks.primus import mmTrainer_PrimusM, mmTrainer_PrimusM_bs4

__all__ = [
    "BaseArchitecture",
    "BaseTrainer",
    "PrimusMClsNetwork",
    "ResidualEncoderClsNetwork",
    "ResNet3D18",
    "create_five_fold_splits",
    "get_architecture_class",
    "get_architecture_registry",
    "get_available_architecture_names",
    "get_fold_sample_ids",
    "load_splits",
    "mmTrainer",
    "mmTrainer_NNUNetEncoder",
    "mmTrainer_NNUNetEncoder_Finetune",
    "mmTrainer_PrimusM",
    "mmTrainer_PrimusM_bs4",
    "predict",
    "predict_case_from_files",
    "predict_from_modelfolder",
    "preview_da",
    "get_available_trainer_names",
    "get_trainer_class",
    "get_trainer_registry",
    "train",
]
