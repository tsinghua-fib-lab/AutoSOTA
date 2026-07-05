from functools import cached_property
from typing import Literal
import torch
from pydantic import Field, AliasChoices
from pydantic_settings import BaseSettings, CliImplicitFlag, SettingsConfigDict
from .language_utils import SUPPORTED_LLM, LargeModel, ModelDtype
from .data import (
    ImageClassificationTask,
    LmClassificationTask,
    LmGenerationTask,
)


class FrozenSetting(BaseSettings, cli_parse_args=True, cli_ignore_unknown_args=True):
    # pydantic's config, not neural network model
    model_config = SettingsConfigDict(frozen=True)


class GeneralSetting(FrozenSetting):
    # general
    seed: int = Field(
        default=365,
        description="Random seed used to initialize model, get dataloaders and to sample the RGE's seeds each round",
    )
    total_samples: int = Field(
        default=80000,
        description="Total number of data samples",
        validation_alias=AliasChoices("total-samples"),
    )
    framework: str = Field(
        default="HO-SFL",
        description="Framework to use, only work in mock setting now",
    )
    log_to_wandb: CliImplicitFlag[bool] = Field(
        default=False,
        description="--log-to-wandb will enable logging to wandb, make sure to have wandb installed and set up",
        validation_alias=AliasChoices("log-to-wandb"),
    )
    project_name: str = Field(
        default="HO_SFL_LLM",
        description="Name of the current project, used for logging purposes",
        validation_alias=AliasChoices("project-name"),
    )
    log_dir: str = Field(
        default="./logs",
        description="Directory to save logs and checkpoints",
        validation_alias=AliasChoices("log-dir"),
    )

    @cached_property
    def general_setting(self) -> "GeneralSetting":
        return GeneralSetting()


class DeviceSetting(FrozenSetting):
    # device
    cuda: CliImplicitFlag[bool] = Field(
        default=True, description="--no-cuda will disable cuda training"
    )
    device: str = Field(
        default="cuda",
        description="Device to use for training, e.g., 'cuda' or 'cuda:0'",
    )

    @cached_property
    def device_setting(self) -> "DeviceSetting":
        return DeviceSetting()


class ModelSetting(FrozenSetting):
    # model
    large_model: LargeModel = Field(
        default=LargeModel.opt_125m,
        validation_alias=AliasChoices("large-model"),
        description="Model name for Hugging Face Lanuguage Model. current only support facebook/opt families",
    )
    model_dtype: ModelDtype = Field(
        default=ModelDtype.float32, validation_alias=AliasChoices("model-dtype")
    )
    # LoRA
    lora: CliImplicitFlag[bool] = Field(default=False)
    lora_r: int = Field(default=8, validation_alias=AliasChoices("lora-r"))
    lora_alpha: int = Field(default=16, validation_alias=AliasChoices("lora-alpha"))

    # Split
    split: CliImplicitFlag[bool] = Field(default=True)
    split_point: int = Field(default=6, validation_alias=AliasChoices("split-point"))

    @cached_property
    def model_setting(self) -> "ModelSetting":
        return ModelSetting()

    def get_hf_model_name(self) -> str:
        return SUPPORTED_LLM[self.large_model.value]

    def get_torch_dtype(self):
        return {
            ModelDtype.float16: torch.float16,
            ModelDtype.float32: torch.float32,
            ModelDtype.bfloat16: torch.bfloat16,
        }[self.model_dtype]


class OptimizerSetting(FrozenSetting):
    # optimizer
    optimizer: Literal["sgd", "adamw"] = Field(default="adamw")
    lr: float = Field(default=1e-5)
    momentum: float = Field(default=0)
    beta1: float = Field(default=0.9)
    beta2: float = Field(default=0.999)

    @cached_property
    def optimizer_setting(self) -> "OptimizerSetting":
        return OptimizerSetting()


class ZerothOrderSetting(FrozenSetting):
    mu: float = Field(
        default=1e-3, description="Perturbation step to measure local gradients"
    )
    num_pert: int = Field(
        default=5,
        validation_alias=AliasChoices("num-pert"),
        description="Number of perturbations needed to perform when estimating gradient",
    )

    @cached_property
    def zeroth_order_setting(self) -> "ZerothOrderSetting":
        return ZerothOrderSetting()


class HybridGradientEstimatorSetting(FrozenSetting):
    mu: float = Field(
        default=1e-3, description="Perturbation step to measure local gradients"
    )
    num_pert: int = Field(
        default=5,
        validation_alias=AliasChoices("num-pert"),
        description="Number of perturbations needed to perform when estimating gradient",
    )

    @cached_property
    def hybrid_gradient_estimator_setting(self) -> "HybridGradientEstimatorSetting":
        return HybridGradientEstimatorSetting()


class SingleTrainingLoopSetting(FrozenSetting):
    # training loop
    total_steps: int = Field(
        default=1000,
        validation_alias=AliasChoices("total-steps"),
        description="Total number of training steps to perform",
    )
    evaluation_interval: int = Field(
        default=50,
        validation_alias=AliasChoices("evaluation-interval"),
        description="Interval (in steps) between evaluations",
    )

    @cached_property
    def training_loop_setting(self) -> "SingleTrainingLoopSetting":
        return SingleTrainingLoopSetting()


class SplitFederatedLearningSetting(FrozenSetting):
    # Split Federated Learning settings
    num_clients: int = Field(
        default=3,
        validation_alias=AliasChoices("num-clients"),
        description="Number of clients in the simulation",
    )
    sampled_client_num: int = Field(
        default=3,
        validation_alias=AliasChoices("sampled-client-num"),
        description="Number of clients sampled in each round",
    )
    total_rounds: int = Field(
        default=50,
        validation_alias=AliasChoices("total-rounds"),
        description="Total number of training rounds to perform",
    )

    local_epochs: int = Field(
        default=1,
        validation_alias=AliasChoices("local-epochs"),
        description="Number of local epochs each client trains before sending updates to server",
    )

    evaluation_interval: int = Field(
        default=50,
        validation_alias=AliasChoices("evaluation-interval"),
        description="Interval (in steps) between evaluations",
    )

    @cached_property
    def sfl_setting(self) -> "SplitFederatedLearningSetting":
        return SplitFederatedLearningSetting()


class DataSetting(BaseSettings, cli_parse_args=True, cli_ignore_unknown_args=True):
    # pydantic's config, not neural network model
    model_config = SettingsConfigDict(frozen=True)

    # data
    dataset: ImageClassificationTask | LmClassificationTask | LmGenerationTask = Field(
        default=LmClassificationTask.sst2
    )
    max_length: int = Field(
        default=256,
        description="Max sequence length for language model tasks. Only valid when dataset is LmClassificationTask or LmGenerationTask",
        validation_alias=AliasChoices("max-length"),
    )
    train_batch_size: int = Field(
        default=16, validation_alias=AliasChoices("train-batch-size")
    )
    test_batch_size: int = Field(
        default=8, validation_alias=AliasChoices("test-batch-size")
    )
    iid: CliImplicitFlag[bool] = Field(
        default=True,
        description="Use dirichlet sampling for data split or iid sampling",
    )
    dirichlet_alpha: float = Field(
        default=1.0, validation_alias=AliasChoices("dirichlet-alpha")
    )
    small_test: CliImplicitFlag[bool] = Field(
        default=False,
        description="Use a smaller test set for quick testing, only valid for LmGenerationTask",
        validation_alias=AliasChoices("small-test"),
    )

    @cached_property
    def data_setting(self):
        return DataSetting()


class HO_SplitFederatedLearningSetting(FrozenSetting):
    # Split Federated Learning settings
    num_clients: int = Field(
        default=3,
        validation_alias=AliasChoices("num-clients"),
        description="Number of clients in the simulation",
    )
    sampled_client_num: int = Field(
        default=3,
        validation_alias=AliasChoices("sampled-client-num"),
        description="Number of clients sampled in each round",
    )
    total_steps: int = Field(
        default=800,
        validation_alias=AliasChoices("total-steps"),
        description="Total number of training steps to perform",
    )
    evaluation_interval: int = Field(
        default=50,
        validation_alias=AliasChoices("evaluation-interval"),
        description="Interval (in steps) between evaluations",
    )

    @cached_property
    def ho_sfl_setting(self) -> "HO_SplitFederatedLearningSetting":
        return HO_SplitFederatedLearningSetting()
