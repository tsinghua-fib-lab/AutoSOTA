import os
import gc
import uuid
import torch
import wandb
import pathlib
import warnings
import importlib
import torch.nn as nn
from datetime import datetime
from appfl.config import ClientAgentConfig
from appfl.algorithm.trainer import BaseTrainer
from omegaconf import DictConfig, OmegaConf
from typing import Union, Dict, OrderedDict, Tuple, Optional
from appfl.logger import ClientAgentFileLogger
from appfl.misc.utils import (
    create_instance_from_file,
    run_function_from_file,
    get_function_from_file,
    create_instance_from_file_source,
    get_function_from_file_source,
    run_function_from_file_source,
    get_appfl_compressor,
)
from appfl.misc.data_readiness.metrics import (
    imbalance_degree,
    completeness,
    get_data_range,
    sparsity,
    variance,
    skewness,
    entropy,
    kurtosis,
    class_distribution,
    brisque,
    total_variation,
    dataset_sharpness,
    calculate_outlier_proportion,
    quantify_time_to_event_imbalance,
)
from appfl.misc.data_readiness.plots import (
    plot_class_distribution,
    plot_data_sample,
    plot_data_distribution,
    plot_class_variance,
    plot_outliers,
    plot_feature_correlations,
    plot_feature_statistics,
    get_feature_space_distribution,
)


class ClientAgent:
    """
    The `ClientAgent` should act on behalf of the FL client to:
    - load configurations received from the server `ClientAgent.load_config`
    - get the size of the local dataset `ClientAgent.get_sample_size`
    - do the local training job using configurations `ClientAgent.train`
    - prepare data for communication `ClientAgent.get_parameters`
    - load parameters from the server `ClientAgent.load_parameters`
    - get a unique client id for server to distinguish clients `ClientAgent.get_id`

    Developers can add new methods to the client agent to support more functionalities,
    and use Fork + Pull Request to contribute to the project.

    Users can overwrite any class method to add custom functionalities of the client agent.

    :param client_agent_config: configurations for the client agent
    """

    def __init__(
        self, client_agent_config: ClientAgentConfig = ClientAgentConfig(), **kwargs
    ) -> None:
        self.client_agent_config = client_agent_config
        # Check for optimize_memory in client_agent_config, default to True
        self.optimize_memory = getattr(client_agent_config, "optimize_memory", True)
        self._create_logger()
        self._init_wandb()
        self._load_model()
        self._load_loss()
        self._load_metric()
        self._load_data()
        self._load_trainer()
        self._load_compressor()

    def load_config(self, config: DictConfig) -> None:
        """Load additional configurations provided by the server."""
        self.client_agent_config = OmegaConf.merge(self.client_agent_config, config)
        self._load_model()
        self._load_loss()
        self._load_metric()
        self._load_trainer()
        self._load_compressor()

    def get_id(self) -> str:
        """Return a unique client id for server to distinguish clients."""
        if not hasattr(self, "client_id"):
            if hasattr(self.client_agent_config, "client_id"):
                self.client_id = str(self.client_agent_config.client_id)
            elif hasattr(self.client_agent_config, "train_configs") and hasattr(
                self.client_agent_config.train_configs, "logging_id"
            ):
                self.client_id = str(self.client_agent_config.train_configs.logging_id)
                # Emit a deprecated warning
                warnings.warn(
                    message="client_agent_config.train_configs.logging_id is deprecated. Please use client_id instead.",
                    category=DeprecationWarning,
                )
            else:
                warnings.warn(
                    message="Client ID (client_id) not specified. Generating a random one.",
                    category=UserWarning,
                )
                self.client_id = str(uuid.uuid4())
        return self.client_id

    def get_sample_size(self) -> int:
        """Return the size of the local dataset."""
        return len(self.train_dataset)

    def train(self, **kwargs) -> None:
        """Train the model locally."""
        self.trainer.train(**kwargs)

        # Memory optimization: Garbage collection after training
        if self.optimize_memory:
            gc.collect()
            # Clear CUDA cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def get_parameters(
        self,
    ) -> Union[Dict, OrderedDict, bytes, Tuple[Union[Dict, OrderedDict, bytes], Dict]]:
        """Return parameters for communication"""
        params = self.trainer.get_parameters()
        if isinstance(params, tuple):
            params, metadata = params
        else:
            metadata = None
        if self.enable_compression:
            params = self.compressor.compress_model(params)
            # Memory optimization: Garbage collection after compression
            if self.optimize_memory:
                gc.collect()

        # Memory optimization: Final cleanup
        if self.optimize_memory:
            gc.collect()

        return params if metadata is None else (params, metadata)

    def load_parameters(self, params) -> None:
        """Load parameters from the server."""
        self.trainer.load_parameters(params)

        # Memory optimization: Garbage collection after parameter loading
        if self.optimize_memory:
            gc.collect()

    def save_checkpoint(self, checkpoint_path: Optional[str] = None) -> None:
        """Save the model to a checkpoint file."""
        if checkpoint_path is None:
            output_dir = self.client_agent_config.train_configs.get(
                "checkpoint_dirname", "./output"
            )
            output_filename = self.client_agent_config.train_configs.get(
                "checkpoint_filename", "checkpoint"
            )
            curr_time_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            checkpoint_path = (
                f"{output_dir}/{output_filename}_{self.get_id()}_{curr_time_str}.pth"
            )

        # Make sure the directory exists
        if not os.path.exists(os.path.dirname(checkpoint_path)):
            pathlib.Path(os.path.dirname(checkpoint_path)).mkdir(
                parents=True, exist_ok=True
            )

        torch.save(self.model.state_dict(), checkpoint_path)

    def generate_readiness_report(self, client_config):
        """
        Generate data readiness report based on the configuration provided by the server.
        """
        if hasattr(client_config, "data_readiness_configs") and hasattr(
            client_config.data_readiness_configs, "dr_metrics"
        ):
            results = {}
            plot_results = {"plots": {}}
            to_combine_results = {"to_combine": {}}

            # Determine how to retrieve data input and labels based on dataset attributes
            if hasattr(self.train_dataset, "data_label"):
                data_labels = self.train_dataset.data_label.tolist()
            else:
                try:
                    data_labels = [label.item() for _, label in self.train_dataset]
                except:  # noqa E722
                    data_labels = [label for _, label in self.train_dataset]

            if hasattr(self.train_dataset, "data_input"):
                data_input = self.train_dataset.data_input
            else:
                data_input = torch.stack(
                    [input_data for input_data, _ in self.train_dataset]
                )

            if hasattr(
                client_config.data_readiness_configs.dr_metrics, "cadremodule_configs"
            ) and hasattr(
                client_config.data_readiness_configs.dr_metrics.cadremodule_configs,
                "cadremodule_kwargs",
            ):
                cadremodule_kwargs = getattr(
                    client_config.data_readiness_configs.dr_metrics.cadremodule_configs,
                    "cadremodule_kwargs",
                    {},
                )
            else:
                cadremodule_kwargs = {}

            # data_input, data_labels = balance_data(data_input, data_labels)
            # data_input, explained_variance = apply_pca(data_input)
            # data_input = normalize_data(data_input)

            # Define metrics with corresponding computation functions
            standard_metrics = {
                "class_imbalance": lambda: round(imbalance_degree(data_labels), 2),
                "sample_size": lambda: len(data_labels),
                "num_classes": lambda: len(set(data_labels)),
                "data_shape": lambda: (len(data_input), *data_input[0].size()),
                "completeness": lambda: completeness(data_input),
                "data_range": lambda: get_data_range(data_input),
                "overall_sparsity": lambda: sparsity(data_input),
                "variance": lambda: variance(data_input),
                "skewness": lambda: skewness(data_input),
                "entropy": lambda: entropy(data_input),
                "kurtosis": lambda: kurtosis(data_input),
                "class_distribution": lambda: class_distribution(data_labels),
                "brisque": lambda: brisque(data_input),
                "total_variation": lambda: total_variation(data_input),
                "sharpness": lambda: dataset_sharpness(data_input),
                "outlier_proportion": lambda: calculate_outlier_proportion(data_input),
                "time_to_event_imbalance": lambda: quantify_time_to_event_imbalance(
                    data_labels
                ),
            }

            plots = {
                "class_distribution_plot": lambda: plot_class_distribution(data_labels),
                "data_sample_plot": lambda: plot_data_sample(data_input),
                "data_distribution_plot": lambda: plot_data_distribution(data_input),
                "class_variance_plot": lambda: plot_class_variance(
                    data_input, data_labels
                ),
                "outlier_detection_plot": lambda: plot_outliers(data_input),
                # "time_to_event_plot": lambda: plot_time_to_event_distribution(data_labels), # TODO: Add time to event plot
                "feature_correlation_plot": lambda: plot_feature_correlations(
                    data_input
                ),
                "feature_statistics_plot": lambda: plot_feature_statistics(data_input),
            }
            combine = {
                "feature_space_distribution": lambda: get_feature_space_distribution(
                    data_input
                ),
            }

            # Handle standard metrics
            for metric_name, compute_function in standard_metrics.items():
                if hasattr(client_config.data_readiness_configs, "dr_metrics"):
                    if metric_name in client_config.data_readiness_configs.dr_metrics:
                        if getattr(
                            client_config.data_readiness_configs.dr_metrics, metric_name
                        ):
                            results[metric_name] = compute_function()

            # Handle plot-specific metrics
            for metric_name, compute_function in plots.items():
                if hasattr(client_config.data_readiness_configs.dr_metrics, "plot"):
                    if (
                        metric_name
                        in client_config.data_readiness_configs.dr_metrics.plot
                    ):
                        if getattr(
                            client_config.data_readiness_configs.dr_metrics.plot,
                            metric_name,
                        ):
                            plot_results["plots"][metric_name] = compute_function()

            # Combine results with plot results
            results.update(plot_results)

            # Handle combined metrics
            for metric_name, compute_function in combine.items():
                if hasattr(client_config.data_readiness_configs.dr_metrics, "combine"):
                    if (
                        metric_name
                        in client_config.data_readiness_configs.dr_metrics.combine
                    ):
                        if getattr(
                            client_config.data_readiness_configs.dr_metrics.combine,
                            metric_name,
                        ):
                            to_combine_results["to_combine"][metric_name] = (
                                compute_function()
                            )

            results.update(to_combine_results)

            # Handle CADRE module metrics
            if hasattr(
                client_config.data_readiness_configs.dr_metrics, "cadremodule_configs"
            ) and hasattr(
                client_config.data_readiness_configs.dr_metrics.cadremodule_configs,
                "cadremodule_path",
            ):
                self.specified_metrics = create_instance_from_file(
                    client_config.data_readiness_configs.dr_metrics.cadremodule_configs.cadremodule_path,
                    client_config.data_readiness_configs.dr_metrics.cadremodule_configs.get(
                        "cadremodule_name", None
                    ),
                    self.train_dataset,
                )
                results["specified_metrics"] = dict(
                    [
                        next(
                            iter(
                                self.specified_metrics.metric(
                                    **cadremodule_kwargs
                                ).items()
                            )
                        )
                    ]
                )

            return results
        else:
            return "Data readiness metrics not available in configuration"

    def adapt_data(self, client_config):
        """
        Modify the data based on the configuration provided by the daragent configs
        """

        if hasattr(
            client_config.data_readiness_configs.dr_metrics, "cadremodule_configs"
        ) and hasattr(
            client_config.data_readiness_configs.dr_metrics.cadremodule_configs,
            "cadremodule_path",
        ):
            self.specified_metrics = create_instance_from_file(
                client_config.data_readiness_configs.dr_metrics.cadremodule_configs.cadremodule_path,
                client_config.data_readiness_configs.dr_metrics.cadremodule_configs.get(
                    "cadremodule_name", None
                ),
                self.train_dataset,
            )

        if hasattr(
            client_config.data_readiness_configs.dr_metrics, "cadremodule_configs"
        ) and hasattr(
            client_config.data_readiness_configs.dr_metrics.cadremodule_configs,
            "cadremodule_kwargs",
        ):
            cadremodule_kwargs = getattr(
                client_config.data_readiness_configs.dr_metrics.cadremodule_configs,
                "cadremodule_kwargs",
                {},
            )
        else:
            cadremodule_kwargs = {}

        ai_ready_data = self.specified_metrics.remedy(
            self.specified_metrics.metric(**cadremodule_kwargs),
            self.logger,
            **cadremodule_kwargs,
        )
        self.train_dataset = ai_ready_data["ai_ready_dataset"]
        metadata = ai_ready_data["metadata"]

        return metadata

    def _create_logger(self):
        """
        Create logger for the client agent to log local training process.
        You can modify or overwrite this method to create your own logger.
        """
        if hasattr(self, "logger"):
            return
        kwargs = {}
        kwargs["logging_id"] = self.get_id()
        if not hasattr(self.client_agent_config, "train_configs"):
            kwargs["file_dir"] = "./output"
            kwargs["file_name"] = "result"
        else:
            kwargs["file_dir"] = self.client_agent_config.train_configs.get(
                "logging_output_dirname", "./output"
            )
            kwargs["file_name"] = self.client_agent_config.train_configs.get(
                "logging_output_filename", "result"
            )
        if hasattr(self.client_agent_config, "experiment_id"):
            kwargs["experiment_id"] = self.client_agent_config.experiment_id
        self.logger = ClientAgentFileLogger(**kwargs)

    def _load_data(self) -> None:
        """Get train and validation dataloaders from local dataloader file."""
        if not hasattr(self.client_agent_config, "data_configs"):
            self.train_dataset, self.val_dataset = None, None
            return
        if hasattr(self.client_agent_config.data_configs, "dataset_source"):
            self.train_dataset, self.val_dataset = run_function_from_file_source(
                self.client_agent_config.data_configs.dataset_source,
                self.client_agent_config.data_configs.dataset_name,
                **(
                    self.client_agent_config.data_configs.dataset_kwargs
                    if hasattr(self.client_agent_config.data_configs, "dataset_kwargs")
                    else {}
                ),
            )
        else:
            self.train_dataset, self.val_dataset = run_function_from_file(
                self.client_agent_config.data_configs.dataset_path,
                self.client_agent_config.data_configs.dataset_name,
                **(
                    self.client_agent_config.data_configs.dataset_kwargs
                    if hasattr(self.client_agent_config.data_configs, "dataset_kwargs")
                    else {}
                ),
            )

            # Convert target to Long if it is not already
            # self.train_dataset = apply_pca_to_dataset(self.train_dataset)
            # self.val_dataset = apply_pca_to_dataset(self.val_dataset)

            # self.train_dataset = normalize_dataset(self.train_dataset)
            # self.val_dataset = normalize_dataset(self.val_dataset)

            # self.train_dataset = balance_classes_undersample(self.train_dataset)

    def _load_model(self) -> None:
        """
        Load model from various sources with optional keyword arguments `model_kwargs`:
        - `model_path` and `model_name`: load model from a local file (usually for local simulation)
        - `model_source` and `model_name`: load model from a raw file source string (usually sent from the server)
        - Users can define their own way to load the model from other sources
        """
        if hasattr(self, "model") and self.model is not None:
            return
        if not hasattr(self.client_agent_config, "model_configs"):
            self.model = None
            return
        if hasattr(self.client_agent_config.model_configs, "model_path"):
            kwargs = self.client_agent_config.model_configs.get("model_kwargs", {})
            if hasattr(self.client_agent_config.model_configs, "model_name"):
                self.model = create_instance_from_file(
                    self.client_agent_config.model_configs.model_path,
                    self.client_agent_config.model_configs.model_name,
                    **kwargs,
                )
            else:
                self.model = run_function_from_file(
                    self.client_agent_config.model_configs.model_path, None, **kwargs
                )
        elif hasattr(self.client_agent_config.model_configs, "model_source"):
            kwargs = self.client_agent_config.model_configs.get("model_kwargs", {})
            if hasattr(self.client_agent_config.model_configs, "model_name"):
                self.model = create_instance_from_file_source(
                    self.client_agent_config.model_configs.model_source,
                    self.client_agent_config.model_configs.model_name,
                    **kwargs,
                )
            else:
                self.model = run_function_from_file_source(
                    self.client_agent_config.model_configs.model_source, None, **kwargs
                )
        else:
            self.model = None

    def _load_loss(self) -> None:
        """
        Load loss function from various sources with optional keyword arguments `loss_fn_kwargs`:
        - `loss_fn_path` and `loss_fn_name`: load loss function from a local file (usually for local simulation)
        - `loss_fn_source` and `loss_fn_name`: load loss function from a raw file source string (usually sent from the server)
        - `loss_fn`: load commonly-used loss function from `torch.nn` module
        - Users can define their own way to load the loss function from other sources
        """
        if hasattr(self, "loss_fn") and self.loss_fn is not None:
            return
        if not hasattr(self.client_agent_config, "train_configs"):
            self.loss_fn = None
            return
        if hasattr(self.client_agent_config.train_configs, "loss_fn_path"):
            kwargs = self.client_agent_config.train_configs.get("loss_fn_kwargs", {})
            self.loss_fn = create_instance_from_file(
                self.client_agent_config.train_configs.loss_fn_path,
                self.client_agent_config.train_configs.loss_fn_name
                if hasattr(self.client_agent_config.train_configs, "loss_fn_name")
                else None,
                **kwargs,
            )
        elif hasattr(self.client_agent_config.train_configs, "loss_fn"):
            kwargs = self.client_agent_config.train_configs.get("loss_fn_kwargs", {})
            if hasattr(nn, self.client_agent_config.train_configs.loss_fn):
                self.loss_fn = getattr(
                    nn, self.client_agent_config.train_configs.loss_fn
                )(**kwargs)
            else:
                self.loss_fn = None
        elif hasattr(self.client_agent_config.train_configs, "loss_fn_source"):
            kwargs = self.client_agent_config.train_configs.get("loss_fn_kwargs", {})
            self.loss_fn = create_instance_from_file_source(
                self.client_agent_config.train_configs.loss_fn_source,
                self.client_agent_config.train_configs.loss_fn_name
                if hasattr(self.client_agent_config.train_configs, "loss_fn_name")
                else None,
                **kwargs,
            )
        else:
            self.loss_fn = None

    def _load_metric(self) -> None:
        """
        Load metric function from various sources:
        - `metric_path` and `metric_name`: load metric function from a local file (usually for local simulation)
        - `metric_source` and `metric_name`: load metric function from a raw file source string (usually sent from the server)
        - Users can define their own way to load the metric function from other sources
        """
        if hasattr(self, "metric") and self.metric is not None:
            return
        if not hasattr(self.client_agent_config, "train_configs"):
            self.metric = None
            return
        if hasattr(self.client_agent_config.train_configs, "metric_path"):
            self.metric = get_function_from_file(
                self.client_agent_config.train_configs.metric_path,
                self.client_agent_config.train_configs.metric_name
                if hasattr(self.client_agent_config.train_configs, "metric_name")
                else None,
            )
        elif hasattr(self.client_agent_config.train_configs, "metric_source"):
            self.metric = get_function_from_file_source(
                self.client_agent_config.train_configs.metric_source,
                self.client_agent_config.train_configs.metric_name
                if hasattr(self.client_agent_config.train_configs, "metric_name")
                else None,
            )
        else:
            self.metric = None

    def _load_trainer(self) -> None:
        """Obtain a local trainer"""
        if hasattr(self, "trainer") and self.trainer is not None:
            return
        if not hasattr(self.client_agent_config, "train_configs"):
            self.trainer = None
            return
        if (
            not hasattr(self.client_agent_config.train_configs, "trainer")
            and not hasattr(self.client_agent_config.train_configs, "trainer_path")
            and not hasattr(self.client_agent_config.train_configs, "trainer_source")
        ):
            self.trainer = None
            return
        if hasattr(self.client_agent_config.train_configs, "trainer_path"):
            self.trainer = create_instance_from_file(
                self.client_agent_config.train_configs.trainer_path,
                self.client_agent_config.train_configs.trainer,
                model=self.model,
                loss_fn=self.loss_fn,
                metric=self.metric,
                train_dataset=self.train_dataset,
                val_dataset=self.val_dataset,
                train_configs=self.client_agent_config.train_configs,
                logger=self.logger,
            )
        elif hasattr(self.client_agent_config.train_configs, "trainer_source"):
            self.trainer = create_instance_from_file_source(
                self.client_agent_config.train_configs.trainer_source,
                self.client_agent_config.train_configs.trainer,
                model=self.model,
                loss_fn=self.loss_fn,
                metric=self.metric,
                train_dataset=self.train_dataset,
                val_dataset=self.val_dataset,
                train_configs=self.client_agent_config.train_configs,
                logger=self.logger,
            )
        else:
            trainer_module = importlib.import_module("appfl.algorithm.trainer")
            if not hasattr(
                trainer_module, self.client_agent_config.train_configs.trainer
            ):
                if self.client_agent_config.train_configs.trainer == "MonaiTrainer":
                    raise ImportError(
                        'Monai is not installed. Please install Monai to use MonaiTrainer using: pip install "appfl[monai]" or pip install -e ".[monai]" if installing from source.'
                    )
                raise ValueError(
                    f"Invalid trainer name: {self.client_agent_config.train_configs.trainer}"
                )
            self.trainer: BaseTrainer = getattr(
                trainer_module, self.client_agent_config.train_configs.trainer
            )(
                model=self.model,
                loss_fn=self.loss_fn,
                metric=self.metric,
                train_dataset=self.train_dataset,
                val_dataset=self.val_dataset,
                train_configs=self.client_agent_config.train_configs,
                logger=self.logger,
                client_id=self.get_id(),  # currently, only useful for MonaiTrainer
            )

    def _load_compressor(self) -> None:
        """
        Create a compressor for compressing the model parameters.
        """
        if hasattr(self, "compressor") and self.compressor is not None:
            return
        self.compressor = None
        self.enable_compression = False
        if not hasattr(self.client_agent_config, "comm_configs"):
            return
        if not hasattr(self.client_agent_config.comm_configs, "compressor_configs"):
            return
        if getattr(
            self.client_agent_config.comm_configs.compressor_configs,
            "enable_compression",
            False,
        ):
            self.enable_compression = True
            self.compressor = get_appfl_compressor(
                compressor_name=self.client_agent_config.comm_configs.compressor_configs.lossy_compressor,
                compressor_config=self.client_agent_config.comm_configs.compressor_configs,
            )

    def _init_wandb(self) -> None:
        """
        Initialize Weights and Biases for logging.
        """
        if not hasattr(self.client_agent_config, "wandb_configs"):
            self.client_agent_config.train_configs.enable_wandb = wandb.run is not None
            self.client_agent_config.train_configs.wandb_logging_id = self.get_id()
            return
        if not self.client_agent_config.wandb_configs.get("enable_wandb", False):
            self.client_agent_config.train_configs.enable_wandb = wandb.run is not None
            self.client_agent_config.train_configs.wandb_logging_id = self.get_id()
            return
        wandb.init(
            entity=self.client_agent_config.wandb_configs.get("entity", None),
            project=self.client_agent_config.wandb_configs.get("project", None),
            dir=self.client_agent_config.train_configs.get(
                "logging_output_dirname", "./output"
            ),
            id=self.client_agent_config.get("experiment_id", wandb.util.generate_id()),
            name=self.client_agent_config.wandb_configs.get("exp_name", "appfl"),
            config=OmegaConf.to_container(self.client_agent_config, resolve=True),
            resume="allow",
        )
        self.client_agent_config.train_configs.enable_wandb = True
        self.client_agent_config.train_configs.wandb_logging_id = self.get_id()

    @property
    def runtime_context(self):
        return getattr(self.trainer, "runtime_context", None)

    @runtime_context.setter
    def runtime_context(self, ctx):
        self.trainer.runtime_context = ctx
