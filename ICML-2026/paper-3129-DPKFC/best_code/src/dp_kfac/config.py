from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from pathlib import Path
import yaml


@dataclass
class ExperimentConfig:
    name: str = "dp_kfac"
    seed: int = 42
    seeds: List[int] = field(default_factory=lambda: [42])
    device: str = "cuda"


@dataclass
class ScenarioConfig:
    name: str = ""
    private_dataset: str = "mnist"
    public_dataset: str = "fashionmnist"
    num_classes: int = 10
    img_size: int = 28
    description: str = ""
    prediction: str = ""


@dataclass
class DataConfig:
    private_dataset: str = "mnist"
    public_dataset: str = "fashionmnist"
    batch_size: int = 256
    num_workers: int = 4
    max_length: int = 128
    max_samples: int = 5000


@dataclass
class ModelConfig:
    type: str = "cnn"
    num_classes: int = 10
    pretrained: bool = True


@dataclass
class TrainingConfig:
    epochs: int = 5
    learning_rate: float = 1e-3
    optimizer: str = "adam"


@dataclass
class PrivacyConfig:
    epsilons: List[float] = field(default_factory=lambda: [1.0, 3.0, 5.0])
    delta: float = 1e-5
    max_grad_norm: float = 1.0


@dataclass
class KFACSpecificConfig:
    enabled: bool = True
    damping: float = 1e-3
    precond_steps: int = 10
    update_frequency: int = 1
    use_public_labels: bool = True


@dataclass
class BaselinesConfig:
    """Hyperparameters for baseline comparison methods (DP-AdamBC, DiSK)."""
    adam_bc_lr: Optional[float] = None  # if None, uses training.learning_rate
    adam_bc_eps_root: float = 1e-2
    adam_bc_betas: List[float] = field(default_factory=lambda: [0.9, 0.999])
    disk_kappa: float = 0.1
    disk_lr: Optional[float] = None  # if None, uses training.learning_rate
    adambc_kfac_lr: Optional[float] = None  # if None, uses adam_bc_lr


@dataclass
class OutputConfig:
    experiments_dir: str = "experiments"
    save_checkpoints: bool = False
    save_metrics: bool = True


@dataclass
class Config:
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    privacy: PrivacyConfig = field(default_factory=PrivacyConfig)
    kfac: KFACSpecificConfig = field(default_factory=KFACSpecificConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    baselines: BaselinesConfig = field(default_factory=BaselinesConfig)
    methods: List[str] = field(default_factory=lambda: ["dp_sgd", "dp_kfac_public"])
    scenarios: List[ScenarioConfig] = field(default_factory=list)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        with open(path, "r") as f:
            data = yaml.safe_load(f)

        config = cls()

        if "experiment" in data:
            config.experiment = ExperimentConfig(**data["experiment"])
        if "data" in data:
            data_cfg = data["data"].copy()
            if "max_length" in data_cfg:
                data_cfg["max_length"] = int(data_cfg["max_length"])
            if "max_samples" in data_cfg:
                data_cfg["max_samples"] = int(data_cfg["max_samples"])
            config.data = DataConfig(**data_cfg)
        if "model" in data:
            config.model = ModelConfig(**data["model"])
        if "training" in data:
            training_data = data["training"].copy()
            if "learning_rate" in training_data:
                training_data["learning_rate"] = float(training_data["learning_rate"])
            config.training = TrainingConfig(**training_data)
        if "privacy" in data:
            privacy_data = data["privacy"].copy()
            if "delta" in privacy_data:
                privacy_data["delta"] = float(privacy_data["delta"])
            if "max_grad_norm" in privacy_data:
                privacy_data["max_grad_norm"] = float(privacy_data["max_grad_norm"])
            if "epsilons" in privacy_data:
                privacy_data["epsilons"] = [float(e) for e in privacy_data["epsilons"]]
            config.privacy = PrivacyConfig(**privacy_data)
        if "kfac" in data:
            kfac_data = data["kfac"].copy()
            if "damping" in kfac_data:
                kfac_data["damping"] = float(kfac_data["damping"])
            config.kfac = KFACSpecificConfig(**kfac_data)
        if "output" in data:
            config.output = OutputConfig(**data["output"])
        if "baselines" in data:
            bl = data["baselines"].copy()
            if "adam_bc_lr" in bl and bl["adam_bc_lr"] is not None:
                bl["adam_bc_lr"] = float(bl["adam_bc_lr"])
            if "adam_bc_eps_root" in bl:
                bl["adam_bc_eps_root"] = float(bl["adam_bc_eps_root"])
            if "disk_kappa" in bl:
                bl["disk_kappa"] = float(bl["disk_kappa"])
            if "disk_lr" in bl:
                bl["disk_lr"] = float(bl["disk_lr"])
            if "adambc_kfac_lr" in bl and bl["adambc_kfac_lr"] is not None:
                bl["adambc_kfac_lr"] = float(bl["adambc_kfac_lr"])
            config.baselines = BaselinesConfig(**bl)
        if "methods" in data:
            config.methods = data["methods"]
        if "scenarios" in data:
            config.scenarios = [
                ScenarioConfig(**scenario) for scenario in data["scenarios"]
            ]

        return config

    def to_yaml(self, path: str | Path) -> None:
        data = {
            "experiment": {
                "name": self.experiment.name,
                "seed": self.experiment.seed,
                "seeds": self.experiment.seeds,
                "device": self.experiment.device,
            },
            "data": {
                "private_dataset": self.data.private_dataset,
                "public_dataset": self.data.public_dataset,
                "batch_size": self.data.batch_size,
                "num_workers": self.data.num_workers,
                "max_length": self.data.max_length,
                "max_samples": self.data.max_samples,
            },
            "model": {
                "type": self.model.type,
                "num_classes": self.model.num_classes,
                "pretrained": self.model.pretrained,
            },
            "training": {
                "epochs": self.training.epochs,
                "learning_rate": self.training.learning_rate,
                "optimizer": self.training.optimizer,
            },
            "privacy": {
                "epsilons": self.privacy.epsilons,
                "delta": self.privacy.delta,
                "max_grad_norm": self.privacy.max_grad_norm,
            },
            "kfac": {
                "enabled": self.kfac.enabled,
                "damping": self.kfac.damping,
                "precond_steps": self.kfac.precond_steps,
                "update_frequency": self.kfac.update_frequency,
                "use_public_labels": self.kfac.use_public_labels,
            },
            "output": {
                "experiments_dir": self.output.experiments_dir,
                "save_checkpoints": self.output.save_checkpoints,
                "save_metrics": self.output.save_metrics,
            },
            "methods": self.methods,
        }

        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
