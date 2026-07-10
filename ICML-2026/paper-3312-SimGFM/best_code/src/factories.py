"""Factory classes for creating data modules, metrics, and model components."""
from functools import cached_property
from typing import Dict

from omegaconf import DictConfig

from src.analysis.spectre_utils import (
    Comm20SamplingMetrics, CommunitySamplingMetrics, EgoSamplingMetrics,
    PlanarSamplingMetrics, SBMSamplingMetrics, SpectreSamplingMetrics,
    TreeSamplingMetrics, TLSSamplingMetrics, CommunitySmallSamplingMetrics,
    CavemanSamplingMetrics, CoraSamplingMetrics, BreastSamplingMetrics,
    EnzymesSamplingMetrics, EgoSmallSamplingMetrics, GridSamplingMetrics
)
from src.datasets.abstract_dataset import (
    AbstractDataModule, InputOutputDimsCalculator, ProportionManager
)
from src.datasets.dataset_utils import DistributionNodes
from src.datasets.guacamol_dataset import GuacamolDataModule
from src.datasets.moses_dataset import MosesDataModule
from src.datasets.qm9_dataset import QM9DataModule
from src.datasets.spectre_dataset import SpectreGraphDataModule
from src.datasets.tls_dataset import TLSDataModule
from src.datasets.zinc_dataset import ZINCDataModule
from src.flow_matching.noise_distribution import NoiseDistribution
from src.metrics.molecular_metrics import SamplingMolecularMetrics


def is_spectre_dataset(name: str) -> bool:
    """Check if dataset is a SPECTRE dataset."""
    return name in [
        "sbm", "comm20", "planar", "tree", "ego", "community",
        "community-small", "caveman", "cora", "breast", "enzymes",
        "ego-small", "grid",
    ]


def is_molecular_dataset(name: str) -> bool:
    """Check if dataset is a molecular dataset."""
    return name in ["qm9", "guacamol", "moses", "zinc"]


def is_tls_dataset(name: str) -> bool:
    """Check if dataset is a TLS dataset."""
    return name == "tls"


class DataModuleFactory:
    """Factory for creating appropriate data module based on dataset type."""
    
    _molecular_dataset_dict = {
        "qm9": QM9DataModule,
        "guacamol": GuacamolDataModule,
        "moses": MosesDataModule,
        "zinc": ZINCDataModule,
    }
    
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.dataset_name = cfg["dataset"]["name"]
        self.remove_h = cfg["dataset"]["remove_h"] if hasattr(cfg["dataset"], "remove_h") else False
    
    @cached_property
    def datamodule(self):
        if is_spectre_dataset(self.dataset_name):
            return SpectreGraphDataModule(self.cfg)
        elif is_molecular_dataset(self.dataset_name):
            return self._molecular_dataset_dict[self.dataset_name](self.cfg)
        elif is_tls_dataset(self.dataset_name):
            return TLSDataModule(self.cfg)
        else:
            raise ValueError(f"Dataset {self.dataset_name} not implemented")

    def get_datamodule(self):
        return self.datamodule


class SamplingMetricsFactory:
    """Factory for creating appropriate sampling metrics based on dataset type."""
    
    _spectre_metrics_dict = {
        "sbm": SBMSamplingMetrics,
        "comm20": Comm20SamplingMetrics,
        "planar": PlanarSamplingMetrics,
        "tree": TreeSamplingMetrics,
        "ego": EgoSamplingMetrics,
        # Only degree, clustering, orbit for the following datasets
        "community": CommunitySamplingMetrics,
        "community-small": CommunitySmallSamplingMetrics,
        "caveman": CavemanSamplingMetrics,
        "cora": CoraSamplingMetrics,
        "breast": BreastSamplingMetrics,
        "enzymes": EnzymesSamplingMetrics,
        "ego-small": EgoSmallSamplingMetrics,
        "grid": GridSamplingMetrics,
    }
    
    _molecular_metrics_dict = {
        "qm9": SamplingMolecularMetrics,
        "guacamol": SamplingMolecularMetrics,
        "moses": SamplingMolecularMetrics,
        "zinc": SamplingMolecularMetrics,
    }
    
    _tls_metrics_dict = {
        "tls": TLSSamplingMetrics,
    }

    def __init__(
        self,
        datamodule: AbstractDataModule,
        cfg: DictConfig,
        portion_manager: ProportionManager,
        input_output_dims: Dict[str, int]
    ):
        self.cfg = cfg
        self.dataset_name = cfg["dataset"]["name"]
        self.datamodule = datamodule
        self.portion_manager = portion_manager
        self.input_output_dims = input_output_dims
    
    @cached_property
    def sampling_metrics(self):
        if is_spectre_dataset(self.dataset_name):
            return self._spectre_metrics_dict[self.dataset_name](self.datamodule)
        elif is_molecular_dataset(self.dataset_name):
            return self._molecular_metrics_dict[self.dataset_name](
                self.datamodule, self.cfg, self.input_output_dims, self.portion_manager
            )
        elif is_tls_dataset(self.dataset_name):
            return self._tls_metrics_dict[self.dataset_name](self.datamodule)
        else:
            raise ValueError(f"Dataset {self.dataset_name} not implemented")

    def get_sampling_metrics(self):
        return self.sampling_metrics


class ModelContextBuilder:
    """
    A builder class responsible for creating and managing all dataset-related components
    and providing them to the main model. It uses lazy initialization with cached_property
    to build components only when they are first accessed.

    Dependency graph:
    datamodule -> portion_manager -> extra_features -> input_output_dims
    datamodule -> sampling_metrics
    portion_manager -> node_dist
    portion_manager, input_output_dims -> noise_dist
    """
    
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

    @cached_property
    def datamodule(self):
        return DataModuleFactory(self.cfg).get_datamodule()

    @cached_property
    def sampling_metrics(self):
        return SamplingMetricsFactory(
            self.datamodule, self.cfg, self.portion_manager, self.out_put_dims
        ).get_sampling_metrics()

    @cached_property
    def portion_manager(self):
        return ProportionManager(self.datamodule, self.cfg)

    @cached_property
    def extra_features(self):
        from src.models.extra_features import ExtraFeatures
        max_nums_of_nodes = self.portion_manager.get_max_nums_of_nodes()
        return ExtraFeatures(
            self.cfg.model.extra_features,
            self.cfg.model.rrwp_steps,
            max_nums_of_nodes,
            self.cfg,
            is_molecular=is_molecular_dataset(self.cfg.dataset.name),
        )

    @cached_property
    def input_output_dims(self):
        return InputOutputDimsCalculator(self.datamodule, self.extra_features).compute()

    @cached_property
    def out_put_dims(self):
        return self.input_output_dims[1]

    @cached_property
    def node_dist(self):
        return DistributionNodes(self.portion_manager.get_proportion("node_nums"))

    @cached_property
    def noise_dist(self):
        return NoiseDistribution(self.cfg.model.transition, self.out_put_dims, self.portion_manager)

    def build(self):
        """Builds and returns a dictionary of components required by the main model."""
        return {
            "sampling_metrics": self.sampling_metrics,
            "extra_features": self.extra_features,
            "input_output_dims": self.input_output_dims,
            "node_dist": self.node_dist,
            "noise_dist": self.noise_dist,
        }

