import sys
import os
from typing import List, Optional
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.config_store import ConfigStore

load_dotenv()

CONFIG_FILE_PATH = Path(os.path.abspath(__file__)).resolve()
MTG_PATH = CONFIG_FILE_PATH.parents[1]
sys.path.append(str(MTG_PATH))

# NOTE: You can define your paths in .env file or directly in the code below
# Where to save data, models, logs, etc.
WORKSPACE_PATH = Path(os.getenv("WORKSPACE_PATH") or "<YOUR_WORKSPACE_PATH>")
MTG_DATASET_PATH = Path(os.getenv("MTG_DATASET_PATH") or "<YOUR_MTG_DATASET_PATH>")
TRAINING_CONFIG_PATH = MTG_PATH / "configs/training"
DATASET_CONFIG_PATH = MTG_PATH / "configs/dataset"

WANDB_PROJECT_NAME = os.getenv("WANDB_PROJECT_NAME") or "<YOUR_WANDB_PROJECT_NAME>"

CLEANDIFT_PATH = str(MTG_PATH / "external/cleandift")
CLEANDIFT_CONFIG_PATH = f"{MTG_PATH}/external/sd21_feature_extractor.yaml"

SAM_CHECKPOINT_PATH = MTG_PATH / "sam_vit_h_4b8939.pth"


@dataclass
class InpaintingFluxConfig:
    num_inference_steps: int = 50
    height: int = 512
    width: int = 512
    guidance_scale: float = 3.5
    max_sequence_length: int = 512


@dataclass
class InpaintingSdxlConfig:
    guidance_scale: float = 7.5
    num_inference_steps: int = 30
    strength: float = 0.98
    padding_mask_crop: int = 32
    dilate_masks: bool = False
    dilation_kernel_size: int = 3


@dataclass
class GenerationConfig:
    # Basic run configuration
    name: str = "default"
    num_imgs: int = 100
    dataset_path: str = f"{MTG_DATASET_PATH}/default"

    img_size: List[int] = field(default_factory=lambda: [768, 768])

    # Generation configuration
    point_idxs_to_segment: List[int] = field(default_factory=lambda: [0, 10, 20, 30])
    matching_score_thresh: float = 0.7
    matching_skewness_thresh: float = 1.3
    part_aspect_ratio_diff_thresh: float = 0.2
    part_relative_sz_min_thresh: float = 0.05
    part_relative_sz_max_thresh: float = 0.6
    part_relative_sz_diff_thresh: float = 0.1
    inpainting_lpips_thresh: float = 0.15  # makes sure that the inpainting part is different from the original image

    # Inpainting configurations
    inpainting_model: str = "sdxl"  # flux | sdxl
    flux_config: InpaintingFluxConfig = field(default_factory=lambda: InpaintingFluxConfig())
    sdxl_config: InpaintingSdxlConfig = field(default_factory=lambda: InpaintingSdxlConfig())


###########################################################################


@dataclass
class DatasetConfig:
    dataset_name: str = "default"
    num_samples: int = -1
    dataset_path: Path = Path()
    img_size: List[int] = field(default_factory=lambda: [0, 0])
    train_val_split: float = 0.9
    num_visualization_samples: int = 10
    seed: int = 42

    @dataclass
    class DatasetFilteringConfig:
        skewness_thresh: float = -1
        lpips_thresh: float = -1
        source_target_rel_sz_diff: float = -1
        matching_score_thresh: float = -1

    @dataclass
    class DatasetAugmentationsConfig:
        enabled: bool = False
        horizontal_flip_prob: float = 0.5
        color_jitter_prob: float = 0.5
        brightness_range: List[float] = field(default_factory=lambda: [0.8, 1.2])
        contrast_range: List[float] = field(default_factory=lambda: [0.8, 1.2])
        saturation_range: List[float] = field(default_factory=lambda: [0.8, 1.2])

    filtering: DatasetFilteringConfig = field(default_factory=DatasetFilteringConfig)
    augmentations: DatasetAugmentationsConfig = field(default_factory=DatasetAugmentationsConfig)


@dataclass
class ExperimentDataConfig:
    train: DatasetConfig = field(default_factory=lambda: DatasetConfig())
    val: DatasetConfig = field(default_factory=lambda: DatasetConfig())


@dataclass
class ExperimentConfig:
    dtype: str = "bfloat16"
    device: str = "cuda:0"
    batch_size: int = 8
    lr: float = 1e-3
    epochs: int = 30
    scheduler_step: int = 10
    losses: List[str] = field(default_factory=lambda: ["bce"])

    @dataclass
    class ExperimentModelConfig:
        name: str = "base_model"
        masked: bool = True

    data: ExperimentDataConfig = field(default_factory=ExperimentDataConfig)
    model: ExperimentModelConfig = field(default_factory=ExperimentModelConfig)


def fetch_cfg_from_dataset(dataset_cfg: DatasetConfig):
    GlobalHydra.instance().clear()

    # Step 1: Init Hydra (use config_path relative to the working dir)
    initialize(
        config_path=str(DATASET_CONFIG_PATH.relative_to(CONFIG_FILE_PATH.parents[0])),
        version_base=None,
    )

    # Step 2: Compose the config (this simulates what Hydra normally does)
    cfg = compose(config_name=dataset_cfg.dataset_name)

    # Step 3: Use your config!
    # print(OmegaConf.to_yaml(cfg))

    dataset_cfg.dataset_path = Path(cfg.dataset_path)
    dataset_cfg.img_size = cfg.img_size

    return dataset_cfg


def load_dataset_generation_config(config_name: str = "default"):
    GlobalHydra.instance().clear()

    # Step 1: Init Hydra (use config_path relative to the working dir)
    initialize(config_path=str(DATASET_CONFIG_PATH.relative_to(CONFIG_FILE_PATH.parents[0])), version_base=None)

    # Step 2: Compose the config (this simulates what Hydra normally does)
    cfg = compose(config_name=config_name)

    # Step 3: Use your config!
    print(OmegaConf.to_yaml(cfg))

    cs = ConfigStore.instance()
    cs.store(name="dataset_generation_config", node=GenerationConfig)
    return cfg


def load_experiment_config(config_name: str = "default"):
    GlobalHydra.instance().clear()

    # Step 1: Init Hydra (use config_path relative to the working dir)
    initialize(
        config_path=str(TRAINING_CONFIG_PATH.relative_to(CONFIG_FILE_PATH.parents[0])),
        version_base=None,
    )

    # Step 2: Compose the config (this simulates what Hydra normally does)
    cfg = compose(config_name=config_name)

    # Step 3: Use your config!
    print(OmegaConf.to_yaml(cfg))

    cs = ConfigStore.instance()
    cs.store(name="experiment_config", node=ExperimentConfig)
    return cfg
