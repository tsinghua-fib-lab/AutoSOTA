from fvcore.common.registry import Registry
from loguru import logger

import dataset.fewshot_video as fewshot_video_datasets
import dataset.image as image_datasets
import dataset.video as video_datasets
from dataset.transforms import create_augmentations

DATASET_REGISTRY = Registry("DATASET")


def register_datasets(dataset_names, source):
    for dataset_name in dataset_names:
        dataset_class = getattr(source, dataset_name)
        DATASET_REGISTRY.register(dataset_class)


register_datasets(image_datasets.AVAIL_DATASETS, image_datasets)
register_datasets(video_datasets.AVAIL_DATASETS, video_datasets)
register_datasets(fewshot_video_datasets.AVAIL_DATASETS, fewshot_video_datasets)


def create_video_datasets(cfg, train_trsf, test_trsf):
    karwgs = dict(
        data_path=cfg.DATA.DATA_PATH,
        n_shot=cfg.DATA.N_SHOT,
        c_way=cfg.DATA.C_WAY,
        num_frames=cfg.DATA.NUM_FRAMES,
        seed=cfg.RNG_SEED,
        target_fps=cfg.DATA.TARGET_FPS,
        n_query_shot=cfg.DATA.N_QUERY_SHOT,
        use_all_classes=cfg.DATA.USE_ALL_CLASSES,
    )

    train_dataset = DATASET_REGISTRY.get(cfg.DATA.DATASET_NAME)(
        transforms=train_trsf,
        split="train",
        num_clips=1,
        sampler_type=cfg.DATA.TRAIN_VIDEO_SAMPLER,
        temporal_strides=cfg.DATA.TRAIN_STRIDES,
        **karwgs,
    )

    # "test" used in the path for fewshot
    val_split = "test" if cfg.DATA.FEWSHOT else "val"
    val_dataset = DATASET_REGISTRY.get(cfg.DATA.DATASET_NAME)(
        transforms=test_trsf,
        split=val_split,
        num_clips=cfg.DATA.TEST_NUM_CLIPS,
        sampler_type=cfg.DATA.TEST_VIDEO_SAMPLER,
        temporal_strides=cfg.DATA.TEST_STRIDES,
        **karwgs,
    )

    return train_dataset, val_dataset


def create_image_datasets(cfg, train_trsf, test_trsf):
    if cfg.DATA.DATASET_NAME == "UCF101":
        cfg.DATA.DATASET_NAME = "ImageUCF101"

    img_data = DATASET_REGISTRY.get(cfg.DATA.DATASET_NAME)(
        data_path=cfg.DATA.DATA_PATH,
        n_shot=cfg.DATA.N_SHOT,
        seed=cfg.RNG_SEED,
        use_base_and_new=cfg.DATA.USE_BASE_AND_NEW,
        train_transforms=train_trsf,
        test_transforms=test_trsf,
    )

    train_dataset, val_dataset = img_data.get_datasets()

    if cfg.DATA.TARGET_DATASET is not None:
        logger.info(f"Using target dataset: {cfg.DATA.TARGET_DATASET}")
        target_img_data = DATASET_REGISTRY.get(cfg.DATA.TARGET_DATASET)(
            data_path=cfg.DATA.DATA_PATH, test_transforms=test_trsf
        )

        new_val_dataset = {
            "source_test": val_dataset,
            "target_test": target_img_data.test_dataset,
        }

        return train_dataset, new_val_dataset

    return train_dataset, val_dataset


def create_datasets(cfg):
    video_dataset = cfg.DATA.TYPE == "video"

    train_augs = create_augmentations(
        augmentations=cfg.DATA.TRAIN_AUGS,
        mean=cfg.DATA.MEAN,
        std=cfg.DATA.STD,
        crop_size=cfg.DATA.TRAIN_CROP_SIZE,
        resize=cfg.DATA.TRAIN_RESIZE,
        video_inputs=video_dataset,
    )

    test_augs = create_augmentations(
        augmentations=cfg.DATA.TEST_AUGS,
        mean=cfg.DATA.MEAN,
        std=cfg.DATA.STD,
        crop_size=cfg.DATA.TEST_CROP_SIZE,
        resize=cfg.DATA.TEST_RESIZE,
        video_inputs=video_dataset,
    )

    if video_dataset:
        return create_video_datasets(cfg, train_augs, test_augs)

    return create_image_datasets(cfg, train_augs, test_augs)
