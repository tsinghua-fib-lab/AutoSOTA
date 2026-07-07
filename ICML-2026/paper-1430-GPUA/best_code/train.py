import argparse

import torch
from loguru import logger

from dataset import create_datasets
from default_configs import get_cfg
from model import ClipPrompts, VideoClipPrompts
from trainer import Trainer
from utils import seed_everything


def parse_args():
    parser = argparse.ArgumentParser()
    # the yaml config file
    parser.add_argument("--config", dest="config_file", type=str, default=None)
    # followed by new config values
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    return parser.parse_args()


def create_dataloaders(cfg, train_dataset, eval_dataset):
    train_bs = cfg.DATALOADER.TRAIN_BATCHSIZE
    test_bs = cfg.DATALOADER.TEST_BATCHSIZE
    n_workers = cfg.DATALOADER.NUM_WORKERS
    pin_memory = torch.cuda.is_available() and cfg.DATALOADER.PIN_MEMORY

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_bs,
        shuffle=False,
        num_workers=n_workers,
        drop_last=False,
        pin_memory=pin_memory,
    )

    eval_loader_kwargs = dict(
        batch_size=test_bs,
        shuffle=False,
        num_workers=n_workers,
        drop_last=False,
        pin_memory=pin_memory,
    )

    if isinstance(eval_dataset, dict):
        eval_dataloader = {
            test_name: torch.utils.data.DataLoader(dataset=eval_dataset,
                                                   **eval_loader_kwargs)
            for test_name, eval_dataset in eval_dataset.items()
        }
        return train_dataloader, eval_dataloader

    else:
        eval_dataloader = {
            "test": torch.utils.data.DataLoader(dataset=eval_dataset,
                                                **eval_loader_kwargs)
        }

    return train_dataloader, eval_dataloader


def create_model(cfg, train_class_names, test_class_names):

    model_kwargs = dict(
        viz_backbone=cfg.MODEL.VIZ_BACKBONE,
        img_size=cfg.DATA.TRAIN_CROP_SIZE,
        train_class_names=train_class_names,
        test_class_names=test_class_names,
        per_class_prompts=cfg.MODEL.PER_CLS_PROMPTS,
        prompt_position=cfg.MODEL.PROMPT_POSITION,
        number_of_prompts=cfg.MODEL.NUM_PROMPTS,
        use_conditioning=cfg.MODEL.IMG_CONDITIONING,
        frame_aggregation=cfg.MODEL.FRAME_AGGREGATION,
        num_frames=cfg.DATA.NUM_FRAMES,
        softmax_temp=cfg.MODEL.SOFTMAX_TEMP,
    )

    if cfg.DATA.TYPE == "video":
        return VideoClipPrompts(**model_kwargs)

    return ClipPrompts(**model_kwargs)


def create_trainer(cfg, model, train_dataloader, eval_dataloader, loss_fn):
    optimizer_kwargs = dict(
        opt_type=cfg.OPT.TYPE,
        learning_rate=cfg.OPT.LR,
        weight_decay=cfg.OPT.WEIGHT_DECAY,
        zero_wd_for_biases=cfg.OPT.ZERO_WD_1D_PARAM,
    )

    lr_policy_kwargs = dict(
        lr_policy_type=cfg.OPT.LR_POLICY,
        lr_step_milestones=cfg.OPT.STEPS,
        warmup_epochs=cfg.OPT.WARMUP_EPOCHS,
        consine_end_lr=cfg.OPT.COSINE_END_LR,
        linear_end_lr=cfg.OPT.LINEAR_END_LR,
    )

    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        loss_fn=loss_fn,
        max_epochs=cfg.OPT.MAX_EPOCHS,
        log_interval=cfg.LOG_INTERVAL,
        save_interval=cfg.SAVE_INTERVAL,
        eval_interval=cfg.EVAL_INTERVAL,
        save_path=cfg.SAVE_PATH,
        training_precision=cfg.TRAIN_PRECISION,
        resume_checkpoint=cfg.RESUME_CHECKPOINT,
        optimizer_kwargs=optimizer_kwargs,
        lr_policy_kwargs=lr_policy_kwargs,
        clip_l2_gradnorm=cfg.OPT.CLIP_L2_GRADNORM,
        log_to_wandb=cfg.LOG_TO_WANDB,
        config=cfg,
    )

    return trainer


def main(cfg):
    seed_everything(cfg.RNG_SEED)

    train_dataset, eval_dataset = create_datasets(cfg)
    train_dataloader, eval_dataloader = create_dataloaders(
        cfg, train_dataset, eval_dataset
    )

    if isinstance(eval_dataset, dict):
        # In case we have base & new test sets, use new class names
        eval_label_names = list(eval_dataset.values())[-1].label_names
    else:
        eval_label_names = eval_dataset.label_names

    model = create_model(cfg, train_dataset.label_names, eval_label_names)

    loss_fn = torch.nn.CrossEntropyLoss()

    trainer = create_trainer(
        cfg, model, train_dataloader, eval_dataloader, loss_fn)

    if cfg.EVAL_ONLY:
        trainer.run_eval_loop()
    else:
        trainer.run_training_loop()

    return trainer.final_results


if __name__ == "__main__":
    args = parse_args()
    cfg = get_cfg(args)

    results = main(cfg)
    logger.info("Final results:")
    logger.info(results)
