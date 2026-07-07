import argparse
import torch

from dassl.utils import setup_logger, set_random_seed, collect_env_info
from dassl.config import get_cfg_default
from dassl.engine import build_trainer
from trainers.alignednorm_config import (
    get_dataset_specified_config as get_alignednorm_cfg,
)
from yacs.config import CfgNode as CN

# custom
import datasets.oxford_pets
import datasets.oxford_flowers
import datasets.fgvc_aircraft
import datasets.dtd
import datasets.eurosat
import datasets.stanford_cars
import datasets.food101
import datasets.sun397
import datasets.caltech101
import datasets.ucf101
import datasets.imagenet

import datasets.imagenet_sketch
import datasets.imagenetv2
import datasets.imagenet_a
import datasets.imagenet_r

import trainers.mmrl
import trainers.mmrlpp
import trainers.alignednorm

def print_args(args, cfg):
    print("***************")
    print("** Arguments **")
    print("***************")
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print("{}: {}".format(key, args.__dict__[key]))
    print("************")
    print("** Config **")
    print("************")
    print(cfg)


def reset_cfg(cfg, args):
    if args.train_epoch:
        cfg.OPTIM.MAX_EPOCH = args.train_epoch

    if args.root:
        cfg.DATASET.ROOT = args.root

    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir

    if args.resume:
        cfg.RESUME = args.resume

    if args.seed:
        cfg.SEED = args.seed

    if args.source_domains:
        cfg.DATASET.SOURCE_DOMAINS = args.source_domains

    if args.target_domains:
        cfg.DATASET.TARGET_DOMAINS = args.target_domains

    if args.transforms:
        cfg.INPUT.TRANSFORMS = args.transforms

    if args.trainer:
        cfg.TRAINER.NAME = args.trainer

    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone

    if args.head:
        cfg.MODEL.HEAD.NAME = args.head


def extend_cfg(cfg):
    """
    Add new config variables.

    E.g.
        from yacs.config import CfgNode as CN
        cfg.TRAINER.MY_MODEL = CN()
        cfg.TRAINER.MY_MODEL.PARAM_A = 1.
        cfg.TRAINER.MY_MODEL.PARAM_B = 0.5
        cfg.TRAINER.MY_MODEL.PARAM_C = False
    """

    cfg.TRAINER.MMRL = CN()
    cfg.TRAINER.MMRL.ALPHA = 0.7
    cfg.TRAINER.MMRL.REG_WEIGHT = 1.0
    cfg.TRAINER.MMRL.REP_LAYERS = []
    cfg.TRAINER.MMRL.REP_DIM = 512
    cfg.TRAINER.MMRL.N_REP_TOKENS = 5  # number of representation tokens per layer
    cfg.TRAINER.MMRL.PREC = "fp16"  # fp16, fp32, amp
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new
    cfg.TASK = "B2N" #B2N, CD, FS

    cfg.TRAINER.MMRLpp = CN()
    cfg.TRAINER.MMRLpp.ALPHA = 0.7
    cfg.TRAINER.MMRLpp.BETA = 0.9
    cfg.TRAINER.MMRLpp.REG_WEIGHT = 1.0
    cfg.TRAINER.MMRLpp.REP_LAYERS = []
    cfg.TRAINER.MMRLpp.REP_DIM = 512
    cfg.TRAINER.MMRLpp.N_REP_TOKENS = 5  # number of representation tokens per layer
    cfg.TRAINER.MMRLpp.PROJ_LORA_DIM = 64
    cfg.TRAINER.MMRLpp.RES_LORA_DIM = 4
    cfg.TRAINER.MMRLpp.PREC = "fp16"  # fp16, fp32, amp
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new
    cfg.TASK = "B2N" #B2N, CD, FS
    
    cfg.TRAINER.ALIGNEDNORM = CN()
    cfg.TRAINER.ALIGNEDNORM.ALPHA = 0.7
    cfg.TRAINER.ALIGNEDNORM.BETA = 0.9
    cfg.TRAINER.ALIGNEDNORM.FINAL_NORM = 0.1
    cfg.TRAINER.ALIGNEDNORM.SEQ_NORM = 0.1
    cfg.TRAINER.ALIGNEDNORM.REG_WEIGHT = 1.0
    cfg.TRAINER.ALIGNEDNORM.REP_LAYERS = []
    cfg.TRAINER.ALIGNEDNORM.REP_DIM = 512
    cfg.TRAINER.ALIGNEDNORM.N_REP_TOKENS = 5  # number of representation tokens per layer
    cfg.TRAINER.ALIGNEDNORM.PROJ_LORA_DIM = 64
    cfg.TRAINER.ALIGNEDNORM.RES_LORA_DIM = 4
    cfg.TRAINER.ALIGNEDNORM.PREC = "fp16"  # fp16, fp32, amp
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new
    cfg.TASK = "B2N" #B2N, CD, FS
    cfg.DATALOADER.RETURN_IMG0 = True
    
def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg)

    # 1. From the dataset config file
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)

    # 2. From the method config file
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    # 3. From input arguments
    reset_cfg(cfg, args)

    # 4. Override dataset specific config
    cfg.merge_from_list(
        get_alignednorm_cfg(
            dataset=cfg.DATASET.NAME,
            trainer=cfg.TRAINER.NAME,
            task=args.opts[args.opts.index("TASK") + 1],
        )
    )

    # 5. From optional input arguments
    cfg.merge_from_list(args.opts)

    cfg.freeze()

    return cfg


def main(args):
    cfg = setup_cfg(args)
    if cfg.SEED >= 0:
        print("Setting fixed seed: {}".format(cfg.SEED))
        set_random_seed(cfg.SEED)
    setup_logger(cfg.OUTPUT_DIR)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    print_args(args, cfg)
    print("Collecting env info ...")
    print("** System info **\n{}\n".format(collect_env_info()))

    trainer = build_trainer(cfg)

    if args.eval_only:
        trainer.load_model(args.model_dir, epoch=args.load_epoch)
        trainer.test()
        return

    if not args.no_train:
        trainer.train()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="", help="path to dataset")
    parser.add_argument("--output-dir", type=str, default="", help="output directory")
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="checkpoint directory (from which the training resumes)",
    )
    parser.add_argument(
        "--seed", type=int, default=-1, help="only positive value enables a fixed seed"
    )
    parser.add_argument(
        "--source-domains", type=str, nargs="+", help="source domains for DA/DG"
    )
    parser.add_argument(
        "--target-domains", type=str, nargs="+", help="target domains for DA/DG"
    )
    parser.add_argument(
        "--transforms", type=str, nargs="+", help="data augmentation methods"
    )
    parser.add_argument(
        "--config-file", type=str, default="", help="path to config file"
    )
    parser.add_argument(
        "--dataset-config-file",
        type=str,
        default="",
        help="path to config file for dataset setup",
    )
    parser.add_argument("--trainer", type=str, default="", help="name of trainer")
    parser.add_argument("--backbone", type=str, default="", help="name of CNN backbone")
    parser.add_argument("--head", type=str, default="", help="name of head")
    parser.add_argument("--eval-only", action="store_true", help="evaluation only")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="",
        help="load model from this directory for eval-only mode",
    )
    parser.add_argument(
        "--train-epoch", type=int, help="the max epoch for training model"
    )
    parser.add_argument(
        "--load-epoch", type=int, help="load model weights at this epoch for evaluation"
    )
    parser.add_argument(
        "--no-train", action="store_true", help="do not call trainer.train()"
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="modify config options using the command-line",
    )
    args = parser.parse_args()
    main(args)
