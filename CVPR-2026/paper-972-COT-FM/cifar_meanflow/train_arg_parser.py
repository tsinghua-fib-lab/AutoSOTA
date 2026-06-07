# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import logging


logger = logging.getLogger(__name__)


def get_args_parser():
    parser = argparse.ArgumentParser("Image dataset training", add_help=False)

    # Optimizer parameters
    parser.add_argument("--batch_size", default=64, type=int, help="Batch size per GPU (effective batch size is batch_size * # gpus")
    parser.add_argument("--epochs", default=4000, type=int)
    parser.add_argument("--lr", default=0.0006, type=float, help="learning rate (absolute lr)")
    parser.add_argument("--optimizer_betas", default=[0.9, 0.999], nargs="+", type=float, help="beta1 and beta2 for Adam optimizer")
    parser.add_argument("--warmup_epochs", default=200, type=int, help="Number of warmup epochs.")
    parser.add_argument("--dropout", default=0.2, type=float, help="Dropout rate.")

    parser.add_argument("--ema_decay", default=0.99995, type=float, help="Exponential moving average decay rate.")
    parser.add_argument("--ema_decays", default=[0.99995], nargs="+", type=float, help="Extra EMA decay rates.")

    # Dataset parameters
    parser.add_argument("--dataset", default='cifar10', type=str, choices=['cifar10', 'mnist'], help="Dataset to use.")
    parser.add_argument("--data_path", default="./data", type=str, help="data root folder with train, val and test subfolders")

    parser.add_argument("--output_dir", default="./output_dir", help="path where to save, empty for no saving")
    parser.add_argument("--fid_samples", default=50000, type=int, help="number of synthetic samples for FID evaluations")
    parser.add_argument("--device", default="cuda", help="device to use for training / testing")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--resume", default="", help="resume from checkpoint")

    parser.add_argument("--start_epoch", default=0, type=int, metavar="N", help="start epoch (used when resumed from checkpoint)")
    parser.add_argument("--eval_only", action="store_true", help="No training, only run evaluation")
    parser.add_argument("--eval_frequency", default=50, type=int, help="Frequency (in number of epochs) for running FID evaluation. -1 to never run evaluation.")
    parser.add_argument("--compute_fid", action="store_true", help="Whether to compute FID in the evaluation loop. When disabled, the evaluation loop still runs and saves snapshots, but skips the FID computation.")
    parser.add_argument("--save_fid_samples", action="store_true", help="Save all samples generated for FID computation.")
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument("--pin_mem", action="store_true", help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.")
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)
    parser.add_argument("--log_per_step", default=100, type=int, metavar="N", help="Log training stats every N iterations",)

    # Distributed training parameters
    parser.add_argument("--world_size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument("--dist_url", default="env://", help="url used to set up distributed training")

    # MeanFlow specific parameters
    parser.add_argument("--ratio", default=0.75, type=float, help="Probability of sampling r (or h) DIFFERENT from t")  

    parser.add_argument("--tr_sampler", default="v1", type=str, choices=["v0", "v1"], help="Joint (t, r) sampler version.")

    parser.add_argument("--P_mean_t", default=-0.6, type=float, help="P_mean_t of lognormal sampler.")
    parser.add_argument("--P_std_t", default=1.6, type=float, help="P_std_t of lognormal sampler.")
    parser.add_argument("--P_mean_r", default=-4.0, type=float, help="P_mean_r of lognormal sampler.")
    parser.add_argument("--P_std_r", default=1.6, type=float, help="P_std_r of lognormal sampler.")
    
    parser.add_argument("--norm_p", default=0.75, type=float, help="Norm power for adaptive weight.")
    parser.add_argument("--norm_eps", default=1e-3, type=float, help="Small constant for adaptive weight division.")
    parser.add_argument("--arch", default="unet", type=str, choices=["unet",], help="Architecture to use.")
    parser.add_argument("--use_edm_aug", action="store_true", dest="use_edm_aug", default=False, help="Enable EDM augmentation with augment labels as conditions.")

    # Debugging settings
    parser.add_argument("--test_run", action="store_true", help="Only run one batch of training and evaluation.")
    parser.add_argument("--not_compile", action="store_false", dest="compile", default=True, help="Disable compilation.")

    parser.add_argument("--generate", action="store_true", help="Generate samples using the trained model.", default=False)
    parser.add_argument("--train_flow", action="store_true", help="Train the flow model.", default=False)


    parser.add_argument("--single_class_id", default=-1, type=int, help="If >=0, use only one class for debugging.")
    parser.add_argument("--accum_steps", default=1, type=int, help="Number of gradient accumulation steps.")
    return parser
