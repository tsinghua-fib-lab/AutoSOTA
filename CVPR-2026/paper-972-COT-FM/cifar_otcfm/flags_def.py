from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_string("model", "AlignFlow", help="flow matching model type")
flags.DEFINE_string("output_dir", "./results/", help="output_directory")
flags.DEFINE_string("wandb_name", 'debug', help="wandb experiment name")

# UNet
flags.DEFINE_integer("num_channel", 128, help="base channel of UNet")

# Training
flags.DEFINE_float("lr", 2e-4, help="target learning rate")  # TRY 2e-4
flags.DEFINE_float("grad_clip", 1.0, help="gradient norm clipping")
flags.DEFINE_integer(
    "total_steps", 400001, help="total training steps"
)  # Lipman et al uses 400k but double batch size
flags.DEFINE_integer("warmup", 5000, help="learning rate warmup")
flags.DEFINE_integer("batch_size", 128, help="batch size")  # Lipman et al uses 128
flags.DEFINE_integer("num_workers", 8, help="workers of Dataloader")
flags.DEFINE_float("ema_decay", 0.9999, help="ema decay rate")
flags.DEFINE_bool("parallel", False, help="multi gpu training")
flags.DEFINE_float("sigma", 0.0, help="do not tune this. This value should always be 0.")
flags.DEFINE_integer("seed", None, help="random seed")
flags.DEFINE_integer("resume_step", 0, help="step to resume training from")
flags.DEFINE_string("space", "v", help="data space: v or x")
flags.DEFINE_string("weighting", "uniform", help="weighting scheme: uniform or adaptive")
flags.DEFINE_float("adaptive_p", 0.75, help="power for adaptive weighting")

# for sdot only
flags.DEFINE_float("entropic_reg", 0, help="entropic regularization strength. We recommand using the same value as when calculating the dual weigth")
flags.DEFINE_integer("resample_interval", 10, help="number of epoch for each rebalance")

flags.DEFINE_integer(
# Evaluation
    "save_step",
    20000,
    help="frequency of saving checkpoints, 0 to disable during training",
)