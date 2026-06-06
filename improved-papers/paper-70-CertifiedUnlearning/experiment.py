import logging
import os

import jax
import wandb
import shutil

from src.data.datamodule import get_datamodule
from src.models.model import ModelFactory
from src.training.trainer import Trainer
from src.utils.utils import load_config, logger, parse_args

LOGS_DIR = "logs"
n_threads_str = "4"
os.environ["OMP_NUM_THREADS"] = n_threads_str
os.environ["OPENBLAS_NUM_THREADS"] = n_threads_str
os.environ["MKL_NUM_THREADS"] = n_threads_str
os.environ["VECLIB_MAXIMUM_THREADS"] = n_threads_str
os.environ["NUMEXPR_NUM_THREADS"] = n_threads_str
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


def main():
    print("JAX process: %d / %d", jax.process_index(), jax.process_count())
    print("JAX local devices: %r", jax.local_devices())

    args = parse_args()
    config = load_config(args.config)
    wandb.init(
        project=config["wandb"]["project"],
        entity=config["wandb"]["entity"],
        config=config,
        dir=config["wandb"]["log_dir"],
    )
    run_dir = os.path.join(LOGS_DIR, wandb.run.id)
    os.makedirs(run_dir, exist_ok=True)
    shutil.copy(args.config, os.path.join(run_dir, "config.yaml"))
    
    checkpoint_path = os.path.join(run_dir, "ckpt")
    os.makedirs(checkpoint_path, exist_ok=True)
    config["checkpoint_path"] = checkpoint_path
    if args.debug >= 1:
        logger.setLevel(level=logging.DEBUG)

    data_module = get_datamodule(config=config, generator=None)
    data_module.setup()

    model = ModelFactory.create_model(
        model_name=config["model"]["name"],
        num_classes=config["model"]["n_classes"],
    )
    key = jax.random.PRNGKey(config["global_seed"])

    trainer = Trainer(
        config=config,
        model=model,
        key=key,
        train_dataloader=data_module.train_dataloader(),
        val_dataloader=data_module.val_dataloader(),
        test_dataloader=data_module.test_dataloader(),
        forget_dataloader=data_module.forget_dataloader(),
        retain_dataloader=data_module.retain_dataloader(),
        use_pretrained=config["training"]["pretrained"],
        run_dir=run_dir,
    )

    state = trainer.fit()
    state = trainer.unlearn(state)
    trainer.test(state=state)


if __name__ == "__main__":
    main()
