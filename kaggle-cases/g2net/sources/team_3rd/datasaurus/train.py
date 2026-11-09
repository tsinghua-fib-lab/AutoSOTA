import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import StochasticWeightAveraging
from pytorch_lightning.plugins import DDPPlugin

from src.config import MODEL_CACHE, OUTPUT_PATH
from src.datasets import GWDataModule
from src.models import GWModel, GWDenoisingModel
from src.utils import prepare_args, prepare_loggers_and_callbacks, resume_helper

torch.hub.set_dir(MODEL_CACHE)


def run_fold(fold: int, args):
    pl.seed_everything(args.seed + fold)
    resume, run_id = resume_helper(args)

    monitor_list = [("loss/valid", "min", "loss"), ("auc", "max", "auc")]
    loggers, callbacks = prepare_loggers_and_callbacks(
        args.timestamp,
        args.encoder,
        fold,
        monitors=monitor_list,
        tensorboard=args.logging,
        wandb=args.logging,
        patience=None,
        run_id=run_id,
    )

    # swa = StochasticWeightAveraging(swa_epoch_start=0.5)
    # callbacks.append(swa)

    if args.denoising:
        model = GWDenoisingModel(**args.__dict__)
    elif resume:
        kwargs = args.__dict__
        del kwargs["checkpoint"]
        model = GWModel.load_from_checkpoint(resume, **kwargs)
    else:
        model = GWModel(**args.__dict__)

    trainer = pl.Trainer().from_argparse_args(
        args,
        logger=loggers,
        callbacks=callbacks,
        resume_from_checkpoint=resume,
        plugins=DDPPlugin(find_unused_parameters=False),
        # fast_dev_run=True,
        # auto_lr_find=True,
    )

    dm = GWDataModule().from_argparse_args(args)
    dm.setup("fit", fold)

    # trainer.tune(model, datamodule=dm)  # Use with auto_lr_find
    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    args = prepare_args()
    run_fold(args.fold - 1, args)
