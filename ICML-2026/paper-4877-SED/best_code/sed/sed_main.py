import os

from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.cli import LightningCLI

from sed.models.callbacks.sed_logging import (SparseCaloImageLogger,
                                              SparseImageLogger,
                                              SparseScrnaLogger)
from sed.models.callbacks.weight_averaging import EMAWeightAveraging


def link_arguments(parser):
    parser.add_argument(
        "-u_name",
        "--unet_name",
        type=str,
        default="default_name",
        help="name of experiment",
    )


class SedCLI(LightningCLI):

    def add_train_args(self, parser):
        parser.add_argument("-n", "--name", type=str, default="default_name")
        parser.add_argument("-d", "--debug", type=bool, default=False)
        parser.add_argument("-se", "--sample_every", type=int, default=10000)

    def add_arguments_to_parser(self, parser):
        self.add_train_args(parser)
        parser.link_arguments("data.init_args.input_mode", "model.init_args.input_mode")
        parser.link_arguments("data.init_args.image_size", "model.init_args.image_size")
        link_arguments(parser)

    def before_instantiate_classes(self):
        try:
            if hasattr(self.config.model.init_args, 'vae_dir'):
                vae_path = self.config.model.init_args.vae_dir
                parts = vae_path.split("/")
                tag = parts[-3] if len(parts) >= 3 else 'vae'
                self.config.name = f"{self.config.name}_{tag}_{self.config.unet_name}"
        except Exception:
            pass

        super().before_instantiate_classes()

        root_dir = getattr(self.config.trainer, 'default_root_dir', None) or '/repo/output'
        self.logdir = getattr(self, 'logdir', None) or root_dir
        self.sampled_dir = getattr(self, 'sampled_dir', None) or os.path.join(self.logdir, 'sampled')
        os.makedirs(self.sampled_dir, exist_ok=True)

        self.config.trainer.val_check_interval = self.config.sample_every
        self.config.trainer.check_val_every_n_epoch = None

        self.reconstr_dir = os.path.join(self.logdir, "reconstructed")
        os.makedirs(self.reconstr_dir, exist_ok=True)

    def after_instantiate_classes(self):
        ema_callback = EMAWeightAveraging(10, 0.9999)
        self.trainer.callbacks.append(ema_callback)
        self.trainer.ema_callback = ema_callback

        if self.model.vae.input_mode == 'image':
            intermediate_logger = SparseImageLogger(
                batch_size=self.datamodule.batch_size,
                sample_every=self.config.sample_every,
                sampled_dir=self.sampled_dir)
        elif self.model.vae.input_mode == 'scrna':
            intermediate_logger = SparseScrnaLogger(
                batch_size=self.datamodule.batch_size,
                sample_every=self.config.sample_every,
                sampled_dir=self.sampled_dir)
        elif self.model.vae.input_mode == 'calo_image':
            intermediate_logger = SparseCaloImageLogger(
                batch_size=self.datamodule.batch_size,
                sample_every=self.config.sample_every,
                sampled_dir=self.sampled_dir)
        self.trainer.callbacks.append(intermediate_logger)

        lr_monitor = LearningRateMonitor(logging_interval='step')
        self.trainer.callbacks.append(lr_monitor)

def cli_main():
    cli = SedCLI(run=False)
    cli.trainer.fit(cli.model, cli.datamodule)

if __name__ == "__main__":
    cli_main()
