import datetime
import os

from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.cli import LightningCLI

from sed.sed_main import SedCLI
from sed.models.callbacks.svae_logging import (CaloImageLogger, ImageLogger,
                                               scrnaLogger)


def link_arguments(parser):
    parser.link_arguments("data.init_args.data_dimensions", "model.init_args.data_dimensions")
    parser.link_arguments("data.init_args.input_mode",
                          "model.init_args.input_mode")
    parser.add_argument(
        "-nl",
        "--no_logging",
        type=bool,
        default=False,
        help="no_logging",
    )


class SvaeCLI(SedCLI):
    def add_arguments_to_parser(self, parser):
        super().add_train_args(parser)
        link_arguments(parser)

    def before_instantiate_classes(self):
        super().before_instantiate_classes()
        self.config.trainer.val_check_interval = self.config.sample_every
        self.config.trainer.check_val_every_n_epoch = None

    def after_instantiate_classes(self):
        if not self.config.no_logging:
            if self.config.model.init_args.input_mode == 'image':
                intermediate_logger = ImageLogger(
                    batch_size=self.datamodule.batch_size, sample_every=self.config.sample_every, sampled_dir=self.sampled_dir, reconstr_dir=self.reconstr_dir)
            elif self.config.model.init_args.input_mode == 'scrna':
                intermediate_logger = scrnaLogger(
                    batch_size=self.datamodule.batch_size, sample_every=self.config.sample_every, sampled_dir=self.sampled_dir, reconstr_dir=self.reconstr_dir)
            elif self.config.model.init_args.input_mode == 'calo_image':
                intermediate_logger = CaloImageLogger(
                    batch_size=self.datamodule.batch_size, sample_every=self.config.sample_every, sampled_dir=self.sampled_dir, reconstr_dir=self.reconstr_dir)
            self.trainer.callbacks.append(intermediate_logger)
        lr_monitor = LearningRateMonitor(logging_interval='step')
        self.trainer.callbacks.append(lr_monitor)
        return


def cli_main():
    cli = SvaeCLI(run=False)
    cli.trainer.fit(cli.model, cli.datamodule)


if __name__ == "__main__":
    cli_main()
