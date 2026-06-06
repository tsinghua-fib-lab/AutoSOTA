from typing import Any, Dict, Optional, Tuple

import torch
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from .custom_dataset import CustomDataset


class DataModule(LightningDataModule):
    """`LightningDataModule` for the ETTh1,ETTh2,ETTm1,ETTm2,weather,traffic,electricity, 7 datasets in total

    A `LightningDataModule` implements 7 key methods:

    ```python
        def prepare_data(self):
        # Things to do on 1 GPU/TPU (not on every GPU/TPU in DDP).
        # Download data, pre-process, split, save to disk, etc...

        def setup(self, stage):
        # Things to do on every process in DDP.
        # Load data, set variables, etc...

        def train_dataloader(self):
        # return train dataloader

        def val_dataloader(self):
        # return validation dataloader

        def test_dataloader(self):
        # return test dataloader

        def predict_dataloader(self):
        # return predict dataloader

        def teardown(self, stage):
        # Called on every process in DDP.
        # Clean up after fit or test.
    ```

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(
            self,
            name: str = "ETTh1",
            data_dir: str = "data/",
            task: str = "forecasting",
            scale: bool = True,
            train_val_test_split: Tuple[float, float, float] = (0.6, 0.2, 0.2),
            input_size: int = 96,
            output_size: int = 96,
            num_channels: int = 7,
            cycle_len: int = 24, 
            batch_size: int = 64,
            num_workers: int = 0,
            pin_memory: bool = False,
    ) -> None:
        """Initialize a `MNISTDataModule`.
        :param data_dir: The data directory. Defaults to `"data/"`.
        :param train_val_test_split: The train, validation and test split. Defaults to `(55_000, 5_000, 10_000)`.
        :param batch_size: The batch size. Defaults to `64`.
        :param num_workers: The number of workers. Defaults to `0`.
        :param pin_memory: Whether to pin memory. Defaults to `False`.
        """
        super().__init__()
        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        self.data_train = CustomDataset(self.hparams.name,
                                        self.hparams.data_dir,
                                        self.hparams.task,
                                        self.hparams.scale,
                                        self.hparams.train_val_test_split,
                                        self.hparams.input_size,
                                        self.hparams.output_size,
                                        self.hparams.cycle_len, 
                                        self.hparams.num_channels,
                                        "train")
        self.data_val = CustomDataset(self.hparams.name,
                                      self.hparams.data_dir, 
                                      self.hparams.task,
                                      self.hparams.scale,
                                      self.hparams.train_val_test_split,
                                      self.hparams.input_size,
                                      self.hparams.output_size,
                                      self.hparams.cycle_len, 
                                      self.hparams.num_channels,
                                      "val")
        self.data_test = CustomDataset(self.hparams.name,
                                       self.hparams.data_dir,
                                       self.hparams.task,
                                       self.hparams.scale,
                                       self.hparams.train_val_test_split,
                                       self.hparams.input_size,
                                       self.hparams.output_size,
                                       self.hparams.cycle_len, 
                                       self.hparams.num_channels,
                                       "test")

        self.scaler = self.data_train.scaler

        self.batch_size_per_device = batch_size

    def prepare_data(self) -> None:
        """Download data if needed. Lightning ensures that `self.prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading logic within. In
        case of multi-node training, the execution of this hook depends upon
        `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        """
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            drop_last=False,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return {'val': DataLoader(dataset=self.data_val,
                                  batch_size=self.batch_size_per_device,
                                  num_workers=self.hparams.num_workers,
                                  pin_memory=self.hparams.pin_memory,
                                  shuffle=False,
                                  drop_last=False,),
                'test': DataLoader(dataset=self.data_test,
                                  batch_size=self.batch_size_per_device,
                                  num_workers=self.hparams.num_workers,
                                  pin_memory=self.hparams.pin_memory,
                                  shuffle=False,
                                  drop_last=False,)
                }

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass


if __name__ == "__main__":
    _ = DataModule()
