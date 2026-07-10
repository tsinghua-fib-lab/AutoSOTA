import torch
import torchvision
import torchvision.transforms.v2 as v2
from custom_callbacks import ImageLabelVisualizationCallback
from lightning import LightningDataModule
from torch.utils.data import DataLoader

FINAL_TRAINING_RUN = False


def set_FINAL_TRAINING_RUN(value: bool):
    global FINAL_TRAINING_RUN
    FINAL_TRAINING_RUN = value


class TorchvisionDataModule(LightningDataModule):
    """Abstract class to easily turn a torchvision dataset into a Lightning DataModule"""

    prediction_callback = ImageLabelVisualizationCallback
    known_shapes: dict[str, tuple[int, ...]]
    transforms: list[v2.Transform]
    dataset: type = torchvision.datasets.VisionDataset
    train_transforms: list[v2.Transform] = []
    dl_kwargs: dict = {}

    @property
    def dataset_name(self):
        return self.dataset.__name__

    def __init__(self, batch_size: int):
        super().__init__()
        self.batch_size = batch_size

        # Add batch size to known_shapes
        self.known_shapes = {
            attr: (batch_size,) + shape for attr, shape in self.known_shapes.items()
        }

    def setup(self, stage: str):
        if stage == "fit":
            train_set = self.dataset(
                root="../data",
                train=True,
                download=True,
                transform=v2.Compose([*self.transforms, *self.train_transforms]),
            )
            self.num_classes = len(train_set.classes)

            if FINAL_TRAINING_RUN:
                self.train_set = train_set
                self.val_set = None
            else:
                self.train_set, self.val_set = torch.utils.data.random_split(
                    train_set, [0.9, 0.1], generator=torch.Generator().manual_seed(42)
                )

        elif stage == "test" or stage == "predict":
            self.test_set = self.dataset(
                root="../data",
                train=False,
                download=True,
                transform=v2.Compose(self.transforms),
            )

    def on_after_batch_transfer(self, batch, dataloader_idx):
        """
        Transforms batch after being placed on device
        Same as the 'transform' argument in torchvision datasets, but batched.

        Transforms:
        * Normalize image to [0, 1]
        * Class label y as one-hot
        """
        img, y = batch

        return {
            "img": img,
            "y": torch.nn.functional.one_hot(y, num_classes=self.num_classes).to(torch.float32),
        }

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            **self.dl_kwargs,
        )

    def val_dataloader(self):
        return (
            DataLoader(
                self.val_set,
                batch_size=self.batch_size,
                shuffle=False,
                drop_last=True,
                **self.dl_kwargs,
            )
            if not FINAL_TRAINING_RUN
            else []  # Empty validation set
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=True,
            **self.dl_kwargs,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            **self.dl_kwargs,
        )

    def metrics(self, node_dict, batch, prefix: str = ""):
        """Returns classification accuracy. Used at val/test time"""
        y = batch["y"]
        y_pred = node_dict["y"]

        class_pred = y_pred.argmax(1, keepdim=True)
        class_target = y.argmax(1, keepdim=True)
        accuracy = (class_pred == class_target).float().mean()

        return {f"{prefix}acc": accuracy}
