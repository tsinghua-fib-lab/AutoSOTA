"""We thank the authors of KDEformer for providing this source code:
- https://github.com/majid-daliri/kdeformer/blob/main/imagenet.py
- License: MIT
- Copyright (c) 2025 KDEFormer
- Paper reference:
    Amir Zandieh, Insu Han, Majid Daliri, Amin Karbasi.
    KDEformer: Accelerating Transformers via Kernel Density Estimation.
    https://arxiv.org/abs/2302.02451
"""

# Adapted from https://github.com/PyTorchLightning/lightning-bolts/blob/master/pl_bolts/datamodules/imagenet_datamodule.py
import os
from pathlib import Path
from typing import Any, List, Union, Callable, Optional

from torch.utils.data import Dataset, DataLoader

from pytorch_lightning import LightningDataModule
from pl_bolts.transforms.dataset_normalizations import imagenet_normalization

from torchvision import transforms
from torchvision.datasets import ImageFolder
import timm


class ImagenetDataModule(LightningDataModule):
    """.. figure:: https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/wp-content/uploads/2017/08/
        Sample-of-Images-from-the-ImageNet-Dataset-used-in-the-ILSVRC-Challenge.png
        :width: 400
        :alt: Imagenet
    Specs:
        - 1000 classes
        - Each image is (3 x varies x varies) (here we default to 3 x 224 x 224)
    Imagenet train, val and test dataloaders.
    The train set is the imagenet train.
    The val set is taken from the train set with `num_imgs_per_val_class` images per class.
    For example if `num_imgs_per_val_class=2` then there will be 2,000 images in the validation set.
    The test set is the official imagenet validation set.
     Example::
        from pl_bolts.datamodules import ImagenetDataModule
        dm = ImagenetDataModule(IMAGENET_PATH)
        model = LitModel()
        Trainer().fit(model, datamodule=dm)
    """

    name = "imagenet"

    def __init__(
        self,
        data_dir: str,
        cache_dir: Optional[str] = None,
        image_size: int = 224,
        num_workers: int = 0,
        batch_size: int = 32,
        shuffle: bool = True,
        pin_memory: bool = True,
        drop_last: bool = False,
        dali: Optional[str] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Args:
        data_dir: path to the imagenet dataset file
        num_imgs_per_val_class: how many images per class for the validation set
        image_size: final image size
        num_workers: how many data workers
        batch_size: batch_size
        shuffle: If true shuffles the data every epoch
        pin_memory: If true, the data loader will copy Tensors into CUDA pinned memory before
                    returning them
        drop_last: If true drops the last incomplete batch

        """
        batch_size_eval = kwargs.pop("batch_size_eval")
        super().__init__(*args, **kwargs)
        self.image_size = image_size
        self.dims = (3, self.image_size, self.image_size)
        self.data_dir = Path(data_dir).expanduser()
        self.cache_dir = cache_dir
        self.use_archive_dataset = (
            self.data_dir.suffix == ".tar" or self.data_dir.suffix == ".zip"
        )
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        assert dali in [None, "cpu", "gpu"]
        if dali is not None and self.use_archive_dataset:
            raise NotImplementedError("dali is not compatible with archive dataset")
        self.dali = dali

    @property
    def num_classes(self) -> int:
        """Return:
        1000

        """
        return 1000

    def _verify_splits(self, data_dir: str, split: str) -> None:
        dirs = os.listdir(data_dir)

        if split not in dirs:
            raise FileNotFoundError(
                f"a {split} Imagenet split was not found in {data_dir},"
                f" make sure the folder contains a subfolder named {split}"
            )

    def prepare_data(self) -> None:
        """This method already assumes you have imagenet2012 downloaded. It validates the data using the meta.bin.
        .. warning:: Please download imagenet on your own first.
        """
        if not self.use_archive_dataset:
            # self._verify_splits(self.data_dir, "train")
            self._verify_splits(self.data_dir, "val")
        else:
            if not self.data_dir.is_file():
                raise FileNotFoundError(
                    f"""Archive file {str(self.data_dir)} not found."""
                )

    def setup(self, stage: Optional[str] = None) -> None:
        """Creates train, val, and test dataset."""
        if self.dali is not None:
            return
        if stage == "fit" or stage is None:
            # train_transforms = (self.train_transform() if self.train_transforms is None
            #                     else self.train_transforms)
            # val_transforms = (self.val_transform() if self.val_transforms is None
            #                   else self.val_transforms)
            train_transforms = self.train_transform()
            val_transforms = self.val_transform()
            if not self.use_archive_dataset:
                try:
                    self.dataset_train = ImageFolder(
                        self.data_dir / "train", transform=train_transforms
                    )
                except:
                    self.dataset_train = None
                self.dataset_val = ImageFolder(
                    self.data_dir / "val", transform=val_transforms
                )
            else:
                from archive_imagefolder import ArchiveImageFolder

                self.dataset_train = ArchiveImageFolder(
                    str(self.data_dir),
                    cache_dir=self.cache_dir,
                    root_in_archive="train",
                    transform=train_transforms,
                )
                self.dataset_val = ArchiveImageFolder(
                    str(self.data_dir),
                    cache_dir=self.cache_dir,
                    root_in_archive="val",
                    transform=val_transforms,
                )

        if stage == "test" or stage is None:
            # test_transforms = (self.val_transform() if self.test_transforms is None
            #                    else self.test_transforms)
            test_transforms = self.val_transform()
            if not self.use_archive_dataset:
                self.dataset_test = ImageFolder(
                    self.data_dir / "val", transform=test_transforms
                )
            else:
                self.dataset_test = ArchiveImageFolder(
                    str(self.data_dir),
                    cache_dir=self.cache_dir,
                    root_in_archive="val",
                    transform=test_transforms,
                )

    def train_transform(self) -> Callable:
        """The standard imagenet transforms.
        .. code-block:: python
            transforms.Compose([
                transforms.RandomResizedCrop(self.image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])
        """
        preprocessing = transforms.Compose(
            [
                transforms.RandomResizedCrop(self.image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                imagenet_normalization(),
            ]
        )

        return preprocessing

    def val_transform(self) -> Callable:
        """The standard imagenet transforms for validation.
        .. code-block:: python
            transforms.Compose([
                transforms.Resize(self.image_size + 32),
                transforms.CenterCrop(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])
        """
        preprocessing = transforms.Compose(
            [
                transforms.Resize(self.image_size + 32),
                transforms.CenterCrop(self.image_size),
                transforms.ToTensor(),
                imagenet_normalization(),
            ]
        )
        return preprocessing

    def train_dataloader(self, *args: Any, **kwargs: Any) -> DataLoader:
        """The train dataloader"""
        if self.dali is None:
            return self._data_loader(self.dataset_train, shuffle=self.shuffle)
        else:
            return self._dali_loader(is_train=True, shuffle=self.shuffle)

    def val_dataloader(
        self, *args: Any, **kwargs: Any
    ) -> Union[DataLoader, List[DataLoader]]:
        """The val dataloader"""
        if self.dali is None:
            return self._data_loader(self.dataset_val)
        else:
            return self._dali_loader(is_train=False)

    def test_dataloader(
        self, *args: Any, **kwargs: Any
    ) -> Union[DataLoader, List[DataLoader]]:
        """The test dataloader"""
        if self.dali is None:
            return self._data_loader(self.dataset_test)
        else:
            return self._dali_loader(is_train=False)

    def _data_loader(self, dataset: Dataset, shuffle: bool = False) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
            # Spinning up worker is slow if we use archive dataset
            # When we don't use archive dataset, I get crashes if I don't set this to True
            # https://github.com/PyTorchLightning/pytorch-lightning/issues/4471
            # https://github.com/PyTorchLightning/pytorch-lightning/issues/8821
            # TD [2021-09-01] I think this bug is finally fixed in pytorch-lightning 1.4.5
            # https://github.com/PyTorchLightning/pytorch-lightning/pull/9239
            persistent_workers=True,
        )


def get_imagenet_datamodule(dataset_path, batch_size=16):
    """Helper for getting imagenet datamodule"""
    kwargs = {
        "data_dir": dataset_path,
        "shuffle": True,
        "batch_size": batch_size,
        "batch_size_eval": 8,
        "num_workers": 8,
        "pin_memory": True,
        "image_size": 224,
    }
    datamodule = ImagenetDataModule(**kwargs)
    datamodule.prepare_data()
    datamodule.setup()
    datamodule.dataset_val.transform = timm.data.create_transform(
        input_size=224, interpolation="bicubic", crop_pct=0.9
    )
    return datamodule
