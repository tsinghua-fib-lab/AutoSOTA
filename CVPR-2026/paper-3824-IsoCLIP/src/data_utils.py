import sys
from pathlib import Path
from typing import Union, List, Dict, Callable

import torch
from torch.utils.data import Dataset

from datasets import *

PROJECT_ROOT = Path(__file__).absolute().parents[1].absolute()
sys.path.insert(1, str(PROJECT_ROOT))


def get_dataset(dataroot: Union[str, Path], dataset_name: str, split: str, preprocess: Callable, **kwargs) -> Dataset:
    """
    This function loads the dataset based on the dataset name provided.

    Args:
        dataroot (Union[str, Path]): The path to the dataset root directory.
        dataset_name (str): The name of the dataset to load.
        split (str): The data split to use ('train', 'test', 'query', 'gallery', etc.).
        preprocess (Callable): The preprocessing function to apply to the dataset.

    Returns:
        Dataset: The dataset object for the given dataset name and split.

    Raises:
        ValueError: If the dataset name is not recognized.
    """
    dataset_name = dataset_name.lower()
    dataroot = Path(dataroot)

    if dataset_name == 'roxford5k':
        dataset = ROxfordRParisDataset(dataroot, 'roxford5k', split, preprocess=preprocess)
    elif dataset_name == 'rparis6k':
        dataset = ROxfordRParisDataset(dataroot, 'rparis6k', split, preprocess=preprocess)
    elif dataset_name == 'cub2011':
        dataset = CUB(dataroot, split, preprocess=preprocess)
    elif dataset_name == 'cub_text':
        dataset = CUBTextDataset()
    elif dataset_name == 'sop':
        dataset = StanfordOnlineProducts(dataroot, split, preprocess=preprocess)
    elif dataset_name == 'stanford_cars':
        dataset = StanfordCars(dataroot, split, preprocess=preprocess)
    elif dataset_name == 'oxford_pets':
        dataset = OxfordPets(dataroot, split, preprocess=preprocess)
    elif dataset_name == 'oxford_flowers':
        dataset = OxfordFlowers(dataroot, split, preprocess=preprocess)
    elif dataset_name == 'fgvc_aircraft':
        dataset = FGVCAircraft(dataroot, split, preprocess=preprocess)
    elif dataset_name == 'dtd':
        dataset = DescribableTextures(dataroot, split, preprocess=preprocess)
    elif dataset_name == 'eurosat':
        dataset = EuroSAT(dataroot, split, preprocess=preprocess)
    elif dataset_name == 'food101':
        dataset = Food101(dataroot, split, preprocess=preprocess)
    elif dataset_name == 'sun397':
        dataset = SUN397(dataroot, split, preprocess=preprocess)
    elif dataset_name == 'caltech101':
        dataset = Caltech101(dataroot, split, preprocess=preprocess)
    elif dataset_name == 'ucf101':
        dataset = UCF101(dataroot, split, preprocess=preprocess)
    elif dataset_name == 'imagenet':
        dataset = ImageNet(dataroot, split, preprocess=preprocess)
    elif dataset_name == 'coco_text':
        dataset = CocoTextDataset(dataroot, split)
    elif dataset_name == 'flickr30k_text':
        dataset = Flickr30KTextDataset(dataroot, split)
    elif dataset_name == 'nocaps_text':
        dataset = NoCapsTextDataset(dataroot, split)
    elif dataset_name == 'imdb_text':
        dataset = IMDBTextDataset(dataroot, split)
    elif dataset_name == 'newsgroup_text':
        dataset = NewsGroupDataset(split)
    elif dataset_name in ['nanoclimatefever', 'nanodbpedia', 'nanofever',
                          'nanonfcorpus', 'nanonq', 'nanoscidocs',
                          'nanoscifact']:
        dataset = NanoBEIRDataset(split, dataset_name)
    elif dataset_name == 'coco':
        dataset = CocoDataset(dataroot, split, preprocess=preprocess, return_image=kwargs.get('return_image', True))
    elif dataset_name == 'modified_coco':
        dataset = ModifiedCocoDataset(dataroot, split, preprocess=preprocess,
                                      return_image=kwargs.get('return_image', True))
    elif dataset_name == 'flickr30k':
        dataset = Flickr30KDataset(dataroot, split, preprocess=preprocess,
                                   return_image=kwargs.get('return_image', True))
    elif dataset_name == 'inaturalist2021':
        dataset = INaturalist2021(dataroot, split, preprocess=preprocess)
    elif dataset_name == 'places365':
        dataset = Places365(dataroot, split, preprocess=preprocess)
    else:
        raise ValueError(f"Dataset {dataset_name} not recognized")
    return dataset


def collate_fn(batch: List[Union[torch.Tensor, None]]) -> torch.Tensor:
    """
    Function to filter out None values from a batch when using torch DataLoader.

    Args:
        batch (List[Union[torch.Tensor, None]]): The input batch containing tensors and possibly None values.

    Returns:
        torch.Tensor: The batch with None values removed.
    """
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


# Definition of the dataset splits for retrieval and classification
RETRIEVAL_SPLTS: Dict[str, Dict[str, str]] = {
    # IMAGE DATASETS
    "cub2011": {'query': 'all', 'gallery': 'all'},
    "cub_text": {'query': 'NO_SPLIT', 'gallery': 'NO_SPLIT'},
    "roxford5k": {'query': 'query', 'gallery': 'gallery'},
    "rparis6k": {'query': 'query', 'gallery': 'gallery'},
    "sop": {'query': 'test', 'gallery': 'test'},
    "inaturalist2021": {'query': 'train', 'gallery': 'train'},
    "places365": {'query': 'val', 'gallery': 'val'},
    "caltech101": {'query': 'test', 'gallery': 'train'},
    "dtd": {'query': 'test', 'gallery': 'train'},
    "eurosat": {'query': 'test', 'gallery': 'train'},
    "fgvc_aircraft": {'query': 'test', 'gallery': 'train'},
    "food101": {'query': 'test', 'gallery': 'train'},
    "imagenet": {'query': 'test', 'gallery': 'train'},
    "oxford_flowers": {'query': 'test', 'gallery': 'train'},
    "oxford_pets": {'query': 'test', 'gallery': 'train'},
    "stanford_cars": {'query': 'test', 'gallery': 'train'},
    "sun397": {'query': 'test', 'gallery': 'train'},
    "ucf101": {'query': 'test', 'gallery': 'train'},
    # TEXT DATASETS
    "coco_text": {'query': 'test_query', 'gallery': 'test_gallery'},
    "flickr30k_text": {'query': 'val_query', 'gallery': 'val_gallery'},
    "imdb_text": {'query': 'query', 'gallery': 'all'},
    "nanoclimatefever": {'query': 'query', 'gallery': 'gallery'},
    "nanodbpedia": {'query': 'query', 'gallery': 'gallery'},
    "nanofever": {'query': 'query', 'gallery': 'gallery'},
    "nanonfcorpus": {'query': 'query', 'gallery': 'gallery'},
    "nanonq": {'query': 'query', 'gallery': 'gallery'},
    "nanoscidocs": {'query': 'query', 'gallery': 'gallery'},
    "nanoscifact": {'query': 'query', 'gallery': 'gallery'},
    "newsgroup_text": {'query': 'query', 'gallery': 'test'},
    "nocaps_text": {'query': 'val_query', 'gallery': 'val_gallery'},
    # IMAGE-TEXT DATASETS
    "coco":  {'query': 'test', 'gallery': 'test'}, 
    "modified_coco":  {'query': 'test', 'gallery': 'test'}, 
    "flickr30k":  {'query': 'test', 'gallery': 'test'},
    
}

CLASSIFICATION_SPLTS: Dict[str, Dict[str, str]] = {
    "caltech101": {'split': 'test'},
    "dtd": {'split': 'test'},
    "eurosat": {'split': 'test'},
    "fgvc_aircraft": {'split': 'test'},
    "food101": {'split': 'test'},
    "imagenet": {'split': 'val'},
    "oxford_flowers": {'split': 'test'},
    "oxford_pets": {'split': 'test'},
    "stanford_cars": {'split': 'test'},
    "sun397": {'split': 'test'},
    "ucf101": {'split': 'test'},
}
