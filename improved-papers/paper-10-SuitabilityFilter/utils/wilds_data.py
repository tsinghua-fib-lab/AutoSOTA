import torch
from examples.models.initializer import (
    initialize_bert_based_model,
    initialize_torchvision_model,
)
from examples.transforms import initialize_transform
from torch.utils.data import DataLoader, Dataset

from wilds import get_dataset

"""
WILDS is a benchmark of in-the-wild distribution shifts spanning diverse data modalities and applications, from tumor identification to wildlife monitoring to poverty mapping.
More information on the benchmark and datasets can be found at https://wilds.stanford.edu/.
"""


class ResConfig:
    def __init__(self, dataset_name):
        self.target_resolution = self.get_target_resolution(dataset_name)

    def get_target_resolution(self, dataset_name):
        if dataset_name == "iwildcam":
            return (448, 448)
        elif dataset_name == "fmow":
            return (224, 224)
        elif dataset_name == "rxrx1":
            return (256, 256)
        else:
            raise ValueError(f"Dataset {dataset_name} not supported")


class BertConfig:
    def __init__(self, dataset_name):
        if dataset_name == "civilcomments":
            self.model = "distilbert-base-uncased"
            self.max_token_length = 300
        elif dataset_name == "amazon":
            self.model = "distilbert-base-uncased"
            self.max_token_length = 512
        else:
            raise ValueError(f"Dataset {dataset_name} not supported")
        self.pretrained_model_path = None
        self.model_kwargs = {}


class WILDSDataset(Dataset):
    def __init__(self, dataset_name, split, root_dir, pre_filter={}):
        """
        dataset_name: the name of the dataset (string)
        split: the split of the dataset (string)
        root_dir: the root directory of the dataset (string)
        pre_filter: the filter to be applied to the data (dict)
        transform: the transformation to be applied to the data (torchvision transform)
        """
        dataset = get_dataset(dataset=dataset_name, download=False, root_dir=root_dir)

        if dataset_name in ["civilcomments", "amazon"]:
            config = BertConfig(dataset_name)
            transform = initialize_transform(
                transform_name="bert", config=config, dataset=dataset, is_training=False
            )
        elif dataset_name in ["iwildcam", "fmow"]:
            config = ResConfig(dataset_name)
            transform = initialize_transform(
                transform_name="image_base",
                config=config,
                dataset=dataset,
                is_training=False,
            )
        elif dataset_name == "rxrx1":
            config = ResConfig(dataset_name)
            transform = initialize_transform(
                transform_name="rxrx1",
                config=config,
                dataset=dataset,
                is_training=False,
            )

        self.dataset = dataset.get_subset(split, transform=transform)
        self.filtered_indices = list(range(len(self.dataset)))

        for key, value in pre_filter.items():
            # Filter the fmow dataset based on pre-filter
            if dataset_name == "fmow":
                if key == "region":
                    key_ind = 0
                    valid_regions = ["Asia", "Europe", "Africa", "Americas", "Oceania"]
                    if not isinstance(value, list):
                        value = [value]
                    for region in value:
                        if region not in valid_regions:
                            raise ValueError(f"Region {region} not supported")
                    val_inds = [valid_regions.index(region) for region in value]
                elif key == "year":
                    key_ind = 1
                    if not isinstance(value, list):
                        value = [value]
                    for year in value:
                        if year < 2002 or year > 2017:
                            raise ValueError("Year must be between 2002 and 2017")
                    val_inds = [year - 2002 for year in value]
                else:
                    raise ValueError(f"Filter property {key} not supported")
                self.filtered_indices = [
                    i
                    for i in self.filtered_indices
                    if self.dataset[i][2][key_ind] in val_inds
                ]

            elif dataset_name == "civilcomments":
                if key == "sensitive":
                    valid_sensitives = [
                        "male",
                        "female",
                        "LGBTQ",
                        "christian",
                        "muslim",
                        "other_religions",
                        "black",
                        "white",
                    ]
                    if not isinstance(value, list):
                        value = [value]
                    for sensitive in value:
                        if sensitive not in valid_sensitives:
                            raise ValueError(
                                f"Sensitive attribute {sensitive} not supported"
                            )
                    val_inds = [
                        valid_sensitives.index(sensitive) for sensitive in value
                    ]
                else:
                    raise ValueError(f"Filter property {key} not supported")
                self.filtered_indices = [
                    i
                    for i in self.filtered_indices
                    if any(self.dataset[i][2][val_ind] == 1 for val_ind in val_inds)
                ]

            elif dataset_name == "rxrx1":
                if key == "cell_type":
                    key_ind = 0
                    valid_cell_types = ["HEPG2", "HUVEC", "RPE", "U2OS"]
                    if not isinstance(value, list):
                        value = [value]
                    for cell_type in value:
                        if cell_type not in valid_cell_types:
                            raise ValueError(f"Cell type {cell_type} not supported")
                    val_inds = [
                        valid_cell_types.index(cell_type) for cell_type in value
                    ]
                else:
                    raise ValueError(f"Filter property {key} not supported")

                self.filtered_indices = [
                    i
                    for i in self.filtered_indices
                    if self.dataset[i][2][key_ind] in val_inds
                ]

            else:
                raise ValueError(f"Filtering not supported for dataset {dataset_name}")

    def __len__(self):
        return len(self.filtered_indices)

    def __getitem__(self, idx):
        return self.dataset[self.filtered_indices[idx]]


def get_wilds_dataset(
    dataset_name,
    root_dir,
    split,
    batch_size,
    shuffle,
    num_workers,
    pre_filter={},
    return_indices=False,
):
    """
    dataset_name: the name of the dataset (string)
    root_dir: the root directory of the dataset (string)
    split: the split of the dataset (string)
    batch_size: the batch size (int)
    shuffle: whether to shuffle the data (bool)
    num_workers: the number of workers (int)
    pre_filter: the filter to be applied to the data (dict)
    return_indices: whether to return the indices of the data
    """

    assert dataset_name in [
        "iwildcam",
        "fmow",
        "civilcomments",
        "rxrx1",
        "amazon",
    ], f"Dataset {dataset_name} not supported"
    assert split in [
        "val",
        "test",
        "id_val",
        "id_test",
    ], "Split must be (id_)val or (id_)test, this should be used for evaluation only"
    root_dir = f"{root_dir}data/"
    dataset = WILDSDataset(dataset_name, split, root_dir, pre_filter=pre_filter)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )
    if return_indices:
        return dataloader, dataset.filtered_indices
    return dataloader


def remove_prefix_from_state_dict(state_dict, prefix="model."):
    """
    Remove the prefix from the keys in state_dict if it exists.
    """
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith(prefix):
            new_state_dict[k[len(prefix) :]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict


def get_wilds_model(dataset_name, root_dir, algorithm, seed=0, model_type="last"):
    """
    dataset_name: the name of the dataset (string)
    root_dir: the root directory of the dataset (string)
    algorithm: the algorithm used to train the model: ERM, IRM, groupDRO (string)
    seed: the seed used to train the model (int)
    model_type: the model to be loaded: last or best (string)
    """

    if dataset_name == "iwildcam":
        model = initialize_torchvision_model("resnet50", d_out=182, pretrained=True)
    elif dataset_name == "fmow":
        model = initialize_torchvision_model("densenet121", d_out=62, pretrained=True)
    elif dataset_name == "civilcomments":
        config = BertConfig(dataset_name)
        model = initialize_bert_based_model(config, d_out=2)
    elif dataset_name == "amazon":
        config = BertConfig(dataset_name)
        model = initialize_bert_based_model(config, d_out=5)
    elif dataset_name == "rxrx1":
        model = initialize_torchvision_model("resnet50", d_out=1139, pretrained=True)
    else:
        raise ValueError(f"Unknown dataset {dataset_name}")

    root_dir = f"{root_dir}experiments"

    try:
        if algorithm == "ERM":
            state_dict = remove_prefix_from_state_dict(
                torch.load(
                    f"{root_dir}/{dataset_name}/{dataset_name}_seed:{seed}_epoch:{model_type}_model.pth"
                )["algorithm"]
            )
        elif algorithm == "IRM":
            state_dict = remove_prefix_from_state_dict(
                torch.load(
                    f"{root_dir}/IRM/{dataset_name}/{dataset_name}_seed:{seed}_epoch:{model_type}_model.pth"
                )["algorithm"]
            )
        elif algorithm == "groupDRO":
            state_dict = remove_prefix_from_state_dict(
                torch.load(
                    f"{root_dir}/groupDRO/{dataset_name}/{dataset_name}_seed:{seed}_epoch:{model_type}_model.pth"
                )["algorithm"]
            )
    except:
        raise ValueError(
            f"Model not found for dataset {dataset_name}, algorithm {algorithm}, seed {seed}, model type {model_type}, path: {root_dir}/{dataset_name}/{dataset_name}_seed:{seed}_epoch:{model_type}_model.pth"
        )

    model.load_state_dict(state_dict)
    return model
