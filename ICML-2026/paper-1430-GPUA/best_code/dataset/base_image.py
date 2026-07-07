import math
import os
import pickle
import random
from abc import abstractmethod
from collections import defaultdict
from typing import List, Union

import torch
import torchvision
from loguru import logger

from dataset.base import AbstractDataset, Example

from .utils import check_path, mkdir_if_missing, read_image, read_json, write_json


class ImageExample(Example):
    def __init__(
        self,
        file_path: str,
        label: int,
        label_name: str,
        check_file_path: bool = False,
        **_,
    ):
        if check_file_path:
            check_path(file_path)

        super().__init__(file_path, label, label_name)

    def set_data_info(self, img: torch.Tensor):
        self.data_info["height"] = img.size(1)
        self.data_info["width"] = img.size(2)

    @abstractmethod
    def load_image(self) -> torch.Tensor:
        img = read_image(self.file_path)
        # C, H, W & normalized
        img = torchvision.transforms.functional.to_tensor(img)

        if len(self.data_info) == 0:
            self.set_data_info(img)
        return img

    def load_example(self) -> Union[torch.Tensor, List]:
        return self.load_image()


class ImageDataset(AbstractDataset):
    def __init__(
        self,
        file_paths: List,
        labels: List,
        label_names: List,
        label2name: dict = None,
        transforms: torchvision.transforms.Compose = None,
        **_,
    ):
        self.transforms = transforms

        examples = self.create_examples(
            file_paths=file_paths, labels=labels, label_names=label_names
        )

        if label2name is None:
            label2name = {i: j for i, j in zip(labels, label_names)}
            label2name = {k: v for k, v in sorted(list(label2name.items()))}

        label_names = [name for name in label2name.values()]

        super().__init__(
            label_names=label_names,
            label2name=label2name,
            examples=examples,
        )

    def preprocess(self, data: torch.Tensor) -> torch.Tensor:
        assert isinstance(
            data, torch.Tensor
        ), "Images to transform must be a torch.Tensor"
        if self.transforms is not None:
            return self.transforms(data)
        return data

    def create_examples(
        self, file_paths: List, labels: List, label_names: List
    ) -> list:
        logger.info("Creating the examples ....")

        examples = []
        for idx, (file_path, label, label_name) in enumerate(
            zip(file_paths, labels, label_names)
        ):
            # check_file_path = True if idx % 1000 == 0 else False

            examples.append(
                ImageExample(
                    file_path=file_path,
                    label=label,
                    label_name=label_name,
                    check_file_path=False,
                )
            )

        return examples


class ImageDatasets(object):
    # Based on CoOp
    # https://github.com/KaiyangZhou/CoOp

    dataset_dir = None
    image_dir = None
    split_path = None
    split_fewshot_dir = None
    ignored_classes = []
    new_class_names = None

    def __init__(
        self,
        data_path: str,
        n_shot: int,
        seed: int,
        use_base_and_new: bool = False,
        train_transforms: torchvision.transforms.Compose = None,
        test_transforms: torchvision.transforms.Compose = None,
    ):
        # for consistency
        random.seed(seed)

        self.data_path = os.path.abspath(os.path.expanduser(data_path))
        self.dataset_dir = os.path.join(self.data_path, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, self.image_dir)
        self.split_path = os.path.join(self.dataset_dir, self.split_path)

        self.split_fewshot_dir = os.path.join(
            self.dataset_dir, self.split_fewshot_dir)

        # self.new_path = "/data/users/ysw/LFA-plus/" 
        # self.split_path = os.path.join(self.new_path, self.split_path)
        # self.split_fewshot_dir = os.path.join(self.new_path, self.split_fewshot_dir)
        
        mkdir_if_missing(self.split_fewshot_dir)

        train, val, test = self.read_data()

        if n_shot >= 1:
            preprocessed = os.path.join(
                self.split_fewshot_dir, f"nshot_{n_shot}-seed_{seed}.pkl"
            )

            if os.path.exists(preprocessed):
                logger.info(
                    f"Loading preprocessed few-shot data from {preprocessed}")

                with open(preprocessed, "rb") as file:
                    data = pickle.load(file)
                    train, val = data["train"], data["val"]
            else:
                train = self.generate_fewshot_dataset(train, n_shot=n_shot)
                if self.dataset_dir.split("/")[-1] != "imagenet":
                    val = self.generate_fewshot_dataset(
                        val, n_shot=min(n_shot, 4))
                else:
                    val = test

                data = {"train": train, "val": val}

                logger.info(
                    f"Saving preprocessed few-shot data to {preprocessed}")

                with open(preprocessed, "wb") as file:
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

        if use_base_and_new:
            logger.info("Dividing the classes into base and new")
            train_base, val_base, test_base = self.subsample_classes(
                train, val, test, subsample="base"
            )
            _, val_new, test_new = self.subsample_classes(
                train, val, test, subsample="new"
            )

            train, val = train_base, val_new

        self.train_dataset = ImageDataset(**train, transforms=train_transforms)
        self.val_dataset = ImageDataset(**val, transforms=test_transforms)

        if use_base_and_new:
            self.test_dataset = {
                "test_base": ImageDataset(**test_base, transforms=test_transforms),
                "test_new": ImageDataset(**test_new, transforms=test_transforms)
            }
            num_test_samples = [
                len(self.test_dataset["test_base"]),
                len(self.test_dataset["test_new"]),
            ]
        else:
            self.test_dataset = ImageDataset(
                **test, transforms=test_transforms)
            num_test_samples = len(self.test_dataset)

        tr_cls = self.train_dataset.num_classes
        test_cls = self.val_dataset.num_classes
        logger.info(f"For dataset: {self.dataset_dir}:")
        logger.info(f" - number of train examples: {len(self.train_dataset)}")
        logger.info(f" - number of val examples: {len(self.val_dataset)}")
        logger.info(f" - number of test examples: {num_test_samples}")
        logger.info(f" - number of train classes: {tr_cls}")
        logger.info(f" - number of test classes: {test_cls}")

    def get_datasets(self, eval_split="test", return_all=False):
        if return_all:
            return self.train_dataset, self.val_dataset, self.test_dataset

        if eval_split == "test":
            return self.train_dataset, self.test_dataset

        return self.train_dataset, self.val_dataset

    def read_data(self):
        if os.path.exists(self.split_path):
            train, val, test = self.read_split(self.split_path, self.image_dir)
        else:
            train, val, test = self.create_split(
                self.image_dir,
                ignored_classes=self.ignored_classes,
                new_class_names=self.new_class_names,
            )

            self.save_split(train, val, test, self.split_path, self.image_dir)

        return train, val, test

    def read_split(self, filepath, path_prefix):
        logger.info(f"Reading splits from {filepath}")

        split = read_json(filepath)

        train = self._joined_lists_to_examples(split["train"], path_prefix)
        val = self._joined_lists_to_examples(split["val"], path_prefix)
        test = self._joined_lists_to_examples(split["test"], path_prefix)

        return train, val, test

    def _joined_lists_to_examples(self, items, path_prefix):
        assert (
            len(items[0]) == 3
        ), "Each item must be a tuple of (file_path, label, class_name)"

        file_paths, labels, label_names = list(zip(*items))

        if path_prefix and len(path_prefix) > 0:
            file_paths = [os.path.join(path_prefix, fp) for fp in file_paths]

        return self._lists_to_examples(file_paths, labels, label_names)

    def _lists_to_examples(self, file_paths, labels, label_names):
        assert (
            len(file_paths) == len(labels) == len(label_names)
        ), "All lists must have the same length"

        dataset_examples = {
            "file_paths": file_paths,
            "labels": labels,
            "label_names": label_names,
        }

        return dataset_examples

    def create_split(
        self,
        image_dir,
        p_trn=0.5,
        p_val=0.2,
        ignored_classes=None,
        new_class_names=None,
    ):
        # The data are supposed to be organized into the following structure
        # images/class1, images/class2, images/class3 ......

        categories = [f for f in os.listdir(
            image_dir) if not f.startswith(".")]

        if ignored_classes is not None and len(ignored_classes) > 0:
            categories = [c for c in categories if c not in ignored_classes]

        categories.sort()

        p_tst = 1 - p_trn - p_val
        logger.info(
            f"Splitting the data into {p_trn:.0%} train,"
            f"{p_val:.0%} val, and {p_tst:.0%} test"
        )

        train_filepaths, val_filepaths, test_filepaths = [], [], []
        train_labels, val_labels, test_labels = [], [], []
        train_label_names, val_label_names, test_label_names = [], [], []

        for label, category in enumerate(categories):
            category_dir = os.path.join(image_dir, category)
            images = [f for f in os.listdir(
                category_dir) if not f.startswith(".")]

            images = [os.path.join(category_dir, im) for im in images]
            random.shuffle(images)

            n_total = len(images)
            n_train = round(n_total * p_trn)
            n_val = round(n_total * p_val)

            n_test = n_total - n_train - n_val
            assert n_train > 0 and n_val > 0 and n_test > 0

            if new_class_names is not None and category in new_class_names:
                category = new_class_names[category]

            train_filepaths.extend(images[:n_train])
            train_labels.extend([label] * n_train)
            train_label_names.extend([category] * n_train)

            val_filepaths.extend(images[n_train: n_train + n_val])
            val_labels.extend([label] * n_val)
            val_label_names.extend([category] * n_val)

            test_filepaths.extend(images[n_train + n_val:])
            test_labels.extend([label] * n_test)
            test_label_names.extend([category] * n_test)

        train_examples = self._lists_to_examples(
            train_filepaths, train_labels, train_label_names
        )
        val_examples = self._lists_to_examples(
            val_filepaths, val_labels, val_label_names
        )
        test_examples = self._lists_to_examples(
            test_filepaths, test_labels, test_label_names
        )

        return train_examples, val_examples, test_examples

    def save_split(
        self, train_examples, val_examples, test_examples, filepath, path_prefix
    ):
        logger.info(f"Saving splits to {filepath}")

        def _extract(examples_dict):
            paths = examples_dict["file_paths"]
            labels = examples_dict["labels"]
            label_names = examples_dict["label_names"]

            out = []
            for path, label, classname in zip(paths, labels, label_names):
                path = path.replace(path_prefix, "")
                path = path[1:] if path.startswith("/") else path
                out.append((path, label, classname))
            return out

        train = _extract(train_examples)
        val = _extract(val_examples)
        test = _extract(test_examples)

        dict_to_save = {"train": train, "val": val, "test": test}

        write_json(dict_to_save, filepath)

    def subsample_classes(
        self,
        train_examples: dict,
        val_examples: dict,
        test_examples: dict,
        subsample: str = "all",
    ):
        # Divide classes into two groups: base and new classes
        # Each input is a dict with paths, labels, and label_namess

        assert subsample in ["all", "base", "new"]

        if subsample == "all":
            return train_examples, val_examples, test_examples

        logger.info(f"Subsamling {subsample} classes!")

        labels = list(set(train_examples["labels"]))
        labels.sort()

        n = len(labels)
        # Divide classes into two halves
        m = math.ceil(n / 2)

        if subsample == "base":
            # take the first half
            selected = labels[:m]
        else:
            # take the second half
            selected = labels[m:]

        relabeler = {y: y_new for y_new, y in enumerate(selected)}

        output = []
        for examples in [train_examples, val_examples, test_examples]:
            dataset_new = {"file_paths": [], "labels": [], "label_names": []}

            for idx, label in enumerate(examples["labels"]):
                if label in selected:
                    dataset_new["file_paths"].append(
                        examples["file_paths"][idx])

                    dataset_new["label_names"].append(
                        examples["label_names"][idx])

                    dataset_new["labels"].append(relabeler[label])

            output.append(dataset_new)

        return output

    def generate_fewshot_dataset(self, *data_sources, n_shot=-1, repeat=False):
        # Generates a few-shot dataset (typically for the training set)
        # each item of data_sources is a dict with paths, labels, and label_names

        def keep_sampled_indices(items: list, indices: list):
            return [items[i] for i in indices]

        if n_shot < 1:
            if len(data_sources) == 1:
                return data_sources[0]
            return data_sources

        logger.info(f"Creating a {n_shot}-shot dataset")

        new_data_sources = []

        for examples in data_sources:
            labels_to_idx = self.split_dataset_by_label(examples["labels"])

            all_sampeled_indices = []

            for label, indices in labels_to_idx.items():
                if len(indices) >= n_shot:
                    sampled_indices = random.sample(indices, n_shot)
                else:
                    if repeat:
                        sampled_indices = random.choices(indices, k=n_shot)
                    else:
                        sampled_indices = indices

                all_sampeled_indices.extend(sampled_indices)

            new_examples = {
                "file_paths": keep_sampled_indices(
                    examples["file_paths"], all_sampeled_indices
                ),
                "labels": keep_sampled_indices(
                    examples["labels"], all_sampeled_indices
                ),
                "label_names": keep_sampled_indices(
                    examples["label_names"], all_sampeled_indices
                ),
            }

            new_data_sources.append(new_examples)

        if len(new_data_sources) == 1:
            return new_data_sources[0]

        return new_data_sources

    def split_dataset_by_label(self, labels):
        labels_to_idx = defaultdict(list)
        for idx, label in enumerate(labels):
            labels_to_idx[label].append(idx)
        return labels_to_idx

    def split_trainval(self, trainval, p_val=0.2):
        p_trn = 1 - p_val
        logger.info(
            f"Splitting trainval into {p_trn:.0%} train and {p_val:.0%} val")

        label_to_idx = defaultdict(list)
        for idx, label in enumerate(trainval["labels"]):
            label_to_idx[label].append(idx)

        train_file_paths, train_labels, train_label_names = [], [], []
        val_file_paths, val_labels, val_label_names = [], [], []

        for label, indices in label_to_idx.items():
            n_val = round(len(indices) * p_val)
            assert n_val > 0

            random.shuffle(indices)

            for n, idx in enumerate(indices):
                if n < n_val:
                    val_file_paths.append(trainval["file_paths"][idx])
                    val_labels.append(trainval["labels"][idx])
                    val_label_names.append(trainval["label_names"][idx])
                else:
                    train_file_paths.append(trainval["file_paths"][idx])
                    train_labels.append(trainval["labels"][idx])
                    train_label_names.append(trainval["label_names"][idx])

        train = self._lists_to_examples(
            train_file_paths, train_labels, train_label_names
        )

        val = self._lists_to_examples(
            val_file_paths, val_labels, val_label_names)

        return train, val
