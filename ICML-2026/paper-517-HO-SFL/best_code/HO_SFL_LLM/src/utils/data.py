from enum import Enum

from .language_utils import (
    LM_DATASET_MAP,
    LM_TEMPLATE_MAP,
    CustomLMDataset,
    CustomLMGenerationDataset,
    LmClassificationTask,
    LmGenerationTask,
    get_collate_fn,
    get_collate_fn_for_gen_model,
    get_hf_tokenizer,
)
from .data_split import dirichlet_split
import torch
from torch.utils.data.dataset import Subset, Dataset
import torchvision
import torchvision.transforms as transforms

from datasets import load_dataset as huggingface_load_dataset


class ImageClassificationTask(Enum):
    mnist = "mnist"
    cifar10 = "cifar10"
    fashion = "fashion"


class DatasetSplit(torch.utils.data.Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


def get_random_split_chunk_length(total_length: int, num_split: int) -> list[int]:
    int_len = total_length // num_split
    rem = total_length % num_split
    ret_base = [int_len] * num_split
    for i in range(rem):
        ret_base[i] += 1
    return ret_base


def get_dataloaders(
    data_setting,
    num_train_split: int,
    seed: int,
    hf_model_name: str | None = None,
) -> tuple[
    list[torch.utils.data.DataLoader],
    torch.utils.data.DataLoader,
]:
    if data_setting.dataset == ImageClassificationTask.mnist:
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        train_dataset = torchvision.datasets.MNIST(
            root="./data", train=True, download=True, transform=transform
        )
        test_dataset = torchvision.datasets.MNIST(
            root="./data", train=False, download=True, transform=transform
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=data_setting.test_batch_size, pin_memory=True
        )
    elif data_setting.dataset == ImageClassificationTask.cifar10:
        transform_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )
        train_dataset = torchvision.datasets.CIFAR10(
            root="./data", train=True, download=True, transform=transform_train
        )
        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root="./data", train=False, download=True, transform=transform_test
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=data_setting.test_batch_size, pin_memory=True
        )
    elif data_setting.dataset == ImageClassificationTask.fashion:
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        )
        train_dataset = torchvision.datasets.FashionMNIST(
            root="./data", train=True, download=True, transform=transform
        )
        test_dataset = torchvision.datasets.FashionMNIST(
            root="./data", train=False, download=True, transform=transform
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=data_setting.test_batch_size, pin_memory=True
        )
    elif isinstance(data_setting.dataset, (LmClassificationTask, LmGenerationTask)):
        assert isinstance(hf_model_name, str)

        max_length = data_setting.max_length

        if isinstance(data_setting.dataset, LmClassificationTask):
            dataset = huggingface_load_dataset(
                LM_DATASET_MAP[data_setting.dataset.value], data_setting.dataset.value
            )
            print("Loaded dataset")
            raw_train_dataset = dataset["train"]
            raw_test_dataset = dataset["validation"]
            if data_setting.small_test:
                raw_test_dataset = raw_test_dataset.select(range(100)).shuffle(seed)
            tokenizer = get_hf_tokenizer(hf_model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.pad_token_id = tokenizer.eos_token_id
            template = LM_TEMPLATE_MAP[data_setting.dataset.value]()
            encoded_train_texts = list(map(template.verbalize, raw_train_dataset))
            encoded_test_texts = list(map(template.verbalize, raw_test_dataset))
            train_dataset = CustomLMDataset(
                encoded_train_texts, tokenizer, max_length=max_length
            )
            test_dataset = CustomLMDataset(
                encoded_test_texts, tokenizer, max_length=max_length
            )
            test_loader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=data_setting.test_batch_size,
                shuffle=True,
                collate_fn=get_collate_fn(tokenizer, max_length),
            )
        elif isinstance(data_setting.dataset, LmGenerationTask):
            dataset = huggingface_load_dataset(
                LM_DATASET_MAP[data_setting.dataset.value]
            )
            raw_train_dataset = dataset["train"].select(range(1000)).shuffle(seed)
            raw_test_dataset = dataset["validation"].select(range(100)).shuffle(seed)
            tokenizer = get_hf_tokenizer(hf_model_name)
            template = LM_TEMPLATE_MAP[data_setting.dataset.value]()
            encoded_train_texts = list(map(template.verbalize, raw_train_dataset))
            encoded_test_texts = list(map(template.encode, raw_test_dataset))
            if data_setting.dataset == LmGenerationTask.squad:
                test_golds = list(
                    map(lambda d: d["answers"]["text"][0], raw_test_dataset)
                )
            elif data_setting.dataset == LmGenerationTask.drop:
                test_golds = list(
                    map(lambda d: d["answers_spans"]["spans"][0], raw_test_dataset)
                )
            elif data_setting.dataset == LmGenerationTask.xsum:
                test_golds = list(map(lambda d: d["summary"], raw_test_dataset))
            train_dataset = CustomLMDataset(
                encoded_train_texts, tokenizer, max_length=max_length
            )
            test_dataset = CustomLMGenerationDataset(
                encoded_test_texts, test_golds, tokenizer, max_length=max_length
            )
            test_loader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=data_setting.test_batch_size,
                shuffle=True,
                collate_fn=get_collate_fn_for_gen_model(tokenizer, max_length),
            )
    else:
        raise Exception(f"Dataset {data_setting.dataset} is not supported")

    # already updated at main function
    splitted_train_sets: list[DatasetSplit] | list[Subset] | list[Dataset]
    if num_train_split == 1:
        splitted_train_sets = [train_dataset]
    else:
        if isinstance(data_setting.dataset, (LmClassificationTask)):
            if data_setting.iid:
                print("Using IID split for LM classification task.")
                generator = torch.Generator().manual_seed(seed)
                splitted_train_sets = torch.utils.data.random_split(
                    train_dataset,
                    get_random_split_chunk_length(len(train_dataset), num_train_split),
                    generator=generator,
                )
            else:
                labels = list(map(lambda x: x["label"], raw_train_dataset))
                splitted_train_sets = dirichlet_split(
                    train_dataset,
                    labels,
                    num_train_split,
                    data_setting.dirichlet_alpha,
                    seed,
                )
        else:
            generator = torch.Generator().manual_seed(seed)
            splitted_train_sets = torch.utils.data.random_split(
                train_dataset,
                get_random_split_chunk_length(len(train_dataset), num_train_split),
                generator=generator,
            )

    splitted_train_loaders = []
    for i in range(num_train_split):
        if isinstance(data_setting.dataset, (LmClassificationTask, LmGenerationTask)):
            dataloader = torch.utils.data.DataLoader(
                splitted_train_sets[i],
                batch_size=data_setting.train_batch_size,
                shuffle=True,
                collate_fn=get_collate_fn(tokenizer, max_length),
            )
        else:
            dataloader = torch.utils.data.DataLoader(
                splitted_train_sets[i],
                batch_size=data_setting.train_batch_size,
                pin_memory=True,
            )
        splitted_train_loaders.append(dataloader)
    return splitted_train_loaders, test_loader
