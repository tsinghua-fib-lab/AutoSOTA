from typing import Tuple, Optional, List
from pathlib import Path
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import datasets, transforms
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
STANDARD_MEAN = [0.5, 0.5, 0.5]
STANDARD_STD = [0.5, 0.5, 0.5]


def get_transforms(
    img_size: int = 28,
    in_channels: int = 1,
    use_imagenet_norm: bool = False,
) -> transforms.Compose:
    mean = IMAGENET_MEAN if use_imagenet_norm else STANDARD_MEAN
    std = IMAGENET_STD if use_imagenet_norm else STANDARD_STD

    transform_list = [transforms.Resize((img_size, img_size))]

    if in_channels == 3:
        transform_list.append(transforms.Grayscale(num_output_channels=3))

    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    return transforms.Compose(transform_list)


def get_mnist_loaders(
    batch_size: int,
    img_size: int = 28,
    num_workers: int = 4,
    use_imagenet_norm: bool = False,
) -> Tuple[DataLoader, DataLoader, int]:
    transform = get_transforms(img_size, in_channels=3 if use_imagenet_norm else 1, use_imagenet_norm=use_imagenet_norm)

    if use_imagenet_norm:
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])

    train_ds = datasets.MNIST("./data", train=True, download=True, transform=transform)
    test_ds = datasets.MNIST("./data", train=False, download=True, transform=transform)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return train_loader, test_loader, len(train_ds)


def get_fashionmnist_loaders(
    batch_size: int,
    img_size: int = 28,
    num_workers: int = 4,
    use_imagenet_norm: bool = False,
) -> Tuple[DataLoader, DataLoader, int]:
    if use_imagenet_norm:
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])

    train_ds = datasets.FashionMNIST("./data", train=True, download=True, transform=transform)
    test_ds = datasets.FashionMNIST("./data", train=False, download=True, transform=transform)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return train_loader, test_loader, len(train_ds)


def get_cifar10_loaders(
    batch_size: int,
    img_size: int = 32,
    num_workers: int = 4,
    use_imagenet_norm: bool = False,
) -> Tuple[DataLoader, DataLoader, int]:
    mean = IMAGENET_MEAN if use_imagenet_norm else STANDARD_MEAN
    std = IMAGENET_STD if use_imagenet_norm else STANDARD_STD

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_ds = datasets.CIFAR10("./data", train=True, download=True, transform=transform)
    test_ds = datasets.CIFAR10("./data", train=False, download=True, transform=transform)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return train_loader, test_loader, len(train_ds)


def get_cifar100_loaders(
    batch_size: int,
    img_size: int = 32,
    num_workers: int = 4,
    use_imagenet_norm: bool = False,
) -> Tuple[DataLoader, DataLoader, int]:
    mean = IMAGENET_MEAN if use_imagenet_norm else STANDARD_MEAN
    std = IMAGENET_STD if use_imagenet_norm else STANDARD_STD

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_ds = datasets.CIFAR100("./data", train=True, download=True, transform=transform)
    test_ds = datasets.CIFAR100("./data", train=False, download=True, transform=transform)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return train_loader, test_loader, len(train_ds)


def get_pathmnist_loaders(
    batch_size: int,
    img_size: int = 28,
    num_workers: int = 4,
    use_imagenet_norm: bool = False,
) -> Tuple[DataLoader, DataLoader, int]:
    import medmnist

    mean = IMAGENET_MEAN if use_imagenet_norm else STANDARD_MEAN
    std = IMAGENET_STD if use_imagenet_norm else STANDARD_STD

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    info = medmnist.INFO['pathmnist']
    DataClass = getattr(medmnist, info['python_class'])

    train_ds = DataClass(split='train', transform=transform, download=True, root='./data')
    test_ds = DataClass(split='test', transform=transform, download=True, root='./data')

    def collate_fn(batch):
        images = torch.stack([item[0] for item in batch])
        labels = torch.tensor([item[1].item() for item in batch], dtype=torch.long)
        return images, labels

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        collate_fn=collate_fn,
    )

    return train_loader, test_loader, len(train_ds)


def get_dataset_loaders(
    dataset_name: str,
    batch_size: int,
    img_size: int = 28,
    num_workers: int = 4,
    use_imagenet_norm: bool = False,
) -> Tuple[DataLoader, DataLoader, int]:
    dataset_name = dataset_name.lower()

    loaders = {
        "mnist": get_mnist_loaders,
        "fashionmnist": get_fashionmnist_loaders,
        "cifar10": get_cifar10_loaders,
        "cifar100": get_cifar100_loaders,
        "pathmnist": get_pathmnist_loaders,
    }

    if dataset_name not in loaders:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(loaders.keys())}")

    return loaders[dataset_name](batch_size, img_size, num_workers, use_imagenet_norm)


def get_dataset_info(dataset_name: str) -> Tuple[int, int, int]:
    dataset_name = dataset_name.lower()

    info = {
        "mnist": (1, 28, 10),
        "fashionmnist": (1, 28, 10),
        "cifar10": (3, 32, 10),
        "cifar100": (3, 32, 100),
        "pathmnist": (3, 28, 9),  # PathMNIST: 9 classes, 28x28, RGB
    }

    if dataset_name not in info:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return info[dataset_name]


def extract_features(
    backbone: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    description: str = "Extracting features",
) -> Tuple[torch.Tensor, torch.Tensor]:
    backbone.eval()
    all_features = []
    all_labels = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task(description, total=len(dataloader))
        with torch.no_grad():
            for data, labels in dataloader:
                data = data.to(device)
                features = backbone(data)
                all_features.append(features.cpu())
                all_labels.append(labels)
                progress.advance(task)

    return torch.cat(all_features, dim=0), torch.cat(all_labels, dim=0)


def get_or_extract_features(
    backbone: nn.Module,
    dataset_name: str,
    model_name: str,
    img_size: int,
    device: torch.device,
    cache_dir: str = "data/features",
    num_workers: int = 4,
    batch_size: int = 256,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    cache_path = Path(cache_dir) / model_name / dataset_name
    cache_path.mkdir(parents=True, exist_ok=True)

    train_features_path = cache_path / "train_features.pt"
    train_labels_path = cache_path / "train_labels.pt"
    test_features_path = cache_path / "test_features.pt"
    test_labels_path = cache_path / "test_labels.pt"

    if all(p.exists() for p in [train_features_path, train_labels_path, test_features_path, test_labels_path]):
        from rich.console import Console
        console = Console()
        console.print(f"[green]Loading cached features from {cache_path}[/green]")
        train_features = torch.load(train_features_path)
        train_labels = torch.load(train_labels_path)
        test_features = torch.load(test_features_path)
        test_labels = torch.load(test_labels_path)
    else:
        from rich.console import Console
        console = Console()
        console.print(f"[yellow]Extracting features for {dataset_name} using {model_name}...[/yellow]")

        train_loader, test_loader, _ = get_dataset_loaders(
            dataset_name, batch_size=batch_size, img_size=img_size,
            num_workers=num_workers, use_imagenet_norm=True,
        )

        backbone = backbone.to(device)
        backbone.eval()

        train_features, train_labels = extract_features(
            backbone, train_loader, device, f"[cyan]{dataset_name}[/cyan] train"
        )
        test_features, test_labels = extract_features(
            backbone, test_loader, device, f"[cyan]{dataset_name}[/cyan] test"
        )

        torch.save(train_features, train_features_path)
        torch.save(train_labels, train_labels_path)
        torch.save(test_features, test_features_path)
        torch.save(test_labels, test_labels_path)
        console.print(f"[green]Cached features to {cache_path}[/green]")

    return train_features, train_labels, test_features, test_labels, len(train_features)


def get_feature_loaders(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    test_features: torch.Tensor,
    test_labels: torch.Tensor,
    batch_size: int,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader]:
    train_ds = TensorDataset(train_features, train_labels)
    test_ds = TensorDataset(test_features, test_labels)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return train_loader, test_loader


class TextDataset(Dataset):
    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer,
        max_length: int = 128,
    ) -> None:
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long),
        }


def get_imdb_data(
    max_samples: int = 25000,
) -> Tuple[List[str], List[int], List[str], List[int]]:
    """Load IMDB movie review dataset from HuggingFace."""
    from datasets import load_dataset

    imdb = load_dataset("imdb")

    train_texts = [item["text"] for item in imdb["train"]]
    train_labels = [item["label"] for item in imdb["train"]]
    test_texts = [item["text"] for item in imdb["test"]]
    test_labels = [item["label"] for item in imdb["test"]]

    if max_samples < len(train_texts):
        indices = random.sample(range(len(train_texts)), max_samples)
        train_texts = [train_texts[i] for i in indices]
        train_labels = [train_labels[i] for i in indices]

    return train_texts, train_labels, test_texts, test_labels


def get_sst2_data(
    max_samples: int = 67000,
) -> Tuple[List[str], List[int], List[str], List[int]]:
    """Load SST-2 from GLUE benchmark via HuggingFace."""
    from datasets import load_dataset

    sst2 = load_dataset("glue", "sst2")

    train_texts = [item["sentence"] for item in sst2["train"]]
    train_labels = [item["label"] for item in sst2["train"]]
    # SST-2 test has no labels; use validation as test
    test_texts = [item["sentence"] for item in sst2["validation"]]
    test_labels = [item["label"] for item in sst2["validation"]]

    if max_samples < len(train_texts):
        indices = random.sample(range(len(train_texts)), max_samples)
        train_texts = [train_texts[i] for i in indices]
        train_labels = [train_labels[i] for i in indices]

    return train_texts, train_labels, test_texts, test_labels


def get_tfidf_features(
    train_texts: List[str],
    test_texts: List[str],
    max_features: int = 10000,
    public_texts: Optional[List[str]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Extract TF-IDF features using sklearn, returning float32 tensors."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(max_features=max_features)
    train_feats = vectorizer.fit_transform(train_texts)
    test_feats = vectorizer.transform(test_texts)

    train_tensor = torch.tensor(train_feats.toarray(), dtype=torch.float32)
    test_tensor = torch.tensor(test_feats.toarray(), dtype=torch.float32)

    # Normalize to unit norm
    train_tensor = train_tensor / (train_tensor.norm(dim=1, keepdim=True) + 1e-8)
    test_tensor = test_tensor / (test_tensor.norm(dim=1, keepdim=True) + 1e-8)

    public_tensor = None
    if public_texts is not None:
        public_feats = vectorizer.transform(public_texts)
        public_tensor = torch.tensor(public_feats.toarray(), dtype=torch.float32)
        public_tensor = public_tensor / (public_tensor.norm(dim=1, keepdim=True) + 1e-8)

    return train_tensor, test_tensor, public_tensor


def get_stackoverflow_data(
    max_samples: int = 5000,
) -> Tuple[List[str], List[int], List[str], List[int]]:
    from datasets import load_dataset

    so_dataset = load_dataset("mteb/stackoverflowdupquestions-reranking")

    private_texts = []
    private_labels = []

    for item in so_dataset["train"]:
        text = f"Question 1: {item['query']} Question 2: {item['positive']}"
        private_texts.append(text)
        private_labels.append(1)

        text = f"Question 1: {item['query']} Question 2: {item['negative']}"
        private_texts.append(text)
        private_labels.append(0)

    indices = random.sample(
        range(len(private_texts)), min(max_samples, len(private_texts))
    )
    private_texts = [private_texts[i] for i in indices]
    private_labels = [private_labels[i] for i in indices]

    test_texts = []
    test_labels = []
    for item in so_dataset["test"]:
        text = f"Question 1: {item['query']} Question 2: {item['positive']}"
        test_texts.append(text)
        test_labels.append(1)

        text = f"Question 1: {item['query']} Question 2: {item['negative']}"
        test_texts.append(text)
        test_labels.append(0)

    return private_texts, private_labels, test_texts, test_labels


def get_agnews_data(
    max_samples: int = 5000,
) -> Tuple[List[str], List[int]]:
    from datasets import load_dataset

    ag_dataset = load_dataset("ag_news")

    public_texts = []
    public_labels = []

    for item in ag_dataset["train"]:
        text = item["text"]
        label = item["label"]
        binary_label = 0 if label < 2 else 1
        public_texts.append(text)
        public_labels.append(binary_label)
        if len(public_texts) >= max_samples:
            break

    return public_texts, public_labels


def get_text_loaders(
    private_dataset: str,
    public_dataset: str,
    batch_size: int,
    max_length: int = 128,
    num_workers: int = 4,
    max_samples: int = 5000,
    tokenizer_name: str = "bert-base-uncased",
) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    if private_dataset.lower() == "stackoverflow":
        train_texts, train_labels, test_texts, test_labels = get_stackoverflow_data(max_samples)
    elif private_dataset.lower() == "imdb":
        train_texts, train_labels, test_texts, test_labels = get_imdb_data(max_samples)
    elif private_dataset.lower() == "sst2":
        train_texts, train_labels, test_texts, test_labels = get_sst2_data(max_samples)
    else:
        raise ValueError(f"Unknown private text dataset: {private_dataset}")

    if public_dataset.lower() == "agnews":
        public_texts, public_labels = get_agnews_data(max_samples)
    elif public_dataset.lower() == "imdb":
        pub_texts, pub_labels, _, _ = get_imdb_data(max_samples)
        public_texts, public_labels = pub_texts, pub_labels
    else:
        raise ValueError(f"Unknown public text dataset: {public_dataset}")

    train_ds = TextDataset(train_texts, train_labels, tokenizer, max_length)
    test_ds = TextDataset(test_texts, test_labels, tokenizer, max_length)
    public_ds = TextDataset(public_texts, public_labels, tokenizer, max_length)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    public_loader = DataLoader(
        public_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )

    return train_loader, test_loader, public_loader, len(train_ds)


def get_text_dataset_info(dataset_name: str) -> int:
    dataset_name = dataset_name.lower()

    info = {
        "stackoverflow": 2,
        "agnews": 2,
        "imdb": 2,
        "sst2": 2,
    }

    if dataset_name not in info:
        raise ValueError(f"Unknown text dataset: {dataset_name}")

    return info[dataset_name]


def is_text_dataset(dataset_name: str) -> bool:
    return dataset_name.lower() in ["stackoverflow", "agnews", "imdb", "sst2"]
