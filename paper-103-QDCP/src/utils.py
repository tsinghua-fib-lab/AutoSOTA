import argparse
import copy
import sys
from typing import Dict, List, Optional, Tuple, Union

import medmnist
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, models, transforms
from tqdm import tqdm

from resnet import (small_resnet14, small_resnet20, small_resnet32,
                    small_resnet44, small_resnet56, small_resnet110)


def get_datasets(
    dataset_name: str, data_dir: str, use_data_augmentation: bool = False
) -> Dict[str, Dataset]:
    if dataset_name == "mnist":
        construct_dataset = datasets.MNIST
        data_augmentation = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ]
        train_transform = [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
        test_transform = [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    elif dataset_name == "fashion":
        construct_dataset = datasets.FashionMNIST
        data_augmentation = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ]
        train_transform = [
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,)),
        ]
        test_transform = [
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,)),
        ]
    elif dataset_name == "svhn":
        construct_dataset = datasets.SVHN
        data_augmentation = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ]
        train_transform = [
            transforms.ToTensor(),
            transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
        ]
        test_transform = [
            transforms.ToTensor(),
            transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
        ]
    elif dataset_name in ["cifar10", "cifar10-2", "cifar10-3"]:
        construct_dataset = datasets.CIFAR10
        data_augmentation = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ]
        train_transform = [
            transforms.ToTensor(),
            transforms.Normalize((0.4913, 0.4821, 0.4465), (0.2470, 0.2434, 0.2615)),
        ]
        test_transform = [
            transforms.ToTensor(),
            transforms.Normalize((0.4913, 0.4821, 0.4465), (0.2470, 0.2434, 0.2615)),
        ]
    elif dataset_name == "cifar100":
        construct_dataset = datasets.CIFAR100
        data_augmentation = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ]
        train_transform = [
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
        ]
        test_transform = [
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
        ]
    elif dataset_name == "bloodmnist":
        info = medmnist.INFO["bloodmnist"]
        construct_dataset = getattr(medmnist, info["python_class"])
        data_augmentation = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ]
        train_transform = [
            transforms.ToTensor(),
            transforms.Normalize((0.7943, 0.6597, 0.6962), (0.2156, 0.2416, 0.1179)),
        ]
        test_transform = [
            transforms.ToTensor(),
            transforms.Normalize((0.7943, 0.6597, 0.6962), (0.2156, 0.2416, 0.1179)),
        ]
        target_transform = transforms.Lambda(lambda x: x[0])
    elif dataset_name == "dermamnist":
        info = medmnist.INFO["dermamnist"]
        construct_dataset = getattr(medmnist, info["python_class"])
        data_augmentation = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ]
        train_transform = [
            transforms.ToTensor(),
            transforms.Normalize((0.7631, 0.5380, 0.5613), (0.1366, 0.1542, 0.1692)),
        ]
        test_transform = [
            transforms.ToTensor(),
            transforms.Normalize((0.7631, 0.5380, 0.5613), (0.1366, 0.1542, 0.1692)),
        ]
        target_transform = transforms.Lambda(lambda x: x[0])
    elif dataset_name == "pathmnist":
        info = medmnist.INFO["pathmnist"]
        construct_dataset = getattr(medmnist, info["python_class"])
        data_augmentation = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ]
        train_transform = [
            transforms.ToTensor(),
            transforms.Normalize((0.7405, 0.533, 0.7058), (0.1237, 0.1768, 0.1244)),
        ]
        test_transform = [
            transforms.ToTensor(),
            transforms.Normalize((0.7405, 0.533, 0.7058), (0.1237, 0.1768, 0.1244)),
        ]
        target_transform = transforms.Lambda(lambda x: x[0])
    elif dataset_name == "tissuemnist":
        info = medmnist.INFO["tissuemnist"]
        construct_dataset = getattr(medmnist, info["python_class"])
        data_augmentation = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ]
        train_transform = [
            transforms.ToTensor(),
            transforms.Normalize((0.1020,), (0.1000,)),
        ]
        test_transform = [
            transforms.ToTensor(),
            transforms.Normalize((0.1020,), (0.1000,)),
        ]
        target_transform = transforms.Lambda(lambda x: x[0])
    else:
        raise ValueError(f"Dataset name: {dataset_name} not valid.".center(20, "="))

    if use_data_augmentation:
        train_transform = data_augmentation + train_transform

    if dataset_name in ("bloodmnist", "dermamnist", "pathmnist", "tissuemnist"):
        train_data = construct_dataset(
            split="train",
            root=data_dir,
            # download=False,
            download=True,
            transform=transforms.Compose(train_transform),
            target_transform=target_transform,
        )
        val_data = construct_dataset(
            split="val",
            root=data_dir,
            # download=False,
            download=True,
            transform=transforms.Compose(test_transform),
            target_transform=target_transform,
        )
        test_data = construct_dataset(
            split="test",
            root=data_dir,
            # download=False,
            download=True,
            transform=transforms.Compose(test_transform),
            target_transform=target_transform,
        )
        return {"train": train_data, "val": val_data, "test": test_data}
    elif dataset_name == "svhn":
        train_data = construct_dataset(
            data_dir,
            split="train",
            download=True,
            transform=transforms.Compose(train_transform),
        )
        test_data = construct_dataset(
            data_dir,
            split="test",
            download=True,
            transform=transforms.Compose(test_transform),
        )
        return {"train": train_data, "test": test_data}
    else:
        train_data = construct_dataset(
            data_dir,
            train=True,
            download=False,
            # download=True,
            transform=transforms.Compose(train_transform),
        )
        test_data = construct_dataset(
            data_dir,
            train=False,
            download=False,
            # download=True,
            transform=transforms.Compose(test_transform),
        )

        return {"train": train_data, "test": test_data}


def partition_dataset(
    dataset: Dataset,
    client_label_map: Dict[str, List[int]] = None,
    samples_per_client: int = None,
    use_iid_partition: bool = False,
    seed: int = 123,
    dirichlet_beta: float = 0.0,
) -> Dict[str, Dataset]:
    client_datasets: Dict[str, Dataset] = {}

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    if use_iid_partition:
        perm = torch.randperm(len(dataset))
        num_clients = len(client_label_map)
        num_samples = round(len(dataset) / num_clients)
        for i, client in enumerate(client_label_map.keys()):
            client_datasets[client] = Subset(
                dataset,
                perm[i * num_samples : (i + 1) * num_samples][:samples_per_client],
            )
    elif dirichlet_beta > 0:
        if hasattr(dataset, "targets"):
            targets = torch.tensor(dataset.targets)
        elif hasattr(dataset, "labels"):
            targets = torch.tensor(dataset.labels)
        else:
            raise ValueError("Cannot find labels")
        classes = targets.unique()
        bool_targets = torch.stack([targets == y for y in classes])
        class_counts = bool_targets.sum(1)
        class_prior = class_counts / len(targets)
        num_clients = len(client_label_map)
        client_class_weights = torch.distributions.dirichlet.Dirichlet(
            torch.tensor(dirichlet_beta * class_prior)
        ).sample((num_clients,))

        for i, client in enumerate(client_label_map.keys()):
            client_samples = []
            for j, weight in enumerate(client_class_weights[i]):
                num_to_sample = round((weight * class_counts[j]).item())
                class_samples = np.random.choice(
                    bool_targets[j].nonzero().flatten(), size=num_to_sample
                )
                client_samples.extend(class_samples)

            client_datasets[client] = Subset(dataset, client_samples)

    elif client_label_map is not None:
        if hasattr(dataset, "targets"):
            targets = torch.tensor(dataset.targets)
        elif hasattr(dataset, "labels"):
            targets = torch.tensor(dataset.labels)
        else:
            raise ValueError("Cannot find labels")
        classes = targets.unique()
        bool_targets = torch.stack([targets == y for y in classes])

        for client, labels in client_label_map.items():
            index = torch.where(bool_targets[labels].sum(0))[0]
            perm = torch.randperm(index.size(0))
            subsample = index[perm][:samples_per_client]
            assert (
                subsample.size(0) != 0
            ), f"Dataset error for client {client} with label {labels}"
            client_datasets[client] = Subset(dataset, subsample)
    else:
        raise ValueError()

    return client_datasets


def init_params(net: torch.nn.Module):
    """Init layer parameters."""
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            torch.nn.init.kaiming_normal(m.weight, mode="fan_out")
            if m.bias:
                torch.nn.init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            torch.nn.init.constant(m.weight, 1)
            torch.nn.init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            torch.nn.init.normal(m.weight, std=1e-3)
            if m.bias:
                torch.nn.init.constant(m.bias, 0)


def make_model(
    architecture: str,
    in_channels: int = 1,
    num_classes: int = 10,
    pretrained: bool = False,
) -> torch.nn.Module:
    if architecture == "cnn":
        model = Net_eNTK(in_channels, num_classes)
    elif architecture == "small_resnet14":
        model = small_resnet14(in_channels=in_channels, num_classes=num_classes)
    elif architecture == "small_resnet20":
        model = small_resnet20(in_channels=in_channels, num_classes=num_classes)
    elif architecture == "small_resnet32":
        model = small_resnet32(in_channels=in_channels, num_classes=num_classes)
    elif architecture == "small_resnet44":
        model = small_resnet44(in_channels=in_channels, num_classes=num_classes)
    elif architecture == "small_resnet56":
        model = small_resnet56(in_channels=in_channels, num_classes=num_classes)
    elif architecture == "resnet18":
        if pretrained:
            model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        else:
            model = models.resnet18()
        model.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        model.bn1 = nn.BatchNorm2d(64)
        model.fc = nn.Linear(in_features=512, out_features=num_classes, bias=True)
    elif architecture == "resnet34":
        if pretrained:
            model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        else:
            model = models.resnet34()
        model.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        # model.bn1 = nn.BatchNorm2d(64)
        model.fc = nn.Linear(in_features=512, out_features=num_classes, bias=True)
    elif architecture == "resnet50":
        if pretrained:
            model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        else:
            model = models.resnet50()
        model.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        # model.bn1 = nn.BatchNorm2d(64)
        model.fc = nn.Linear(in_features=2048, out_features=num_classes, bias=True)
    elif architecture == "resnet101":
        if pretrained:
            model = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
        else:
            model = models.resnet101()
        model.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        # model.bn1 = nn.BatchNorm2d(64)
        model.fc = nn.Linear(in_features=2048, out_features=num_classes, bias=True)
    elif architecture == "efficientnet-b0":
        if pretrained:
            model = models.efficientnet_b0(
                weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
            )
        else:
            model = models.efficientnet_b0()
        model.features[0][0] = nn.Conv2d(
            in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False
        )
        model.classifier[1] = nn.Linear(
            in_features=1280, out_features=num_classes, bias=True
        )
    elif architecture == "efficientnet-b1":
        if pretrained:
            model = models.efficientnet_b1(
                weights=models.EfficientNet_B1_Weights.IMAGENET1K_V1
            )
        else:
            model = models.efficientnet_b1()
        model.features[0][0] = nn.Conv2d(
            in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False
        )
        model.classifier[1] = nn.Linear(
            in_features=1280, out_features=num_classes, bias=True
        )
    elif architecture == "efficientnet-b2":
        if pretrained:
            model = models.efficientnet_b2(
                weights=models.EfficientNet_B2_Weights.IMAGENET1K_V1
            )
        else:
            model = models.efficientnet_b2()
        model.features[0][0] = nn.Conv2d(
            in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False
        )
        model.classifier[1] = nn.Linear(
            in_features=1408, out_features=num_classes, bias=True
        )
    elif architecture == "efficientnet-b3":
        if pretrained:
            model = models.efficientnet_b3(
                weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1
            )
        else:
            model = models.efficientnet_b3()
        model.features[0][0] = nn.Conv2d(
            in_channels, 40, kernel_size=3, stride=2, padding=1, bias=False
        )
        model.classifier[1] = nn.Linear(
            in_features=1536, out_features=num_classes, bias=True
        )
    elif architecture == "efficientnet-b4":
        if pretrained:
            model = models.efficientnet_b4(
                weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1
            )
        else:
            model = models.efficientnet_b4()
        model.features[0][0] = nn.Conv2d(
            in_channels, 48, kernel_size=3, stride=2, padding=1, bias=False
        )
        model.classifier[1] = nn.Linear(
            in_features=1792, out_features=num_classes, bias=True
        )
    elif architecture == "efficientnet-b5":
        if pretrained:
            model = models.efficientnet_b5(
                weights=models.EfficientNet_B5_Weights.IMAGENET1K_V1
            )
        else:
            model = models.efficientnet_b5()
        model.features[0][0] = nn.Conv2d(
            in_channels, 48, kernel_size=3, stride=2, padding=1, bias=False
        )
        model.classifier[1] = nn.Linear(
            in_features=2048, out_features=num_classes, bias=True
        )
    else:
        raise ValueError(f'Architecture "{architecture}" not supported.')
    return model


def replace_last_layer(model: torch.nn.Module, architecture: str, num_classes: int = 1):
    if architecture == "cnn":
        model.fc2 = nn.Linear(128, num_classes).cuda()
    elif architecture in (
        "small_resnet14",
        "small_resnet20",
        "small_resnet32",
        "small_resnet44",
        "small_resnet56",
    ):
        model.linear = nn.Linear(64, 1)
    elif architecture in (
        "resnet18",
        "resnet34",
    ):
        model.fc = nn.Linear(
            in_features=512, out_features=num_classes, bias=True
        ).cuda()
    elif architecture == "resnet50":
        model.fc = nn.Linear(
            in_features=2048, out_features=num_classes, bias=True
        ).cuda()
    elif architecture in (
        "efficientnet-b0",
        "efficientnet-b1",
    ):
        model.classifier[1] = nn.Linear(
            in_features=1280, out_features=num_classes, bias=True
        )
    elif architecture == "efficientnet-b2":
        model.classifier[1] = nn.Linear(
            in_features=1408, out_features=num_classes, bias=True
        )
    elif architecture == "efficientnet-b3":
        model.classifier[1] = nn.Linear(
            in_features=1536, out_features=num_classes, bias=True
        )
    elif architecture == "efficientnet-b4":
        model.classifier[1] = nn.Linear(
            in_features=1792, out_features=num_classes, bias=True
        )
    elif architecture == "efficientnet-b5":
        model.classifier[1] = nn.Linear(
            in_features=2048, out_features=num_classes, bias=True
        )
    else:
        raise ValueError(f'Architecture "{architecture}" not supported.')

    return model


class Net(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 10):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.fc1 = nn.Linear(2048, 128)
        self.fc2 = nn.Linear(128, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        # output = F.log_softmax(x, dim=1)
        # return output
        return x


def client_update(
    client_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    epoch: int = 5,
    fedprox_mu: float = 0,
    global_model: Optional[torch.nn.Module] = None,
) -> float:
    """Train a client_model on the train_loder data."""
    criterion = nn.CrossEntropyLoss().cuda()
    client_model.train()

    # for fedprox
    if fedprox_mu > 0 and global_model is not None:
        global_weight_collector = list(global_model.parameters())

    for e in range(epoch):
        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = client_model(data)
            # score = F.log_softmax(output, dim=1)
            # loss = F.nll_loss(score, target)
            loss = criterion(output, target)

            if fedprox_mu > 0 and global_model is not None:
                fedprox_reg = 0.0
                for param_index, param in enumerate(client_model.parameters()):
                    fedprox_reg += (fedprox_mu / 2) * torch.norm(
                        param - global_weight_collector[param_index]
                    ) ** 2

                loss += fedprox_reg

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
    return total_loss / len(train_loader)


def average_models(
    global_model: torch.nn.Module, client_models: List[torch.nn.Module]
) -> None:
    """Average models across all clients."""
    global_dict = global_model.state_dict()
    for k in global_dict.keys():
        global_dict[k] = (
            torch.stack(
                [client_models[i].state_dict()[k] for i in range(len(client_models))],
                0,
            )
            .float()
            .mean(0)
        )
    global_model.load_state_dict(global_dict)


def evaluate_model(
    model: torch.nn.Module,
    data_loader: DataLoader,
    return_logits: bool = False,
    num_batches: int = 1,
) -> Union[Tuple[float, float], Tuple[float, float, torch.Tensor, torch.Tensor]]:
    """Compute loss and accuracy of a single model on a data_loader."""
    model.eval()
    loss = 0
    correct = 0
    total = 0
    logits, targets = [], []
    with torch.no_grad():
        for i, (data, target) in enumerate(data_loader):
            if i > num_batches:
                break
            data, target = data.cuda(), target.cuda()
            output = model(data)
            score = F.log_softmax(output, dim=1)
            loss += F.nll_loss(
                score, target, reduction="sum"
            ).item()  # sum up batch loss
            pred = score.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            # correct += pred.eq(target.view_as(pred)).sum().item()
            correct += (score.argmax(1) == target).sum()
            total += data.shape[0]

            if return_logits:
                logits.append(output.detach().cpu())
                targets.append(target.detach().cpu())

    loss /= (i + 1) * data_loader.batch_size
    acc = correct / total
    loss = loss
    acc = acc.detach().cpu()

    if return_logits:
        return (
            loss,
            acc,
            torch.cat(logits),
            torch.cat(targets),
        )
    else:
        return loss, acc


def evaluate_many_models(
    models: List[torch.nn.Module], data_loader: List[DataLoader], num_batches: int = 1
) -> Tuple[float, float]:
    """Compute average loss and accuracy of multiple models on a data_loader."""
    num_models = len(models)
    total_loss = 0
    total_accuracy = 0
    for i in range(num_models):
        loss, accuracy = evaluate_model(models[i], data_loader, num_batches=num_batches)
        total_loss += loss
        total_accuracy += accuracy
    return total_loss / num_models, total_accuracy / num_models


class Net_eNTK(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 10):
        super(Net_eNTK, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.fc1 = nn.Linear(2048, 128)
        self.fc2 = nn.Linear(128, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


def scaffold_update(
    grads_data: torch.Tensor,
    targets_onehot: torch.Tensor,
    theta_client: torch.Tensor,
    h_i_client_pre: torch.Tensor,
    theta_global: torch.Tensor,
    M: int = 200,
    lr_local: float = 0.00001,
    num_classes: int = 10,
    use_squared_loss: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # set up data / eNTK
    grads_data = grads_data.float().cuda()
    targets_onehot = targets_onehot.cuda()
    theta_client = theta_client.cuda()
    h_i_client_pre = h_i_client_pre.cuda()
    theta_global = theta_global.cuda()

    # compute transformed onehot label
    # targets_onehot = F.one_hot(targets, num_classes=num_classes).cuda() - (
    #    1.0 / num_classes
    # )
    # targets_onehot = targets_onehot - (1.0 / num_classes)
    num_samples = targets_onehot.shape[0]

    # compute updates
    h_i_client_update = h_i_client_pre + (1 / (M * lr_local)) * (
        theta_global - theta_client
    )
    theta_hat_local = (theta_global) * 1.0

    # local gd
    for local_iter in range(M):
        if use_squared_loss:
            theta_hat_local -= lr_local * (
                (1.0 / num_samples)
                * grads_data.t()
                @ (grads_data @ theta_hat_local - targets_onehot)
                - h_i_client_update
            )
        else:
            theta_hat_local -= lr_local * (
                (1.0 / num_samples)
                * grads_data.t()
                @ (torch.softmax(grads_data @ theta_hat_local, 1) - targets_onehot)
                - h_i_client_update
            )

    # del targets
    del targets_onehot
    del grads_data
    del theta_client
    del theta_global
    del h_i_client_pre
    torch.cuda.empty_cache()
    return theta_hat_local, h_i_client_update


def compute_eNTK(
    model: torch.nn.Module,
    loader: DataLoader,
    subsample_size: int = 100000,
    seed: int = 123,
    num_classes: int = 10,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """ "compute eNTK"""
    model.eval()
    params = list(model.parameters())
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("num_params:", num_params)

    random_index = torch.randperm(num_params)[:subsample_size]
    dataset = loader.dataset
    grads = torch.zeros((len(dataset), len(random_index)), dtype=torch.float)
    targets = []

    for i, (x, y) in tqdm(enumerate(dataset)):
        model.zero_grad()
        model.forward(x.unsqueeze(0).cuda())[0].backward()

        grad = []
        for param in params:
            if param.requires_grad:
                grad.append(param.grad.flatten())

        grads[i, :] = torch.cat(grad)[random_index]

        targets.append(y)

    return grads, torch.tensor(targets)
