import argparse
import collections
import copy
import json
import logging
import os
import sys
import time
from pathlib import Path

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import matplotlib.pyplot as plt
import medmnist
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
import torchvision
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, models, transforms
from tqdm import tqdm

from conformal import calibrate_lac, inference_lac
from skin_dataset import (TEST_TRANSFORM, TRAIN_TRANSFORM, SkinDataset,
                          get_weighted_sampler)
from temperature import tune_temp
from utils import (Net, Net_eNTK, average_models, client_update, compute_eNTK,
                   evaluate_many_models, evaluate_model, get_datasets,
                   make_model, partition_dataset, replace_last_layer,
                   scaffold_update)

# plt.style.use("seaborn")


def main():
    """Main script for TCT, FedAvg, and Centrally hosted experiments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_client", default=5, type=int)
    parser.add_argument("--seed", default=123, type=int)
    parser.add_argument("--samples_per_client", default=10000, type=int)
    parser.add_argument("--rounds_stage1", default=100, type=int)
    parser.add_argument("--local_epochs_stage1", default=5, type=int)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--local_lr_stage1", default=0.01, type=float)
    parser.add_argument("--rounds_stage2", default=100, type=int)
    parser.add_argument("--local_steps_stage2", default=500, type=int)
    parser.add_argument("--local_lr_stage2", default=0.0001, type=float)
    parser.add_argument("--data_dir", default="data", type=str)
    parser.add_argument("--save_dir", default="experiments", type=str)
    parser.add_argument("--dataset", default="fashion", type=str)
    parser.add_argument("--architecture", default="cnn", type=str)
    parser.add_argument("--start_from_stage2", action="store_true")
    parser.add_argument("--num_workers", default=16, type=int)
    parser.add_argument("--num_test_samples", default=10000, type=int)
    parser.add_argument("--use_iid_partition", action="store_true")
    parser.add_argument("--central", action="store_true")
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--use_data_augmentation", action="store_true")
    parser.add_argument("--fitzpatrick_csv", default="csv/fitzpatrick.csv", type=str)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--num_random_grad", default=100000, type=int)
    parser.add_argument("--start_from_stage1", action="store_true")
    parser.add_argument("--override", action="store_true")
    parser.add_argument("--tag", default="", type=str)
    parser.add_argument("--use_three_label_partition", action="store_true")
    parser.add_argument("--use_nine_label_partition", action="store_true")
    parser.add_argument("--use_squared_loss", action="store_true")
    parser.add_argument("--use_fedprox", action="store_true")
    parser.add_argument("--fedprox_mu", default=0.1, type=float)
    parser.add_argument("--inference_only", action="store_true")
    parser.add_argument(
        "--fitzpatrick_image_dir",
        default="../data/fitzpatrick17k/images",
        type=str,
    )
    parser.add_argument("--dirichlet_beta", default=0, type=float)
    args = vars(parser.parse_args())

    start_time = time.perf_counter()

    # Set random seed for Repeatability
    seed = args["seed"]
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Hyperparameters
    num_clients = args["num_client"]
    num_rounds_stage1 = args["rounds_stage1"]
    num_rounds_stage2 = args["rounds_stage2"]
    local_epochs_stage1 = args["local_epochs_stage1"]
    lr_stage1 = args["local_lr_stage1"]
    samples_per_client = args["samples_per_client"]
    batch_size = args["batch_size"]
    num_local_steps = args["local_steps_stage2"]
    lr_stage2 = args["local_lr_stage2"]
    architecture = args["architecture"]
    start_from_stage1 = args["start_from_stage1"]
    start_from_stage2 = args["start_from_stage2"]
    num_workers = args["num_workers"]
    num_test_samples = args["num_test_samples"]
    use_iid_partition = args["use_iid_partition"]
    central = args["central"]
    momentum = args["momentum"]
    use_data_augmentation = args["use_data_augmentation"]
    fitzpatrick_csv = args["fitzpatrick_csv"]
    fitzpatrick_image_dir = Path(args["fitzpatrick_image_dir"]).resolve()
    pretrained = args["pretrained"]
    num_random_grad = args["num_random_grad"]
    override = args["override"]
    tag = args["tag"]
    use_three_label_partition = args["use_three_label_partition"]
    use_nine_label_partition = args["use_nine_label_partition"]
    use_squared_loss = args["use_squared_loss"]
    use_fedprox = args["use_fedprox"]
    fedprox_mu = args["fedprox_mu"]
    inference_only = args["inference_only"]
    dirichlet_beta = args["dirichlet_beta"]

    dataset_name = args["dataset"]

    if dataset_name == "mnist":
        in_channels = 1
        num_classes = 10
        client_label_map = {
            "client_0": [0],
            "client_1": [1],
            "client_2": [2],
            "client_3": [3],
            "client_4": [4],
            "client_5": [5],
            "client_6": [6],
            "client_7": [7],
            "client_8": [8],
            "client_9": [9],
        }
        num_clients = len(client_label_map)
    elif dataset_name == "fashion":
        in_channels = 1
        num_classes = 10
        client_label_map = {
            "client_0": [0],
            "client_1": [1],
            "client_2": [2],
            "client_3": [3],
            "client_4": [4],
            "client_5": [5],
            "client_6": [6],
            "client_7": [7],
            "client_8": [8],
            "client_9": [9],
        }
        num_clients = len(client_label_map)
    elif dataset_name == "cifar10":
        in_channels = 3
        num_classes = 10
        client_label_map = {
            "client_0": [0, 1],
            "client_1": [2, 3],
            "client_2": [4, 5],
            "client_3": [6, 7],
            "client_4": [8, 9],
        }
        num_clients = len(client_label_map)
    elif dataset_name == "cifar10-2":
        in_channels = 3
        num_classes = 10
        client_label_map = {
            "client_0": [0, 1, 2],
            "client_1": [2, 3, 4],
            "client_2": [4, 5, 6],
            "client_3": [6, 7, 8],
            "client_4": [8, 9, 0],
        }
        num_clients = len(client_label_map)
    elif dataset_name == "cifar10-3":
        in_channels = 3
        num_classes = 10
        client_label_map = {
            "client_0": [0, 1, 2, 3],
            "client_1": [2, 3, 4, 5],
            "client_2": [4, 5, 6, 7],
            "client_3": [6, 7, 8, 9],
            "client_4": [8, 9, 0, 1],
        }
        num_clients = len(client_label_map)
    elif dataset_name == "cifar100":
        in_channels = 3
        num_classes = 100
        client_label_map = {
            "client_0": [4, 30, 55, 72, 95],
            "client_1": [1, 32, 67, 73, 91],
            "client_2": [54, 62, 70, 82, 92],
            "client_3": [9, 10, 16, 28, 61],
            "client_4": [0, 51, 53, 57, 83],
            "client_5": [22, 39, 40, 86, 87],
            "client_6": [5, 20, 25, 84, 94],
            "client_7": [6, 7, 14, 18, 24],
            "client_8": [3, 42, 43, 88, 97],
            "client_9": [12, 17, 37, 68, 76],
            "client_10": [23, 33, 49, 60, 71],
            "client_11": [15, 19, 21, 31, 38],
            "client_12": [34, 63, 64, 66, 75],
            "client_13": [26, 45, 77, 79, 99],
            "client_14": [2, 11, 35, 46, 98],
            "client_15": [27, 29, 44, 78, 93],
            "client_16": [36, 50, 65, 74, 80],
            "client_17": [47, 52, 56, 59, 96],
            "client_18": [8, 13, 48, 58, 90],
            "client_19": [41, 69, 81, 85, 89],
        }
        # client_label_map = {
        #    "client_0": [0, 1, 2, 3, 4],
        #    "client_1": [5, 6, 7, 8, 9],
        #    "client_2": [10, 11, 12, 13, 14],
        #    "client_3": [15, 16, 17, 18, 19],
        #    "client_4": [20, 21, 22, 23, 24],
        #    "client_5": [25, 26, 27, 28, 29],
        #    "client_6": [30, 31, 32, 33, 34],
        #    "client_7": [35, 36, 37, 38, 39],
        #    "client_8": [40, 41, 42, 43, 44],
        #    "client_9": [45, 46, 47, 48, 49],
        #    "client_10": [50, 51, 52, 53, 54],
        #    "client_11": [55, 56, 57, 58, 59],
        #    "client_12": [60, 61, 62, 63, 64],
        #    "client_13": [65, 66, 67, 68, 69],
        #    "client_14": [70, 71, 72, 73, 74],
        #    "client_15": [75, 76, 77, 78, 79],
        #    "client_16": [80, 81, 82, 83, 84],
        #    "client_17": [85, 86, 87, 88, 89],
        #    "client_18": [90, 91, 92, 93, 94],
        #    "client_19": [95, 96, 97, 98, 99],
        # }
        num_clients = len(client_label_map)
    elif dataset_name == "svhn":
        in_channels = 3
        num_classes = 10
        client_label_map = {
            "client_0": [0, 1],
            "client_1": [2, 3],
            "client_2": [4, 5],
            "client_3": [6, 7],
            "client_4": [8, 9],
        }
        num_clients = len(client_label_map)
    elif dataset_name == "bloodmnist":
        in_channels = 3
        num_classes = 8
        client_label_map = {
            "client_0": [0, 1],
            "client_1": [2, 3],
            "client_2": [4, 5],
            "client_3": [6, 7],
        }
        num_clients = len(client_label_map)
    elif dataset_name == "dermamnist":
        in_channels = 3
        num_classes = 7
        client_label_map = {
            "client_0": [0, 1],
            "client_1": [2, 3],
            "client_2": [4, 5, 6],
        }
        num_clients = len(client_label_map)
    elif dataset_name == "pathmnist":
        in_channels = 3
        num_classes = 9
        client_label_map = {
            "client_0": [0, 1, 2],
            "client_1": [3, 4, 5],
            "client_2": [6, 7, 8],
        }
        num_clients = len(client_label_map)
    elif dataset_name == "tissuemnist":
        in_channels = 1
        num_classes = 8
        client_label_map = {
            "client_0": [0, 1],
            "client_1": [2, 3],
            "client_2": [4, 5],
            "client_3": [6, 7],
        }
        num_clients = len(client_label_map)
    elif dataset_name == "fitzpatrick":
        in_channels = 3
        num_classes = 114
        client_label_map = None
        if use_three_label_partition:
            num_clients = 3
        elif use_nine_label_partition:
            num_clients = 9
        else:
            # num_clients = 6
            num_clients = 11
    else:
        raise ValueError(f'dataset "{dataset_name}" not supported')

    if central:
        experiment = "central"
        num_clients = 1
        local_epochs_stage1 = 1
        num_rounds_stage2 = 0
        client_label_map = {"central_server": list(range(num_classes))}
    elif num_rounds_stage2 == 0 and use_fedprox:
        experiment = "fedprox"
    elif num_rounds_stage2 == 0 and not use_fedprox:
        experiment = "fedavg"
    else:
        experiment = "tct"

    save_name = f"{dataset_name}_{experiment}_{architecture}"

    if use_iid_partition:
        save_name = save_name + "_iid_partition"

    if pretrained:
        save_name = save_name + "_pretrained"

    if use_squared_loss:
        save_name = save_name + "_squared_loss"

    if dirichlet_beta > 0:
        save_name = save_name + f"_dirichlet_{dirichlet_beta}"

    if tag:
        save_name = save_name + f"_{tag}"

    if dataset_name == "fitzpatrick":
        if use_three_label_partition:
            save_name = save_name + "_three_label_partition"
        elif use_nine_label_partition:
            save_name = save_name + "_nine_label_partition"
        else:
            save_name = save_name + "_skin_type_partition"

    # Make directory to write outputs
    save_dir = Path(args["save_dir"]) / save_name
    data_dir = Path(args["data_dir"]).resolve()

    if not override and (save_dir / "finished.txt").exists():
        print("experiment already exists")
        exit()

    save_dir.mkdir(exist_ok=True, parents=True)
    data_dir.mkdir(exist_ok=True, parents=True)

    checkpoint_dir = save_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True, parents=True)

    score_dir = save_dir / "scores"
    score_dir.mkdir(exist_ok=True, parents=True)

    figure_dir = save_dir / "figures"
    figure_dir.mkdir(exist_ok=True, parents=True)

    with open(save_dir / "commands.txt", "w") as f:
        json.dump(args, f, indent=4)

    # Setup logging to write console output to file
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # print to stdout
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)

    # write to file
    file_handler = logging.FileHandler(save_dir / "console.log")
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)

    logger.info(f" {save_name.upper()} ".center(50, "="))

    if dataset_name == "fitzpatrick":
        df = pd.read_csv(fitzpatrick_csv)
        if use_three_label_partition:
            skin_labels = sorted(df.three_partition_label.unique())
        elif use_nine_label_partition:
            skin_labels = sorted(df.nine_partition_label.unique())
        else:
            skin_types = sorted(
                # df.aggregated_fitzpatrick_scale.unique()
                [x for x in df.aggregated_fitzpatrick_scale.unique() if x != -1]
            )
        if central:
            assert 1 == num_clients
        elif use_three_label_partition:
            assert len(skin_labels) == num_clients
        elif use_nine_label_partition:
            assert len(skin_labels) == num_clients
        else:
            assert len(skin_types) == num_clients

        if use_iid_partition:
            train_df = df.query("split == 'train'").sample(frac=1, random_state=seed)
            samples_per_client = round(len(train_df) / num_clients)
            train_partition = {
                str(i): train_df[i * samples_per_client : (i + 1) * samples_per_client]
                for i in range(num_clients)
            }
        else:
            if central:
                train_partition = {"central": df.query("split == 'train'")}
            elif use_three_label_partition:
                train_partition = {
                    str(sl): df.query(
                        "three_partition_label == @sl and split == 'train'"
                    )
                    for sl in skin_labels
                }
            elif use_nine_label_partition:
                train_partition = {
                    str(sl): df.query(
                        "nine_partition_label == @sl and split == 'train'"
                    )
                    for sl in skin_labels
                }
            else:
                train_partition = {
                    str(st): df.query(
                        "aggregated_fitzpatrick_scale == @st and split == 'train'"
                    )
                    for st in skin_types
                }
        val_df = df.query("split == 'val'")
        test_df = df.query("split == 'test'")
        samplers = {
            str(st): get_weighted_sampler(df) for st, df in train_partition.items()
        }

        train_datasets = {
            str(client): SkinDataset(
                image_dir=fitzpatrick_image_dir,
                label_mapping=dict(df[["md5hash", "target"]].values),
                transform=TRAIN_TRANSFORM if use_data_augmentation else TEST_TRANSFORM,
            )
            for client, df in train_partition.items()
        }
        val_map = dict(val_df[["md5hash", "target"]].values)
        val_dataset = SkinDataset(
            image_dir=fitzpatrick_image_dir,
            label_mapping=val_map,
            transform=TEST_TRANSFORM,
        )
        test_map = dict(test_df[["md5hash", "target"]].values)
        test_dataset = SkinDataset(
            image_dir=fitzpatrick_image_dir,
            label_mapping=test_map,
            transform=TEST_TRANSFORM,
        )

        train_loaders = [
            DataLoader(
                ds,
                batch_size=batch_size,
                sampler=samplers[client],
                num_workers=num_workers,
                pin_memory=True,
            )
            for client, ds in train_datasets.items()
        ]
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            shuffle=False,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            shuffle=False,
        )

        val_items = list(val_map.items())
        test_items = list(test_map.items())
        if val_dataset[0][1] != val_items[0][1]:
            raise ValueError(f"{val_dataset[0][1]=} {val_items[0]=}")
        if val_dataset[-1][1] != val_items[-1][1]:
            raise ValueError(f"{val_dataset[-1][1]=} != {val_items[-1]=}")
        if test_dataset[0][1] != test_items[0][1]:
            raise ValueError(f"{test_dataset[0][1]=} != {test_items[0]=}")
        if test_dataset[-1][1] != test_items[-1][1]:
            raise ValueError(f"{test_dataset[-1][1]=} != {test_items[-1]=}")
        val_df.to_csv(save_dir / "val_df.csv")
        test_df.to_csv(save_dir / "test_df.csv")

    else:
        _datasets = get_datasets(
            dataset_name, data_dir, use_data_augmentation=use_data_augmentation
        )
        client_train_datasets = partition_dataset(
            _datasets["train"],
            client_label_map,
            samples_per_client,
            use_iid_partition=use_iid_partition,
            seed=seed,
            dirichlet_beta=dirichlet_beta,
        )
        train_loaders = [
            DataLoader(
                train_subset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True,
            )
            for train_subset in client_train_datasets.values()
        ]

        test_dataset = _datasets["test"]

        if dataset_name in ("bloodmnist", "pathmnist", "tissuemnist"):
            val_dataset = _datasets["val"]
            val_index = torch.randperm(len(val_dataset))[:1000]
            test_index = torch.randperm(len(test_dataset))[:num_test_samples]
            val_subsample = Subset(val_dataset, val_index.tolist())
            test_subsample = Subset(test_dataset, test_index.tolist())
        else:
            val_split: float = 0.1
            num_val = round(val_split * len(test_dataset))
            # index = torch.randperm(len(test_dataset))
            index = torch.arange(len(test_dataset))
            val_index = index[:num_val]
            test_index = index[num_val:][:num_test_samples]
            val_subsample = Subset(test_dataset, val_index.tolist())
            test_subsample = Subset(test_dataset, test_index.tolist())

        val_loader = DataLoader(
            val_subsample,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        test_loader = DataLoader(
            test_subsample,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    for i, loader in enumerate(train_loaders):
        logger.info(
            f"client {i} train samples".ljust(20, "-") + f" {len(loader.dataset)}"
        )
    logger.info("val samples".ljust(20, "-") + f" {len(val_loader.dataset)}")
    logger.info("test samples".ljust(20, "-") + f" {len(test_loader.dataset)}")

    if not start_from_stage2:
        logger.info("===================== Start Stage-1 =====================")

        # Instantiate models and optimizers
        global_model = make_model(architecture, in_channels, num_classes, pretrained=pretrained).cuda()
        client_models = [
            make_model(
                architecture, in_channels, num_classes, pretrained=pretrained
            ).cuda()
            for _ in range(num_clients)
        ]

        if start_from_stage1 or inference_only:
            checkpoint = max(
                checkpoint_dir.glob(f"{save_name}_stage1_model_*.pth"),
                key=lambda x: int(x.stem.split("_")[-1]),
            )
            global_model.load_state_dict(torch.load(checkpoint))
            r = int(checkpoint.stem.split("_")[-1])
            logger.info(f"starting stage 1 with {checkpoint} at round {r}")
        else:
            r = 1

        for model in client_models:
            model.load_state_dict(global_model.state_dict())
        opt = [
            optim.SGD(
                model.parameters(),
                lr=lr_stage1,
                momentum=momentum,
            )
            for model in client_models
        ]

        logger.info(f"{len(client_models)=}")
        logger.info(f"{len(train_loaders)=}")
        logger.info(f"{opt[0]=}")

        if inference_only:
            val_loss, val_acc, val_scores, val_targets = evaluate_model(
                global_model,
                val_loader,
                num_batches=len(val_loader),
                return_logits=True,
            )
            test_loss, test_acc, test_scores, test_targets = evaluate_model(
                global_model,
                test_loader,
                num_batches=len(test_loader),
                return_logits=True,
            )
            logger.info(
                f"global model -- {val_loss=:.3f} -- {val_acc=:.3f} -- {test_loss=:.3f} -- {test_acc=:.3f}"
            )

            torch.save(val_scores, score_dir / f"{save_name}_stage1_val_scores.pth")
            torch.save(val_targets, score_dir / f"{save_name}_stage1_val_targets.pth")
            torch.save(test_scores, score_dir / f"{save_name}_stage1_test_scores.pth")
            torch.save(test_targets, score_dir / f"{save_name}_stage1_test_targets.pth")
            exit()

        stage1_loss = collections.defaultdict(list)
        stage1_accuracy = collections.defaultdict(list)
        stage1_max_score = collections.defaultdict(list)

        # Run TCT-Stage1 (i.e., FedAvg)
        for r in range(r, num_rounds_stage1 + 1):

            # load global weights
            for model in client_models:
                model.load_state_dict(global_model.state_dict())

            # client update
            loss = 0
            for i in range(num_clients):
                loss += client_update(
                    client_models[i],
                    opt[i],
                    train_loaders[i],
                    epoch=local_epochs_stage1,
                    fedprox_mu=fedprox_mu,
                    global_model=global_model if use_fedprox else None,
                )
            loss /= num_clients

            # average params across neighbors
            average_models(global_model, client_models)

            # save global model
            if r % 25 == 0:
                torch.save(
                    global_model.state_dict(),
                    checkpoint_dir / f"{save_name}_stage1_model_{r}.pth",
                )

            # evaluate clients
            clients_train_loss, clients_train_acc = 0, 0
            for i in range(num_clients):
                train_loss, train_acc = evaluate_model(
                    client_models[i],
                    train_loaders[i],
                    num_batches=8,
                )
                clients_train_loss += train_loss
                clients_train_acc += train_acc
            clients_train_loss /= num_clients
            clients_train_acc /= num_clients

            stage1_loss["clients_train"].append(clients_train_loss)
            stage1_accuracy["clients_train"].append(clients_train_acc)

            # save mean and variance of client accuracy and max score
            clients_test_loss = 0
            clients_test_acc = []
            clients_test_max_score = []
            for i, model in enumerate(client_models):
                test_loss, test_acc, logits, _ = evaluate_model(
                    model,
                    test_loader,
                    num_batches=12,
                    return_logits=True,
                )
                clients_test_loss += test_loss
                clients_test_acc.append(test_acc)
                clients_test_max_score.append(torch.softmax(logits, 1).max(1).values)

            clients_test_loss /= num_clients
            clients_test_max_score = torch.cat(clients_test_max_score).tolist()

            stage1_loss["clients_test"].append(clients_test_loss)
            stage1_accuracy["clients_test_mean"].append(
                float(np.mean(clients_test_acc))
            )
            stage1_accuracy["clients_test_std"].append(float(np.std(clients_test_acc)))

            stage1_max_score["clients_test_mean"].append(
                float(np.mean(clients_test_max_score))
            )
            stage1_max_score["clients_test_std"].append(
                float(np.std(clients_test_max_score))
            )

            # evaluate global model
            global_train_loss, global_train_acc = 0, 0
            for i in range(num_clients):
                train_loss, train_acc = evaluate_model(
                    global_model, train_loaders[i], num_batches=8
                )
                global_train_loss += train_loss
                global_train_acc += train_acc
            global_train_loss /= num_clients
            global_train_acc /= num_clients

            global_test_loss, global_test_acc = evaluate_model(
                global_model,
                test_loader,
                num_batches=8,
            )

            stage1_loss["global_train"].append(global_train_loss)
            stage1_loss["global_test"].append(global_test_loss)
            stage1_accuracy["global_train"].append(global_train_acc)
            stage1_accuracy["global_test"].append(global_test_acc)

            logger.info(
                f"{str(r).zfill(3)} == clients =="
                f" {loss:.3f} train loss {clients_train_loss:.3f} "
                f"| test loss {clients_test_loss:.3f} "
                f"| train acc {clients_train_acc:.3f} "
                f"| test acc {np.mean(clients_test_acc):.3f} "
                f"   == global model =="
                f"  train loss {global_train_loss:.3f} "
                f"| test loss {global_test_loss:.3f} "
                f"| train acc {global_train_acc:.3f} "
                f"| test acc {global_test_acc:.3f} "
            )

            if r % 50 == 0:
                val_loss, val_acc, val_scores, val_targets = evaluate_model(
                    global_model,
                    val_loader,
                    num_batches=len(val_loader),
                    return_logits=True,
                )
                test_loss, test_acc, test_scores, test_targets = evaluate_model(
                    global_model,
                    test_loader,
                    num_batches=len(test_loader),
                    return_logits=True,
                )
                logger.info(
                    f"global model -- {val_loss=:.3f} -- {val_acc=:.3f} -- {test_loss=:.3f} -- {test_acc=:.3f}"
                )

                torch.save(val_scores, score_dir / f"{save_name}_stage1_val_scores.pth")
                torch.save(
                    val_targets, score_dir / f"{save_name}_stage1_val_targets.pth"
                )
                torch.save(
                    test_scores, score_dir / f"{save_name}_stage1_test_scores.pth"
                )
                torch.save(
                    test_targets, score_dir / f"{save_name}_stage1_test_targets.pth"
                )

        with open(save_dir / "history.json", "w") as f:
            json.dump(
                dict(
                    loss=stage1_loss,
                    accuracy=stage1_accuracy,
                    score=stage1_max_score,
                ),
                f,
                indent=4,
                default=float,
            )

        fig, ax = plt.subplots(ncols=2, figsize=(18, 6))
        fontsize = 24
        ax[0].plot(stage1_loss["clients_train"], ":", label="clients_train")
        ax[0].plot(stage1_loss["clients_test"], ":", label="clients_test")
        ax[0].plot(stage1_loss["global_train"], "--", label="global_train")
        ax[0].plot(stage1_loss["global_test"], "--", label="global_test")
        ax[0].set_xlabel("round", fontsize=fontsize)
        ax[0].set_ylabel("loss", fontsize=fontsize)
        ax[0].legend(fontsize=fontsize)
        ax[1].plot(stage1_accuracy["clients_train"], ":", label="clients_train")
        ax[1].plot(stage1_accuracy["clients_test_mean"], ":", label="clients_test")
        ax[1].plot(stage1_accuracy["global_train"], "--", label="global_train")
        ax[1].plot(stage1_accuracy["global_test"], "--", label="global_test")
        ax[1].set_xlabel("round", fontsize=fontsize)
        ax[1].set_ylabel("accuracy", fontsize=fontsize)
        ax[1].legend(fontsize=fontsize)
        plt.savefig(figure_dir / f"{save_name}_stage1_loss_curve.png")

    if num_rounds_stage2 != 0 and experiment == "tct":

        # turn off data augmentation in training datasets
        if dataset_name == "fitzpatrick":
            if use_iid_partition:
                train_df = df.query("split == 'train'").sample(
                    frac=1, random_state=seed
                )
                samples_per_client = round(len(train_df) / num_clients)
                train_partition = {
                    str(i): train_df[
                        i * samples_per_client : (i + 1) * samples_per_client
                    ]
                    for i in range(num_clients)
                }
            elif use_three_label_partition:
                train_partition = {
                    str(sl): df.query(
                        "three_partition_label == @sl and split == 'train'"
                    )
                    for sl in skin_labels
                }
            elif use_nine_label_partition:
                train_partition = {
                    str(sl): df.query(
                        "nine_partition_label == @sl and split == 'train'"
                    )
                    for sl in skin_labels
                }
            else:
                train_partition = {
                    str(st): df.query(
                        "aggregated_fitzpatrick_scale == @st and split == 'train'"
                    )
                    for st in skin_types
                }
            train_datasets = {
                str(st): SkinDataset(
                    image_dir=fitzpatrick_image_dir,
                    label_mapping=dict(df[["md5hash", "target"]].values),
                    transform=TEST_TRANSFORM,
                )
                for st, df in train_partition.items()
            }
            train_loaders = [
                DataLoader(
                    ds,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=True,
                )
                for ds in train_datasets.values()
            ]
        else:
            _datasets = get_datasets(
                dataset_name, data_dir, use_data_augmentation=False
            )
            client_train_datasets = partition_dataset(
                _datasets["train"],
                client_label_map,
                samples_per_client,
                use_iid_partition=use_iid_partition,
                seed=seed,
                dirichlet_beta=dirichlet_beta,
            )
            train_loaders = [
                DataLoader(
                    train_subset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=True,
                )
                for train_subset in client_train_datasets.values()
            ]

        logger.info("===================== Start Stage-2 =====================")
        checkpoint = (
            checkpoint_dir / f"{save_name}_stage1_model_{num_rounds_stage1}.pth"
        )
        logger.info(f"{checkpoint=}")

        # Init and load model ckpt
        global_model = make_model(architecture, in_channels, num_classes).cuda()
        global_model.load_state_dict(torch.load(checkpoint, map_location='cpu'))
        global_model = global_model.cuda()

        client_train_loss, client_train_acc = 0, 0
        for i in range(num_clients):
            train_loss, train_acc, train_scores, train_targets = evaluate_model(
                global_model,
                train_loaders[i],
                num_batches=len(train_loaders[i]),
                return_logits=True,
            )
            client_train_loss += train_loss
            client_train_acc += train_acc

        client_train_loss /= num_clients
        client_train_acc /= num_clients

        test_loss, test_acc, test_scores, test_targets = evaluate_model(
            global_model,
            test_loader,
            return_logits=True,
            num_batches=len(test_loader),
        )
        logger.info(
            f"stage2 model -- {client_train_loss=:.3f} -- "
            + f"{client_train_acc=:.3f} -- {test_loss=:.3f} -- {test_acc=:.3f}"
        )

        global_model = replace_last_layer(global_model, architecture, num_classes=1)
        global_model = global_model.cuda()
        logger.info("loaded eNTK model")

        logger.info("Compute eNTK representations")
        # Train
        grad_all = []
        target_all = []
        for i in range(num_clients):
            grad_i, target_i = compute_eNTK(
                global_model,
                train_loaders[i],
                subsample_size=num_random_grad,
                seed=seed,
                num_classes=num_classes,
            )
            grad_all.append(copy.deepcopy(grad_i).cpu())
            target_all.append(copy.deepcopy(target_i).cpu())
            del grad_i
            del target_i
            torch.cuda.empty_cache()

        assert len(grad_all) == num_clients

        # For calibration
        grad_val, target_val = compute_eNTK(
            global_model,
            val_loader,
            subsample_size=num_random_grad,
            seed=seed,
            num_classes=num_classes,
        )
        # Test
        grad_test, target_test = compute_eNTK(
            global_model,
            test_loader,
            subsample_size=num_random_grad,
            seed=seed,
            num_classes=num_classes,
        )

        # normalization
        logger.info("normalization")
        scaler = StandardScaler()
        scaler.fit(torch.cat(grad_all).cpu().numpy())
        for idx in range(len(grad_all)):
            grad_all[idx] = torch.from_numpy(
                scaler.transform(grad_all[idx].cpu().numpy())
            )
        grad_test = torch.from_numpy(scaler.transform(grad_test.cpu().numpy()))
        grad_val = torch.from_numpy(scaler.transform(grad_val.cpu().numpy()))

        # Init linear models
        theta_global = torch.zeros(num_random_grad, num_classes)
        theta_global = torch.tensor(theta_global, requires_grad=False)
        client_thetas = [torch.zeros_like(theta_global) for _ in range(num_clients)]
        client_hi_s = [torch.zeros_like(theta_global) for _ in range(num_clients)]

        train_targets = []
        for i in range(num_clients):
            train_targets.append(F.one_hot(target_all[i], num_classes=num_classes))

        # Run TCT-Stage2
        for round_idx in range(1, num_rounds_stage2 + 1):
            theta_list = []
            for i in range(num_clients):
                theta_hat_update, h_i_client_update = scaffold_update(
                    grad_all[i],
                    train_targets[i],
                    client_thetas[i],
                    client_hi_s[i],
                    theta_global,
                    M=num_local_steps,
                    lr_local=lr_stage2,
                    num_classes=num_classes,
                    use_squared_loss=use_squared_loss,
                )
                client_hi_s[i] = h_i_client_update * 1.0
                client_thetas[i] = theta_hat_update * 1.0
                theta_list.append(theta_hat_update)

            # averaging
            theta_global = torch.zeros_like(theta_list[0])
            for theta_idx in range(num_clients):
                theta_global += (1.0 / num_clients) * theta_list[theta_idx]

            # eval on train
            logits_class_train = torch.cat(grad_all).cpu() @ theta_global.cpu()
            target_train = torch.cat(target_all).cpu()
            train_acc = (
                (logits_class_train.argmax(1) == target_train.cpu()).sum()
                / logits_class_train.shape[0]
            ).item()
            train_max_score = (
                torch.softmax(logits_class_train, 1).max(1).values.mean().item()
            )

            # eval on val
            logits_class_val = grad_val.cpu() @ theta_global.cpu()

            # eval on test
            logits_class_test = grad_test.cpu() @ theta_global.cpu()
            test_acc = (
                (logits_class_test.argmax(1) == target_test.cpu()).sum()
                / logits_class_test.shape[0]
            ).item()
            T = tune_temp(logits_class_val, target_val)
            # T = 1
            val_scores = torch.softmax(logits_class_val / T, 1)
            test_scores = torch.softmax(logits_class_test / T, 1)
            q_10 = calibrate_lac(val_scores, target_val, alpha=0.1)
            q_20 = calibrate_lac(val_scores, target_val, alpha=0.2)
            q_30 = calibrate_lac(val_scores, target_val, alpha=0.3)
            psets_10 = inference_lac(test_scores, q_10)
            psets_20 = inference_lac(test_scores, q_20)
            psets_30 = inference_lac(test_scores, q_30)
            size_10 = psets_10.sum(1).float().mean().item()
            size_20 = psets_20.sum(1).float().mean().item()
            size_30 = psets_30.sum(1).float().mean().item()

            test_max_score = test_scores.max(1).values.mean().item()

            logger.info(
                f"{round_idx=}: {train_acc=:.3f} {test_acc=:.3f}   "
                + f" {train_max_score=:.3f} {test_max_score=:.3f}  "
                + f" {q_10=:.3f} {size_10=:.1f}"
                + f" {q_20=:.3f} {size_20=:.1f}"
                + f" {q_30=:.3f} {size_30=:.1f}"
            )

            if round_idx % 25 == 0 or round_idx == num_rounds_stage2:
                torch.save(
                    logits_class_train,
                    score_dir / f"{save_name}_stage2_train_scores.pth",
                )
                torch.save(
                    target_train, score_dir / f"{save_name}_stage2_train_targets.pth"
                )
                torch.save(
                    logits_class_val, score_dir / f"{save_name}_stage2_val_scores.pth"
                )
                torch.save(
                    target_val, score_dir / f"{save_name}_stage2_val_targets.pth"
                )
                torch.save(
                    logits_class_test, score_dir / f"{save_name}_stage2_test_scores.pth"
                )
                torch.save(
                    target_test, score_dir / f"{save_name}_stage2_test_targets.pth"
                )

        logger.info(" Finished TCT Stage-2 ".center(20, "="))

    end_time = time.perf_counter()
    total_runtime = end_time - start_time
    logger.info(f"total runtime {total_runtime:.0f}".center(40, "="))

    with open(save_dir / "finished.txt", "w") as f:
        f.write(f"total runtime: {total_runtime:.0f}")


if __name__ == "__main__":
    print('cuda', torch.cuda.is_available())
    print(torch.cuda.device_count())
    main()
