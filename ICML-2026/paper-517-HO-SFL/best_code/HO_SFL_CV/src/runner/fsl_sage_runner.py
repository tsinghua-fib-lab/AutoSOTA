import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import random
import wandb
from copy import deepcopy
from typing import Dict, Tuple

from torchvision.models.resnet import BasicBlock, conv1x1
from src.models.registry import get_model
from src.models.split_wrapper import split_model
from src.core.communicator import Communicator

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models.resnet import BasicBlock, conv1x1


class ResNet18Auxiliary(nn.Module):
    def __init__(
        self,
        in_planes=128,
        num_classes=10,
        align_epochs=5,
        align_batch_size=100,
        max_dataset_size=1000,
        lr=1e-3,
        device="cpu",
    ):
        super(ResNet18Auxiliary, self).__init__()

        self.device = device
        self.align_epochs = align_epochs
        self.align_batch_size = align_batch_size
        self.max_dataset_size = max_dataset_size
        self.lr = lr

        self.inplanes = in_planes
        self._norm_layer = nn.BatchNorm2d

        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256 * BasicBlock.expansion, num_classes)

        self.data_x = torch.tensor([], device=device)
        self.data_labels = torch.tensor([], device=device)

        self.data_y = None

        self.to(device)
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

    def _make_layer(self, block, planes, blocks, stride=1):
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )
        layers = []
        layers.append(
            block(self.inplanes, planes, stride, downsample, 1, 64, 1, norm_layer)
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, None, 1, 64, 1, norm_layer))
        return nn.Sequential(*layers)

    def forward_inner(self, x):
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def forward(self, x, label, server_criterion):
        x = x.requires_grad_(True)
        preds = self.forward_inner(x)
        loss = server_criterion(preds, label)
        aux_grad = torch.autograd.grad(loss, x, create_graph=True)[0]
        return aux_grad

    def add_datapoint(self, x, label):
        x = x.detach()
        label = label.detach()

        if self.data_x.device != x.device:
            self.data_x = self.data_x.to(x.device)
        if self.data_labels.device != label.device:
            self.data_labels = self.data_labels.to(label.device)

        self.data_x = torch.cat((self.data_x, x), dim=0)
        self.data_labels = torch.cat((self.data_labels, label), dim=0).long()

        if self.data_x.shape[0] > self.max_dataset_size:
            self.data_x = self.data_x[-self.max_dataset_size :]
            self.data_labels = self.data_labels[-self.max_dataset_size :]

    def refresh_data(self, server_model, server_criterion):
        if self.data_x.shape[0] == 0:
            self.data_y = torch.tensor([], device=self.device)
            return

        with torch.enable_grad():
            server_model.eval()
            server_model.zero_grad()

            all_ins = self.data_x.clone().detach().to(self.device).requires_grad_(True)

            labels = self.data_labels.to(self.device)

            all_out = server_model(all_ins)
            all_loss = server_criterion(all_out, labels)

            all_loss.backward()
            self.data_y = all_ins.grad.clone().detach()
            server_model.zero_grad()

    def align(self, server_criterion):

        if self.data_x.shape[0] == 0 or self.data_y is None:
            return 0.0

        dataset = torch.utils.data.TensorDataset(
            self.data_x, self.data_y, self.data_labels
        )
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.align_batch_size, shuffle=True
        )

        self.train()
        total_loss = 0.0
        steps = 0

        for _ in range(self.align_epochs):
            for x, y_true, labels in loader:
                x, y_true, labels = (
                    x.to(self.device),
                    y_true.to(self.device),
                    labels.to(self.device),
                )

                self.optimizer.zero_grad()
                grad_est = self.forward(x, labels, server_criterion)
                loss = nn.functional.mse_loss(grad_est, y_true, reduction="sum")
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                steps += 1

        return total_loss / max(1, steps)

    def get_state(self):
        return {
            "state_dict": {k: v.cpu() for k, v in self.state_dict().items()},
            "data_x": self.data_x.cpu(),
            "data_labels": self.data_labels.cpu(),
            "optimizer": self.optimizer.state_dict(),
        }

    def load_state(self, state):

        self.load_state_dict(state["state_dict"])

        self.data_x = state["data_x"].to(self.device)
        self.data_labels = state["data_labels"].to(self.device)

        self.optimizer.load_state_dict(state["optimizer"])


class FSLSAGERunner:
    def __init__(self, config, data_manager):
        self.cfg = config
        self.data_manager = data_manager
        self.device = torch.device(config.system.device)
        self.communicator = Communicator()

        print(
            "Initializing FSL-SAGE Runner (Aligned with Local Epochs & Independent Aux)"
        )

        self.total_rounds = config.runner.communication_rounds
        self.local_epochs = config.runner.local_epochs
        self.server_update_interval = config.algo.get("server_update_interval", 5)
        self.align_interval = config.algo.get("align_interval", 1)

        self.align_epochs = config.algo.get("align_epochs", 5)
        self.aux_lr = config.algo.get("aux_lr", 1e-3)
        self.server_lr = config.algo.get("server_lr", 1e-3)
        self.max_aux_data_size = config.algo.get("alignment_buffer_size", 1000)

        full_model = get_model(config)
        client_part, server_part = split_model(full_model, config.algo.split_point)

        self.client_model_template = client_part.to(self.device)
        self.server_model_template = server_part.to(self.device)

        self.server_optimizer = self._get_specific_optimizer(
            self.server_model_template.parameters(), lr=self.server_lr
        )
        self.criterion = nn.CrossEntropyLoss()

        dummy_input = torch.randn(1, *config.data.input_shape).to(self.device)
        with torch.no_grad():
            cut_layer_out = client_part(dummy_input)
            in_planes = cut_layer_out.shape[1]

        self.aux_kwargs = {
            "in_planes": in_planes,
            "num_classes": config.data.num_classes,
            "align_epochs": self.align_epochs,
            "max_dataset_size": self.max_aux_data_size,
            "lr": self.aux_lr,
            "device": self.device,
        }

        self.global_client_weights = deepcopy(client_part.state_dict())
        self.global_server_weights = deepcopy(server_part.state_dict())

        self.client_aux_states: Dict[int, dict] = {}

        self.num_active = config.runner.sampled_clients
        self.active_client_models = [
            deepcopy(self.client_model_template) for _ in range(self.num_active)
        ]

        self.active_aux_models = [None] * self.num_active

        self.client_optimizers = [None] * self.num_active
        self.best_acc = 0.0

        self.client_iterators = {}

    def _get_optimizer(self, params):
        return self._get_specific_optimizer(params, lr=self.cfg.algo.client_lr)

    def _get_specific_optimizer(self, params, lr):
        if self.cfg.algo.optimizer == "adamw":
            return optim.AdamW(params, lr=lr)
        elif self.cfg.algo.optimizer == "sgd":
            return optim.SGD(params, lr=lr)
        else:
            return optim.Adam(params, lr=lr)

    def sample_clients(self):
        total_clients = self.cfg.runner.num_clients
        return np.random.choice(total_clients, self.num_active, replace=False)

    def _get_next_batch(self, cid):
        if cid not in self.client_iterators:
            loader = self.data_manager.get_client_loader(cid)
            self.client_iterators[cid] = (loader, iter(loader))

        loader, iterator = self.client_iterators[cid]
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
            self.client_iterators[cid] = (loader, iterator)
        return batch

    def _load_aux_model(self, cid) -> ResNet18Auxiliary:
        aux_model = ResNet18Auxiliary(**self.aux_kwargs)

        if cid in self.client_aux_states:
            aux_model.load_state(self.client_aux_states[cid])

        return aux_model

    def _save_aux_model(self, cid, aux_model: ResNet18Auxiliary):
        self.client_aux_states[cid] = aux_model.get_state()

    def run(self):
        print(
            f"Start Training: {self.total_rounds} rounds, Local Epochs: {self.local_epochs}, Align Interval: {self.align_interval}, Server Update Interval: {self.server_update_interval}"
        )
        wandb_table = None
        self.communicator.reset()

        self.server_model_template.load_state_dict(self.global_server_weights)

        for round_idx in range(self.total_rounds):
            sampled_ids = self.sample_clients()
            active_client_samples = []

            avg_server_loss = 0
            total_server_updates = 0

            for i, cid in enumerate(sampled_ids):
                self.active_client_models[i].load_state_dict(self.global_client_weights)
                self.client_optimizers[i] = self._get_optimizer(
                    self.active_client_models[i].parameters()
                )

                self.active_aux_models[i] = self._load_aux_model(cid)

                active_client_samples.append(
                    self.data_manager.get_client_sample_count(cid)
                )

            if round_idx % self.align_interval == 0:
                for i, cid in enumerate(sampled_ids):
                    aux_model = self.active_aux_models[i]

                    aux_model.refresh_data(self.server_model_template, self.criterion)

                    align_loss = aux_model.align(self.criterion)
                    self.communicator.log_model_download(aux_model)

            for i, cid in enumerate(sampled_ids):
                client_model = self.active_client_models[i]
                aux_model = self.active_aux_models[i]
                client_opt = self.client_optimizers[i]

                client_model.train()
                aux_model.train()

                steps_per_epoch = self.data_manager.get_client_steps_per_epoch(cid)
                total_local_steps = steps_per_epoch * self.local_epochs

                for step in range(total_local_steps):
                    inputs, labels = self._get_next_batch(cid)
                    if inputs.size(0) == 1:
                        continue
                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                    self.communicator.log_proceeded_samples(inputs)

                    client_opt.zero_grad()
                    z_f = client_model(inputs)

                    local_smashed_data = z_f.clone().detach().requires_grad_(True)

                    if step % self.server_update_interval == 0:
                        smashed_data = z_f.clone().detach().requires_grad_(True)
                        self.communicator.log_activation(smashed_data)
                        self.server_optimizer.zero_grad()

                        out = self.server_model_template(smashed_data)
                        s_loss = self.criterion(out, labels)

                        s_loss.backward()
                        self.server_optimizer.step()

                        avg_server_loss += s_loss.item()
                        total_server_updates += 1

                        aux_model.add_datapoint(z_f, labels)

                    aux_grad = aux_model(local_smashed_data, labels, self.criterion)

                    z_f.backward(aux_grad)
                    client_opt.step()

                self.communicator.log_model_upload(client_model)
                self._save_aux_model(cid, aux_model)

            self.global_client_weights = self._aggregate_weights(
                self.active_client_models,
                active_client_samples,
                self.global_client_weights,
            )

            self.global_server_weights = deepcopy(
                self.server_model_template.state_dict()
            )

            mean_server_loss = (
                avg_server_loss / total_server_updates
                if total_server_updates > 0
                else 0
            )
            print(
                f"Round {round_idx+1}/{self.total_rounds} | Server Loss: {mean_server_loss:.4f}"
            )

            if (round_idx + 1) % self.cfg.runner.evaluation_interval == 0 or (
                stats["proceeded_samples"] >= self.cfg.runner.total_samples
            ):
                acc = self.evaluate()
                if acc > self.best_acc:
                    self.best_acc = acc
            else:
                acc = None

            if self.cfg.logging.use_wandb:
                stats = self.communicator.get_stats()
                row_dict = {
                    "Round": round_idx + 1,
                    "Train/Server_Loss": mean_server_loss,
                    "Val/Accuracy": acc,
                    "Val/Best_Accuracy": self.best_acc,
                }
                for k, v in stats.items():
                    if k != "proceeded_samples":
                        row_dict[f"Comm/{k}_MB"] = v / (1024**2)
                    else:
                        row_dict[f"Comm/{k}"] = v

                wandb.log(row_dict, step=round_idx + 1)
                if wandb_table is None:
                    wandb_table = wandb.Table(columns=list(row_dict.keys()))
                wandb_table.add_data(*row_dict.values())

                if stats["proceeded_samples"] >= self.cfg.runner.total_samples:
                    print(
                        f"Reached total proceeded samples: {stats['proceeded_samples']}. Stopping training."
                    )
                    break
        if self.cfg.logging.use_wandb and wandb_table is not None:
            self._save_wandb_history(wandb_table)

    def _aggregate_weights(self, model_list, sample_counts, ref_state_dict):
        total_samples = sum(sample_counts)
        weighted_state = {}

        proxy_model = model_list[0]
        frozen_params = set()
        for name, param in proxy_model.named_parameters():
            if not param.requires_grad:
                frozen_params.add(name)

        keys_to_aggregate = []
        for key in ref_state_dict:
            if not torch.is_floating_point(ref_state_dict[key]):
                weighted_state[key] = ref_state_dict[key].to(self.device)
                continue
            if key in frozen_params:
                weighted_state[key] = ref_state_dict[key].to(self.device)
                continue
            weighted_state[key] = torch.zeros_like(
                ref_state_dict[key], dtype=torch.float32, device=self.device
            )
            keys_to_aggregate.append(key)

        for i, model in enumerate(model_list):
            weight = sample_counts[i] / total_samples
            state = model.state_dict()
            for key in keys_to_aggregate:
                weighted_state[key] += state[key].float() * weight

        final_state = {}
        for key in ref_state_dict:
            final_state[key] = weighted_state[key].to(ref_state_dict[key].dtype)
        return final_state

    def evaluate(self):
        test_client = deepcopy(self.client_model_template)
        test_client.load_state_dict(self.global_client_weights)
        test_client.to(self.device)
        test_client.eval()

        test_server = deepcopy(self.server_model_template)
        test_server.load_state_dict(self.global_server_weights)
        test_server.to(self.device)
        test_server.eval()

        test_loader = self.data_manager.get_global_test_loader()
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                feat = test_client(inputs)
                out = test_server(feat)
                _, predicted = torch.max(out.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        acc = 100 * correct / total
        print(f"Validation Accuracy: {acc:.2f}%")
        return acc

    def _save_wandb_history(self, wandb_table):
        random.seed(None)
        print("Uploading detailed run history table to WandB...")
        wandb.log({"Detailed_Run_History": wandb_table})
        log_data = wandb_table.data
        log_columns = wandb_table.columns
        df = pd.DataFrame(data=log_data, columns=log_columns)

        log_name = (
            f"{self.cfg.algo.algo_name}-"
            f"{self.cfg.data.dataset}-"
            f"{self.cfg.data.partition.algo}-"
            f"{self.cfg.seed}"
        )
        log_path = f"{self.cfg.logging.log_dir}/{log_name}_log.csv"
        df.to_csv(log_path, index=False)
