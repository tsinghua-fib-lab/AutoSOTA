import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import random
import wandb
from copy import deepcopy

from src.models.registry import get_model
from src.models.split_wrapper import split_model
from src.core.communicator import Communicator


class SFLZORunner:
    def __init__(self, config, data_manager):
        self.cfg = config
        self.data_manager = data_manager
        self.device = torch.device(config.system.device)
        self.communicator = Communicator()

        print("Initializing ZO-SFL Runner")

        self.evaluation_interval = config.runner.get("evaluation_interval", 10)

        self.P = config.algo.get("zo_p", 1)
        self.mu = config.algo.get("zo_mu", 1e-3)

        full_model = get_model(config)
        client_part, server_part = split_model(full_model, config.algo.split_point)

        self.global_client_weights = deepcopy(client_part.state_dict())
        self.global_server_weights = deepcopy(server_part.state_dict())

        self.client_model_template = client_part
        self.server_model_template = server_part

        self.num_active = config.runner.sampled_clients

        self.active_client_models = []
        self.active_server_models = []
        self.best_acc = 0.0

        for _ in range(self.num_active):
            self.active_client_models.append(deepcopy(client_part).to(self.device))
            self.active_server_models.append(deepcopy(server_part).to(self.device))

        self.client_optimizers = [None] * self.num_active
        self.server_optimizers = [None] * self.num_active

        self.criterion = nn.CrossEntropyLoss()

    def _get_optimizer(self, params):
        if self.cfg.algo.optimizer == "adamw":
            return optim.AdamW(
                params,
                lr=self.cfg.algo.lr,
                betas=self.cfg.algo.get("betas", (0.9, 0.999)),
                weight_decay=self.cfg.algo.get("weight_decay", 1e-4),
            )
        elif self.cfg.algo.optimizer == "sgd":
            return optim.SGD(
                params,
                lr=self.cfg.algo.lr,
                momentum=self.cfg.algo.get("momentum", 0.9),
                weight_decay=self.cfg.algo.get("weight_decay", 1e-4),
            )
        else:
            raise ValueError(f"Unknown optimizer type: {self.cfg.algo.optimizer}")

    def sample_clients(self):
        total_clients = self.cfg.runner.num_clients
        return np.random.choice(total_clients, self.num_active, replace=False)

    def _generate_perturbation_and_apply(self, model, seed, scale_factor):
        rng_state = torch.get_rng_state()
        torch.manual_seed(seed)
        with torch.no_grad():
            for param in model.parameters():
                if not param.requires_grad:
                    continue
                u = torch.randn_like(param)
                param.add_(u, alpha=scale_factor)
        torch.set_rng_state(rng_state)

    def _generate_perturbation_and_accumulate_grad(self, model, seed, scalar_weight):
        rng_state = torch.get_rng_state()
        torch.manual_seed(seed)
        with torch.no_grad():
            for param in model.parameters():
                if not param.requires_grad:
                    continue
                u = torch.randn_like(param)
                if param.grad is None:
                    param.grad = torch.zeros_like(param)
                param.grad.add_(u, alpha=scalar_weight)
        torch.set_rng_state(rng_state)

    def run(self):
        total_rounds = self.cfg.runner.communication_rounds
        local_epochs = self.cfg.runner.local_epochs
        print(
            f"Start Training ZO-SFL for {total_rounds} rounds with {local_epochs} local epochs..."
        )

        wandb_table = None
        self.communicator.reset()

        for round_idx in range(total_rounds):
            sampled_ids = self.sample_clients()
            active_client_samples = []
            avg_loss = 0
            total_steps_in_round = 0

            for i in range(self.num_active):
                self.active_client_models[i].load_state_dict(self.global_client_weights)
                self.communicator.log_model_download(self.active_client_models[i])

                self.active_server_models[i].load_state_dict(self.global_server_weights)

                self.client_optimizers[i] = self._get_optimizer(
                    self.active_client_models[i].parameters()
                )
                self.server_optimizers[i] = self._get_optimizer(
                    self.active_server_models[i].parameters()
                )

            for i, cid in enumerate(sampled_ids):
                client_model = self.active_client_models[i]
                server_model = self.active_server_models[i]
                client_opt = self.client_optimizers[i]
                server_opt = self.server_optimizers[i]

                sample_count = self.data_manager.get_client_sample_count(cid)
                active_client_samples.append(sample_count)

                steps_per_epoch = self.data_manager.get_client_steps_per_epoch(cid)
                total_local_steps = steps_per_epoch * local_epochs

                loader = self.data_manager.get_client_loader(cid)
                iterator = iter(loader)

                client_loss_sum = 0

                client_model.train()
                server_model.train()

                for step in range(total_local_steps):
                    inputs, labels = next(iterator)

                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    self.communicator.log_proceeded_samples(inputs)
                    if inputs.size(0) == 1:
                        continue

                    client_opt.zero_grad()
                    server_opt.zero_grad()

                    current_step_loss_sum = 0.0

                    for _ in range(self.P):
                        c_seed = np.random.randint(0, 1000000)
                        s_seed = np.random.randint(0, 1000000)

                        self._generate_perturbation_and_apply(
                            client_model, c_seed, self.mu
                        )
                        self._generate_perturbation_and_apply(
                            server_model, s_seed, self.mu
                        )

                        with torch.no_grad():
                            smashed_p = client_model(inputs)
                            self.communicator.log_activation(smashed_p)

                            out_p = server_model(smashed_p)
                            loss_p = self.criterion(out_p, labels)

                        self._generate_perturbation_and_apply(
                            client_model, c_seed, -2 * self.mu
                        )
                        self._generate_perturbation_and_apply(
                            server_model, s_seed, -2 * self.mu
                        )

                        with torch.no_grad():
                            smashed_n = client_model(inputs)
                            self.communicator.log_activation(smashed_n)

                            out_n = server_model(smashed_n)
                            loss_n = self.criterion(out_n, labels)

                        self._generate_perturbation_and_apply(
                            client_model, c_seed, self.mu
                        )
                        self._generate_perturbation_and_apply(
                            server_model, s_seed, self.mu
                        )

                        loss_diff = loss_p.item() - loss_n.item()

                        scalar = loss_diff / (2 * self.mu * self.P)

                        self._generate_perturbation_and_accumulate_grad(
                            client_model, c_seed, scalar
                        )
                        self._generate_perturbation_and_accumulate_grad(
                            server_model, s_seed, scalar
                        )

                        self.communicator.log_scalars_downlink(1)

                        current_step_loss_sum += (loss_p.item() + loss_n.item()) / 2.0

                    client_opt.step()
                    server_opt.step()

                    client_loss_sum += current_step_loss_sum / self.P

                avg_loss += client_loss_sum
                total_steps_in_round += total_local_steps

                self.communicator.log_model_upload(client_model)

            self.global_client_weights = self._aggregate_weights(
                self.active_client_models,
                active_client_samples,
                self.global_client_weights,
            )

            self.global_server_weights = self._aggregate_weights(
                self.active_server_models,
                active_client_samples,
                self.global_server_weights,
            )

            stats = self.communicator.get_stats()
            mean_loss = (
                avg_loss / total_steps_in_round if total_steps_in_round > 0 else 0
            )

            print(f"Round {round_idx+1}/{total_rounds} | Loss: {mean_loss:.4f} |")

            if (
                (round_idx + 1) % self.evaluation_interval == 0
                or (round_idx + 1) == total_rounds
                or (stats["proceeded_samples"] >= self.cfg.runner.total_samples)
            ):
                acc = self.evaluate()
            else:
                acc = None
            if acc > self.best_acc:
                self.best_acc = acc

            if self.cfg.logging.use_wandb:
                row_dict = {
                    "Round": round_idx + 1,
                    "Train/Loss": mean_loss,
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
            log_name += f"-p{self.cfg.algo.zo_p}-mu{self.cfg.algo.zo_mu}"
            log_path = f"{self.cfg.logging.log_dir}/{log_name}_log.csv"
            df.to_csv(log_path, index=False)

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
