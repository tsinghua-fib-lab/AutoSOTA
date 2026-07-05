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


class MU_SplitFedRunner:
    def __init__(self, config, data_manager):
        self.cfg = config
        self.data_manager = data_manager
        self.device = torch.device(config.system.device)
        self.communicator = Communicator()

        print(f"Initializing MU-SplitFed Runner")

        self.evaluation_interval = config.runner.get("evaluation_interval", 10)

        self.tau = config.algo.get("tau", 2)
        self.mu = config.algo.get("zo_mu", 1e-3)

        full_model = get_model(config)
        client_part, server_part = split_model(full_model, config.algo.split_point)

        self.global_client_weights = deepcopy(client_part.state_dict())
        self.global_server_weights = deepcopy(server_part.state_dict())

        self.client_model = client_part.to(self.device)
        self.server_model = server_part.to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.best_acc = 0.0

    def _get_optimizer(self, params, is_server=False):
        if is_server:
            lr = self.cfg.algo.get("lr_s", 0.01)
        else:
            lr = self.cfg.algo.get("lr_c", 0.005)
        return optim.SGD(params, lr=lr)

    def sample_clients(self):
        total_clients = self.cfg.runner.num_clients
        return np.random.choice(
            total_clients, self.cfg.runner.sampled_clients, replace=False
        )

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

    def _aggregate_weights(self, global_state, state_dicts):
        eta_g = self.cfg.algo.get("lr_g", 0.3)
        num_clients = len(state_dicts)
        new_state = {}

        for key in global_state.keys():
            if not torch.is_floating_point(global_state[key]):
                new_state[key] = global_state[key]
                continue

            delta_avg = torch.zeros_like(
                global_state[key], dtype=torch.float32, device=self.device
            )

            for state in state_dicts:
                delta_avg += (
                    state[key].to(self.device) - global_state[key].to(self.device)
                ) / num_clients

            new_param = global_state[key].to(self.device) + eta_g * delta_avg
            new_state[key] = new_param.to(global_state[key].dtype)

        return new_state

    def run(self):
        total_rounds = self.cfg.runner.communication_rounds
        print(
            f"Start Training MU-SplitFed for {total_rounds} rounds with tau={self.tau}..."
        )

        wandb_table = None
        self.communicator.reset()

        for round_idx in range(total_rounds):
            sampled_ids = self.sample_clients()
            avg_loss = 0

            client_weights_buffer = []
            server_weights_buffer = []

            for cid in sampled_ids:
                self.client_model.load_state_dict(self.global_client_weights)
                self.server_model.load_state_dict(self.global_server_weights)
                self.communicator.log_model_download(self.client_model)

                client_optimizer = self._get_optimizer(
                    self.client_model.parameters(), is_server=False
                )
                server_optimizer = self._get_optimizer(
                    self.server_model.parameters(), is_server=True
                )

                self.client_model.train()
                self.server_model.train()

                loader = self.data_manager.get_client_loader(cid)
                inputs, labels = next(iter(loader))
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.communicator.log_proceeded_samples(inputs)

                if inputs.size(0) == 1:
                    continue

                client_optimizer.zero_grad()
                server_optimizer.zero_grad()

                with torch.no_grad():
                    h_fixed = self.client_model(inputs)

                self.communicator.log_activation(h_fixed)

                for i in range(self.tau):
                    server_optimizer.zero_grad()
                    s_seed = np.random.randint(0, 1000000)

                    self._generate_perturbation_and_apply(
                        self.server_model, s_seed, self.mu
                    )
                    with torch.no_grad():
                        out_p = self.server_model(h_fixed)
                        loss_p = self.criterion(out_p, labels)

                    self._generate_perturbation_and_apply(
                        self.server_model, s_seed, -2 * self.mu
                    )
                    with torch.no_grad():
                        out_n = self.server_model(h_fixed)
                        loss_n = self.criterion(out_n, labels)

                    self._generate_perturbation_and_apply(
                        self.server_model, s_seed, self.mu
                    )

                    loss_diff_s = loss_p.item() - loss_n.item()
                    scalar_s = loss_diff_s / (2 * self.mu)

                    self._generate_perturbation_and_accumulate_grad(
                        self.server_model, s_seed, scalar_s
                    )
                    server_optimizer.step()

                c_seed = np.random.randint(0, 1000000)

                self._generate_perturbation_and_apply(
                    self.client_model, c_seed, self.mu
                )
                with torch.no_grad():
                    h_pos = self.client_model(inputs)

                self.communicator.log_activation(h_pos)

                self._generate_perturbation_and_apply(
                    self.client_model, c_seed, -2 * self.mu
                )
                with torch.no_grad():
                    h_neg = self.client_model(inputs)

                self.communicator.log_activation(h_neg)

                self._generate_perturbation_and_apply(
                    self.client_model, c_seed, self.mu
                )

                with torch.no_grad():
                    out_c_p = self.server_model(h_pos)
                    loss_c_p = self.criterion(out_c_p, labels)

                    out_c_n = self.server_model(h_neg)
                    loss_c_n = self.criterion(out_c_n, labels)

                loss_diff_c = loss_c_p.item() - loss_c_n.item()
                scalar_c = loss_diff_c / (2 * self.mu)

                self.communicator.log_scalars_downlink(1)

                self._generate_perturbation_and_accumulate_grad(
                    self.client_model, c_seed, scalar_c
                )
                client_optimizer.step()

                avg_loss += (loss_c_p.item() + loss_c_n.item()) / 2.0

                client_weights_buffer.append(deepcopy(self.client_model.state_dict()))
                server_weights_buffer.append(deepcopy(self.server_model.state_dict()))

                self.communicator.log_model_upload(self.client_model)

            self.global_client_weights = self._aggregate_weights(
                self.global_client_weights, client_weights_buffer
            )
            self.global_server_weights = self._aggregate_weights(
                self.global_server_weights, server_weights_buffer
            )

            stats = self.communicator.get_stats()
            mean_loss = avg_loss / len(sampled_ids)

            print(f"Round {round_idx+1}/{total_rounds} | Loss: {mean_loss:.4f} |")

            acc = None
            if (
                (round_idx + 1) % self.evaluation_interval == 0
                or (round_idx + 1) == total_rounds
                or (stats["proceeded_samples"] >= self.cfg.runner.total_samples)
            ):
                acc = self.evaluate()
            else:
                acc = None
            if acc is not None and acc > self.best_acc:
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
            log_name += f"-tau{self.tau}-mu{self.mu}-lr_c{self.cfg.algo.get('lr_c',0.005)}-lr_s{self.cfg.algo.get('lr_s',0.01)}-lr_g{self.cfg.algo.get('lr_g',0.3)}"
            log_path = f"{self.cfg.logging.log_dir}/{log_name}_log.csv"
            df.to_csv(log_path, index=False)

    def evaluate(self):
        self.client_model.load_state_dict(self.global_client_weights)
        self.server_model.load_state_dict(self.global_server_weights)

        self.client_model.eval()
        self.server_model.eval()

        test_loader = self.data_manager.get_global_test_loader()
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                feat = self.client_model(inputs)
                out = self.server_model(feat)
                _, predicted = torch.max(out.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        acc = 100 * correct / total
        print(f"Validation Accuracy: {acc:.2f}%")
        return acc
