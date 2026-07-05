import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import random
import wandb

from src.models.registry import get_model
from src.models.split_wrapper import split_model
from src.core.communicator import Communicator


class HO_SFLRunner:
    def __init__(self, config, data_manager):
        self.cfg = config
        self.data_manager = data_manager
        self.device = torch.device(config.system.device)
        self.communicator = Communicator()

        print(f"Initializing HO-SFL Runner")

        self.evaluation_interval = config.runner.get("evaluation_interval", 10)
        # --- Models ---
        full_model = get_model(config)
        client_part, server_part = split_model(full_model, config.algo.split_point)

        self.server_model = server_part.to(self.device)
        self.server_optimizer = self._get_optimizer(self.server_model.parameters())

        self.client_model = client_part.to(self.device)

        # Client Optimizer
        self.client_optimizer = self._get_optimizer(self.client_model.parameters())

        self.criterion = nn.CrossEntropyLoss()

        # ZO Params
        self.P = config.algo.zo_p
        self.mu = config.algo.zo_mu
        self.best_acc = 0.0
        self.client_timestamps = [0] * self.cfg.runner.num_clients

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
                # grad += scalar * u
                param.grad.add_(u, alpha=scalar_weight)
        torch.set_rng_state(rng_state)

    def run(self):
        total_rounds = self.cfg.runner.communication_rounds
        print(f"Start Training for {total_rounds} rounds...")

        wandb_table = None

        self.communicator.reset()

        for round_idx in range(total_rounds):
            sampled_ids = self.sample_clients()

            total_missed_rounds_batch = 0
            for cid in sampled_ids:
                # Calculate how many rounds this client missed since last update
                last_update_round = self.client_timestamps[cid]
                missed = round_idx - last_update_round

                if missed > 0:
                    total_missed_rounds_batch += missed

            if total_missed_rounds_batch > 0:
                # For every missed round, client downloads P scalars and P seeds
                # Log Downlink traffic for synchronization
                sync_items_count = total_missed_rounds_batch * self.P
                self.communicator.log_scalars_downlink(sync_items_count)
                self.communicator.log_seeds(sync_items_count)

            clients_data = []

            self.client_model.eval()
            self.server_model.train()

            self.server_optimizer.zero_grad()
            self.client_optimizer.zero_grad()
            avg_loss = 0.0
            for cid in sampled_ids:
                loader = self.data_manager.get_client_loader(cid)
                inputs, labels = next(loader)
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                if inputs.size(0) == 1:
                    # Skip the last batch if its size is 1
                    continue
                self.communicator.log_proceeded_samples(inputs)

                # --- 1. Client Forward ---
                with torch.no_grad():
                    a_m = self.client_model(inputs)

                self.communicator.log_activation(a_m)

                # --- 2. Server Forward & Backward ---
                server_input = a_m.clone().detach().requires_grad_(True)

                # Note: No zero_grad here, accumulating gradients
                server_output = self.server_model(server_input)
                loss = self.criterion(server_output, labels)
                avg_loss += loss.item()

                # Server Backward
                scale = torch.tensor(1 / len(sampled_ids), device=loss.device)
                loss.backward(scale)

                # Get g_a (Activation Gradient) for ZO
                g_a_m = server_input.grad.clone().detach()
                self.communicator.log_activation_gradient(g_a_m)

                clients_data.append({"inputs": inputs, "a_m": a_m, "g_a_m": g_a_m})

            for param in self.server_model.parameters():
                if param.grad is not None:
                    param.grad.div_(len(sampled_ids))

            # Server Update
            self.server_optimizer.step()

            seeds = [np.random.randint(0, 1000000) for _ in range(self.P)]
            self.communicator.log_seeds(len(seeds) * len(sampled_ids))

            all_clients_v_list = []

            for c_data in clients_data:
                inputs = c_data["inputs"]
                a_m = c_data["a_m"]
                g_a_m = c_data["g_a_m"]

                client_v_tensor = torch.zeros(self.P, device=self.device)

                for p_idx, seed in enumerate(seeds):
                    self._generate_perturbation_and_apply(
                        self.client_model, seed, self.mu
                    )
                    with torch.no_grad():
                        a_tilde = self.client_model(inputs)

                    diff = a_tilde - a_m
                    v_scalar = torch.sum(diff * g_a_m)
                    client_v_tensor[p_idx] = v_scalar

                    self._generate_perturbation_and_apply(
                        self.client_model, seed, -self.mu
                    )

                self.communicator.log_scalars_uplink(self.P)
                all_clients_v_list.append(client_v_tensor)

            all_clients_v_stack = torch.stack(all_clients_v_list)
            bar_v = torch.mean(all_clients_v_stack, dim=0)
            self.communicator.log_scalars_downlink(self.P * len(sampled_ids))

            scale = 1.0 / (self.P * self.mu)

            for p_idx, seed in enumerate(seeds):
                scalar_w = bar_v[p_idx].item() * scale
                self._generate_perturbation_and_accumulate_grad(
                    self.client_model, seed, scalar_w
                )

            self.client_optimizer.step()

            for cid in sampled_ids:
                self.client_timestamps[cid] = round_idx + 1

            stats = self.communicator.get_stats()
            mean_loss = avg_loss / len(sampled_ids)

            print(f"Round {round_idx+1}/{total_rounds} | " f"Loss: {mean_loss:.4f} | ")
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
            log_name += f"-p{self.cfg.algo.zo_p}-mu{self.cfg.algo.zo_mu}"
            log_path = f"{self.cfg.logging.log_dir}/{log_name}_log.csv"
            df.to_csv(log_path, index=False)

    def evaluate(self):
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
