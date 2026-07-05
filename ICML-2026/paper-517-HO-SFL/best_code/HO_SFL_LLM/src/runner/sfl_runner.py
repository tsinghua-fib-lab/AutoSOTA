import torch
import numpy as np
import wandb
import pandas as pd
import os
import random
from tqdm import tqdm
import math
from collections import OrderedDict

from ..utils.metrics import Metric
from ..core.communicator import Communicator
from ..utils import prepare_settings
from ..utils.data import get_dataloaders


class SFLRunner:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)
        self.communicator = Communicator()

        print("Initializing Standard SFL Runner...")

        # --- Data Setup ---
        self.num_clients = args.sfl_setting.num_clients
        self.train_loaders, self.test_loader = get_dataloaders(
            args.data_setting,
            num_train_split=self.num_clients,
            seed=args.seed,
            hf_model_name=args.model_setting.get_hf_model_name(),
        )

        self.train_iters = [self._inf_loader(loader) for loader in self.train_loaders]

        self.client_steps_per_epoch = self._calculate_steps_per_epoch()

        client_model, server_model = prepare_settings.get_model(
            args.data_setting.dataset, args.model_setting, args.seed
        )
        self.client_model = client_model.to(self.device)
        self.server_model = server_model.to(self.device)

        # --- Weights Management (LoRA Optimized) ---
        self.global_client_weights = self._get_trainable_state(self.client_model)
        self.global_server_weights = self._get_trainable_state(self.server_model)

        # --- Metrics ---
        self.metric_packs = prepare_settings.get_metrics(
            args.data_setting.dataset, args.model_setting
        )

        self.best_acc = 0.0

    def _inf_loader(self, dl):
        while True:
            for v in dl:
                yield v

    def _calculate_steps_per_epoch(self):

        steps_list = []
        batch_size = self.args.data_setting.train_batch_size
        for loader in self.train_loaders:
            try:
                total_samples = len(loader.dataset)
            except (AttributeError, TypeError):
                raise ValueError(
                    "Dataset does not support len(). Please provide a fixed number of steps."
                )

            steps = math.ceil(total_samples / batch_size)
            steps_list.append(steps)
        return steps_list

    def _get_trainable_state(self, model):

        state = OrderedDict()
        for name, param in model.named_parameters():
            if param.requires_grad:
                state[name] = param.data.clone()
        return state

    def _load_trainable_state(self, model, state):

        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in state:
                    param.data.copy_(state[name])

    def _get_optimizer(self, model):

        return prepare_settings.get_optimizer(
            model, self.args.data_setting.dataset, self.args.optimizer_setting
        )

    def _aggregate_weights(self, weights_list, sample_counts):

        total_samples = sum(sample_counts)
        aggregated_weights = OrderedDict()

        first_weights = weights_list[0]
        for key in first_weights.keys():
            aggregated_weights[key] = torch.zeros_like(first_weights[key])

        for weights, count in zip(weights_list, sample_counts):
            factor = count / total_samples
            for key in weights.keys():
                aggregated_weights[key] += weights[key] * factor

        return aggregated_weights

    def run(self):
        total_rounds = self.args.sfl_setting.total_rounds
        local_epochs = self.args.sfl_setting.local_epochs
        sampled_client_num = self.args.sampled_client_num

        if sampled_client_num > self.num_clients:
            sampled_client_num = self.num_clients

        print(
            f"Start SFL Training: {total_rounds} Rounds, Clients/Round: {sampled_client_num}, Local epochs: {local_epochs}"
        )

        self.communicator.reset()
        wandb_table = None

        for step in tqdm(range(total_rounds)):
            round_idx = step + 1

            # 1. Sample Clients
            sampled_ids = np.random.choice(
                self.num_clients, sampled_client_num, replace=False
            )

            active_client_weights = []
            active_server_weights = []
            active_counts = []

            round_loss = 0.0

            # 2. Sequential Simulation of Parallel Clients
            for cid in sampled_ids:
                # A. Downlink: Load Global Weights to Client & Server Model
                self._load_trainable_state(
                    self.client_model, self.global_client_weights
                )
                self._load_trainable_state(
                    self.server_model, self.global_server_weights
                )

                self.communicator.log_model_download(self.global_client_weights)
                client_opt = self._get_optimizer(self.client_model)
                server_opt = self._get_optimizer(self.server_model)

                self.client_model.train()
                self.server_model.train()

                client_loss_sum = 0.0

                iterator = self.train_iters[cid]

                # current_local_steps = local_steps
                current_local_steps = self.client_steps_per_epoch[cid] * local_epochs
                total_samples = 0
                for _ in range(current_local_steps):
                    inputs, labels = next(iterator)
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    self.communicator.log_proceeded_samples(labels.size(0))
                    total_samples += labels.size(0)
                    client_opt.zero_grad()
                    server_opt.zero_grad()

                    # --- SFL Forward ---
                    # 1. Client Forward -> Activations
                    client_hidden, attention_mask = self.client_model(
                        input_ids=inputs.input_ids, attention_mask=inputs.attention_mask
                    )

                    # 2. Communication: Client -> Server (Upload Activation)
                    # Detach from graph to simulate network cut, require grad for backward
                    smashed_data = (
                        client_hidden.detach()
                        .clone()
                        .to(self.device)
                        .requires_grad_(True)
                    )
                    self.communicator.log_activation(smashed_data)

                    # 3. Server Forward
                    server_outputs = self.server_model(
                        hidden_states=smashed_data,
                        attention_mask=attention_mask.to(self.device),
                    )

                    # 4. Loss Computation
                    loss = self.metric_packs.train_loss(server_outputs, labels)
                    client_loss_sum += loss.item()

                    # --- SFL Backward ---
                    # 5. Server Backward
                    loss.backward()

                    # 6. Communication: Server -> Client (Download Gradient)
                    smashed_grad = smashed_data.grad.clone()
                    self.communicator.log_activation_gradient(smashed_grad)

                    # 7. Client Backward
                    client_hidden.backward(smashed_grad)

                    # 8. Updates
                    client_opt.step()
                    server_opt.step()

                # D. Store updated weights for aggregation
                active_client_weights.append(
                    self._get_trainable_state(self.client_model)
                )
                active_server_weights.append(
                    self._get_trainable_state(self.server_model)
                )

                active_counts.append(total_samples)

                round_loss += client_loss_sum / current_local_steps

                # E. Log Model Upload (Client -> Aggregator)
                self.communicator.log_model_upload(self.global_client_weights)

            # 3. Aggregation
            self.global_client_weights = self._aggregate_weights(
                active_client_weights, active_counts
            )
            self.global_server_weights = self._aggregate_weights(
                active_server_weights, active_counts
            )

            # 4. Logging & Eval
            mean_loss = round_loss / len(sampled_ids)
            stats = self.communicator.get_stats()

            print(f"Round {round_idx}/{total_rounds} | Loss: {mean_loss:.4f} |")

            if (
                (round_idx % self.args.sfl_setting.evaluation_interval == 0)
                or (round_idx == total_rounds)
                or (stats["proceeded_samples"] >= self.args.total_samples)
            ):
                # Load global weights for evaluation
                self._load_trainable_state(
                    self.client_model, self.global_client_weights
                )
                self._load_trainable_state(
                    self.server_model, self.global_server_weights
                )
                acc = self.evaluate(round_idx)
                if stats["proceeded_samples"] >= self.args.total_samples:
                    print(
                        f"Reached total proceeded samples {self.args.total_samples}. Stopping training."
                    )
                    break
            else:
                acc = None

            if acc is not None and acc > self.best_acc:
                self.best_acc = acc

            # WandB
            if self.args.log_to_wandb:
                row_dict = {
                    "Round": round_idx,
                    "Train/Loss": mean_loss,
                    "Val/Accuracy": acc,
                    "Val/Best_Accuracy": self.best_acc,
                }
                for k, v in stats.items():
                    if k != "proceeded_samples":
                        row_dict[f"Comm/{k}_MB"] = v / (1024**2)
                    else:
                        row_dict[f"Comm/{k}"] = v

                wandb.log(row_dict, step=round_idx)

                if wandb_table is None:
                    wandb_table = wandb.Table(columns=list(row_dict.keys()))
                row_values = [row_dict.get(k, None) for k in wandb_table.columns]
                wandb_table.add_data(*row_values)

        # End of Training: Save Log
        if self.args.log_to_wandb and wandb_table is not None:
            random.seed(None)
            print("Uploading detailed run history table to WandB...")
            wandb.log({"Detailed_Run_History": wandb_table})

            log_data = wandb_table.data
            log_columns = wandb_table.columns
            df = pd.DataFrame(data=log_data, columns=log_columns)

            log_dir = getattr(self.args, "log_dir", "./logs")
            os.makedirs(log_dir, exist_ok=True)
            log_name = (
                f"SFL-{self.args.data_setting.dataset.value}-seed{self.args.seed}"
            )
            log_path = os.path.join(log_dir, f"{log_name}_log.csv")

            print(f"Saving local log to {log_path}...")
            df.to_csv(log_path, index=False)

    def evaluate(self, step):
        self.client_model.eval()
        self.server_model.eval()
        acc = Metric("accuracy")

        with torch.no_grad():
            for batch_input_dict, batch_output_tensor in self.test_loader:
                batch_input_dict = batch_input_dict.to(self.device)
                num_samples = batch_output_tensor.size(0)
                batch_output_tensor = batch_output_tensor.to(self.device)
                hidden_state, attention_mask = self.client_model(
                    input_ids=batch_input_dict.input_ids,
                    attention_mask=batch_input_dict.attention_mask,
                )
                sended_hidden_state = hidden_state.detach().to(self.device)
                attention_mask = attention_mask.to(self.device)
                outputs = self.server_model(
                    hidden_states=sended_hidden_state, attention_mask=attention_mask
                )

                batch_acc = self.metric_packs.test_acc(outputs, batch_output_tensor)
                acc.update(batch_acc, samples=num_samples)
                del batch_input_dict, batch_output_tensor, outputs, batch_acc
                torch.cuda.empty_cache()
        print(f"[Eval] Round {step} | Validation Accuracy: {acc.avg:.4f}")
        return acc.avg
