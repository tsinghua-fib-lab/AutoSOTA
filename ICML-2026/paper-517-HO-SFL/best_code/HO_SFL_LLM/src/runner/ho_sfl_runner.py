import torch
import torch.nn as nn
import numpy as np
import wandb
import pandas as pd
import os
import random
from tqdm import tqdm

from ..utils.metrics import Metric
from ..core.communicator import Communicator
from ..utils import prepare_settings
from ..utils.data import get_dataloaders
from ..gradient_estimators.hybrid_gradient_estimator import (
    HybridGradientEstimator,
)


class HO_SFLRunner:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)
        self.communicator = Communicator()

        print("Initializing HO-SFL Runner")

        self.num_clients = args.ho_sfl_setting.num_clients
        self.train_loaders, self.test_loader = get_dataloaders(
            args.data_setting,
            num_train_split=self.num_clients,
            seed=args.seed,
            hf_model_name=args.model_setting.get_hf_model_name(),
        )
        self.train_iters = [self._inf_loader(loader) for loader in self.train_loaders]
        print("Data loaders prepared.")

        client_model, server_model = prepare_settings.get_model(
            args.data_setting.dataset, args.model_setting, args.seed
        )
        self.client_model = client_model.to(self.device)
        self.server_model = server_model.to(self.device)
        print("Models initialized.")
        self.client_optimizer = prepare_settings.get_optimizer(
            self.client_model, args.data_setting.dataset, args.optimizer_setting
        )

        self.server_optimizer = prepare_settings.get_optimizer(
            self.server_model, args.data_setting.dataset, args.optimizer_setting
        )

        self.estimator = HybridGradientEstimator(
            self.client_model,
            device=self.device,
            dtype=args.model_setting.get_torch_dtype(),
        )
        print("Optimizer and Estimator set up.")
        self.metric_packs = prepare_settings.get_metrics(
            args.data_setting.dataset, args.model_setting
        )
        self.mu = args.hybrid_gradient_estimator_setting.mu
        self.num_pert = args.hybrid_gradient_estimator_setting.num_pert

        self.best_acc = 0.0
        self.client_timestamps = [0] * self.num_clients

    def _inf_loader(self, dl):
        while True:
            for v in dl:
                yield v

    def run(self):
        total_steps = self.args.ho_sfl_setting.total_steps
        sampled_client_num = self.args.ho_sfl_setting.sampled_client_num
        if sampled_client_num > self.num_clients:
            raise ValueError(
                f"Sampled client num {sampled_client_num} exceeds total clients {self.num_clients}."
            )
        print(
            f"Start Training: {total_steps} Rounds, Clients per round: {sampled_client_num}"
        )

        self.communicator.reset()
        wandb_table = None

        start_acc = self.evaluate(step=0)

        for step in tqdm(range(total_steps)):
            round_idx = step + 1

            sampled_ids = np.random.choice(
                self.num_clients, sampled_client_num, replace=False
            )

            total_missed_rounds_batch = 0
            for cid in sampled_ids:
                last_update_round = self.client_timestamps[cid]

                missed = (round_idx - 1) - last_update_round

                if missed > 0:
                    total_missed_rounds_batch += missed

            if total_missed_rounds_batch > 0:
                sync_items_count = total_missed_rounds_batch * self.num_pert
                self.communicator.log_scalars_downlink(sync_items_count, 1)
                self.communicator.log_seeds(sync_items_count, 1)

            self.server_optimizer.zero_grad()
            self.client_optimizer.zero_grad()

            seeds = [np.random.randint(0, 10000000) for _ in range(self.num_pert)]
            self.communicator.log_seeds(len(seeds), len(sampled_ids))

            v_list = []
            avg_loss = 0.0

            for cid in sampled_ids:
                inputs, labels = next(self.train_iters[cid])
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                self.communicator.log_proceeded_samples(labels.size(0))

                v, loss_val = self.estimator.compute_v_with_server(
                    inputs=inputs,
                    labels=labels,
                    server_model=self.server_model,
                    loss_fn=self.metric_packs.train_loss,
                    seeds=seeds,
                    mu=self.mu,
                    communicator=self.communicator,
                    scale_loss=1.0 / len(sampled_ids),
                )

                v_list.append(v)
                avg_loss += loss_val

            self.server_optimizer.step()

            v_stack = torch.stack(v_list)
            bar_v = torch.mean(v_stack, dim=0)

            self.communicator.log_scalars_downlink(self.num_pert, len(sampled_ids))

            self.estimator.update_from_aggregated_v(bar_v, seeds, self.mu)
            self.client_optimizer.step()

            for cid in sampled_ids:
                self.client_timestamps[cid] = round_idx

            stats = self.communicator.get_stats()
            mean_loss = avg_loss / len(sampled_ids)

            print(f"Round {round_idx}/{total_steps} | Loss: {mean_loss:.4f} |")

            if (
                round_idx % self.args.ho_sfl_setting.evaluation_interval == 0
                or round_idx == total_steps
            ) or (stats["proceeded_samples"] >= self.args.total_samples):
                acc = self.evaluate(round_idx)
            else:
                acc = None

            if acc is not None and acc > self.best_acc:
                self.best_acc = acc
            if stats["proceeded_samples"] >= self.args.total_samples:
                print(
                    f"Reached total samples {self.args.total_samples}. Stopping training."
                )
                break

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

        if self.args.log_to_wandb and wandb_table is not None:
            random.seed(None)
            print("Uploading detailed run history table to WandB...")
            wandb.log({"Detailed_Run_History": wandb_table})

            log_data = wandb_table.data
            log_columns = wandb_table.columns
            df = pd.DataFrame(data=log_data, columns=log_columns)

            log_dir = self.args.log_dir
            os.makedirs(log_dir, exist_ok=True)
            log_name = (
                f"HO_SFL-{self.args.data_setting.dataset.value}-"
                f"p{self.num_pert}-mu{self.mu}-"
                f"seed{self.args.seed}"
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
                outputs = self.server_model(
                    hidden_states=sended_hidden_state, attention_mask=attention_mask
                )

                batch_acc = self.metric_packs.test_acc(outputs, batch_output_tensor)
                acc.update(batch_acc, samples=num_samples)
                del batch_input_dict, batch_output_tensor, outputs, batch_acc
                torch.cuda.empty_cache()
        print(f"[Eval] Round {step} | Validation Accuracy: {acc.avg:.4f}")
        return acc.avg
