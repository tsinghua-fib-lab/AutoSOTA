import torch
import random
import numpy as np
import wandb
import pandas as pd
import os
import math
from tqdm import tqdm
from collections import OrderedDict

from ..utils.metrics import Metric
from ..core.communicator import Communicator
from ..utils import prepare_settings
from ..utils.data import get_dataloaders

# ==========================================
# ZO Estimator Classes
# ==========================================


class BaseZOEstimator:
    def __init__(self, parameters, device=None, torch_dtype=torch.float32):
        self.parameters_list = [p for p in parameters if p.requires_grad]
        self.device = device
        self.torch_dtype = torch_dtype

    def get_rng(self, seed, perturb_index):
        return torch.Generator(device=self.device).manual_seed(
            seed * (perturb_index + 17) + perturb_index
        )

    def _sample_noise_like_params(self, rng):
        zs = []
        for p in self.parameters_list:
            z = torch.randn(
                *p.shape,
                device=p.device,
                dtype=p.dtype,
                generator=rng,
            )
            zs.append(z)
        return zs

    def _apply_perturb(self, zs, alpha):
        with torch.no_grad():
            for p, z in zip(self.parameters_list, zs):
                p.add_(z, alpha=alpha)


class ZOSPSAGradientEstimator(BaseZOEstimator):
    """
    Adapted ZO-SPSA for End-to-End Joint Optimization (Client + Server).
    """

    def compute_grad(self, forward_fn, mu, seed, num_pert):
        """
        Args:
            forward_fn: 回调函数，执行完整的前向传播并返回 Loss 标量。
            mu: 微扰强度。
            seed: 随机种子。
            num_pert: 微扰次数 (q).
        """
        dir_grads = []

        for i in range(num_pert):
            rng = self.get_rng(seed, perturb_index=i)
            zs = self._sample_noise_like_params(rng)

            with torch.no_grad():
                self._apply_perturb(zs, alpha=mu)
                loss_plus = forward_fn()

                self._apply_perturb(zs, alpha=-2 * mu)
                loss_minus = forward_fn()

                self._apply_perturb(zs, alpha=mu)

                loss_diff = loss_plus - loss_minus
                dir_grad = loss_diff / (2 * mu)

            dir_grads.append(dir_grad.detach())

            w = float(dir_grad) / float(num_pert)

            with torch.no_grad():
                for p, z in zip(self.parameters_list, zs):
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)
                    p.grad.add_(z, alpha=w)

        return torch.stack(dir_grads)


# ==========================================
# End-to-End Joint ZO SFL Runner
# ==========================================


class SFLZORunner:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)
        self.communicator = Communicator()

        print("Initializing End-to-End Joint ZO SFL Runner (Client + Server)...")

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

        # --- Model Setup ---
        client_model, server_model = prepare_settings.get_model(
            args.data_setting.dataset, args.model_setting, args.seed
        )
        self.client_model = client_model.to(self.device)
        self.server_model = server_model.to(self.device)

        # --- Weights Management ---
        self.global_client_weights = self._get_trainable_state(self.client_model)
        self.global_server_weights = self._get_trainable_state(self.server_model)

        # --- ZO Estimator Setup (Joint) ---
        all_trainable_params = list(self.client_model.parameters()) + list(
            self.server_model.parameters()
        )

        self.zo_estimator = ZOSPSAGradientEstimator(
            all_trainable_params,
            device=self.device,
            torch_dtype=args.model_setting.get_torch_dtype(),
        )
        self.mu = args.zeroth_order_setting.mu
        self.num_pert = args.zeroth_order_setting.num_pert

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
            f"Start End-to-End Joint ZO-SFL Training: {total_rounds} Rounds, "
            f"Clients/Round: {sampled_client_num}, Local epochs: {local_epochs}, "
            f"Perturbations: {self.num_pert}, Mu: {self.mu}"
        )

        self.communicator.reset()
        wandb_table = None

        global_step_counter = 0

        for step in tqdm(range(total_rounds)):
            round_idx = step + 1

            sampled_ids = np.random.choice(
                self.num_clients, sampled_client_num, replace=False
            )

            active_client_weights = []
            active_server_weights = []
            active_counts = []

            round_loss = 0.0

            for cid in sampled_ids:
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

                    def e2e_forward_pass():
                        # 1. Client Forward (NO GRAD)
                        with torch.no_grad():
                            c_h, attention_mask = self.client_model(
                                input_ids=inputs.input_ids,
                                attention_mask=inputs.attention_mask,
                            )

                        self.communicator.log_activation(c_h)

                        # 3. Server Forward (NO GRAD)
                        s_h = c_h.to(self.device)
                        with torch.no_grad():
                            s_out = self.server_model(
                                hidden_states=s_h, attention_mask=attention_mask
                            )
                            loss_val = self.metric_packs.train_loss(s_out, labels)
                        return loss_val

                    global_step_counter += 1

                    self.zo_estimator.compute_grad(
                        forward_fn=e2e_forward_pass,
                        mu=self.mu,
                        seed=global_step_counter,
                        num_pert=self.num_pert,
                    )

                    with torch.no_grad():
                        clean_loss = e2e_forward_pass()
                    client_loss_sum += clean_loss.item()

                    # Comm: Downlink (Scalar Loss Differences)
                    self.communicator.log_scalars_downlink(self.num_pert * 2, 1)

                    client_opt.step()
                    server_opt.step()

                active_client_weights.append(
                    self._get_trainable_state(self.client_model)
                )
                active_server_weights.append(
                    self._get_trainable_state(self.server_model)
                )

                active_counts.append(total_samples)
                round_loss += client_loss_sum / current_local_steps

                self.communicator.log_model_upload(self.global_client_weights)

            self.global_client_weights = self._aggregate_weights(
                active_client_weights, active_counts
            )
            self.global_server_weights = self._aggregate_weights(
                active_server_weights, active_counts
            )

            mean_loss = round_loss / len(sampled_ids)
            stats = self.communicator.get_stats()

            print(f"Round {round_idx}/{total_rounds} | Loss: {mean_loss:.4f} |")

            if (
                (round_idx % self.args.sfl_setting.evaluation_interval == 0)
                or (round_idx == total_rounds)
                or (stats["proceeded_samples"] >= self.args.total_samples)
            ):
                self._load_trainable_state(
                    self.client_model, self.global_client_weights
                )
                self._load_trainable_state(
                    self.server_model, self.global_server_weights
                )
                acc = self.evaluate(round_idx)
                if stats["proceeded_samples"] >= self.args.total_samples:
                    print("Reached total sample processing limit. Stopping training.")
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

        if self.args.log_to_wandb and wandb_table is not None:
            random.seed(None)
            print("Uploading detailed run history table to WandB...")
            wandb.log({"Detailed_Run_History": wandb_table})

            log_data = wandb_table.data
            log_columns = wandb_table.columns
            df = pd.DataFrame(data=log_data, columns=log_columns)

            log_dir = getattr(self.args, "log_dir", "./logs")
            os.makedirs(log_dir, exist_ok=True)
            log_name = f"SFL_JointZO-{self.args.data_setting.dataset.value}-seed{self.args.seed}"
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

                # Client Forward
                hidden_state, attention_mask = self.client_model(
                    input_ids=batch_input_dict.input_ids,
                    attention_mask=batch_input_dict.attention_mask,
                )

                if isinstance(hidden_state, tuple):
                    hidden_state = hidden_state[0]
                elif hasattr(hidden_state, "last_hidden_state"):
                    hidden_state = hidden_state.last_hidden_state

                sended_hidden_state = hidden_state.detach().to(self.device)
                attention_mask = attention_mask.to(self.device)

                # Server Forward
                outputs = self.server_model(
                    hidden_states=sended_hidden_state, attention_mask=attention_mask
                )

                batch_acc = self.metric_packs.test_acc(outputs, batch_output_tensor)
                acc.update(batch_acc, samples=num_samples)
                del batch_input_dict, batch_output_tensor, outputs, batch_acc
                torch.cuda.empty_cache()
        print(f"[Eval] Round {step} | Validation Accuracy: {acc.avg:.4f}")
        return acc.avg
