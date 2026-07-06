import torch
import torch.nn as nn
import numpy as np
import random
from tqdm import tqdm
import math
from collections import OrderedDict

from ..utils import prepare_settings
from ..utils.data import get_dataloaders


class MockServer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states, attention_mask=None):
        return None


class SFLMemoryProfilingRunner:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)

        print("Initializing SFL Memory Profiling Runner (Client Focus)...")

        self.num_clients = args.sfl_setting.num_clients
        self.train_loaders, _ = get_dataloaders(
            args.data_setting,
            num_train_split=self.num_clients,
            seed=args.seed,
            hf_model_name=args.model_setting.get_hf_model_name(),
        )
        self.train_iters = [self._inf_loader(loader) for loader in self.train_loaders]

        self.client_steps_per_epoch = self._calculate_steps_per_epoch()

        # --- 2. Model Setup ---
        client_model, _ = prepare_settings.get_model(
            args.data_setting.dataset, args.model_setting, args.seed
        )
        self.client_model = client_model.to(self.device)
        self.server_model = MockServer().to(self.device)

        # --- 3. Weights Management ---
        self.global_client_weights = self._get_trainable_state(self.client_model)

        # --- 4. Optimizer ---

        print("Client Model & Data ready. Server is mocked.")

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
                raise ValueError("Dataset len() error.")
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

        print(f"Start SFL Profiling: {total_rounds} Rounds...")

        # GPU Warmup
        torch.cuda.synchronize()

        for step in tqdm(range(total_rounds)):

            # 1. Sample Clients
            sampled_ids = np.random.choice(
                self.num_clients, sampled_client_num, replace=False
            )

            active_client_weights = []
            active_counts = []

            # 2. Sequential Simulation
            for cid in sampled_ids:
                # A. Load Weights
                self._load_trainable_state(
                    self.client_model, self.global_client_weights
                )

                client_opt = self._get_optimizer(self.client_model)

                self.client_model.train()

                iterator = self.train_iters[cid]
                current_local_steps = self.client_steps_per_epoch[cid] * local_epochs
                total_samples = 0

                for _ in range(current_local_steps):
                    inputs, labels = next(iterator)
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    total_samples += labels.size(0)
                    client_opt.zero_grad()

                    # --- SFL Client Forward ---
                    client_hidden, attention_mask = self.client_model(
                        input_ids=inputs.input_ids, attention_mask=inputs.attention_mask
                    )


                    # --- Mock Server Gradient ---
                    fake_smashed_grad = torch.randn_like(
                        client_hidden, device=self.device
                    )

                    # --- SFL Client Backward ---
                    client_hidden.backward(fake_smashed_grad)

                    # --- Client Update ---
                    client_opt.step()

                    del fake_smashed_grad

                # D. Store updated weights
                active_client_weights.append(
                    self._get_trainable_state(self.client_model)
                )
                active_counts.append(total_samples)

                del client_opt
                break
