import torch
import torch.nn as nn
import numpy as np
import random
from tqdm import tqdm
import math
from collections import OrderedDict

from ..utils import prepare_settings
from ..utils.data import get_dataloaders

# ==========================================
# Mock Server
# ==========================================


class MockServer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states, attention_mask=None):
        return None


# ==========================================
# Inference Memory Profiling Runner
# ==========================================


class InferenceMemoryProfilingRunner:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)

        print("Initializing Inference-Only Memory Profiling Runner...")

        # 1. Data Setup
        self.num_clients = args.sfl_setting.num_clients
        self.train_loaders, _ = get_dataloaders(
            args.data_setting,
            num_train_split=self.num_clients,
            seed=args.seed,
            hf_model_name=args.model_setting.get_hf_model_name(),
        )
        self.train_iters = [self._inf_loader(loader) for loader in self.train_loaders]

        # 2. Model Setup
        # Server: Mock
        client_model, _ = prepare_settings.get_model(
            args.data_setting.dataset, args.model_setting, args.seed
        )
        self.client_model = client_model.to(self.device)
        self.server_model = MockServer().to(self.device)

        # 3. Optimizer Setup
        self.client_optimizer = None

        print(f"Client Model Loaded. Optimizer skipped (Inference Mode).")

    def _inf_loader(self, dl):
        while True:
            for v in dl:
                yield v

    def run(self):
        total_steps = self.args.total_steps
        sampled_client_num = self.args.sampled_client_num

        if sampled_client_num > self.num_clients:
            sampled_client_num = self.num_clients

        print(f"Start Inference Profiling: {total_steps} Steps...")

        # GPU Warmup
        torch.cuda.synchronize()

        self.client_model.eval()

        for step in tqdm(range(total_steps)):

            # 1. Sample Clients
            sampled_ids = np.random.choice(
                self.num_clients, sampled_client_num, replace=False
            )

            # 2. Sequential Simulation
            for cid in sampled_ids:
                iterator = self.train_iters[cid]

                inputs, labels = next(iterator)
                inputs = inputs.to(self.device)
                labels = labels.to(
                    self.device
                )

                # =========================================================
                # Inference Forward Pass
                # =========================================================
                with torch.no_grad():

                    # 1. Client Forward
                    client_hidden = self.client_model(
                        input_ids=inputs.input_ids, attention_mask=inputs.attention_mask
                    )

                    if hasattr(client_hidden, "last_hidden_state"):
                        client_hidden = client_hidden.last_hidden_state
                    elif isinstance(client_hidden, tuple):
                        client_hidden = client_hidden[0]

                    # 2. Mock Server Forward
                    self.server_model(client_hidden, inputs.attention_mask)

                # 3. Cleanup
                del client_hidden

            # Step cleanup
            torch.cuda.empty_cache()

        print("Inference Profiling Completed.")
