import torch
import torch.nn as nn
import numpy as np
import random
from tqdm import tqdm

from ..utils import prepare_settings
from ..utils.data import get_dataloaders
from ..gradient_estimators.hybrid_gradient_estimator import (
    HybridGradientEstimator,
)


class MockServer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states, attention_mask=None):
        return None


class MemoryProfilingRunner:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)

        print("Initializing HO-SFL Memory Profiling Runner (Client Focus)...")

        self.num_clients = args.ho_sfl_setting.num_clients
        self.train_loaders, _ = get_dataloaders(
            args.data_setting,
            num_train_split=self.num_clients,
            seed=args.seed,
            hf_model_name=args.model_setting.get_hf_model_name(),
        )
        self.train_iters = [self._inf_loader(loader) for loader in self.train_loaders]
        print("Data loaders prepared.")

        client_model, _ = prepare_settings.get_model(
            args.data_setting.dataset, args.model_setting, args.seed
        )
        self.client_model = client_model.to(self.device)
        self.server_model = MockServer().to(self.device)
        print("Client Model initialized (Server is Mocked).")

        self.client_optimizer = prepare_settings.get_optimizer(
            self.client_model, args.data_setting.dataset, args.optimizer_setting
        )

        self.estimator = HybridGradientEstimator(
            self.client_model,
            device=self.device,
            dtype=args.model_setting.get_torch_dtype(),
        )

        self.mu = args.hybrid_gradient_estimator_setting.mu
        self.num_pert = args.hybrid_gradient_estimator_setting.num_pert

    def _inf_loader(self, dl):
        while True:
            for v in dl:
                yield v

    def compute_v_with_mock_server(
        self,
        inputs,
        seeds,
        mu,
    ):
        P = len(seeds)

        # --- 1. Client Forward (Anchor) ---
        self.client_model.eval()
        with torch.no_grad():
            outputs = self.client_model(
                input_ids=inputs.input_ids, attention_mask=inputs.attention_mask
            )
            if hasattr(outputs, "last_hidden_state"):
                h_anchor = outputs.last_hidden_state
            elif isinstance(outputs, tuple):
                h_anchor = outputs[0]
            else:
                h_anchor = outputs

        # --- 2. Mock Server Interaction ---
        g_act = torch.randn_like(h_anchor, device=self.device)

        # --- 3. ZO Projection (Compute v) ---
        v_list = []

        for i, seed in enumerate(seeds):
            rng = self.estimator.get_rng(seed, i)

            # 3.1 Apply Perturbation (+mu)
            self.estimator._generate_noise_and_apply(rng, mu)

            # 3.2 Forward Perturbed
            with torch.no_grad():
                out_p = self.client_model(
                    input_ids=inputs.input_ids, attention_mask=inputs.attention_mask
                )
                if hasattr(out_p, "last_hidden_state"):
                    h_p = out_p.last_hidden_state
                elif isinstance(out_p, tuple):
                    h_p = out_p[0]
                else:
                    h_p = out_p

            rng_restore = self.estimator.get_rng(seed, i)
            self.estimator._generate_noise_and_apply(rng_restore, -mu)

            delta = h_p - h_anchor
            v_scalar = torch.sum(delta * g_act)

            v_list.append(v_scalar)

            del h_p, delta

        return torch.stack(v_list)

    def run(self):
        total_steps = self.args.ho_sfl_setting.total_steps
        sampled_client_num = self.args.ho_sfl_setting.sampled_client_num

        if sampled_client_num > self.num_clients:
            sampled_client_num = self.num_clients

        print(f"Start Profiling Run: {total_steps} Steps...")

        for step in tqdm(range(total_steps)):

            sampled_ids = np.random.choice(
                self.num_clients, sampled_client_num, replace=False
            )

            self.client_optimizer.zero_grad()

            seeds = [random.randint(0, 10000000) for _ in range(self.num_pert)]

            v_list = []

            for cid in sampled_ids:
                inputs, labels = next(self.train_iters[cid])
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                v = self.compute_v_with_mock_server(
                    inputs=inputs,
                    seeds=seeds,
                    mu=self.mu,
                )

                v_list.append(v)

            v_stack = torch.stack(v_list)
            bar_v = torch.mean(v_stack, dim=0)

            self.estimator.update_from_aggregated_v(bar_v, seeds, self.mu)

            # Optimizer Step
            self.client_optimizer.step()

            del v_list, v_stack, bar_v
