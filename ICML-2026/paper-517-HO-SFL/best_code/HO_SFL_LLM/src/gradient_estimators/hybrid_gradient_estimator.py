import torch


class HybridGradientEstimator:
    def __init__(self, client_model: torch.nn.Module, device=None, dtype=torch.float32):
        self.client_model = client_model
        self.dtype = dtype
        self.device = (
            device if device is not None else next(client_model.parameters()).device
        )

        self.trainable_params = [
            p for p in client_model.parameters() if p.requires_grad
        ]
        self.num_params = sum(p.numel() for p in self.trainable_params)
        self.max_grad_norm = 1.0  # gradient clipping threshold

        print(f"[HybridEstimator] Initialized. Trainable Params: {self.num_params}, Max Grad Norm: {self.max_grad_norm}")

    def get_rng(self, seed, index):
        return torch.Generator(device=self.device).manual_seed(
            seed * (index + 13) + index
        )

    def _generate_noise_and_apply(self, rng, alpha):
        """Apply Rademacher perturbations (lower variance than Gaussian for ZO)."""
        with torch.no_grad():
            for p in self.trainable_params:
                noise = 2 * torch.randint(
                    0, 2, p.size(), device=self.device, dtype=torch.long, generator=rng
                ).to(self.dtype) - 1
                p.add_(noise, alpha=alpha)

    def compute_v_with_server(
        self,
        inputs,
        labels,
        server_model,
        loss_fn,
        seeds,
        mu,
        communicator=None,
        scale_loss=1.0,
    ):
        P = len(seeds)

        # --- 1. Client Forward (Anchor) ---
        self.client_model.eval()
        with torch.no_grad():
            h_anchor, attention_mask = self.client_model(
                input_ids=inputs.input_ids, attention_mask=inputs.attention_mask
            )

        if communicator:
            communicator.log_activation(h_anchor)

        # --- 2. Server Forward & Backward ---
        server_input = h_anchor.detach().clone().to(self.device).requires_grad_(True)
        server_model.train()

        server_outputs = server_model(
            hidden_states=server_input,
            attention_mask=(
                attention_mask.to(self.device)
                if type(attention_mask) == torch.Tensor
                else attention_mask
            ),
        )

        loss = loss_fn(server_outputs, labels)
        loss_val = loss.item()

        # Server Backward
        (loss * scale_loss).backward()

        g_act = server_input.grad.detach()  # [B, Seq, Dim]

        if communicator:
            communicator.log_activation_gradient(g_act)

        # --- 3. ZO Projection (Compute v) ---
        v_list = []

        for i, seed in enumerate(seeds):
            rng = self.get_rng(seed, i)

            # 3.1 Apply Perturbation (+mu)
            self._generate_noise_and_apply(rng, mu)

            # 3.2 Forward Perturbed
            with torch.no_grad():
                out_p = self.client_model(
                    input_ids=inputs.input_ids, attention_mask=inputs.attention_mask
                )
                if hasattr(out_p, "last_hidden_state"):
                    h_p = out_p.last_hidden_state
                else:
                    h_p = out_p[0] if isinstance(out_p, tuple) else out_p

            # 3.3 Restore (-mu)
            rng_restore = self.get_rng(seed, i)
            self._generate_noise_and_apply(rng_restore, -mu)

            # 3.4 Compute Delta & Project
            v_scalar = torch.sum(h_p * g_act) - torch.sum(h_anchor * g_act)
            v_list.append(v_scalar)
            del h_p

        # Uplink: Scalars (Client -> Server)
        if communicator:
            communicator.log_scalars_uplink(P, 1)

        return torch.stack(v_list), loss_val

    def update_from_aggregated_v(self, aggregated_v, seeds, mu):
        P = len(seeds)
        base_scale = 1.0 / (mu * P)

        for i, seed in enumerate(seeds):
            v_val = aggregated_v[i].item()
            scale = v_val * base_scale

            rng = self.get_rng(seed, i)

            with torch.no_grad():
                for p in self.trainable_params:
                    # Rademacher noise: lower variance than Gaussian for ZO estimation
                    noise = 2 * torch.randint(
                        0, 2, p.size(), device=self.device, dtype=torch.long, generator=rng
                    ).to(self.dtype) - 1
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)
                    p.grad.add_(noise, alpha=scale)

        # Gradient clipping to prevent destabilizing updates from outlier ZO estimates
        torch.nn.utils.clip_grad_norm_(self.trainable_params, self.max_grad_norm)

    def zero_grad(self):
        for p in self.trainable_params:
            if p.grad is not None:
                p.grad.zero_()
