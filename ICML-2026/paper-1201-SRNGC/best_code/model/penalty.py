import torch
import torch.nn as nn

class Shapley_Penalty(nn.Module):
    def __init__(self, num_proj=1, approx=True, individual_effect_only=False, device='cpu'):
        super().__init__()
        # num_proj = -1 means exact method (call the full jacobian matrix)
        # else use random projections
        self.num_proj = num_proj
        # approx = False means use the exact method as in shapley value (L-1 norm on interaction term)
        # else use the approximation (F-norm on interaction term)
        self.approx = approx
        self.device = device
        self.individual_effect_only = individual_effect_only

        if approx == False and self.num_proj != -1:
            raise NotImplementedError("L-1 norm version only supports exact method (num_proj=-1)")

    # only when num_proj = -1 we consider full jacobian
    def _full_jacobian(self, y, x):
        batch_size, output_dim = y.shape
        input_dim = x.shape[1]
        jac = torch.zeros(batch_size, output_dim, input_dim, device=self.device)
        for i in range(output_dim):
            grad_y = torch.zeros_like(y)
            grad_y[:, i] = 1.0
            grad_x, = torch.autograd.grad(
                outputs=y,
                inputs=x,
                grad_outputs=grad_y,
                create_graph=True
            )
            jac[:, i, :] = grad_x
        return jac

    def interaction_effect_rand_proj(self, model, x, num_prob=1, dist='rademacher'):
        batch_size, input_dim = x.shape

        est_total = 0.0
        for _ in range(num_prob):
            if dist == 'rademacher':
                prob_1 = (torch.randint(0, 2, (batch_size, input_dim), device=self.device).float() * 2 - 1)
                prob_2 = (torch.randint(0, 2, (batch_size, input_dim), device=self.device).float() * 2 - 1)
            else:
                prob_1 = torch.randn(batch_size, input_dim, device=self.device)
                prob_2 = torch.randn(batch_size, input_dim, device=self.device)

            y1 = torch.autograd.functional.jvp(model, x, prob_1, create_graph=True)[1]
            y2 = torch.autograd.functional.jvp(model, x, prob_2, create_graph=True)[1]

            est_total += ((y1 * y2) ** 2).sum()

        return est_total / (num_prob * batch_size)
    
    def individual_effect_rand_proj(self, model, x, num_prob=1, dist='rademacher'):
        y = model(x)

        individual_effect = 0
        for _ in range(num_prob):
            if dist == 'rademacher':
                prob = torch.randint(0, 2, y.shape, device=self.device, dtype=x.dtype) * 2 - 1
            else:
                prob = torch.randn(y.shape, device=self.device, dtype=x.dtype)

            grad,  = torch.autograd.grad(y, x, prob, create_graph=True)
            individual_effect += (grad ** 2).sum()

        return individual_effect / (num_prob * x.shape[0])

    def forward(self, model, x):
        if self.num_proj == -1: # no random projection
            y = model(x)
            full_jacobian = self._full_jacobian(y, x) # [batch, output_dim, input_dim]

            if self.individual_effect_only and not self.approx:
                # L-1 norm of individual effect (Jacobian regularization)
                individual_effect = full_jacobian.abs().sum() / x.shape[0]
            else:
                # F-norm of individual effect
                individual_effect = torch.sum(full_jacobian.pow(2)) / x.shape[0]

            if self.individual_effect_only:
                return individual_effect, 0.0

            if self.approx:
                # F-norm of interaction effect
                # E[ \sum_i |J_i:^T J_i:|_F^2 ]
                interaction_effect = (full_jacobian.pow(2).sum(dim=2).pow(2).sum(dim=1)).mean()
            else:
                # L1 norm (absolute value) of interaction effect
                # Exact form of Shapley value
                # \sum_i \sum_j \sum_k E( |J_ij J_ik| )
                interaction_effect = full_jacobian.abs().sum(dim=2).pow(2).sum() / x.shape[0]
        else:
            # only for approx = True, use the approx version of Shapley value
            # we split the computation of individual effect and interaction effect for readability
            # this can be optimized by combining the two parts
            individual_effect = self.individual_effect_rand_proj(model, x, num_prob=self.num_proj)
            if self.individual_effect_only: # JRNGC penalty
                return individual_effect, 0.0
            interaction_effect = self.interaction_effect_rand_proj(model, x, num_prob=self.num_proj)

        if self.approx:
            return 0.5 * individual_effect, 0.5 * interaction_effect
        else:
            return individual_effect, 0.5 * (interaction_effect - individual_effect)

class Weight_Penalty(nn.Module):
    def __init__(self, model_type):
        super().__init__()
        self.model_type = model_type

    # ---------- LSTM penalty (both componentwise and full) ----------

    def _lstm_penalty_single(self, lstm_block: nn.Module) -> torch.Tensor:
        W = lstm_block.lstm.weight_ih_l0      # [4*hidden, p]
        return torch.sum(torch.norm(W, dim=0))

    def _lstm_penalty(self, model) -> torch.Tensor:
        # componentwise: sum over all per-target LSTMs
        if getattr(model, "componentwise", False):
            return sum(self._lstm_penalty_single(block) for block in model.blocks)
        else:
            return self._lstm_penalty_single(model.block)

    # ---------- MLP penalty (GL / GSGL / H), both componentwise and full ----------

    def _mlp_penalty_single(self, mlp_block: nn.Module, dim: int, lag: int) -> torch.Tensor:
        # First module in the Sequential is always a Linear for our MLPBlock.
        first_layer = mlp_block.net[0]
        assert isinstance(first_layer, nn.Linear), \
            "First layer of MLPBlock must be nn.Linear for penalty to work."

        W_raw = first_layer.weight          # [hidden_out, p*lag]
        hidden_out = W_raw.shape[0]
        assert W_raw.shape[1] == dim * lag, \
            f"Expected in_features = dim*lag, got {W_raw.shape[1]} vs {dim}*{lag}"

        # Reshape to [hidden, p, lag] to match original conv1d semantics.
        W = W_raw.view(hidden_out, dim, lag)  # [hidden, p, lag]

        total = 0.0
        for i in range(lag):
            total = total + torch.sum(torch.norm(W[:, :, :(i + 1)], dim=(0, 2)))
        return total

    def _mlp_penalty(self, model) -> torch.Tensor:
        dim = model.dim
        lag = model.lag

        if getattr(model, "componentwise", False):
            # Sum penalty across all per-target MLPs.
            return sum(self._mlp_penalty_single(block, dim, lag) for block in model.blocks)
        else:
            # Single full MLP.
            return self._mlp_penalty_single(model.block, dim, lag)

    # ---------- dispatcher ----------

    def forward(self, model) -> torch.Tensor:
        if self.model_type == 'cMLP':
            return self._mlp_penalty(model)
        elif self.model_type == 'cLSTM':
            return self._lstm_penalty(model)
        else:
            raise ValueError(f"Unknown model_type {self.model_type} for Weight_Penalty.")