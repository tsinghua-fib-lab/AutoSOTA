import torch


class ProbabilityProjector:
    def project(self, R_t_X: torch.Tensor, R_t_E: torch.Tensor, X_t: torch.Tensor, E_t: torch.Tensor, dt: torch.Tensor):
        raise NotImplementedError


class RowStochasticProjector(ProbabilityProjector):
    """Zero diagonal and normalize rows to 1 after multiplying by dt."""

    def project(self, R_t_X: torch.Tensor, R_t_E: torch.Tensor, X_t: torch.Tensor, E_t: torch.Tensor, dt: torch.Tensor):
        if dt.dim() == 2:
            dt_x = dt[:, None, :]
            dt_e = dt[:, None, None, :]
        else:
            dt_x = dt
            dt_e = dt
        step_probs_X = R_t_X * dt_x  # (B, N, D)
        step_probs_E = R_t_E * dt_e  # (B, N, N, D)

        # zero diagonal entries and set diagonal so each row sums to 1
        x_argmax = X_t.argmax(-1)[:, :, None]
        e_argmax = E_t.argmax(-1)[:, :, :, None]

        step_probs_X = step_probs_X.scatter(-1, x_argmax, 0.0)
        step_probs_E = step_probs_E.scatter(-1, e_argmax, 0.0)

        step_probs_X = step_probs_X.scatter(
            -1, x_argmax, (1.0 - step_probs_X.sum(dim=-1, keepdim=True)).clamp(min=0.0)
        )
        step_probs_E = step_probs_E.scatter(
            -1, e_argmax, (1.0 - step_probs_E.sum(dim=-1, keepdim=True)).clamp(min=0.0)
        )

        return step_probs_X, step_probs_E


class ResidualAddProjector(ProbabilityProjector):
    """Residual add: prob = R*dt + one_hot; then clamp to valid range.

    Assumes X_t, E_t are one-hot encodings.
    """

    def project(self, R_t_X: torch.Tensor, R_t_E: torch.Tensor, X_t: torch.Tensor, E_t: torch.Tensor, h: torch.Tensor):
        step = h[0]
        prob_X = R_t_X * step + X_t
        prob_E = R_t_E * step + E_t
        prob_X[prob_X < 0] = 0.0
        prob_E[prob_E < 0] = 0.0
        torch.nan_to_num(prob_X, nan=0.0, posinf=1e5, neginf=0.0, out=prob_X)
        torch.nan_to_num(prob_E, nan=0.0, posinf=1e5, neginf=0.0, out=prob_E)
        # prob_X.clamp_(min=1e-6, max=1e3)
        # prob_E.clamp_(min=1e-6, max=1e3)
        # torch.nan_to_num(prob_X, nan=1e-6, posinf=1e3, neginf=0.0, out=prob_X)
        # torch.nan_to_num(prob_E, nan=1e-6, posinf=1e3, neginf=0.0, out=prob_E)
        return prob_X, prob_E


