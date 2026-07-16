import torch
import numpy as np
import logging


def loss(inp, params):
    a, b, e, alpha, beta = params
    pre_lse = torch.stack([
        a - alpha * torch.log(inp[:, 0]),
        b - beta * torch.log(inp[:, 1]),
        torch.as_tensor(e).expand((inp.shape[0]))
    ])
    post_lse = torch.logsumexp(pre_lse, dim=0)
    huber_loss = torch.nn.functional.huber_loss(post_lse, torch.log(inp[:, 2]), delta=1e-3, reduction='none')
    return huber_loss.sum()

def minimize_loss(inp, init_params=[6, 6, -1, 0.28, 0.32], steps=50):
    init_params = torch.Tensor(init_params).to("cpu")
    params = torch.nn.Parameter(init_params, requires_grad=True)

    lbfgs = torch.optim.LBFGS(
        [params],
        lr=1e-1,
        history_size=10,
        max_iter=20,
        line_search_fn="strong_wolfe"
    )

    def closure():
        lbfgs.zero_grad()
        l = loss(inp, params)
        l.backward()
        return l

    for i in range(steps):
        lbfgs.step(closure)
    return loss(inp, params), params

class ScalingLaw:
    def __init__(self, lin_space=4, device="cpu"):
        self.lin_space = lin_space
        self.device = device

    def map_parameters(self, model_size_str, include_embeddings=False):
        if include_embeddings:
            mapping = {
                "150m": 190532352,
                "300m": 371491840,
                "530m": 597683520,
                "750m": 758564352,
                "1b": 1279854592,
            }
        else:
            mapping = {
                "150m": 151898880,
                "300m": 319980544,
                "530m": 530074944,
                "750m": 681297408,
                "1b": 1176832000,
            }
        return mapping[model_size_str.lower()]

    def compute_predicted_value(self, a, b, e, alpha, beta, compute_value):
        N = np.sqrt(compute_value / (6 * 20 * 5 * 0.8547))
        D = 20 * N * 5 * 0.8547
        inp = torch.Tensor([[N, D]]).to(self.device)
        pre_lse = torch.stack([
            a - alpha * torch.log(inp[:, 0]),
            b - beta * torch.log(inp[:, 1]),
            torch.as_tensor(e).expand((inp.shape[0]))
        ])
        post_lse = torch.logsumexp(pre_lse, dim=0)
        return torch.exp(post_lse).item()

    def fit_scaling_law(self, df):
        inp = torch.Tensor([[N, D, L] for N, D, L in zip(df["param_count"], df["tokens"], df["values"])]).to(self.device)
        min_loss = float('inf')
        best_params = None

        param_combinations = [
            (a, b, e, alpha, beta)
            for a in np.linspace(0, 12, self.lin_space)
            for b in np.linspace(0, 12, self.lin_space)
            for e in np.linspace(-1, 1, self.lin_space)
            for alpha in np.linspace(0, 1, self.lin_space)
            for beta in np.linspace(0, 1, self.lin_space)
        ]

        for params_combination in param_combinations:
            l, params = self._evaluate_params(params_combination, inp)
            if l < min_loss:
                min_loss = l
                best_params = params

        if best_params is None:
            logging.warning(f"Best params returns None:\n{df}")
        return best_params

    def _evaluate_params(self, params_combination, inp):
        a, b, e, alpha, beta = params_combination
        l, params = minimize_loss(inp, [a, b, e, alpha, beta])
        return l.item(), params.detach().cpu().numpy()
