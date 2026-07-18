import torch
from clm_kernels import selective_adam_update


class SelectiveAdam(torch.optim.Adam):
    """
    A custom optimizer that extends the standard Adam optimizer by
    incorporating selective updates.

    This class is useful for situations where only a subset of parameters
    should be updated at each step, such as in sparse models or in cases where
    parameter visibility is controlled by an external mask.

    Additionally, the operations are fused into a single kernel. This optimizer
    leverages the `selective_adam_update` function from a CUDA backend for
    optimized sparse updates.

    This is one of the two optimizers mentioned in the Taming3DGS paper.

    Args:
        params (iterable): Iterable of parameters to optimize or dicts defining parameter groups.
        eps (float): Term added to the denominator to improve numerical stability (default: 1e-8).
        betas (Tuple[float, float]): Coefficients used for computing running averages of gradient and its square (default: (0.9, 0.999)).

    Examples:

        >>> N = 100
        >>> param = torch.randn(N, requires_grad=True)
        >>> optimizer = SelectiveAdam([param], eps=1e-8, betas=(0.9, 0.999))
        >>> visibility_mask = torch.cat([torch.ones(50), torch.zeros(50)])  # Visible first half, hidden second half

        >>> # Forward pass
        >>> loss = torch.sum(param ** 2)

        >>> # Backward pass
        >>> loss.backward()

        >>> # Optimization step with selective updates
        >>> optimizer.step(visibility=visibility_mask)

    """

    def __init__(self, params, eps, betas):
        super().__init__(params=params, eps=eps, betas=betas)

    @torch.no_grad()
    def step(self, visibility):
        N = visibility.numel()
        for group in self.param_groups:
            lr = group["lr"]
            eps = group["eps"]
            beta1, beta2 = group["betas"]

            assert len(group["params"]) == 1, "more than one tensor in group"
            param = group["params"][0]
            if param.grad is None:
                continue

            # Lazy state initialization
            state = self.state[param]
            if len(state) == 0:
                state["step"] = torch.tensor(0.0, dtype=torch.float32)
                state["exp_avg"] = torch.zeros_like(
                    param, memory_format=torch.preserve_format
                )
                state["exp_avg_sq"] = torch.zeros_like(
                    param, memory_format=torch.preserve_format
                )

            stored_state = self.state.get(param, None)
            exp_avg = stored_state["exp_avg"]
            exp_avg_sq = stored_state["exp_avg_sq"]
            M = param.numel() // N

            selective_adam_update(
                param,
                param.grad,
                exp_avg,
                exp_avg_sq,
                visibility,
                lr,
                beta1,
                beta2,
                eps,
                N,
                M,
            )


class ResidentAdamContext:
    """Metadata context for the paper-mode resident-block Adam updater.

    This class intentionally does not perform Adam updates or own optimizer
    moments. The real update path is GPUResidentAdam; it reads this context for
    parameter groups, Adam hyperparameters, and per-column learning rates.
    """

    def __init__(
        self,
        params,
        columns_sizes,
        columns_lr,
        lr=1e-3,
        bias_correction=True,
        betas=(0.9, 0.999),
        eps=1e-15,
        weight_decay=0,
        amsgrad=False,
        adamw_mode=False,
        fp32_optimizer_states=True,
        fused=False,
        sparse=False,
    ):
        if len(params) != 1 or params[0].get("name") != "unified_params":
            raise RuntimeError(
                "ResidentAdamContext expects the single unified_params metadata group "
                "used by the Pure SSD/Tide release path."
            )

        group = dict(params[0])
        param_list = list(group.get("params", []))
        if len(param_list) != 1:
            raise RuntimeError("ResidentAdamContext expects one metadata tensor.")

        metadata_tensor = param_list[0]
        if metadata_tensor.is_cuda:
            raise AssertionError(
                "ResidentAdamContext metadata tensor must stay on CPU; resident "
                "blocks are materialized separately for GPUResidentAdam."
            )

        group["params"] = param_list
        group.setdefault("lr", lr)
        group.setdefault("bias_correction", bias_correction)
        group.setdefault("betas", betas)
        group.setdefault("eps", eps)
        group.setdefault("weight_decay", weight_decay)
        group.setdefault("amsgrad", amsgrad)
        group.setdefault("adamw_mode", adamw_mode)
        group.setdefault("fp32_optimizer_states", fp32_optimizer_states)

        self.param_groups = [group]
        self.state = {}
        self.columns_sizes = columns_sizes
        self.columns_lr = (
            torch.tensor(columns_lr, dtype=torch.float32)
            if isinstance(columns_lr, list)
            else columns_lr
        )
        self.is_ssd_offload_mode = True
        self.cpu_adam = None
        self.gpu_adam = None
        self.fused = fused
        self.sparse = sparse

    def get_all_states(self):
        return []

    def zero_grad(self, set_to_none=False):
        for group in self.param_groups:
            for param in group.get("params", []):
                if param.grad is None:
                    continue
                if set_to_none:
                    param.grad = None
                else:
                    param.grad.zero_()

    def step(self, closure=None):
        if closure is not None:
            with torch.enable_grad():
                return closure()
        return None
