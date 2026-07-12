# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
from typing import Optional, Union, Tuple, Callable
import tqdm

import torch
from torch import nn, Tensor

from bayes_opt import BayesianOptimization
from bayes_opt.acquisition import ExpectedImprovement
from sklearn.gaussian_process.kernels import RBF

from .groups import Group, ConnectedMatrixLieGroup
from .energies import Energy


class GroupOptimizer(nn.Module):
    def __init__(self, group: Union[Group, ConnectedMatrixLieGroup]):
        super().__init__()
        self.group = group

    def forward(self, x: Tensor, y: Optional[Tensor]) -> Tensor:
        """
        Optimize group elements based on the input tensors.
        Return:
            g: [bsize, degree, degree]
        """
        raise NotImplementedError


class GroupEnergyOptimizer(GroupOptimizer):  # pylint: disable=abstract-method
    def __init__(self, energy: Energy):
        super().__init__(energy.group)
        self.energy = energy


class LieLACOptimizer(GroupEnergyOptimizer):
    def __init__(
        self,
        energy: Energy,
        step_size: float,
        steps: int,
        num_hypothesis: int,
        init_scale: float,
        verbose: bool,
    ):
        super().__init__(energy)
        self.step_size = step_size
        self.steps = steps
        self.num_hp = num_hypothesis
        self.init_scale = init_scale
        self.verbose = verbose

    def _repeat_hypothesis_and_flatten(self, x: Tensor) -> Tensor:
        bsize = x.shape[0]
        if self.num_hp == 1:
            return x.clone()
        return x[:, None].expand(bsize, self.num_hp, *x.shape[1:]).flatten(0, 1)

    def _deflatten_hypothesis(self, x: Tensor, bsize: int) -> Tensor:
        return x.view(bsize, self.num_hp, *x.shape[1:])

    @torch.enable_grad()
    @torch.inference_mode(False)
    def forward(self, x: Tensor, y: Optional[Tensor]) -> Tensor:
        """
        Optimize group elements based on the input tensors.
        Arguments:
            x: [bsize, ...]
            y: [bsize, ...] (optional)
        Return:
            g: [bsize, ...]
        """
        bsize = x.shape[0]

        coeff = self.group.random_coeff(bsize * self.num_hp) * self.init_scale
        coeff.requires_grad_(True)

        x = self._repeat_hypothesis_and_flatten(x)
        y = self._repeat_hypothesis_and_flatten(y) if y is not None else None

        optimizer = torch.optim.Adam([coeff], lr=self.step_size)

        best_energy = torch.zeros(bsize * self.num_hp, device=x.device) + float("inf")
        best_coeff = torch.zeros_like(coeff)
        for _ in tqdm.tqdm(range(self.steps), bar_format='{desc:<5.5}{percentage:3.0f}%|{bar:10}{r_bar}'):
            optimizer.zero_grad()

            g = self.group.exp(coeff)
            energy = self.energy(self.group.inverse(g), x, y).to(best_energy.dtype)
            energy.sum().backward()

            is_best = energy < best_energy
            best_energy[is_best] = energy[is_best]
            best_coeff[is_best] = coeff[is_best]

            # print grad norm
            if self.verbose:
                assert coeff.grad is not None
                grad_norm = coeff.grad.norm()
                print(f"grad norm: {grad_norm.item():.2e}", end=", ")

            optimizer.step()

            if self.verbose:
                energy_ = self._deflatten_hypothesis(best_energy, bsize).min(dim=1)[0]
                print(f"\nenergy mean {energy_.mean().item():.4e} " +
                      f"median {energy_.median().item():.4e} " +
                      f"max {energy_.max().item():.4e} " +
                      f"min {energy_.min().item():.4e}")

        best_energy = self._deflatten_hypothesis(best_energy, bsize)
        best_coeff = self._deflatten_hypothesis(best_coeff, bsize)
        hypothesis_index = best_energy.argmin(dim=1)
        best_coeff = best_coeff[torch.arange(bsize), hypothesis_index]

        g = self.group.exp(best_coeff)
        return g

class LieLACOptimizerPDE(GroupEnergyOptimizer):
    def __init__(
        self,
        energy: Energy,
        optimizer: str,
        step_size: float,
        steps: int,
        num_hypothesis: int,
        init_scale: float,
        verbose: bool,
    ):
        super().__init__(energy)
        self.step_size = step_size
        self.optimizer = optimizer
        self.steps = steps
        self.num_hp = num_hypothesis
        self.init_scale = init_scale
        self.verbose = verbose

    def _repeat_hypothesis_and_flatten(self, x: Tensor) -> Tensor:
        bsize = x.shape[0]
        if self.num_hp == 1:
            return x.clone()
        return x[:, None].expand(bsize, self.num_hp, *x.shape[1:]).flatten(0, 1)

    def _deflatten_hypothesis(self, x: Tensor, bsize: int) -> Tensor:
        return x.view(bsize, self.num_hp, *x.shape[1:])

    @torch.enable_grad()
    @torch.inference_mode(False)
    def forward(self, x: Tuple[Tensor, Tensor], y: Optional[Tensor]) -> Tensor:  # type: ignore
        """
        Optimize group elements based on the input tensors.
        Arguments:
            x: tuple([bsize, ...], [bsize, ...])
            y: [bsize, ...] (optional)
        Return:
            g: [bsize, ...]
        """
        jet_0, X_f = x

        bsize = jet_0.shape[0]
        device = jet_0.device

        coeff = self.group.random_coeff(bsize * self.num_hp) * self.init_scale
        coeff.requires_grad_(True)

        jet_0 = self._repeat_hypothesis_and_flatten(jet_0)
        X_f = self._repeat_hypothesis_and_flatten(X_f)
        x = (jet_0, X_f)
        y = self._repeat_hypothesis_and_flatten(y) if y is not None else None

        if self.optimizer == 'sgd':
            optimizer = torch.optim.SGD([coeff], lr=self.step_size)
        elif self.optimizer == 'adam':
            optimizer = torch.optim.Adam([coeff], lr=self.step_size)
        else:
            raise ValueError(f"Unknown optimizer {self.optimizer}")

        best_energy = torch.zeros(bsize * self.num_hp, device=device) + float("inf")
        best_coeff = torch.zeros_like(coeff)
        for i in range(self.steps):
            optimizer.zero_grad()

            g = self.group.exp(coeff)
            energy = self.energy(self.group.inverse(g), x, y).to(best_energy.dtype)
            energy.sum().backward()

            is_best = energy < best_energy
            best_energy[is_best] = energy[is_best]
            best_coeff[is_best] = coeff[is_best]

            # print grad norm
            if self.verbose:
                assert coeff.grad is not None
                grad_norm = coeff.grad.norm()
                print(f"grad norm: {grad_norm.item():.2e}", end=", ")

            optimizer.step()

            if self.verbose:
                energy_ = self._deflatten_hypothesis(best_energy, bsize).min(dim=1)[0]

                print(f"\nstep {i + 1} / {self.steps}")

                print(f"energy mean {energy_.mean().item():.4e} " +
                      f"median {energy_.median().item():.4e} " +
                      f"max {energy_.max().item():.4e} " +
                      f"min {energy_.min().item():.4e}")

        best_energy = self._deflatten_hypothesis(best_energy, bsize)
        best_coeff = self._deflatten_hypothesis(best_coeff, bsize)
        hypothesis_index = best_energy.argmin(dim=1)
        best_coeff = best_coeff[torch.arange(bsize), hypothesis_index]

        g = self.group.exp(best_coeff)
        return g


class FoCalOptimizer(GroupEnergyOptimizer):
    def __init__(
        self,
        energy: Energy,
        num_hypothesis: int,
        init_scale: float,
        init_points: int = 450,
        n_iter: int = 150,
        opt_range: Tuple[float, float] = (-1.0, 1.0),
        seed: int = 1,
        verbose: bool = False
    ):
        super().__init__(energy)
        self.num_hp = num_hypothesis
        self.init_scale = init_scale
        self.verbose = verbose
        self.init_points = init_points
        self.n_iter = n_iter
        self.opt_range = opt_range
        self.seed = seed

    def _repeat_hypothesis_and_flatten(self, x: Tensor) -> Tensor:
        bsize = x.shape[0]
        if self.num_hp == 1:
            return x.clone()
        return x[:, None].expand(bsize, self.num_hp, *x.shape[1:]).flatten(0, 1)

    def _deflatten_hypothesis(self, x: Tensor, bsize: int) -> Tensor:
        return x.view(bsize, self.num_hp, *x.shape[1:])

    def run_bayesian_optimization(
        self,
        target_fn: Callable[[float], float],
        init_prob: Tensor,
        init_points: int = 450,
        n_iter: int = 150,
        opt_range: Tuple[float, float] = (-1.0, 1.0),
        seed: int = 1
    ) -> BayesianOptimization:
        """Run Bayesian optimization for gamma/contrast alignment.

        Args:
            target_fn: Function to optimize
            init_gamma_grid: Initial gamma grid to probe (in log space)
            init_random_points: Number of initial points sampled randomly
            n_iter: Number of iterations
            opt_range: Range for optimization in log space

        Returns:
            Optimized BayesianOptimization object
        """
        alg_dim = self.group.num_generators
        pbounds = {}

        # search range for optimization
        for i in range(alg_dim):
            low = opt_range[0]
            high = opt_range[1]
            pbounds[f"b{i}"] = (low, high)

        optimizer = BayesianOptimization(
            f=target_fn,
            acquisition_function=ExpectedImprovement(xi=0.0),
            pbounds=pbounds,
            random_state=seed,
            verbose=0,
            allow_duplicate_points=True,
        )

        optimizer.set_gp_params(alpha=0.01, kernel=RBF(length_scale=0.01))

        params = {j: init_prob[i].item() for i, j in enumerate(pbounds.keys())}
        optimizer.probe(
            params=params, lazy=False
        )

        optimizer.maximize(
            init_points=init_points,
            n_iter=n_iter,
        )

        return optimizer

    @torch.enable_grad()
    @torch.inference_mode(False)
    def forward(self, x: Tensor, y: Optional[Tensor]) -> Tensor:
        """
        Optimize group elements based on the input tensors.
        Arguments:
            x: [bsize, ...]
            y: [bsize, ...] (optional)
        Return:
            g: [bsize, ...]
        """
        bsize = x.shape[0]

        coeff = self.group.random_coeff(bsize * self.num_hp) * self.init_scale

        x = self._repeat_hypothesis_and_flatten(x)
        y = self._repeat_hypothesis_and_flatten(y) if y is not None else None

        best_energy = torch.zeros(bsize * self.num_hp, device=x.device) - float("inf")
        best_coeff = torch.zeros(bsize * self.num_hp, self.group.num_generators).to(x.device)

        for i in range(bsize * self.num_hp):
            x_bo = x[i:i+1]
            y_bo = y[i:i+1] if y is not None else None
            init_probe = coeff[i]

            def bo_target_fn(**kwargs) -> Tensor:
                coeff = torch.tensor([float(kwargs[k]) for k in kwargs]).unsqueeze(0).to(x.device)
                g = self.group.exp(coeff)
                energy = self.energy(self.group.inverse(g), x_bo, y_bo)
                return - energy.sum().item()

            optimizer = self.run_bayesian_optimization(
                bo_target_fn,  # type: ignore
                init_probe,
                init_points=self.init_points,
                n_iter=self.n_iter,
                opt_range=self.opt_range,
                seed=self.seed
            )

            assert isinstance(optimizer.max, dict)
            energy = float(optimizer.max["target"])
            coeff_dict = optimizer.max["params"]
            _coeff = torch.tensor([v for k, v in coeff_dict.items()]).unsqueeze(0)

            if best_energy[i] < energy:
                best_energy[i] = energy
                best_coeff[i] = _coeff

        best_energy = self._deflatten_hypothesis(best_energy, bsize)
        best_coeff = self._deflatten_hypothesis(best_coeff, bsize)
        hypothesis_index = best_energy.argmin(dim=1)
        best_coeff = best_coeff[torch.arange(bsize), hypothesis_index]

        g = self.group.exp(best_coeff)
        return g


class FoCalOptimizerPDE(FoCalOptimizer):
    @torch.enable_grad()
    @torch.inference_mode(False)
    def forward(self, x: Tuple[Tensor, Tensor], y: Optional[Tensor]) -> Tensor:  # type: ignore
        """
        Optimize group elements based on the input tensors.
        Arguments:
            x: [bsize, ...]
            y: [bsize, ...] (optional)
        Return:
            g: [bsize, ...]
        """
        jet_0, X_f = x

        bsize = jet_0.shape[0]
        device = jet_0.device
        dtype = jet_0.dtype

        coeff = self.group.random_coeff(bsize * self.num_hp) * self.init_scale

        jet_0 = self._repeat_hypothesis_and_flatten(jet_0)
        X_f = self._repeat_hypothesis_and_flatten(X_f)
        x = (jet_0, X_f)
        y = self._repeat_hypothesis_and_flatten(y) if y is not None else None

        best_energy = torch.zeros(bsize * self.num_hp, device=device) - float("inf")
        best_coeff = torch.zeros(bsize * self.num_hp, self.group.num_generators).to(device)

        for i in range(bsize * self.num_hp):
            x_bo = (x[0][i:i+1], x[1][i:i+1])
            y_bo = y[i:i+1] if y is not None else None
            init_probe = coeff[i]

            def bo_target_fn(**kwargs) -> Tensor:
                coeff = torch.tensor([float(kwargs[k]) for k in kwargs], dtype=dtype, device=device).unsqueeze(0)
                g = self.group.exp(coeff)
                energy = self.energy(self.group.inverse(g), x_bo, y_bo)
                return - energy.sum().item()

            optimizer = self.run_bayesian_optimization(
                bo_target_fn,  # type: ignore
                init_probe,
                init_points=self.init_points,
                n_iter=self.n_iter,
                opt_range=self.opt_range,
                seed=self.seed
            )

            assert isinstance(optimizer.max, dict)
            energy = float(optimizer.max["target"])
            coeff_dict = optimizer.max["params"]
            _coeff = torch.tensor([v for k, v in coeff_dict.items()]).unsqueeze(0)

            if best_energy[i] < energy:
                best_energy[i] = energy
                best_coeff[i] = _coeff

        best_energy = self._deflatten_hypothesis(best_energy, bsize)
        best_coeff = self._deflatten_hypothesis(best_coeff, bsize)
        hypothesis_index = best_energy.argmin(dim=1)
        best_coeff = best_coeff[torch.arange(bsize), hypothesis_index].to(dtype)

        g = self.group.exp(best_coeff)
        return g
