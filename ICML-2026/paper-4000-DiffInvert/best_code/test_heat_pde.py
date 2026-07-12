# pylint: disable=invalid-name
from typing import Tuple, Optional, Union
import argparse
import random
import yaml
import tqdm
from easydict import EasyDict
from matplotlib import pyplot as plt
import numpy as np
import torch

import deepxde as dde
import deepxde.config as dde_config
from deepxde.data.function_spaces import FunctionSpace
from deepxde.nn.pytorch.deeponet import DeepONetCartesianProd
from deepxde.metrics import l2_relative_error

from src import groups, energies, group_optimizer, group_sampler


dde.backend.set_default_backend("pytorch")


def set_global_seed(seed: int):
    """Set random seed for reproducibility."""
    dde_config.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def heat_pde(xt: np.ndarray, u: np.ndarray, nu: float) -> np.ndarray:
    """Heat equation and conditions"""
    # ∂u/∂t
    du_t = dde.grad.jacobian(u, xt, i=0, j=1)
    # ∂²u/∂x²
    du_xx = dde.grad.hessian(u, xt, i=0, j=0)
    assert du_xx is not None
    return du_t - nu * du_xx


def gen_testdata(data_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Import and preprocess the dataset with the exact solution."""
    # Load the data:
    data = np.load(data_path)

    # Obtain the values for t, x, and the excat solution:
    t, x, exact = data["t"], data["x"], data["usol"].T

    # Process the data and flatten it out (like labels and features):
    xx, tt = np.meshgrid(x, t)
    X = np.vstack((np.ravel(xx), np.ravel(tt))).T
    y = exact.flatten()[:, None]

    return X, y


class SineIC(FunctionSpace):
    """
    Function space for initial conditions u0(x) = sum_k A_k sin(l_k x + phi_k)
    for lielac, keep K=1, l=2, phi=0; only amplitude A varies between regimes.
    """
    def __init__(
        self,
        l: float = 2.0,
        L: float = 2 * np.pi,
        phi: float = 0.0,
        mode: str = "fixed",
        A_range: tuple = (0.5, 5.0)
    ):
        self.l = float(l)
        self.L = float(L)
        self.phi = float(phi)
        self.mode = mode
        self.A_range = A_range

    def random(self, size: int) -> np.ndarray:  # type: ignore
        if self.mode == "fixed":
            # in-distribution regime
            return np.ones((size, 1), dtype=np.float32)

        assert self.mode == "amp"
        # out-of-distribution regime
        # broader operator training distribution
        lo, hi = self.A_range
        return np.random.uniform(lo, hi, size=(size, 1)).astype(np.float32)

    def eval_one(self, feature: np.ndarray, x: float) -> float:  # type: ignore
        A = feature[0]
        return (A * np.sin(self.l * 2 * np.pi * x / self.L + self.phi)).astype(np.float32)

    def eval_batch(self, features: np.ndarray, xs: np.ndarray) -> np.ndarray:  # type: ignore
        A = features[:, :1]
        return (A * np.sin(self.l * 2 * np.pi * xs.T  / self.L + self.phi)).astype(np.float32)


def test(
    A_array: np.ndarray,
    ic_space: FunctionSpace,
    sensors_x: np.ndarray,
    inner_model: dde.Model,
    X: np.ndarray,
    y_true_base: np.ndarray,
    outer_model: Optional[Union[group_optimizer.GroupOptimizer, group_sampler.GroupSampler]],
    device: torch.device
) -> Tuple[float, float]:

    errs = []

    for A in tqdm.tqdm(A_array):
        y_true = A * y_true_base.squeeze()

        feat = np.array([[A]], dtype=np.float32)
        u_0 = ic_space.eval_batch(feat, sensors_x)
        assert isinstance(u_0, np.ndarray)

        if outer_model is None:
            y_pred = inner_model.predict((u_0, X))
            assert isinstance(y_pred, np.ndarray)
            y_pred = y_pred.squeeze()

        else:
            group = outer_model.group
            assert isinstance(group, groups.Heat1D)

            # initial condition jet_0: (x_0=sensors_x, t_0=0, u_0)
            # prediction points X: (x_f, t_f)
            jet_0 = torch.cat([
                torch.tensor(sensors_x, device=device).t(),
                torch.zeros(1, sensors_x.shape[0], device=device),
                torch.tensor(u_0, device=device)
            ], dim=0)  # [3, N_ic]
            X_f = torch.tensor(X, device=device).T  # [2, N_f]

            # find canonical transformation
            g = outer_model((jet_0[None], X_f[None]), None)

            # transform to canonical frame
            g_inv = group.inverse(g)
            jet_0_transformed = group.act(g_inv, jet_0[None])[0]
            X_f_transformed = group.act(g_inv, X_f[None])[0]

            # predict in canonical frame
            u_0_transformed = group.resample_to_sensors(jet_0_transformed[None])[0]
            t_0_transformed = jet_0_transformed[1]

            assert (t_0_transformed[1:] - t_0_transformed[:-1]).abs().max().item() < 1e-5, \
                "t_0 should be constant"

            X_f_transformed = torch.stack([
                X_f_transformed[0],
                X_f_transformed[1] - t_0_transformed.mean()
            ], dim=0)

            y_pred_transformed = inner_model.predict((
                u_0_transformed[None].float().cpu().detach().numpy(),
                X_f_transformed.t().float().cpu().detach().numpy()
            ))

            # transform back to original frame
            jet_f_transformed = torch.cat([
                X_f_transformed,
                torch.tensor(y_pred_transformed, device=device, dtype=torch.float64)
            ], dim=0)  # [3, N_f]

            jet_f = group.act(g, jet_f_transformed[None])[0]
            y_pred = jet_f[2].float().cpu().detach().numpy().squeeze()

        errs.append(l2_relative_error(y_true, y_pred))
        print(f"A={A:.3f} | rel L2 err: {errs[-1]:.6f}")

    errs = np.array(errs, dtype=float)
    mean = float(errs.mean())
    std_within_seed = float(errs.std())
    return mean, std_within_seed


def run_one_seed(seed, config) -> dict:
    """Train + test for one seed"""
    set_global_seed(seed)
    device = torch.device("cuda:0")

    # setup dataset
    X, y_true_base = gen_testdata("pde_data/heat.npz")

    # spatial-time geometry
    geom = dde.geometry.Interval(config.d_min, config.d_max)
    timedomain = dde.geometry.TimeDomain(config.t0, config.t1)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    # function spaces for initial condition
    ic_space = SineIC(l=2.0, L=2*np.pi, phi=0.0, mode=config.training_mode)
    ic_space_for_eval = SineIC(l=2.0, L=2*np.pi, phi=0.0, mode="fixed")
    sensors_x = np.linspace(config.d_min, config.d_max, config.N_ic, endpoint=True)[:, None]
    bounds = dict(
        x_min=config.d_min,
        x_max=config.d_max,
        t_min=config.t0,
        t_max=config.t1,
        u_min=-1.0,
        u_max=1.0
    )

    # setup group
    sensors_x_ = torch.tensor(sensors_x, device=device, dtype=torch.float64)[:, 0]
    group = groups.Heat1D(config.nu, sensors_x_)

    # setup energy
    energy = energies.LieLACL2Target(group, target=bounds)

    # setup outer model
    if config.outer == "none":
        outer_model = None

    elif config.outer == "kinetic_langevin":
        outer_model = group_sampler.EnergyKineticLangevinSamplerPDE(
            energy,
            temperature=config.temperature,
            step_size=config.step_size,
            steps=config.steps,
            friction=config.friction,
            clip_norm=config.clip_norm,
            num_hypothesis=config.num_hypothesis,
            init_scale=config.init_scale,
            dtype=config.dtype
        ).to(device)

    elif config.outer == "focal":
        outer_model = group_optimizer.FoCalOptimizerPDE(
            energy,
            num_hypothesis=config.num_hypothesis,
            init_scale=config.init_scale,
            init_points=config.init_points,
            n_iter=config.n_iter,
            opt_range=(config.opt_range_lower, config.opt_range_upper),
            seed=config.seed,
            verbose=config.verbose
        )

    elif config.outer == "lielac":
        outer_model = group_optimizer.LieLACOptimizerPDE(
            energy,
            optimizer=config.optimizer,
            step_size=config.step_size,
            steps=config.steps,
            num_hypothesis=config.num_hypothesis,
            init_scale=config.init_scale,
            verbose=config.verbose
        ).to(device).eval()

    elif config.outer == "diffusion":
        outer_model = group_sampler.EnergyDiffusionSamplerPDE(
            energy,
            temperature=config.temperature,
            steps=config.steps,
            noise_min=config.noise_min,
            noise_max=config.noise_max,
            clip_norm=config.clip_norm,
            num_mc=config.num_mc,
            num_hypothesis=config.num_hypothesis,
            dtype=config.dtype,
            verbose=config.verbose
        ).to(device).eval()

    else:
        raise NotImplementedError(f"Unknown outer model type: {config.outer}")

    # setup inner model
    dim_X = 2
    def on_x_boundary(X, on_boundary):
        x, _ = X
        return on_boundary and (
            np.isclose(x, config.d_min) or
            np.isclose(x, config.d_max)
        )
    bc0 = dde.icbc.PeriodicBC(
        geomtime,
        component_x=0,
        on_boundary=on_x_boundary,
        derivative_order=0
    )
    bc1 = dde.icbc.PeriodicBC(
        geomtime,
        component_x=0,
        on_boundary=on_x_boundary,
        derivative_order=1
    )
    ic = dde.icbc.IC(
        geomtime,
        func=(lambda _, aux: aux),
        on_initial=(lambda _, on_initial: on_initial)
    )
    time_pde = dde.data.TimePDE(
        geomtime,
        lambda xt, u, _: heat_pde(xt, u, config.nu),
        [bc0, bc1, ic],
        num_domain=config.N_dom,
        num_boundary=config.N_bc,
        num_initial=config.N_ic,
        num_test=config.N_dom
    )
    inner_model = dde.Model(
        dde.data.PDEOperatorCartesianProd(
            pde=time_pde,
            function_space=ic_space,
            evaluation_points=sensors_x,
            num_function=config.N_f_train,
            function_variables=[0],
            batch_size=config.batch_size,
        ),
        DeepONetCartesianProd(
            layer_sizes_branch=[config.N_ic, 100, 100, 100, 100, 100, 100, 100],
            layer_sizes_trunk=[dim_X, 100, 100, 100, 100, 100, 100, 100],
            activation="tanh",
            kernel_initializer="Glorot normal"
        ).to(device)
    )

    loss_weights = [config.alpha_PINN, 1.0, 1.0, config.gamma_data]
    inner_model.compile("adam", lr=0.001, loss_weights=loss_weights)

    if config.training_mode == "fixed":
        ckpt_path = "pretrained_checkpoints/heat_pde_deeponet.pt"
    else:
        assert config.training_mode == "amp"
        ckpt_path = "pretrained_checkpoints/heat_pde_deeponet_data_aug.pt"

    inner_model.restore(ckpt_path, device=device, verbose=2)
    inner_model.net.to(device)

    # ID: 10 samples per one seed
    As_id = np.ones(config.ID_N_f_test, dtype=np.float32)
    id_mean, id_std_within_seed = test(
        As_id,
        ic_space_for_eval,
        sensors_x,
        inner_model,
        X, y_true_base,
        outer_model,
        device
    )

    # OOD: 10 samples per one seed
    rng_ood = np.random.default_rng(13 + seed)
    As_ood = rng_ood.uniform(
        config.A_low,
        config.A_high,
        size=config.OOD_N_f_test
    ).astype(np.float32)
    ood_mean, ood_std_within_seed = test(
        As_ood,
        ic_space_for_eval,
        sensors_x,
        inner_model,
        X, y_true_base,
        outer_model,
        device
    )

    return {
        "seed": seed,
        "ID_mean": id_mean,
        "ID_std_within_seed": id_std_within_seed,
        "OOD_mean": ood_mean,
        "OOD_std_within_seed": ood_std_within_seed,
    }


def main(config):
    # fixed settings
    config.nu = 0.1
    config.d_min = 0.0
    config.d_max = 2 * np.pi
    config.t0 = 0.0
    config.t1 = 16.0
    config.N_dom = 500
    config.N_bc = 100
    config.N_ic = 200
    config.N_f_train = 10
    config.alpha_PINN = 150.0
    config.gamma_data = 20.0
    # Regime 1 (ID): "fixed" (A = 1)
    # Regime 2 (OOD): "amp" (A ~ U[0.5, 5.0])
    config.A_low = 0.5
    config.A_high = 5.0

    # repeat over seeds
    results = []

    for seed in range(config.num_seeds):
        r = run_one_seed(seed, config)

        results.append(r)

        print(f"[seed {seed:02d}] "
              f"ID(mean over {config.ID_N_f_test}): {r['ID_mean']:.6f} (± {r['ID_std_within_seed']:.6f}) | "
              f"OOD(mean over {config.OOD_N_f_test}): {r['OOD_mean']:.6f} (± {r['OOD_std_within_seed']:.6f})")

    # result statistics
    id_seed_means = np.array([r["ID_mean"] for r in results], dtype=float)
    ood_seed_means = np.array([r["OOD_mean"] for r in results], dtype=float)
    id_within_seed_stds = np.array([r["ID_std_within_seed"] for r in results], dtype=float)
    ood_within_seed_stds = np.array([r["OOD_std_within_seed"] for r in results], dtype=float)

    print("\n==== Summary over seeds ====")
    print(f"ID : {id_seed_means.mean():.6f} ± {id_seed_means.std():.6f}  "
          f"(within-seed std avg: {id_within_seed_stds.mean():.6f})")
    print(f"OOD: {ood_seed_means.mean():.6f} ± {ood_seed_means.std():.6f}  "
          f"(within-seed std avg: {ood_within_seed_stds.mean():.6f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to the config file.')
    args = parser.parse_args()
    with open(args.config, 'r', encoding="utf-8") as f:
        config_ = EasyDict(yaml.safe_load(f))

    main(config_)
