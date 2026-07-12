# pylint: disable=missing-function-docstring,missing-class-docstring,missing-module-docstring,invalid-name
from typing import Optional, Tuple, Union
import argparse
import yaml
import tqdm
from easydict import EasyDict
import numpy as np
from matplotlib import pyplot as plt
import scipy.io
import torch
import jax.numpy as jnp
from jax import random

from src import groups, energies, group_optimizer, group_sampler
from src.utils.burgers.model import PIDeepONet
from src.utils.burgers.data import generate_one_test_data


def test(
    idx: np.ndarray,
    usol: jnp.ndarray,
    P: int,
    model: PIDeepONet,
    params: Tuple,
    outer_model: Optional[Union[group_optimizer.GroupOptimizer, group_sampler.GroupSampler]],
    device: torch.device
) -> float:
    # u_test: (101 * 101, 101), ic
    # y_test: (101 * 101, 2), (t, x)
    # s_test: (101 * 101), u
    u_test, y_test, s_test = generate_one_test_data(idx, usol, P)

    if outer_model is None:
        # s_pred: (101 * 101,)
        s_pred = model.predict_s(params, u_test, y_test)

    else:
        group = outer_model.group
        assert isinstance(group, groups.Burgers1D)

        s_pred_orig = model.predict_s(params, u_test, y_test)

        u0 = torch.tensor(np.array(u_test), device=device, dtype=torch.float64)[0]
        x0 = torch.linspace(0, 1, P, device=device, dtype=torch.float64)
        t0 = torch.zeros_like(u0, dtype=torch.float64)

        # initial condition jet_0: (x_0, t_0=0, u_0)
        # prediction points X: (x_f, t_f)
        jet_0 = torch.stack([x0, t0, u0], dim=0)  # [3, N_ic = 101]
        x_f = torch.tensor(np.array(y_test[:, 1]), device=device, dtype=torch.float64)
        t_f = torch.tensor(np.array(y_test[:, 0]), device=device, dtype=torch.float64)
        X_f = torch.stack([x_f, t_f], dim=0)  # [2, N_f = 101 * 101]

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

        u_test = jnp.tile(jnp.array(u_0_transformed[None].float().cpu().detach()), (P**2, 1))
        y_test = jnp.array(X_f_transformed[[1, 0]].t().float().cpu().detach())
        s_pred_transformed = model.predict_s(params, u_test, y_test)

        # transform back to original frame
        jet_f_transformed = torch.cat([
            X_f_transformed,
            torch.tensor(np.array(s_pred_transformed), device=device)[None]
        ], dim=0)  # [3, N_f]
        jet_f = group.act(g, jet_f_transformed[None])[0]
        s_pred = jnp.array(jet_f[2].float().cpu().detach()).squeeze()

    error = jnp.linalg.norm(s_test - s_pred) / jnp.linalg.norm(s_test)
    return error


def main(config):
    # fixed settings
    config.N = 2000  # number of total input samples
    config.N_train = 1000  # number of input samples used for training
    config.m = 101  # number of sensors for input samples
    config.P_ics_train = 101  # number of locations for evaluating the initial condition
    config.P_bcs_train = 100  # number of locations for evaluating the boundary condition
    config.P_res_train = 2500  # number of locations for evaluating the PDE residual
    config.P_test = 101  # resolution of uniform grid for the test data

    device = torch.device("cuda:0")

    # setup dataset
    data_path = "pde_data/burgers.mat" if config.mode == "id" else "pde_data/burgers_ood.mat"
    data = scipy.io.loadmat(data_path)
    usol = jnp.array(data['output'])
    assert usol.shape == (config.N, config.m, config.m)
    sensors_x = torch.linspace(0, 1, config.P_test, device=device)
    bounds = dict(
        t_min=0.0,
        t_max=1.0,
        x_min=0.0,
        x_max=1.0,
        u_min=-1.0,
        u_max=1.0
    )

    # setup group
    group = groups.Burgers1D(sensors_x)

    # setup energy
    energy = energies.LieLACL2Target(group, target=bounds).to(device)

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
    inner_model = PIDeepONet(
        branch_layers=[config.m, 100, 100, 100, 100, 100, 100, 100],
        trunk_layers=[2, 100, 100, 100, 100, 100, 100, 100]
    )
    ckpt_path = "pretrained_checkpoints/burgers_pde_deeponet.npy"
    params = inner_model.unravel_params(jnp.load(ckpt_path))

    # compute relative l2 error over test data
    idx = random.randint(
        key=random.PRNGKey(12345),
        shape=(400,),
        minval=config.N_train,
        maxval=2000
    )
    k = 1500
    idx = np.arange(k, k + config.N_test)

    errors_list = []
    for i in tqdm.tqdm(range(config.N_test)):
        error_i = test(idx[i], usol, config.P_test, inner_model, params, outer_model, device)
        print(f"idx={idx[i]}, relative L2 error of s: {error_i:.6f}")
        errors_list.append(error_i)

    errors = np.array(errors_list)
    mean_error = errors.mean()

    print(f"Mean relative L2 error of s: {mean_error:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to the config file.')
    args = parser.parse_args()
    with open(args.config, 'r', encoding="utf-8") as f:
        config_ = EasyDict(yaml.safe_load(f))

    main(config_)
