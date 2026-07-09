# %%
import math

import numpy as np
import torch
from scipy.linalg import toeplitz
from simu_data import (
    construct_discretize_mlp,
    construct_invertible_mlp,
    construct_poly_mixing,
    generate_latent_cov_mat,
)
from utils_train import (
    random_normal,
    random_normal_clamped,
    random_orthogonal_cayley,
    random_orthogonal_qr,
)

# %%


def setup_dgp_args(cfg):
    """
    Constructs the dataset arguments on CPU to ensure reproducibility and stability.
    """

    dgp_device = torch.device("cpu")  # generator on cpu
    dim_u = cfg.data.dim_v_true + cfg.data.dim_w_true
    poly_mix_weights = None

    # Calculate observed dimension
    if "poly" in cfg.data.mixing_type:
        dim_z = sum(
            math.comb(dim_u + d - 1, d)
            for d in range(cfg.data.polymix_degree + 1)
        )

        seed_val = cfg.mix_seed if cfg.mix_seed else cfg.data_seed
        generator = torch.Generator(device=dgp_device).manual_seed(seed_val)

        if cfg.data.mixing_type == "eyepoly":
            poly_mix_weights = torch.eye(dim_z, device=dgp_device)
        elif cfg.data.mixing_type == "qrpoly":
            poly_mix_weights = random_orthogonal_qr(
                dim_z, device=dgp_device, generator=generator
            )
        elif cfg.data.mixing_type == "cayleypoly":
            poly_mix_weights = random_orthogonal_cayley(
                dim_z,
                max_norm=cfg.data.cayley_max_norm,
                device=dgp_device,
                generator=generator,
            )
        elif cfg.data.mixing_type == "normalpoly":
            poly_mix_weights = random_normal(
                dim_z, device=dgp_device, generator=generator
            )
        elif cfg.data.mixing_type == "normalclamppoly":
            poly_mix_weights = random_normal_clamped(
                dim_z,
                max_cond=5,
                method="clip",
                device=dgp_device,
                generator=generator,
            )
        elif cfg.data.mixing_type == "normalsmoothpoly":
            poly_mix_weights = random_normal_clamped(
                dim_z,
                max_cond=5,
                method="exact",
                device=dgp_device,
                generator=generator,
            )
        print(
            f"\nCondition number of mixing weight: {torch.linalg.cond(poly_mix_weights)}"
        )
    else:
        dim_z = (
            cfg.data.dim_z
            if cfg.data.dim_z
            else (dim_u + int(cfg.data.intercept))
        )

    print(f"observed instrument dimension: {dim_z}.")

    # Covariances of V and W
    if cfg.data.fixed_sig:
        Sig_v = (
            torch.from_numpy(
                toeplitz(
                    (np.arange(cfg.data.dim_v_true, 0, -1))
                    / cfg.data.dim_v_true
                )
            )
            .float()
            .to(dgp_device)
        )
        Sig_w = (
            torch.from_numpy(
                toeplitz(
                    (np.arange(cfg.data.dim_w_true, 0, -1))
                    / cfg.data.dim_w_true
                )
            )
            .float()
            .to(dgp_device)
        )
    else:
        Sig_v = generate_latent_cov_mat(
            dim=cfg.data.dim_v_true, df=cfg.data.dim_v_true * 3
        ).to(dgp_device)
        Sig_w = generate_latent_cov_mat(
            dim=cfg.data.dim_w_true, df=cfg.data.dim_w_true * 3
        ).to(dgp_device)

    # SCM Parameters
    alpha1 = torch.zeros((cfg.data.dim_v_true, 1), device=dgp_device) + 1.0
    alpha2 = torch.zeros((cfg.data.dim_v_true, 1), device=dgp_device) + 1.0
    beta = torch.ones(
        cfg.data.dim_v_true + cfg.data.dim_w_true, 1, device=dgp_device
    )
    theta = torch.tensor(1.0, device=dgp_device)

    etas = [
        torch.eye(cfg.data.dim_v_true, cfg.data.dim_v_true, device=dgp_device)
    ]
    if cfg.data.n_pop == 2:
        etas.append(
            torch.eye(
                cfg.data.dim_v_true, cfg.data.dim_v_true, device=dgp_device
            )
            * 2
        )
    elif cfg.data.n_pop == 3 and cfg.data.dim_v_true == 2:
        tmp = torch.eye(
            cfg.data.dim_v_true, cfg.data.dim_v_true, device=dgp_device
        )
        tmp[0, 0] = 2.0
        etas.append(tmp)
        tmp = torch.eye(
            cfg.data.dim_v_true, cfg.data.dim_v_true, device=dgp_device
        )
        tmp[1, 1] = 2.0
        etas.append(tmp)

    # Mixing functions
    Beta = None
    mixing_fn = None
    seed_val = cfg.mix_seed if cfg.mix_seed else cfg.data_seed

    if cfg.data.mixing_type == "linear":
        Beta = torch.randn(
            cfg.data.dim_v_true + cfg.data.dim_w_true + int(cfg.data.intercept),
            cfg.data.dim_z,
            generator=torch.Generator(device=dgp_device).manual_seed(seed_val),
            device=dgp_device,
        )
    elif "poly" in cfg.data.mixing_type:
        mixing_fn = construct_poly_mixing(
            degree=cfg.data.polymix_degree,
            latent_dim=cfg.data.dim_v_true + cfg.data.dim_w_true,
            output_dim=None,
            weights=poly_mix_weights,  # on cpu
        )
    elif cfg.data.mixing_type == "invmlp":
        if cfg.data.discretize:
            mixing_fn = construct_discretize_mlp(
                ns=[cfg.data.dim_z, 2 * cfg.data.dim_z, 3 * cfg.data.dim_z],
                act_fct=cfg.data.invmlp_actfun,
                seed=seed_val,
            )
        else:
            mixing_fn = construct_invertible_mlp(
                n=cfg.data.dim_z,
                n_layers=2,
                act_fct=cfg.data.invmlp_actfun,
                seed=seed_val,
            )
            print(mixing_fn[0].weight[:10])

    dataset_args = {
        "Sig_v": Sig_v,
        "Sig_w": Sig_w,
        "Sig_hs": [
            torch.eye(cfg.data.dim_v_true, device=dgp_device)
            for _ in range(cfg.data.n_pop)
        ],
        "mean_hs": [
            torch.zeros(cfg.data.dim_v_true, device=dgp_device) + pp * 2
            for pp in range(cfg.data.n_pop)
        ],
        "etas": etas,
        "alpha1s": [alpha1 for _ in range(cfg.data.n_pop)],
        "alpha2s": [alpha2 for _ in range(cfg.data.n_pop)],
        "thetas": [theta for _ in range(cfg.data.n_pop)],
        "betas": [beta for _ in range(cfg.data.n_pop)],
        "Beta": Beta,
        "intercept": cfg.data.intercept,
        "dim_z": dim_z,
        "mixing_fn": mixing_fn,
        "discretize": cfg.data.discretize,
    }

    return dataset_args, dim_z, poly_mix_weights
