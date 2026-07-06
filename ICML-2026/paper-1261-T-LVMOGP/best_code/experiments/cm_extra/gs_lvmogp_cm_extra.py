import os
import time
import h5py
from datetime import datetime
import argparse
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from kernels.rbf_kernel import MyRBFKernel

from baselines.gs_lvmogp_base import (
    gs_lvmogp_base,
    gs_Prior_H,
    gs_Variational_H,
    gs_Variational_inducing_dist,
    gs_Inducing_points
)
from utils.build_datasets import MyDataset
from utils.helpers import float_or_none, rp_h5file, print_model_summary
from utils.metrics import gaussian_nll


class gs_lvmogp_cm_extra(gs_lvmogp_base):
    """
    copernicus marine dataset
    """
    def __init__(
            self, Q: int, pH_mean: Tensor, M_H: int, M_X: int, qH_mean_field: bool,
            whitening: bool = True, jitter = 1e-6, pH_cov_value: float = 0.01, # covariance value for prior of H
    ):
        # Hardcoded parameters for the cm_extra dataset
        P = 10873
        D_X, D_H, batch_shape = 1, 2, ()
        sigma_joint, sigma_init = True, 0.01

        input_kernels = [
            MyRBFKernel(
                multi_output=False, has_outputscale=True, ard_num_dims=D_X, batch_shape=torch.Size(batch_shape),
            )
        ]

        latent_kernels = [
            MyRBFKernel(
                multi_output=False, has_outputscale=True, ard_num_dims=D_H, batch_shape=torch.Size(batch_shape),
            )
        ]

        # pH
        mean_pH_shape = batch_shape + (Q, P, D_H,)
        # mean_pH = torch.zeros(mean_pH_shape, dtype=torch.get_default_dtype())
        mean_pH = pH_mean.squeeze(-1).expand(Q, P, D_H) # TODO: for generic usage, consider batch_shape
        diag_cov_pH = pH_cov_value * torch.ones(mean_pH_shape, dtype=torch.get_default_dtype())
        pH = gs_Prior_H(Q, mean_pH, diag_cov_pH)

        # qH
        qH = gs_Variational_H(Q, P, D_H, batch_shape=batch_shape, mean_field=qH_mean_field)

        # qU
        qU = gs_Variational_inducing_dist(M_H, M_X, batch_shape=batch_shape)

        # Z, inducing points on input space and latent space
        zH_IP_init_shape = batch_shape + (Q, M_H, D_H,)
        zX_IP_init_shape = batch_shape + (M_X, D_X,)
        zH_IP_init = torch.randn(zH_IP_init_shape, dtype=torch.get_default_dtype())
        zX_IP_init = torch.randn(zX_IP_init_shape, dtype=torch.get_default_dtype())
        zH = gs_Inducing_points(M_H, D_H, IP_init=zH_IP_init, IP_name="zH", IP_joint=True)
        zX = gs_Inducing_points(M_X, D_X, IP_init=zX_IP_init, IP_name="zX", IP_joint=True)

        super(gs_lvmogp_cm_extra, self).__init__(
            input_kernels=input_kernels, latent_kernels=latent_kernels, Q=Q, pH=pH, qH=qH, qU=qU, zH=zH, zX=zX,
            lik_model={"type": "Gaussian", "sigma_joint": sigma_joint, "sigma_init": sigma_init},
            whitening=whitening, jitter=jitter,
        )

    # override
    def train_gs_lvmogp(self, *args, **kwargs):
        coeff_exp_log_lik = int(10873 * 24)

        return super(gs_lvmogp_cm_extra, self).train_lvmogp(*args, coeff_exp_log_lik=coeff_exp_log_lik, **kwargs)


def run_gs_lvmogp_cm_extra(args):
    # set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.float64:
        torch.set_default_dtype(torch.float64)
    else:
        torch.set_default_dtype(torch.float32)

    jitter = 1e-8 if args.float64 else 1e-6

    if torch.cuda.is_available():
        device = 'cuda'
        data_device = 'cuda'  # or 'cpu'
        torch.cuda.manual_seed(args.seed)
    else:
        device = 'cpu'
        data_device = 'cpu'

    current_time = datetime.now().strftime("%m%d_%H_%M_%S")
    print("Parsed arguments:")
    for key, value in vars(args).items():
        print(f"{key}: {value}")

    exp_result_folder = f"./results/cm_extra/gs_lvmogp/Q={args.Q}/{current_time}"
    os.makedirs(exp_result_folder, exist_ok=True)

    # Save configs
    with open(f'{exp_result_folder}/configs.txt', 'w') as f:
        for arg, value in vars(args).items():
            f.write(f"{arg}={value};\n")

    # Load data to get pH mean, specify model
    file_path = Path(f"./data/cm_extra.h5")
    assert file_path.exists()

    data_dict = rp_h5file(file_path, device=device)
    pH_mean = data_dict['train_sc']

    # Specify model
    model = gs_lvmogp_cm_extra(
        Q=args.Q, pH_mean=pH_mean, M_H=args.M_H, M_X=args.M_X,
        qH_mean_field=args.qH_mean_field, whitening=args.whitening, jitter=jitter,
        pH_cov_value=args.pH_cov_value,
    )

    # Print model summary
    print_model_summary(model, exp_result_folder)

    # Specify dataset and dataloader for training
    train_dataset = MyDataset(
        X=data_dict["all_X"], Y=data_dict['train_Y'], m=data_dict['train_mask'], data_device=data_device
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.input_batch_size, shuffle=True,
    )

    # Specify optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Train
    s = time.time()
    model.train_gs_lvmogp(
        train_dataloader=train_dataloader, optimizer=optimizer, epochs=args.epochs,
        output_batch_size=args.output_batch_size,
        beta_u=args.beta_u, beta_h=args.beta_h, max_norm=args.max_norm, device=device,
        print_epochs=args.num_print_epochs,
    )
    e = time.time()

    total_training_time = e - s
    print(f"Total training time: {total_training_time}\n")

    # Save model
    torch.save(model.state_dict(), f"{exp_result_folder}/model.pt")

    # Predict on training data (observed outputs)
    s_train_qf_mean, s_train_qy_cov = model.predict(
        x_star=data_dict['all_X'], num_samples=20, device=device, noiseless=True,
    )  # [s, N, P_train]

    train_qf_mean = s_train_qf_mean.mean(dim=0)  # [N, P_train]
    train_qy_cov = s_train_qy_cov.mean(dim=0) + s_train_qf_mean.var(dim=0)  # [N, P_train]

    # Predict on test data (unseen outputs)
    test_qf_mean, test_qy_cov = model.predict_given_H(
        x_star=data_dict['all_X'], H_values=data_dict['test_sc'], num_samples=20,
        pH_cov_value=args.new_output_pH_cov_value, device=device, noiseless=True,
    )  # [N, P_test], pH_cov_value=args.pH_cov_value

    train_mse = ((train_qf_mean - data_dict['train_Y']).square()).mean().item()
    train_nll = gaussian_nll(data_dict['train_Y'], train_qf_mean, train_qy_cov).mean().item()
    test_mse = ((test_qf_mean - data_dict['test_Y']).square()).mean().item()
    test_nll = gaussian_nll(data_dict['test_Y'], test_qf_mean, test_qy_cov).mean().item()

    print(f"Train mse: {train_mse}, Train nll: {train_nll}, Test mse: {test_mse}, Test nll: {test_nll}\n")

    # Save results
    with open(f'{exp_result_folder}/metrics.txt', 'w') as f:
        f.write(f"Total training time is: {total_training_time}\n")
        f.write(f"Train mse is: {train_mse}\n")
        f.write(f"Train nll is: {train_nll}\n")
        f.write(f"Test mse is: {test_mse}\n")
        f.write(f"Test nll is: {test_nll}\n")
        f.write(f"Sigma is: {model.lik_model.sigma.item()}\n")

    # Save prediction results on dataset input points
    with h5py.File(f'{exp_result_folder}/pred_dict.h5', 'w') as f:
        f.create_dataset('all_X', data=data_dict['all_X'].cpu().numpy())
        f.create_dataset('train_sc', data=data_dict['train_sc'].cpu().numpy())
        f.create_dataset('test_sc', data=data_dict['test_sc'].cpu().numpy())
        f.create_dataset('train_pred_means', data=train_qf_mean.cpu().numpy())
        f.create_dataset('train_pred_vars', data=train_qy_cov.cpu().numpy())
        f.create_dataset('test_pred_means', data=test_qf_mean.cpu().numpy())
        f.create_dataset('test_pred_vars', data=test_qy_cov.cpu().numpy())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GS-LVMOGP on cm_extra Dataset")

    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument('--float64', action='store_true', help='whether to use float64')

    # Specific to cm_extra dataset for extrapolation experiments
    parser.add_argument("--pH_cov_value", type=float, default=0.01, help="covariance value for prior of H")
    parser.add_argument("--new_output_pH_cov_value", type=float, default=0.01, help="covariance value for new output H")

    parser.add_argument("--Q", type=int, default=3, help="Number of Coregionalization Matrices")
    parser.add_argument("--M_H", type=int, default=20, help="Number of inducing points on latent space")
    parser.add_argument("--M_X", type=int, default=15, help="Number of inducing points on input space")
    parser.add_argument("--qH_mean_field", action='store_true', help="Whether to use mean-field approximation for q(H)")
    parser.add_argument("--whitening", action='store_true', help="Whether to use whitening parametrisation")

    parser.add_argument("--output_batch_size", type=int, default=16, help="batch size for output")
    parser.add_argument("--input_batch_size", type=int, default=256, help="batch size for input data")
    parser.add_argument("--beta_u", type=float, default=1., help="beta parameter for KL_qU_pU")
    parser.add_argument("--beta_h", type=float, default=1., help="beta parameter for KL_qH_pH")
    parser.add_argument('--max_norm', type=float_or_none, default="none", help="max norm of gradients in grad norm clipping")
    parser.add_argument("--epochs", type=int, default=1000, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
    parser.add_argument('--num_samples', type=int, default=20, help='number of samples for prediction')
    parser.add_argument('--num_print_epochs', type=int, default=10, help='number of printing epochs')

    args_cm = parser.parse_args()

    run_gs_lvmogp_cm_extra(args_cm)