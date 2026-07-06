import os
import time
import h5py
from datetime import datetime
import argparse
from pathlib import Path

import numpy as np
import torch
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


class gs_lvmogp_sarcos(gs_lvmogp_base):
    """
    sarcos dataset
    """
    def __init__(
            self, Q: int, D_H: int, M_H: int, M_X: int, qH_mean_field: bool,
            whitening: bool = True, jitter = 1e-6,
    ):
        # Hardcoded parameters for the sarcos dataset
        P = 7  # number of outputs
        D_X, batch_shape = 21, ()
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
        mean_pH = torch.zeros(mean_pH_shape, dtype=torch.get_default_dtype())
        diag_cov_pH = torch.ones(mean_pH_shape, dtype=torch.get_default_dtype())
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

        super(gs_lvmogp_sarcos, self).__init__(
            input_kernels=input_kernels, latent_kernels=latent_kernels, Q=Q, pH=pH, qH=qH, qU=qU, zH=zH, zX=zX,
            lik_model={"type": "Gaussian", "sigma_joint": sigma_joint, "sigma_init": sigma_init},
            whitening=whitening, jitter=jitter,
        )

    # override
    def train_gs_lvmogp(self, *args, **kwargs):
        coeff_exp_log_lik = 171266

        return super(gs_lvmogp_sarcos, self).train_lvmogp(*args, coeff_exp_log_lik=coeff_exp_log_lik, **kwargs)


def run_gs_lvmogp_sarcos(args):
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

    exp_result_folder = f"./results/sarcos/gs_lvmogp/{current_time}"
    os.makedirs(exp_result_folder, exist_ok=True)

    # Save configs
    with open(f'{exp_result_folder}/configs.txt', 'w') as f:
        for arg, value in vars(args).items():
            f.write(f"{arg}={value};\n")

    # Specify model
    model = gs_lvmogp_sarcos(
        Q=args.Q, D_H=args.D_H, M_H=args.M_H, M_X=args.M_X,
        qH_mean_field=args.qH_mean_field, whitening=args.whitening, jitter=jitter,
    )

    # Print model summary
    print_model_summary(model, exp_result_folder)

    # Specify dataset and dataloader for training
    file_path = Path(f"./data/sarcos.h5")
    assert file_path.exists()

    data_dict = rp_h5file(file_path, device=device)
    train_dataset = MyDataset(
        X=data_dict["all_X"], Y=data_dict['train_Y'], m=data_dict['train_mask'], data_device=data_device
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.input_batch_size, shuffle=True,
    )

    # Specify optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Train
    torch.cuda.synchronize()
    s = time.perf_counter()
    model.train_gs_lvmogp(
        train_dataloader=train_dataloader, optimizer=optimizer, epochs=args.epochs,
        output_batch_size=args.output_batch_size,
        beta_u=args.beta_u, beta_h=args.beta_h, max_norm=args.max_norm, device=device,
        print_epochs=args.num_print_epochs,
    )
    torch.cuda.synchronize()
    e = time.perf_counter()

    total_training_time = e - s
    print(f"Total training time: {total_training_time}\n")

    # Save model
    torch.save(model.state_dict(), f"{exp_result_folder}/model.pt")

    # Prediction
    metric_dict, pred_dict, _ = model.predict_lvmogp_gaussian(
        data_dict=data_dict, num_samples=args.num_samples, noiseless=False,
        num_plot_points=None, device=device
    )

    # all scalars:
    train_mse, train_nll, test_mse, test_nll = metric_dict["train_mse"], metric_dict["train_nll"], metric_dict["test_mse"], metric_dict["test_nll"]
    print(f"Total training time: {total_training_time}, train_mse: {train_mse}, train_nll: {train_nll}, test_mse: {test_mse}, test_nll: {test_nll}\n")

    # Save results
    with open(f'{exp_result_folder}/metrics.txt', 'w') as f:
        f.write(f"Total training time is: {total_training_time}\n")
        f.write(f"Train mse is: {train_mse}\n")
        f.write(f"Train nll is: {train_nll}\n")
        f.write(f"Test mse is: {test_mse}\n")
        f.write(f"Test nll is: {test_nll}\n")

    # Save prediction results on dataset input points
    with h5py.File(f'{exp_result_folder}/pred_dict.h5', 'w') as f:
        f.create_dataset('all_X', data=pred_dict['all_X'].cpu().numpy())
        f.create_dataset('pred_means', data=pred_dict['pred_means'].cpu().numpy())
        f.create_dataset('pred_vars', data=pred_dict['pred_vars'].cpu().numpy())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GS-LVMOGP on sarcos Dataset")

    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument('--float64', action='store_true', help='whether to use float64')

    parser.add_argument("--Q", type=int, default=3, help="Number of Coregionalization Matrices")
    parser.add_argument("--D_H", type=int, default=3, help="Latent dimension")
    parser.add_argument("--M_H", type=int, default=5, help="Number of inducing points on latent space")
    parser.add_argument("--M_X", type=int, default=40, help="Number of inducing points on input space")
    parser.add_argument("--qH_mean_field", action='store_true', help="Whether to use mean-field approximation for q(H)")
    parser.add_argument("--whitening", action='store_true', help="Whether to use whitening parametrisation")

    parser.add_argument("--output_batch_size", type=int, default=7, help="batch size for output")
    parser.add_argument("--input_batch_size", type=int, default=256, help="batch size for input data")
    parser.add_argument("--beta_u", type=float, default=1., help="beta parameter for KL_qU_pU")
    parser.add_argument("--beta_h", type=float, default=1., help="beta parameter for KL_qH_pH")
    parser.add_argument('--max_norm', type=float_or_none, default="none", help="max norm of gradients in grad norm clipping")
    parser.add_argument("--epochs", type=int, default=1000, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
    parser.add_argument('--num_samples', type=int, default=20, help='number of samples for prediction')
    parser.add_argument('--num_print_epochs', type=int, default=10, help='number of printing epochs')

    args_sarcos = parser.parse_args()

    run_gs_lvmogp_sarcos(args_sarcos)
