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

from gpytorch.means import ZeroMean

from kernels.rbf_kernel import MyRBFKernel

from baselines.graphical_mogp_base import graphical_mogp_base
from models.building_blocks.gp_modules import mo_Variational_inducing_dist, Inducing_points, svgp_base
from utils.build_datasets import MyDataset
from utils.helpers import float_or_none, rp_h5file, print_model_summary


class graphical_mogp_era5(graphical_mogp_base):
    def __init__(
            self, M: int, all_Y: Tensor, all_m: Tensor,  # used to register cos_sim
            whitening: bool = True, jitter: float = 1e-6,
    ):
        # Hardcoded parameters for the ERA5 dataset
        P, D_X, batch_shape = 3395, 1, ()
        sigma_joint, sigma_init = True, 0.01

        # Z, inducing points
        IP_init_shape = batch_shape + (P, M, D_X,)
        IP_init = torch.randn(IP_init_shape, dtype=torch.get_default_dtype())
        Z = Inducing_points(M, D_X, IP_init, IP_name="Z", IP_joint=True)

        # qU
        qU = mo_Variational_inducing_dist(P, M, batch_shape=batch_shape, jitter=jitter)

        # mean function
        my_mean = ZeroMean(batch_shape=torch.Size(batch_shape + (P,)))

        MyKernel = MyRBFKernel(
            multi_output=True, has_outputscale=True, ard_num_dims=D_X, batch_shape=torch.Size(batch_shape + (P,)),
        )

        lik_model = {"type": "Gaussian", "sigma_joint": sigma_joint, "sigma_init": sigma_init}

        super(graphical_mogp_era5, self).__init__(
            mean=my_mean, kernel=MyKernel, Z=Z, qU=qU, lik_model=lik_model, whitening=whitening, jitter=jitter
        )

        # cos_sim should be registered before training
        self.register_cos_sim(Y=all_Y, m=all_m)

    # override
    def train_gmogp(self, *args, coeff_exp_log_lik = 71850, **kwargs):
        return super(graphical_mogp_era5, self).train_gmogp(*args, coeff_exp_log_lik=coeff_exp_log_lik, **kwargs)


def run_gmogp_era5(args):
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

    exp_result_folder = f"./results/era5/{args.missingness}/graphical_mogp/{current_time}"
    os.makedirs(exp_result_folder, exist_ok=True)

    # Save configs
    with open(f'{exp_result_folder}/configs.txt', 'w') as f:
        for arg, value in vars(args).items():
            f.write(f"{arg}={value};\n")

    # Read data
    file_path = Path(f"./data/era5_{args.missingness}.h5")
    assert file_path.exists()

    data_dict = rp_h5file(file_path, device=device)

    # Specify model
    model = graphical_mogp_era5(
        M=args.M, all_Y=data_dict['train_Y'], all_m=data_dict['train_mask'], whitening=args.whitening, jitter=jitter
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
    torch.cuda.synchronize()
    s = time.perf_counter()
    model.train_gmogp(
        train_dataloader=train_dataloader, output_batch_size=args.output_batch_size, optimizer=optimizer,
        epochs=args.epochs, beta=args.beta, max_norm=args.max_norm, device=device, print_epochs=args.num_print_epochs
    )
    torch.cuda.synchronize()
    e = time.perf_counter()

    total_training_time = e - s
    print(f"Total Training time: {total_training_time}\n")

    # Prediction
    metric_dict, pred_dict, _ = model.predict_gmogp_gaussian(
        data_dict=data_dict, noiseless=False, num_plot_points=None, device=device
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
    parser = argparse.ArgumentParser(description="Graphical MOGP on ERA5 Dataset")

    parser.add_argument("--missingness", type=str, default="block", choices=["block", "random"], help="Type of missing data pattern")

    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument('--float64', action='store_true', help='whether to use float64')

    parser.add_argument("--M", type=int, default=10, help="Number of inducing points for each output")
    parser.add_argument("--whitening", action='store_true', help="Whether to use whitening parametrisation")

    parser.add_argument("--output_batch_size", type=int, default=128, help="batch size for output")
    parser.add_argument("--input_batch_size", type=int, default=30, help="batch size for input data")
    parser.add_argument("--beta", type=float, default=1., help="beta parameter for KL_qU_pU")
    parser.add_argument('--max_norm', type=float_or_none, default="none", help="max norm of gradients in grad norm clipping")
    parser.add_argument("--epochs", type=int, default=1000, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
    parser.add_argument('--num_print_epochs', type=int, default=10, help='number of printing epochs')

    args_era5 = parser.parse_args()

    run_gmogp_era5(args_era5)

