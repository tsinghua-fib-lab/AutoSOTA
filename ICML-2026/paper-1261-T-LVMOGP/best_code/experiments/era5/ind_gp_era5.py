import os
import time
import h5py
from datetime import datetime
import argparse
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from kernels.rbf_kernel import MyRBFKernel

from baselines.ind_gp_base import ind_exact_gp
from utils.helpers import float_or_none, rp_h5file, print_model_summary


__all__ = ["ind_gp_era5"]


class ind_gp_era5(ind_exact_gp):
    def __init__(
            self, train_X: Tensor, train_Y: Tensor, train_mask: Tensor, jitter: float = 1e-6
    ):
        # Hardcoded parameters for the ERA5 dataset
        D_X, batch_shape = 1, ()
        sigma_joint, sigma_init = True, 0.01
        # init_lengthscale = 0.05

        MyKernel = MyRBFKernel(
            multi_output=False, has_outputscale=True, ard_num_dims=D_X, batch_shape=torch.Size(batch_shape),
        )

        # MyKernel.lengthscale = init_lengthscale

        super(ind_gp_era5, self).__init__(
            kernel=MyKernel, train_X=train_X, train_Y=train_Y, train_mask=train_mask, sigma_joint=sigma_joint,
            sigma_init=sigma_init, jitter=jitter
        )


def run_ind_gp_era5(args):
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

    exp_result_folder = f"./results/era5/{args.missingness}/ind_gp/{current_time}"
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
    model = ind_gp_era5(
        train_X=data_dict["all_X"], train_Y=data_dict["train_Y"], train_mask=data_dict["train_mask"], jitter=jitter
    )

    # Print model summary
    print_model_summary(model, exp_result_folder)

    # Specify optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Train
    # torch.cuda.synchronize()
    s = time.perf_counter()
    model.train_ind_exact_gp(
        optimizer=optimizer, epochs=args.epochs, method='approach1', device=device, print_epochs=args.num_print_epochs,
    )
    # torch.cuda.synchronize()
    e = time.perf_counter()

    total_training_time = e - s
    print(f"Total Training time: {total_training_time}\n")

    # Save model
    torch.save(model.state_dict(), f"{exp_result_folder}/model.pt")

    # Prediction
    metric_dict, pred_dict, _ = model.predict_ind_gp(
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
    parser = argparse.ArgumentParser(description="Independent Exact GP on ERA5 Dataset")

    parser.add_argument("--missingness", type=str, default="block", choices=["block", "random"], help="Type of missing data pattern")

    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument('--float64', action='store_true', help='whether to use float64')

    parser.add_argument("--max_norm", type=float_or_none, default="none", help="max norm of gradients in grad norm clipping")
    parser.add_argument("--epochs", type=int, default=1000, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
    parser.add_argument('--num_print_epochs', type=int, default=10, help='number of printing epochs')

    args_era5 = parser.parse_args()

    run_ind_gp_era5(args_era5)


