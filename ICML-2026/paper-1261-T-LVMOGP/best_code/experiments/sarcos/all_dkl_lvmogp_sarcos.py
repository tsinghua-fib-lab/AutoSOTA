import os
import time
import h5py
from datetime import datetime
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import gpytorch

from models.dkl_lvmogp_sarcos import dkl_lvmogp_sarcos
from utils.helpers import float_or_none, rp_h5file, print_model_summary
from utils.build_datasets import MyDataset
from utils.helpers import check_and_filter_args


def run_dkl_lvmogp_sarcos(args):
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

    args = check_and_filter_args(args)

    print("Parsed arguments:")
    for key, value in vars(args).items():
        print(f"{key}: {value}")

    sn = "no_sn"
    try:
        if args.spectral_norm:
            sn = f"with_sn={args.sn_ub}"
    except AttributeError:
        pass

    if args.tighter_elbo:
        exp_result_folder = f"./results/sarcos/dkl_lvmogp/{args.neural_network_type}_{args.qH_type}_{sn}_tighterELBO_init_sigma={args.sigma_init}_freeze={args.freeze_lik_before_epoch}/{current_time}"
    else:
        exp_result_folder = f"./results/sarcos/dkl_lvmogp/{args.neural_network_type}_{args.qH_type}_{sn}_init_sigma={args.sigma_init}_freeze={args.freeze_lik_before_epoch}/{current_time}"

    os.makedirs(exp_result_folder, exist_ok=True)

    # Save configs
    with open(f'{exp_result_folder}/configs.txt', 'w') as f:
        for arg, value in vars(args).items():
            f.write(f"{arg}={value};\n")

    assert args.qH_type == "Gaussian", "Only Gaussian qH is implemented for this dataset."

    if args.neural_network_type == "FCNet":
        model = dkl_lvmogp_sarcos(
            D_H=args.D_H, M=args.M, qH_mean_field=args.qH_mean_field, whitening=args.whitening,
            tighter_elbo=args.tighter_elbo, qU_type=args.qU_type,
            neural_network_type="FCNet", out_dim=args.out_dim, hidden_dims=args.hidden_dims, num_blocks=None,
            spectral_norm=args.spectral_norm, sn_ub=args.sn_ub, jitter=jitter,
            use_cache_for_svgp=args.use_cache_for_svgp, sigma_init=args.sigma_init,
            freeze_lik_before_epoch=args.freeze_lik_before_epoch,
        )
    elif args.neural_network_type == "ResNet":
        model = dkl_lvmogp_sarcos(
            D_H=args.D_H, M=args.M, qH_mean_field=args.qH_mean_field, whitening=args.whitening,
            tighter_elbo=args.tighter_elbo, qU_type=args.qU_type,
            neural_network_type="ResNet", out_dim=None, hidden_dims=None, num_blocks=args.num_blocks,
            spectral_norm=args.spectral_norm, sn_ub=args.sn_ub, jitter=jitter,
            use_cache_for_svgp=args.use_cache_for_svgp, sigma_init=args.sigma_init,
            freeze_lik_before_epoch=args.freeze_lik_before_epoch,
        )
    elif args.neural_network_type == "Identity":
        model = dkl_lvmogp_sarcos(
            D_H=args.D_H, M=args.M, qH_mean_field=args.qH_mean_field, whitening=args.whitening,
            tighter_elbo=args.tighter_elbo, qU_type=args.qU_type,
            neural_network_type="Identity", out_dim=None, hidden_dims=None, num_blocks=None,
            spectral_norm=None, sn_ub=None, jitter=jitter, use_cache_for_svgp=args.use_cache_for_svgp,
            sigma_init=args.sigma_init, freeze_lik_before_epoch=args.freeze_lik_before_epoch,
        )
    else:
        raise NotImplementedError

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
    if args.qU_type == "standard":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        optimizer_natural = None
    else:
        variational_params = list(model.qU.parameters())
        variational_ids = {id(p) for p in variational_params}
        other_params = [p for p in model.parameters() if id(p) not in variational_ids]
        optimizer = torch.optim.Adam(other_params, lr=args.lr)
        optimizer_natural = gpytorch.optim.NGD(variational_params, num_data=1, lr=args.natural_lr)

    # Train
    torch.cuda.synchronize()
    s = time.perf_counter()
    model.train_lvmogp(
        train_dataloader=train_dataloader, output_batch_size=args.output_batch_size,
        optimizer=optimizer, epochs=args.epochs, beta_u=args.beta_u, beta_h=args.beta_h,
        max_norm=args.max_norm, device=device, print_epochs=args.num_print_epochs,
        optimizer_natural=optimizer_natural,
    )
    torch.cuda.synchronize()
    e = time.perf_counter()

    total_training_time = e - s
    print(f"Total Training time: {total_training_time}\n")

    # Save model
    torch.save(model.state_dict(), f"{exp_result_folder}/model.pt")

    # Prediction
    metric_dict, pred_dict, _ = model.predict_lvmogp_gaussian(
        data_dict=data_dict, num_samples=args.num_samples, noiseless=False, num_plot_points=None, device=device
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
    parser = argparse.ArgumentParser(description="DKL-LVMOGP on sarcos dataset")

    parser.add_argument("--seed", type=int, default=3, help="Random seed")
    parser.add_argument("--float64", action='store_true', help='whether to use float64')

    parser.add_argument("--neural_network_type", type=str, default="Identity", choices=["FCNet", "ResNet", "Identity"])
    parser.add_argument("--qH_type", type=str, default="Gaussian", choices=["Gaussian", "Delta"])

    # Only for FCNet
    parser.add_argument("--hidden_dims", type=int, nargs="+", help="list of hidden layer dimensions")
    parser.add_argument("--out_dim", default=5, type=int, help="Neural Network output dimension, dimensionality of embedding space")

    # Only for ResNet
    parser.add_argument("--num_blocks", type=int, default=3, help="Number of residual blocks")

    # For FCNet and ResNet, but not Identity
    parser.add_argument("--spectral_norm", action='store_true', help="Whether to use spectral normalisation")
    parser.add_argument("--sn_ub", type=float, default=1., help="Upper bound for spectral normalisation")

    # Only for Gaussian qH
    parser.add_argument("--qH_mean_field", action='store_true', help="Whether to use mean-field approximation for q(H)")
    parser.add_argument("--beta_h", type=float, default=1., help="beta parameter for KL_qH_pH")
    parser.add_argument('--num_samples', type=int, default=20, help='number of samples for prediction')

    # Specific to sarcos dataset
    parser.add_argument("--sigma_init", type=float, default=0.01, help="initial sigma for Gaussian likelihood")
    parser.add_argument("--freeze_lik_before_epoch", type=int, default=500, help="freeze lik params before specified epoch")

    # Common hyperparameters
    parser.add_argument("--D_H", type=int, default=5, help="Latent dimension")
    parser.add_argument("--M", type=int, default=200, help="Number of inducing points")
    parser.add_argument("--qU_type", type=str, default="standard", choices=["standard", "natural", "tril-natural"])
    parser.add_argument("--whitening", action='store_true', help="Whether to use whitening parametrisation")
    parser.add_argument("--tighter_elbo", action='store_true', help="Whether to use the tighter ELBO")
    parser.add_argument("--use_cache_for_svgp", action='store_true', help="Whether to use cache mechanism for KL and exp_log_lik computation in SVGP.")

    parser.add_argument("--output_batch_size", type=int, default=16, help="batch size for output")
    parser.add_argument("--input_batch_size", type=int, default=256, help="batch size for input data")
    parser.add_argument("--beta_u", type=float, default=1., help="beta parameter for KL_qU_pU")
    parser.add_argument('--max_norm', type=float_or_none, default="none", help="max norm of gradients in grad norm clipping")
    parser.add_argument("--epochs", type=int, default=1000, help="number of epochs")
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
    parser.add_argument("--natural_lr", type=float, default=0.1, help="natural gradient descent learning rate, used for qU_type != standard")
    parser.add_argument('--num_print_epochs', type=int, default=10, help='number of printing epochs')

    args_sarcos = parser.parse_args()

    run_dkl_lvmogp_sarcos(args_sarcos)