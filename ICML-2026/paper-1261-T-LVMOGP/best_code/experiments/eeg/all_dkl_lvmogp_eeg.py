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

from models.dkl_lvmogp_eeg import dkl_lvmogp_eeg, det_dkl_lvmogp_eeg
from utils.eeg import eeg_plot
from utils.helpers import float_or_none, rp_h5file, print_model_summary
from utils.build_datasets import MyDataset
from utils.helpers import check_and_filter_args
from utils.diagnosis import visualize_embeddings
from torch.optim.lr_scheduler import CosineAnnealingLR


def run_dkl_lvmogp_eeg(args):
    # set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # for debugging:
    # torch.set_printoptions(precision=50)
    # set number of threads
    # torch.set_num_threads(1)

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
        exp_result_folder = f"./results/eeg/dkl_lvmogp/{args.neural_network_type}_{args.qH_type}_{sn}_tighterELBO/{current_time}"
    else:
        exp_result_folder = f"./results/eeg/dkl_lvmogp/{args.neural_network_type}_{args.qH_type}_{sn}/{current_time}"

    os.makedirs(exp_result_folder, exist_ok=True)

    # Save configs
    with open(f'{exp_result_folder}/configs.txt', 'w') as f:
        for arg, value in vars(args).items():
            f.write(f"{arg}={value};\n")

    # Specify model
    if args.qH_type == "Gaussian":
        if args.neural_network_type == "FCNet":
            model = dkl_lvmogp_eeg(
                D_H=args.D_H, M=args.M, qH_mean_field=args.qH_mean_field, whitening=args.whitening,
                tighter_elbo=args.tighter_elbo, qU_type=args.qU_type,
                neural_network_type="FCNet", out_dim=args.out_dim, hidden_dims=args.hidden_dims, num_blocks=None,
                spectral_norm=args.spectral_norm, sn_ub=args.sn_ub, jitter=jitter, use_cache_for_svgp=args.use_cache_for_svgp
            )
        elif args.neural_network_type == "ResNet":
            model = dkl_lvmogp_eeg(
                D_H=args.D_H, M=args.M, qH_mean_field=args.qH_mean_field, whitening=args.whitening,
                tighter_elbo=args.tighter_elbo, qU_type=args.qU_type,
                neural_network_type="ResNet", out_dim=None, hidden_dims=None, num_blocks=args.num_blocks,
                spectral_norm=args.spectral_norm, sn_ub=args.sn_ub, jitter=jitter, use_cache_for_svgp=args.use_cache_for_svgp
            )
        elif args.neural_network_type == "Identity":
            model = dkl_lvmogp_eeg(
                D_H=args.D_H, M=args.M, qH_mean_field=args.qH_mean_field, whitening=args.whitening,
                tighter_elbo=args.tighter_elbo, qU_type=args.qU_type,
                neural_network_type="Identity", out_dim=None, hidden_dims=None, num_blocks=None,
                spectral_norm=None, sn_ub=None, jitter=jitter, use_cache_for_svgp=args.use_cache_for_svgp
            )
        else:
            raise NotImplementedError

    elif args.qH_type == "Delta":
        if args.neural_network_type == "FCNet":
            model = det_dkl_lvmogp_eeg(
                D_H=args.D_H, M=args.M, whitening=args.whitening, tighter_elbo=args.tighter_elbo, qU_type=args.qU_type,
                neural_network_type="FCNet", out_dim=args.out_dim, hidden_dims=args.hidden_dims, num_blocks=None,
                spectral_norm=args.spectral_norm, sn_ub=args.sn_ub, jitter=jitter, use_cache_for_svgp=args.use_cache_for_svgp
            )
        elif args.neural_network_type == "ResNet":
            model = det_dkl_lvmogp_eeg(
                D_H=args.D_H, M=args.M, whitening=args.whitening, tighter_elbo=args.tighter_elbo, qU_type=args.qU_type,
                neural_network_type="ResNet", out_dim=None, hidden_dims=None, num_blocks=args.num_blocks,
                spectral_norm=args.spectral_norm, sn_ub=args.sn_ub, jitter=jitter, use_cache_for_svgp=args.use_cache_for_svgp
            )
        elif args.neural_network_type == "Identity":
            model = det_dkl_lvmogp_eeg(
                D_H=args.D_H, M=args.M, whitening=args.whitening, tighter_elbo=args.tighter_elbo, qU_type=args.qU_type,
                neural_network_type="Identity", out_dim=None, hidden_dims=None, num_blocks=None,
                spectral_norm=None, sn_ub=None, jitter=jitter, use_cache_for_svgp=args.use_cache_for_svgp
            )
        else:
            raise NotImplementedError

    else:
        raise NotImplementedError

    # To rebuttal against Reviewer 6zJP, we may want to fix qH in the model to be non-trainable.
    if args.fix_qH:
        for param in model.qH.parameters():
            param.requires_grad = False

    # Print model summary
    print_model_summary(model, exp_result_folder)

    # Specify dataset and dataloader for training
    file_path = Path("./data/eeg.h5")
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

    # LR scheduler (cosine annealing)
    if args.lr_schedule:
        lr_scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr_min)
        _orig_hook = model._epoch_start_hook
        def _hooked_start(epoch):
            _orig_hook(epoch)
            lr_scheduler.step()
        model._epoch_start_hook = _hooked_start

    # Train
    s = time.time()
    if args.qH_type == "Gaussian":
        model.train_lvmogp(
            train_dataloader=train_dataloader, output_batch_size=args.output_batch_size,
            optimizer=optimizer, epochs=args.epochs, beta_u=args.beta_u, beta_h=args.beta_h,
            max_norm=args.max_norm, device=device, print_epochs=args.num_print_epochs,
            optimizer_natural=optimizer_natural,
        )

    elif args.qH_type == "Delta":
        model.train_lvmogp(
            train_dataloader=train_dataloader, output_batch_size=args.output_batch_size,
            optimizer=optimizer, epochs=args.epochs, beta_u=args.beta_u,
            max_norm=args.max_norm, device=device, print_epochs=args.num_print_epochs,
            optimizer_natural=optimizer_natural,
        )

    else:
        raise NotImplementedError

    e = time.time()

    total_training_time = e - s
    print(f"Total Training time: {total_training_time}\n")

    # Save model
    torch.save(model.state_dict(), f"{exp_result_folder}/model.pt")

    # Prediction
    if args.qH_type == "Gaussian":
        metric_dict, pred_dict, plot_pred_dict = model.predict_lvmogp_gaussian(
            data_dict=data_dict, num_samples=args.num_samples, noiseless=False, num_plot_points=2000, device=device
        )
    elif args.qH_type == "Delta":
        metric_dict, pred_dict, plot_pred_dict = model.predict_lvmogp_gaussian(
            data_dict=data_dict, noiseless=False, num_plot_points=2000, device=device
        )
    else:
        raise NotImplementedError

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

    # Plot
    eeg_plot(data_dict, plot_pred_dict, exp_result_folder=exp_result_folder)

    # Model diagnosis
    visualize_embeddings(
        model=model, data_dict=data_dict, save_path=f"{exp_result_folder}/embeddings.pdf", figsize=(16, 9)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DKL-LVMOGP on EEG Dataset")

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

    # Common hyperparameters
    parser.add_argument("--D_H", type=int, default=5, help="Latent dimension")
    parser.add_argument("--M", type=int, default=200, help="Number of inducing points")
    parser.add_argument("--qU_type", type=str, default="standard", choices=["standard", "natural", "tril-natural"])
    parser.add_argument("--whitening", action='store_true', help="Whether to use whitening parametrisation")
    parser.add_argument("--tighter_elbo", action='store_true', help="Whether to use the tighter ELBO")
    parser.add_argument("--use_cache_for_svgp", action='store_true', help="Whether to use cache mechanism for KL and exp_log_lik computation in SVGP.")

    parser.add_argument("--output_batch_size", type=int, default=8, help="batch size for output")
    parser.add_argument("--input_batch_size", type=int, default=128, help="batch size for input data")
    parser.add_argument("--beta_u", type=float, default=1., help="beta parameter for KL_qU_pU")
    parser.add_argument('--max_norm', type=float_or_none, default=10., help="max norm of gradients in grad norm clipping")
    parser.add_argument("--epochs", type=int, default=1000, help="number of epochs")
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
    parser.add_argument("--natural_lr", type=float, default=0.1, help="natural gradient descent learning rate, used for qU_type != standard")
    parser.add_argument('--lr_schedule', action='store_true', help='Use cosine annealing LR schedule')
    parser.add_argument('--lr_min', type=float, default=1e-5, help='Minimum LR for cosine annealing')
    parser.add_argument('--num_print_epochs', type=int, default=10, help='number of printing epochs')

    # For rebuttal again 6zJP only
    parser.add_argument("--fix_qH", action='store_true', help='whether to fix qH to be non-trainable')

    args_eeg = parser.parse_args()

    run_dkl_lvmogp_eeg(args_eeg)





















