import os
import h5py
import argparse

import torch
from torch import nn


def float_or_none(v):
    if v.lower() == "none":
        return None
    try:
        return float(v)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{v} is not a float or 'None'")


def rp_h5file(file_pth, device="cpu"):
    # read and prepare data in torch tensors
    h5_dict = h5py.File(file_pth, "r")

    for key_name in ["all_X", "train", "test"]:
        assert key_name in h5_dict.keys(), f"{key_name} not in h5_dict.keys()={h5_dict.keys()}."

    all_X = torch.as_tensor(h5_dict['all_X'][:], dtype=torch.get_default_dtype(), device=device)  # [..., N, D_X]
    train_Y = torch.as_tensor(h5_dict['train']['train_Y'][:], dtype=torch.get_default_dtype(), device=device)  # [..., N, P]
    train_mask = torch.as_tensor(h5_dict['train']['train_mask'][:], dtype=torch.bool, device=device)  # [..., N, P]
    test_Y = torch.as_tensor(h5_dict['test']['test_Y'][:], dtype=torch.get_default_dtype(), device=device)  # [..., N, P]
    test_mask = torch.as_tensor(h5_dict['test']['test_mask'][:], dtype=torch.bool, device=device)  # [..., N, P]

    data_dict = {
        "all_X": all_X,
        "train_Y": train_Y,
        "train_mask": train_mask,
        "test_Y": test_Y,
        "test_mask": test_mask,
    }

    if "stats" in h5_dict.keys():
        means = torch.as_tensor(h5_dict['stats']['mean'][:], dtype=torch.get_default_dtype(), device=device)  # [..., P]
        stds = torch.as_tensor(h5_dict['stats']['std'][:], dtype=torch.get_default_dtype(), device=device)  # [..., P]

        data_dict["means"] = means
        data_dict["stds"] = stds

    if "rmnist_oe_spatial_coords" in h5_dict.keys(): # specific to rmnist output extrapolation
        train_sc = torch.as_tensor(h5_dict['rmnist_oe_spatial_coords']['train_sc'][:], dtype=torch.get_default_dtype(), device=device)  # [..., N, 2]
        test_sc = torch.as_tensor(h5_dict['rmnist_oe_spatial_coords']['test_sc'][:], dtype=torch.get_default_dtype(), device=device)  # [..., N, 2]
        data_dict["train_sc"] = train_sc
        data_dict["test_sc"] = test_sc

    elif "cm_oe_spatial_coords" in h5_dict.keys():
        train_sc = torch.as_tensor(h5_dict['cm_oe_spatial_coords']['train_sc'][:], dtype=torch.get_default_dtype(), device=device)  # [..., N, 2]
        test_sc = torch.as_tensor(h5_dict['cm_oe_spatial_coords']['test_sc'][:], dtype=torch.get_default_dtype(), device=device)  # [..., N, 2]
        data_dict["train_sc"] = train_sc
        data_dict["test_sc"] = test_sc

    return data_dict


def print_model_summary(model: nn.Module, exp_result_folder: str):
    # Save model summary to a text file in a given folder
    with open(f'{exp_result_folder}/model_summary.txt', 'w') as f:
        print(model, file=f)

        for name, param in model.named_parameters():
            print(f"{name}: {param.shape}, requires_grad={param.requires_grad}\n", file=f)

        num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total trainable parameters: {num_trainable_params}\n", file=f)

        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters (including non-trainable): {total_params}\n", file=f)


def check_and_filter_args(args):
    """
    Check essential arguments & Filter out redundant arguments according to 'model' type (neural_network_type and qH_type)
    """
    if args.neural_network_type == "FCNet":
        assert hasattr(args, "hidden_dims")
        assert hasattr(args, "out_dim")
        if hasattr(args, "num_blocks"):
            del args.num_blocks
    elif args.neural_network_type == "ResNet":
        assert hasattr(args, "num_blocks")
        if hasattr(args, "out_dim"):
            del args.out_dim
        if hasattr(args, "hidden_dims"):
            del args.hidden_dims
    elif args.neural_network_type == "Identity":
        if hasattr(args, "num_blocks"):
            del args.num_blocks
        if hasattr(args, "hidden_dims"):
            del args.hidden_dims
        if hasattr(args, "out_dim"):
            del args.out_dim
        if hasattr(args, "spectral_norm"):
            del args.spectral_norm
        if hasattr(args, "sn_ub"):
            del args.sn_ub
    else:
        raise NotImplementedError

    if args.qH_type == "Gaussian":
        assert hasattr(args, "qH_mean_field")
        assert hasattr(args, "beta_h")
    elif args.qH_type == "Delta":
        if hasattr(args, "qH_mean_field"):
            del args.qH_mean_field
        if hasattr(args, "beta_h"):
            del args.beta_h
        if hasattr(args, "num_samples"):
            del args.num_samples
    else:
        raise NotImplementedError

    if args.qU_type == "standard":
        if hasattr(args, "natural_lr"):
            del args.natural_lr
    else:  # natural or tril-natural
        assert hasattr(args, "natural_lr")
    return args

def pca_reduce(X: torch.Tensor, k: int = 2) -> torch.Tensor:
    """
    Perform PCA on X and reduce to k dimensions.
    
    Args:
        X: [..., N, D] input tensor.
        k: target dimension (default=2).
        
    Returns:
        [..., N, k] tensor of PCA-reduced features.
    """
    if X.ndim < 2:
        raise ValueError("X must be at least 2D with trailing dimensions [N, D].")
    N, D = X.size(-2), X.size(-1)
    if N < 2:
        raise ValueError("Need at least 2 samples to compute covariance.")
    if k > D:
        raise ValueError(f"k={k} cannot exceed feature dimension D={D}.")

    # Center along sample dim
    Xc = X - X.mean(dim=-2, keepdim=True)  # [..., N, D]

    # covariance: [..., D, D] = (Xc^T @ Xc) / (N-1)
    cov = Xc.mT @ Xc / (N - 1) # [..., D, D]

    # Eigen decomposition ascending eigenvalues
    eigvals, eigvecs = torch.linalg.eigh(cov)  #  [..., D], [..., D, D]

    # Take top-k eigenvectors (last k columns corresponding to largest eigenvalues)
    Vk = eigvecs[..., :, -k:]  # [..., D, k]

    # Project: [..., N, k] = Xc @ Vk
    X_reduced = Xc @ Vk  # [..., N, k]

    return X_reduced

def wrap_func_by_batch(
        model: nn.Module, func_args: dict, name: str, 
        input_batch_size: int = 64, output_batch_size: int = 32, device: str = "cpu"
    ):
    """
    Wrap predict function to handle mini-batch processing across inputs and outputs. 

    Used in dkl_lvmogp_base, lmc_base, ind_svgp_base, oilmm_base models.
    """
    ### Check Arguments & Obtain Chunks ###
    if name in ["dkl_lvmogp_base", "gs_lvmogp_base"]:
        args_keys = ["x_star", "output_idx", "num_samples", "noiseless"]
        for key in args_keys:
           if key not in func_args:
               raise ValueError(f"Missing argument '{key}' for using predict function in {name}.")
    
    elif name in ["lmc_base", "ind_svgp_base", "oilmm_base", "graphical_mogp_base"]:
        args_keys = ["x_star", "output_idx", "noiseless"]
        for key in args_keys:
           if key not in func_args:
               raise ValueError(f"Missing argument '{key}' for using predict function in {name}.")

    else:
        raise NotImplementedError(f"wrap_func_by_batch not implemented for model '{name}'.")
    
    input_chunks = torch.split(func_args["x_star"], input_batch_size, dim=-2)
    output_chunks = torch.split(func_args["output_idx"], output_batch_size, dim=-1)

    ### Wrap Function ###
    list_output_means, list_output_vars = [], []  # mini-batch across outputs
    for output_chunk in output_chunks:
        list_input_means, list_input_vars = [], []  # mini-batch across inputs
        for input_chunk in input_chunks:
            if name in ["dkl_lvmogp_base", "gs_lvmogp_base"]:
                tmp_qy_means, tmp_qy_vars = model.predict(
                    input_chunk, output_chunk, func_args["num_samples"], device, func_args["noiseless"]
                )
            elif name in ["lmc_base", "ind_svgp_base", "oilmm_base", "graphical_mogp_base"]:
                tmp_qy_means, tmp_qy_vars = model.predict(
                    input_chunk, output_chunk, device, func_args["noiseless"]
                )
            list_input_means.append(tmp_qy_means)
            list_input_vars.append(tmp_qy_vars)
        # Cat mini-batch across inputs
        chunk_output_means = torch.cat(list_input_means, dim=-2)  # [..., n_star, P_chunk]
        chunk_output_vars = torch.cat(list_input_vars, dim=-2)    # [..., n_star, P_chunk]
        list_output_means.append(chunk_output_means)
        list_output_vars.append(chunk_output_vars)
    # Cat mini-batch across outputs
    final_means = torch.cat(list_output_means, dim=-1)  # [..., n_star, P]
    final_vars = torch.cat(list_output_vars, dim=-1)    # [..., n_star, P]

    return final_means, final_vars

def read_config_file(filepath):
    """
    Read a configs.txt file and return a dictionary of key-value pairs.
    Supports types: int, float, bool, None, and str.
    """
    config = {}
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            # Skip empty lines or comments
            if not line or line.startswith("#"):
                continue
            # Each line like key=value;
            if not line.endswith(";"):
                raise ValueError(f"Line does not end with ';': {line}")
            key, value = line[:-1].split("=", 1)  # remove trailing ';'
            key = key.strip()
            value = value.strip()

            # Type conversion
            if value.lower() in {"true", "false"}:
                value = value.lower() == "true"
            elif value.lower() == "none":
                value = None
            else:
                # Try int, then float
                try:
                    value = int(value)
                except ValueError:
                    try:
                        value = float(value)
                    except ValueError:
                        pass  # leave as string

            config[key] = value
    return config

def find_best_seed(result_folder):
    """
    Example:
        result_folder = "./results/eeg/dkl_lvmogp/ResNet_Gaussian_with_sn=0.005"
        which contains subfolders like:
            1121_17_16_00, 1121_17_22_21, 1121_17_28_41, ...
        for each subfolder, there is a metrics.txt file, like:
            Total training time is: 24.623374223709106
            Train mse is: 0.058660600458089865
            Train nll is: 1.5039918364142537
            Test mse is: 0.11486745404933675
            Test nll is: 0.7064689255273426
        This function finds the subfolder with the lowest Test nll and returns its name.
    Return: subfolder name
    """
    best_nll = float('inf')
    best_subfolder = None

    if not os.path.exists(result_folder):
        print(f"Directory {result_folder} does not exist.")
        return None

    subfolders = [
        f for f in os.listdir(result_folder) if os.path.isdir(os.path.join(result_folder, f))
    ]

    for subfolder in subfolders:
        subfolder_path = os.path.join(result_folder, subfolder)
        metrics_file = os.path.join(subfolder_path, "metrics.txt")

        assert os.path.exists(metrics_file)
        with open(metrics_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if "Test nll is:" in line:
                    parts = line.strip().split(":")
                    current_nll = float(parts[1].strip())

                    if current_nll < best_nll:
                        best_nll = current_nll
                        best_subfolder = subfolder

                    break # skip out of this file after the Test nll line being found

    if best_subfolder:
        print(f"Found best seed: {best_subfolder} with Test NLL: {best_nll}")
    else:
        print("No valid metrics found.")

    return best_subfolder

def get_sin_rebuttal_dataset(N_input = 128):
    # Generate synthetic sinusoidal dataset with 2 outputs.
    # N_input: number of input points (both train and test)

    all_X = torch.linspace(0, 2 * torch.pi, N_input).unsqueeze(-1)  # [N_input, 1]

    # both outputs are missing or present for a particular input
    # missing rate 50% if N_input is an even number
    assert N_input // 2 * 2 == N_input, "N_input should be an even number."
    train_pattern = ((torch.arange(N_input) + 1) % 2).to(all_X.dtype)  # [1,0,1,0,...], of length N_input
    test_pattern = 1 - train_pattern  # [0,1,0,1,...], of length N_input
    train_mask = train_pattern.unsqueeze(-1).repeat(1, 2)  # [N_input, 2]
    test_mask = test_pattern.unsqueeze(-1).repeat(1, 2)  # [N_input, 2]

    all_Y = torch.cat(
        [torch.sin(all_X), torch.sin(torch.pi + all_X)], dim=-1
    )  # [N_input, 2]

    # random noise
    all_Y = all_Y + 0.01 * torch.randn_like(all_Y)

    train_Y = all_Y * train_mask.to(all_Y.dtype)  # [N_input, 2], with missing values set to 0
    test_Y = all_Y * test_mask.to(all_Y.dtype)  # [N_input, 2]

    translated_input = (all_X / torch.pi) - 1 # translate to [-1, 1], of shape [N_input, 1]

    data_dict = {
        "all_X": translated_input, "train_Y": train_Y, "train_mask": train_mask, "test_Y": test_Y, "test_mask": test_mask
    }

    return data_dict
