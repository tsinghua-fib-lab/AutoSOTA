import logging
import os
import pickle
from functools import partial
from multiprocessing import Pool

import hydra
import numpy as np
import torch
from dysts.analysis import gp_dim  # type: ignore
from tqdm import tqdm

from scaleformer.scaleformer.pipeline import PatchTSTPipeline
from scaleformer.utils import (
    compute_gp_dimension,
    get_eval_data_dict,
    log_on_main,
)

logger = logging.getLogger(__name__)
log = partial(log_on_main, logger=logger)


def _compute_gp_dims_worker(args):
    """Worker function to compute GP dimensions for a single system"""
    dyst_name, data, use_custom_method = args
    dimension_func = compute_gp_dimension if use_custom_method else gp_dim

    # Transpose arrays once
    completions_T = data["completions"].T
    groundtruth_T = data["processed_context"].T

    return dyst_name, {
        "groundtruth": dimension_func(groundtruth_T),
        "completions": dimension_func(completions_T),
    }


def get_gp_dims(
    completions_dict: dict[str, dict[str, np.ndarray]],
    use_custom_method: bool = False,
    n_jobs: int | None = None,
) -> dict[str, float]:
    """
    Compute GP dimensions for multiple systems using multiprocessing.

    Args:
        completions_dict: Dictionary containing completions and processed_context for each system
        use_custom_method: Whether to use custom GP dimension computation method
        n_jobs: Number of processes to use. If None, uses all available CPU cores

    Returns:
        Dictionary containing GP dimensions for groundtruth and completions for each system
    """
    # Validate all data upfront
    for dyst_name, data in completions_dict.items():
        if "completions" not in data or "processed_context" not in data:
            raise ValueError(f"Missing required data for {dyst_name}")

    # Prepare arguments for parallel processing
    worker_args = [
        (dyst_name, data, use_custom_method)
        for dyst_name, data in completions_dict.items()
    ]

    # Use multiprocessing to compute dimensions in parallel
    with Pool(processes=n_jobs) as pool:
        results = list(
            tqdm(
                pool.imap(_compute_gp_dims_worker, worker_args),
                total=len(worker_args),
                desc="Computing GP dimensions",
            )
        )

    # Convert results list back to dictionary
    return dict(results)


def get_model_completion(
    pipeline,
    context: np.ndarray,
    return_normalized_completions: bool = False,
    verbose: bool = True,
    **kwargs,
):
    # Prepare input tensor
    context_tensor = torch.from_numpy(context.T).float().to(pipeline.device)[None, ...]
    # Generate completions
    completions_output = pipeline.model.generate_completions(
        context_tensor,
        past_observed_mask=None,
        **kwargs,
    )

    if verbose:
        print(f"context_tensor shape: {context_tensor.shape}")
        print(f"completions output shape: {completions_output.completions.shape}")

    # Extract shapes and data
    patch_size = completions_output.completions.shape[-1]

    # Check for required outputs
    if any(
        x is None
        for x in [completions_output.mask, completions_output.patched_past_values]
    ):
        raise ValueError("Required completion outputs are None")

    # Process tensors to numpy arrays
    def process_tensor(tensor, reshape=True):
        if reshape:
            return (
                tensor.reshape(context_tensor.shape[0], context_tensor.shape[-1], -1)
                .detach()
                .cpu()
                .numpy()
                .transpose(0, 2, 1)
            )
        return tensor.detach().cpu().numpy()

    completions = process_tensor(completions_output.completions)
    processed_context = process_tensor(completions_output.patched_past_values)
    patch_mask = process_tensor(completions_output.mask, reshape=False)
    timestep_mask = np.repeat(patch_mask, repeats=patch_size, axis=2)

    # Denormalize if needed
    if not return_normalized_completions:
        if completions_output.loc is None or completions_output.scale is None:
            raise ValueError("Loc or scale is None")
        loc = completions_output.loc.detach().cpu().numpy()
        scale = completions_output.scale.detach().cpu().numpy()
        completions = completions * scale + loc
        processed_context = processed_context * scale + loc

    # Reshape for plotting
    processed_context = processed_context.squeeze(0).transpose(1, 0)
    completions = completions.squeeze(0).transpose(1, 0)
    timestep_mask = timestep_mask.squeeze(0)

    if verbose:
        print(f"processed context shape: {processed_context.shape}")
        print(f"completions shape: {completions.shape}")
        print(f"timestep mask shape: {timestep_mask.shape}")

    return completions, processed_context, timestep_mask


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg):
    start_time = 512  # to be extra sure we cut off transient

    test_data_dict = get_eval_data_dict(
        cfg.eval.data_paths_lst,
        num_subdirs=cfg.eval.num_subdirs,
        num_samples_per_subdir=cfg.eval.num_samples_per_subdir,
    )
    log(f"Number of combined test data subdirectories: {len(test_data_dict)}")

    # Load MLM checkpoint
    checkpoint_path = cfg.eval.checkpoint_path
    log(f"Using checkpoint: {checkpoint_path}")
    # torch_dtype = getattr(torch, cfg.eval.torch_dtype)
    # assert isinstance(torch_dtype, torch.dtype)
    model_pipeline = PatchTSTPipeline.from_pretrained(
        mode=cfg.eval.mode,
        pretrain_path=checkpoint_path,
        device_map=cfg.eval.device,
        # torch_dtype=torch_dtype,
    )

    completions_dict = {}
    for subdir_name, datasets in tqdm(
        list(test_data_dict.items())[: cfg.eval.num_subdirs],
        desc="Generating completions for subdirectories",
    ):
        log(f"Processing {len(datasets)} datasets in {subdir_name}")
        for file_dataset in datasets[: cfg.eval.num_samples_per_subdir]:
            filepath = file_dataset.iterable.path  # type: ignore
            sample_idx = int(os.path.basename(filepath).split("_")[0])
            system_name = f"{subdir_name}_pp{sample_idx}"
            coords, _ = zip(
                *[(coord["target"], coord["start"]) for coord in file_dataset]
            )
            coordinates = np.stack(coords)
            if coordinates.ndim > 2:  # if not one_dim_target:
                coordinates = coordinates.squeeze()

            completions, processed_context, timestep_mask = get_model_completion(
                model_pipeline,
                coordinates[:, start_time:],  # context
                return_normalized_completions=False,
                verbose=False,
            )
            completions_dict[system_name] = {
                "completions": completions,
                "processed_context": processed_context,
                "timestep_mask": timestep_mask,
            }

    gp_dims = get_gp_dims(completions_dict, n_jobs=cfg.eval.num_processes)

    metrics_save_dir = cfg.eval.metrics_save_dir
    metrics_fname = f"{cfg.eval.metrics_fname}.pkl"
    os.makedirs(metrics_save_dir, exist_ok=True)
    with open(os.path.join(metrics_save_dir, metrics_fname), "wb") as f:
        pickle.dump(gp_dims, f)

    if cfg.eval.save_completions:
        log(
            f"Saving completions to {metrics_save_dir}"
        )  # instead of cfg.eval.completions_save_dir
        with open(
            os.path.join(metrics_save_dir, f"{cfg.eval.metrics_fname}_completions.pkl"),
            "wb",
        ) as f:
            pickle.dump(completions_dict, f)


if __name__ == "__main__":
    main()
