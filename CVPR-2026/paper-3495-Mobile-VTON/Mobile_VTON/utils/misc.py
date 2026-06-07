import importlib
import os
import os.path as osp
import shutil
import sys
import random
import logging
from typing import List
import numpy as np
import torch
from PIL import Image
from omegaconf import OmegaConf
from typing import Dict
from accelerate.utils import set_seed
from diffusers.training_utils import EMAModel
import matplotlib.pyplot as plt

# Setup logging
logger = logging.getLogger(__name__)

# Setup working directory
WORK_DIR = osp.abspath(osp.join(osp.dirname(__file__), "../../.."))
logger.debug(f"Working directory: {WORK_DIR}")
if WORK_DIR not in sys.path:
    logger.warning(f"Working directory ({WORK_DIR}) is not in sys.path. Adding it.")
    sys.path.append(WORK_DIR)


def get_real_path(path):
    if osp.isabs(path):
        return path
    return osp.abspath(osp.join(WORK_DIR, path))

def snapshot_code(src_dir, tgt_dir, items_list):
    # ["configs", "hallo", "scripts", "accelerate_config.yaml"]
    for i in items_list:
        src_path = osp.join(src_dir, i)
        assert osp.exists(src_path), f"{src_path} does not exist."
        tgt_path = osp.join(tgt_dir, i)
        if osp.isdir(src_path):
            cmd = f"cp -r {src_path} {tgt_path}"
        else:
            cmd = f"cp {src_path} {tgt_path}"
        ret = os.system(cmd)
        assert ret == 0, f"Failed to execute: {cmd}"
    logger.info(f"code snapshot of {src_dir} is saved to {tgt_dir}")


def seed_everything(seed):
    """
    Seeds all random number generators to ensure reproducibility.

    Args:
        seed (int): The seed value to set for all random number generators.
    """
    set_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed % (2**32))
    random.seed(seed)


def init_output_dir(dir_list: List[str]):
    """
    Initialize the output directories.

    This function creates the directories specified in the `dir_list`. If a directory already exists, it does nothing.

    Args:
        dir_list (List[str]): List of directory paths to create.
    """
    for path in dir_list:
        os.makedirs(path, exist_ok=True)


def import_filename(filename):
    """
    Import a module from a given file location.

    Args:
        filename (str): The path to the file containing the module to be imported.

    Returns:
        module: The imported module.

    Raises:
        ImportError: If the module cannot be imported.

    Example:
        >>> imported_module = import_filename('path/to/your/module.py')
    """
    spec = importlib.util.spec_from_file_location("mymodule", filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def filter_non_none(dict_obj: Dict):
    """
    Filters out key-value pairs from the given dictionary where the value is None.

    Args:
        dict_obj (Dict): The dictionary to be filtered.

    Returns:
        Dict: The dictionary with key-value pairs removed where the value was None.

    This function creates a new dictionary containing only the key-value pairs from
    the original dictionary where the value is not None. It then clears the original
    dictionary and updates it with the filtered key-value pairs.
    """
    non_none_filter = {k: v for k, v in dict_obj.items() if v is not None}
    dict_obj.clear()
    dict_obj.update(non_none_filter)
    return dict_obj


def load_config(config_path: str) -> dict:
    """
    Loads the configuration file.

    Args:
        config_path (str): Path to the configuration file.

    Returns:
        dict: The configuration dictionary.
    """

    if config_path.endswith(".yaml"):
        return OmegaConf.load(config_path)
    if config_path.endswith(".py"):
        return import_filename(config_path).cfg
    raise ValueError("Unsupported format for config file")


def load_checkpoint(cfg, save_dir, accelerator, ema_model=None, ema_dir=None, ema_prefix=None, ema_model_cls=None):
    """
    Load the most recent checkpoint from the specified directory.

    This function loads the latest checkpoint from the `save_dir` if the `resume_from_checkpoint` parameter is set to "latest".
    If a specific checkpoint is provided in `resume_from_checkpoint`, it loads that checkpoint. If no checkpoint is found,
    it starts training from scratch.

    Args:
        cfg: The configuration object containing training parameters.
        save_dir (str): The directory where checkpoints are saved.
        accelerator: The accelerator object for distributed training.

    Returns:
        int: The global step at which to resume training.
    """
    if cfg.resume_from_checkpoint != "latest":
        resume_dir = get_real_path(cfg.resume_from_checkpoint)
    else:
        resume_dir = save_dir
    # Get the most recent checkpoint
    dirs = os.listdir(resume_dir)

    dirs = [d for d in dirs if d.startswith("checkpoint")]
    if len(dirs) > 0:
        dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
        path = dirs[-1]
        accelerator.load_state(os.path.join(resume_dir, path))
        accelerator.print(f"Resuming from checkpoint {path}")
        global_step = int(path.split("-")[1])
        if ema_model is not None:
            assert ema_dir is not None, "ema_dir must be provided if ema_model is not None"
            assert ema_prefix is not None, "ema_prefix must be provided if ema_model is not None"
            assert ema_model_cls is not None, "ema_model_cls must be provided if ema_model is not None"
            ema_save_dir = osp.join(ema_dir, f"{ema_prefix}-{global_step}-ema")
            if not osp.exists(ema_save_dir):
                accelerator.print(
                    f"Could not find EMA checkpoint under {ema_save_dir}. EMAModel will be initialized from scratch")
            else:
                load_ema_model = EMAModel.from_pretrained(ema_save_dir, ema_model_cls)
                ema_model.load_state_dict(load_ema_model.state_dict())
                ema_model.to(accelerator.device)
                accelerator.print(f"EMA model loaded from {ema_save_dir}")
                del load_ema_model
    else:
        accelerator.print(
            f"Could not find checkpoint under {resume_dir}, start training from scratch")
        global_step = 0

    return global_step, ema_model


def delete_additional_ckpt(base_path, num_keep):
    """
    Deletes additional checkpoint files in the given directory.

    Args:
        base_path (str): The path to the directory containing the checkpoint files.
        num_keep (int): The number of most recent checkpoint files to keep.

    Returns:
        None

    Raises:
        FileNotFoundError: If the base_path does not exist.

    Example:
        >>> delete_additional_ckpt('path/to/checkpoints', 1)
        # This will delete all but the most recent checkpoint file in 'path/to/checkpoints'.
    """
    dirs = []
    for d in os.listdir(base_path):
        if d.startswith("checkpoint-"):
            dirs.append(d)
    num_tot = len(dirs)
    print("delete_additional_ckpt", dirs)
    print("delete_additional_ckpt", num_tot)
    if num_tot <= num_keep:
        return
    # ensure ckpt is sorted and delete the ealier!
    del_dirs = sorted(dirs, key=lambda x: int(
        x.split("-")[-1]))[: num_tot - num_keep]
    print("delete_additional_ckpt", del_dirs)
    for d in del_dirs:
        path_to_dir = osp.join(base_path, d)
        if osp.exists(path_to_dir):
            shutil.rmtree(path_to_dir)


def save_checkpoint(model: torch.nn.Module, save_dir: str, prefix: str, ckpt_num: int, total_limit: int = -1, ema_model=None) -> None:
    """
    Save the model's state_dict to a checkpoint file.

    If `total_limit` is provided, this function will remove the oldest checkpoints
    until the total number of checkpoints is less than the specified limit.

    Args:
        model (nn.Module): The model whose state_dict is to be saved.
        save_dir (str): The directory where the checkpoint will be saved.
        prefix (str): The prefix for the checkpoint file name.
        ckpt_num (int): The checkpoint number to be saved.
        total_limit (int, optional): The maximum number of checkpoints to keep.
            Defaults to None, in which case no checkpoints will be removed.

    Raises:
        FileNotFoundError: If the save directory does not exist.
        ValueError: If the checkpoint number is negative.
        OSError: If there is an error saving the checkpoint.
    """

    if not osp.exists(save_dir):
        raise FileNotFoundError(
            f"The save directory {save_dir} does not exist.")

    if ckpt_num < 0:
        raise ValueError(f"Checkpoint number {ckpt_num} must be non-negative.")

    save_path = osp.join(save_dir, f"{prefix}-{ckpt_num}.pth")
    if ema_model is not None:
        ema_save_dir = osp.join(save_dir, f"{prefix}-{ckpt_num}-ema")

    if total_limit > 0:
        total_checkpoints = []
        total_removing_checkpoints = []

        checkpoints = os.listdir(save_dir)
        checkpoints = [d for d in checkpoints if d.startswith(prefix) and osp.isfile(osp.join(save_dir, d))]
        checkpoints = sorted(
            checkpoints, key=lambda x: int(x.split("-")[1].split(".")[0])
        )
        print("save_checkpoint", checkpoints)
        total_checkpoints.extend(checkpoints)

        if len(checkpoints) >= total_limit:
            num_to_remove = len(checkpoints) - total_limit + 1
            removing_checkpoints = checkpoints[0:num_to_remove]
            print("save_checkpoint", removing_checkpoints)
            total_removing_checkpoints.extend(removing_checkpoints)

        if ema_model is not None:
            ema_checkpoints = os.listdir(save_dir)
            ema_checkpoints = [d for d in ema_checkpoints if d.startswith(prefix) and osp.isdir(osp.join(save_dir, d))]
            ema_checkpoints = sorted(
                ema_checkpoints, key=lambda x: int(x.split("-")[1].split("-")[0])
            )
            print("save_checkpoint", ema_checkpoints)
            total_checkpoints.extend(ema_checkpoints)

            if len(ema_checkpoints) >= total_limit:
                num_to_remove = len(ema_checkpoints) - total_limit + 1
                removing_ema_checkpoints = ema_checkpoints[0:num_to_remove]
                print("save_checkpoint", removing_ema_checkpoints)
                total_removing_checkpoints.extend(removing_ema_checkpoints)

        if len(total_removing_checkpoints) > 0:
            print(
                f"{len(total_checkpoints)} checkpoints already exist, removing {len(total_removing_checkpoints)} checkpoints"
            )
            print(
                f"Removing checkpoints: {', '.join(total_removing_checkpoints)}"
            )

            for removing_checkpoint in total_removing_checkpoints:
                removing_checkpoint_path = osp.join(
                    save_dir, removing_checkpoint)
                try:
                    if osp.exists(removing_checkpoint_path):
                        if osp.isdir(removing_checkpoint_path):
                            shutil.rmtree(removing_checkpoint_path)
                        else:
                            os.remove(removing_checkpoint_path)
                except OSError as e:
                    print(
                        f"Error removing checkpoint {removing_checkpoint_path}: {e}")

    state_dict = model.state_dict()
    try:
        torch.save(state_dict, save_path)
        print(f"Checkpoint saved at {save_path}")
        if ema_model is not None:
            os.makedirs(ema_save_dir, exist_ok=True)
            ema_model.save_pretrained(ema_save_dir)
            print(f"EMA model saved at {ema_save_dir}")
    except OSError as e:
        raise OSError(f"Error saving checkpoint at {save_path}: {e}") from e


def move_final_checkpoint(save_dir, module_dir, prefix):
    """
    Move the final checkpoint file to the save directory.

    This function identifies the latest checkpoint file based on the given prefix and moves it to the specified save directory.

    Args:
        save_dir (str): The directory where the final checkpoint file should be saved.
        module_dir (str): The directory containing the checkpoint files.
        prefix (str): The prefix used to identify checkpoint files.

    Raises:
        ValueError: If no checkpoint files are found with the specified prefix.
    """
    # only file, not dir
    entries = os.listdir(module_dir)
    checkpoints = [entry for entry in entries if os.path.isfile(os.path.join(module_dir, entry))]
    checkpoints = [d for d in checkpoints if d.startswith(prefix)]
    checkpoints = sorted(
        checkpoints, key=lambda x: int(x.split("-")[1].split(".")[0])
    )
    shutil.copy2(os.path.join(
        module_dir, checkpoints[-1]), os.path.join(save_dir, prefix + '.pth'))


def make_grid(val_pil_images):
    n_rows = np.sqrt(len(val_pil_images)).astype(int)
    n_cols = np.ceil(len(val_pil_images) / n_rows).astype(int)
    row_list = []
    for r in range(n_rows):
        col_list = []
        for c in range(n_cols):
            idx = r * n_cols + c
            if idx >= len(val_pil_images):
                col_list.append(np.zeros_like(np.array(val_pil_images[0])))
            else:
                col_list.append(np.array(val_pil_images[idx]))
        row_list.append(np.concatenate(col_list, axis=1))
    val_image = np.concatenate(row_list, axis=0)
    val_image = Image.fromarray(val_image)
    return val_image


def display_image_safely(image_pil, safe_size=256):
    image_pil_safe = image_pil.copy()
    image_width, image_height = image_pil_safe.size
    if image_width > safe_size or image_height > safe_size:
        max_size = max(image_width, image_height)
        ratio = safe_size / max_size
        new_width = int(image_width * ratio)
        new_height = int(image_height * ratio)
        image_pil_safe = image_pil_safe.resize((new_width, new_height))
    logger.debug(f"Original size: {image_width}x{image_height}")
    logger.debug(f"Safe size: {image_pil_safe.size}")
    plt.figure()
    plt.imshow(image_pil_safe)
    plt.show()
