import os
from typing import Optional, Dict, Any, Literal, Tuple
import datetime
import json
import yaml
from pathlib import Path
import re


def create_experiment_dir(
    root_dir: Path,
    model_name: str,
    dataset_name: str,
    experiment_id: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    config_format: Literal['json', 'yaml'] = 'yaml',
    verbosity: int = 0,
    create_subdirs: bool = True,
) -> Dict[str, str]:
    """
    Creates a hierarchical directory structure for experiment results:
    root_dir/experiment_id/model_name[_target_key]/

    experiments/
    ├── experiment_id/
    │   ├── model_name_target/
    │   │   ├── config/
    │   │   ├── metrics/
    │   │   ├── models/
    │   │   └── logs/
    
    Args:
        root_dir: Base directory for all results
        model_name: Name of the model being used
        dataset_name: Name of the dataset being used
        experiment_id: Optional experiment ID. If None, generates one based on timestamp
        config: Optional configuration dictionary (not saved, just used for path construction)
        config_format: Format to save config file in ('json' or 'yaml')
        verbosity: Verbosity level for printing warnings
        create_subdirs: Whether to create subdirectories for metrics, models, and logs
        
    Returns:
        Dictionary containing paths to:
        - 'exp_dir': The experiment directory
        - 'metrics': Directory for metrics and results
        - 'models': Directory for model weights
        - 'logs': Directory for training logs
        - 'config_save_path': Path where config file should be saved
    """
    # Generate experiment ID if not provided
    if experiment_id is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_id = f"exp_{timestamp}"
        
    # Ensure model directory is unique WITHIN the experiment directory.
    # Keep experiment_id fixed; increment the model directory name instead.
    candidate_model_name = model_name
    existing_dir = os.path.join(root_dir, experiment_id, candidate_model_name)
    while os.path.exists(existing_dir):
        match = re.match(r"(.*)_(\d+)$", candidate_model_name)
        if match:
            base_name, num = match.groups()
            candidate_model_name = f"{base_name}_{int(num) + 1}"
        else:
            candidate_model_name = f"{candidate_model_name}_1"
        existing_dir = os.path.join(root_dir, experiment_id, candidate_model_name)
    
    # Create directory structure
    base_dir = os.path.join(root_dir, experiment_id, candidate_model_name)
    dirs = {
        'exp_dir': base_dir,
        'metrics': os.path.join(base_dir, 'metrics'),
        'models': os.path.join(base_dir, 'models'),
        'logs': os.path.join(base_dir, 'logs')
    }

    # Always create just the experiment directory; subdirs are optional
    try:
        os.makedirs(dirs['exp_dir'], exist_ok=True)
    except PermissionError:
        if verbosity > -1:
            print(f"Warning: Could not create directory {dirs['exp_dir']} due to permission error.")
        return None

    if create_subdirs:
        for key in ('metrics', 'models', 'logs'):
            try:
                os.makedirs(dirs[key], exist_ok=True)
            except PermissionError:
                if verbosity > -1:
                    print(f"Warning: Could not create directory {dirs[key]} due to permission error.")
                return None

        # Config directory under exp_dir only when creating subdirs
        config_dir = os.path.join(dirs['exp_dir'], 'config')
        os.makedirs(config_dir, exist_ok=True)
        dirs['config_save_path'] = os.path.join(config_dir, f'config.{config_format}')
    else:
        # Do not create metrics/models/logs/config here.
        # Provide a default config path under exp_dir but do not create the dir.
        dirs['config_save_path'] = os.path.join(dirs['exp_dir'], 'config', f'config.{config_format}')
    
    return dirs


def ensure_dir_exists(
    dir_path: str,
    raise_exception: bool = False
) -> bool:
    """
    Safely ensure a directory exists, handling permission errors gracefully.
    
    Args:
        dir_path: Path to the directory to create
        verbosity: Verbosity level for printing warnings (default: 0)
        
    Returns:
        bool: True if directory exists or was created successfully,
              False if creation failed due to permission error
    """
    # First check if directory exists
    if os.path.exists(dir_path):
        return True
        
    try:
        os.makedirs(dir_path, exist_ok=True)
        return True
    except PermissionError:
        if raise_exception:
            raise Exception(
                f"Permission error: could not create directory {dir_path}"
            )
        else:
            print(
                f"Warning: could not create directory {dir_path}...skipping."
            )
            return False
        


def smart_pickle(
    filepath: str, 
    obj: Any, 
    overwrite: bool = False
) -> None:
    """
    More robust pickling function.

    Args:
        path: full filepath to which to save
            pickle.
        obj: object to be pickled.
    Returns:
        None (pickles object to file).
    """
    import pickle
    
    if (filepath is not None) and (filepath != ""):
        try:
            file_exists = os.path.isfile(filepath)
            if file_exists and (not overwrite):
            # if file already exists, and we don't want to overwrite it,
            # append, e.g., '_1' to filename before .filetype suffix
                while file_exists:
                    dir = os.path.dirname(os.path.realpath(filepath))
                    filename = os.path.basename(filepath)
                    filename_pref_suf = filename.split('.')
                    filename_suffix = filename_pref_suf[-1]
                    filename_pref = filename_pref_suf[0]
                    filename_pref_parts = filename_pref.split('_')
                    filename_pref_no_last_thing = '_'.join(filename_pref_parts[:-1])
                    last_filename_thing = filename_pref_parts[-1]
                    
                    if last_filename_thing.isdigit():
                        new_version = str(int(last_filename_thing) + 1)
                        new_filename = filename_pref_no_last_thing + f'_{new_version}'
                    else:
                        new_filename = filename_pref + '_1'
                        
                    new_filename += f'.{filename_suffix}'
                    # print(new_filename)
                    filepath = os.path.join(dir, new_filename)
                    file_exists = os.path.isfile(filepath)

            else:  # file does not exist: give it a .pkl suffix
                if '.' not in filepath:
                    filepath = f"{filepath}.pkl"
                
            with open(filepath, "wb") as f:
                pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
                
        except pickle.UnpicklingError as e:
            # normal, somewhat expected
            pass
        except (AttributeError, EOFError, ImportError, IndexError) as e:
            # secondary errors
            print(e)
        except FileNotFoundError as e:
            print(e)
            print(f"FileNotFoundError: attempted path: {filepath}")
            print("File not saved!")
            return
        except Exception as e:
            # everything else, possibly fatal
            print(e)
            print(f"Attempted path: {filepath}")
            print("File not saved!")
            return
    else:
        print("No save path given; file not saved!")


def get_unique_path_with_suffix(base_path: Path) -> Path:
    """
    Returns a non-existing path by suffixing an integer when needed.
    Example: if 'foo' exists, returns 'foo_1', then 'foo_2', etc.
    """
    candidate = base_path
    if not candidate.exists():
        return candidate

    stem = base_path.name
    parent = base_path.parent
    suffix = 1
    while True:
        candidate = parent / f"{stem}_{suffix}"
        if not candidate.exists():
            return candidate
        suffix += 1


def get_time_hr_min_sec(
    t_1: float, 
    t_0: Optional[float] = None,
    return_str: bool = False
) -> Tuple[float, float]:
    """
    Calculates hours, minutes, and seconds 
    elapsed between 2 timepoints, or
    for one length of time.

    Args:
        t_1: 'end' timepoint or full
            time.
        t_0: 'start' timepoint. Leave
            'None' to calculate for a
            t_1 length of time.
        return_str: bool. If True, return a string
            of the form '1h, 2min, 3.4sec.'.
    Returns:
        2-tuple of floats: min and sec
        elapsed.
    """
    if t_0 is None:
        t = t_1
    else:
        t = t_1 - t_0
    t_hr, t_min, t_sec = t // 3600, t % 3600 // 60, t % 60
    if return_str:
        if t_hr > 0:
            return f'{t_hr:.0f}h, {t_min:.0f}min, {t_sec:.1f}sec.'
        else:
            return f'{t_min:.0f}min, {t_sec:.1f}sec.'
    else:
        return t_hr, t_min, t_sec

