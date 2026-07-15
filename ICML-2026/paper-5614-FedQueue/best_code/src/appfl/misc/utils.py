import os
import re
import ast
import sys
import copy
import stat
import yaml
import torch
import random
import string
import logging
import pathlib
import importlib
import numpy as np
import pickle as pkl
import os.path as osp
import importlib.util
import torch.nn as nn
from omegaconf import DictConfig
from .deprecation import deprecated
from typing import Any, Optional, Union, Tuple, List, Dict


def _ensure_secure_dir(path: pathlib.Path) -> None:
    """Create `path` with mode 0o700 if missing, then verify it is a real
    directory (not a symlink), owned by the current user, and has mode
    exactly 0o700. Raises PermissionError if the verification fails."""
    try:
        path.mkdir(mode=0o700, exist_ok=False)
    except FileExistsError:
        pass
    os.chmod(path, 0o700)
    st = os.lstat(path)
    if not stat.S_ISDIR(st.st_mode):
        raise PermissionError(f"{path} is not a directory (or is a symlink)")
    if hasattr(os, "getuid") and st.st_uid != os.getuid():
        raise PermissionError(
            f"{path} is owned by uid {st.st_uid}, expected {os.getuid()}"
        )
    mode_bits = stat.S_IMODE(st.st_mode)
    if mode_bits & (stat.S_IRWXG | stat.S_IRWXO):
        raise PermissionError(
            f"{path} has insecure mode {oct(mode_bits)}; expected 0o700"
        )


def secure_appfl_dir(*parts: str) -> str:
    """Return a path under the user's APPFL working tree, creating any
    missing components with mode 0o700.

    Tries `$HOME/.appfl/<parts...>` first. If the home directory is
    read-only (common in containers), falls back to
    `/tmp/.appfl-<uid>/<parts...>` — the uid suffix prevents a co-tenant
    from pre-creating the directory and trapping the operator's writes.

    Every directory from the `.appfl(-uid)` root down to the leaf is
    created (or verified) with mode 0o700 and owned by the current user;
    symlinked components are refused.
    """
    if hasattr(os, "getuid"):
        fallback_root = pathlib.Path("/tmp") / f".appfl-{os.getuid()}"
    else:
        fallback_root = pathlib.Path("/tmp") / ".appfl"
    candidates = [pathlib.Path.home() / ".appfl", fallback_root]

    last_err: Optional[Exception] = None
    for root in candidates:
        try:
            _ensure_secure_dir(root)
            current = root
            for part in parts:
                current = current / part
                _ensure_secure_dir(current)
            return str(current)
        except OSError as e:
            last_err = e
            continue
    assert last_err is not None
    raise last_err


@deprecated(silent=True)
def get_appfl_algorithm(
    algorithm_name: str,
    args: Union[Tuple, List],
    kwargs: Dict,
):
    try:
        appfl_module = importlib.import_module("appfl.algorithm")
        AlgorithmClass = getattr(appfl_module, algorithm_name)
        algorithm = AlgorithmClass(*args, **kwargs)
        return algorithm
    except AttributeError:
        raise ValueError(f"Invalid algorithm name: {algorithm_name}")


def get_proxystore_connector(
    connector_name: str,
    connector_args: Dict[str, Any],
):
    assert connector_name in ["RedisConnector", "FileConnector", "EndpointConnector"], (
        f"Invalid connector name: {connector_name}, only RedisConnector, FileConnector, and EndpointConnector are supported"
    )
    if connector_name == "RedisConnector":
        from proxystore.connectors.redis import RedisConnector

        connector = RedisConnector(**connector_args)
    elif connector_name == "FileConnector":
        from proxystore.connectors.file import FileConnector

        connector = FileConnector(**connector_args)
    elif connector_name == "EndpointConnector":
        from proxystore.connectors.endpoint import EndpointConnector

        connector = EndpointConnector(**connector_args)
    return connector


def get_appfl_authenticator(
    authenticator_name: str,
    authenticator_args: Dict[str, Any],
):
    try:
        appfl_module = importlib.import_module("appfl.login_manager")
        AuthenticatorClass = getattr(appfl_module, authenticator_name)
        authenticator = AuthenticatorClass(**authenticator_args)
        return authenticator
    except AttributeError:
        raise ValueError(f"Invalid authenticator name: {authenticator_name}")


def get_appfl_aggregator(
    aggregator_name: str,
    model: Optional[Any],
    aggregator_config: DictConfig,
    logger: Optional[Any] = None,
):
    try:
        appfl_module = importlib.import_module("appfl.algorithm.aggregator")
        AggregatorClass = getattr(appfl_module, aggregator_name)
        aggregator = AggregatorClass(model, aggregator_config, logger)
        return aggregator
    except AttributeError:
        raise ValueError(f"Invalid aggregator name: {aggregator_name}")


def get_appfl_scheduler(
    scheduler_name: str,
    scheduler_config: DictConfig,
    aggregator: Optional[Any] = None,
    logger: Optional[Any] = None,
):
    try:
        appfl_module = importlib.import_module("appfl.algorithm.scheduler")
        SchedulerClass = getattr(appfl_module, scheduler_name)
        scheduler = SchedulerClass(scheduler_config, aggregator, logger)
        return scheduler
    except AttributeError:
        raise ValueError(f"Invalid scheduler name: {scheduler_name}")


def get_appfl_compressor(compressor_name: str, compressor_config: DictConfig):
    try:
        appfl_module = importlib.import_module("appfl.compressor")
        CompressorClass = getattr(appfl_module, compressor_name)
        compressor = CompressorClass(compressor_config)
        return compressor
    except AttributeError:
        raise ValueError(f"Invalid compressor name: {compressor_name}")


def get_torch_optimizer(optimizer_name, model_parameters, **kwargs):
    try:
        optim_module = importlib.import_module("torch.optim")
        OptimizerClass = getattr(optim_module, optimizer_name)
        optimizer = OptimizerClass(model_parameters, **kwargs)
        return optimizer
    except AttributeError:
        raise ValueError(f"Invalid optimizer name: {optimizer_name}")


def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return "".join(random.choice(chars) for _ in range(size))


def get_last_class_name(file_path):
    with open(file_path) as file:
        file_content = file.read()

    # Parse the file content
    tree = ast.parse(file_content)

    # Get all class definitions
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]

    # Return the name of the last class if there are any
    if classes:
        return classes[-1].name
    else:
        return None


def get_last_function_name(file_path):
    with open(file_path) as file:
        file_content = file.read()

    # Parse the file content
    tree = ast.parse(file_content)

    # Get all function definitions
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]

    # Return the name of the last function if there are any
    if functions:
        return functions[-1].name
    else:
        return None


def create_instance_from_file(file_path, class_name=None, *args, **kwargs):
    """
    Creates an instance of a class from a given file path.

    :param file_path: The file path where the class is defined.
    :param class_name: The name of the class to be instantiated.
    :param args: Positional arguments to be passed to the class constructor.
    :param kwargs: Keyword arguments to be passed to the class constructor.
    :return: An instance of the specified class, or None if creation fails.
    """
    # Read the last class name if not provided
    if class_name is None:
        class_name = get_last_class_name(file_path)

    # Normalize the file path
    file_path = os.path.abspath(file_path)

    # Check if the file exists
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # Extract module name and directory from the file path
    module_dir, module_file = os.path.split(file_path)
    module_name, _ = os.path.splitext(module_file)

    # Add module directory to sys.path
    if module_dir not in sys.path:
        sys.path.append(module_dir)

    # Load the module
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Get the class and create an instance
    cls = getattr(module, class_name)
    instance = cls(*args, **kwargs)

    return instance


def get_function_from_file(file_path, function_name=None):
    """
    Gets a function from a given file path.

    :param file_path: The file path where the function is defined.
    :param function_name: The name of the function to be retrieved.
    :return: The function object, or None if retrieval fails.
    """
    try:
        # Read the last function name if not provided
        if function_name is None:
            function_name = get_last_function_name(file_path)

        # Normalize the file path
        file_path = os.path.abspath(file_path)

        # Check if the file exists
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        # Extract module name and directory from the file path
        module_dir, module_file = os.path.split(file_path)
        module_name, _ = os.path.splitext(module_file)

        # Add module directory to sys.path
        if module_dir not in sys.path:
            sys.path.append(module_dir)

        # Load the module
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Get the function
        function = getattr(module, function_name)

        return function

    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def run_function_from_file(file_path, function_name=None, *args, **kwargs):
    """
    Runs a function from a given file path.

    :param file_path: The file path where the function is defined.
    :param function_name: The name of the function to be executed.
    :param args: Positional arguments to be passed to the function.
    :param kwargs: Keyword arguments to be passed to the function.
    :return: The result of the function execution, or None if execution fails.
    """
    try:
        # Read the last function name if not provided
        if function_name is None:
            function_name = get_last_function_name(file_path)

        # Normalize the file path
        file_path = os.path.abspath(file_path)

        # Check if the file exists
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        # Extract module name and directory from the file path
        module_dir, module_file = os.path.split(file_path)
        module_name, _ = os.path.splitext(module_file)

        # Add module directory to sys.path
        if module_dir not in sys.path:
            sys.path.append(module_dir)

        # Load the module
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Get the function and run it
        function = getattr(module, function_name)
        result = function(*args, **kwargs)

        return result

    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def create_instance_from_file_source(source, class_name=None, *args, **kwargs):
    """
    Creates an instance of a class from a given source code.

    :param source: The source code where the class is defined.
    :param class_name: The name of the class to be instantiated.
    :param args: Positional arguments to be passed to the class constructor
    """
    # Create a temporary file to store the source code
    dirname = secure_appfl_dir("tmp")
    file_path = osp.join(dirname, f"{id_generator()}.py")
    with open(file_path, "w") as file:
        file.write(source)

    # Create an instance from the temporary file
    instance = create_instance_from_file(file_path, class_name, *args, **kwargs)

    # Remove the temporary file
    os.remove(file_path)

    return instance


def get_function_from_file_source(source, function_name=None):
    """
    Gets a function from a given source code.

    :param source: The source code where the function is defined.
    :param function_name: The name of the function to be retrieved.
    :return: The function object, or None if retrieval fails.
    """
    # Create a temporary file to store the source code
    dirname = secure_appfl_dir("tmp")
    file_path = osp.join(dirname, f"{id_generator()}.py")
    with open(file_path, "w") as file:
        file.write(source)

    # Get the function from the temporary file
    function = get_function_from_file(file_path, function_name)

    # Remove the temporary file
    os.remove(file_path)

    return function


def run_function_from_file_source(source, function_name=None, *args, **kwargs):
    """
    Runs a function from a given source code.

    :param source: The source code where the function is defined.
    :param function_name: The name of the function to be executed.
    :param args: Positional arguments to be passed to the function.
    :param kwargs: Keyword arguments to be passed to the function.
    :return: The result of the function execution, or None if execution fails.
    """
    function = get_function_from_file_source(source, function_name)
    if function is None:
        return None
    result = function(*args, **kwargs)
    return result


def get_unique_filename(
    dirname: str,
    filename: str,
):
    """
    Create the directory (if needed) and get a unique filename by appending a number to the filename.
    :param dirname: The directory where the file is located.
    :param filename: The original filename.
    :return dirname, unique_filename: The directory and the unique filename.
    """
    if not osp.exists(dirname):
        pathlib.Path(dirname).mkdir(parents=True, exist_ok=True)
    unique = 1
    unique_filename = filename
    filename_base, ext = osp.splitext(filename)
    while pathlib.Path(osp.join(dirname, unique_filename)).exists():
        unique_filename = f"{filename_base}_{unique}{ext}"
        unique += 1
    return dirname, unique_filename


def load_data_from_file(file_path: str, to_device=None):
    """
    Read data from file using the corresponding readers.
    For uncompressed model weights of PyTorch models, the weights are stored in a dictionary in the `pt` or `pth` file.
    For compressed model weights of PyTorch models, the weights are stored as bytes in the `pkl` file.
    """
    TORCH_EXT = [".pt", ".pth"]
    PICKLE_EXT = [".pkl"]
    file_ext = osp.splitext(osp.basename(file_path))[-1]
    if file_ext in TORCH_EXT:
        results = torch.load(file_path, map_location=to_device)
    elif file_ext in PICKLE_EXT:
        with open(file_path, "rb") as fi:
            results = pkl.load(fi)
    else:
        raise RuntimeError("File extension %s is not supported" % file_ext)
    return results


def deserialize_yaml(
    stream,
    trusted: bool = False,
    warning_message: str = None,
):
    """
    Deserialize a YAML object to a string.
    :param stream: The YAML object to be serialized.
    :param trusted: Whether the YAML object is trusted.
    :param warning_message: The warning message to be displayed if the YAML object is not trusted.
    :return: The serialized YAML object as a string.
    """
    try:
        return yaml.safe_load(stream)
    except yaml.YAMLError as e:
        if trusted:
            return yaml.load(stream, Loader=yaml.UnsafeLoader)
        else:
            if warning_message is not None:
                raise ValueError(warning_message)
            else:
                raise ValueError(
                    f"An error occurred: {e}, you may need to use the `trusted` flag to load the YAML object"
                )


def dump_data_to_file(obj, file_path: str):
    """
    Write data to file using the corresponding readers
    """
    TORCH_EXT = [".pt", ".pth"]
    PICKLE_EXT = [".pkl"]
    file_ext = osp.splitext(osp.basename(file_path))[-1]
    if file_ext in TORCH_EXT:
        torch.save(obj, file_path)
    elif file_ext in PICKLE_EXT:
        with open(file_path, "wb") as f:
            pkl.dump(obj, f)
    else:
        raise RuntimeError("File extension %s is not supported" % file_ext)
    return True


def save_partial_model_iteration(t, model, cfg: DictConfig, client_id=None):
    # This function saves the model weights (instead of the entire model).
    # If personalization is enabled, only the shared layer weights will be saved for the server.
    dir = cfg.save_model_dirname
    if not os.path.isdir(dir):
        os.mkdir(dir)
    if client_id is not None:
        if not os.path.isdir(dir + "/client_%d" % client_id):
            try:
                os.mkdir(dir + "/client_%d" % client_id)
            except:  # noqa: E722
                pass

    file_ext = ".pt"
    if client_id is None:
        file = dir + f"/{cfg.save_model_filename}_Round_{t}{file_ext}"
    else:
        file = (
            dir
            + "/client_%d" % client_id
            + f"/{cfg.save_model_filename}_Round_{t}{file_ext}"
        )
    uniq = 1
    while os.path.exists(file):
        file = dir + "/%s_Round_%s_%d%s" % (cfg.save_model_filename, t, uniq, file_ext)
        uniq += 1

    state_dict = copy.deepcopy(model.state_dict())
    if client_id is None:
        if cfg.personalization:
            keys = [key for key, _ in enumerate(model.named_parameters())]
            for key in keys:
                if key in cfg.p_layers:
                    _ = state_dict.pop(key)

    torch.save(state_dict, file)


def model_parameters_clip_factor(model, pre_update_params, C, norm_type=1):
    """
    Compute the norm of all the parameters of a model and return that C / norm
    if norm is below the threshold C, otherwise return 1.
    :model: The PyTorch model
    :C: The threshold value
    :param norm_type: The type of norm to compute (default is 2, which is the L2 norm)
    """
    total_norm = 0.0
    for param_old, param_new in zip(pre_update_params, list(model.parameters())):
        param_norm = torch.norm(param_old - param_new, p=norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1.0 / norm_type)
    return (C / total_norm) if total_norm > C else 1


def scale_update(model, pre_update_params, scale=1.0):
    """
    Scale update of <<updated>> model.
    :model: <<updated>> model
    :pre_update_params: list of <<pre-update>> model parameters
    :scale: Scaling factor
    """
    for param_old, param_new in zip(pre_update_params, list(model.parameters())):
        param_new = param_old + scale * (param_new - param_old)


def parse_device_str(devices_str: str):
    """
    Parse a string like `cpu`, `cuda:0`, `cuda:0,cuda:1`. Raise `ValueError` if invalid devices or out-of-range indices.
    :param devices_str: Device config provided by user
    :return: `device_config` device configuration dictionary
    ```python
        {
            "device_type": "cpu" | "gpu-single" | "gpu-multi",
            "device_ids": [list_of_ints_if_applicable]
        }
    ```
    :return: `xy_device`: str for the main device to place input/output tensors, e.g. `cuda:0` or `cpu`.
    """
    devices = [d.strip().lower() for d in devices_str.split(",")]

    # CASE 1: single device
    # e.g. "cuda" or "cpu"
    if len(devices) == 1:
        dev = devices[0]
        if dev == "cpu":
            return ({"device_type": "cpu", "device_ids": []}, "cpu")
        elif dev == "cuda":
            return ({"device_type": "gpu-single", "device_ids": []}, "cuda")
        elif dev.startswith("cuda:"):
            match = re.match(r"cuda:(\d+)$", dev)
            if not match:
                raise ValueError(
                    f"Invalid device format: '{dev}'. Expected 'cuda:<index>' or 'cpu'"
                )
            index = int(match.group(1))
            if index < 0 or index >= torch.cuda.device_count():
                raise ValueError(
                    f"Requested {dev}, but only {torch.cuda.device_count()} GPUs available."
                )
            return ({"device_type": "gpu-single", "device_ids": [index]}, dev)
        else:
            raise ValueError(
                f"Unsupported device string: '{dev}'. Use 'cpu' or 'cuda:<index>'."
            )

    # CASE 2: multiple devices
    # e.g. "cuda:0,cuda:1"
    device_ids = []
    for d in devices:
        if d == "cpu":
            raise ValueError(
                "Cannot mix 'cpu' with other devices in multi-device usage."
            )
        match = re.match(r"cuda:(\d+)$", d)
        if not match:
            raise ValueError(f"Invalid device format: '{d}'. Expected 'cuda:<index>'.")
        index = int(match.group(1))
        if index < 0 or index >= torch.cuda.device_count():
            raise ValueError(
                f"Requested {d}, but only {torch.cuda.device_count()} GPUs available."
            )
        device_ids.append(index)

    device_ids = list(set(device_ids))
    device_ids.sort()
    if not device_ids:
        raise ValueError("No valid CUDA devices parsed from string.")

    # For multi-GPU, use the first in device_ids as primary
    first_dev = f"cuda:{device_ids[0]}"
    return ({"device_type": "gpu-multi", "device_ids": device_ids}, first_dev)


def apply_model_device(model, config: dict, xy_device: str):
    """
    This function extends pytorch's `model.to()` functionality, which applies
    the model to a device given the configuration and return the updated model.
    :param: `model` current nn.Module
    :param: `config` config returned from parse_device_str
    :param: `xy_device` main device string (e.g., "cuda:0" or "cpu")
    :return: updated model moved to the device
    """
    device_type = config["device_type"]

    if device_type == "cpu":
        # Single CPU
        model.to("cpu")
        # The model is now on CPU. (xy_device is also "cpu".)
        return model

    elif device_type == "gpu-single":
        # Single GPU
        if len(config["device_ids"]) == 0:
            # device is `cuda` without index
            model.to(xy_device)
        else:
            device_id = config["device_ids"][0]
            d = torch.device(f"cuda:{device_id}")
            model.to(d)
        return model

    elif device_type == "gpu-multi":
        # Wrap in DataParallel
        model = nn.DataParallel(model, device_ids=config["device_ids"])
        # Move base model to the first device
        first_dev_id = config["device_ids"][0]
        model.to(torch.device(f"cuda:{first_dev_id}"))
        return model

    else:
        raise ValueError(f"Unknown device_type: {device_type}")


@deprecated(silent=True)
def set_seed(seed=233):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@deprecated(silent=True)
def load_source_file(file_path):
    with open(file_path) as fi:
        source = fi.read()
    return source


@deprecated(silent=True)
def compute_gradient(original_model, trained_model):
    """Compute the difference (i.e. gradient) between the original model and the trained model"""
    list_named_parameters = []
    for name, _ in trained_model.named_parameters():
        list_named_parameters.append(name)
    local_gradient = {}
    trained_model_state_dict = trained_model.state_dict()
    for name in trained_model_state_dict:
        if name in list_named_parameters:
            local_gradient[name] = original_model[name] - trained_model_state_dict[name]
        else:
            local_gradient[name] = trained_model_state_dict[name]
    return local_gradient


@deprecated(silent=True)
def validation(self, dataloader, metric):
    if self.loss_fn is None or dataloader is None:
        return 0.0, 0.0

    device = "cuda" if torch.cuda.is_available() else "cpu"
    validation_model = copy.deepcopy(self.model)
    validation_model.to(device)
    validation_model.eval()

    loss, tmpcnt = 0, 0
    with torch.no_grad():
        for img, target in dataloader:
            tmpcnt += 1
            img = img.to(device)
            target = target.to(device)
            output = validation_model(img)
            loss += self.loss_fn(output, target).item()
    loss = loss / tmpcnt
    accuracy = _evaluate_model_on_tests(validation_model, dataloader, metric)
    return loss, accuracy


@deprecated(silent=True)
def _evaluate_model_on_tests(model, test_dataloader, metric):
    if metric is None:
        metric = _default_metric
    model.eval()
    with torch.no_grad():
        test_dataloader_iterator = iter(test_dataloader)
        y_pred_final = []
        y_true_final = []
        for X, y in test_dataloader_iterator:
            if torch.cuda.is_available():
                X = X.cuda()
                y = y.cuda()
            y_pred = model(X).detach().cpu()
            y = y.detach().cpu()
            y_pred_final.append(y_pred.numpy())
            y_true_final.append(y.numpy())

        y_true_final = np.concatenate(y_true_final)
        y_pred_final = np.concatenate(y_pred_final)
        accuracy = float(metric(y_true_final, y_pred_final))
    return accuracy


@deprecated(silent=True)
def _default_metric(y_true, y_pred):
    if len(y_pred.shape) == 1:
        y_pred = np.round(y_pred)
    else:
        y_pred = y_pred.argmax(axis=1)
    return 100 * np.sum(y_pred == y_true) / y_pred.shape[0]


@deprecated(silent=True)
def create_custom_logger(logger, cfg: DictConfig):
    dir = cfg.output_dirname
    if not os.path.isdir(dir):
        os.makedirs(dir, exist_ok=True)
    output_filename = cfg.output_filename + "_server"

    file_ext = ".txt"
    filename = dir + f"/{output_filename}{file_ext}"
    uniq = 1
    while os.path.exists(filename):
        filename = dir + "/%s_%d%s" % (output_filename, uniq, file_ext)
        uniq += 1

    logger.setLevel(logging.INFO)
    # Create handlers
    c_handler = logging.StreamHandler()
    f_handler = logging.FileHandler(filename)
    c_handler.setLevel(logging.INFO)
    f_handler.setLevel(logging.INFO)

    # Add handlers to the logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

    return logger


@deprecated(silent=True)
def client_log(dir, output_filename):
    if not os.path.isdir(dir):
        os.makedirs(dir, exist_ok=True)

    file_ext = ".txt"
    filename = dir + f"/{output_filename}{file_ext}"
    uniq = 1
    while os.path.exists(filename):
        filename = dir + "/%s_%d%s" % (output_filename, uniq, file_ext)
        uniq += 1

    outfile = open(filename, "a")

    return outfile


@deprecated(silent=True)
def load_model(cfg: DictConfig):
    file = cfg.load_model_dirname + "/{}{}".format(cfg.load_model_filename, ".pt")
    model = torch.load(file)
    model.eval()
    return model


@deprecated(silent=True)
def save_model_iteration(t, model, cfg: DictConfig):
    dir = cfg.save_model_dirname
    if not os.path.isdir(dir):
        os.mkdir(dir)

    file_ext = ".pt"
    file = dir + f"/{cfg.save_model_filename}_Round_{t}{file_ext}"
    uniq = 1
    while os.path.exists(file):
        file = dir + "/%s_Round_%s_%d%s" % (cfg.save_model_filename, t, uniq, file_ext)
        uniq += 1

    torch.save(model, file)


@deprecated(silent=True)
def load_model_state(cfg: DictConfig, model, client_id=None):
    # This function allows to use partial model weights into a model.
    # Useful since server model will only have shared layer weights when personalization is enabled.
    if client_id is None:
        file = cfg.load_model_dirname + "/{}{}".format(cfg.load_model_filename, ".pt")
    else:
        file = (
            cfg.load_model_dirname
            + "/client_%d" % client_id
            + "/{}{}".format(cfg.load_model_filename, ".pt")
        )

    model.load_state_dict(torch.load(file), strict=False)

    return model
