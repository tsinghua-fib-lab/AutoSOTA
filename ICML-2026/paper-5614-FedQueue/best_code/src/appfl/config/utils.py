"""
[Deprecated] This file contains utility functions to load configurations from yaml files.
"""

import yaml
import inspect
import importlib.util
import os.path as osp
from datetime import datetime
from omegaconf import OmegaConf
from typing import List
from appfl.config import (
    Config,
    GlobusComputeServerConfig,
    GlobusComputeClientConfig,
    ExecutableFunc,
    GlobusComputeConfig,
)
from appfl.config.fed import Federated


def show():
    conf = OmegaConf.structured(Config)
    print(OmegaConf.to_yaml(conf))


def load_executable_func(cfg_dict):
    """Load the executable function from the configuration dictionary."""
    exct_func = OmegaConf.create(ExecutableFunc(**cfg_dict))
    assert exct_func.module != "" or exct_func.script_file != "", (
        "Need to specify the executable function by (module, call) or script file"
    )
    assert not (exct_func.module != "" and exct_func.script_file != ""), (
        "Can only specify the executable function by (module, call) or script file but not both"
    )
    assert exct_func.call != "", (
        "Need to specify the function's name by setting 'call: <func name>' in the config file"
    )
    if exct_func.script_file != "":
        with open(exct_func.script_file) as fi:
            exct_func.source = fi.read()
        assert len(exct_func.source) > 0, "Source file is empty."
    return exct_func


def check_asynchronous(alg_name):
    """Check whether a given algorithm is an asynchronous FL algorithm."""
    async_list = [
        "ServerFedAsynchronous",
        "ServerFedBuffer",
        "ServerFedCompass",
        "ServerFedCompassMom",
        "ServerFedCompassNova",
    ]
    return alg_name in async_list


def check_compass(alg_name):
    """Check whether a given algorithm uses the compass scheduler."""
    compass_list = ["ServerFedCompass", "ServerFedCompassMom", "ServerFedCompassNova"]
    return alg_name in compass_list


def check_step_optimizer(optim_name):
    """Check whether a client local optimizer (trainer) runs for a certain number of steps (batches) or not."""
    step_optim_list = ["GlobusComputeClientStepOptim"]
    return optim_name in step_optim_list


def load_globus_compute_server_config(cfg: GlobusComputeConfig, config_file: str):
    """Load the server configurations from the yaml configuration file to the GlobusComputeConfig object."""
    assert osp.exists(config_file), "Config file {config_file} not found!"
    with open(config_file) as fi:
        data = yaml.load(fi, Loader=yaml.SafeLoader)
    cfg.server = OmegaConf.structured(GlobusComputeServerConfig(**data["server"]))
    assert "func" in data and "get_model" in data["func"], (
        "Please specify the function to obtain the model."
    )
    assert "get_model" in data["func"], (
        "Please specify the function to obtain the model."
    )
    assert "val_metric" in data["func"], (
        "Please specify the validation metric function."
    )
    cfg.get_model = load_executable_func(data["func"]["get_model"])
    cfg.val_metric = load_executable_func(data["func"]["val_metric"])
    # TODO: Zilinghan what is this data - this is a general dataset if each client does not specify a local dataloader
    if "get_data" in data["func"]:
        cfg.get_data = load_executable_func(data["func"]["get_data"])
    if "get_loss" in data["func"]:
        cfg.get_loss = load_executable_func(data["func"]["get_loss"])
        cfg.loss = ""
    elif "loss" in data:
        cfg.loss = data["loss"]
    if "train_data_batch_size" in data:
        cfg.train_data_batch_size = data["train_data_batch_size"]
    if "test_data_batch_size" in data:
        cfg.test_data_batch_size = data["test_data_batch_size"]
    # Load FL algorithm configs
    is_async = check_asynchronous(data["algorithm"]["servername"])
    use_compass = check_compass(data["algorithm"]["servername"])
    is_step_optimizer = check_step_optimizer(data["algorithm"]["clientname"])
    # Perform some sanity checks
    if is_step_optimizer:
        assert "num_local_steps" in data["algorithm"]["args"], (
            "Please provide the number of local steps for step-based client optimizer."
        )
    else:
        assert "num_local_epochs" in data["algorithm"]["args"], (
            "Please provide the number of local epochs for epoch-based client optimizer."
        )
    if use_compass:
        assert is_step_optimizer, (
            "Compass scheduler only works with step-based client optimizer."
        )
    # Load FL algorithm configs
    cfg.fed = Federated()
    cfg.fed.servername = data["algorithm"]["servername"]
    cfg.fed.clientname = data["algorithm"]["clientname"]
    cfg.fed.args = OmegaConf.create(data["algorithm"]["args"])
    cfg.fed.args.is_async = is_async
    cfg.fed.args.use_compass = use_compass
    # Load training configs
    cfg.num_epochs = data["training"]["num_epochs"]
    if "save_model_dirname" in data["training"]:
        cfg.save_model_dirname = data["training"]["save_model_dirname"]
    cfg.save_model_filename = data["training"]["save_model_filename"]
    # Load model configs
    cfg.model_kwargs = data["model"]
    # Load dataset configs
    cfg.dataset = data["dataset"][
        "name"
    ]  # TODO: Zilinghan I think this is not very useful


def load_globus_compute_client_config(cfg: GlobusComputeConfig, config_file: str):
    """Load the client configurations from the yaml configuration file to the GlobusComputeConfig object."""
    assert osp.exists(config_file), "Config file {config_file} not found!"
    with open(config_file) as fi:
        data = yaml.load(fi, Loader=yaml.SafeLoader)
    for client in data["clients"]:
        if "get_data" in client:
            client["get_data"] = load_executable_func(client["get_data"])
        if "data_pipeline" in client:
            client["data_pipeline"] = OmegaConf.create(client["data_pipeline"])
        client_cfg = OmegaConf.structured(GlobusComputeClientConfig(**client))
        # Make sure the output directory is unique for each client
        client_cfg.output_dir = osp.join(client_cfg.output_dir, client_cfg.endpoint_id)
        cfg.clients.append(client_cfg)
    cfg.num_clients = len(cfg.clients)
    return cfg


# ====================================================================================
# The following functions are for APPFLx web application


def load_appfl_server_config_funcx(cfg: GlobusComputeConfig, config_file: str):
    # Modified (ZL): load the configuration file for the appfl server
    assert osp.exists(config_file), "Config file {config_file} not found!"
    with open(config_file) as fi:
        data = yaml.load(fi, Loader=yaml.SafeLoader)

    ## Load configs for server
    cfg.server = OmegaConf.structured(GlobusComputeServerConfig(**data["server"]))

    ## Load module configs for get_model and get_dataset method
    if "get_data" in data["func"]:
        cfg.get_data = load_executable_func(data["func"]["get_data"])

    if "loss" in data:
        cfg.loss = data["loss"]
    if "train_data_batch_size" in data:
        cfg.train_data_batch_size = data["train_data_batch_size"]
    if "test_data_batch_size" in data:
        cfg.test_data_batch_size = data["test_data_batch_size"]
    cfg.get_model = load_executable_func(data["func"]["get_model"])

    ## Load FL algorithm configs
    cfg.fed = Federated()
    cfg.fed.servername = data["algorithm"]["servername"]
    cfg.fed.clientname = data["algorithm"]["clientname"]
    cfg.fed.args = OmegaConf.create(data["algorithm"]["args"])
    ## Load training configs
    cfg.num_epochs = data["training"]["num_epochs"]
    if "save_model_dirname" in data["training"]:
        cfg.save_model_dirname = data["training"]["save_model_dirname"]
    cfg.save_model_filename = data["training"]["save_model_filename"]
    ## Load model configs
    cfg.model_kwargs = data["model"]
    ## Load dataset configs
    cfg.dataset = data["dataset"]["name"]


def load_appfl_client_config_funcx(cfg: GlobusComputeConfig, config_file: str):
    # Modified (ZL): Load the configuration file for appfl clients
    assert osp.exists(config_file), "Config file {config_file} not found!"
    with open(config_file) as fi:
        data = yaml.load(fi, Loader=yaml.SafeLoader)

    ## Load configs for clients
    for client in data["clients"]:
        if "get_data" in client:
            client["get_data"] = load_executable_func(client["get_data"])
        if "data_pipeline" in client:
            client["data_pipeline"] = OmegaConf.create(client["data_pipeline"])
        # else:
        #     client['data_pipeline']= OmegaConf.create({})
        client_cfg = OmegaConf.structured(GlobusComputeClientConfig(**client))
        cfg.clients.append(client_cfg)

    cfg.num_clients = len(cfg.clients)
    return cfg


def get_call(script: str):
    """
    return the name of the first function inside a python script
    """
    module_spec = importlib.util.spec_from_file_location("module", script)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    functions = inspect.getmembers(module, inspect.isfunction)
    function_names = [func[0] for func in functions]
    assert len(function_names) == 1, (
        f"More than one function defined in script {script}"
    )
    return function_names[0]


def load_appfl_client_config_funcx_web(
    cfg: GlobusComputeConfig, config_files: List[str], dataloaders: List[str]
):
    assert len(config_files) == len(dataloaders), (
        "The number of configuration files and dataloader files are different!"
    )
    for config_file, dataloader_file in zip(config_files, dataloaders):
        assert osp.exists(config_file), f"Config file {config_file} not found!"
        assert osp.exists(dataloader_file), (
            f"Dataloader file {dataloader_file} not found!"
        )
        # load the client configuration file
        with open(config_file) as fi:
            data = yaml.load(fi, Loader=yaml.SafeLoader)
        client = data["client"]
        start_time = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        client["output_dir"] += f"_{cfg.dataset}_UTC_Time_{start_time}"
        # load the client dataloader
        src = OmegaConf.create(ExecutableFunc())
        src.script_file = dataloader_file
        with open(dataloader_file) as fi:
            src.source = fi.read()
        src.call = get_call(dataloader_file)
        client["get_data"] = src
        # add the client
        client_cfg = OmegaConf.structured(
            GlobusComputeClientConfig(
                device=client["device"],
                endpoint_id=client["endpoint_id"],
                output_dir=client["output_dir"],
                get_data=client["get_data"],
            )
        )
        cfg.clients.append(client_cfg)
    cfg.num_clients = len(cfg.clients)
    return cfg


def load_appfl_server_config_funcx_web(
    cfg: GlobusComputeConfig, server_config: str, model_config: str, model_file: str
):
    assert osp.exists(server_config), f"Config file {server_config} not found!"
    assert osp.exists(model_config), f"Config file {model_config} not found!"
    assert osp.exists(model_file), f"Model loader {model_file} not found!"

    # Load server configuration file
    with open(server_config) as fi:
        data = yaml.load(fi, Loader=yaml.SafeLoader)
    cfg.server = OmegaConf.structured(GlobusComputeServerConfig(**data["server"]))
    ## Load module configs for get_dataset method
    if "get_data" in data:
        cfg.get_data = load_executable_func(data["func"]["get_data"])
    if "loss" in data:
        cfg.loss = data["loss"]
    if "train_data_batch_size" in data:
        cfg.train_data_batch_size = data["train_data_batch_size"]
    if "test_data_batch_size" in data:
        cfg.test_data_batch_size = data["test_data_batch_size"]

    ## Load the dataloader
    src = OmegaConf.create(ExecutableFunc())
    src.script_file = model_file
    with open(model_file) as fi:
        src.source = fi.read()
    src.call = get_call(model_file)
    cfg.get_model = src

    ## Load FL algorithm configs
    cfg.fed = Federated()
    cfg.fed.servername = data["algorithm"]["servername"]
    cfg.fed.clientname = data["algorithm"]["clientname"]
    cfg.fed.args = OmegaConf.create(data["algorithm"]["args"])
    ## Load training configs
    cfg.num_epochs = data["training"]["num_epochs"]
    if "save_model_dirname" in data["training"]:
        cfg.save_model_dirname = data["training"]["save_model_dirname"]
    cfg.save_model_filename = data["training"]["save_model_filename"]
    ## Load dataset configs
    cfg.dataset = data["dataset"]["name"]

    ## Load model configs
    with open(model_config) as fi:
        data = yaml.load(fi, Loader=yaml.SafeLoader)
    cfg.model_kwargs = data["model"]


def load_appfl_server_config_funcx_web_v2(cfg: GlobusComputeConfig, server_config: str):
    assert osp.exists(server_config), f"Config file {server_config} not found!"
    # assert osp.exists(model_config), f"Config file {model_config} not found!"
    # assert osp.exists(model_file), f"Model loader {model_file} not found!"

    # Load server configuration file
    with open(server_config) as fi:
        data = yaml.load(fi, Loader=yaml.SafeLoader)
    cfg.server = OmegaConf.structured(GlobusComputeServerConfig(**data["server"]))
    ## Load module configs for get_dataset method
    if "get_data" in data:
        cfg.get_data = load_executable_func(data["func"]["get_data"])
    if "loss" in data:
        cfg.loss = data["loss"]
    if "train_data_batch_size" in data:
        cfg.train_data_batch_size = data["train_data_batch_size"]
    if "test_data_batch_size" in data:
        cfg.test_data_batch_size = data["test_data_batch_size"]

    ## Load the model
    src = OmegaConf.create(ExecutableFunc())
    use_hugging_face = False
    if "hf_model_arc" in data and "hf_model_weights" in data:
        if data["hf_model_arc"] is not None and data["hf_model_weights"] is not None:
            use_hugging_face = True
    if use_hugging_face:
        print(
            "Using hugging face models......",
            data["hf_model_arc"],
            data["hf_model_weights"],
        )
        cfg.hf_model_arc = data["hf_model_arc"]
        cfg.hf_model_weights = data["hf_model_weights"]
    else:
        print("Using custom models......")
        src.script_file = data["model_file"]
        with open(src.script_file) as fi:
            src.source = fi.read()
        src.call = get_call(src.script_file)
        cfg.get_model = src

    ## Load FL algorithm configs
    cfg.fed = Federated()
    cfg.fed.servername = data["algorithm"]["servername"]
    cfg.fed.clientname = data["algorithm"]["clientname"]
    cfg.fed.args = OmegaConf.create(data["algorithm"]["args"])
    ## Load training configs
    cfg.num_epochs = data["training"]["num_epochs"]
    # TODO: Currently, the save model is disabled
    if "save_model_dirname" in data["training"]:
        cfg.save_model_dirname = data["training"]["save_model_dirname"]
    cfg.save_model_filename = data["training"]["save_model_filename"]
    ## Load dataset configs
    cfg.dataset = data["dataset"]["name"]
