"""
This file contains all the functions to be run on the globus compute endpoint.
"""


def client_validate_data(cfg, client_idx, mode="train"):
    """
    client_validate_data:
        Validate the dataloader provided by the clients, and return the statistics of the dataset of different `mode` (train, val, test).
    """
    from appfl.comm.utils.client_utils import get_dataset
    from appfl.comm.globus_compute.utils.logging import GlobusComputeClientLogger

    modes = [mode] if type(mode) is str else mode
    client_logger = GlobusComputeClientLogger()
    client_logger.mark_event("Start endpoint execution")
    data_info = {}
    for m in modes:
        client_logger.start_timer(f"Load {m} dataset")
        dataset = get_dataset(cfg, client_idx, mode=m)
        data_info[m] = len(dataset)
        client_logger.stop_timer(f"Load {m} dataset")
    client_logger.mark_event("Stop endpoint execution")
    return data_info, client_logger.to_dict()


def client_training(
    cfg,
    client_idx,
    weights,
    global_state,
    local_model_key="",
    local_model_url="",
    do_validation=False,
    global_epoch=0,
):
    """
    client_training:
        Do client local training using local data and send the trained model back to the server via S3 bucket.
    Inputs:
        - cfg: FL experiment configuration
        - client_idx: index of the local client, used for finding corresponding info from `cfg`
        - weights: weights for all FL clients
        - global_state: state dictionary for the global model
        - local_model_key: s3 object key the local model to be sent back to the server via S3
        - local_model_url: presigned url for uploading local model to S3
        - do_validation: whether to perform local validation
        - global_epoch: current global epoch
    """
    import importlib
    import os.path as osp
    from appfl.misc.utils import client_log
    from appfl.comm.utils.utils import get_dataloader
    from appfl.comm.globus_compute.utils.logging import GlobusComputeClientLogger
    from appfl.comm.utils.client_utils import (
        get_dataset,
        get_model,
        get_loss,
        get_val_metric,
        load_global_state,
        send_client_state,
    )

    ## Create logger
    cli_logger = GlobusComputeClientLogger()
    cli_logger.mark_event("Start endpoint execution")

    ## Load client configs
    cfg.device = cfg.clients[client_idx].device
    cfg.output_dirname = cfg.clients[client_idx].output_dir

    cli_logger.start_timer("Load dataset")
    ## Prepare training/validation data
    train_dataset = get_dataset(cfg, client_idx, mode="train")
    train_dataloader = get_dataloader(cfg, train_dataset, mode="train")
    if do_validation:
        val_dataset = get_dataset(cfg, client_idx, mode="val")
        val_dataloader = get_dataloader(cfg, val_dataset, mode="val")
    else:
        val_dataloader = None
    cli_logger.stop_timer("Load dataset")

    ## Get training model, loss, and validation metric
    model = get_model(cfg)
    loss_fn = get_loss(cfg)
    val_metric = get_val_metric(cfg)

    ## Prepare output directory
    output_filename = cfg.output_filename + "_client_%s" % (client_idx)
    outfile = client_log(cfg.output_dirname, output_filename)

    ## Instantiate training agent at client
    appfl_alg = importlib.import_module("appfl.algorithm")
    ClientOptim = getattr(appfl_alg, cfg.fed.clientname)
    client = ClientOptim(
        client_idx,
        weights,
        model,
        loss_fn,
        train_dataloader,
        cfg,
        outfile,
        val_dataloader,
        val_metric,
        global_epoch,
        **cfg.fed.args,
    )

    ## Download global state
    cli_logger.start_timer("Download global state")
    temp_dir = osp.join(cfg.output_dirname, "tmp")
    global_state = load_global_state(cfg, global_state, temp_dir)
    cli_logger.stop_timer("Download global state")

    ## Initial state for a client
    cli_logger.start_timer("Load global state to device")
    client.model.load_state_dict(global_state)
    cli_logger.stop_timer("Load global state to device")

    ## Perform a client update
    cli_logger.start_timer("Client local training")
    client_state, cli_logger = client.update(cli_logger=cli_logger)
    cli_logger.stop_timer("Client local training")

    ## Send client state to server
    cli_logger.start_timer("Upload client state")
    res = send_client_state(
        cfg, client_state, temp_dir, local_model_key, local_model_url
    )
    cli_logger.stop_timer("Upload client state")

    cli_logger.mark_event("Stop endpoint execution")
    return res, cli_logger.to_dict()


def client_testing(cfg, client_idx, weights, global_state):
    """
    client_testing:
        Test the global model performance on the client local testing/validation dataset.
    Inputs:
        - cfg: FL experiment configuration
        - client_idx: index of the local client, used for finding corresponding info from `cfg`
        - weights: weights for all FL clients
        - global_state: state dictionary for the global model
    """
    import importlib
    import os.path as osp
    from appfl.misc.utils import client_log
    from appfl.comm.utils.utils import get_dataloader
    from appfl.comm.globus_compute.utils.logging import GlobusComputeClientLogger
    from appfl.comm.utils.client_utils import (
        get_dataset,
        load_global_state,
        get_model,
        get_loss,
        get_val_metric,
    )

    ## Create logger
    cli_logger = GlobusComputeClientLogger()
    cli_logger.mark_event("Start endpoint execution")

    ## Load client configs
    cfg.device = cfg.clients[client_idx].device
    cfg.output_dirname = cfg.clients[client_idx].output_dir

    ## Prepare testing data
    cli_logger.start_timer("Load dataset")
    try:
        test_dataset = get_dataset(cfg, client_idx, mode="test")
        test_dataloader = get_dataloader(cfg, test_dataset, mode="test")
    except:  # noqa E722
        test_dataset = get_dataset(cfg, client_idx, mode="val")
        test_dataloader = get_dataloader(cfg, test_dataset, mode="val")
    cli_logger.stop_timer("Load dataset")

    ## Get testing model, loss, and validation metric
    model = get_model(cfg)
    loss_fn = get_loss(cfg)
    val_metric = get_val_metric(cfg)

    ## Prepare output directory
    output_filename = cfg.output_filename + "_client_test_%s" % (client_idx)
    outfile = client_log(cfg.output_dirname, output_filename)

    ## Instantiate training agent at client
    appfl_alg = importlib.import_module("appfl.algorithm")
    ClientOptim = getattr(appfl_alg, cfg.fed.clientname)
    client = ClientOptim(
        client_idx,
        weights,
        model,
        loss_fn,
        None,
        cfg,
        outfile,
        test_dataloader,
        val_metric,
        **cfg.fed.args,
    )

    ## Download global state
    cli_logger.start_timer("Download global state")
    temp_dir = osp.join(cfg.output_dirname, "tmp")
    global_state = load_global_state(cfg, global_state, temp_dir)
    cli_logger.stop_timer("Download global state")

    ## Initial state for a client
    cli_logger.start_timer("Load global state to device")
    client.model.load_state_dict(global_state)
    cli_logger.stop_timer("Load global state to device")

    ## Do validation
    cli_logger.start_timer("Testing")
    loss, acc = client.client_validation()
    cli_logger.start_timer("Testing")

    cli_logger.mark_event("Stop endpoint execution")
    return {
        "client_idx": client_idx,
        "test_loss": loss,
        "test_acc": acc,
    }, cli_logger.to_dict()


def client_model_saving(cfg, client_idx, global_state):
    """
    client_model_saving:
        Save the final global model in the client's local file system.
    Inputs:
        - cfg: FL experiment configuration
        - client_idx: index of the local client, used for finding corresponding info from `cfg`
        - global_state: state dictionary for the global model
    """
    import os.path as osp
    from appfl.comm.globus_compute.utils.logging import GlobusComputeClientLogger
    from appfl.comm.utils.client_utils import save_global_model

    ## Create logger
    cli_logger = GlobusComputeClientLogger()
    cli_logger.mark_event("Start endpoint execution")

    ## Load client configs
    cfg.device = cfg.clients[client_idx].device
    cfg.output_dirname = cfg.clients[client_idx].output_dir

    ## Download global state
    cli_logger.start_timer("Download global state")
    save_dir = osp.join(cfg.output_dirname, f"final_model_client{client_idx}")
    global_state = save_global_model(cfg, global_state, save_dir)
    cli_logger.stop_timer("Download global state")

    cli_logger.mark_event("Stop endpoint execution")
    return True, cli_logger.to_dict()
