"""Legacy training functions.

DEPRECATION NOTICE:
This module is deprecated in favor of the new `training` module.
Use the trainers from `training.trainers` instead:

    from training import get_trainer, TrainerConfig

    config = TrainerConfig(epochs=1000, lr=1e-4)
    trainer = get_trainer("npe", config, model_path, task_config)
    posterior = trainer.train(data=(theta, x), device=device)

The functions in this file are kept for backward compatibility and
are used internally by some trainers in the new module.

Migration guide:
- `train_npe` -> `get_trainer("npe", ...)`
- `train_flow_matching` -> `get_trainer("fmpe", ...)`
- `train_rope` -> `get_trainer("rope", ...)`
- `train_fm_post_transform` -> `get_trainer("fm_post_transform", ...)`
- etc.
"""

import copy
import json
import pickle
import warnings
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch

# Issue deprecation warning when module is imported
warnings.warn(
    "baselines.trainers is deprecated. Use the training module instead: "
    "from training import get_trainer, TrainerConfig",
    DeprecationWarning,
    stacklevel=2,
)
from lampe.utils import GDStep
from torch import Tensor
from tqdm.auto import trange

from posteriors import (
    DirectFlowMatchingPosterior,
    DirectPosteriorEstimator,
    DualFlowPosteriorEstimator,
    Estimator,
    FlowMatchingPosterior,
    Posterior,
    RopePosterior,
)
from flow_matching.torch_flow import FlowMatching
from simulator import get_simulator
from utils.metrics import MMD
from utils.misc import train_val_split, train_val_split_n
from utils.timing import HierarchicalTimer

# Optional MLXP import for backward compatibility
try:
    from mlxp.logger import Logger as MLXPLogger
except ImportError:
    MLXPLogger = None

# Type alias for logger
Logger = Optional[Any]  # Accept any logger-like object or None


class Trainer:
    def __init__(self, name: str, logger: Logger, model_path: Path, task: Dict):
        self.name = name
        self.logger = logger  # Can be None
        self.path = model_path
        self.config = task[name]["config"]
        self.training_params = task[name].get("training_params", {})

    def train(
        self,
        data_sim: Tuple[Tensor, Tensor],
        data_cal: Tuple[Tensor, Tensor, Tensor],
        cal_values: List[int],
        device: torch.device,
        load: bool = False,
        compute_reference: bool = False,
        timing_path: Path = Path("timings"),
        **kwargs,
    ) -> Dict[str, Posterior]:
        """Train the model on the provided data."""
        train_fn = globals().get(f"train_{self.name}")
        if train_fn is None:
            raise ValueError(f"Training function for {self.name} not found.")
        models = {}
        for ncal in cal_values:
            if ncal == cal_values[-1] and not compute_reference:
                print(
                    f"Skipping reference model for {self.name} with {ncal} calibration data points"
                )
                continue
            print(f"Training {self.name} with {ncal} calibration data points")
            key = str(ncal) if (ncal != cal_values[-1] or not compute_reference) else "ref"
            theta_cal, x_cal, y_cal = (
                data_cal[0][:ncal],
                data_cal[1][:ncal],
                data_cal[2][:ncal],
            )
            timer = HierarchicalTimer()
            posterior = train_fn(
                (theta_cal, x_cal, y_cal),
                self.config,
                self.training_params,
                device,
                logger=self.logger,
                logname=f"{self.name}_{key}",
                model_path=self.path,
                load=load,
                timer=timer,
                **kwargs,
            )
            models[key] = posterior
            timer.save(timing_path / f"timing_{self.name}_{key}.json")
        return models


def train_npe(
    task: Dict,
    theta: Tensor,
    x: Tensor,
    device: torch.device,
    model_path: Path,
    logger: Optional[Logger] = None,
    logname: str = "",
    load=False,
    save=True,
    use_pretrained: bool = False,
    timer: Optional[HierarchicalTimer] = None,
    timer_operation_name: str = "npe",
) -> Estimator:
    """Neural Posterior estimation.

    Args:
        task: Configuration for the task.
        theta: Samples from the prior.
        x: Samples from the simulator or the true data.
        simulator: Simulator object.
        device: Device for computation.
        training_params: Training parameters.
        **kwargs: Additional arguments.
    Returns:
        Estimator: Estimator object.
    """
    # Training parameters
    training_params = task["npe"]["training"]
    epochs = training_params.get("epochs", 1000)
    lr = training_params.get("lr", 5e-4)
    batch_size = training_params.get("batch_size", 256)
    train_size = training_params.get("train_size", 0.9)
    max_patience = training_params.get("max_patience", 20)
    rescale = training_params.get("rescale", "z_score")

    estimator = Estimator(
        task["name"],
        theta.shape[1:],
        x.shape[1:],
        **task["npe"]["params"],
    )

    if load:
        # Load
        try:
            estimator.load_state_dict(
                torch.load(
                    model_path / (logname + ".pth"),
                    weights_only=True,
                    map_location=device,
                )
            )
            # Validate rescaler mode matches config
            loaded_mode = estimator.data_rescaler.mode
            config_mode = rescale
            if loaded_mode != config_mode:
                print(
                    f"WARNING: Loaded model uses rescaling '{loaded_mode}', "
                    f"but config specifies '{config_mode}'. Using loaded model's rescaling."
                )
            # Set rescale_name for backwards compatibility
            estimator.rescale_name = loaded_mode
            return estimator
        except FileNotFoundError:
            print("Model not found, training from scratch.")
    if use_pretrained:
        try:
            print("\tUsing pretrained NPE model")
            estimator.load_state_dict(
                torch.load(
                    model_path / ("npe.pth"),
                    weights_only=True,
                    map_location=device,
                )
            )
        except FileNotFoundError:
            raise FileNotFoundError(f"Pretrained model {'npe.pth'} not found. ")

    estimator.set_scales(theta, x, rescale)
    estimator.to(device)

    optimizer = torch.optim.Adam(estimator.parameters(), lr=lr)
    step = GDStep(optimizer, clip=5.0)
    train_loader, val_loader = train_val_split(theta, x, train_size, batch_size)

    best_loss = float("inf")
    best_state_dict = estimator.state_dict()
    patience = 0

    def _training_loop():
        nonlocal best_loss, best_state_dict, patience
        for epoch in (bar := trange(epochs, desc="Training NPE estimator", total=epochs)):
            if timer:
                with timer.time_operation("training", f"{timer_operation_name}_epoch"):
                    estimator.train()
                    loss_list_train = []
                    loss_list_val = []
                    for thb, xb in train_loader:
                        loss = estimator.compute_loss(thb.to(device), xb.to(device))
                        loss_list_train.append(step(loss))
                    estimator.eval()
                    with torch.no_grad():
                        for thb, xb in val_loader:
                            loss_list_val.append(estimator.compute_loss(thb.to(device), xb.to(device)).item())
                    loss_value_train = torch.stack(loss_list_train).mean().item()
                    loss_value_val = torch.tensor(loss_list_val).mean().item()
            else:
                estimator.train()
                loss_list_train = []
                loss_list_val = []
                for thb, xb in train_loader:
                    loss = estimator.compute_loss(thb.to(device), xb.to(device))
                    loss_list_train.append(step(loss))
                estimator.eval()
                with torch.no_grad():
                    for thb, xb in val_loader:
                        loss_list_val.append(estimator.compute_loss(thb.to(device), xb.to(device)).item())
                loss_value_train = torch.stack(loss_list_train).mean().item()
                loss_value_val = torch.tensor(loss_list_val).mean().item()

            bar.set_postfix(npe_loss=loss_value_train, npe_val_loss=loss_value_val)

            # Logging
            if logger is not None:
                logger.log_metrics(
                    {
                        "train_loss": loss_value_train,
                        "val_loss": loss_value_val,
                        "epoch": epoch,
                    },
                    log_name=logname,
                )

            patience += 1

            if loss_value_val < best_loss:
                best_loss = loss_value_val
                patience = 0
                best_state_dict = deepcopy(estimator.state_dict())

            if patience >= max_patience:
                bar.write("\tEarly stopping, best model saved.")
                bar.write(f"\tBest validation loss: {best_loss}")
                torch.save(estimator, model_path / f"{logname}.pkl")
                break
        estimator.load_state_dict(
            best_state_dict,
            strict=False,  # Allow missing rescaler keys (loaded separately)
        )

    if timer:
        with timer.time_operation("training", timer_operation_name):
            _training_loop()
    else:
        _training_loop()
    # Save
    if save:
        torch.save(estimator.state_dict(), model_path / f"{logname}.pth")
    return estimator


def train_npe_rs(
    task: Dict,
    theta: Tensor,
    x: Tensor,
    y_cal: Tensor,
    device: torch.device,
    model_path: Path,
    logger: Optional[Logger] = None,
    logname: str = "",
    load=False,
    save=True,
    mmd_weight: float = 10.0,
    timer: Optional[HierarchicalTimer] = None,
    timer_operation_name: str = "npe_rs",
) -> Estimator:
    """Neural Posterior Estimation with Robust Specification (NPE-RS).

    NPE-RS trains on simulation data (theta, x) while adding an MMD penalty
    between embeddings of x and true calibration observations y_cal. This
    encourages the embedding network to produce similar representations for
    both misspecified and true observations.

    Args:
        task: Configuration for the task.
        theta: Samples from the prior (simulation).
        x: Samples from the simulator (misspecified observations).
        y_cal: Calibration observations (true data, no theta needed).
        device: Device for computation.
        model_path: Path for saving/loading models.
        logger: Optional logger for metrics.
        logname: Name for logging and saving.
        load: If True, attempt to load pre-trained model.
        save: If True, save the trained model.
        mmd_weight: Weight for MMD regularization penalty (lambda).
        timer: Optional timer for profiling.
        timer_operation_name: Name for timer operations.

    Returns:
        Estimator: Trained NPE-RS estimator.
    """
    # Training parameters
    training_params = task["npe_rs"]["training"]
    epochs = training_params.get("epochs", 1000)
    lr = training_params.get("lr", 5e-4)
    batch_size = training_params.get("batch_size", 256)
    train_size = training_params.get("train_size", 0.9)
    max_patience = training_params.get("max_patience", 20)
    rescale = training_params.get("rescale", "z_score")
    mmd_weight = training_params.get("mmd_weight", mmd_weight)

    estimator = Estimator(
        task["name"],
        theta.shape[1:],
        x.shape[1:],
        **task["npe_rs"]["params"],
    )

    if load:
        # Load
        try:
            estimator.load_state_dict(
                torch.load(
                    model_path / (logname + ".pth"),
                    weights_only=True,
                    map_location=device,
                )
            )
            # Validate rescaler mode matches config
            loaded_mode = estimator.data_rescaler.mode
            config_mode = rescale
            if loaded_mode != config_mode:
                print(
                    f"WARNING: Loaded model uses rescaling '{loaded_mode}', "
                    f"but config specifies '{config_mode}'. Using loaded model's rescaling."
                )
            # Set rescale_name for backwards compatibility
            estimator.rescale_name = loaded_mode
            return estimator
        except FileNotFoundError:
            print("Model not found, training from scratch.")

    estimator.set_scales(theta, x, rescale)
    estimator.to(device)

    optimizer = torch.optim.Adam(estimator.parameters(), lr=lr)
    step = GDStep(optimizer, clip=5.0)
    train_loader, val_loader = train_val_split(theta, x, train_size, batch_size)

    # Move y_cal to device
    y_cal = y_cal.to(device)

    best_loss = float("inf")
    best_state_dict = estimator.state_dict()
    patience = 0

    def _training_loop():
        nonlocal best_loss, best_state_dict, patience
        for epoch in (bar := trange(epochs, desc="Training NPE-RS estimator", total=epochs)):
            if timer:
                with timer.time_operation("training", f"{timer_operation_name}_epoch"):
                    estimator.train()
                    loss_list_train = []
                    loss_list_val = []
                    mmd_list_train = []

                    for thb, xb in train_loader:
                        thb, xb = thb.to(device), xb.to(device)

                        # Sample corresponding batch from y_cal
                        batch_size_current = xb.shape[0]
                        y_cal_indices = torch.randint(0, y_cal.shape[0], (batch_size_current,))
                        y_cal_batch = y_cal[y_cal_indices]

                        # Compute embeddings (need rescaled data for embedding comparison)
                        with torch.no_grad():
                            _, xb_rescaled = estimator._auto_rescale(thb, xb)
                            _, y_cal_rescaled = estimator._auto_rescale(thb, y_cal_batch)

                        embed_x = estimator.embedding_net(xb_rescaled)
                        embed_y = estimator.embedding_net(y_cal_rescaled)

                        # Compute MMD penalty
                        mmd_loss = MMD(embed_x, embed_y, kernel="rbf")

                        # Compute NPE loss (with automatic rescaling)
                        npe_loss = estimator.compute_loss(thb, xb)

                        # Total loss
                        total_loss = npe_loss + mmd_weight * mmd_loss

                        # Optimization step
                        optimizer.zero_grad()
                        total_loss.backward()
                        torch.nn.utils.clip_grad_norm_(estimator.parameters(), 5.0)
                        optimizer.step()

                        loss_list_train.append(total_loss.detach())
                        mmd_list_train.append(mmd_loss.detach())

                    estimator.eval()
                    with torch.no_grad():
                        for thb, xb in val_loader:
                            loss_list_val.append(estimator.compute_loss(thb.to(device), xb.to(device)).item())

                    loss_value_train = torch.stack(loss_list_train).mean().item()
                    loss_value_val = torch.tensor(loss_list_val).mean().item()
                    mmd_value_train = torch.stack(mmd_list_train).mean().item()
            else:
                estimator.train()
                loss_list_train = []
                loss_list_val = []
                mmd_list_train = []

                for thb, xb in train_loader:
                    thb, xb = thb.to(device), xb.to(device)

                    # Sample corresponding batch from y_cal
                    batch_size_current = xb.shape[0]
                    y_cal_indices = torch.randint(0, y_cal.shape[0], (batch_size_current,))
                    y_cal_batch = y_cal[y_cal_indices]

                    # Compute embeddings (need rescaled data for embedding comparison)
                    with torch.no_grad():
                        _, xb_rescaled = estimator._auto_rescale(thb, xb)
                        _, y_cal_rescaled = estimator._auto_rescale(thb, y_cal_batch)

                    embed_x = estimator.embedding_net(xb_rescaled)
                    embed_y = estimator.embedding_net(y_cal_rescaled)

                    # Compute MMD penalty
                    mmd_loss = MMD(embed_x, embed_y, kernel="rbf")

                    # Compute NPE loss (with automatic rescaling)
                    npe_loss = estimator.compute_loss(thb, xb)

                    # Total loss
                    total_loss = npe_loss + mmd_weight * mmd_loss

                    # Optimization step
                    optimizer.zero_grad()
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(estimator.parameters(), 5.0)
                    optimizer.step()

                    loss_list_train.append(total_loss.detach())
                    mmd_list_train.append(mmd_loss.detach())

                estimator.eval()
                with torch.no_grad():
                    for thb, xb in val_loader:
                        loss_list_val.append(estimator.compute_loss(thb.to(device), xb.to(device)).item())

                loss_value_train = torch.stack(loss_list_train).mean().item()
                loss_value_val = torch.tensor(loss_list_val).mean().item()
                mmd_value_train = torch.stack(mmd_list_train).mean().item()

            bar.set_postfix(
                npe_loss=loss_value_train,
                npe_val_loss=loss_value_val,
                mmd=mmd_value_train,
            )

            # Logging
            if logger is not None:
                logger.log_metrics(
                    {
                        "train_loss": loss_value_train,
                        "val_loss": loss_value_val,
                        "mmd_loss": mmd_value_train,
                        "epoch": epoch,
                    },
                    log_name=logname,
                )

            patience += 1

            if loss_value_val < best_loss:
                best_loss = loss_value_val
                patience = 0
                best_state_dict = deepcopy(estimator.state_dict())

            if patience >= max_patience:
                bar.write("\tEarly stopping, best model saved.")
                bar.write(f"\tBest validation loss: {best_loss}")
                torch.save(estimator, model_path / f"{logname}.pkl")
                break
        estimator.load_state_dict(
            best_state_dict,
            strict=False,  # Allow missing rescaler keys (loaded separately)
        )

    if timer:
        with timer.time_operation("training", timer_operation_name):
            _training_loop()
    else:
        _training_loop()
    # Save
    if save:
        torch.save(estimator.state_dict(), model_path / f"{logname}.pth")
    return estimator


def train_flow_matching(
    target: Tensor,
    cond: Tensor,
    config: Dict,
    device: torch.device,
    logger: Optional[Logger],
    logname: str = "",
    load: bool = False,
    save: bool = True,
    model_path: Path = Path("models"),
    epochs: int = 100,
    lr: float = 1e-4,
    batch_size: int = 128,
    train_size: float = 0.8,
    max_patience: int = 20,
    rescale: str = "z_score",
    use_pretrained: bool = False,
    timer: Optional[HierarchicalTimer] = None,
    timer_operation_name: str = "flow_matching",
):
    flow_matching = FlowMatching(
        config["probability_path"],
        config["prior"],
        config["base_dist"],
        target.shape[1:],
        cond_dim=cond.shape[1:],
        **config["params"],
    )
    if use_pretrained:
        print("\tUsing pretrained flow matching model")
        try:
            flow_matching.load_state_dict(
                torch.load(
                    model_path / (logname.rsplit("_", 1)[0] + "_pretrained" + ".pth"),
                    weights_only=True,
                    map_location=device,
                )
            )
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Pretrained model {logname + '_pretrained' + '.pth'} not found. "
            )
    if load:
        print("\tLoading model from", model_path / (logname + ".pth"))
        try:
            flow_matching.load_state_dict(
                torch.load(
                    model_path / (logname + ".pth"),
                    weights_only=True,
                    map_location=device,
                )
            )
            # Validate rescaler mode matches config
            loaded_mode = flow_matching.target_rescaler.mode
            config_mode = rescale
            if loaded_mode != config_mode:
                print(
                    f"WARNING: Loaded flow model uses rescaling '{loaded_mode}', "
                    f"but config specifies '{config_mode}'. Using loaded model's rescaling."
                )
            # Set rescale_name for backwards compatibility
            flow_matching.rescale_name = loaded_mode
            return flow_matching
        except FileNotFoundError:
            print("Model not found, training from scratch.")
    flow_matching.to(device)
    flow_matching.set_scales(target, cond, rescale)
    train_loader, val_loader = train_val_split(target, cond, train_size, batch_size)
    optimizer = torch.optim.Adam(flow_matching.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    best_loss = float("inf")
    best_state_dict = flow_matching.state_dict()
    patience = 0
    bar = trange(
        epochs,
        desc="Training Flow Matching",
        total=epochs,
    )

    def _training_loop():
        nonlocal best_loss, best_state_dict, patience
        for epoch in bar:
            if timer:
                with timer.time_operation("training", f"{timer_operation_name}_epoch"):
                    # Training
                    flow_matching.train()
                    loss_list_train = []
                    loss_list_val = []
                    for xbatch, ybatch in train_loader:
                        optimizer.zero_grad()
                        loss = flow_matching.compute_loss(xbatch.to(device), ybatch.to(device))
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(flow_matching.parameters(), 1.0)
                        optimizer.step()
                        loss_list_train.append(loss.detach())
                    scheduler.step()

                    # Validation
                    flow_matching.eval()
                    with torch.no_grad():
                        for xbatch, ybatch in val_loader:
                            loss = flow_matching.compute_loss(xbatch.to(device), ybatch.to(device))
                            loss_list_val.append(loss.detach())
                    loss_value_train = torch.stack(loss_list_train).mean().item()
                    loss_value_val = torch.stack(loss_list_val).mean().item()
            else:
                # Training
                flow_matching.train()
                loss_list_train = []
                loss_list_val = []
                for xbatch, ybatch in train_loader:
                    optimizer.zero_grad()
                    loss = flow_matching.compute_loss(xbatch.to(device), ybatch.to(device))
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(flow_matching.parameters(), 1.0)
                    optimizer.step()
                    loss_list_train.append(loss.detach())
                scheduler.step()

                # Validation
                flow_matching.eval()
                with torch.no_grad():
                    for xbatch, ybatch in val_loader:
                        loss = flow_matching.compute_loss(xbatch.to(device), ybatch.to(device))
                        loss_list_val.append(loss.detach())
                loss_value_train = torch.stack(loss_list_train).mean().item()
                loss_value_val = torch.stack(loss_list_val).mean().item()
            bar.set_postfix(train=loss_value_train, val=loss_value_val)

            # Logging
            if logger is not None:
                logger.log_metrics(
                    {
                        "train_loss": loss_value_train,
                        "val_loss": loss_value_val,
                        "epoch": epoch,
                    },
                    log_name=logname,
                )

            patience += 1

            if loss_value_val < best_loss:
                best_loss = loss_value_val
                patience = 0
                best_state_dict = deepcopy(flow_matching.state_dict())

            if patience >= max_patience:
                bar.write("\tEarly stopping, best model saved.")
                bar.write(f"\tBest validation loss: {best_loss}")
                break

        flow_matching.load_state_dict(best_state_dict, strict=False)

    if timer:
        with timer.time_operation("training", timer_operation_name):
            _training_loop()
    else:
        _training_loop()
    if save:
        torch.save(flow_matching.state_dict(), model_path / (logname + ".pth"))
    return flow_matching


#####################################
######### METHODS ##################
######################################


def train_fmcpe_fm(
    data: Tuple[Tensor, Tensor, Tensor],
    config: Dict,
    training_params: Dict,
    device: torch.device,
    logger: Optional[Logger],
    logname: str = "",
    load: bool = False,
    save: bool = True,
    model_path: Path = Path("models"),
    rescale: str = "z_score",
    timer: Optional[HierarchicalTimer] = None,
):
    theta, x, y = data
    ncal = x.shape[0]

    # ---- Load simulator ---- #
    with open(model_path / "simulator.json", "r") as f:
        config_simulator = json.load(f)
        params = config_simulator["params"]
        name = config_simulator["name"]
        simulator = get_simulator(name, **params)
    try:
        simulate = simulator.get_simulator(misspecified=True)
    except NotImplementedError:
        simulate = None

    # Get Flow in x space
    assert timer is not None
    with timer.time_operation("training", "fmcpe_fm_flow_x"):
        flow = get_or_train_flow_x(
            theta,
            x,
            y,
            simulate,
            config,
            model_path,
            device,
            logger,
            logname + "_x",
            load,
            save,
            **training_params,
        )

    # FMPE config
    with open(model_path / "npe_fmpe.json", "r") as f:
        config_npe = json.load(f)
    fmpe_flow = FlowMatching(
        config_npe["probability_path"],
        config_npe["prior"],
        config_npe["base_dist"],
        theta.shape[1:],
        cond_dim=y.shape[1:],
        **config_npe["params"],
    )
    fmpe_flow.load_state_dict(
        torch.load(model_path / "npe_fmpe.pth", weights_only=True, map_location=device)
    )
    fmpe_flow.rescale_name = rescale
    fmpe = DirectFlowMatchingPosterior("npe_fmpe", fmpe_flow)
    posterior = FlowMatchingPosterior(
        "fmcpe_fm",
        posterior=fmpe,
        flow=flow,
        theta_dim=theta.shape[1:],
        obs_dim=x.shape[1:],
    )

    torch.save(flow.state_dict(), model_path / f"flow_x_y_{ncal}.pth")

    return posterior


def train_fm_post_transform_only_ut(
    data: Tuple[Tensor, Tensor, Tensor],
    config: Dict,
    training_params: Dict,
    device: torch.device,
    logger: Optional[Logger] = None,
    logname: str = "",
    load=False,
    save=False,
    model_path: Path = Path("models"),
    rescale: str = "z_score",
    timer: Optional[HierarchicalTimer] = None,
    timer_operation_name: str = "fm_post_transform_only_ut",
) -> DualFlowPosteriorEstimator:
    theta, x, y = data

    # Extract rescaling parameter from training_params
    rescale = training_params.get("rescale", "z_score")

    # ---- Load simulator ---- #
    with open(model_path / "simulator.json", "r") as f:
        config_simulator = json.load(f)
        params = config_simulator["params"]
        name = config_simulator["name"]
        simulator = get_simulator(name, **params)
    try:
        simulate = simulator.get_simulator(misspecified=True)
    except NotImplementedError:
        simulate = None

    # ---- Load npe ---- #
    if config["npe"] == "npe":
        with open(model_path / "npe.pkl", "rb") as f:
            npe = pickle.load(f)
        npe.cpu()

    elif config["npe"] == "fmpe":
        with open(model_path / "npe_fmpe.json", "rb") as f:
            config_npe = json.load(f)
        flow_matching = FlowMatching(
            config_npe["probability_path"],
            config_npe["prior"],
            config_npe["base_dist"],
            theta.shape[1:],
            cond_dim=x.shape[1:],
            **config_npe["params"],
        )
        flow_matching.rescale_name = rescale
        npe = DirectFlowMatchingPosterior("npe_fmpe", flow_matching)
        # drift: ContinuousFlow = flow_matching.drift
        # embedding_net_x = copy.deepcopy(drift.context_embedding_net)
    else:
        raise ValueError(f"Unknown npe type {config['npe']}, should be 'npe' or 'fmpe'")
    npe.to(device)

    # ---- Training/ loading x flow ---- #

    flow_matching_x = get_or_train_flow_x(
        theta,
        x,
        y,
        simulate,
        config["flow_x"],
        model_path,
        device,
        logger,
        logname + "_x",
        load,
        save,
        **training_params,
    )

    flow_matching_theta = FlowMatching(
        config["flow_theta"]["probability_path"],
        config["flow_theta"]["prior"],
        config["flow_theta"]["base_dist"],
        theta.shape[1:],
        cond_dim=y.shape[1:],
        **config["flow_theta"]["params"],
    )

    # Training parameters
    if load:
        print(f"\tLoading model from {model_path / f'{logname}.pth'}")
        try:
            flow_matching_theta.load_state_dict(
                torch.load(
                    model_path / f"{logname}_theta.pth",
                    weights_only=True,
                    map_location=device,
                )
            )
            flow_matching_x.load_state_dict(
                torch.load(
                    model_path / f"{logname}_x.pth",
                    weights_only=True,
                    map_location=device,
                )
            )
        except FileNotFoundError:
            raise FileNotFoundError(
                "Model path does not exist, set load_models=False in config.yaml"
            )
        return DualFlowPosteriorEstimator(
            "fm_post_transform_only_ut",
            base_dist=npe,
            flow_theta=flow_matching_theta,
            flow_x=flow_matching_x,
        )

    flow_matching_theta.to(device)
    flow_matching_x.to(device)

    # FIX: Set rescaling parameters for both flow matching models
    flow_matching_theta.set_scales(theta, y, rescale)
    flow_matching_x.set_scales(x, y, rescale)

    # Load training params
    epochs = training_params.get("epochs", 1000)
    batch_size = training_params.get("batch_size", 128)
    lr = training_params.get("lr", 1e-4)
    train_size = training_params.get("train_size", 0.8)
    max_patience = training_params.get("max_patience", 20)

    optimizer = torch.optim.Adam(flow_matching_theta.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    # Prepare data loaders
    train_loader, val_loader = train_val_split_n(
        theta, x, y, train_size=train_size, batch_size=batch_size
    )

    # --- Initialize weights --- #
    # init_method = config["flow_theta"].get("init", None)
    # if init_method == "emdedding":
    #     print("\tInitializing theta flow context net with embedding net of npe")
    #     state_dict = embedding_net_x.state_dict()
    #     flow_matching_theta.drift.context_embedding_net.load_state_dict(state_dict)
    #
    # elif init_method == "flow_x":
    #     print("\tInitializing theta flow context net with context net of x flow")
    #     state_dict = flow_matching_x.drift.context_embedding_net.state_dict()
    #     flow_matching_theta.drift.context_embedding_net.load_state_dict(state_dict)
    # else:
    #     pass

    assert timer is not None
    with timer.time_operation("training", "fm_post_transform_only_ut"):
        flow_matching_theta, flow_matching_x = _train_loop_flow_theta_x(
            flow_matching_theta,
            flow_matching_x,
            npe,
            train_loader,
            val_loader,
            None,
            device,
            optimizer,
            scheduler,
            epochs,
            max_patience,
            train_flow_x=False,
            timer=timer,
            model_path=model_path,
            logname=logname,
            logger=logger,
            desc="Training sequential FM",
        )

    posterior = DualFlowPosteriorEstimator(
        "fm_post_transform_only_ut",
        base_dist=npe,
        flow_theta=flow_matching_theta,
        flow_x=flow_matching_x,
        embedding_net=None,
    )
    return posterior


def train_fm_post_transform(
    data: Tuple[Tensor, Tensor, Tensor],
    config: Dict,
    training_params: Dict,
    device: torch.device,
    logger: Optional[Logger] = None,
    logname: str = "",
    load=False,
    save=False,
    model_path: Path = Path("models"),
    rescale: str = "z_score",
    timer: Optional[HierarchicalTimer] = None,
    timer_operation_name: str = "fm_post_transform",
) -> DualFlowPosteriorEstimator:
    theta, x, y = data

    # Extract rescaling parameter from training_params
    rescale = training_params.get("rescale", "z_score")

    # ---- Load simulator ---- #
    with open(model_path / "simulator.json", "r") as f:
        config_simulator = json.load(f)
        params = config_simulator["params"]
        name = config_simulator["name"]
        simulator = get_simulator(name, **params)
    try:
        simulate = simulator.get_simulator(misspecified=True)
    except NotImplementedError:
        simulate = None

    flow_matching_theta = FlowMatching(
        config["flow_theta"]["probability_path"],
        config["flow_theta"]["prior"],
        config["flow_theta"]["base_dist"],
        theta.shape[1:],
        cond_dim=y.shape[1:],
        **config["flow_theta"]["params"],
    )
    flow_matching_x = FlowMatching(
        config["flow_x"]["probability_path"],
        config["flow_x"]["prior"],
        config["flow_x"]["base_dist"],
        x.shape[1:],
        cond_dim=y.shape[1:],
        **config["flow_x"]["params"],
    )

    # ---- Load npe ---- #
    if config["npe"] == "npe":
        with open(model_path / "npe.pkl", "rb") as f:
            npe = pickle.load(f)
        # embedding_net_x = copy.deepcopy(npe.embedding_net)
        npe = DirectPosteriorEstimator("npe", npe)

    elif config["npe"] == "fmpe":
        with open(model_path / "npe_fmpe.json", "rb") as f:
            config_npe = json.load(f)
        flow_matching = FlowMatching(
            config_npe["probability_path"],
            config_npe["prior"],
            config_npe["base_dist"],
            theta.shape[1:],
            cond_dim=x.shape[1:],
            **config_npe["params"],
        )
        flow_matching.rescale_name = rescale
        npe = DirectFlowMatchingPosterior("npe_fmpe", flow_matching)
        # embedding_net_x = copy.deepcopy(drift.context_embedding_net)
    else:
        raise ValueError(f"Unknown npe type {config['npe']}, should be 'npe' or 'fmpe'")
    npe.to(device)

    # Training parameters
    if load:
        print(f"\tLoading model from {model_path / f'{logname}.pth'}")
        try:
            flow_matching_theta.load_state_dict(
                torch.load(
                    model_path / f"{logname}_theta.pth",
                    weights_only=True,
                    map_location=device,
                )
            )
            flow_matching_x.load_state_dict(
                torch.load(
                    model_path / f"{logname}_x.pth",
                    weights_only=True,
                    map_location=device,
                )
            )
        except FileNotFoundError:
            raise FileNotFoundError(
                "Model path does not exist, set load_models=False in config.yaml"
            )
        posterior = DualFlowPosteriorEstimator(
            "fm_post_transform",
            base_dist=npe,
            flow_theta=flow_matching_theta,
            flow_x=flow_matching_x,
        )
        return posterior

    flow_matching_theta.to(device)
    flow_matching_x.to(device)

    # FIX: Set rescaling parameters for both flow matching models
    flow_matching_theta.set_scales(theta, y, rescale)
    flow_matching_x.set_scales(x, y, rescale)

    # Load training params
    epochs = training_params.get("epochs", 1000)
    batch_size = training_params.get("batch_size", 128)
    lr = training_params.get("lr", 1e-4)
    train_size = training_params.get("train_size", 0.8)
    max_patience = training_params.get("max_patience", 20)
    # Combine parameters of both models
    optimizer = torch.optim.Adam(
        list(flow_matching_theta.parameters()) + list(flow_matching_x.parameters()), lr=lr
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    # Prepare data loaders
    batch_size = max(theta.shape[0] // 5, 4)  # Ensure at least batch size of 4
    train_loader, val_loader = train_val_split_n(
        theta, x, y, train_size=train_size, batch_size=batch_size
    )

    # Initialize early stopping and checkpoint tracking

    assert timer is not None
    with timer.time_operation("training", "fm_post_transform"):
        flow_matching_theta, flow_matching_x = _train_loop_flow_theta_x(
            flow_matching_theta,
            flow_matching_x,
            npe,
            train_loader,
            val_loader,
            simulate,
            device,
            optimizer,
            scheduler,
            epochs,
            max_patience,
            train_flow_x=True,
            timer=timer,
            model_path=model_path,
            logname=logname,
            logger=logger,
            save=save,
            desc="Training jointly coupled flow-matching models",
        )

    posterior = DualFlowPosteriorEstimator(
        "fm_post_transform",
        base_dist=npe,
        flow_theta=flow_matching_theta,
        flow_x=flow_matching_x,
    )
    return posterior


def train_fm_post_transform_plug_y(
    data: Tuple[Tensor, Tensor, Tensor],
    config: Dict,
    training_params: Dict,
    device: torch.device,
    logger: Optional[Logger] = None,
    logname: str = "",
    load=False,
    save=False,
    model_path: Path = Path("models"),
    rescale: str = "z_score",
    timer: Optional[HierarchicalTimer] = None,
    timer_operation_name: str = "fm_post_transform_plug_y",
) -> DualFlowPosteriorEstimator:
    theta, x, y = data

    # Extract rescaling parameter from training_params
    rescale = training_params.get("rescale", "z_score")

    flow_matching_theta = FlowMatching(
        config["flow_theta"]["probability_path"],
        config["flow_theta"]["prior"],
        config["flow_theta"]["base_dist"],
        theta.shape[1:],
        cond_dim=y.shape[1:],
        **config["flow_theta"]["params"],
    )
    # ---- Load npe ---- #
    if config["npe"] == "npe":
        with open(model_path / "npe.pkl", "rb") as f:
            npe = pickle.load(f)
        # embedding_net_x = copy.deepcopy(npe.embedding_net)
        npe = DirectPosteriorEstimator("npe", npe)

    elif config["npe"] == "fmpe":
        with open(model_path / "npe_fmpe.json", "rb") as f:
            config_npe = json.load(f)
        flow_matching = FlowMatching(
            config_npe["probability_path"],
            config_npe["prior"],
            config_npe["base_dist"],
            theta.shape[1:],
            cond_dim=x.shape[1:],
            **config_npe["params"],
        )
        flow_matching.rescale_name = rescale
        npe = DirectFlowMatchingPosterior("npe_fmpe", flow_matching)
        # embedding_net_x = copy.deepcopy(drift.context_embedding_net)
    else:
        raise ValueError(f"Unknown npe type {config['npe']}, should be 'npe' or 'fmpe'")
    npe.to(device)

    # Training parameters
    if load:
        print(f"\tLoading model from {model_path / f'{logname}.pth'}")
        try:
            flow_matching_theta.load_state_dict(
                torch.load(
                    model_path / f"{logname}_theta.pth",
                    weights_only=True,
                    map_location=device,
                )
            )
        except FileNotFoundError:
            raise FileNotFoundError(
                "Model path does not exist, set load_models=False in config.yaml"
            )
        posterior = DualFlowPosteriorEstimator(
            "fm_post_transform",
            base_dist=npe,
            flow_theta=flow_matching_theta,
            flow_x=None,
        )
        return posterior

    flow_matching_theta.to(device)

    # FIX: Set rescaling parameters for flow matching model
    flow_matching_theta.set_scales(theta, y, rescale)

    # Load training params
    epochs = training_params.get("epochs", 1000)
    batch_size = training_params.get("batch_size", 128)
    lr = training_params.get("lr", 1e-4)
    train_size = training_params.get("train_size", 0.8)
    max_patience = training_params.get("max_patience", 20)

    # Combine parameters of both models
    optimizer = torch.optim.Adam(flow_matching_theta.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    # Prepare data loaders
    train_loader, val_loader = train_val_split_n(
        theta, x, y, train_size=train_size, batch_size=batch_size
    )

    assert timer is not None
    with timer.time_operation("training", "fm_post_transform_plug_y"):
        flow_matching_theta, flow_matching_x = _train_loop_flow_theta_x(
            flow_matching_theta,
            None,
            npe,
            train_loader,
            val_loader,
            None,
            device,
            optimizer,
            scheduler,
            epochs,
            max_patience,
            train_flow_x=True,
            timer=timer,
            model_path=model_path,
            logname=logname,
            logger=logger,
            save=save,
            desc="Training FM with plug y",
        )

    posterior = DualFlowPosteriorEstimator(
        "fm_post_transform_plug_y",
        base_dist=npe,
        flow_theta=flow_matching_theta,
        flow_x=None,
    )
    return posterior


def train_rope(
    data: Tuple[Tensor, Tensor, Tensor],
    config: Dict,
    training_params: Dict,
    device: torch.device,
    logger: Optional[Logger],
    logname: str = "",
    load: bool = False,
    save: bool = True,
    model_path: Path = Path("models"),
    timer: Optional[HierarchicalTimer] = None,
    timer_operation_name: str = "rope",
    **kwargs,
) -> RopePosterior:
    """Train RoPE (Robust Posterior Estimation) by fine-tuning an embedding network.

    RoPE learns to align embeddings of misspecified observations (y) with embeddings
    of well-specified simulations (x) by fine-tuning a copy of the NPE embedding
    network. This enables robust posterior estimation under model misspecification.

    Args:
        data: Tuple of (theta, x, y) tensors containing parameters, simulation
            observations, and real observations respectively.
        config: Configuration dictionary containing RoPE hyperparameters.
        device: PyTorch device for computation.
        logger: Optional logger for tracking training metrics.
        logname: Name prefix for logging and model checkpoints.
        load: If True, attempt to load pre-trained fine-tuned embedding network.
        save: If True, save the best fine-tuned embedding network.
        model_path: Directory path for loading/saving model checkpoints.
        epochs: Number of training epochs.
        lr: Learning rate for optimizer.
        batch_size: Batch size for training.
        train_size: Fraction of data to use for training (rest for validation).
        rescale: Data rescaling method ('z_score', 'whiten', or 'none').

    Returns:
        RopePosterior containing the original and fine-tuned embedding networks.

    Raises:
        FileNotFoundError: If the required NPE model cannot be loaded.
    """
    theta, x, y = data
    epochs = training_params.get("epochs", 100)
    lr = training_params.get("lr", 1e-4)
    batch_size = training_params.get("batch_size", 128)
    train_size = training_params.get("train_size", 0.8)
    rescale = training_params.get("rescale", "none")

    # Load pre-trained NPE and extract embedding network.
    npe_path = model_path / "npe.pkl"
    if not npe_path.exists():
        raise FileNotFoundError(f"NPE model not found at {npe_path}")

    with open(npe_path, "rb") as f:
        npe: Estimator = pickle.load(f)

    print("rescaler:", "npe:", npe.rescale_name, "rope:", rescale)
    print(npe.data_rescaler, npe.cond_rescaler)
    npe.set_scales(theta, x, rescale)
    embedding_net_original = npe.embedding_net

    # Handle Identity embedding network (no fine-tuning needed).
    if isinstance(embedding_net_original, torch.nn.Identity):
        print("\tEmbedding network is Identity - no fine-tuning required")
        return RopePosterior(
            name="rope",
            theta_dim=theta.shape[-1],
            obs_dim=x.shape[-1],
            embedding_net=embedding_net_original,
            fine_tuned_embedding_net=embedding_net_original,
            flow_model=npe,
        )

    # Attempt to load pre-trained fine-tuned model.
    embedding_net_finetuned = copy.deepcopy(embedding_net_original)
    checkpoint_path = model_path / f"{logname}.pth"

    if load and checkpoint_path.exists():
        print(f"\tLoading fine-tuned embedding from {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, weights_only=True, map_location=device)
        embedding_net_finetuned.load_state_dict(state_dict)
        return RopePosterior(
            name="rope",
            theta_dim=theta.shape[-1],
            obs_dim=x.shape[-1],
            embedding_net=embedding_net_original,
            fine_tuned_embedding_net=embedding_net_finetuned,
            flow_model=npe,
        )

    print(f"\tTraining RoPE embedding network for {epochs} epochs")

    # Setup training.
    embedding_net_original.to(device).requires_grad_(False)
    embedding_net_finetuned.to(device).train()

    optimizer = torch.optim.Adam(embedding_net_finetuned.parameters(), lr=lr)
    train_loader, val_loader = train_val_split_n(
        theta, x, y, train_size=train_size, batch_size=batch_size
    )

    # Early stopping state.
    best_val_loss = float("inf")
    best_state_dict = copy.deepcopy(embedding_net_finetuned.state_dict())

    # Training loop.
    patience = 0
    max_patience = kwargs.get("max_patience", 20)

    def _training_loop():
        nonlocal patience, best_val_loss, best_state_dict
        for epoch in (pbar := trange(epochs, desc="Fine-tuning embedding network")):
            if timer:
                with timer.time_operation("training", f"{timer_operation_name}_epoch"):
                    # Training phase.
                    train_losses = _rope_train_epoch(
                        embedding_net_original=embedding_net_original,
                        embedding_net_finetuned=embedding_net_finetuned,
                        npe=npe,
                        train_loader=train_loader,
                        optimizer=optimizer,
                        device=device,
                    )

                    # Validation phase.
                    val_losses = _rope_validate_epoch(
                        embedding_net_original=embedding_net_original,
                        embedding_net_finetuned=embedding_net_finetuned,
                        npe=npe,
                        val_loader=val_loader,
                        device=device,
                    )

                    # Aggregate metrics.
                    mean_train_loss = torch.stack(train_losses).mean().item()
                    mean_val_loss = torch.stack(val_losses).mean().item()
            else:
                # Training phase.
                train_losses = _rope_train_epoch(
                    embedding_net_original=embedding_net_original,
                    embedding_net_finetuned=embedding_net_finetuned,
                    npe=npe,
                    train_loader=train_loader,
                    optimizer=optimizer,
                    device=device,
                )

                # Validation phase.
                val_losses = _rope_validate_epoch(
                    embedding_net_original=embedding_net_original,
                    embedding_net_finetuned=embedding_net_finetuned,
                    npe=npe,
                    val_loader=val_loader,
                    device=device,
                )

                # Aggregate metrics.
                mean_train_loss = torch.stack(train_losses).mean().item()
                mean_val_loss = torch.stack(val_losses).mean().item()
            pbar.set_postfix(train=mean_train_loss, val=mean_val_loss)

            # Track best model.
            if mean_val_loss < best_val_loss:
                best_val_loss = mean_val_loss
                best_state_dict = copy.deepcopy(embedding_net_finetuned.state_dict())
                patience = 0

            # Log metrics.
            if logger is not None:
                logger.log_metrics(
                    {
                        "train_loss": mean_train_loss,
                        "val_loss": mean_val_loss,
                        "epoch": epoch,
                    },
                    log_name=logname,
                )

            # Stop training if no improvement over max_patience epochs.
            if patience >= max_patience:
                print(f"\tEarly stopping at epoch {epoch + 1}")
                break

        # Restore best model and save.
        embedding_net_finetuned.load_state_dict(best_state_dict)

    if timer:
        with timer.time_operation("training", "rope"):
            _training_loop()
    else:
        _training_loop()
    if save:
        torch.save(best_state_dict, checkpoint_path)
        print(f"\tSaved fine-tuned embedding to {checkpoint_path}")

    # Move networks to CPU and construct posterior.
    embedding_net_original.cpu()
    embedding_net_finetuned.cpu()
    npe.cpu()

    return RopePosterior(
        name="rope",
        theta_dim=theta.shape[-1],
        obs_dim=x.shape[-1],
        embedding_net=embedding_net_original,
        fine_tuned_embedding_net=embedding_net_finetuned,
        flow_model=npe,
    )


def _rope_train_epoch(
    embedding_net_original: torch.nn.Module,
    embedding_net_finetuned: torch.nn.Module,
    npe: Estimator,
    train_loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> list[Tensor]:
    """Execute one training epoch for RoPE embedding fine-tuning.

    Args:
        embedding_net_original: Frozen original embedding network from NPE.
        embedding_net_finetuned: Fine-tuned embedding network being trained.
        npe: NPE estimator for data rescaling.
        train_loader: DataLoader for training batches.
        optimizer: Optimizer for updating fine-tuned network.
        device: PyTorch device for computation.

    Returns:
        List of loss tensors for each training batch.
    """
    embedding_net_finetuned.train()
    losses = []

    for theta_batch, x_batch, y_batch in train_loader:
        theta_batch = theta_batch.to(device)
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        # Rescale inputs using NPE scaling.
        with torch.no_grad():
            _, x_rescaled = npe._auto_rescale(theta_batch, x_batch)
            _, y_rescaled = npe._auto_rescale(theta_batch, y_batch)

            # Compute original embeddings (frozen).
            embedding_original = embedding_net_original(x_rescaled)

        # Compute fine-tuned embeddings.
        embedding_finetuned = embedding_net_finetuned(y_rescaled)

        # MSE loss to align embeddings.
        loss = (embedding_finetuned - embedding_original).pow(2).mean()

        # Optimization step.
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(embedding_net_finetuned.parameters(), max_norm=5.0)
        optimizer.step()

        losses.append(loss.detach())

    return losses


def _rope_validate_epoch(
    embedding_net_original: torch.nn.Module,
    embedding_net_finetuned: torch.nn.Module,
    npe: Estimator,
    val_loader,
    device: torch.device,
) -> list[Tensor]:
    """Execute one validation epoch for RoPE embedding fine-tuning.

    Args:
        embedding_net_original: Frozen original embedding network from NPE.
        embedding_net_finetuned: Fine-tuned embedding network being evaluated.
        npe: NPE estimator for data rescaling.
        val_loader: DataLoader for validation batches.
        device: PyTorch device for computation.

    Returns:
        List of loss tensors for each validation batch.
    """
    embedding_net_finetuned.eval()
    losses = []

    with torch.no_grad():
        for theta_batch, x_batch, y_batch in val_loader:
            theta_batch = theta_batch.to(device)
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            # Rescale inputs using NPE scaling.
            _, x_rescaled = npe._auto_rescale(theta_batch, x_batch)
            _, y_rescaled = npe._auto_rescale(theta_batch, y_batch)

            # Compute embeddings.
            embedding_original = embedding_net_original(x_rescaled)
            embedding_finetuned = embedding_net_finetuned(y_rescaled)

            # MSE loss to align embeddings.
            loss = (embedding_finetuned - embedding_original).pow(2).mean()
            losses.append(loss)

    return losses


def get_or_train_flow_x(
    theta: Tensor,
    x: Tensor,
    y: Tensor,
    simulate: Callable | None,
    config: Dict,
    model_path: Path,
    device: torch.device,
    logger,
    logname: str,
    load: bool,
    save: bool,
    rescale: str,
    **training_params,
):
    """
    Train or load the x-flow (flow_matching_x). Handles simulation,
    repetition, training-space selection, loading, and saving.
    """
    epochs = training_params.get("epochs", 1000)
    batch_size = training_params.get("batch_size", 128)
    lr = training_params.get("lr", 1e-4)
    train_size = training_params.get("train_size", 0.8)
    max_patience = training_params.get("max_patience", 20)
    ncal = x.shape[0]
    # ---- Training / loading x-flow ---- #

    # ----- Repeat or simulate training data -----
    if simulate is None:
        theta_repeated = theta
        x_train = x
        y_train = y
        epochs_x = epochs
    else:
        n_repeat = 10
        perm = torch.randperm(x.shape[0] * n_repeat)
        theta_repeated = theta.repeat_interleave(n_repeat, dim=0)[perm]
        x_train = simulate(theta_repeated)
        y_train = y.repeat_interleave(n_repeat, dim=0)[perm]
        epochs_x = epochs // 10

    # ----- Choose training space (data or latent) -----
    space = config["space"]  # "data" or "latent"
    if space == "latent":
        raise NotImplementedError("Latent space training not implemented in this snippet")

    # ----- Load path -----
    saved_path = model_path / f"flow_x_y_{space}_{ncal}.pth"

    # ----- Attempt loading -----
    if saved_path.exists():
        print(f"\tFound existing flow matching model from y to x in {space} for ncal= {ncal}")

        flow_matching_x = FlowMatching(
            config["probability_path"],
            config["prior"],
            config["base_dist"],
            x_train.shape[1:],
            cond_dim=y.shape[1:],
            **config["params"],
        )

        flow_matching_x.load_state_dict(torch.load(saved_path, map_location=device))
        flow_matching_x.rescale_name = rescale
        return flow_matching_x

    # ----- Otherwise, train -----
    flow_matching_x = train_flow_matching(
        target=x_train,
        cond=y_train,
        config=config,
        device=device,
        logger=logger,
        logname=logname + "_x",
        load=load,
        save=save,
        model_path=model_path,
        epochs=epochs_x,
        lr=lr,
        batch_size=batch_size,
        train_size=train_size,
        max_patience=max_patience,
        rescale=rescale,
        use_pretrained=False,
    )

    # Save model
    torch.save(flow_matching_x.state_dict(), saved_path)

    return flow_matching_x


def _train_loop_flow_theta_x(
    flow_matching_theta: FlowMatching,
    flow_matching_x: Optional[FlowMatching],
    npe,
    train_loader,
    val_loader,
    simulate,
    device,
    optimizer,
    scheduler,
    epochs,
    max_patience,
    train_flow_x,
    timer,
    model_path,
    logname="flow",
    logger=None,
    save=True,
    desc="Training coupled flow-matching models",
):
    # ----------------------------------------
    # Helper functions
    # ----------------------------------------
    def sample_x_given_y(flow, y):
        if flow is None:
            return y
        source = flow.sample_base(y).to(device)
        with timer.time_operation("training", "sample_x_flow") as ctx:
            xp, nfe = flow.sample(
                source, y, device, only_last=True, return_nfe=True
            )
            ctx.add_nfe(nfe)
        return xp.to(device)

    def sample_theta_given_x(x):
        with timer.time_operation("training", "sample_npe"):
            th = npe.sample(x, 1, device, show_progress_bars=False)
        return th.squeeze(0).to(device)

    def compute_flow_loss(flow, source, target, cond):
        return flow.compute_loss_with_source(source, target, cond)

    def maybe_simulate(theta, x):
        return simulate(theta) if simulate is not None else x

    use_flow_x = train_flow_x and flow_matching_x is not None
    lambda_ = 0.5 if use_flow_x else None

    best_loss = float("inf")
    patience = 0

    for epoch in (bar := trange(epochs, desc=desc)):
        flow_matching_theta.train()
        if use_flow_x:
            flow_matching_x.train()

        loss_train = []

        # -------------------------
        # TRAIN LOOP
        # -------------------------
        with timer.time_operation("training", "epoch"):
            for thb, xb, yb in train_loader:
                xb = maybe_simulate(thb, xb).to(device)
                thb = thb.to(device)
                yb = yb.to(device)

                # Sample x|y
                x_pred = sample_x_given_y(flow_matching_x, yb)

                # Sample theta|x
                source_theta = sample_theta_given_x(x_pred)

                # Loss θ
                loss_theta = compute_flow_loss(flow_matching_theta, source_theta, thb, yb)

                # Loss x
                if use_flow_x:
                    loss_x = flow_matching_x.compute_loss(xb, yb)
                    total_loss = lambda_ * loss_theta + (1 - lambda_) * loss_x
                else:
                    total_loss = loss_theta

                # Backprop
                with timer.time_operation("training", "gradient_step"):
                    optimizer.zero_grad()
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(flow_matching_theta.parameters(), 5.0)
                    if use_flow_x:
                        torch.nn.utils.clip_grad_norm_(flow_matching_x.parameters(), 5.0)
                    optimizer.step()

                loss_train.append(total_loss.detach())

        scheduler.step()

        # -------------------------
        # VALIDATION
        # -------------------------
        flow_matching_theta.eval()
        if use_flow_x:
            flow_matching_x.eval()

        loss_val = []
        with torch.no_grad():
            for thb, xb, yb in val_loader:
                xb = maybe_simulate(thb, xb).to(device)
                thb = thb.to(device)
                yb = yb.to(device)

                x_pred = sample_x_given_y(flow_matching_x, yb)
                source_theta = sample_theta_given_x(x_pred)

                loss_theta = compute_flow_loss(flow_matching_theta, source_theta, thb, yb)

                if use_flow_x:
                    loss_x = flow_matching_x.compute_loss(xb, yb)
                    total_loss = lambda_ * loss_theta + (1 - lambda_) * loss_x
                else:
                    total_loss = loss_theta

                loss_val.append(total_loss.detach())

        # -------------------------
        # LOGGING / CHECKPOINTS
        # -------------------------
        train_l = torch.stack(loss_train).mean().item()
        val_l = torch.stack(loss_val).mean().item()
        bar.set_postfix(train=train_l, val=val_l)

        if logger:
            logger.log_metrics(
                {"train_loss": train_l, "val_loss": val_l, "epoch": epoch},
                log_name=logname,
            )

        if val_l < best_loss:
            best_loss = val_l
            patience = 0

            best_theta = deepcopy(flow_matching_theta.drift.state_dict())
            if use_flow_x:
                best_x = deepcopy(flow_matching_x.drift.state_dict())

            if save:
                torch.save(best_theta, model_path / f"{logname}_theta.pth")
                if use_flow_x:
                    torch.save(best_x, model_path / f"{logname}_x.pth")
        else:
            patience += 1

        if patience >= max_patience:
            bar.write("\tEarly stopping.")
            bar.write(f"\tBest validation loss: {best_loss:.2f}")
            break

    # Restore best drifts
    flow_matching_theta.drift.load_state_dict(best_theta)
    if use_flow_x:
        flow_matching_x.drift.load_state_dict(best_x)

    return flow_matching_theta, flow_matching_x


