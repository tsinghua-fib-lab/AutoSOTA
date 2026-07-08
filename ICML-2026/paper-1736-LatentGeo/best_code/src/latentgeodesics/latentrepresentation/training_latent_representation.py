import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False

from .network import setup_latent_network
from .utils import compute_perturbed_points, L2


def train_latent_representation(
    dataloader,
    save_path,
    args,
    load_path=None,
    load_file_name=None,
    start_epoch=0,
    end_epoch=None,
    device=None,
    dataloader_test=None,
    eval_every=10,
    save_every=None,
):
    if load_path is not None or start_epoch > 0:
        if load_path is None:
            load_path = save_path
        load_path = Path(load_path)
        if args is None:
            with open(load_path / "args.json", "r") as f:
                args = json.load(f)
        if end_epoch is not None:
            args["train"]["epochs"] = end_epoch

    if save_path is not None:
        save_path = Path(save_path)

        with open(f"{save_path}/args.json", "w") as f:
            json.dump(args, f, indent=4)
    else:
        ValueError("save path not correctly set up")


    # process parameters
    if save_every is None:
        save_every = args["train"]["epochs"] // 5
    if end_epoch is None:
        end_epoch = args["train"]["epochs"]

    scheduler_step = args["train"].get("scheduler_step", 1000)
    
    network = setup_latent_network(args["architecture"]).to(device)
    
    parameters = network.parameters()
    optimizer = torch.optim.Adam(
        parameters, lr=args["train"]["lr"], weight_decay=args["train"]["weight_decay"]
    )
    # decay per epoch
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99) 
    
    # loading for continued training
    if start_epoch > 0:
        if load_file_name is None:
            load_file_name = f"state_dict_{start_epoch}.pth"
        state_dict = torch.load(load_path / load_file_name)
        network.load_state_dict(state_dict)
        try:
            optimizer_state_dict = torch.load(load_path / f"opt_{load_file_name}")
        except FileNotFoundError as e:
            optimizer_state_dict = None
            print(e)
        if optimizer_state_dict is not None:
            optimizer.load_state_dict(optimizer_state_dict)

    if TENSORBOARD_AVAILABLE:            
        writer = SummaryWriter(save_path)
    last_time = time.time()
    min_loss = np.inf
    best_epoch = -1


    # define sampling of points
    std = args["train"]["sampling"]["std"]
    std = std * dataloader.dataset.reach


    for epoch in range(start_epoch, args["train"]["epochs"]):
        network.train()
        epoch_losses = defaultdict(lambda: 0)

        # console output
        if time.time() - last_time >= 60 or epoch == 0:
            print(f'{epoch}/{args["train"]["epochs"]}')
            last_time = time.time()

        for points in dataloader:

            perturbed_points = compute_perturbed_points(points, std)
            points_projected = network(perturbed_points)
            loss = L2(points-points_projected)

            # optimization step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch % eval_every == 0:
                with torch.no_grad():
                    epoch_losses["loss_manifold"] += L2(
                        network.latent_space_deviation(points)).item()
                    epoch_losses["loss"] += loss.item()

        if epoch % scheduler_step == 0:
            scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        global_step = epoch
        if TENSORBOARD_AVAILABLE:
            writer.add_scalar("LearningRate", current_lr, global_step)

        if epoch % eval_every == 0:
            with torch.no_grad():
                # evaluate on test set
                epoch_losses_test = defaultdict(lambda: 0)
                if dataloader_test is not None:
                    for points in dataloader_test:
                        epoch_losses_test["loss_manifold"] += L2(
                            network.latent_space_deviation(points)
                        ).item()
                        perturbed_points = compute_perturbed_points(points, std)
                        points_projected = network(perturbed_points)
                        epoch_losses_test["loss"] += L2(points-points_projected).item()
                else:
                    #track training loss if no test set is provided
                    epoch_losses_test["loss_manifold"] = epoch_losses["loss_manifold"]
                    epoch_losses_test["loss"] = epoch_losses["loss"]
            # tracking
            if TENSORBOARD_AVAILABLE:
                if dataloader_test is not None:
                    for key, value in epoch_losses_test.items():
                        writer.add_scalar(
                            str(key) + "_test", value / dataloader_test.num, global_step=epoch
                        )

                for key, value in epoch_losses.items():
                    writer.add_scalar(key, value / dataloader.num, global_step=epoch)

            #save best epoch
            if epoch_losses_test["loss"] < min_loss:
                best_epoch = epoch
                torch.save(network.state_dict(), f=f"{save_path}/state_dict_best.pth")
                torch.save(
                    optimizer.state_dict(), f"{save_path}/opt_state_dict_best.pth"
                )
                min_loss = epoch_losses_test["loss"]

        if epoch % save_every == 0:
            torch.save(network.state_dict(), f=f"{save_path}/state_dict_{epoch}.pth")
            torch.save(
                optimizer.state_dict(), f"{save_path}/opt_state_dict_{epoch}.pth"
            )


    if save_path is not None:
        torch.save(network.state_dict(), f=f"{save_path}/state_dict_final.pth")
        torch.save(optimizer.state_dict(), f"{save_path}/opt_state_dict_final.pth")
        with open(f"{save_path}/best.json", "w") as f:
            json.dump({"best epoch": best_epoch}, f, indent=4)
    return network
