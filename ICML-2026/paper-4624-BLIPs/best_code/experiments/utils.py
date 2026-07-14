import os
import copy
import random
import math
import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils import clip_grad_norm_


########## Processing utils ##########
def save_checkpoint(model, optimizer, scheduler, epoch, val_loss, path):
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch,
            "val_loss": val_loss,
        },
        path,
    )


def load_checkpoint(path, model, device, optimizer=None, scheduler=None):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    start_epoch = checkpoint["epoch"] + 1
    best_val_loss = checkpoint["val_loss"]
    print(f"Loaded checkpoint from {path}")
    return start_epoch, best_val_loss


def directory_handler(dir):
    if not isinstance(dir, (list, tuple)):
        dir = [dir]
    for d in dir:
        assert isinstance(d, str)
        if not os.path.exists(d):
            os.makedirs(d, exist_ok=True)


def set_seed_precision(seed):
    torch.set_float32_matmul_precision("high")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


########## Model utils ##########
def choose_activation(activation):
    if activation == "relu":
        return nn.ReLU
    if activation == "silu":
        return nn.SiLU
    if activation == "tanh":
        return nn.Tanh
    raise NotImplementedError


########## Training utils ##########
def regression_loss(model, criterion, data):
    output = model(batch=data)
    return criterion(output, data.y)


def mlff_loss(model, criterion, data):
    output = model(batch=data)
    energy_loss = 0.1 * criterion(output["energy"], data["energy"])
    energy_grad_loss = criterion(output["energy_grad"], data["energy_grad"])
    return energy_loss + energy_grad_loss


def mlff_orb_train(model, criterion, data):
    output = model(batch=data)
    # compute loss as in ORB v3
    total_loss = torch.tensor(
        0.0,
        device=data.positions.device,
        dtype=data.positions.dtype,
    )
    if hasattr(model, "posterior"):
        state = model.model
    else:
        state = model
    for name, head in state.heads.items():
        if name == "confidence":
            continue
        head_out = head.loss(output[name], data)
        weight = state.loss_weights.get(name, 1.0)
        state_loss = weight * head_out.loss
        total_loss += state_loss
    return total_loss


def mlff_orb_validate(model, criterion, data):
    output = model.predict(batch=data)
    energy_loss = criterion(output["energy"], data.system_targets["energy"])
    force_loss = criterion(output["forces"], data.node_targets["forces"])
    return energy_loss + force_loss


def get_batch_size(data):
    if "y" in data and data["y"] is not None:
        return len(data["y"])
    elif "energy" in data and data["energy"] is not None:
        return len(data["energy"])
    elif "energy" in data.system_targets and data.system_targets["energy"] is not None:
        return len(data.system_targets["energy"])
    else:
        raise RuntimeError("batch size")


def get_edge_index(data):
    if "edge_index" in data and data["edge_index"] is not None:
        return data["edge_index"]
    elif "nbr_list" in data and data["nbr_list"] is not None:
        return data["nbr_list"].T
    elif hasattr(data, "senders") and data.senders is not None:
        return torch.stack([data.senders, data.receivers], dim=0)
    else:
        raise RuntimeError("edge index")


def train(
    model,
    loader,
    optimizer,
    scheduler,
    criterion,
    device,
    is_bayesian,
    loss="regression",
    clip=False,
    kl_scale=1.0,
):
    model.train()
    # chose loss
    if loss == "regression":
        loss_fn = regression_loss
    elif loss == "mlff":
        loss_fn = mlff_loss
    elif loss == "mlff_orb":
        loss_fn = mlff_orb_train
    else:
        raise NotImplementedError("Invalid loss function.")
    # initialize stats
    total_loss = 0
    total_kl = 0
    counter = 0
    # starts training
    for data in loader:
        data = data.to(device)
        batch_size = get_batch_size(data)
        optimizer.zero_grad()
        if is_bayesian:
            first_loss = loss_fn(model, criterion, data)
            kl = model.kl_loss(edge_index=get_edge_index(data))
            loss = first_loss + kl_scale * kl
        else:
            loss = loss_fn(model, criterion, data)
        loss.backward()
        if clip:
            clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        # logging
        counter += batch_size
        if is_bayesian:
            total_loss += (loss - kl_scale * kl).item() * batch_size
            total_kl += kl.item() * batch_size
        else:
            total_loss += loss.item() * batch_size
    # schedule
    try:
        scheduler.step()
    except:
        scheduler.step(total_loss / counter)
    if is_bayesian:
        return [total_loss / counter, total_kl / counter]
    return total_loss / counter


def validation(model, loader, criterion, device, loss=None):
    total_loss = 0
    counter = 0
    if loss == "regression":
        loss_fn = regression_loss
    elif loss == "mlff":
        loss_fn = mlff_loss
    elif loss == "mlff_orb":
        loss_fn = mlff_orb_validate
    else:
        raise NotImplementedError("Invalid loss function.")
    model.eval()
    for data in loader:
        data = data.to(device)
        batch_size = get_batch_size(data)
        loss = loss_fn(model, criterion, data)
        total_loss += loss.item() * batch_size
        counter += batch_size
    return total_loss / counter


def compute_kl_scale(epoch, warmup_epochs, total_epochs, min_scale=0.0, max_scale=1.0):
    """Linear warmup of KL weight from min_scale to max_scale."""
    if epoch < warmup_epochs:
        return min_scale + (max_scale - min_scale) * (epoch / warmup_epochs)
    return max_scale


def run(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    criterion,
    device,
    writer,
    max_epochs,
    ckpt_file=None,
    run_type="train",
    loss="regression",
    kl_warmup_epochs=0,
    clip_grad=False,
):
    # use bayesian training
    use_bayesian = hasattr(model, "posterior")
    clip = True if loss == "mlff_orb" else clip_grad
    if run_type == "train":
        start_epoch = 0
        best_val_loss = 1e20
        print("Training starts")
        print(
            f"Total number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
        )
        if use_bayesian and kl_warmup_epochs > 0:
            print(f"KL annealing: warmup over {kl_warmup_epochs} epochs")
        for epoch in range(start_epoch, max_epochs):
            # compute KL scale for annealing
            kl_scale = 1.0
            if use_bayesian and kl_warmup_epochs > 0:
                kl_scale = compute_kl_scale(epoch, kl_warmup_epochs, max_epochs)

            # train and validate
            train_loss = train(
                model,
                train_loader,
                optimizer,
                scheduler,
                criterion,
                device,
                use_bayesian,
                loss,
                clip,
                kl_scale=kl_scale,
            )
            val_loss = validation(model, val_loader, criterion, device, loss)
            if use_bayesian:
                writer.add_scalar("Train/loss", train_loss[0], epoch)
                writer.add_scalar("Train/kl", train_loss[1], epoch)
                writer.add_scalar("Train/kl_scale", kl_scale, epoch)
                writer.add_scalar("Val/loss", val_loss, epoch)
            else:
                writer.add_scalar("Train/loss", train_loss, epoch)
                writer.add_scalar("Val/loss", val_loss, epoch)
            # Save if val improves
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    val_loss,
                    ckpt_file,
                )
            if (epoch + 1) % 500 == 0:
                print(f"Epoch {epoch+1}/{max_epochs}: train_loss={train_loss[0] if use_bayesian else train_loss:.6f}, val_loss={val_loss:.6f}, best_val_loss={best_val_loss:.6f}")
        print(f"Training complete. Best val_loss={best_val_loss:.6f}")
    if run_type == "test":
        load_checkpoint(
            ckpt_file,
            model,
            device,
            optimizer,
            scheduler,
        )
