"""
Original code from: https://github.com/VicenteVivan/geo-clip

Original source:
Vivanco, Vicente; Nayak, Gaurav Kumar; Shah, Mubarak.
"GeoCLIP: CLIP-Inspired Alignment between Locations and Images for Effective Worldwide Geo-localization."
NeurIPS 2023. arXiv preprint published September 27, 2023.
"""

import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm


def train(
    train_dataloader,
    model,
    optimizer,
    epoch,
    batch_size,
    device,
    scheduler=None,
    criterion=nn.CrossEntropyLoss(),
):
    print("Starting Epoch", epoch)

    bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))

    targets_img_gps = (
        torch.Tensor([i for i in range(batch_size)]).long().to(device)
    )

    for i, (imgs, gps) in bar:
        imgs = imgs.to(device)
        gps = gps.to(device)
        gps_queue = model.get_gps_queue()

        optimizer.zero_grad()

        # Append GPS Queue & Queue Update
        gps_all = torch.cat([gps, gps_queue], dim=0)
        model.dequeue_and_enqueue(gps)

        # Forward pass
        logits_img_gps = model(imgs, gps_all)

        # Compute the loss
        img_gps_loss = criterion(logits_img_gps, targets_img_gps)
        loss = img_gps_loss

        # Backpropagate
        loss.backward()
        optimizer.step()

        bar.set_description(f"Epoch {epoch} loss: {loss.item():.5f}")

    if scheduler is not None:
        scheduler.step()
