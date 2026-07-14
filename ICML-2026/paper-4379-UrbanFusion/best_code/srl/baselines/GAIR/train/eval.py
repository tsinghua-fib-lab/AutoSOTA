"""
Original code from: https://github.com/VicenteVivan/geo-clip

Original source:
Vivanco, Vicente; Nayak, Gaurav Kumar; Shah, Mubarak.
"GeoCLIP: CLIP-Inspired Alignment between Locations and Images for Effective Worldwide Geo-localization."
NeurIPS 2023. arXiv preprint published September 27, 2023.
"""

import numpy as np
import pandas as pd
import torch
from geopy.distance import geodesic as GD
from tqdm import tqdm


def distance_accuracy(targets, preds, dis=2500, gps_gallery=None):
    total = len(targets)
    correct = 0
    gd_avg = 0

    for i in range(total):
        gd = GD(gps_gallery[preds[i]], targets[i]).km
        gd_avg += gd
        if gd <= dis:
            correct += 1

    gd_avg /= total
    return correct / total, gd_avg


def eval_images(val_dataloader, model, device="cpu"):
    model.eval()
    preds = []
    targets = []

    gps_gallery = model.gps_gallery

    with torch.no_grad():
        for imgs, labels in tqdm(val_dataloader, desc="Evaluating"):
            labels = labels.cpu().numpy()
            imgs = imgs.to(device)

            # Get predictions (probabilities for each location based on similarity)
            logits_per_image = model(imgs, gps_gallery)
            probs = logits_per_image.softmax(dim=-1)

            # Predict gps location with the highest probability (index)
            outs = torch.argmax(probs, dim=-1).detach().cpu().numpy()

            preds.append(outs)
            targets.append(labels)

    preds = np.concatenate(preds, axis=0)
    targets = np.concatenate(targets, axis=0)

    model.train()

    distance_thresholds = [2500, 750, 200, 25, 1]  # km
    accuracy_results = {}
    for dis in distance_thresholds:
        acc, avg_distance_error = distance_accuracy(
            targets, preds, dis, gps_gallery
        )
        print(
            f"Accuracy at {dis} km: {acc}, Average Distance Error: {avg_distance_error}"
        )
        accuracy_results[f"acc_{dis}_km"] = acc

    return accuracy_results
