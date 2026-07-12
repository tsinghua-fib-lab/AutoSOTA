"""
Copyright 2020 Twitter, Inc.
SPDX-License-Identifier: Apache-2.0
"""
import torch


@torch.no_grad()
def test(model, x, data, logits=None, evaluator=None, inference_loader=None, device="cuda"):
    if logits is None:
        model.eval()
        logits = (
            inference_sampled(model, x, inference_loader, device)
            if inference_loader
            else inference_full_batch(model, x, data.edge_index)
        )

    accs = []
    for _, mask in data("train_mask", "val_mask", "test_mask"):
        pred = logits[mask].max(1)[1]
        if evaluator:
            acc = evaluator.eval({"y_true": data.y[mask], "y_pred": pred.unsqueeze(1)})["acc"]
        else:
            acc = pred.eq(data.y[mask].squeeze()).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs, logits


def inference_full_batch(model, x, edge_index):
    out = model(x, edge_index)

    return out


def inference_sampled(model, x, inference_loader, device):
    return model.inference(x, inference_loader, device)
