# eval.py

import torch
import torch.nn.functional as F

@torch.no_grad()
def evaluate(model, data, split_idx, eval_func, criterion, args=None, result=None, device=None):
    model.eval()

    if device is not None:
        model = model.to(device)
        data = data.to(device)

    out = result if result is not None else model(data.x, data.edge_index)

    y = data.y
    if y.dim() > 1:
        y = y.view(-1)
    y = y.to(torch.long)

    # --- PATCH: move indices to out.device ---
    idx_train = split_idx["train"].to(out.device)
    idx_valid = split_idx["valid"].to(out.device)
    idx_test  = split_idx["test"].to(out.device)

    train_acc = eval_func(y[idx_train], out[idx_train])
    valid_acc = eval_func(y[idx_valid], out[idx_valid])
    test_acc  = eval_func(y[idx_test],  out[idx_test])

    if isinstance(criterion, torch.nn.NLLLoss):
        out_for_loss = F.log_softmax(out, dim=1)
        valid_loss = criterion(out_for_loss[idx_valid], y[idx_valid])
        out_return = out_for_loss
    else:
        valid_loss = criterion(out[idx_valid], y[idx_valid])
        out_return = out

    return train_acc, valid_acc, test_acc, valid_loss, out_return


@torch.no_grad()
def evaluate_cpu(model, data, split_idx, eval_func, criterion, args=None, result=None, device=None):
    model_was_device = next(model.parameters()).device

    model = model.to(torch.device("cpu"))
    data_cpu = data.to(torch.device("cpu"))

    out = result if result is not None else model(data_cpu.x, data_cpu.edge_index)

    y = data_cpu.y
    if y.dim() > 1:
        y = y.view(-1)
    y = y.to(torch.long)

    # --- PATCH: ensure indices are CPU ---
    idx_train = split_idx["train"].to(torch.device("cpu"))
    idx_valid = split_idx["valid"].to(torch.device("cpu"))
    idx_test  = split_idx["test"].to(torch.device("cpu"))

    train_acc = eval_func(y[idx_train], out[idx_train])
    valid_acc = eval_func(y[idx_valid], out[idx_valid])
    test_acc  = eval_func(y[idx_test],  out[idx_test])

    if isinstance(criterion, torch.nn.NLLLoss):
        out_for_loss = F.log_softmax(out, dim=1)
        valid_loss = criterion(out_for_loss[idx_valid], y[idx_valid])
        out_return = out_for_loss
    else:
        valid_loss = criterion(out[idx_valid], y[idx_valid])
        out_return = out

    # Restore model
    if device is not None:
        model = model.to(device)
    else:
        model = model.to(model_was_device)

    return train_acc, valid_acc, test_acc, valid_loss, out_return
