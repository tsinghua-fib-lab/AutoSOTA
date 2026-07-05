import torch
import torchmetrics


def GPPTEva(data, idx_test, gnn, prompt, num_class, device):
    # gnn.eval()
    prompt.eval()
    accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_class).to(device)
    macro_f1 = torchmetrics.classification.F1Score(task="multiclass", num_classes=num_class, average="macro").to(device)
    auroc = torchmetrics.classification.AUROC(task="multiclass", num_classes=num_class).to(device)
    auprc = torchmetrics.classification.AveragePrecision(task="multiclass", num_classes=num_class).to(device)

    accuracy.reset()
    macro_f1.reset()
    auroc.reset()
    auprc.reset()

    node_embedding = gnn(data.x, data.edge_index)
    out = prompt(node_embedding, data.edge_index)
    pred = out.argmax(dim=1)

    acc = accuracy(pred[idx_test], data.y[idx_test])
    f1 = macro_f1(pred[idx_test], data.y[idx_test])
    roc = auroc(out[idx_test], data.y[idx_test])
    prc = auprc(out[idx_test], data.y[idx_test])
    return acc.item(), f1.item(), roc.item(), prc.item()


def GPPTGraphEva(loader, gnn, prompt, num_class, device):
    # batch must be 1
    prompt.eval()
    accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_class).to(device)
    macro_f1 = torchmetrics.classification.F1Score(task="multiclass", num_classes=num_class, average="macro").to(device)
    auroc = torchmetrics.classification.AUROC(task="multiclass", num_classes=num_class).to(device)
    auprc = torchmetrics.classification.AveragePrecision(task="multiclass", num_classes=num_class).to(device)
    accuracy.reset()
    macro_f1.reset()
    auroc.reset()
    auprc.reset()
    with torch.no_grad():
        for batch_id, batch in enumerate(loader):
            batch = batch.to(device)
            node_embedding = gnn(batch.x, batch.edge_index)
            out = prompt(node_embedding, batch.edge_index)

            predicted_classes = out.argmax(dim=1)

            votes = predicted_classes.bincount(minlength=out.shape[1])

            pred = votes.argmax()
            pred = pred.unsqueeze(dim=-1)
            average_out = torch.nn.functional.softmax(votes.float(), dim=0).unsqueeze(dim=0)

            # correct += int((pred == batch.y).sum())  
            acc = accuracy(pred, batch.y)
            ma_f1 = macro_f1(pred, batch.y)
            roc = auroc(average_out, batch.y)
            prc = auprc(average_out, batch.y)
            if len(loader) > 20:
                print("Batch {}/{} Acc: {:.4f} | Macro-F1: {:.4f}| AUROC: {:.4f}| AUPRC: {:.4f}".format(batch_id,
                                                                                                        len(loader),
                                                                                                        acc.item(),
                                                                                                        ma_f1.item(),
                                                                                                        roc.item(),
                                                                                                        prc.item()))

    acc = accuracy.compute()
    ma_f1 = macro_f1.compute()
    roc = auroc.compute()
    prc = auprc.compute()

    return acc.item(), ma_f1.item(), roc.item(), prc.item()
