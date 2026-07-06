import torch
import torchmetrics


def AllInOneEva(loader, prompt, gnn, answering, num_class, device):
    prompt.eval()
    answering.eval()

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
            prompted_graph = prompt(batch)
            graph_emb = gnn(prompted_graph.x, prompted_graph.edge_index, prompted_graph.batch)
            pre = answering(graph_emb)
            pred = pre.argmax(dim=1)

            acc = accuracy(pred, batch.y)
            ma_f1 = macro_f1(pred, batch.y)
            roc = auroc(pre, batch.y)
            prc = auprc(pre, batch.y)
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


def AllInOneEvaWithoutAnswer(loader, prompt, gnn, num_class, device):
    prompt.eval()
    accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_class).to(device)
    macro_f1 = torchmetrics.classification.F1Score(task="multiclass", num_classes=num_class, average="macro").to(device)
    accuracy.reset()
    macro_f1.reset()
    for batch_id, test_batch in enumerate(loader):
        test_batch = test_batch.to(device)
        emb0 = gnn(test_batch.x, test_batch.edge_index, test_batch.batch)
        pg_batch = prompt.token_view()
        pg_batch = pg_batch.to(device)
        pg_emb = gnn(pg_batch.x, pg_batch.edge_index, pg_batch.batch)
        dot = torch.mm(emb0, torch.transpose(pg_emb, 0, 1))
        pre = torch.softmax(dot, dim=1)

        y = test_batch.y
        pre_cla = torch.argmax(pre, dim=1)

        acc = accuracy(pre_cla, y)
        ma_f1 = macro_f1(pre_cla, y)

    acc = accuracy.compute()
    ma_f1 = macro_f1.compute()
    return acc
