import torch
import torchmetrics
import torch.nn.functional as F


def GpromptEva(loader, gnn, prompt, center_embedding, num_class, device):
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
            out = gnn(batch.x, batch.edge_index, batch.batch, prompt, 'Gprompt')
            similarity_matrix = F.cosine_similarity(out.unsqueeze(1), center_embedding.unsqueeze(0), dim=-1)
            pred = similarity_matrix.argmax(dim=1)
            acc = accuracy(pred, batch.y)
            ma_f1 = macro_f1(pred, batch.y)
            roc = auroc(similarity_matrix, batch.y)
            prc = auprc(similarity_matrix, batch.y)
            if len(loader) > 20:
                print("Batch {}/{} Acc: {:.4f} | Macro-F1: {:.4f}| AUROC: {:.4f}| AUPRC: {:.4f}".format(batch_id,
                                                                                                        len(loader),
                                                                                                        acc.item(),
                                                                                                        ma_f1.item(),
                                                                                                        roc.item(),
                                                                                                        prc.item()))

            # print(acc)
    acc = accuracy.compute()
    ma_f1 = macro_f1.compute()
    roc = auroc.compute()
    prc = auprc.compute()

    return acc.item(), ma_f1.item(), roc.item(), prc.item()
