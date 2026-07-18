from torchmetrics import Metric
from torchmetrics import Accuracy
from torchmetrics.classification import MulticlassRecall
from torch import nn
import torch


class MultiLabelAcc(Metric):
    def __init__(self, num_classes_per_label: int, device=None):
        super().__init__()
        self.acc = nn.ModuleList([Accuracy(num_classes=num_classes,task="multiclass") for num_classes in num_classes_per_label])
        if device is not None:
            self.acc = self.acc.to(device)
        self.num_classes_per_label = num_classes_per_label

    def update(self,preds, target):
        for i in range(len(self.num_classes_per_label)):
            self.acc[i].update(preds[i], target[:,i])

    def compute(self):
        return [acc.compute() for acc in self.acc]
    
    def reset(self):
        for acc in self.acc:
            acc.reset()


class MultiLabelRecall(Metric):
    def __init__(self, num_classes_per_label: list[int], average='none', device=None):
        """
        Args:
            num_classes_per_label: List of number of classes for each label.
            average: 'none', 'macro', 'micro', or 'weighted'. 
                     Use 'none' to get recall for EACH class separately (recommended for imbalance).
        """
        super().__init__()
        self.recalls = nn.ModuleList([
            MulticlassRecall(num_classes=num_classes, average=average) 
            for num_classes in num_classes_per_label
        ])
        
        if device is not None:
            self.recalls = self.recalls.to(device)
            
        self.num_classes_per_label = num_classes_per_label

    def update(self, preds, target):
        """
        preds: List of logits [batch, num_classes] for each label
        target: Tensor [batch, num_labels]
        """
        for i in range(len(self.num_classes_per_label)):
            self.recalls[i].update(preds[i], target[:, i])

    def compute(self):
        """
        Returns a list of tensors.
        If average='none': Returns [Tensor(recall_class0, recall_class1), ...]
        If average='macro': Returns [Tensor(scalar_recall), ...]
        """
        return [recall.compute() for recall in self.recalls]

    def reset(self):
        for recall in self.recalls:
            recall.reset()


class MultiLabelGroupAcc(Metric):
    def __init__(self, num_classes_per_label):
        super().__init__()
        self.num_classes_per_label = num_classes_per_label
        n = len(num_classes_per_label)

        # Register states AS TENSORS
        self.add_state("correct", default=torch.zeros(n, dtype=torch.long), dist_reduce_fx="sum")
        self.add_state("total",   default=torch.zeros(n, dtype=torch.long), dist_reduce_fx="sum")

    def update(self, logits, target):
        """
        logits: [B, sum(C_k)]
        target: [B, sum(C_k)]
        """
        B = logits.size(0)

        offset = 0
        for i, n_cls in enumerate(self.num_classes_per_label):
            log_i = logits[:, offset:offset+n_cls]   # [B, n_cls]
            tgt_i = target[:, offset:offset+n_cls]   # [B, n_cls]

            pred = log_i.argmax(dim=1)               # [B]
            true = tgt_i.argmax(dim=1)               # [B]

            # increment tensor states
            self.correct[i] += (pred == true).sum()
            self.total[i] += B

            offset += n_cls

    def compute(self):
        return self.correct.float() / self.total.float()
