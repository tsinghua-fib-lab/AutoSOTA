import numpy as np
import os
from collections import OrderedDict, defaultdict
import torch
from sklearn.metrics import f1_score, confusion_matrix
from scipy.sparse import coo_matrix
from scipy.stats import hmean, gmean


class Evaluator:
    """Evaluator for classification."""

    def __init__(self, cfg, many_idxs=None, med_idxs=None, few_idxs=None):
        self.cfg = cfg
        self.many_idxs = many_idxs
        self.med_idxs = med_idxs
        self.few_idxs = few_idxs
        self.reset()

    def reset(self):
        self._correct = 0
        self._correct5 = 0
        self._total = 0
        self._y_true = []
        self._y_pred = []
        self._y_conf = []  # Store prediction confidences

    def process(self, mo, gt):
        # mo (torch.Tensor): model output [batch, num_classes]
        # gt (torch.LongTensor): ground truth [batch]
        pred = mo.max(1)[1]
        conf = torch.softmax(mo, dim=1).max(1)[0]  # Compute prediction confidences
        matches = pred.eq(gt).float()
        self._correct += int(matches.sum().item())
        self._total += gt.shape[0]

        _, pred2 = mo.topk(5, dim=1, largest=True, sorted=True)
        pred2 = pred2.t()
        matches = pred2.eq(gt.view(1, -1).expand_as(pred2)).float()
        self._correct5 += int(matches.sum().item())

        self._y_true.extend(gt.data.cpu().numpy().tolist())
        self._y_pred.extend(pred.data.cpu().numpy().tolist())
        self._y_conf.extend(conf.data.cpu().numpy().tolist())

    def evaluate(self):
        results = OrderedDict()
        acc = 100.0 * self._correct / self._total
        acc5 = 100.0 * self._correct5 / self._total
        err = 100.0 - acc
        macro_f1 = 100.0 * f1_score(
            self._y_true,
            self._y_pred,
            average="macro",
            labels=np.unique(self._y_true)
        )

        # The first value will be returned by trainer.test()
        results["accuracy"] = acc
        results["accuracy5"] = acc5
        results["error_rate"] = err
        results["macro_f1"] = macro_f1

        print(
            "=> result\n"
            f"* total: {self._total:,}\n"
            f"* correct: {self._correct:,}\n"
            f"* accuracy: {acc:.1f}%\n"
            f"* accuracy5: {acc5:.1f}%\n"
            f"* error: {err:.1f}%\n"
            f"* macro_f1: {macro_f1:.1f}%"
        )

        self._per_class_res = defaultdict(list)

        for label, pred in zip(self._y_true, self._y_pred):
            matches = int(label == pred)
            self._per_class_res[label].append(matches)

        labels = list(self._per_class_res.keys())
        labels.sort()

        cls_accs = []
        for label in labels:
            res = self._per_class_res[label]
            correct = sum(res)
            total = len(res)
            acc = 100.0 * correct / total
            cls_accs.append(acc)
        
        accs_string = np.array2string(np.array(cls_accs), precision=2)
        print(f"* class acc: {accs_string}")

        # Compute worst case accuracy
        worst_case_acc = min([acc for acc in cls_accs])

        # Compute lowest recall
        # lowest_recall = min([100.0 * sum(res) / self.cls_num_list[label] for label, res in self._per_class_res.items()])

        # Compute harmonic mean
        hmean_acc = 100.0 / np.mean([1.0 / (max(acc, 0.001) / 100.0) for acc in cls_accs])

        # Compute geometric mean
        gmean_acc = 100.0 * np.prod([acc / 100.0 for acc in cls_accs]) ** (1.0 / len(cls_accs))

        results["worst_case_acc"] = worst_case_acc
        # results["lowest_recall"] = lowest_recall
        results["hmean_acc"] = hmean_acc
        results["gmean_acc"] = gmean_acc

        print(
            f"* worst_case_acc: {worst_case_acc:.1f}%\n"
            # f"* lowest_recall: {lowest_recall:.1f}%\n"
            f"* hmean_acc: {hmean_acc:.1f}%\n"
            f"* gmean_acc: {gmean_acc:.1f}%"
        )

        if self.many_idxs is not None and self.med_idxs is not None and self.few_idxs is not None:
            many_acc = np.mean(np.array(cls_accs)[self.many_idxs])
            med_acc = np.mean(np.array(cls_accs)[self.med_idxs])
            few_acc = np.mean(np.array(cls_accs)[self.few_idxs])
            results["many_acc"] = many_acc
            results["med_acc"] = med_acc
            results["few_acc"] = few_acc
            print(f"* many: {many_acc:.1f}%  med: {med_acc:.1f}%  few: {few_acc:.1f}%")

        mean_acc = np.mean(cls_accs)
        results["mean_acc"] = mean_acc
        print(f"* average: {mean_acc:.1f}%")

        # Compute expected calibration error
        # ece = 100.0 * expected_calibration_error(
        #     self._y_conf,
        #     self._y_pred,
        #     self._y_true
        # )
        # results["expected_calibration_error"] = ece
        # print(f"* expected_calibration_error: {ece:.2f}%")

        # Compute confusion matrix
        # cmat = confusion_matrix(self._y_true, self._y_pred)
        # cmat = coo_matrix(cmat)
        # save_path = os.path.join(self.cfg.output_dir, "cmat.pt")
        # torch.save(cmat, save_path)
        # print(f"Confusion matrix is saved to {save_path}")

        return results


def compute_accuracy(output, target, topk=(1, )):
    """Computes the accuracy over the k top predictions for
    the specified values of k.

    Args:
        output (torch.Tensor): prediction matrix with shape (batch_size, num_classes).
        target (torch.LongTensor): ground truth labels with shape (batch_size).
        topk (tuple, optional): accuracy at top-k will be computed. For example,
            topk=(1, 5) means accuracy at top-1 and top-5 will be computed.

    Returns:
        list: accuracy at top-k.
    """
    maxk = max(topk)
    batch_size = target.size(0)

    if isinstance(output, (tuple, list)):
        output = output[0]

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        acc = correct_k.mul_(100.0 / batch_size)
        res.append(acc)

    return res


def expected_calibration_error(confs, preds, labels, num_bins=10):
    def _populate_bins(confs, preds, labels, num_bins):
        bin_dict = defaultdict(lambda: {'bin_accuracy': 0, 'bin_confidence': 0, 'count': 0})
        bins = np.linspace(0, 1, num_bins + 1)
        for conf, pred, label in zip(confs, preds, labels):
            bin_idx = np.searchsorted(bins, conf) - 1
            bin_dict[bin_idx]['bin_accuracy'] += int(pred == label)
            bin_dict[bin_idx]['bin_confidence'] += conf
            bin_dict[bin_idx]['count'] += 1
        return bin_dict

    bin_dict = _populate_bins(confs, preds, labels, num_bins)
    num_samples = len(labels)
    ece = 0
    for i in range(num_bins):
        bin_accuracy = bin_dict[i]['bin_accuracy']
        bin_confidence = bin_dict[i]['bin_confidence']
        bin_count = bin_dict[i]['count']
        ece += (float(bin_count) / num_samples) * \
               abs(bin_accuracy / bin_count - bin_confidence / bin_count)
    return ece
