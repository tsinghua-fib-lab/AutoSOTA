import copy
from timeit import default_timer as timer

import numpy as np
import torch
from torch import nn
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score


class Linear(nn.Linear):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        bias = self.bias
        if (
            bias is not None
            and hasattr(self, "scale_bias")
            and self.scale_bias is not None
        ):
            bias = self.scale_bias * bias

        out = torch.nn.functional.linear(
            input,
            self.weight,
            bias,
        )
        return out

    def fit(
        self,
        Xtr: torch.Tensor,
        ytr: torch.Tensor,
        criterion: torch.nn.Module,
        reg: float = 0.0,
        epochs: int = 100,
        optimizer: torch.optim.Optimizer | None = None,
        device: str = "cuda",
    ) -> None:
        if optimizer is None:
            optimizer = torch.optim.LBFGS(self.parameters(), lr=1.0, history_size=10)
        if self.bias is not None:
            scale_bias = (Xtr**2).mean(-1).sqrt().mean().item()
            self.scale_bias = scale_bias
        self.train()
        self.to(device)
        Xtr = Xtr.to(device)
        ytr = ytr.to(device)

        def closure():
            optimizer.zero_grad()
            output = self(Xtr)
            loss = criterion(output, ytr)
            loss = loss + 0.5 * reg * self.weight.pow(2).sum()
            loss.backward()
            return loss

        for epoch in range(epochs):
            optimizer.step(closure)
        if self.bias is not None:
            self.bias.data.mul_(self.scale_bias)
        self.scale_bias = None

    @torch.no_grad()
    def score(self, X: torch.Tensor, y: torch.Tensor) -> float:
        self.eval()
        scores = self(X)
        scores = scores.argmax(-1)
        scores = scores.cpu()
        return torch.mean((scores == y).float()).item()


def accuracy(
    output: torch.Tensor,
    target: torch.Tensor,
    topk: tuple[int, ...] = (1,),
) -> list[float]:
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size).item())
    return res


def train_and_eval_linear(
    X_tr: torch.Tensor,
    y_tr: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    X_te: torch.Tensor | list[torch.Tensor],
    y_te: torch.Tensor | list[torch.Tensor],
    num_class: int,
    device: str = "cuda",
) -> tuple[float, list[list[float]]]:
    """Train a linear classifier with L-BFGS and evaluate on test sets.

    Performs cross-validation over a regularisation grid
    ``[100, 10, 1, 0.1, 0.01, 0.001]`` (as described in Appendix B.3 of the
    paper), selects the best ``α`` by validation accuracy, then reports top-1
    accuracy and macro F1 on each test set.

    Args:
        X_tr: Training representations ``(N_tr, D)``.
        y_tr: Training labels ``(N_tr,)``.
        X_val: Validation representations ``(N_val, D)``.
        y_val: Validation labels ``(N_val,)``.
        X_te: Test representations — either a single tensor ``(N_te, D)`` or a
            list of tensors for multiple test sets (e.g. TEDBench + CATH 4.4).
        y_te: Test labels matching ``X_te``.
        num_class: Number of output classes (965 for TEDBench).
        device: Torch device string for training (default ``"cuda"``).

    Returns:
        Tuple ``(val_score, test_scores)`` where ``val_score`` is the best
        validation accuracy (float) and ``test_scores`` is a list of
        ``[top1, top5, top10]`` accuracy lists for each test set.
    """
    embed_dim = X_tr.shape[1]
    search_grid = 10.0 ** np.arange(-2, 4)
    search_grid = 1.0 / search_grid
    best_score = -np.inf
    clf = Linear(embed_dim, num_class)
    criterion = torch.nn.CrossEntropyLoss(reduction="sum")
    if X_tr.shape[1] > 20000:
        optimizer = torch.optim.Adam(clf.parameters(), lr=0.01)
        epochs = 800
    else:
        optimizer = torch.optim.LBFGS(
            clf.parameters(),
            lr=0.1,
            max_eval=20,
            history_size=20,
        )
        epochs = 1000
    torch.cuda.empty_cache()
    print("Start cross validation")
    for alpha in search_grid:
        tic = timer()
        clf.fit(
            X_tr,
            y_tr,
            criterion,
            reg=alpha,
            epochs=epochs,
            optimizer=optimizer,
            device=device,
        )
        toc = timer()
        X_val = X_val.to(device)
        score = clf.score(X_val, y_val)
        print(
            "CV alpha={}, acc={:.2f}, ts={:.2f}s".format(
                alpha, score * 100.0, toc - tic
            )
        )
        if score > best_score:
            best_score = score
            best_alpha = alpha
            best_weight = copy.deepcopy(clf.state_dict())

    clf.load_state_dict(best_weight)

    print("Finished, elapsed time: {:.2f}s".format(toc - tic))

    if not isinstance(X_te, list):
        X_te = [X_te]
        y_te = [y_te]

    scores_all = []
    for X_te_i, y_te_i in zip(X_te, y_te):
        X_te_i = X_te_i.to(device)
        with torch.no_grad():
            y_pred = clf(X_te_i).cpu()

        scores = accuracy(y_pred, y_te_i, (1, 5, 10))
        print(scores)
        scores_all.append(scores)
        metric_fn = MetricCollection(
            {
                "balanced_acc": MulticlassAccuracy(
                    num_classes=num_class, average="macro"
                ),
                "macro_f1": MulticlassF1Score(num_classes=num_class, average="macro"),
            }
        )
        metrics = metric_fn(y_pred, y_te_i)
        metrics = {k: v.item() for k, v in metrics.items()}
        print(metrics)

    return best_score, scores_all
