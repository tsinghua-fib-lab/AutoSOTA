import torch
import torchmetrics as tm


def RSE(pred, true):
    return torch.sqrt(torch.sum((true - pred) ** 2)) / torch.sqrt(torch.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = torch.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return torch.mean(torch.abs(pred - true))


def MSE(pred, true):
    return torch.mean((pred - true) ** 2)


def RMSE(pred, true):
    return torch.sqrt((MSE(pred, true)))


def MAPE(pred, true):
    return torch.mean(torch.abs((pred - true) / true))


def MSPE(pred, true):
    return torch.mean(torch.square((pred - true) / true))


def metric_torch(pred, true):
    mae = MAE(pred, true).item()
    mse = MSE(pred, true).item()
    rmse = RMSE(pred, true).item()
    mape = MAPE(pred, true).item()
    mspe = MSPE(pred, true).item()

    return mae, mse, rmse, mape, mspe


class MeanAE(tm.Metric):
    is_differentiable: bool = True
    higher_is_better: bool = False
    full_state_update: bool = False

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.add_state("sum_abs_error", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds, target) -> None:
        """Update state with predictions and targets."""
        sum_abs_error = torch.sum(torch.abs(preds - target))
        num_obs = target.numel()
        self.sum_abs_error += sum_abs_error
        self.total += num_obs

    def compute(self):
        """Compute mean absolute error over state."""
        value = self.sum_abs_error / self.total
        return value.item()


class MeanSE(tm.Metric):
    is_differentiable = True
    higher_is_better = False
    full_state_update = False

    def __init__(self, squared=True, num_outputs=1, **kwargs) -> None:
        super().__init__(**kwargs)
        self.squared = squared
        self.num_outputs = num_outputs

        self.add_state("sum_squared_error", default=torch.zeros(num_outputs), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds, target) -> None:
        """Update state with predictions and targets."""
        preds, target = preds.view(-1), target.view(-1)
        sum_squared_error = torch.sum((preds - target) ** 2, dim=0)
        num_obs = target.shape[0]
        self.sum_squared_error += sum_squared_error
        self.total += num_obs

    def compute(self):
        """Compute mean squared error over state."""
        value = self.sum_squared_error / self.total if self.squared else torch.sqrt(self.sum_squared_error / self.total)
        return value.item()


class MAbsPE(tm.Metric):
    is_differentiable: bool = True
    higher_is_better: bool = False
    full_state_update: bool = False

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.add_state("sum_abs_per_error", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds, target) -> None:
        """Update state with predictions and targets."""
        abs_per_error = torch.abs((preds - target) / target)
        sum_abs_per_error = torch.sum(abs_per_error)
        num_obs = target.numel()
        self.sum_abs_per_error += sum_abs_per_error
        self.total += num_obs

    def compute(self):
        """Compute mean absolute percentage error over state."""
        value = self.sum_abs_per_error / self.total
        return value.item()


class MSqrPE(tm.Metric):
    is_differentiable: bool = True
    higher_is_better: bool = False
    full_state_update: bool = False

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.add_state("sum_sqr_per_error", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds, target) -> None:
        """Update state with predictions and targets."""
        sqr_per_error = torch.square((preds - target) / target)
        sum_sqr_per_error = torch.sum(sqr_per_error)
        num_obs = target.numel()
        self.sum_sqr_per_error += sum_sqr_per_error
        self.total += num_obs

    def compute(self):
        """Compute mean square percentage error over state."""
        value = self.sum_sqr_per_error / self.total
        return value.item()


def create_metric_collector(device='cpu'):
    collector = tm.MetricCollection({
        "mae": MeanAE(),
        "mse": MeanSE(),
        "rmse": MeanSE(squared=False),
        "mape": MAbsPE(),
        "mspe": MSqrPE()
    }).to(device)
    collector.reset()
    return collector