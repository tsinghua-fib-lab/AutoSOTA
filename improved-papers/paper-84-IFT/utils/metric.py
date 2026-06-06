from environment import *


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def RSE(pred, true):
    return np.sqrt(np.sum((pred - true) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))


def CORR(pred, true):
    pred_term = pred - pred.mean(0)
    true_term = true - true.mean(0)
    u = (pred_term * true_term).sum(0)
    d = np.sqrt((pred_term ** 2 * true_term ** 2).sum(0))
    return (u / d).mean(-1)


class Metric:
    def __init__(self):
        self.mse, self.mae, self.rse = MSE, MAE, RSE
        self.rmse, self.mape, self.mspe = RMSE, MAPE, MSPE
        self.corr = CORR

    def __call__(self, pred, true):
        mse = self.mse(pred, true)
        mae = self.mae(pred, true)
        rse = self.rse(pred, true)

        rmse = self.rmse(pred, true)
        mape = self.mape(pred, true)
        mspe = self.mspe(pred, true)

        return {'MSE': mse, 'MAE': mae, 'RSE': rse, 'RMSE': rmse, 'MAPE': mape, 'MSPE': mspe}
