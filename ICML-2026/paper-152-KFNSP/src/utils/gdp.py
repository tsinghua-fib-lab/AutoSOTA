import torch
from torch import nn
from math import pi, sqrt


class kde_fair:
    """ TAKEN FROM https://github.com/zhimengj0326/GDP/blob/master/tabular/kde.py. UNMODIFIED
    A Gaussian KDE implemented in pytorch for the gradients to flow in pytorch optimization.
    Keep in mind that KDE are not scaling well with the number of dimensions and this implementation is not really
    optimized...
    """
    def __init__(self, x_test):
        # self.train_x = x_train
        # self.train_y = y_train
        self.x_test = x_test
    
    def forward(self, y_train, x_train, device_gpu = "cpu"):
        n = x_train.size()[0]
        # print(f'n={n}')
        d = 1
        bandwidth = torch.tensor((n * (d + 2) / 4.) ** (-1. / (d + 4))).to(device_gpu)

        y_hat = self.kde_regression(bandwidth, x_train, y_train)
        y_mean = torch.mean(y_train)
        pdf_values = self.pdf(bandwidth, x_train)

        DP = torch.sum(torch.abs(y_hat-y_mean) * pdf_values) / torch.sum(pdf_values)
        return DP

    def kde_regression(self, bandwidth, x_train, y_train):
        n = x_train.size()[0]
        X_repeat = self.x_test.repeat_interleave(n).reshape((-1, n))
        attention_weights = nn.functional.softmax(-(X_repeat - x_train)**2/(bandwidth ** 2) / 2, dim=1)
        y_hat = torch.matmul(attention_weights, y_train)
        return y_hat

    def pdf(self, bandwidth, x_train):
        n = x_train.size()[0]
        # data = x.unsqueeze(-2)
        # train_x = _unsqueeze_multiple_times(self.train_x, 0, len(s))

        data = self.x_test.repeat_interleave(n).reshape((-1, n))
        train_x = x_train.unsqueeze(0)
        # print(f'data={data.shape}')
        # print(f'train_x={train_x.shape}')

        pdf_values = (torch.exp(-((data - train_x) ** 2 / (bandwidth ** 2) / 2))
                     ).mean(dim=-1) / sqrt(2 * pi) / bandwidth

        return pdf_values




##################################################################################


# Interface to access the fairness measure
def gdp(y,p):
    test_sol = 1e-3 # Basically binning if I understand correctly
    x_appro = torch.arange(test_sol, 1-test_sol, test_sol)
    KDE_FAIR = kde_fair(x_appro)

    fairness = KDE_FAIR.forward

    GDP = fairness(y, p).item()

    return GDP