from math import ceil
import numpy as np
import matplotlib.pyplot as plt
import torch

from tueplots import bundles, axes


class TensorDataLoader:
    """Combination of torch's DataLoader and TensorDataset for efficient batch
    sampling and adaptive augmentation on GPU.
    """

    def __init__(self, x, y, transform=None, transform_y=None, batch_size=500, 
                 data_factor=1, shuffle=False):
        assert x.size(0) == y.size(0), 'Size mismatch'
        self.x = x
        self.y = y
        self.data_factor = data_factor
        self.n_data = y.size(0)
        self.batch_size = batch_size
        self.n_batches = ceil(self.n_data / self.batch_size)
        self.shuffle = shuffle
        identity = lambda x: x
        self.transform = transform if transform is not None else identity
        self.transform_y = transform_y if transform_y is not None else identity

    def __iter__(self):
        if self.shuffle:
            permutation = torch.randperm(self.n_data)
            self.x = self.x[permutation]
            self.y = self.y[permutation]
        self.i_batch = 0
        return self

    def __next__(self):
        if self.i_batch >= self.n_batches:
            raise StopIteration
        
        start = self.i_batch * self.batch_size
        end = start + self.batch_size
        batch = (self.transform(self.x[start:end]), self.transform_y(self.y[start:end]))
        self.i_batch += 1
        return batch

    def __len__(self):
        return self.n_batches

    @property
    def dataset(self):
        return DatasetDummy(self.n_data * self.data_factor)


class DatasetDummy:
    def __init__(self, N):
        self.N = N
        
    def __len__(self):
        return self.N


def plot_pair(x, y, f_forward, f_reverse, dark=False):
    x_test = np.linspace(x.min()-1, x.max()+1, 500)
    y_test = np.linspace(y.min()-1, y.max()+1, 500)
    m_for_test, s_for_test = f_forward(x_test)
    m_rev_test, s_rev_test = f_reverse(y_test)
    m_for, s_for = f_forward(x)
    m_rev, s_rev = f_reverse(y)

    if dark:
        plt.style.use('dark_background')
        col = 'white'
    else:
        col = 'black'

    with plt.rc_context({**bundles.aistats2022(column='half'), **axes.lines()}):
        fig, axss = plt.subplots(nrows=2, ncols=2, figsize=(4, 3))
        axs = axss[0]
        axs[0].scatter(x, y, alpha=0.3, color=col, lw=0.5, s=8)
        axs[0].plot(x_test, m_for_test, label='LSNM', color='tab:blue', lw=1.5)
        axs[0].fill_between(x_test, m_for_test-2*s_for_test, m_for_test+2*s_for_test, color='tab:blue', alpha=0.3)
        axs[0].set_xlim(x_test.min()+0.4, x_test.max())
        axs[0].set_ylim(y.min()-1, y.max()+1)
        axs[0].grid()
        axs[0].set_xlabel('Cause $X$')
        axs[0].set_ylabel('Effect $Y$')
        axs[0].set_title('Causal')

        axs[1].scatter(y, x, alpha=0.3, color=col, lw=0.5, s=8)
        axs[1].plot(y_test, m_rev_test, label='LSNM', color='tab:blue', lw=1.5)
        axs[1].fill_between(y_test, m_rev_test-2*s_rev_test, m_rev_test+2*s_rev_test, color='tab:blue', alpha=0.3)
        # legend = axs[1].legend()
        axs[1].set_xlim(y_test.min(), y_test.max())
        axs[1].set_ylim(x.min()-1, x.max()+1)
        axs[1].set_ylabel('Cause $X$')
        axs[1].set_xlabel('Effect $Y$')
        axs[1].set_title('Anticausal')
        axs[1].grid()

        axs = axss[1]
        axs[0].set_xlim(x_test.min()+0.4, x_test.max())
        axs[0].scatter(x, (y - m_for) / s_for, lw=0.5, s=8, color=col, alpha=0.3)
        axs[0].grid()
        axs[0].set_ylabel('Residual $\\frac{Y - f(X)}{g(X)}$')
        axs[0].set_xlabel('Cause $X$')

        axs[1].set_xlim(y_test.min(), y_test.max())
        axs[1].scatter(y, (x - m_rev) / s_rev, lw=0.5, s=8, color=col, alpha=0.3)
        axs[1].set_xlabel('Effect $Y$')
        axs[1].set_ylabel('Residual $\\frac{X - h(Y)}{k(Y)}$')
        axs[1].grid()
        plt.savefig(f'MNU_{dark}.png', transparent=True)
        plt.show()