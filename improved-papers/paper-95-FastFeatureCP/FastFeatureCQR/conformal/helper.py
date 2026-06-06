import sys
import copy
import torch
import numpy as np
import torch.nn as nn
import abc
import ipdb
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from functools import partial

from conformal.icp import IcpRegressor
import conformal.cqr_torch_models as torch_models


def run_icp(nc, X_train, y_train, X_test, idx_train, idx_cal, significance, condition=None):
    """ Run split conformal method

    Parameters
    ----------
    nc : class of nonconformist object
    X_train : numpy array, training features (n1Xp)
    y_train : numpy array, training labels (n1)
    X_test : numpy array, testing features (n2Xp)
    idx_train : numpy array, indices of proper training set examples
    idx_cal : numpy array, indices of calibration set examples
    significance : float, significance level (e.g. 0.1)
    condition : function, mapping feature vector to group id

    Returns
    -------
    y_lower : numpy array, estimated lower bound for the labels (n2)
    y_upper : numpy array, estimated upper bound for the labels (n2)

    """
    icp = IcpRegressor(nc, condition=condition)
    icp.fit(X_train[idx_train, :], y_train[idx_train])
    icp.calibrate(X_train[idx_cal, :], y_train[idx_cal])
    predictions = icp.predict(X_test, significance=significance)

    y_lower = predictions[:, 0]
    y_upper = predictions[:, 1]
    return y_lower, y_upper


class BaseModelAdapter(BaseEstimator):
    __metaclass__ = abc.ABCMeta

    def __init__(self, model, fit_params=None):
        super(BaseModelAdapter, self).__init__()

        self.model = model
        self.last_x, self.last_y = None, None
        self.clean = False
        self.fit_params = {} if fit_params is None else fit_params

    def fit(self, x, y):
        self.model.fit(x, y, **self.fit_params)
        self.clean = False

    def predict(self, x):
        if (not self.clean or
            self.last_x is None or
            self.last_y is None or
            not np.array_equal(self.last_x, x)
        ):
            self.last_x = x
            self.last_y = self._underlying_predict(x)
            self.clean = True

        return self.last_y.copy()

    @abc.abstractmethod
    def _underlying_predict(self, x):
        pass


class RegressorAdapter(BaseModelAdapter):
    def __init__(self, model, fit_params=None):
        super(RegressorAdapter, self).__init__(model, fit_params)

    def _underlying_predict(self, x):
        return self.model.predict(x)


class MSENet_RegressorAdapter(RegressorAdapter):
    def __init__(self, model, device, fit_params=None, in_shape=1, out_shape=1, hidden_size=1,
                 learn_func=torch.optim.Adam, epochs=1000, batch_size=10,
                 dropout=0.1, lr=0.01, wd=1e-6, test_ratio=0.2, random_state=0):
        super(MSENet_RegressorAdapter, self).__init__(model, fit_params)
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout = dropout
        self.lr = lr
        self.wd = wd
        self.test_ratio = test_ratio
        self.random_state = random_state
        self.device = device
        if model is None:
            self.model = mse_model(in_shape=in_shape, out_shape=out_shape, hidden_size=hidden_size, dropout=dropout)
        self.loss_func = torch.nn.MSELoss()
        self.learner = LearnerOptimized(self.model, partial(learn_func, lr=lr, weight_decay=wd),
                                        self.loss_func, device=device, test_ratio=self.test_ratio, random_state=self.random_state)

        self.hidden_size = hidden_size
        self.in_shape = in_shape
        self.learn_func = learn_func

    def fit(self, x, y):
        self.learner.fit(x, y, self.epochs, batch_size=self.batch_size, verbose=True)

    def predict(self, x):
        return self.learner.predict(x)


class AllQNet_RegressorAdapter(RegressorAdapter):
    """ Conditional quantile estimator, formulated as neural net
    """
    def __init__(self,
                 model,
                 fit_params=None,
                 in_shape=1,
                 hidden_size=1,
                 quantiles=[.05, .95],
                 learn_func=torch.optim.Adam,
                 epochs=1000,
                 batch_size=10,
                 dropout=0.1,
                 lr=0.01,
                 wd=1e-6,
                 test_ratio=0.2,
                 random_state=0,
                 device='cpu',
                 use_rearrangement=False):
        """ Initialization

        Parameters
        ----------
        model : None, unused parameter (for compatibility with nc class)
        fit_params : None, unused parameter (for compatibility with nc class)
        in_shape : integer, input signal dimension
        hidden_size : integer, hidden layer dimension
        quantiles : numpy array, low and high quantile levels in range (0,1)
        learn_func : class of Pytorch's SGD optimizer
        epochs : integer, maximal number of epochs
        batch_size : integer, mini-batch size for SGD
        dropout : float, dropout rate
        lr : float, learning rate for SGD
        wd : float, weight decay
        test_ratio : float, ratio of held-out data, used in cross-validation
        random_state : integer, seed for splitting the data in cross-validation
        use_rearrangement : boolean, use the rearrangement algorithm (True)
                            of not (False). See reference [1].

        References
        ----------
        .. [1]  Chernozhukov, Victor, Iván Fernández‐Val, and Alfred Galichon.
                "Quantile and probability curves without crossing."
                Econometrica 78.3 (2010): 1093-1125.

        """
        super(AllQNet_RegressorAdapter, self).__init__(model, fit_params)
        # Instantiate model
        self.quantiles = quantiles
        if use_rearrangement:
            self.all_quantiles = torch.from_numpy(np.linspace(0.01,0.99,99)).float()
        else:
            self.all_quantiles = self.quantiles
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout = dropout
        self.lr = lr
        self.wd = wd
        self.test_ratio = test_ratio
        self.random_state = random_state
        self.device = device
        if self.model is None:
            self.model = torch_models.all_q_model(quantiles=self.all_quantiles,
                                                  in_shape=in_shape,
                                                  hidden_size=hidden_size,
                                                  dropout=dropout)
        else:
            self.model = model
        self.loss_func = torch_models.AllQuantileLoss(self.all_quantiles)
        self.learner = torch_models.LearnerOptimizedCrossing(self.model,
                                                             partial(learn_func, lr=lr, weight_decay=wd),
                                                             self.loss_func,
                                                             device=device,
                                                             test_ratio=self.test_ratio,
                                                             random_state=self.random_state,
                                                             qlow=self.quantiles[0],
                                                             qhigh=self.quantiles[1],
                                                             use_rearrangement=use_rearrangement)

    def fit(self, x, y):
        """ Fit the model to data

        Parameters
        ----------

        x : numpy array of training features (nXp)
        y : numpy array of training labels (n)

        """
        self.learner.fit(x, y, self.epochs, self.batch_size, verbose=True)

    def predict(self, x):
        """ Estimate the conditional low and high quantiles given the features

        Parameters
        ----------
        x : numpy array of training features (nXp)

        Returns
        -------
        ret_val : numpy array of estimated conditional quantiles (nX2)

        """
        return self.learner.predict(x)


def epoch_internal_train(model, loss_func, x_train, y_train, batch_size, optimizer, cnt=0, best_cnt=np.Inf):
    model.train()
    shuffle_idx = np.arange(x_train.shape[0])
    np.random.shuffle(shuffle_idx)
    x_train = x_train[shuffle_idx]
    y_train = y_train[shuffle_idx]
    epoch_losses = []
    for idx in range(0, x_train.shape[0], batch_size):
        cnt = cnt + 1
        optimizer.zero_grad()
        batch_x = x_train[idx: min(idx + batch_size, x_train.shape[0]), :]
        batch_y = y_train[idx: min(idx + batch_size, y_train.shape[0])]
        preds = model(batch_x)
        loss = loss_func(preds, batch_y)
        loss.backward()
        optimizer.step()
        epoch_losses.append(loss.cpu().detach().numpy())

        if cnt >= best_cnt:
            break

    epoch_loss = np.mean(epoch_losses)

    return epoch_loss, cnt


class mse_model(nn.Module):
    def __init__(self, in_shape=1, out_shape=1, hidden_size=64, dropout=0.5):

        super().__init__()
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.build_model()
        self.init_weights()

    def build_model(self):
        self.encoder = nn.Sequential(
            nn.Linear(self.in_shape, self.hidden_size),
            nn.ReLU(), nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, self.hidden_size), nn.ReLU())
        self.g = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size), nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, self.out_shape))

        self.base_model = nn.Sequential(self.encoder, self.g)

    def init_weights(self):
        for m in self.base_model:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, layer_index=0, return_intermediate=False):

        if return_intermediate and layer_index == 0:
            intermediate = x
        x = self.encoder[0](x)

        if return_intermediate and layer_index == 1:
            intermediate = x
        x = self.encoder[1](x)
        x = self.encoder[2](x)
        x = self.encoder[3](x)
        if return_intermediate and layer_index == 2:
            intermediate = x
        x = self.encoder[4](x)

        x = self.g[0](x)
        if return_intermediate and layer_index == 3:
            intermediate = x
        x = self.g[1](x)
        x = self.g[2](x)
        x = self.g[3](x)
        if return_intermediate and layer_index == 4:
            intermediate = x

        if return_intermediate:
            # Ensure intermediate requires gradient
            intermediate.retain_grad()
            return intermediate,torch.squeeze(x)
        else:
            return torch.squeeze(x)

    def predict_with_grads(self, X, layer_index):
        self.eval()
        X = torch.from_numpy(X).float()
        X.requires_grad = True  # Ensure X requires gradients

        with torch.enable_grad():  # Enable gradient computation
            intermediate, outputs = self.forward(X, layer_index, return_intermediate=True)
            outputs.backward(torch.ones_like(outputs))  # Perform backward pass

            # Get the gradients
            intermediate_grads = intermediate.grad  # Gradients of outputs w.r.t. intermediate
            outputs_grads = X.grad  # Gradients of outputs w.r.t. X

        return outputs.detach().numpy(), intermediate_grads.detach().numpy(), outputs_grads.detach().numpy()


class LearnerOptimized:
    def __init__(self, model, optimizer_class, loss_func, device='cpu', test_ratio=0.2, random_state=0):
        self.model = model.to(device)
        self.optimizer_class = optimizer_class
        self.optimizer = optimizer_class(self.model.parameters())
        self.loss_func = loss_func.to(device)
        self.device = device
        self.test_ratio = test_ratio
        self.random_state = random_state
        self.loss_history = []
        self.test_loss_history = []
        self.full_loss_history = []

    def fit(self, x, y, epochs, batch_size, verbose=False):
        sys.stdout.flush()
        model = copy.deepcopy(self.model)
        model = model.to(self.device)
        optimizer = self.optimizer_class(model.parameters())
        best_epoch = epochs

        x_train, xx, y_train, yy = train_test_split(x, y, test_size=self.test_ratio, random_state=self.random_state)

        x_train = torch.from_numpy(x_train).float().to(self.device).requires_grad_(False)
        xx = torch.from_numpy(xx).float().to(self.device).requires_grad_(False)
        y_train = torch.from_numpy(y_train).float().to(self.device).requires_grad_(False)
        yy = torch.from_numpy(yy).float().to(self.device).requires_grad_(False)

        best_cnt = 1e10
        best_test_epoch_loss = 1e10

        cnt = 0
        for e in range(epochs):
            epoch_loss, cnt = epoch_internal_train(model, self.loss_func, x_train, y_train, batch_size, optimizer, cnt)
            self.loss_history.append(epoch_loss)

            # test
            model.eval()
            preds = model(xx)
            test_epoch_loss = self.loss_func(preds, yy).cpu().detach().numpy()
            self.test_loss_history.append(test_epoch_loss)

            test_preds = preds.cpu().detach().numpy()
            test_preds = np.squeeze(test_preds)

            if (test_epoch_loss <= best_test_epoch_loss):
                best_test_epoch_loss = test_epoch_loss
                best_epoch = e
                best_cnt = cnt

            if (e + 1) % 100 == 0 and verbose:
                print("CV: Epoch {}: Train {}, Test {}, Best epoch {}, Best loss {}".format(e + 1, epoch_loss,
                                            test_epoch_loss, best_epoch, best_test_epoch_loss))
                sys.stdout.flush()

        # use all the data to train the model, for best_cnt steps
        x = torch.from_numpy(x).float().to(self.device).requires_grad_(False)
        y = torch.from_numpy(y).float().to(self.device).requires_grad_(False)

        cnt = 0
        for e in range(best_epoch + 1):
            if cnt > best_cnt:
                break

            epoch_loss, cnt = epoch_internal_train(self.model, self.loss_func, x, y, batch_size,
                                                   self.optimizer, cnt, best_cnt)
            self.full_loss_history.append(epoch_loss)

            if (e + 1) % 100 == 0 and verbose:
                print("Full: Epoch {}: {}, cnt {}".format(e + 1, epoch_loss, cnt))
                sys.stdout.flush()

    def predict(self, x):
        self.model.eval()
        if isinstance(x, torch.Tensor):
            x = x.to(self.device).requires_grad_(False)
        else:
            x = torch.from_numpy(x).to(self.device).requires_grad_(False)
        ret_val = self.model(x).cpu().detach().numpy()
        return ret_val


