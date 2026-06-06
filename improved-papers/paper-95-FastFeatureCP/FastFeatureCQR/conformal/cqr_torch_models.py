import sys
import copy
import torch
import numpy as np
import torch.nn as nn
from conformal import helper, utils
from sklearn.model_selection import train_test_split


def epoch_internal_train(model, loss_func, x_train, y_train, batch_size, optimizer, cnt=0, best_cnt=np.Inf):
    """ Sweep over the data and update the model's parameters

    Parameters
    ----------

    model : class of neural net model
    loss_func : class of loss function
    x_train : pytorch tensor n training features, each of dimension p (nXp)
    batch_size : integer, size of the mini-batch
    optimizer : class of SGD solver
    cnt : integer, counting the gradient steps
    best_cnt: integer, stop the training if current cnt > best_cnt

    Returns
    -------

    epoch_loss : mean loss value
    cnt : integer, cumulative number of gradient steps

    """

    model.train()
    shuffle_idx = np.arange(x_train.shape[0])
    np.random.shuffle(shuffle_idx)
    x_train = x_train[shuffle_idx]
    y_train = y_train[shuffle_idx]
    epoch_losses = []
    for idx in range(0, x_train.shape[0], batch_size):
        cnt = cnt + 1
        optimizer.zero_grad()
        batch_x = x_train[idx : min(idx + batch_size, x_train.shape[0]),:]
        batch_y = y_train[idx : min(idx + batch_size, y_train.shape[0])]
        preds = model(batch_x)
        loss = loss_func(preds, batch_y)
        loss.backward()
        optimizer.step()
        epoch_losses.append(loss.cpu().detach().numpy())

        if cnt >= best_cnt:
            break

    epoch_loss = np.mean(epoch_losses)

    return epoch_loss, cnt


def rearrange(all_quantiles, quantile_low, quantile_high, test_preds):
    """ Produce monotonic quantiles

    Parameters
    ----------

    all_quantiles : numpy array (q), grid of quantile levels in the range (0,1)
    quantile_low : float, desired low quantile in the range (0,1)
    quantile_high : float, desired high quantile in the range (0,1)
    test_preds : numpy array of predicted quantile (nXq)

    Returns
    -------

    q_fixed : numpy array (nX2), containing the rearranged estimates of the
              desired low and high quantile

    References
    ----------
    .. [1]  Chernozhukov, Victor, Iván Fernández‐Val, and Alfred Galichon.
            "Quantile and probability curves without crossing."
            Econometrica 78.3 (2010): 1093-1125.

    """
    scaling = all_quantiles[-1] - all_quantiles[0]
    low_val = (quantile_low - all_quantiles[0])/scaling
    high_val = (quantile_high - all_quantiles[0])/scaling
    q_fixed = np.quantile(test_preds,(low_val, high_val),interpolation='linear',axis=1)
    return q_fixed.T


# Define the training procedure
class LearnerOptimized:
    """ Fit a neural network (conditional mean) to training data
    """
    def __init__(self, model, optimizer_class, loss_func, device='cpu', test_ratio=0.2, random_state=0):
        """ Initialization

        Parameters
        ----------

        model : class of neural network model
        optimizer_class : class of SGD optimizer (e.g. Adam)
        loss_func : loss to minimize
        device : string, "cuda:0" or "cpu"
        test_ratio : float, test size used in cross-validation (CV)
        random_state : int, seed to be used in CV when splitting to train-test

        """
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
        """ Fit the model to data

        Parameters
        ----------

        x : numpy array, containing the training features (nXp)
        y : numpy array, containing the training labels (n)
        epochs : integer, maximal number of epochs
        batch_size : integer, mini-batch size for SGD

        """

        sys.stdout.flush()
        model = copy.deepcopy(self.model)
        model = model.to(self.model.device)
        optimizer = self.optimizer_class(model.parameters())
        best_epoch = epochs

        x_train, xx, y_train, yy = train_test_split(x, y, test_size=self.test_ratio,random_state=self.random_state)

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
            test_preds = preds.cpu().detach().numpy()
            test_preds = np.squeeze(test_preds)
            test_epoch_loss = self.loss_func(preds, yy).cpu().detach().numpy()

            self.test_loss_history.append(test_epoch_loss)

            if (test_epoch_loss <= best_test_epoch_loss):
                best_test_epoch_loss = test_epoch_loss
                best_epoch = e
                best_cnt = cnt

            if (e+1) % 100 == 0 and verbose:
                print("CV: Epoch {}: Train {}, Test {}, Best epoch {}, Best loss {}".format(e+1, epoch_loss, test_epoch_loss, best_epoch, best_test_epoch_loss))
                sys.stdout.flush()

        # use all the data to train the model, for best_cnt steps
        x = torch.from_numpy(x).float().to(self.device).requires_grad_(False)
        y = torch.from_numpy(y).float().to(self.device).requires_grad_(False)

        cnt = 0
        for e in range(best_epoch+1):
            if cnt > best_cnt:
                break

            epoch_loss, cnt = epoch_internal_train(self.model, self.loss_func, x, y, batch_size, self.optimizer, cnt, best_cnt)
            self.full_loss_history.append(epoch_loss)

            if (e+1) % 100 == 0 and verbose:
                print("Full: Epoch {}: {}, cnt {}".format(e+1, epoch_loss, cnt))
                sys.stdout.flush()

    def predict(self, x):
        """ Estimate the label given the features

        Parameters
        ----------
        x : numpy array of training features (nXp)

        Returns
        -------
        ret_val : numpy array of predicted labels (n)

        """
        self.model.eval()
        ret_val = self.model(torch.from_numpy(x).to(self.device).requires_grad_(False)).cpu().detach().numpy()
        return ret_val


##############################################################################
# Quantile regression
# Implementation inspired by:
# https://github.com/ceshine/quantile-regression-tensorflow
##############################################################################

class AllQuantileLoss(nn.Module):
    """ Pinball loss function
    """
    def __init__(self, quantiles):
        """ Initialize

        Parameters
        ----------
        quantiles : pytorch vector of quantile levels, each in the range (0,1)


        """
        super().__init__()
        self.quantiles = quantiles

    def forward(self, preds, target):
        """ Compute the pinball loss

        Parameters
        ----------
        preds : pytorch tensor of estimated labels (n)
        target : pytorch tensor of true labels (n)

        Returns
        -------
        loss : cost function value

        """
        assert not target.requires_grad
        assert preds.size(0) == target.size(0)
        losses = []

        for i, q in enumerate(self.quantiles):
            errors = target - preds[:, i]
            losses.append(torch.max((q-1) * errors, q * errors).unsqueeze(1))

        loss = torch.mean(torch.sum(torch.cat(losses, dim=1), dim=1))
        return loss


class all_q_model(nn.Module):
    """ Conditional quantile estimator, formulated as neural net
    """
    def __init__(self,
                 quantiles,
                 in_shape=1,
                 hidden_size=64,
                 dropout=0.5):
        """ Initialization

        Parameters
        ----------
        quantiles : numpy array of quantile levels (q), each in the range (0,1)
        in_shape : integer, input signal dimension (p)
        hidden_size : integer, hidden layer dimension
        dropout : float, dropout rate

        """
        super().__init__()
        self.quantiles = quantiles
        self.num_quantiles = len(quantiles)
        self.hidden_size = hidden_size
        self.in_shape = in_shape
        self.out_shape = len(quantiles)
        self.dropout = dropout
        self.build_model()
        self.init_weights()

    def build_model(self):
        """ Construct the network
        """

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
        """ Initialize the network parameters
        """
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
            return intermediate, torch.squeeze(x)
        else:
            return torch.squeeze(x)

    def predict_with_grads(self, X, layer_index, output_index):
        self.eval()
        X = torch.from_numpy(X).float()
        X.requires_grad = True  # Ensure X requires gradients

        with torch.enable_grad():  # Enable gradient computation
            intermediate, outputs = self.forward(X, layer_index, return_intermediate=True)
            outputs_shape = outputs.shape

            output_index = torch.tensor(output_index, dtype=torch.long)
            intermediate_grads = torch.zeros(outputs_shape[0], intermediate.shape[1])
            self.zero_grad()
            outputs[:, output_index].backward(torch.ones_like(outputs[:, output_index]), retain_graph=True)
            intermediate_grads[:, :] = intermediate.grad.clone()
            intermediate.grad.zero_()

            outputs_grads = torch.autograd.grad(outputs[:, output_index], X, torch.ones_like(outputs[:, output_index]),
                                                retain_graph=True)[0]

        return outputs.detach().numpy(), intermediate_grads.detach().numpy(), outputs_grads.detach().numpy()


class LearnerOptimizedCrossing:
    """ Fit a neural network (conditional quantile) to training data
    """
    def __init__(self, model, optimizer_class, loss_func, device='cpu', test_ratio=0.2, random_state=0,
                 qlow=0.05, qhigh=0.95, use_rearrangement=False):
        """ Initialization

        Parameters
        ----------

        model : class of neural network model
        optimizer_class : class of SGD optimizer (e.g. pytorch's Adam)
        loss_func : loss to minimize
        device : string, "cuda:0" or "cpu"
        test_ratio : float, test size used in cross-validation (CV)
        random_state : integer, seed used in CV when splitting to train-test
        qlow : float, low quantile level in the range (0,1)
        qhigh : float, high quantile level in the range (0,1)
        use_rearrangement : boolean, use the rearrangement  algorithm (True)
                            of not (False)

        """
        self.model = model.to(device)
        self.use_rearrangement = use_rearrangement
        self.compute_coverage = True
        self.quantile_low = qlow
        self.quantile_high = qhigh
        self.target_coverage = 100.0*(self.quantile_high - self.quantile_low)
        self.all_quantiles = loss_func.quantiles
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
        """ Fit the model to data

        Parameters
        ----------

        x : numpy array of training features (nXp)
        y : numpy array of training labels (n)
        epochs : integer, maximal number of epochs
        batch_size : integer, mini-batch size used in SGD solver

        """
        sys.stdout.flush()
        model = copy.deepcopy(self.model)
        model = model.to(self.device)
        optimizer = self.optimizer_class(model.parameters())
        best_epoch = epochs

        x_train, xx, y_train, yy = train_test_split(x,
                                                    y,
                                                    test_size=self.test_ratio,
                                                    random_state=self.random_state)

        x_train = torch.from_numpy(x_train).float().to(self.device).requires_grad_(False)
        xx = torch.from_numpy(xx).float().to(self.device).requires_grad_(False)
        y_train = torch.from_numpy(y_train).float().to(self.device).requires_grad_(False)
        yy_cpu = yy
        yy = torch.from_numpy(yy).float().to(self.device).requires_grad_(False)

        best_avg_length = 1e10
        best_coverage = 0
        best_cnt = 1e10

        cnt = 0
        for e in range(epochs):
            model.train()
            epoch_loss, cnt = epoch_internal_train(model, self.loss_func, x_train, y_train, batch_size, optimizer, cnt)
            self.loss_history.append(epoch_loss)

            model.eval()
            preds = model(xx)
            test_epoch_loss = self.loss_func(preds, yy).cpu().detach().numpy()
            self.test_loss_history.append(test_epoch_loss)

            test_preds = preds.cpu().detach().numpy()
            test_preds = np.squeeze(test_preds)

            if self.use_rearrangement:
                test_preds = rearrange(self.all_quantiles, self.quantile_low, self.quantile_high, test_preds)

            y_lower = test_preds[:,0]
            y_upper = test_preds[:,1]
            coverage, avg_length = utils.compute_coverage_len(yy_cpu, y_lower, y_upper)

            if (coverage >= self.target_coverage) and (avg_length < best_avg_length):
                best_avg_length = avg_length
                best_coverage = coverage
                best_epoch = e
                best_cnt = cnt

            if (e+1) % 100 == 0 and verbose:
                print("CV: Epoch {}: Train {}, Test {}, Best epoch {}, Best Coverage {} Best Length {} Cur Coverage {}".format(e+1, epoch_loss, test_epoch_loss, best_epoch, best_coverage, best_avg_length, coverage))
                sys.stdout.flush()

        x = torch.from_numpy(x).float().to(self.device).requires_grad_(False)
        y = torch.from_numpy(y).float().to(self.device).requires_grad_(False)

        cnt = 0
        for e in range(best_epoch+1):
            if cnt > best_cnt:
                break
            epoch_loss, cnt = epoch_internal_train(self.model, self.loss_func, x, y, batch_size, self.optimizer, cnt, best_cnt)
            self.full_loss_history.append(epoch_loss)

            if (e+1) % 100 == 0 and verbose:
                print("Full: Epoch {}: {}, cnt {}".format(e+1, epoch_loss, cnt))
                sys.stdout.flush()

    def predict(self, x):
        """ Estimate the conditional low and high quantile given the features

        Parameters
        ----------
        x : numpy array of training features (nXp)

        Returns
        -------
        test_preds : numpy array of predicted low and high quantiles (nX2)

        """
        self.model.eval()
        test_preds = self.model(torch.from_numpy(x).to(self.device).requires_grad_(False)).cpu().detach().numpy()
        if self.use_rearrangement:
            test_preds = rearrange(self.all_quantiles, self.quantile_low, self.quantile_high, test_preds)
        else:
            test_preds[:,0] = np.min(test_preds,axis=1)
            test_preds[:,1] = np.max(test_preds,axis=1)
        return test_preds
