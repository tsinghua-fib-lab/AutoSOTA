import numpy as np
from scipy.special import softmax
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as tdata
import pandas as pd
import time
from sklearn.model_selection import KFold
from tqdm import tqdm
from utils import validate, get_logits_targets, sort_sum
from torch.autograd.functional import jacobian
import torch.nn.functional as F
import pdb
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import random

def compute_gradients_and_norm(softmax_input, T=1):

    device = softmax_input.device
    if isinstance(T, torch.Tensor):
        T = T.to(device)
    else:
        T = torch.tensor(T).detach.cpu()

    def softmax_single(input_logits):
        return F.softmax(input_logits / T.detach().numpy(), dim=1)

    softmax_output = softmax_single(softmax_input)
    softmax_output = np.asarray(softmax_output)
    batch_size, num_classes = softmax_output.shape
    jacobian = np.zeros((batch_size, num_classes, num_classes))
    norms = np.zeros((batch_size, num_classes))
    for i in range(batch_size):
        s = softmax_output[i]
        diag = np.diag(s)
        jacobian[i] = diag - np.outer(s, s)
        jacobian[i] /= T.detach().numpy()
        norms[i] = np.linalg.norm(jacobian[i], axis=1)

    return jacobian, norms

# Conformalize a model with a calibration set.
# Save it to a file in .cache/modelname
# The only difference is that the forward method of ConformalModel also outputs a set.
class ConformalModel(nn.Module):
    def __init__(self, model, calib_loader, alpha, kreg=None, lamda=None, randomized=True, allow_zero_sets=False,
                 pct_paramtune=0.3, batch_size=32, lamda_criterion='size', delta=None, T=None):
        super(ConformalModel, self).__init__()
        self.model = model
        self.alpha = alpha

        if T == None:
            self.T = torch.Tensor([1.3])  # initialize (1.3 is usually a good value)
            self.T, calib_logits = platt(self, calib_loader)
        else:
            calib_logits = get_logits_targets(model, calib_loader)
            self.T = T
            print(f"Optimal T={T.item()}")

        self.randomized = randomized
        self.allow_zero_sets = allow_zero_sets
        self.num_classes = len(calib_loader.dataset.dataset.classes)

        if kreg == None or lamda == None or delta == None:
            kreg, lamda, calib_logits, self.delta = pick_parameters(model, calib_logits, alpha, kreg, lamda, randomized,
                                                        allow_zero_sets, pct_paramtune, batch_size, lamda_criterion, delta, self.T)

        self.penalties = np.zeros((1, self.num_classes))
        self.penalties[:, kreg:] += lamda

        calib_loader = tdata.DataLoader(calib_logits, batch_size=batch_size, shuffle=False, pin_memory=True)

        self.Qhat = conformal_calibration_logits(self, calib_loader, self.T, self.delta)

    def forward(self, *args, randomized=None, allow_zero_sets=None, **kwargs):
        if randomized == None:
            randomized = self.randomized
        if allow_zero_sets == None:
            allow_zero_sets = self.allow_zero_sets
        logits = self.model(*args, **kwargs)

        logits_numpy = logits.detach().cpu().numpy()
        scores = softmax(logits_numpy / self.T.item(), axis=1)

        # Get gradient information
        gradients, grad_norms_tensor = compute_gradients_and_norm(logits.detach().cpu(), self.T)
        grad_norms = grad_norms_tensor

        # Using gradient information to approximate
        delta = self.delta
        scores = scores + delta * grad_norms

        I, ordered, cumsum = sort_sum(scores)

        S = gcq(scores, self.Qhat, I=I, ordered=ordered, cumsum=cumsum, penalties=self.penalties,
                randomized=randomized, allow_zero_sets=allow_zero_sets)

        return logits, S


# Computes the conformal calibration
# Not used
def conformal_calibration(cmodel, calib_loader):
    T = torch.Tensor([1.3])
    print("Conformal calibration")
    with torch.no_grad():
        E = np.array([])
        for x, targets in tqdm(calib_loader):
            logits = cmodel.model(x.cuda()).detach().cpu().numpy()
            scores = softmax(logits / cmodel.T.item(), axis=1)

            I, ordered, cumsum = sort_sum(scores)

            E = np.concatenate((E, giq(scores, targets, I=I, ordered=ordered, cumsum=cumsum, penalties=cmodel.penalties,
                                       randomized=True, allow_zero_sets=True)))

        Qhat = np.quantile(E, 1 - cmodel.alpha, interpolation='higher')

        return Qhat

    # Temperature scaling
def platt(cmodel, calib_loader, max_iters=10, lr=0.01, epsilon=0.01):
    print("Begin Platt scaling.")
    # Save logits so don't need to double compute them
    logits_dataset = get_logits_targets(cmodel.model, calib_loader)
    logits_loader = torch.utils.data.DataLoader(logits_dataset, batch_size=calib_loader.batch_size, shuffle=False,
                                                pin_memory=True)

    T = platt_logits(cmodel, logits_loader, max_iters=max_iters, lr=lr, epsilon=epsilon)

    print(f"Optimal T={T.item()}")
    return T, logits_dataset

"""


        INTERNAL FUNCTIONS


"""

### Precomputed-logit versions of the above functions.

class ConformalModelLogits(nn.Module):
    def __init__(self, model, calib_loader, alpha, kreg=None, lamda=None, randomized=True, allow_zero_sets=False,
                 naive=False, LAC=False, pct_paramtune=0.3, batch_size=32, lamda_criterion='size', delta=None, T=None):
        super(ConformalModelLogits, self).__init__()
        self.model = model
        self.alpha = alpha
        self.randomized = randomized
        self.LAC = LAC
        self.allow_zero_sets = allow_zero_sets
        if T == None:
            self.T = platt_logits(self, calib_loader)
        else:
            self.T = T

        if (kreg == None or lamda == None or delta == None) and not naive and not LAC:
            kreg, lamda, calib_logits, delta = pick_parameters(model, calib_loader.dataset, alpha, kreg, lamda, randomized,
                                                        allow_zero_sets, pct_paramtune, batch_size, lamda_criterion, delta, self.T)
            calib_loader = tdata.DataLoader(calib_logits, batch_size=batch_size, shuffle=False, pin_memory=True)

        self.delta = delta

        self.penalties = np.zeros((1, calib_loader.dataset[0][0].shape[0]))
        if not (kreg == None) and not naive and not LAC:
            self.penalties[:, kreg:] += lamda
        self.Qhat = 1 - alpha
        if not naive and not LAC:
            self.Qhat = conformal_calibration_logits(self, calib_loader, self.T, self.delta)
        elif not naive and LAC:
            gt_locs_cal = np.array(
                [np.where(np.argsort(x[0]).flip(dims=(0,)) == x[1])[0][0] for x in calib_loader.dataset])
            scores_cal = 1 - np.array(
                [np.sort(torch.softmax(calib_loader.dataset[i][0] / self.T.item(), dim=0))[::-1][gt_locs_cal[i]] for i
                 in range(len(calib_loader.dataset))])
            self.Qhat = np.quantile(scores_cal, np.ceil((scores_cal.shape[0] + 1) * (1 - alpha)) / scores_cal.shape[0])

    def forward(self, logits, randomized=None, allow_zero_sets=None):
        if randomized == None:
            randomized = self.randomized
        if allow_zero_sets == None:
            allow_zero_sets = self.allow_zero_sets

        logits_numpy = logits.detach().cpu().numpy()
        scores = softmax(logits_numpy / self.T.item(), axis=1)

        gradients, grad_norms_tensor = compute_gradients_and_norm(logits.detach().cpu(), self.T)
        grad_norms = grad_norms_tensor

        delta = self.delta
        scores = scores + delta * grad_norms

        if not self.LAC:
            I, ordered, cumsum = sort_sum(scores)

            S = gcq(scores, self.Qhat, I=I, ordered=ordered, cumsum=cumsum, penalties=self.penalties,
                    randomized=randomized, allow_zero_sets=allow_zero_sets)
        else:
            S = [np.where((1 - scores[i, :]) < self.Qhat)[0] for i in range(scores.shape[0])]

        return logits, S


def conformal_calibration_logits(cmodel, calib_loader, T, delta):

    E = np.array([])
    for logits, targets in calib_loader:

        logits_numpy = logits.detach().cpu().numpy()
        scores = softmax(logits_numpy / T.item(), axis=1)

        gradients, grad_norms_tensor = compute_gradients_and_norm(logits, T)
        grad_norms = grad_norms_tensor

        scores = scores + delta * grad_norms

        I, ordered, cumsum = sort_sum(scores)

        E = np.concatenate((E, giq(scores, targets, I=I, ordered=ordered, cumsum=cumsum, penalties=cmodel.penalties,
                                   randomized=True, allow_zero_sets=True)))

    Qhat = np.quantile(E, 1 - cmodel.alpha, interpolation='higher')

    return Qhat


def platt_logits(cmodel, calib_loader, max_iters=10, lr=0.01, epsilon=0.01):
    nll_criterion = nn.CrossEntropyLoss().cuda()

    T = nn.Parameter(torch.Tensor([1.3]).cuda())

    optimizer = optim.SGD([T], lr=lr)
    for iter in range(max_iters):
        T_old = T.item()
        for x, targets in calib_loader:
            optimizer.zero_grad()
            x = x.cuda()
            x.requires_grad = True
            out = x / T
            loss = nll_criterion(out, targets.long().cuda())
            loss.backward()
            optimizer.step()
        if abs(T_old - T.item()) < epsilon:
            break
    return T


### CORE CONFORMAL INFERENCE FUNCTIONS

# Generalized conditional quantile function.
def gcq(scores, tau, I, ordered, cumsum, penalties, randomized, allow_zero_sets):
    penalties_cumsum = np.cumsum(penalties, axis=1)
    sizes_base = ((cumsum + penalties_cumsum) <= tau).sum(axis=1) + 1  # 1 - 1001
    sizes_base = np.minimum(sizes_base, scores.shape[1])  # 1-1000

    if randomized:
        V = np.zeros(sizes_base.shape)
        for i in range(sizes_base.shape[0]):
            V[i] = 1 / ordered[i, sizes_base[i] - 1] * \
                   (tau - (cumsum[i, sizes_base[i] - 1] - ordered[i, sizes_base[i] - 1]) - penalties_cumsum[
                       0, sizes_base[i] - 1])  # -1 since sizes_base \in {1,...,1000}.

        sizes = sizes_base - (np.random.random(V.shape) >= V).astype(int)
    else:
        sizes = sizes_base

    if tau == 1.0:
        sizes[:] = cumsum.shape[1]  # always predict max size if alpha==0. (Avoids numerical error.)

    if not allow_zero_sets:
        sizes[
            sizes == 0] = 1  # allow the user the option to never have empty sets (will lead to incorrect coverage if 1-alpha < model's top-1 accuracy

    S = list()

    # Construct S from equation (5)
    for i in range(I.shape[0]):
        S = S + [I[i, 0:sizes[i]], ]

    return S


# Get the 'p-value'
def get_tau(score, target, I, ordered, cumsum, penalty, randomized, allow_zero_sets):  # For one example
    idx = np.where(I == target)
    tau_nonrandom = cumsum[idx]

    if not randomized:
        return tau_nonrandom + penalty[0]

    U = np.random.random()

    if idx == (0, 0):
        if not allow_zero_sets:
            return tau_nonrandom + penalty[0]
        else:
            return U * tau_nonrandom + penalty[0]
    else:
        return U * ordered[idx] + cumsum[(idx[0], idx[1] - 1)] + (penalty[0:(idx[1][0] + 1)]).sum()


# Gets the histogram of Taus.
def giq(scores, targets, I, ordered, cumsum, penalties, randomized, allow_zero_sets):
    """
        Generalized inverse quantile conformity score function.
        E from equation (7) in Romano, Sesia, Candes.  Find the minimum tau in [0, 1] such that the correct label enters.
    """
    E = -np.ones((scores.shape[0],))
    for i in range(scores.shape[0]):
        E[i] = get_tau(scores[i:i + 1, :], targets[i].item(), I[i:i + 1, :], ordered[i:i + 1, :], cumsum[i:i + 1, :],
                       penalties[0, :], randomized=randomized, allow_zero_sets=allow_zero_sets)

    return E


### AUTOMATIC PARAMETER TUNING FUNCTIONS
def pick_kreg(paramtune_logits, alpha):
    gt_locs_kstar = np.array([np.where(np.argsort(x[0]).flip(dims=(0,)) == x[1])[0][0] for x in paramtune_logits])
    kstar = np.quantile(gt_locs_kstar, 1 - alpha, interpolation='higher') + 1
    return kstar


def pick_lamda_size(model, paramtune_loader, alpha, kreg, randomized, allow_zero_sets):
    print("start pick lamda")
    # Calculate lamda_star
    best_size = iter(paramtune_loader).__next__()[0][1].shape[0]  # number of classes
    # Use the paramtune data to pick lamda.  Does not violate exchangeability.
    for temp_lam in tqdm([0.001, 0.01, 0.1, 0.2, 0.5]):  # predefined grid, change if more precision desired.
        conformal_model = ConformalModelLogits(model, paramtune_loader, alpha=alpha, kreg=kreg, lamda=temp_lam,
                                               randomized=randomized, allow_zero_sets=allow_zero_sets, naive=False)
        top1_avg, top5_avg, cvg_avg, sz_avg = validate(paramtune_loader, conformal_model, print_bool=False)
        temp_delta = conformal_model.delta
        if sz_avg < best_size:
            best_size = sz_avg
            lamda_star = temp_lam
            delta_star = temp_delta
    return lamda_star, delta_star


def pick_lamda_adaptiveness(model, paramtune_loader, alpha, kreg, randomized, allow_zero_sets,
                            strata=[[0, 1], [2, 3], [4, 6], [7, 10], [11, 100], [101, 1000]]):
    # Calculate lamda_star
    lamda_star = 0
    best_violation = 1
    # Use the paramtune data to pick lamda.  Does not violate exchangeability.
    for temp_lam in [0, 1e-5, 1e-4, 8e-4, 9e-4, 1e-3, 1.5e-3,
                     2e-3]:  # predefined grid, change if more precision desired.
        conformal_model = ConformalModelLogits(model, paramtune_loader, alpha=alpha, kreg=kreg, lamda=temp_lam,
                                               randomized=randomized, allow_zero_sets=allow_zero_sets, naive=False)
        curr_violation = get_violation(conformal_model, paramtune_loader, strata, alpha)
        if curr_violation < best_violation:
            best_violation = curr_violation
            lamda_star = temp_lam
    return lamda_star


def pick_parameters(model, calib_logits, alpha, kreg, lamda, randomized, allow_zero_sets, pct_paramtune, batch_size,
                    lamda_criterion, delta, T):
    num_paramtune = int(np.ceil(pct_paramtune * len(calib_logits)))
    paramtune_logits, calib_logits = tdata.random_split(calib_logits,
                                                        [num_paramtune, len(calib_logits) - num_paramtune])
    calib_loader = tdata.DataLoader(calib_logits, batch_size=batch_size, shuffle=False, pin_memory=True)
    paramtune_loader = tdata.DataLoader(paramtune_logits, batch_size=batch_size, shuffle=False, pin_memory=True)

    if kreg == None:
        kreg = pick_kreg(paramtune_logits, alpha)
    if lamda == None:
        if lamda_criterion == "size":
            lamda, delta = pick_lamda_size(model, paramtune_loader, alpha, kreg, randomized, allow_zero_sets)
        elif lamda_criterion == "adaptiveness":
            lamda = pick_lamda_adaptiveness(model, paramtune_loader, alpha, kreg, randomized, allow_zero_sets)
    if delta == None:
        delta = pick_delta(model, paramtune_loader, alpha, kreg, lamda, randomized, allow_zero_sets, T)

    return kreg, lamda, calib_logits, delta

def pick_delta(model, paramtune_loader, alpha, kreg, lamda, randomized, allow_zero_sets, T):

    # Calculate delta_star
    best_size = iter(paramtune_loader).__next__()[0][1].shape[0]  # number of classes
    # Use the paramtune data to pick delta.  Does not violate exchangeability.
    print("start pick delta")
    for temp_delta in [0.00001, 0.0001, 0.001, 0.00005, 0.0005, 0.005, 0.0075, 0.0025, 0.01, 0.025, 0.05, 0.075]:  # predefined grid, change if more precision desired.
        conformal_model = ConformalModelLogits(model, paramtune_loader, alpha=alpha, kreg=kreg, lamda=lamda,
                                               randomized=randomized, allow_zero_sets=allow_zero_sets, naive=False, delta=temp_delta, T=T)
        top1_avg, top5_avg, cvg_avg, sz_avg = validate(paramtune_loader, conformal_model, print_bool=False)
        if sz_avg < best_size:
            best_size = sz_avg
            delta_star = temp_delta

    return delta_star

def get_violation(cmodel, loader_paramtune, strata, alpha):
    df = pd.DataFrame(columns=['size', 'correct'])
    for logit, target in loader_paramtune:
        # compute output
        output, S = cmodel(logit)  # This is a 'dummy model' which takes logits, for efficiency.
        # measure accuracy and record loss
        size = np.array([x.size for x in S])
        I, _, _ = sort_sum(logit.numpy())
        correct = np.zeros_like(size)
        for j in range(correct.shape[0]):
            correct[j] = int(target[j] in list(S[j]))
        batch_df = pd.DataFrame({'size': size, 'correct': correct})
        df = df.append(batch_df, ignore_index=True)
    wc_violation = 0
    for stratum in strata:
        temp_df = df[(df['size'] >= stratum[0]) & (df['size'] <= stratum[1])]
        if len(temp_df) == 0:
            continue
        stratum_violation = abs(temp_df.correct.mean() - (1 - alpha))
        wc_violation = max(wc_violation, stratum_violation)
    return wc_violation  # the violation

