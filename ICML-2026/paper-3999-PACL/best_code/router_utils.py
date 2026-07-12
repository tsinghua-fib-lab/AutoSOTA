import numpy as np
import torch as ch
from torch.utils.data import TensorDataset, DataLoader
from scipy.optimize import root_scalar
from tqdm import tqdm
from pac_utils import zero_one_loss_vec

class SigmoidLike:
    def __init__(self, scale: float = 1.0):
        self.scale = scale

    def forward(self, x):
        raise NotImplementedError
    
    def backward(self, x):
        x.requires_grad = True
        return ch.autograd.grad(self.forward(x), x, grad_outputs=ch.ones_like(x))[0]
    
    def log_backward(self, x):
        if not x.requires_grad:
            x.requires_grad = True
        return ch.autograd.grad(ch.log(self.forward(x)), x, grad_outputs=ch.ones_like(x), create_graph=True)[0]
    
class Sigmoid(SigmoidLike):
    def forward(self, x):
        return ch.sigmoid(x * self.scale)

class Tanh(SigmoidLike):
    def forward(self, x):
        return ch.tanh(x * self.scale) * 0.5 + 0.5

sigmoid = Sigmoid(scale=0.1)


## Uncertainty models

class LinearUncertainty(ch.nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.linear_base = ch.nn.Linear(dimension, dimension)
        self.linear = ch.nn.Linear(dimension, 1)
        self.tie_breaking_noise = None

    def forward(self, X, log=False, with_noise=True):
        x_raw = self.linear_base(X)
        x_raw = ch.relu(x_raw)
        w_raw = self.linear(x_raw)
        y = ch.sigmoid(w_raw)
        if self.tie_breaking_noise is None:
            self.tie_breaking_noise = ch.randn_like(y) * 1e-6
        if with_noise:
            return y + self.tie_breaking_noise
        else:
            return y


## Weighters

class LinearWeighter(ch.nn.Module):
    def __init__(self, dimension, Yhats):
        super().__init__()
        self.linear = ch.nn.Linear(dimension, Yhats)

    def forward(self, X, log=False):
        w_raw = self.linear(X)
        if log:
            return ch.log_softmax(w_raw, dim=-1)
        return ch.softmax(w_raw, dim=-1)

class DeterministicWeighter(ch.nn.Module):
    def __init__(self, dimension, Yhats):
        self.Yhats = Yhats
        super().__init__()

    def forward(self, X, log=False):
        mask = ch.ones(X.shape[0], self.Yhats) / self.Yhats
        if log:
            return ch.log(mask)
        return mask

#### Thresholding methods

def threshold(*, weighter, X, Y, Yhat, confs, uncertainty_model, num_classes, epsilon=0.1):
    errors = zero_one_loss_vec(Y, Yhat)
    X_for_uncertainty = make_uncertainty_input(X=X, 
                                               Yhat=Yhat, 
                                               confs=confs, 
                                               weighter=weighter,
                                               num_classes=num_classes)
    def func(uhat):
        weights = weighter(X)
        uncertainties = uncertainty_model(X_for_uncertainty)
        return (weights * errors * (uncertainties > uhat).float()).sum(1).mean() - epsilon
    if np.sign(func(-0.1))*np.sign(func(100)) > 0:
        return ch.tensor(0.5)
    uhat = root_scalar(func, bracket=[-0.1, 100]).root
    return uhat

class DumbAutograd(ch.autograd.Function):
    @staticmethod
    def forward(ctx, weights, log_weights, uncertainties, u_value, *back_tensors):
        ctx.save_for_backward(*back_tensors)
        return u_value

    @staticmethod
    def backward(ctx, grad_output):
        back_tensors = ctx.saved_tensors
        return *[None if bt is None else grad_output * bt for bt in back_tensors], None, *[None for _ in back_tensors]

def threshold_smooth(*, weights, log_weights, Y, Yhat, uncertainties, epsilon=0.05):
    errors = zero_one_loss_vec(Y, Yhat)
    # Solve for mean(weighter(X) * errors * (1 - sigmoid(uhat - uncertainties))) = epsilon 
    # Use scipy.optimize.root_scalar to solve for uhat
    def func(uhat):
        return (weights * errors * sigmoid.forward(uncertainties - uhat)).sum(1).mean() - epsilon
    if np.sign(func(-100))*np.sign(func(100)) > 0:
        return ch.tensor(0.5)
    uhat = root_scalar(func, bracket=[-100, 100]).root
    return ch.tensor(uhat)

def threshold_smooth_surrogate(*, weights, log_weights, Y, Yhat, uncertainties, uhat_value):
    ## Weighter grad
    w_uncertainties = uncertainties.clone().detach()
    with ch.no_grad():
        j_probs = weights * zero_one_loss_vec(Y, Yhat) * sigmoid.forward(uhat_value - w_uncertainties) 
        j_probs = j_probs.detach()

    loss = (j_probs * log_weights).sum(1) / (j_probs * sigmoid.log_backward(uhat_value - w_uncertainties)).sum(1)
    loss = -loss.mean()

    ## Uncertainty grad
    sigs = sigmoid.backward((uhat_value - w_uncertainties).detach())
    with ch.no_grad():
        i_probs = weights * zero_one_loss_vec(Y, Yhat) * sigs
        i_probs = i_probs.detach()
    
    loss = loss + (i_probs * uncertainties).sum(1).mean() / i_probs.sum(1).mean()

    return loss


def complex_step(*, optimizer, weighter, uncertainty_model, X_batch, Y_batch, Yhat_batch, raw_confs, num_classes, epsilon=0.1,
                    costs=None, expert_cost=1):

    if costs is None:
        costs = ch.zeros(Yhat_batch.shape[1])
    added_expert_cost = expert_cost - costs

    X_for_uncertainty = make_uncertainty_input(X=X_batch, 
                                               Yhat=Yhat_batch, 
                                               confs=raw_confs, 
                                               weighter=weighter,
                                               num_classes=num_classes)
    
    weights = weighter(X_batch)
    log_weights = weighter(X_batch, log=True)
    uncertainties = uncertainty_model(X_for_uncertainty) 
    with ch.no_grad():
        uhat_value = threshold_smooth(
            weights=weights, 
            log_weights=log_weights, 
            Y=Y_batch, 
            Yhat=Yhat_batch, 
            uncertainties=uncertainties,
            epsilon=epsilon)
        
    surrogate_u = threshold_smooth_surrogate(
        weights=weights, 
        log_weights=log_weights, 
        Y=Y_batch, 
        Yhat=Yhat_batch, 
        uncertainties=uncertainties, 
        uhat_value=uhat_value)
    duhat_value = ch.autograd.grad(surrogate_u, (weights, log_weights, uncertainties), allow_unused=True)
    uhat = DumbAutograd.apply(weights, log_weights, uncertainties, uhat_value, *duhat_value)

    surrogate_loss = (weighter(X_batch) * (costs + added_expert_cost * sigmoid.forward(uncertainties - uhat))).sum(1).mean()
    optimizer.zero_grad()
    surrogate_loss.backward()
    # Normalize the gradient
    for param in weighter.parameters():
        if hasattr(param, 'grad'):
            param.grad = ch.sign(param.grad)
    for param in uncertainty_model.parameters():
        if hasattr(param, 'grad'):
            param.grad = ch.sign(param.grad)
    optimizer.step()
    return uhat_value, surrogate_loss


def train_weighter(input_df, calibration_frac=0.2, dimension=150, Yhats=2, costs=[0,0], expert_cost=1, epochs=50, batch_size=50, update_weighter=True, update_uncertainty=True, epsilon=0.1):
    weighter = LinearWeighter(dimension, Yhats)
    weighter.linear.weight.data = ch.zeros_like(weighter.linear.weight.data)

    full_df = input_df.copy()
    # Subselect the calibration set without replacement
    input_df = input_df.sample(frac=calibration_frac, replace=False)
    sampled_mask = full_df.index.isin(input_df.index)

    X = np.array(input_df[[f'X_{i}' for i in range(dimension)]]).astype(np.float32)
    Y = np.array(input_df['Y'])[:, None].astype(np.float32)
    Yhat = np.array(input_df[[f'Yhat_{i}' for i in range(Yhats)]]).astype(np.float32)
    corrects = (Y == Yhat)
    conf = np.array(input_df[[f'confidence_{i}' for i in range(Yhats)]]).astype(np.float32)

    num_classes = len(np.unique(Y))

    uncertainty_model = LinearUncertainty(dimension + 1 + num_classes)
    uncertainty_model.linear.weight.data = ch.zeros_like(uncertainty_model.linear.weight.data)

    dataset = TensorDataset(ch.tensor(X), ch.tensor(Y), ch.tensor(Yhat), ch.tensor(conf), ch.tensor(corrects))
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    params_to_update = []
    if update_weighter:
        params_to_update.extend(list(weighter.parameters()))
    if update_uncertainty:
        params_to_update.extend(list(uncertainty_model.parameters()))

    optimizer = ch.optim.AdamW(params_to_update, lr=0.1, weight_decay=0.001)
    # optimizer = ch.optim.SGD(params_to_update, lr=1e-3, weight_decay=0., momentum=0.0)
    it = tqdm(range(epochs), total=epochs)

    costs = ch.tensor(costs) if costs is not None else None
    added_expert_cost = expert_cost - costs
    for epoch in it:
        for batch in train_loader:
            X_batch, Y_batch, Yhat_batch, conf_batch, corrects_batch = batch
            X_for_uncertainty = make_uncertainty_input(X=X_batch, 
                                                       Yhat=Yhat_batch, 
                                                       confs=conf_batch, 
                                                       weighter=weighter,
                                                       num_classes=num_classes)

            learned_uncertainty = uncertainty_model(X_for_uncertainty)
            
            smooth_u, surrogate_u = complex_step(optimizer=optimizer, 
                                                           weighter=weighter, 
                                                           uncertainty_model=uncertainty_model,
                                                           X_batch=X_batch, 
                                                           Y_batch=Y_batch, 
                                                           Yhat_batch=Yhat_batch, 
                                                           raw_confs=conf_batch,
                                                           costs=costs,
                                                           num_classes=num_classes,
                                                           epsilon=epsilon)
            with ch.no_grad():
                real_u = threshold(weighter=weighter, 
                                        X=X_batch, 
                                        Y=Y_batch, 
                                        Yhat=Yhat_batch, 
                                        confs=conf_batch,
                                        uncertainty_model=uncertainty_model,
                                        num_classes=num_classes,
                                        epsilon=epsilon)
                avg_correct = (weighter(X_batch) * corrects_batch).sum(1).mean()
                real_cost = (weighter(X_batch) * (costs + added_expert_cost * (learned_uncertainty < real_u).float())).sum(1).mean()
            it.set_description(f"Ep {epoch} "
                               f"Smooth u: {smooth_u:.3f} "
                               f"Surrogate u: {surrogate_u:.3f} "
                               f"Real u: {real_u:.3f} "
                               f"Avg correct: {avg_correct:.3f} "
                               f"Real cost: {real_cost:.3f}")
    
    # Use the weighter to select the right Yhat column for each example
    all_X = np.array(full_df[[f'X_{i}' for i in range(dimension)]]).astype(np.float32)
    all_Yhats = np.array(full_df[[f'Yhat_{i}' for i in range(Yhats)]]).astype(np.float32)
    all_confs = np.array(full_df[[f'confidence_{i}' for i in range(Yhats)]]).astype(np.float32)
    best_Yhats = np.argmax(weighter(ch.tensor(all_X)).detach().numpy(), axis=-1)
    full_df['Yhat_routed'] = all_Yhats[np.arange(len(all_Yhats)), best_Yhats]
    X_for_uncertainty = ch.cat([ch.tensor(all_X), 
                                ch.tensor(all_confs[np.arange(len(all_confs)), best_Yhats])[:, None],
                                ch.nn.functional.one_hot(ch.tensor(full_df['Yhat_routed'].values).long() - 1, num_classes=num_classes)], dim=-1)
    full_df['confidence_routed'] = 1 - uncertainty_model(X_for_uncertainty, with_noise=False).detach().numpy().squeeze()
    full_df['label_collected'] = sampled_mask
    full_df['routed_model'] = best_Yhats
    return full_df

def make_uncertainty_input(*, X, Yhat, confs, weighter, num_classes):
    weights = weighter(X)
    agg_conf = (weights * confs).sum(dim=1, keepdim=True).clone().detach()
    agg_Yhat = ch.nn.functional.one_hot(Yhat.long() - 1, num_classes=num_classes)
    agg_Yhat = (weights[..., None] * agg_Yhat).sum(dim=1)
    return ch.cat([X, agg_conf, agg_Yhat], dim=-1)