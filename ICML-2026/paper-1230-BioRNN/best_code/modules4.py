import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, floor
import random
import time
def loss_mse(output, target, mask):
    """
    Mean squared error loss
    :param output: torch tensor of shape (num_trials, num_timesteps, output_dim)
    :param target: idem
    :param mask: torch tensor of shape (num_trials, num_timesteps, 1)
    :return: float
    """
    # Compute loss for each (trial, timestep) (average accross output dimensions)
    loss_tensor = (mask * (target - output)).pow(2).mean(dim=-1)
    # Account for different number of masked values per trial
    loss_by_trial = loss_tensor.sum(dim=-1) / mask[:, :, 0].sum(dim=-1)
    return loss_by_trial.mean()

def net_loss(net, _input, _target, _mask, n_epochs, lr=1e-2, batch_size=32, cuda=False):
    """
    Train a network
    :param net: nn.Module
    :param _input: torch tensor of shape (num_trials, num_timesteps, input_dim)
    :param _target: torch tensor of shape (num_trials, num_timesteps, output_dim)
    :param _mask: torch tensor of shape (num_trials, num_timesteps, 1)
    :param n_epochs: int
    :param lr: float, learning rate
    :param batch_size: int
    :param plot_learning_curve: bool
    :param plot_gradient: bool
    :param clip_gradient: None or float, if not None the value at which gradient norm is clipped
    :param keep_best: bool, if True, model with lower loss from training process will be kept (for this option, the
        network has to implement a method clone())
    :return: nothing
    """
    
    # CUDA management
    if cuda:
        if not torch.cuda.is_available():
            print("Warning: CUDA not available on this machine, switching to CPU")
            device = torch.device('cpu')
        else:
            device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    net.to(device=device)
    input = _input.to(device=device)
    target = _target.to(device=device)
    mask = _mask.to(device=device)

    with torch.no_grad():
        initial_loss = loss_mse(net(input), target, mask)
        print("initial loss: %.3f" % (initial_loss.item()))
    return(initial_loss.item())

def train(net, _input, _target, _mask, n_epochs, lr=1e-2, batch_size=32, plot_learning_curve=False, plot_gradient=False,
          clip_gradient=None, keep_best=False, cuda=False, resample=False, save_loss=False):
    """
    Train a network
    :param net: nn.Module
    :param _input: torch tensor of shape (num_trials, num_timesteps, input_dim)
    :param _target: torch tensor of shape (num_trials, num_timesteps, output_dim)
    :param _mask: torch tensor of shape (num_trials, num_timesteps, 1)
    :param n_epochs: int
    :param lr: float, learning rate
    :param batch_size: int
    :param plot_learning_curve: bool
    :param plot_gradient: bool
    :param clip_gradient: None or float, if not None the value at which gradient norm is clipped
    :param keep_best: bool, if True, model with lower loss from training process will be kept (for this option, the
        network has to implement a method clone())
    :return: nothing
    """
    print("Training...")
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    num_examples = _input.shape[0]
    all_losses = []
    if plot_gradient:
        gradient_norms = []
    
    # CUDA management
    if cuda:
        if not torch.cuda.is_available():
            print("Warning: CUDA not available on this machine, switching to CPU")
            device = torch.device('cpu')
        else:
            device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    net.to(device=device)
    input = _input.to(device=device)
    target = _target.to(device=device)
    mask = _mask.to(device=device)

    with torch.no_grad():
        initial_loss = loss_mse(net(input), target, mask)
        print("initial loss: %.3f" % (initial_loss.item()))
        if keep_best:
            best = net.clone()
            best_loss = initial_loss.item()

    for epoch in range(n_epochs):
        begin = time.time()
        losses = []
        for i in range(num_examples // batch_size):
            optimizer.zero_grad()
            random_batch_idx = random.sample(range(num_examples), batch_size)
            batch = input[random_batch_idx]
            output = net(batch)
            loss = loss_mse(output, target[random_batch_idx], mask[random_batch_idx])
            losses.append(loss.item())
            all_losses.append(loss.item())
            loss.backward()
            if clip_gradient is not None:
                torch.nn.utils.clip_grad_norm_(net.parameters(), clip_gradient)
            if plot_gradient:
                tot = 0
                for param in [p for p in net.parameters() if p.requires_grad]:
                    tot += (param.grad ** 2).sum()
                gradient_norms.append(sqrt(tot))
            optimizer.step()
            # These 2 lines important to prevent memory leaks
            loss.detach_()
            output.detach_()
            if resample:
                net.resample_basis()
        if keep_best and np.mean(losses) < best_loss:
            best = net.clone()
            best_loss = np.mean(losses)
            print("epoch %d:  loss=%.9f  (took %.6f s) *" % (epoch, np.mean(losses), time.time() - begin))
        else:
            print("epoch %d:  loss=%.9f  (took %.6f s)" % (epoch, np.mean(losses), time.time() - begin))

    if plot_learning_curve:
        plt.plot(all_losses)
        plt.title("Learning curve")
        plt.show()

    if plot_gradient:
        plt.plot(gradient_norms)
        plt.title("Gradient norm")
        plt.show()

    if keep_best:
        net.load_state_dict(best.state_dict())
    if save_loss==True:
        return(all_losses)
    else:
        return()

def initial_loss(net, _input, _target, _mask, n_epochs, lr=1e-2, batch_size=32, plot_learning_curve=False, plot_gradient=False,
          clip_gradient=None, keep_best=False, cuda=False, resample=False, save_loss=False):
    """
    Train a network
    :param net: nn.Module
    :param _input: torch tensor of shape (num_trials, num_timesteps, input_dim)
    :param _target: torch tensor of shape (num_trials, num_timesteps, output_dim)
    :param _mask: torch tensor of shape (num_trials, num_timesteps, 1)
    :param n_epochs: int
    :param lr: float, learning rate
    :param batch_size: int
    :param plot_learning_curve: bool
    :param plot_gradient: bool
    :param clip_gradient: None or float, if not None the value at which gradient norm is clipped
    :param keep_best: bool, if True, model with lower loss from training process will be kept (for this option, the
        network has to implement a method clone())
    :return: nothing
    """

    
    # CUDA management
    if cuda:
        if not torch.cuda.is_available():
            print("Warning: CUDA not available on this machine, switching to CPU")
            device = torch.device('cpu')
        else:
            device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    net.to(device=device)
    input = _input.to(device=device)
    target = _target.to(device=device)
    mask = _mask.to(device=device)

    with torch.no_grad():
        initial_loss = loss_mse(net(input), target, mask)

        return(initial_loss.item())

class FullRankRNN(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, noise_std, alpha=0.2, rho=1,
                 train_wi=False, train_wo=False, train_wrec=True, train_h0=False,
                 wi_init=None, wo_init=None, wrec_init=None, si_init=None, so_init=None, h0_init=None):
        """

        :param input_size: int
        :param hidden_size: int
        :param output_size: int
        :param noise_std: float
        :param alpha: float
        :param rho: float, std of gaussian distribution for initialization
        :param train_wi: bool
        :param train_wo: bool
        :param train_wrec: bool
        :param train_h0: bool
        :param wi_init: torch tensor of shape (input_dim, hidden_size)ð
        :param wo_init: torch tensor of shape (hidden_size, output_dim)
        :param wrec_init: torch tensor of shape (hidden_size, hidden_size)
        :param si_init: input scaling, torch tensor of shape (input_dim)
        :param so_init: output scaling, torch tensor of shape (output_dim)
        """
        super(FullRankRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.noise_std = noise_std
        self.alpha = alpha
        self.rho = rho
        self.train_wi = train_wi
        self.train_wo = train_wo
        self.train_wrec = train_wrec
        self.train_h0 = train_h0
        self.non_linearity = torch.tanh

        # Define parameters
        self.wi = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.si = nn.Parameter(torch.Tensor(input_size))
        if train_wi:
            self.si.requires_grad = False
        else:
            self.wi.requires_grad = False
        self.wrec = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        if not train_wrec:
            self.wrec.requires_grad = False
        self.wo = nn.Parameter(torch.Tensor(hidden_size, output_size))
        self.so = nn.Parameter(torch.Tensor(output_size))
        if train_wo:
            self.so.requires_grad = False
        if not train_wo:
            self.wo.requires_grad = False
        self.h0 = nn.Parameter(torch.Tensor(hidden_size))
        if not train_h0:
            self.h0.requires_grad = False

        # Initialize parameters
        with torch.no_grad():
            if wi_init is None:
                self.wi.normal_()
            else:
                self.wi.copy_(wi_init)
            if si_init is None:
                self.si.set_(torch.ones_like(self.si))
            else:
                self.si.copy_(si_init)
            if wrec_init is None:
                self.wrec.normal_(std=rho / sqrt(hidden_size))
            else:
                self.wrec.copy_(wrec_init)
            if wo_init is None:
                self.wo.normal_(std=1 / sqrt(hidden_size))
            else:
                self.wo.copy_(wo_init)
            if so_init is None:
                self.so.set_(torch.ones_like(self.so))
            else:
                self.so.copy_(so_init)
            if h0_init is None:
                self.h0.zero_()
            else:
                self.h0.copy_(h0_init)
        self.wi_full, self.wo_full = [None] * 2
        self.define_proxy_parameters()

    def define_proxy_parameters(self):
        self.wi_full = (self.wi.t() * self.si).t()
        self.wo_full = self.wo * self.so

    def forward(self, input, return_dynamics=False, return_noise=False):
        """
        :param input: tensor of shape (batch_size, #timesteps, input_dimension)
        Important: the 3 dimensions need to be present, even if they are of size 1.
        :param return_dynamics: bool
        :return: if return_dynamics=False, output tensor of shape (batch_size, #timesteps, output_dimension)
                 if return_dynamics=True, (output tensor, trajectories tensor of shape (batch_size, #timesteps, #hidden_units))
        """
        batch_size = input.shape[0]
        seq_len = input.shape[1]
        h = self.h0
        r = self.non_linearity(h)
        self.define_proxy_parameters()
        noise = torch.randn(batch_size, seq_len, self.hidden_size, device=self.wrec.device)
        output = torch.zeros(batch_size, seq_len, self.output_size, device=self.wrec.device)
        if return_dynamics:
            trajectories = torch.zeros(batch_size, seq_len, self.hidden_size, device=self.wrec.device)

        # simulation loop
        for i in range(seq_len):
            h = h + self.noise_std * noise[:, i, :] + self.alpha * \
                (-h + r.matmul(self.wrec.t()) + input[:, i, :].matmul(self.wi_full))
            r = self.non_linearity(h)
            output[:, i, :] = r.matmul(self.wo_full)
            if return_dynamics:
                trajectories[:, i, :] = h

        if not return_dynamics and not return_noise:
            return output
        elif return_dynamics==True and not return_noise:
            return output, trajectories
        else :
            return output, trajectories, noise

    def clone(self):
        new_net = FullRankRNN(self.input_size, self.hidden_size, self.output_size, self.noise_std, self.alpha,
                              self.rho, self.train_wi, self.train_wo, self.train_wrec, self.train_h0,
                              self.wi, self.wo, self.wrec, self.si, self.so)
        return new_net

    def plot_eigenvalues(self):
        eig, _ = np.linalg.eig(self.w_rec.detach().numpy())
        ax = plt.axes()
        ax.scatter(np.real(eig), np.imag(eig))
        ax.axvline(1, color="red", alpha=0.5)
        ax.set_aspect(1)
        if ax.get_xlim()[0] > -1.1:
            ax.set_xlim(left=-1.1)
        if ax.get_xlim()[1] < 1.1:
            ax.set_xlim(right=1.1)
        if ax.get_ylim()[0] > -1.1:
            ax.set_ylim(bottom=-1.1)
        if ax.get_ylim()[1] < 1.1:
            ax.set_ylim(top=1.1)
        ax.set_title("Connectivity matrix eigenvalues")
        plt.show()

class LowRankRNN(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, noise_std, alpha, rho=0., rank=1,
                 train_wi=False, train_wo=False, train_wrec=True, train_h0=False,
                 wi_init=None, wo_init=None, m_init=None, n_init=None, si_init=None, so_init=None, h0_init=None):
        """
        :param input_size: int
        :param hidden_size: int
        :param output_size: int
        :param noise_std: float
        :param alpha: float
        :param rho: float, std of quenched noise matrix
        :param rank: int
        :param train_wi: bool
        :param train_wo: bool
        :param train_wrec: bool
        :param train_h0: bool
        :param wi_init: torch tensor of shape (input_dim, hidden_size)
        :param wo_init: torch tensor of shape (hidden_size, output_dim)
        :param m_init: torch tensor of shape (hidden_size, rank)
        :param n_init: torch tensor of shape (hidden_size, rank)
        :param si_init: input scaling, torch tensor of shape (input_dim)
        :param so_init: output scaling, torch tensor of shape (output_dim)
        :param h0_init: torch tensor of shape (hidden_size)
        """
        super(LowRankRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.noise_std = noise_std
        self.alpha = alpha
        self.rho = rho
        self.rank = rank
        self.train_wi = train_wi
        self.train_wo = train_wo
        self.train_wrec = train_wrec
        self.train_h0 = train_h0
        self.non_linearity = torch.tanh

        # Define parameters
        self.wi = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.si = nn.Parameter(torch.Tensor(input_size))
        if train_wi:
            self.si.requires_grad = False
        else:
            self.wi.requires_grad = False
        self.m = nn.Parameter(torch.Tensor(hidden_size, rank))
        self.n = nn.Parameter(torch.Tensor(hidden_size, rank))
        if not train_wrec:
            self.m.requires_grad = False
            self.n.requires_grad = False
        self.wo = nn.Parameter(torch.Tensor(hidden_size, output_size))
        self.so = nn.Parameter(torch.Tensor(output_size))
        if train_wo:
            self.so.requires_grad = False
        if not train_wo:
            self.wo.requires_grad = False
        self.h0 = nn.Parameter(torch.Tensor(hidden_size))
        if not train_h0:
            self.h0.requires_grad = False
        self.rec_noise = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.rec_noise.requires_grad = False

        # Initialize parameters
        with torch.no_grad():
            if wi_init is None:
                self.wi.normal_()
            else:
                self.wi.copy_(wi_init)
            if si_init is None:
                self.si.set_(torch.ones_like(self.si))
            else:
                self.si.copy_(si_init)
            if self.rho > 0:
                self.rec_noise.normal_(std=self.rho / sqrt(hidden_size))
            else:
                self.rec_noise.zero_()
            if m_init is None:
                self.m.normal_(std=1 / sqrt(hidden_size))
            else:
                self.m.copy_(m_init)
            if n_init is None:
                self.n.normal_(std=1 / sqrt(hidden_size))
            else:
                self.n.copy_(n_init)
            if wo_init is None:
                self.wo.normal_(std=1 / sqrt(hidden_size))
            else:
                self.wo.copy_(wo_init)
            if so_init is None:
                self.so.set_(torch.ones_like(self.so))
            else:
                self.so.copy_(so_init)
            if h0_init is None:
                self.h0.zero_()
            else:
                self.h0.copy_(h0_init)
        self.wrec, self.wi_full, self.wo_full = [None] * 3
        self.define_proxy_parameters()

    def define_proxy_parameters(self):
        self.wrec = self.m.matmul(self.n.t()) + self.rec_noise
        self.wi_full = (self.wi.t() * self.si).t()
        self.wo_full = self.wo * self.so

    def forward(self, input, return_dynamics=False):
        """
        :param input: tensor of shape (batch_size, #timesteps, input_dimension)
        Important: the 3 dimensions need to be present, even if they are of size 1.
        :param return_dynamics: boolean
        :return: if return_dynamics=False, output tensor of shape (batch_size, #timesteps, output_dimension)
                 if return_dynamics=True, (output tensor, trajectories tensor of shape (batch_size, #timesteps, #hidden_units))
        """
        batch_size = input.shape[0]
        seq_len = input.shape[1]
        h = self.h0
        r = self.non_linearity(h)
        self.define_proxy_parameters()
        noise = torch.randn(batch_size, seq_len, self.hidden_size, device=self.m.device)
        output = torch.zeros(batch_size, seq_len, self.output_size, device=self.m.device)
        if return_dynamics:
            trajectories = torch.zeros(batch_size, seq_len+1, self.hidden_size, device=self.m.device)
            trajectories[:, 0, :] = h

        # simulation loop
        for i in range(seq_len):
            h = h + self.noise_std * noise[:, i, :] + self.alpha * (-h + r.matmul(self.wrec.t()) +
                                                                    input[:, i, :].matmul(self.wi_full))
            r = self.non_linearity(h)
            output[:, i, :] = r.matmul(self.wo_full)
            if return_dynamics:
                trajectories[:, i+1, :] = h

        if not return_dynamics:
            return output
        else:
            return output, trajectories

    def clone(self):
        new_net = LowRankRNN(self.input_size, self.hidden_size, self.output_size, self.noise_std, self.alpha,
                             self.rho, self.rank, self.train_wi, self.train_wo, self.train_wrec, self.train_h0,
                             self.wi, self.wo, self.m, self.n, self.si, self.so)
        new_net.rec_noise.copy_(self.rec_noise)  # in case correlations with the noise were relevant
        new_net.define_proxy_parameters()
        return new_net

    def resample_connectivity_noise(self):
        self.rec_noise.normal_(std=self.rho / sqrt(self.hidden_size))
        self.define_proxy_parameters()

    def load_state_dict(self, state_dict, strict=True):
        """
        override to recompute w_rec on loading
        """
        super().load_state_dict(state_dict, strict)
        self.define_proxy_parameters()

    def svd_reparametrization(self):
        """
        Orthogonalize m and n via SVD
        """
        with torch.no_grad():
            structure = (self.m @ self.n.t()).numpy()
            m, s, n = np.linalg.svd(structure, full_matrices=False)
            m, s, n = m[:, :self.rank], s[:self.rank], n[:self.rank, :]
            self.m.set_(torch.from_numpy(m * np.sqrt(s)))
            self.n.set_(torch.from_numpy(n.transpose() * np.sqrt(s)))
            self.define_proxy_parameters()

class SupportLowRankRNN(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, noise_std, alpha, rank=1, n_supports=1, weights=None,
                 gaussian_basis_dim=None, m_weights_init=None, n_weights_init=None, wi_weights_init=None,
                 wo_weights_init=None, m_biases_init=None, n_biases_init=None, wi_biases_init=None):
        """

        :param input_size: int
        :param hidden_size: int
        :param output_size: int
        :param noise_std: float
        :param alpha: float
        :param rho: float, std of quenched noise matrix
        :param rank: int
        :param n_supports: int, number of cell classes used
        :param weights: list, proportion of total population for each cell class (GMM components weights)
        :param gaussian_basis_dim: dimensionality of the gaussian basis on which weights are learned
        :param m_weights_init: torch tensor of shape (rank, n_supports, gaussian_basis_dim)
        :param n_weights_init: torch tensor of shape (rank, n_supports, gaussian_basis_dim)
        :param wi_weights_init: torch tensor of shape (input_size, n_supports, self.gaussian_basis_dim)
        :param wo_weights_init: torch tensor of shape (output_size, n_supports, self.gaussian_basis_dim)
        :param m_biases_init: torch tensor of shape (rank, n_supports)
        :param n_biases_init: torch tensor of shape (rank, n_supports)
        :param wi_biases_init: torch tensor of shape (input_size, n_supports)
        """
        super(SupportLowRankRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.noise_std = noise_std
        self.alpha = alpha
        self.rank = rank
        self.n_supports = n_supports
        self.weights = weights
        self.gaussian_basis_dim = 2 * rank + input_size if gaussian_basis_dim is None else gaussian_basis_dim
        self.non_linearity = torch.tanh

        self.gaussian_basis = nn.Parameter(torch.randn((self.gaussian_basis_dim, hidden_size)), requires_grad=False)
        self.supports = nn.Parameter(torch.zeros((n_supports, hidden_size)), requires_grad=False)
        if self.weights is None:
            l_support = hidden_size // n_supports
            for i in range(n_supports):
                self.supports[i, l_support * i: l_support * (i + 1)] = 1
            self.weights = [l_support / hidden_size] * n_supports
        else:
            k = 0
            for i in range(n_supports):
                self.supports[i, k: k + floor(weights[i] * hidden_size)] = 1
                k += floor(weights[i] * hidden_size)

        # Define parameters
        self.wi_weights = nn.Parameter(torch.Tensor(input_size, n_supports, self.gaussian_basis_dim))
        self.m_weights = nn.Parameter(torch.Tensor(rank, n_supports, self.gaussian_basis_dim))
        self.n_weights = nn.Parameter(torch.Tensor(rank, n_supports, self.gaussian_basis_dim))
        self.wo_weights = nn.Parameter(torch.Tensor(output_size, n_supports, self.gaussian_basis_dim))
        self.wi_biases = nn.Parameter(torch.Tensor(input_size, n_supports))
        self.m_biases = nn.Parameter(torch.Tensor(rank, n_supports))
        self.n_biases = nn.Parameter(torch.Tensor(rank, n_supports))
        self.h0_weights = nn.Parameter(torch.Tensor(n_supports, self.gaussian_basis_dim))
        self.h0_weights.requires_grad = False

        # Initialize parameters
        with torch.no_grad():
            if wi_weights_init is not None:
                self.wi_weights.copy_(wi_weights_init)
            else:
                self.wi_weights.normal_()
            if m_weights_init is not None:
                self.m_weights.copy_(m_weights_init)
            else:
                self.m_weights.normal_(std=1 / sqrt(hidden_size))
            if n_weights_init is not None:
                self.n_weights.copy_(n_weights_init)
            else:
                self.n_weights.normal_(std=1 / sqrt(hidden_size))
            if wo_weights_init is not None:
                self.wo_weights.copy_(wo_weights_init)
            else:
                self.wo_weights.normal_(std=1 / hidden_size)
            if wi_biases_init is not None:
                self.wi_biases.copy_(wi_biases_init)
            else:
                self.wi_biases.zero_()
            if m_biases_init is not None:
                self.m_biases.copy_(m_biases_init)
            else:
                self.m_biases.zero_()
            if n_biases_init is not None:
                self.n_biases.copy_(n_biases_init)
            else:
                self.n_biases.zero_()
            self.h0_weights.zero_()
        self.wi, self.m, self.n, self.wo, self.h0, self.wi_full, self.wo_full = [None] * 7
        self.define_proxy_parameters()

    def define_proxy_parameters(self):
        self.wi = torch.sum((self.wi_weights @ self.gaussian_basis) * self.supports, dim=(1,)) + \
            self.wi_biases @ self.supports
        self.wi_full = self.wi
        self.m = torch.sum((self.m_weights @ self.gaussian_basis) * self.supports, dim=(1,)).t() + \
                 (self.m_biases @ self.supports).t()
        self.n = torch.sum((self.n_weights @ self.gaussian_basis) * self.supports, dim=(1,)).t() + \
                 (self.n_biases @ self.supports).t()
        self.wo = torch.sum((self.wo_weights @ self.gaussian_basis) * self.supports, dim=(1,)).t()
        self.wo_full = self.wo
        self.h0 = torch.sum((self.h0_weights @ self.gaussian_basis) * self.supports, dim=(0,))

    def forward(self, input, return_dynamics=False):
        batch_size = input.shape[0]
        seq_len = input.shape[1]
        self.define_proxy_parameters()
        h = self.h0
        r = self.non_linearity(h)
        noise = torch.randn(batch_size, seq_len, self.hidden_size, device=self.m.device)
        output = torch.zeros(batch_size, seq_len, self.output_size, device=self.m.device)
        if return_dynamics:
            trajectories = torch.zeros(batch_size, seq_len, self.hidden_size, device=self.m.device)

        # simulation loop
        for i in range(seq_len):
            h = h + self.noise_std * noise[:, i, :] + self.alpha * (-h + r.matmul(self.n).matmul(self.m.t()) +
                                                                    input[:, i, :].matmul(self.wi_full))
            r = self.non_linearity(h)
            output[:, i, :] = r.matmul(self.wo_full)
            if return_dynamics:
                trajectories[:, i, :] = h

        if not return_dynamics:
            return output
        else:
            return output, trajectories

    def clone(self):
        new_net = SupportLowRankRNN(self.input_size, self.hidden_size, self.output_size, self.noise_std, self.alpha,
                                    self.rank, self.n_supports, self.weights, self.gaussian_basis_dim, self.m_weights,
                                    self.n_weights, self.wi_weights, self.wo_weights, self.m_biases, self.n_biases,
                                    self.wi_biases)
        new_net.gaussian_basis.copy_(self.gaussian_basis)
        new_net.define_proxy_parameters()
        return new_net

    def load_state_dict(self, state_dict, strict=True):
        """
        override to recompute w_rec on loading
        """
        super().load_state_dict(state_dict, strict)
        self.define_proxy_parameters()

    def resample_basis(self):
        self.gaussian_basis.normal_()
        self.define_proxy_parameters()

    # def orthogonalize_basis(self):
    #     for i in range(self.n_supports):
    #         gaussian_chunk = self.gaussian_basis[:, self.supports[i] == 1].view(self.gaussian_basis_dim, -1)
    #         gram_schmidt_pt(gaussian_chunk)
    #         self.gaussian_basis[:, self.supports[i] == 1] = gaussian_chunk
    #     self.gaussian_basis *= sqrt(self.hidden_size // self.n_supports)
    #     self.define_proxy_parameters()

class OptimizedLowRankRNN(LowRankRNN):

    def __init__(self, input_size, hidden_size, output_size, noise_std, alpha, rho=0., rank=1,
                 train_wi=False, train_wo=False, train_wrec=True, train_h0=False,
                 wi_init=None, wo_init=None, m_init=None, n_init=None, si_init=None, so_init=None, h0_init=None):
        rho = 0.  # enforce no high-rank noise
        super(OptimizedLowRankRNN, self).__init__(input_size, hidden_size, output_size, noise_std, alpha, rho, rank,
                 train_wi, train_wo, train_wrec, train_h0, wi_init, wo_init, m_init, n_init, si_init, so_init, h0_init)
        self.rec_noise = None

    def define_proxy_parameters(self):
        self.wrec = None
        self.wi_full = (self.wi.t() * self.si).t()
        self.wo_full = self.wo * self.so

    def forward(self, input, return_dynamics=False, return_noise=False):
        """
        :param input: tensor of shape (batch_size, #timesteps, input_dimension)
        Important: the 3 dimensions need to be present, even if they are of size 1.
        :param return_dynamics: boolean
        :return: if return_dynamics=False, output tensor of shape (batch_size, #timesteps, output_dimension)
                 if return_dynamics=True, (output tensor, trajectories tensor of shape (batch_size, #timesteps, #hidden_units))
        """
        batch_size = input.shape[0]
        seq_len = input.shape[1]
        h = self.h0
        r = self.non_linearity(h)
        self.define_proxy_parameters()
        noise = torch.randn(batch_size, seq_len, self.hidden_size, device=self.m.device)
        output = torch.zeros(batch_size, seq_len, self.output_size, device=self.m.device)
        if return_dynamics:
            trajectories = torch.zeros(batch_size, seq_len+1, self.hidden_size, device=self.m.device)
            trajectories[:, 0, :] = h

        # simulation loop
        for i in range(seq_len):
            h = h + self.noise_std * noise[:, i, :] + self.alpha * (-h + r.matmul(self.n).matmul(self.m.t()) +
                                                                    input[:, i, :].matmul(self.wi_full))

            r = self.non_linearity(h)
            output[:, i, :] = r.matmul(self.wo_full)
            if return_dynamics:
                trajectories[:, i+1, :] = h

        if not return_dynamics and not return_noise:
            return output
        elif return_dynamics==True and not return_noise:
            return output, trajectories
        else :
            return output, trajectories, noise


    def clone(self):
        new_net = OptimizedLowRankRNN(self.input_size, self.hidden_size, self.output_size, self.noise_std, self.alpha,
                             0., self.rank, self.train_wi, self.train_wo, self.train_wrec, self.train_h0,
                             self.wi, self.wo, self.m, self.n, self.si, self.so)
        new_net.define_proxy_parameters()
        return new_net

    def load_state_dict(self, state_dict, strict=True):
        """
        override to recompute w_rec on loading
        """
        if 'rec_noise' in state_dict:
            del state_dict['rec_noise']
        super().load_state_dict(state_dict, strict)
        self.define_proxy_parameters()

class OptimizedLowRankRNN_PosNL(LowRankRNN):

    def __init__(self, input_size, hidden_size, output_size, noise_std, alpha, rho=0., rank=1,
                 train_wi=False, train_wo=False, train_wrec=True, train_h0=False,
                 wi_init=None, wo_init=None, m_init=None, n_init=None, si_init=None, so_init=None, h0_init=None):
        rho = 0.  # enforce no high-rank noise
        super(OptimizedLowRankRNN, self).__init__(input_size, hidden_size, output_size, noise_std, alpha, rho, rank,
                 train_wi, train_wo, train_wrec, train_h0, wi_init, wo_init, m_init, n_init, si_init, so_init, h0_init)
        self.rec_noise = None

    def define_proxy_parameters(self):
        self.wrec = None
        self.wi_full = (self.wi.t() * self.si).t()
        self.wo_full = self.wo * self.so

    def forward(self, input, return_dynamics=False):
        """
        :param input: tensor of shape (batch_size, #timesteps, input_dimension)
        Important: the 3 dimensions need to be present, even if they are of size 1.
        :param return_dynamics: boolean
        :return: if return_dynamics=False, output tensor of shape (batch_size, #timesteps, output_dimension)
                 if return_dynamics=True, (output tensor, trajectories tensor of shape (batch_size, #timesteps, #hidden_units))
        """
        batch_size = input.shape[0]
        seq_len = input.shape[1]
        h = self.h0
        r = self.non_linearity(h)
        self.define_proxy_parameters()
        noise = torch.randn(batch_size, seq_len, self.hidden_size, device=self.m.device)
        output = torch.zeros(batch_size, seq_len, self.output_size, device=self.m.device)
        if return_dynamics:
            trajectories = torch.zeros(batch_size, seq_len+1, self.hidden_size, device=self.m.device)
            trajectories[:, 0, :] = h

        # simulation loop
        for i in range(seq_len):
            h = h + self.noise_std * noise[:, i, :] + self.alpha * (-h + r.matmul(self.n).matmul(self.m.t()) +
                                                                    input[:, i, :].matmul(self.wi_full))
            r = self.non_linearity(h)
            output[:, i, :] = r.matmul(self.wo_full)
            if return_dynamics:
                trajectories[:, i+1, :] = h

        if not return_dynamics:
            return output
        else:
            return output, trajectories

    def clone(self):
        new_net = OptimizedLowRankRNN(self.input_size, self.hidden_size, self.output_size, self.noise_std, self.alpha,
                             0., self.rank, self.train_wi, self.train_wo, self.train_wrec, self.train_h0,
                             self.wi, self.wo, self.m, self.n, self.si, self.so)
        new_net.define_proxy_parameters()
        return new_net

    def load_state_dict(self, state_dict, strict=True):
        """
        override to recompute w_rec on loading
        """
        if 'rec_noise' in state_dict:
            del state_dict['rec_noise']
        super().load_state_dict(state_dict, strict)
        self.define_proxy_parameters()