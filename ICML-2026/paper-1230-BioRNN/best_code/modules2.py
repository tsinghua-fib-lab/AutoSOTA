import torch
import torch.nn as nn
from math import sqrt
import random
import numpy as np
import matplotlib.pyplot as plt
#from helpers import *

def loss_mse(output, target, mask):
    loss_tensor = (mask * (target - output)).pow(2).mean(dim=-1)
    loss_by_trial = loss_tensor.sum(dim=-1) / mask[:, :, 0].sum(dim=-1)
    return loss_by_trial.mean()

def train(net, _input, _target, _mask, n_epochs, lr=1e-2, batch_size=32, plot_learning_curve=False, plot_gradient=False,
          clip_gradient=None, keep_best=False, cuda=False):
    """
    Train a network
    :param net: nn.Module
    :param _input: torch tensor of shape (num_trials, num_timesteps, input_dim)
    :param _target: torch tensor of shape (num_trials, num_timesteps, output_dim)
    :param _mask: torch tensor of shape (num_trials, num_timesteps, 1)
    :param n_epochs: int
    :param lr: float, learning rate
    :param batch_size: float
    :param plot_learning_curve: bool
    :param plot_gradient: bool
    :param clip_gradient: None or float, if not None the value at which gradient norm is clipped
    :param keep_best: bool, if True, model with lower loss from training process will be kept (for this option, the
        network has to implement a method clone())
    :return:
    """
    print("Training...")
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    num_examples = _input.shape[0]
    all_losses = []
    if plot_gradient:
        gradient_norms = []
    if keep_best:
        best = net.clone()
        best_loss = float("inf")

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

    for epoch in range(n_epochs):
        losses = []
        for i in range(num_examples // batch_size):
            optimizer.zero_grad()
            random_batch_idx = random.sample(range(num_examples), batch_size)
            batch = input[random_batch_idx]
            output = net(batch)
            loss = loss_mse(output, target[random_batch_idx], mask[random_batch_idx])
            losses.append(loss.item())
            if keep_best and loss.item() < best_loss:
                best = net.clone()
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
        print("epoch %d:  loss=%.3f" % (epoch, np.mean(losses)))

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

class FullRankLeakyNoisyRNN(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, noise_std, alpha, rho=1,
                 train_wi=True, train_wo=True, train_wrec=True, train_h0=False,
                 initial_wi=None, initial_wo=None, initial_wrec=None,
                 initial_si=None, initial_so=None):
        """
        :param noise_std: standard deviation of recurrent white noise on the units (you have to include the scaling
        factor sqrt(dt/tau) yourself)
        :param rho: The scaling factor of connectivity noise
        """
        super(FullRankLeakyNoisyRNN, self).__init__()
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
        self.w_i = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.s_i = nn.Parameter(torch.Tensor(input_size))
        if train_wi:
            self.s_i.requires_grad = False
        else:
            self.w_i.requires_grad = False
        self.w_rec = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        if not train_wrec:
            self.w_rec.requires_grad = False
        self.w_o = nn.Parameter(torch.Tensor(hidden_size, output_size))
        self.s_o = nn.Parameter(torch.Tensor(output_size))
        if train_wo:
            self.s_o.requires_grad = False
        if not train_wo:
            self.w_o.requires_grad = False
        self.h0 = nn.Parameter(torch.Tensor(hidden_size))
        if not train_h0:
            self.h0.requires_grad = False

        # Initialize parameters
        with torch.no_grad():
            if initial_wi is None:
                self.w_i.normal_()
            else:
                self.w_i.copy_(initial_wi)
            if initial_si is None:
                self.s_i.set_(torch.ones_like(self.s_i))
            else:
                self.s_i.copy_(initial_si)
            if initial_wrec is None:
                self.w_rec.normal_(std=rho / sqrt(hidden_size))
            else:
                self.w_rec.copy_(initial_wrec)
            if initial_wo is None:
                self.w_o.normal_()
            else:
                self.w_o.copy_(initial_wo)
            if initial_so is None:
                self.s_o.set_(torch.ones_like(self.s_o))
            else:
                self.s_o.copy_(initial_so)
            self.h0.zero_()

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
        w_i_mat = (self.w_i.t() * self.s_i).t()
        w_o_mat = self.w_o * self.s_o
        noise = torch.randn(batch_size, seq_len, self.hidden_size, device=self.w_rec.device)
        output = torch.zeros(batch_size, seq_len, self.output_size, device=self.w_rec.device)
        if return_dynamics:
            trajectories = torch.zeros(batch_size, seq_len, self.hidden_size, device=self.w_rec.device)

        # simulation loop
        for i in range(seq_len):
            h = h + self.noise_std * noise[:, i, :] + self.alpha * (-h + r.matmul(self.w_rec.t()) + input[:, i, :].matmul(w_i_mat))
            r = self.non_linearity(h)
            output[:, i, :] = r.matmul(w_o_mat) / self.hidden_size
            if return_dynamics:
                trajectories[:, i, :] = h

        if not return_dynamics:
            return output
        else:
            return output, trajectories

    def clone(self):
        new_net = FullRankLeakyNoisyRNN(self.input_size, self.hidden_size, self.output_size, self.noise_std, self.alpha,
                                        self.rho, self.train_wi, self.train_wo, self.train_wrec, self.train_h0,
                                        self.w_i, self.w_o, self.w_rec)
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

    def response_plots(self, input):
        """
        plot responses to a batch of inputs
        """
        fig, axes = plt.subplots()
        output = self.forward(input)
        for i in range(input.shape[0]):
            axes.plot(output[i, :, 0].detach().numpy())
        axes.set_title("Outputs")
        plt.show()

class LowRankLeakyNoisyRNN(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, noise_std, alpha, rho=1, rank=1, correlated_noise=False,
                 train_wi=True, train_wo=True, train_wrec=True, train_h0=False,
                 initial_wi=None, initial_wo=None, initial_mrec=None, initial_nrec=None,
                 initial_si=None, initial_so=None, initial_h0=None):
        """
        :param noise_std: standard deviation of recurrent white noise on the units (you have to include the scaling
        factor sqrt(dt/tau) yourself)
        :param rho: The scaling factor of connectivity noise
        """
        super(LowRankLeakyNoisyRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.noise_std = noise_std
        self.alpha = alpha
        self.rho = rho
        self.rank = rank
        self.correlated_noise = correlated_noise
        self.train_wi = train_wi
        self.train_wo = train_wo
        self.train_wrec = train_wrec
        self.train_h0 = train_h0
        self.non_linearity = torch.tanh

        # Define parameters
        self.w_i = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.s_i = nn.Parameter(torch.Tensor(input_size))
        if train_wi:
            self.s_i.requires_grad = False
        else:
            self.w_i.requires_grad = False
        self.m_rec = nn.Parameter(torch.Tensor(hidden_size, rank))
        self.n_rec = nn.Parameter(torch.Tensor(hidden_size, rank))
        if not train_wrec:
            self.m_rec.requires_grad = False
            self.n_rec.requires_grad = False
        self.w_o = nn.Parameter(torch.Tensor(hidden_size, output_size))
        self.s_o = nn.Parameter(torch.Tensor(output_size))
        if train_wo:
            self.s_o.requires_grad = False
        if not train_wo:
            self.w_o.requires_grad = False
        self.h0 = nn.Parameter(torch.Tensor(hidden_size))
        if not train_h0:
            self.h0.requires_grad = False
        self.rec_noise = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.rec_noise.requires_grad = False

        # Initialize parameters
        with torch.no_grad():
            if initial_wi is None:
                self.w_i.normal_()
            else:
                self.w_i.copy_(initial_wi)
            if initial_si is None:
                self.s_i.set_(torch.ones_like(self.s_i))
            else:
                self.s_i.copy_(initial_si)
            self.rec_noise.normal_(std=self.rho / sqrt(hidden_size))
            if initial_mrec is None:
                self.m_rec.normal_()
            else:
                self.m_rec.copy_(initial_mrec)
            if initial_nrec is None:
                self.n_rec.normal_()
            else:
                self.n_rec.copy_(initial_nrec)
            if initial_wo is None:
                self.w_o.normal_()
            else:
                self.w_o.copy_(initial_wo)
            if initial_so is None:
                self.s_o.set_(torch.ones_like(self.s_o))
            else:
                self.s_o.copy_(initial_so)
            if initial_h0 is None:
                self.h0.zero_()
            else:
                self.h0.copy_(initial_h0)
        self.w_rec, self.wi_full, self.wo_full = [None]*3
        self.define_proxy_parameters()

    def define_proxy_parameters(self):
        self.w_rec = (self.m_rec.matmul(self.n_rec.t()) / self.hidden_size) + self.rec_noise
        self.wi_full = (self.w_i.t() * self.s_i).t()
        self.wo_full = self.w_o * self.s_o

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
        noise = torch.randn(batch_size, seq_len, self.hidden_size, device=self.m_rec.device)
        output = torch.zeros(batch_size, seq_len, self.output_size, device=self.m_rec.device)
        if return_dynamics:
            trajectories = torch.zeros(batch_size, seq_len, self.hidden_size, device=self.m_rec.device)

        # simulation loop
        for i in range(seq_len):
            h = h + self.noise_std * noise[:, i, :] + self.alpha * (-h + r.matmul(self.w_rec.t()) +
                                                                    input[:, i, :].matmul(self.wi_full))
            r = self.non_linearity(h)
            output[:, i, :] = r.matmul(self.wo_full) / self.hidden_size
            if return_dynamics:
                trajectories[:, i, :] = h

        if not return_dynamics:
            return output
        else:
            return output, trajectories

    def clone(self):
        new_net = LowRankLeakyNoisyRNN(self.input_size, self.hidden_size, self.output_size, self.noise_std, self.alpha,
                                       self.rho, self.rank, self.correlated_noise, self.train_wi, self.train_wo,
                                       self.train_wrec, self.train_h0, self.w_i, self.w_o, self.m_rec, self.n_rec,
                                       self.s_i, self.s_o)
        return new_net

    def resample_connectivity_noise(self):
        if self.correlated_noise:
            print("Error: one should resample the noise only for networks that are uncorrelated")
        else:
            self.rec_noise = torch.Tensor(self.hidden_size, self.hidden_size)
            self.rec_noise.normal_(std=self.rho / sqrt(self.hidden_size))
            self.define_proxy_parameters()

    def load_state_dict(self, state_dict, strict=True):
        """
        override to recompute w_rec on loading
        """
        super().load_state_dict(state_dict, strict)
        self.define_proxy_parameters()

    def plot_eigenvalues(self, with_noise=False):
        if with_noise:
            fig, axes = plt.subplots(1, 2)
        else:
            fig, axes = plt.subplots(1, 1)
            axes = [axes]
        eig, _ = np.linalg.eig(self.w_rec.detach())
        axes[0].scatter(np.real(eig), np.imag(eig))
        axes[0].axvline(1, color="red", alpha=0.5)
        axes[0].set_aspect(1)
        axes[0].set_axis_on()
        axes[0].set_title("Trained matrix eigenvalues")
        if with_noise:
            eign, _ = np.linalg.eig(self.rec_noise.numpy())
            axes[1].scatter(np.real(eign), np.imag(eign))
            axes[1].axvline(1, color="red", alpha=0.5)
            axes[1].set_aspect(1)
            axes[1].set_xlim(axes[0].get_xlim())
            axes[1].set_axis_on()
            axes[1].set_title("Noise matrix eigenvalues")
        if axes[0].get_xlim()[0] > -1.1:
            axes[0].set_xlim(left=-1.1)
        if axes[0].get_xlim()[1] < 1.1:
            axes[0].set_xlim(right=1.1)
        if axes[0].get_ylim()[0] > -1.1:
            axes[0].set_ylim(bottom=-1.1)
        if axes[0].get_ylim()[1] < 1.1:
            axes[0].set_ylim(top=1.1)
        plt.show()

    def response_plots(self, input):
        """
        plot responses to a batch of inputs
        """

        fig, axes = plt.subplots(1, 1)
        output = self.forward(input)
        for i in range(input.shape[0]):
            axes.plot(output[i, :, 0].detach().numpy())
        plt.show()

    def svd_reparametrization(self):
        """
        Orthogonalize m_rec and n_rec via SVD
        """
        with torch.no_grad():
            structure = (self.m_rec @ self.n_rec.t()).numpy()
            m, s, n = np.linalg.svd(structure, full_matrices=False)
            m, s, n = m[:, :self.rank], s[:self.rank], n[:self.rank, :]
            self.m_rec.set_(torch.from_numpy(m * np.sqrt(s)))
            self.n_rec.set_(torch.from_numpy(n.transpose() * np.sqrt(s)))
            self.w_rec = (self.m_rec @ self.n_rec.t()) / self.hidden_size + self.rec_noise

class HandMadeTrainableRNN(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, noise_std, alpha,
                 overlap_wi_initial=1, overlap_wo_initial=1, overlap_mn_initial=1, rho=1,
                 train_ov_wi=True, train_ov_wo=True, train_ov_mn=True):
        super(HandMadeTrainableRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.noise_std = noise_std
        self.alpha = alpha
        self.non_linearity = torch.tanh

        self.wi_flat = torch.randn(hidden_size, 1)
        self.wi = self.wi_flat.view(1, hidden_size)
        self.wo = torch.randn(hidden_size, 1)
        self.A = torch.randn(hidden_size, 1)
        self.A /= np.linalg.norm(self.A)
        self.h0 = torch.Tensor(hidden_size)
        self.rec_noise = torch.Tensor(hidden_size, hidden_size)

        if train_ov_wi:
            self.overlap_wi = nn.Parameter(torch.Tensor([overlap_wi_initial]))
        else:
            self.overlap_wi = torch.Tensor([overlap_wi_initial])
        if train_ov_wo:
            self.overlap_wo = nn.Parameter(torch.Tensor([overlap_wo_initial]))
        else:
            self.overlap_wo = torch.Tensor([overlap_wo_initial])
        if train_ov_mn:
            self.overlap_mn = nn.Parameter(torch.Tensor([overlap_mn_initial]))
        else:
            self.overlap_mn = torch.Tensor([overlap_mn_initial])
        self.h0.zero_()
        self.rec_noise.normal_(std=rho / sqrt(hidden_size))

        self.n_rec = (self.overlap_wi * self.hidden_size / self.wi.norm() ** 2) * self.wi_flat + \
                      torch.sqrt(self.overlap_mn * self.hidden_size) * self.A
        self.m_rec = (self.overlap_wo * self.hidden_size / self.wo.norm() ** 2) * self.wo + \
                      torch.sqrt(self.overlap_mn * self.hidden_size) * self.A

    def forward(self, input, return_dynamics=False):
        batch_size = input.shape[0]
        seq_len = input.shape[1]
        h = self.h0
        r = self.non_linearity(h)
        self.n_rec = (self.overlap_wi * self.hidden_size / self.wi.norm() ** 2) * self.wi_flat + \
                     torch.sqrt(self.overlap_mn * self.hidden_size) * self.A
        self.m_rec = (self.overlap_wo * self.hidden_size / self.wo.norm() ** 2) * self.wo + \
                     torch.sqrt(self.overlap_mn * self.hidden_size) * self.A
        w_rec = (self.n_rec.matmul(self.m_rec.t()) / self.hidden_size) + self.rec_noise
        noise = torch.randn(batch_size, seq_len, self.hidden_size)
        output = torch.zeros(batch_size, seq_len, self.output_size)
        if return_dynamics:
            hiddens = [np.tile(h.clone().detach().numpy(), (batch_size, 1))]

        for i in range(seq_len):
            h = h + self.noise_std * noise[:, i, :] + self.alpha * (-h + r.matmul(w_rec) + input[:, i, :].matmul(self.wi))
            r = self.non_linearity(h)
            output[:, i, :] = r.matmul(self.wo) / self.hidden_size
            if return_dynamics:
                hiddens.append(h.clone().detach().numpy())

        if not return_dynamics:
            return output
        else:
            # Manipulations to have hidden states from a same batch contiguous in the result
            hidden_states = np.stack(hiddens)
            hidden_states = np.swapaxes(hidden_states, 0, 1)
            hidden_states = np.reshape(hidden_states, ((seq_len + 1) * batch_size, self.hidden_size))
            return output, hidden_states

class HandmadeRank2Net(LowRankLeakyNoisyRNN):
    """
    Can only take one input
    """

    def __init__(self, hidden_size, overlaps, diagonal, bias_percentages, input_properties, noise_std=0.01,
                 alpha=0.2):
        """
        :param hidden_size:
        :param overlaps: overlaps n1-m1, n2-m2, n2-m1, n1-m2
        :param diagonal: orthogonal component on m1, m2
        :param bias_percentages:
        :param input_properties: overlaps of input with m1, m2, n1, n2
        :param noise_std:
        :param alpha:
        """
        gaussian_basis = torch.randn(11, hidden_size)
        bernoulli_basis = 1 - 2 * torch.randint(0, 2, size=(4, hidden_size)).type(torch.float32)

        z1 = sqrt(bias_percentages[0]) * bernoulli_basis[0] + sqrt(1 - bias_percentages[0]) * gaussian_basis[2]
        z2 = sqrt(bias_percentages[1]) * bernoulli_basis[1] + sqrt(1 - bias_percentages[1]) * gaussian_basis[3]
        z3 = sqrt(bias_percentages[2]) * bernoulli_basis[2] + sqrt(1 - bias_percentages[2]) * gaussian_basis[4]
        z4 = sqrt(bias_percentages[3]) * bernoulli_basis[3] + sqrt(1 - bias_percentages[3]) * gaussian_basis[5]
        m1 = diagonal[0] * gaussian_basis[0] + sqrt(abs(overlaps[0])) * z1 + sqrt(abs(overlaps[2])) * z3 + \
             gaussian_basis[6]
        m2 = diagonal[1] * gaussian_basis[1] + sqrt(abs(overlaps[1])) * z2 + sqrt(abs(overlaps[3])) * z4 + \
             gaussian_basis[7]
        n1 = sign(overlaps[0]) * sqrt(abs(overlaps[0])) * z1 + sign(overlaps[3]) * sqrt(abs(overlaps[3])) * z4 + \
             gaussian_basis[8]
        n2 = sign(overlaps[1]) * sqrt(abs(overlaps[1])) * z2 + sign(overlaps[2]) * sqrt(abs(overlaps[2])) * z3 + \
             gaussian_basis[9]

        wi = input_properties[0] * gaussian_basis[6] + input_properties[1] * gaussian_basis[7] + \
             input_properties[2] * gaussian_basis[8] + input_properties[3] * gaussian_basis[9] + \
             input_properties[4] * gaussian_basis[10]

        m = torch.empty(hidden_size, 2)
        n = torch.empty(hidden_size, 2)
        m[:, 0] = m1
        m[:, 1] = m2
        n[:, 0] = n1
        n[:, 1] = n2

        super(HandmadeRank2Net, self).__init__(1, hidden_size, 1, noise_std, alpha, rho=0, rank=2,
                                               initial_wi=wi, initial_mrec=m, initial_nrec=n)

class SupportRank2Net(LowRankLeakyNoisyRNN):

    def __init__(self, input_size, hidden_size, n_supports, m1, m2, n1, n2, wis, gaussian_basis_dim=None,
                 noise_std=0.01, alpha=0.2):
        """
        m1, m2, n1, n2: tensors of shape (#supports, gaussian_basis_dim)
        wis: tensor of shape (input_dim, #supports, gaussian_basis_dim)
        gaussian_basis_dim: if None, will equal 4+input_size
        """
        if gaussian_basis_dim is None:
            gaussian_basis_dim = 4 + input_size
        gaussian_basis = torch.randn(gaussian_basis_dim, hidden_size)
        supports = torch.zeros(n_supports, hidden_size)
        l_support = hidden_size // n_supports
        for i in range(n_supports):
            supports[i, l_support * i: l_support * (i+1)] = 1

        m1_full = torch.sum((m1 @ gaussian_basis) * supports, dim=(0,))
        m2_full = torch.sum((m2 @ gaussian_basis) * supports, dim=(0,))
        n1_full = torch.sum((n1 @ gaussian_basis) * supports, dim=(0,))
        n2_full = torch.sum((n2 @ gaussian_basis) * supports, dim=(0,))

        m = torch.empty(hidden_size, 2)
        n = torch.empty(hidden_size, 2)
        m[:, 0] = m1_full
        m[:, 1] = m2_full
        n[:, 0] = n1_full
        n[:, 1] = n2_full

        wis_full = torch.sum((wis @ gaussian_basis) * supports, dim=(1,))

        super(SupportRank2Net, self).__init__(input_size, hidden_size, 1, noise_std, alpha, rho=0, rank=2,
                                              initial_wi=wis_full, initial_mrec=m, initial_nrec=n)

class SupportLowRankRNN(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, noise_std, alpha, rank=1, n_supports=1,
                 gaussian_basis_dim=None, m_init=None, n_init=None, wi_init=None, wo_init=None):
        super(SupportLowRankRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.noise_std = noise_std
        self.alpha = alpha
        self.rank = rank
        self.n_supports = n_supports
        self.gaussian_basis_dim = 2 * rank + input_size if gaussian_basis_dim is None else gaussian_basis_dim
        self.non_linearity = torch.tanh

        self.gaussian_basis = nn.Parameter(torch.randn((self.gaussian_basis_dim, hidden_size)), requires_grad=False)
        self.supports = nn.Parameter(torch.zeros((n_supports, hidden_size)), requires_grad=False)
        l_support = hidden_size // n_supports
        for i in range(n_supports):
            self.supports[i, l_support * i: l_support * (i + 1)] = 1

        # Define parameters
        self.wi = nn.Parameter(torch.Tensor(input_size, n_supports, self.gaussian_basis_dim))
        self.m = nn.Parameter(torch.Tensor(rank, n_supports, self.gaussian_basis_dim))
        self.n = nn.Parameter(torch.Tensor(rank, n_supports, self.gaussian_basis_dim))
        self.wo = nn.Parameter(torch.Tensor(output_size, n_supports, self.gaussian_basis_dim))
        self.h0 = nn.Parameter(torch.zeros(hidden_size))
        self.h0.requires_grad = False

        # Initialize parameters
        with torch.no_grad():
            if wi_init is not None:
                self.wi.copy_(wi_init)
            else:
                self.wi.normal_()
            if m_init is not None:
                self.m.copy_(m_init)
            else:
                self.m.normal_()
            if n_init is not None:
                self.n.copy_(n_init)
            else:
                self.n.normal_()
            if wo_init is not None:
                self.wo.copy_(wo_init)
            else:
                self.wo.normal_()
        self.wi_full, self.m_rec, self.n_rec, self.wo_full, self.w_rec = [None]*5
        self.define_proxy_parameters()

    def define_proxy_parameters(self):
        self.wi_full = torch.sum((self.wi @ self.gaussian_basis) * self.supports, dim=(1,))
        self.m_rec = torch.sum((self.m @ self.gaussian_basis) * self.supports, dim=(1,)).t()
        self.n_rec = torch.sum((self.n @ self.gaussian_basis) * self.supports, dim=(1,)).t()
        self.wo_full = torch.sum((self.wo @ self.gaussian_basis) * self.supports, dim=(1,)).t()
        self.w_rec = (self.m_rec.matmul(self.n_rec.t()) / self.hidden_size)

    def forward(self, input, return_dynamics=False):
        batch_size = input.shape[0]
        seq_len = input.shape[1]
        h = self.h0
        r = self.non_linearity(h)
        self.define_proxy_parameters()
        noise = torch.randn(batch_size, seq_len, self.hidden_size, device=self.m_rec.device)
        output = torch.zeros(batch_size, seq_len, self.output_size, device=self.m_rec.device)
        if return_dynamics:
            trajectories = torch.zeros(batch_size, seq_len, self.hidden_size, device=self.m_rec.device)

        # simulation loop
        for i in range(seq_len):
            h = h + self.noise_std * noise[:, i, :] + self.alpha * (-h + r.matmul(self.w_rec.t()) +
                                                                    input[:, i, :].matmul(self.wi_full))
            r = self.non_linearity(h)
            output[:, i, :] = r.matmul(self.wo_full) / self.hidden_size
            if return_dynamics:
                trajectories[:, i, :] = h

        if not return_dynamics:
            return output
        else:
            return output, trajectories

    def clone(self):
        new_net = SupportLowRankRNN(self.input_size, self.hidden_size, self.output_size, self.noise_std, self.alpha,
                                    self.rank, self.n_supports, self.gaussian_basis_dim, self.m, self.n, self.wo,
                                    self.wo)
        new_net.gaussian_basis.copy_(self.gaussian_basis)
        new_net.define_proxy_parameters()
        return new_net

    def load_state_dict(self, state_dict, strict=True):
        """
        override to recompute w_rec on loading
        """
        super().load_state_dict(state_dict, strict)
        self.define_proxy_parameters()

