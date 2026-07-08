import os
from jinja2.nodes import Break
from sklearn.decomposition import NMF
import modules4 as md
# import lib_rnns as lr
# import tools_MF as tm
from mpl_toolkits.mplot3d import Axes3D
from math import sqrt
import random
import time as Time
import funcs_Sphere as fs
import scipy.stats
import scipy
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.stats import lognorm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
from scipy import stats
from matplotlib.colors import LinearSegmentedColormap, SymLogNorm
from matplotlib import colors
from matplotlib import cm
from datetime import datetime
from scipy.stats import kruskal

def run_net(dims, dt, net, batch_size=1, test=True, return_dynamics=False, h_init=None):
    # Generate batch
    input, target, mask = perceptual_vs(dims, dt, batch_size, noise=0.5*std_noise_rec, return_ts=False, test=test)

    # Convert training data1 to pytorch tensors
    # input = torch.from_numpy(input)
    # target = torch.from_numpy(target)
    # mask = torch.from_numpy(mask)
    with torch.no_grad():
        # Run dynamics
        if return_dynamics:
            output, trajectories = net(input, return_dynamics, h_init=h_init)
        else:
            output = net(input)
        loss = loss_mse(output, target, mask)
    res = [input, target, mask, output, loss]
    if return_dynamics:
        res.append(trajectories)
    res = [r.numpy() for r in res]
    return res

def plot_weight_matrix(wrec, size):
    """
       Plot the heat map of the weight matrix

       parameters:
           wrec: Weight matrix (numpy array)
           size: Matrix size (int)
    """
    # Create a color map
    colors = [(0.16, 0.47, 0.95), (1, 1, 1), (0.92, 0.23, 0.18)]
    cmap = LinearSegmentedColormap.from_list('synaptic', colors, N=size)
    fig_width = 1.5 * 2.2  # width in inches
    fig_height = 1.5 * 2  # height in inches
    fig_size = [fig_width, fig_height]
    fig = plt.figure(figsize=fig_size)
    # Create a canvas
    ax = fig.add_subplot(111)

    # Draw a heat map
    im = ax.imshow(wrec, cmap=cmap, aspect='equal',
                   vmin=-0.15, vmax=0.15,
                   interpolation='none')

    # Add vertical dividing line (front 80% and back 20% junction)
    vertical_line = 0.8 * size - 0.5  # Accurately calculate the boundary location
    ax.axvline(vertical_line, color='#444444',
               linestyle='--', linewidth=1.5, alpha=0.9)

    # Add diagonal demarcation line (no self-concatenation)
    ax.plot([-0.5, size - 0.5], [-0.5, size - 0.5],
            color='#666666', linestyle='--',
            linewidth=1.2, alpha=0.8)

    # Axis settings
    ax.set_xticks(np.linspace(0, size, 5))
    ax.set_yticks(np.linspace(0, size, 5))
    ax.set_xticklabels(['0', '64', '128', '192', '256'])
    ax.set_yticklabels(['0', '64', '128', '192', '256'])

    # Axis label
    ax.set_xlabel("Excitatory Synaptic Neuron", labelpad=10, fontsize=12)
    ax.set_ylabel("Postsynaptic Neuron", labelpad=10, fontsize=12)

    # Add inhibitive dimensions
    ax.text(0.85 * size, 0.5 * size, 'Inhibitory',
            ha='left', va='top', color='darkblue',
            fontsize=15, transform=ax.transData)
    ax.text(0.35 * size, 0.5 * size, 'Excitatory',
            ha='left', va='top', color='darkblue',
            fontsize=15, transform=ax.transData)

    # Color bar settings
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.set_ticks([-0.15, -0.10, -0.05, 0, 0.05, 0.10, 0.15])
    cbar.set_label('Synaptive Weight', rotation=270, labelpad=15, fontsize=12)

    # Title
    plt.title("Wrec (No Self-connections)",
              pad=20, fontsize=14)

    plt.tight_layout()
    plt.show()

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def set_plot(ll=7):
    '''
    Set plotting parameters. Returns colors for plots

    Parameters
    ----------
    ll : int, optional
        Number of colors. 5 or 7. The default is 7.

    Returns
    -------
    clS : colors

    '''

    """Base style and size"""
    plt.style.use('ggplot')  # Use ggplot classic style (light gray background + grid)

    fig_width = 1.5 * 2.2  # width in inches# Graphic width (inches), about 3.3 inches
    fig_height = 1.5 * 2  # height in inches# Figure height (inches), about 3 inches
    fig_size = [fig_width, fig_height]
    plt.rcParams['figure.figsize'] = fig_size  # Set the size of the graph
    plt.rcParams['figure.autolayout'] = True  # Automatically adjust the subgraph layout

    """Line and marker parameters"""
    plt.rcParams['lines.linewidth'] = 1.2  # Line width
    plt.rcParams['lines.markeredgewidth'] = 0.003  # Mark edge width (very fine)
    plt.rcParams['lines.markersize'] = 3  # Mark size

    """Font and axis parameters"""
    plt.rcParams['font.size'] = 14  # 9# Global font size
    plt.rcParams['legend.fontsize'] = 12  # 7. # Legend font size


    """Axis and background"""
    plt.rcParams['axes.facecolor'] = '1'  # The background color of the coordinate axis is white when the plot is set up ('1' means white)
    plt.rcParams['axes.edgecolor'] = '0'  ## The edge color of the axis is black when setting up the drawing ('0' means black)
    plt.rcParams['axes.linewidth'] = '0.7'  # Set the edge line width of the coordinate axis to 0.7 when drawing

    plt.rcParams['axes.labelcolor'] = '0'
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 14  # 9# Axis label font size
    plt.rcParams['xtick.labelsize'] = 12  # 7 x-axis scale font size
    plt.rcParams['ytick.labelsize'] = 12  # 7# Y-axis scale font size

    plt.rcParams['xtick.color'] = '0'  # Set the color of the x-axis scale to black
    plt.rcParams['ytick.color'] = '0'  # Set the color of the y-axis scale to black
    plt.rcParams['xtick.major.size'] = 2  # Set the size of the main scale line on the x-axis to 2
    plt.rcParams['ytick.major.size'] = 2  # Set the size of the main tick line on the y-axis to 2
    # plt.rcParams['font.sans-serif'] = 'Times New Roman'  # Set the font to Times New Roman
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

    # Key: Make the math formula use Times (via LaTeX's mathptmx macro) as well
    plt.rcParams['text.usetex'] = True
    plt.rcParams['text.latex.preamble'] = r'\usepackage{mathptmx}'

    """Color scheme definition"""
    clS = np.zeros((ll, 3))

    cl11 = np.array((102, 153, 255)) / 255.
    cl12 = np.array((53, 153, 53)) / 255.

    cl21 = np.array((255, 204, 51)) / 255.
    cl22 = np.array((204, 0, 0)) / 255.

    if ll == 7:
        clS[0, :] = 0.4 * np.ones((3,))

        clS[1, :] = cl11
        clS[2, :] = 0.5 * cl11 + 0.5 * cl12
        clS[3, :] = cl12

        clS[4, :] = cl21
        clS[5, :] = 0.5 * cl21 + 0.5 * cl22
        clS[6, :] = cl22

        clS = clS[1:]
        clS = clS[::-1]

        c2 = [67 / 256, 90 / 256, 162 / 256]
        c1 = [220 / 256, 70 / 256, 51 / 256]
        clS[0, :] = c1
        clS[5, :] = c2
    elif ll == 5:
        clS[0, :] = 0.4 * np.ones((3,))

        clS[2, :] = cl12

        clS[3, :] = cl21

        clS[4, :] = cl22
    return (clS)

class DScoSGD:
    """
    DScoSGD (Distribution-Based Stochastic Gradient Descent) optimizer class
    This class implements a special weight update strategy that optimizes the network by mapping the
    weight distribution to the target log-normal distribution
    """

    def __init__(self, net, mu_e, mu_i, loc_e, loc_i, sigma_e, sigma_i, learning_rate2):
        """
        Initialize the DScoSGD optimizer

        Parameters:
            net: Neural network to optimize
            mu_e: Lognormal mean parameter of the excitability weight
            mu_i: Lognormal distribution mean parameters of the inhibitory weights
            loc_e: Lognormal distribution position parameters of excitability weights
            loc_i: Lognormal distribution position parameters of the inhibitory weights
            sigma_e: Lognormal distribution standard deviation parameter of excitability weights
            sigma_i: Lognormal distribution standard deviation parameter of inhibitory weights
            learning_rate2: The mixing rate of weight updates controls the mixing ratio of new and old weights
        """
        self.net = net
        self.learning_rate2 = learning_rate2
        self.mu_e = mu_e
        self.mu_i = mu_i
        self.loc_e = loc_e
        self.loc_i = loc_i
        self.sigma_e = sigma_e
        self.sigma_i = sigma_i

    def transform_matrix(self, matrix):
        """
        Convert the weight distribution of the input matrix to a target lognormal distribution

        Parameters:
            matrix: The input weight matrix(torch.Tensor)

        return:
            The converted weight matrix(torch.Tensor)
        """
        # Convert the PyTorch tensor to a numpy array for processing
        matrix_np = matrix.cpu().numpy()
        # Initialize the result matrix
        result = np.zeros_like(matrix_np)

        # ====================================Positive weight processing part ====================================
        # Create a positive weight mask
        positive_mask = matrix_np > 0
        # Extract all positive weights
        positive = matrix_np[positive_mask]
        if len(positive) > 0:
            # Computation of the Cumulative Distribution Function (ECDF)
            ecdf = ECDF(positive)
            # Get the ECDF value for each positive weight
            u = np.array(ecdf(positive))

            # Generate a random sample from the target lognormal distribution
            # Using excitatory parameters (mu_e, loc_e, sigma_e)
            lognorm_dist = np.array(lognorm(s=self.sigma_e, loc=self.loc_e, scale=np.exp(self.mu_e)).rvs(len(u)))

            # Maintain the ordering relationship of the original weights
            u_sorted_indices = np.argsort(u)
            lognorm_dist_sorted = np.sort(lognorm_dist)
            new_lognorm_dist = np.empty_like(lognorm_dist)
            new_lognorm_dist[u_sorted_indices] = lognorm_dist_sorted
            target = new_lognorm_dist

            # Linear interpolation mixes raw weights and target distributions
            blended = positive * (1 - self.learning_rate2) + target * self.learning_rate2
            result[positive_mask] = blended

        # ====================================Negative weight processing part ====================================
        # Create a negative weight mask
        negative_mask = matrix_np < 0
        # Extract the absolute values of all negative weights
        negative = -matrix_np[negative_mask]
        if len(negative) > 0:
            # Calculate the cumulative distribution function of experience(ECDF)
            ecdf = ECDF(negative)
            # Get the ECDF value for each negative weight absolute value
            u = np.array(ecdf(negative))

            # Generate a random sample from the target lognormal distribution
            # Use inhibitory parameters(mu_i, loc_i, sigma_i)
            lognorm_dist = np.array(lognorm(s=self.sigma_i, loc=self.loc_i, scale=np.exp(self.mu_i)).rvs(len(u)))

            # Maintain the ordering relationship of the original weights
            u_sorted_indices = np.argsort(u)
            lognorm_dist_sorted = np.sort(lognorm_dist)
            new_lognorm_dist = np.empty_like(lognorm_dist)
            new_lognorm_dist[u_sorted_indices] = lognorm_dist_sorted
            target = new_lognorm_dist

            # Linear interpolation mixes raw weights and target distributions
            blended = negative * (1 - self.learning_rate2) + target * self.learning_rate2
            # Restore the negative sign
            result[negative_mask] = -blended

        # Convert the result back to the PyTorch tensor, keeping the original device type
        return torch.from_numpy(result).to(matrix.device).type_as(matrix)

    def apply(self):
        """
        Apply weights to a plasticity weight matrix for the network
        This method modifies the wrec_plastic weights of the network
        """
        # If the learning rate is too small, it will not be updated
        if self.learning_rate2 <= 1e-6:
            return

        # Weight updates without calculating gradients
        with torch.no_grad():
            # Obtain the plasticity weight matrix of the network
            wrec_plastic = self.net.wrec_plastic
            # Get the symbol of the weight
            sign = torch.sign(wrec_plastic)
            # Apply D_mask and take ReLU to ensure that it is not negative
            weight = self.net.D_mask * torch.relu(wrec_plastic)
            # Transform the weight matrix
            transformed = torch.abs(self.transform_matrix(weight.data))
            # Update the weights to keep the original symbol intact
            wrec_plastic.data.copy_(transformed * sign)  # Only the weight value range is adjusted

def create_wrec_init(hidden_size):
    w0_exc = np.random.normal(0, 1 / hidden_size, (hidden_size, int(0.8 * hidden_size)))
    w0_inh = np.random.normal(0, 1 / hidden_size, (hidden_size, hidden_size - int(0.8 * hidden_size)))
    w0_exc = np.abs(w0_exc)
    w0_inh = np.abs(w0_inh)
    # Make sure there are no self-concatenations (diagonal elements are 0)
    np.fill_diagonal(w0_exc[:, :int(0.8 * hidden_size)], 0)
    np.fill_diagonal(w0_inh[int(0.8 * hidden_size):, :], 0)
    # # # Calculate the mean of excitability and inhibitory weights
    # mean_exc = np.mean(np.abs(w0_exc))
    # mean_inh = np.mean(np.abs(w0_inh))
    # # # Balance excitatory and inhibitory inputs
    # if mean_exc != 0 and mean_inh != 0:
    #     balance_factor = mean_exc / mean_inh
    #     mean_inh *= balance_factor
    # # Combined excitatory and inhibitory weights
    w0_rec_plus = np.hstack((w0_exc, 4 * w0_inh))
    # # Calculate the initial spectral radius
    # rho_0 = np.max(np.abs(np.linalg.eigvals(w0_rec_plus)))
    # # Spectral radius scaling
    # w_rec_plus = (1.0 / rho_0) * w0_rec_plus
    wrec_0 = w0_rec_plus
    return wrec_0

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

def net_loss(net, _input, _target, _mask, n_epochs, lr=1e-2, batch_size=32, cuda=True):
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
        print("initial loss:{initial_loss.item()} ")
    return (initial_loss.item())

def train(net, _input, _target, _mask, n_epochs, lr=1e-2, batch_size=32, if_dscosgd=False, clip_gradient=None,
          cuda=True, plot_gradient=False):
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
    losses = []
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
        print(f"initial loss: {initial_loss.item()}")

    for epoch in range(n_epochs):
        begin = Time.time()
        for i in range(num_examples // batch_size):
            optimizer.zero_grad()
            random_batch_idx = random.sample(range(num_examples), batch_size)
            batch = input[random_batch_idx]
            output = net(batch)
            loss = loss_mse(output, target[random_batch_idx], mask[random_batch_idx])
            losses.append(loss.item())
            loss.backward()
            if clip_gradient is not None:
                torch.nn.utils.clip_grad_norm_(net.parameters(), clip_gradient)
            optimizer.step()

            with torch.no_grad():
                if if_dscosgd and epoch > 120:
                    wrec = (net.D_mask * torch.relu(net.wrec_plastic.data)).detach().cpu().numpy().copy()
                    wrec_positive = wrec[wrec > 0]
                    wrec_negative = -wrec[wrec < 0]
                    params_pos = scipy.stats.lognorm.fit(wrec_positive, method='mle')
                    params_neg = scipy.stats.lognorm.fit(wrec_negative, method='mle')
                    learning_rate2 = 0.1 * epoch/n_epochs
                    mu_e = np.log(params_pos[2])
                    mu_i = np.log(params_neg[2])
                    loc_e = params_pos[1]
                    loc_i = params_neg[1]
                    sigma_e = params_pos[0]
                    sigma_i = params_neg[0]
                    dsco_sgd = DScoSGD(net, mu_e, mu_i, loc_e, loc_i, sigma_e, sigma_i, learning_rate2)
                    dsco_sgd.apply()
            # # These 2 lines important to prevent memory leaks
            loss.detach_()
            output.detach_()
            if epoch != 0:
                print(f"epoch {15 * epoch + i}:  loss={loss.item()}  (took: {Time.time() - begin} s) ")
            else:
                print(f"epoch {i}:  loss={loss.item()}  (took: {Time.time() - begin} s)")
    return losses

def initial_loss(net, _input, _target, _mask, n_epochs, lr=1e-2, batch_size=32, plot_learning_curve=False,
                 plot_gradient=False, clip_gradient=None, keep_best=False, cuda=True, resample=False, save_loss=False):
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
        return (initial_loss.item())

class BIORNN(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, noise_std, alpha=0.2, rho=0.1,
                 train_wi=False, train_wo=False, train_wrec=True, train_h0=False,wi_init=None,
                 wo_init=None, wrec_init=None, si_init=None, so_init=None, h0_init=None,e_ratio=0.8,apply_dale=True):
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
        super(BIORNN, self).__init__()
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
        self.e_size = int(hidden_size * e_ratio)
        self.i_size = hidden_size - self.e_size

        # Define parameters
        # Initialize the Dale matrix (D = diag([1]*E + [-1]*I))
        self.D = torch.cat([torch.ones(hidden_size, self.e_size), -torch.ones(hidden_size, self.i_size)], 1).float()
        self.wrec_plastic = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
        # Create a diagonal mask matrix (0 diagonal, 1 elsewhere)
        self.mask = torch.ones(hidden_size, hidden_size) - torch.eye(hidden_size)  # Self-attaching masks
        self.register_buffer('D_mask', self.D * self.mask)  # Registered as a non-trainable constant

        self.wi = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.si = nn.Parameter(torch.Tensor(input_size))
        if train_wi:
            self.si.requires_grad = False
        else:
            self.wi.requires_grad = False

        if not train_wrec:
            self.wrec_plastic.requires_grad = False
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
                if apply_dale:
                    self.wrec_plastic.normal_(std=rho / np.sqrt(hidden_size))
                else:
                    self.wrec_plastic.normal_(std=rho / np.sqrt(hidden_size))
            else:
                if not apply_dale:
                    if type(wrec_init) == np.ndarray:
                        wrec_init = torch.from_numpy(wrec_init)
                    self.wrec_plastic.copy_(wrec_init)
                else:
                    if type(wrec_init) == np.ndarray:
                        wrec_plastic_init = torch.from_numpy(wrec_init)
                    self.wrec_plastic.copy_(wrec_plastic_init)
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

    def forward(self, input, return_dynamics=False, return_noise=False,apply_dale=True):
        """
        :param input: tensor of shape (batch_size, #timesteps, input_dimension)
        Important: the 3 dimensions need to be present, even if they are of size 1.
        :param return_dynamics: bool
        :return: if return_dynamics=False, output tensor of shape (batch_size, #timesteps, output_dimension)
                 if return_dynamics=True, (output tensor, trajectories tensor of shape (batch_size, #timesteps, #hidden_units))
        """
        # Apply the Dale principle to generate cyclic weights
        if apply_dale:
            wrec = self.D_mask * torch.relu(self.wrec_plastic)
        else:
            wrec = self.wrec_plastic
        batch_size = input.shape[0]
        seq_len = input.shape[1]
        h = self.h0
        r = self.non_linearity(h)
        self.define_proxy_parameters()
        noise = torch.randn(batch_size, seq_len, self.hidden_size, device=self.wrec_plastic.device)
        output = torch.zeros(batch_size, seq_len, self.output_size, device=self.wrec_plastic.device)
        if return_dynamics:
            trajectories = torch.zeros(batch_size, seq_len, self.hidden_size, device=self.wrec_plastic.device)

        # simulation loop
        for i in range(seq_len):
            h = h + self.noise_std * noise[:, i, :] + self.alpha * \
                (-h + r.matmul(wrec.t()) + input[:, i, :].matmul(self.wi_full))
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
        new_net = BIORNN(self.input_size, self.hidden_size, self.output_size, self.noise_std, self.alpha,
                              self.rho, self.train_wi, self.train_wo, self.train_wrec, self.train_h0,
                              self.wi, self.wo, self.wrec, self.si, self.so)
        return new_net

    def plot_eigenvalues(self):
        eig, _ = np.linalg.eig((self.D_mask * torch.relu(self.wrec_plastic)).detach().numpy())
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

def distribution_test(matrix, distribution, plot_dist=False, plot_QQ=False):
    """
    The distribution of matrix data is tested, and the fitting and testing of lognormal distributions are supported

    Parameters:
        matrix: Matrix data entered
        distribution: Specify the type of distribution to test, currently only 'lognorm' (lognormal distribution) is supported
        plot_dist: Boolean value, whether to plot a histogram of data distribution, is False by default
        plot_QQ: Boolean value, whether to plot a QQ chart, defaults to False

    return:
        There is no return value, but the p-value of the KS test is printed and the corresponding graph is drawn according to the parameters
    """

    # Check if it is a log-normal distribution
    if distribution == 'lognorm':
        # The matrix data is divided into two parts: positive and negative
        matrix_pos = matrix[matrix > 0]  # Extract all positive elements
        matrix_neg = matrix[matrix < 0]  # Extract all negative elements

        # Lognormal distribution fitting of positive and negative values separately (using maximum likelihood estimation)
        params_pos = scipy.stats.lognorm.fit(matrix_pos, method='mle')
        params_neg = scipy.stats.lognorm.fit(-matrix_neg, method='mle')

        # KS test for positive and negative values (Kolmogorov-Smirnov test)
        s_pos, p_pos = scipy.stats.kstest(matrix_pos, 'lognorm', args=params_pos)
        s_neg, p_neg = scipy.stats.kstest(-matrix_neg, 'lognorm', args=params_neg)

        # Print the p-value of the KS test
        print('positive weight{lognorm}KS-test p-value: ', p_pos)
        print('negative weight{lognorm}KS-test p-value: ', p_neg)

        # draw a QQplot
        if plot_QQ:
            # Draw a QQplot with positive values
            stats.probplot(matrix_pos, dist=stats.lognorm, sparams=params_pos, plot=plt)
            plt.title("QQ Plot for Positive Weights", fontsize=12)
            plt.xlabel("Theoretical Quantiles")
            plt.ylabel("Sample Quantiles")
            plt.show()

            # Draw a QQplot with negative values (pay attention to negative values)
            stats.probplot(-matrix_neg, dist=stats.lognorm, sparams=params_neg, plot=plt)
            plt.title("QQ Plot for Negative Weights", fontsize=12)
            plt.xlabel("Theoretical Quantiles")
            plt.ylabel("Sample Quantiles")
            plt.show()

        # If necessary, a distribution histogram is drawn
        if plot_dist:
            # Draw a histogram of positive and negative values
            plt.hist(matrix_pos, bins=50, density=True, alpha=0.5, label='Positive Weights')
            plt.hist(matrix_neg, bins=50, density=True, alpha=0.5, label='Negative Weights')
            plt.legend()
            plt.show()

def perceptual_vs(dims, dt, batch_size, noise, choices=None, fixation_duration=100, stim_min=80, stim_max=1500, stim_mean=330,
                  decision_duration=300, target_high=1.0, target_low=0.2, coherences=[1, 2, 4, 8, 16],
                  fraction_catch_trails=0., return_ts=False, test=False, fixation_duration_test=300,
                  stim_test=600):
    """
    Perceptual decision task generator

    Parameters:
        dims (tuple): Input and output dimensions in the format of: (dim_in, dim_hidden, dim_out)
        dt (float): Time step (ms)
        batch_size (int): Batch size
        noise (float): Standard deviation of input noise
        choices (list, optional): Optional list of decision options
        fixation_duration (int): Fixed-point duration (milliseconds)
        stim_min (int): Minimum duration of stimulation (ms)
        stim_max (int): Stimulation maximum duration (milliseconds)
        stim_mean (int): Average stimulus duration (ms)
        decision_duration (int): Decision period duration (milliseconds)
        target_high (float): High value of the target output
        target_low (float): Low value of the target output
        coherences (list): List of stimulus consistency levels
        fraction_catch_trails (float): Capture the proportion of trials
        return_ts (bool): Whether the time series is returned
        test (bool): Whether it's test mode
        fixation_duration_test (int): Fixed-point duration in test mode
        stim_test (int): Duration of stimulation in test mode

    return:
        input_batch
        target_batch
        mask_batch
        ts (optional)
    """
    # Get input and output dimensions
    dim_in, _,  dim_out = dims
    SCALE = 3.2  # Stimulus intensity scaling factor

    # Check if the dimension meets the requirements
    assert dim_in == 3, "The VS task input dimension must be 3 (2 decision directions + 1 starting signal)"
    assert dim_out == 2, "VS task output dimension must be 2 (two decision options)"

    # If no choices are provided, all possible decision options are used
    if choices is None:
        choices = np.arange(dim_out)
    # Make sure that the maximum value in choices does not exceed the output dimension
    assert np.max(choices) <= (dim_out - 1), "The max choice must agree with input dimension!"

    # Define the truncation index distribution function for generating stimulation duration
    def truncated_exponential(mean, xmin, xmax, dt):
        while True:
            x = np.random.exponential(mean)
            if xmin <= x < xmax:
                # Make sure it is an integer multiple of the time step
                return int(round(x / dt) * dt)

    # Generate stimulus duration based on pattern
    stim_all = [stim_test] if test else [truncated_exponential(stim_mean, stim_min, stim_max, dt) for _ in range(batch_size)]
    stimulus_duration_discrete_max = int(max(stim_all)/dt)

    # Calculate the time points of each stage (discrete time steps)
    fixation_duration_discrete = batch_size * [int(fixation_duration_test / dt)] if test else batch_size * [int(fixation_duration / dt)]
    stim_begin = fixation_duration_discrete

    stimulus_duration_discrete = batch_size * [int(stim_test / dt)] if test else [int(stim/dt) for stim in stim_all]
    stim_end = [sum(time) for time in zip(stim_begin, stimulus_duration_discrete)]

    response_begin = stim_end

    decision_duration_discrete = batch_size * [int(decision_duration / dt)]
    response_end = [sum(time) for time in zip(response_begin, decision_duration_discrete)]

    # Calculate the total length of time
    n_t_max_all = [sum(time) for time in zip(fixation_duration_discrete, stimulus_duration_discrete, decision_duration_discrete)]
    t_max = [n_t * dt for n_t in n_t_max_all]
    n_t_max_MAX = fixation_duration_discrete[0] + stimulus_duration_discrete_max + decision_duration_discrete[0]

    # Handle coherences parameters
    if coherences is None:
        coherences = np.array([1, 2, 4, 8, 16])
    elif type(coherences) == list:
        coherences = np.array(coherences)

    # Define the stimulus intensity scaling function
    def scale(coh):
        return (1 + SCALE * coh / 100) / 2

    # Initialize the input, target, and mask batches
    input_batch = np.zeros((batch_size, n_t_max_MAX, dim_in), dtype=np.float32)
    target_batch = np.zeros((batch_size, n_t_max_MAX, dim_out), dtype=np.float32)
    mask_batch = np.zeros((batch_size, n_t_max_MAX, dim_out), dtype=np.float32)

    # Generate inputs, targets, and masks for each sample
    for b_idx in range(batch_size):
        # Initializes the inputs, targets, and masks of the current sample
        input_samp = np.zeros((n_t_max_all[b_idx], dim_in - 1))
        input_pulse = np.zeros((n_t_max_all[b_idx], 1))
        target_samp = np.zeros((n_t_max_all[b_idx], dim_out))
        mask_samp = np.zeros((n_t_max_all[b_idx], dim_out))

        # Generate sensory input noise
        input_noise_samp = np.random.randn(n_t_max_all[b_idx], dim_in-1) * noise

        # Stimuli and targets are generated based on whether or not they are generated for the capture trial
        if b_idx < (1 - fraction_catch_trails) * batch_size:
            # Random selection of stimulus consistency and decision-making options
            if test and b_idx == 0:
                coh_i = coherences[3]  # Use fixed consistency in test mode
                choice = 1  # Use the fixed decision option in test mode
            else:
                coh_i = np.random.choice(coherences)
                choice = np.random.choice(choices)

            # Set the input, context, and target
            # Set the stimulus intensity according to the decision direction
            input_samp[stim_begin[b_idx]:stim_end[b_idx],choice] += scale(+coh_i)
            input_samp[stim_begin[b_idx]:stim_end[b_idx], 1-choice] += scale(-coh_i)
            # Set the stimulus start pulse
            input_pulse[stim_begin[b_idx]:stim_begin[b_idx]+1] += 1
            # Add noise
            input_samp += input_noise_samp

            # Set the target output
            target_samp[:stim_begin[b_idx]] = target_low
            target_samp[response_begin[b_idx]:response_end[b_idx], choice] = target_high
            target_samp[response_begin[b_idx]:response_end[b_idx], 1 - choice] = target_low

            # Set the mask
            mask_samp[:stim_begin[b_idx]] = 1
            mask_samp[response_begin[b_idx]:response_end[b_idx]] = 1
        else:
            # Capture test: only noise, no target stimulus
            input_samp += input_noise_samp
            target_samp[:] = target_low
            # The mask of the capture test is all 1
            mask_samp[:] = 1

        # Merge inputs, targets, and masks
        input_batch[b_idx, :response_end[b_idx], :dim_in-1] = input_samp
        input_batch[b_idx, :response_end[b_idx], dim_in-1:] = input_pulse

        target_batch[b_idx, :response_end[b_idx], 0:dim_out] = target_samp
        mask_batch[b_idx, :response_end[b_idx]] = mask_samp

    # Convert to PyTorch tensor
    dtype = torch.FloatTensor
    input_batch = torch.from_numpy(input_batch).type(dtype)
    target_batch = torch.from_numpy(target_batch).type(dtype)
    mask_batch = torch.from_numpy(mask_batch).type(dtype)

    # Decide whether to return the time series based on return_ts
    if return_ts:
        # Generate time series
        ts = np.arange(0, t_max[0], dt) / dt
        return input_batch, target_batch, mask_batch, ts
    else:
        return input_batch, target_batch, mask_batch

def shannon_effect_rank(Wrec, hidden_size):
    # Calculate Singular values
    singular_values = np.linalg.svd(Wrec, compute_uv=False)
    # Calculate the square of the Frobenius norm (sum of all singular values squared)
    frob_norm_sq = np.sum(singular_values ** 2)
    # Calculate qk = (γk^2) / ||A||_F^2
    qk = (singular_values ** 2) / frob_norm_sq
    # Use np.where to avoid taking ln for 0
    log_qk = np.where(qk > 0, np.log(qk), 0)
    entropy_sum = np.sum(qk * log_qk)
    # Calculate exp(-entropy_sum)
    r_eff = np.exp(-entropy_sum)
    return r_eff / hidden_size

dt = 10  # ms
tau = 100  # ms
alpha = dt / tau
std_noise_rec = np.sqrt(2 * alpha) * 0.1
input_size = 3
hidden_size = 200
output_size = 2

os.makedirs('TrainedNets_variable_stim/',exist_ok=True)
os.makedirs('Figures_variable_stim/', exist_ok=True)

begin_time = datetime.now()
begin_time = begin_time.strftime("%Y-%m-%d %H:%M:%S")
train_ = False
trails_train = 1000

# Network parameters
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU is available.")
else:
    device = torch.device("cpu")
    print("GPU is not available, using CPU.")

dims = [input_size, hidden_size, output_size]
wrec_dscosgd = create_wrec_init(hidden_size)
net_DScoSGD = BIORNN(input_size, hidden_size, output_size, 0.0 * std_noise_rec, alpha, wrec_init=wrec_dscosgd,
                     train_wi=True, train_wrec=True, train_wo=True, train_h0=True, e_ratio=0.8, apply_dale=True)
if train_ == True:

    input_train, output_train, mask_train = perceptual_vs(dims, dt, trails_train, noise=0.5 * std_noise_rec,
                                                          fraction_catch_trails=0.1, return_ts=False, test=False)

    print('train BIORNN')
    loss_dscosgd = train(net_DScoSGD, input_train, output_train, mask_train, n_epochs=180,
                         lr=1e-3, if_dscosgd=True, clip_gradient=1., cuda=True)
    np.save("TrainedNets_variable_stim/" + "RT_BioRNN_Train_loss", loss_dscosgd)
    torch.save(net_DScoSGD.state_dict(), "TrainedNets_variable_stim/" + "RT_BioRNN_Train_net.pt")
    net_DScoSGD.load_state_dict(torch.load("TrainedNets_variable_stim/" + "RT_BioRNN_Train_net.pt", map_location=device))
else:
    net_DScoSGD.load_state_dict(torch.load("TrainedNets_variable_stim/" + "RT_BioRNN_Train_net.pt", map_location=device))
    loss_origin = np.load("TrainedNets_variable_stim/" + "RT_BioRNN_Train_loss.npy")
    wrec = (net_DScoSGD.D_mask * torch.relu(net_DScoSGD.wrec_plastic.data)).detach().cpu().numpy().copy()
    set_plot()
    print(shannon_effect_rank(wrec,hidden_size))

    # =================================================================================
    #  Network input and output
    # =================================================================================
    input_train, output_train, mask_train, task_plt = perceptual_vs(dims, dt, 1, noise=0.5 * std_noise_rec,
                                                                    return_ts=True, test=True)
    res_plt = run_net(dims, dt, net_DScoSGD, batch_size=1, test=True, h_init=None)
    _input, _target, _mask, net_output, loss = res_plt
    # inputt = input_train.detach().cpu().numpy().copy()
    # outputt = output_train.detach().cpu().numpy().copy()
    # mask = mask_train.detach().cpu().numpy().copy()
    # input1 = inputt[0, :, 0]
    # input2 = inputt[0, :, 1]
    # input3 = inputt[0, :, 2]
    #
    # output1 = outputt[0, :, 0]
    # output2 = outputt[0, :, 1]
    # net_out = net_DScoSGD.forward(input_train).detach().cpu().numpy().copy()
    # net_out1 = net_out[0, :, 0]
    # net_out2 = net_out[0, :, 1]
    #
    # mask1 = mask[0, :, 0]
    # mask2 = mask[0, :, 1]
    #
    # output1 = np.where(mask1 != 0, output1, np.nan)
    # output2 = np.where(mask2 != 0, output2, np.nan)
    #
    # plt.plot(task_plt, input1, label='choice1')
    # plt.plot(task_plt, input2, label='choice2')
    # plt.plot(task_plt, input3, label='pulse')
    # plt.legend()
    # plt.grid()
    # plt.show()
    #
    # plt.plot(task_plt, output1, label='output1')
    # plt.plot(task_plt, output2, label='output2')
    #
    # plt.plot(task_plt, net_out1, '--', color='k', label='net_output1')
    # plt.plot(task_plt, net_out2, '--', color='r', label='net_output2')
    # plt.legend()
    # plt.grid()
    # plt.show()

    fig_width = 1.5 * 2.2  # width in inches
    fig_height = 1.5 * 2.0  # height in inches
    fig_size = [fig_width, fig_height]
    fig = plt.figure(figsize=fig_size)
    ax0, ax1, ax2 = fig.add_subplot(311), fig.add_subplot(312), fig.add_subplot(313)

    input_train, output_train, mask_train, ts_plt = perceptual_vs(dims, dt, 1, noise=0.5 * std_noise_rec,
                                                                    return_ts=True, test=True)
    res_plt = run_net(dims, dt, net_DScoSGD, batch_size=1, test=True)
    _input, _target, _mask, net_output, loss = res_plt

    dim_in = _input.shape[-1]
    dim_out = net_output.shape[-1]

    x0f = 0.05
    x1f = 1.55
    colors = ["0.6", "0.1"]
    for i in range(dim_in-1):
        # Sensory input
        ax0.plot(ts_plt, _input[0, :, i], '-', lw=2, c=colors[i], label="$x_%d$" % (i + 1))
    # # Pulse input
    # ax1.plot(ts_plt, _input[0, :, 2], '-', lw=2, c=colors[1], label="$c_%d$" % (i + 1))

    for i in range(dim_out):
        # # Output
        # ax2.plot(ts_plt, net_output[0, :, i], '-', lw=2, c=colors[0], label="$y_%d$")
        # # Target
        # m = np.bool_(_mask[0, :, i])
        # z_hat = _target[0, :, i]
        # z_hat = np.where(m, z_hat, np.nan)
        # ax2.plot(ts_plt, z_hat, '-', lw=2, c=colors[1], label="$\hat{y}$")
        # output
        ax = [ax1, ax2][i]
        ax.plot(ts_plt, net_output[0, :, i], '-', lw=2, c=colors[0], label='$y_%d$' % (i + 1))
        # target
        m = np.bool_(_mask[0, :, i])
        z_hat = _target[0, :, i]
        z_hat = np.where(m, z_hat, np.nan)
        ax.plot(ts_plt, z_hat, '-', lw=2, c=colors[1], label='$\hat{y}_{%d}$' % (i + 1))
    # Indicate task phases
    t0 = 30
    t1 = t0 + 60

    for i, ti in enumerate([t0, t1]):
        for ax in [ax0, ax1, ax2]:
            ax.axvline(ti, ls='--', lw=1, c='0.7', zorder=-1)

    ax0.set_yticks([0.0, 1.0])
    ax0.set_yticklabels([0.0, 1.0])
    ax1.set_ylim(-0.1, 1.1)
    ax1.set_yticklabels([0.2, 1.0])
    ax2.set_ylim(-0.1, 1.1)
    ax2.set_yticks([0.2, 1.0])

    ax0.set_ylabel("input", fontsize=12)
    ax1.set_ylabel("output1", fontsize=12)
    ax2.set_ylabel("output2", fontsize=12)
    ax2.set_xlabel("time", fontsize=12)

    dt = ts_plt[1] - ts_plt[0]
    t_max_plt = ts_plt[-1] + dt
    x0 = -t_max_plt * x0f
    x1a = t_max_plt * (1 + x0f)
    x1b = t_max_plt * x1f

    for i, ax in enumerate([ax0, ax1, ax2]):
        ax.set_yticklabels([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        ax.plot([x0, x1a], [0, 0], '--', c='0.7', zorder=-1)
        ax.set_xlim(x0, x1b)
        ax.legend(loc=5, frameon=False, framealpha=1., labelspacing=0.1, handlelength=1.,fontsize=10)
        if not i == 2:
            ax.set_xticklabels([])
    #     ax2.set_xticks([0, t_max_plt//2, t_max_plt])
    ax2.set_xticks([0, 60, 120])

    string = f'Perceptual decision-making_VS_Input&Output.png'
    print(string)
    plt.savefig('Figures_variable_stim/' + string, dpi=300, bbox_inches='tight')
    string = f'Perceptual decision-making_VS_Input&Output.pdf'
    print(string)
    plt.savefig('Figures_variable_stim/' + string, bbox_inches='tight')
    plt.show()

    # =================================================================================
    #  Network Training error
    # =================================================================================
    fig_width = 1.5 * 2.2  # width in inches
    fig_height = 1.5 * 2.0  # height in inches
    fig_size = [fig_width, fig_height]
    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(111)

    only_epoch = 2700
    loss_0 = loss_origin[0]
    ax.set_xlim(-0.05*only_epoch, only_epoch)
    ax.set_yticks([0, loss_0])
    ax.set_xticks([0, only_epoch//2, only_epoch])
    ax.set_yticklabels([0, r"$l_0$"], fontsize=12, usetex=True)
    ax.set_xlabel("iteration", fontsize=14)
    ax.set_ylabel("loss", labelpad=0, fontsize=14)
    ax.axhline(0, ls='--', c='0.7', zorder=-1)
    ax.plot(loss_origin, '-', c='0.35', lw=1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    string = f'Perceptual decision-making_VS_TrainingError.png'
    print(string)
    plt.savefig('Figures_variable_stim/' + string, dpi=300, bbox_inches='tight')
    string = f'Perceptual decision-making_VS_TrainingError.pdf'
    print(string)
    plt.savefig('Figures_variable_stim/' + string, bbox_inches='tight')
    string = f'Perceptual decision-making_VS_TrainingError.eps'
    print(string)
    plt.savefig('Figures_variable_stim/' + string, bbox_inches='tight')
    plt.show()

    # =================================================================================
    # Heat map + QQplot
    # =================================================================================
    fig_width = 1.5 * 2.2  # width in inches
    fig_height = 1.5 * 2.0    # height in inches
    fig_size = [fig_width, fig_height]
    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(111)

    abs_max = np.abs(wrec).max()
    linthresh = abs_max * 0.001
    n_front = int(wrec.shape[0] * 0.8)
    n_back = wrec.shape[0] - n_front

    im = ax.imshow(
        wrec,
        cmap='RdBu_r',
        norm=SymLogNorm(linthresh=linthresh, linscale=1, vmin=-abs_max, vmax=abs_max),
        aspect='equal',
        interpolation='nearest'
    )
    color_e = (209 / 255, 87 / 255, 73 / 255)
    color_i = (28 / 255, 92 / 255, 158 / 255)

    # plt.axvline(x=80, color='k', linestyle='--',lw=2 )
    # plt.axhline(y=160, color='k', linestyle='--', alpha=0.5)

    x_e = (n_front - 1) / 2
    x_i = n_front + (n_back - 1) / 2
    y_e = ((n_front / 0.8) - 1) * 1.1
    y_i = ((n_front / 0.8) - 1) * 1.1
    plt.text(x_e, y_e, "excitatory", ha='center', fontsize=10, color=color_e)
    plt.text(x_i, y_i, "inhibitory", ha='center', fontsize=10, color=color_i)

    cbar = fig.colorbar(im, ax=ax, fraction=0.0465, pad=0.05, aspect=20)
    cbar.set_label('wrec(log scale)', fontsize=10, labelpad=-10)
    ticks = [-1, 0, 1]
    tick_labels = ['-0.1', '0', '0.1']
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(tick_labels)
    plt.xticks([])
    plt.yticks([])

    string = f'Perceptual decision-making_VS_Hotmap.png'
    print(string)
    plt.savefig('Figures_variable_stim/' + string, dpi=300, bbox_inches='tight')
    string = f'Perceptual decision-making_VS_Hotmap.pdf'
    print(string)
    plt.savefig('Figures_variable_stim/' + string, bbox_inches='tight')
    plt.show()

    # QQplot
    matrix_pos = wrec[wrec > 0]
    matrix_neg = wrec[wrec < 0]

    params_pos = scipy.stats.lognorm.fit(matrix_pos, method='mle')
    s_e, loc_e, scale_e = params_pos

    mu_e = np.log(scale_e)
    sigma_e = s_e

    params_neg = scipy.stats.lognorm.fit(-matrix_neg, method='mle')
    s_i, loc_i, scale_i = params_neg

    mu_i = np.log(scale_i)
    sigma_i = s_i

    n_column_e = int(wrec.shape[0] * 0.8)
    n_column_i = wrec.shape[0] - n_column_e

    s_pos, p_pos = scipy.stats.kstest(matrix_pos, 'lognorm', args=params_pos)
    s_neg, p_neg = scipy.stats.kstest(-matrix_neg, 'lognorm', args=params_neg)
    print('positive {lognorm}KS-test p-value & s_pos: ', p_pos, s_pos)
    print('negative {lognorm}KS-test p-value & s_neg: ', p_neg, s_neg)

    # positive QQplot
    fig_width = 1.5 * 2.2  # width in inches
    fig_height = 1.5 * 2.0  # height in inches
    fig_size = [fig_width, fig_height]
    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(111)
    osm, osr = stats.probplot(matrix_pos, dist=stats.lognorm, sparams=params_pos, plot=ax)
    points = ax.get_lines()[0]
    line = ax.get_lines()[1]

    points.set_color(color_e)
    points.set_marker('o')
    points.set_markersize(4)
    line.set_color('grey')
    line.set_linewidth(1.5)

    ax.set_xlabel("theoretical quantiles", fontsize=12)
    ax.set_ylabel("sample quantiles", fontsize=12)
    ax.set_title('')
    info_text = (f'$\mu^E={mu_e:.3f}$\n$\sigma^E={sigma_e:.3f}$')
    ax.annotate(info_text,
                xy=(0.05, 0.95),
                xycoords='axes fraction',
                fontsize=8,
                ha='left', va='top',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='gray'))

    ax.grid(False)

    string = f'Perceptual decision-making_VS_QQplot_E.png'
    print(string)
    plt.savefig('Figures_variable_stim/' + string, dpi=300, bbox_inches='tight')
    string = f'Perceptual decision-making_VS_QQplot_E.pdf'
    print(string)
    plt.savefig('Figures_variable_stim/' + string, bbox_inches='tight')
    plt.show()

    # negative QQplot
    fig_width = 1.5 * 2.2  # width in inches
    fig_height = 1.5 * 2.0  # height in inches
    fig_size = [fig_width, fig_height]
    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(111)
    osm, osr = stats.probplot(-matrix_neg, dist=stats.lognorm, sparams=params_neg, plot=ax)
    points = ax.get_lines()[0]
    line = ax.get_lines()[1]

    points.set_color(color_i)
    points.set_marker('o')
    points.set_markersize(4)
    line.set_color('grey')
    line.set_linewidth(1.5)

    ax.set_xlabel("theoretical quantiles", fontsize=12)
    ax.set_ylabel("sample quantiles", fontsize=12)
    ax.set_title('')
    info_text = (
        f'$\mu^I={mu_i:.3f}$\n$\sigma^I={sigma_i:.3f}$')
    ax.annotate(info_text,
                xy=(0.05, 0.95),
                xycoords='axes fraction',
                fontsize=8,
                ha='left', va='top',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='gray'))
    ax.grid(False)

    string = f'Perceptual decision-making_VS_QQplot_I.png'
    print(string)
    plt.savefig('Figures_variable_stim/' + string, dpi=300, bbox_inches='tight')
    string = f'Perceptual decision-making_VS_QQplot_I.pdf'
    print(string)
    plt.savefig('Figures_variable_stim/' + string, bbox_inches='tight')
    plt.show()