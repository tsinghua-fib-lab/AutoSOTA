import random
import torch.optim as optim
from matplotlib.colors import LinearSegmentedColormap, SymLogNorm
from matplotlib.patches import FancyArrowPatch
from scipy.stats import lognorm
from sklearn.decomposition import NMF
from torch.onnx.symbolic_opset9 import detach
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import statsmodels.api as sm
from scipy import stats
import scipy
from statsmodels.distributions.empirical_distribution import ECDF
import os, time
import pickle
import scipy.stats
# RNN model and task
from task_generators import flipflop, mante, romo
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
from warnings import warn
from statsmodels.graphics.gofplots import qqplot
import torch.nn.functional as F
from torch.distributions import Beta
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import StepLR
import pandas as pd
import seaborn as sns
from matplotlib.colors import LogNorm

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

def loss_mse(output, target, mask):
    """
    Mean squared error loss
    :param output: torch tensor of shape (num_trials, num_timesteps, output_dim)
    :param target: idem
    :param mask: idem -- or torch tensor of shape (num_trials, num_timesteps, 1)
    :return: float
    """
    # wrec = net.wrec
    # If mask has the same shape as output:
    if output.shape == mask.shape:
        loss = (mask * (target - output).pow(2)).sum() / mask.sum()
    else:
        raise Exception("This is problematic...")
        output_dim = output.shape[-1]
        loss = (mask * (target - output).pow(2)).sum() / (mask.sum() * output_dim)
    # Take half:
    loss = 0.5 * loss
    return loss

def run_net(net, task, batch_size=32, return_dynamics=False, h_init=None):
    # Generate batch
    input, target, mask = task(batch_size)
    # Convert training data1 to pytorch tensors
    input = torch.from_numpy(input)
    target = torch.from_numpy(target)
    mask = torch.from_numpy(mask)
    with torch.no_grad():
        # Run dynamics
        if return_dynamics:
            output, trajectories = net(input, return_dynamics, h_init=h_init)
        else:
            output = net(input, h_init=h_init)
        loss = loss_mse(output, target, mask)
    res = [input, target, mask, output, loss]
    if return_dynamics:
        res.append(trajectories)
    res = [r.numpy() for r in res]
    return res

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

class BioRNN(nn.Module):
    """
    Biologically inspired recurrent neural network class that implements an RNN model with biological properties
    """
    def __init__(self, dims, noise_std, dt=0.5,nonlinearity='tanh', readout_nonlinearity=None,
                 g=None, wi_init=None, wrec_init=None, wo_init=None, brec_init=None, h0_init=None,
                 train_wi=False, train_wrec=True, train_wo=False, train_brec=False, train_h0=False,
                 ML_RNN=False,e_ratio=0.8,apply_dale=True,
                 ):
        """
        :param dims: list = [input_size, hidden_size, output_size] - Network dimension configuration
        :param noise_std: float - Noise standard deviation
        :param dt: float - Integration time steps
        :param nonlinearity: str - Activate the function type, optionally 'tanh' or 'id'
        :param readout_nonlinearity: str - Output layer activation function type, optional 'tanh' or 'id'
        :param g: float - The standard deviation of the Gaussian distribution initialization
        :param wi_init: torch tensor - Enter the weight matrix initialization value, shaped like (input_dim, hidden_size)
        :param wo_init: torch tensor - 输The initialization value of the weight matrix is in the shape of (hidden_size, output_dim)
        :param wrec_init: torch tensor - The value initialized by the cyclic weight matrix is in the shape of (hidden_size, hidden_size)
        :param brec_init: torch tensor - The cyclic layer bias initialization value is shaped like (hidden_size)
        :param h0_init: torch tensor - The initial hidden state is shaped like (hidden_size)
        :param train_wi: bool - Whether the input weights are trained
        :param train_wo: bool - Whether the output weights are trained
        :param train_wrec: bool - Whether to train loop weights
        :param train_brec: bool - Whether to train the cyclic layer bias
        :param train_h0: bool - Whether to train the initial hidden state
        :param ML_RNN: bool - Whether to use the machine learning convention forward propagation method f(Wr)
        """

        super(BioRNN, self).__init__()

        # Save the network dimension configuration
        self.dims = dims
        input_size, hidden_size, output_size = dims
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Save other hyperparameters
        self.noise_std = noise_std
        self.dt = dt
        self.train_wi = train_wi
        self.train_wo = train_wo
        self.train_wrec = train_wrec
        self.train_brec = train_brec
        self.train_h0 = train_h0
        self.ML_RNN = ML_RNN

        # Count excitatory and inhibitory neuron counts
        self.e_size = int(hidden_size * e_ratio)  # Number of excitatory neurons
        self.i_size = hidden_size - self.e_size   # Number of inhibitory neurons

        # If the Dale principle is applied
        if apply_dale:
            # Initialize the Dale matrix (D = diag([1]*E + [-1]*I))
            # Excitatory neurons are represented by 1, and inhibitory neurons are denoted by -1
            self.D = torch.cat([torch.ones(hidden_size, self.e_size), -torch.ones(hidden_size, self.i_size)], 1).float()

            # Define a trainable cyclic weight matrix
            self.wrec_plastic = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size))

            # Create a diagonal mask matrix (0 diagonal, 1 elsewhere)
            self.mask = torch.ones(hidden_size, hidden_size) - torch.eye(hidden_size)  # Self-attaching masks

            # Register Dale masks as non-trainable constants
            self.register_buffer('D_mask', self.D * self.mask)

        # Check if the g or wrec_init parameter is provided
        # Either set g or choose initial parameters. Otherwise, there's a conflict!
        assert (g is not None) or (wrec_init is not None), "Choose g or initial wrec!"

        # If both g and wrec_init are provided, check that they are consistent
        if (g is not None) and (wrec_init is not None):
            g_wrec = wrec_init.std() * np.sqrt(hidden_size)
            tol_g = 0.01
            if np.abs(g_wrec - g) > tol_g:
                warn("Nominal g and wrec_init disagree: g = %.2f, g_wrec = %.2f" % (g, g_wrec))
        self.g = g

        # Set the activation function
        # Nonlinearity
        if nonlinearity == 'tanh':
            self.nonlinearity = torch.tanh
        elif nonlinearity == 'leakyrelu':
            self.nonlinearity = nn.LeakyReLU(negative_slope=0.01)
        elif nonlinearity == 'id':
            # Linear activation function
            self.nonlinearity = lambda x: x
            # For linear networks, dynamic stability needs to be checked
            if g is not None:
                if g > 1:
                    warn("g > 1. For a linear network, we need stable dynamics!")
        elif nonlinearity.lower() == 'relu':
            self.nonlinearity = nn.ReLU()
        elif nonlinearity == 'softplus':
            # Softplus activation function, the scale parameter controls how close it is to ReLU
            softplus_scale = 1  # Note that scale 1 is quite far from relu
            self.nonlinearity = lambda x: torch.log(1. + torch.exp(softplus_scale * x)) / softplus_scale
        elif type(nonlinearity) == str:
            raise NotImplementedError("Nonlinearity not yet implemented.")
        else:
            # If the function object is passed, use it directly
            self.nonlinearity = nonlinearity

        # Set the output layer activation function
        # Readout nonlinearity
        if readout_nonlinearity is None:
            # If not specified, use the same activation function as the loop layer
            self.readout_nonlinearity = self.nonlinearity
        elif readout_nonlinearity == 'tanh':
            self.readout_nonlinearity = torch.tanh
        elif readout_nonlinearity == 'logistic':
            # Logistic function, output range[0, 1]
            self.readout_nonlinearity = lambda x: 1. / (1. + torch.exp(-x))
        elif readout_nonlinearity == 'id':
            self.readout_nonlinearity = lambda x: x
        elif type(readout_nonlinearity) == str:
            raise NotImplementedError("readout_nonlinearity not yet implemented.")
        else:
            self.readout_nonlinearity = readout_nonlinearity

        # Define network parameters

        self.wi = nn.Parameter(torch.Tensor(input_size, hidden_size))  #weight matrix
        if not train_wi:
            self.wi.requires_grad = False

        if not apply_dale:
            # If you don't apply the Dale principle, define a normal cyclic weight matrix
            self.wrec = nn.Parameter(torch.Tensor(hidden_size, hidden_size))

        if not train_wrec:
            # The training state of the loop weights is set according to whether the Dale principle is applied
            if apply_dale:
                self.wrec_plastic.requires_grad = False
            else:
                self.wrec.requires_grad = False

        self.wo = nn.Parameter(torch.Tensor(hidden_size, output_size))  # Output weight matrix
        if not train_wo:
            self.wo.requires_grad = False

        self.brec = nn.Parameter(torch.Tensor(hidden_size))  # Cyclic layer bias
        if not train_brec:
            self.brec.requires_grad = False

        self.h0 = nn.Parameter(torch.Tensor(hidden_size))  # Initial hidden state
        if not train_h0:
            self.h0.requires_grad = False

        # Initialize parameters
        with torch.no_grad():
            # Initialize the input weight matrix
            if wi_init is None:
                self.wi.normal_()
            else:
                if type(wi_init) == np.ndarray:
                    wi_init = torch.from_numpy(wi_init)
                self.wi.copy_(wi_init)

            # Initialize the cyclic weight matrix
            if wrec_init is None:
                # If no initialization value is provided, initialize with a Gaussian distribution
                if apply_dale:
                    self.wrec_plastic.normal_(std=g / np.sqrt(hidden_size))
                else:
                    self.wrec_plastic.normal_(std=g / np.sqrt(hidden_size))
            else:
                # Use the provided initialization values
                if not apply_dale:
                    if type(wrec_init) == np.ndarray:
                        wrec_init = torch.from_numpy(wrec_init)
                    self.wrec.copy_(wrec_init)
                else:
                    if type(wrec_init) == np.ndarray:
                        wrec_plastic_init = torch.from_numpy(wrec_init)
                    self.wrec_plastic.copy_(wrec_plastic_init)

            # Initialize the output weight matrix
            if wo_init is None:
                self.wo.normal_(std=1 / hidden_size)
            else:
                if type(wo_init) == np.ndarray:
                    wo_init = torch.from_numpy(wo_init)
                self.wo.copy_(wo_init)

            # Initialize the cyclic layer bias
            if brec_init is None:
                self.brec.zero_()
            else:
                if type(brec_init) == np.ndarray:
                    brec_init = torch.from_numpy(brec_init)
                self.brec.copy_(brec_init)

            # Initialize the initial hidden state
            if h0_init is None:
                self.h0.zero_()
            else:
                if type(h0_init) == np.ndarray:
                    h0_init = torch.from_numpy(h0_init)
                self.h0.copy_(h0_init)


    def forward(self, input, return_dynamics=False, h_init=None, apply_dale=True):
        """
        Forward propagation function

        :param input: tensor - Enter the tensor, shaped by (batch_size, #timesteps, input_dimension)
                      Note: Even if some dimensions are 1 in size, the 3-dimensional structure must be maintained
        :param return_dynamics: bool - Whether to return a hidden state track
        :param h_init: tensor - Custom initial hiding state
        :param apply_dale: bool - Whether the Dale principle is applied
        :return: If return_dynamics=False，return an output tensor of the shape (batch_size, #timesteps, output_dimension)
                 If return_dynamics=True，return (output tensor, hidden state trajectory tensor),
                 and the trajectory shape is (batch_size, #timesteps, #hidden_units)
        """
        # Apply the Dale principle to generate cyclic weights
        if apply_dale:
            # Use ReLU to ensure that the weights are not negative, and then multiply by the Dale mask
            wrec = self.D_mask * torch.relu(self.wrec_plastic)
        else:
            wrec = self.wrec

        # Get the dimension information for the input tensor
        batch_size = input.shape[0]
        seq_len = input.shape[1]

        # Set the initial hiding state
        if h_init is None:
            h = self.h0
        else:
            # Use a custom initial hiding state
            h_init_torch = nn.Parameter(torch.Tensor(batch_size, self.hidden_size))
            h_init_torch.requires_grad = False

            with torch.no_grad():
                h = h_init_torch.copy_(torch.from_numpy(h_init))

        # Generate noise
        noise = torch.randn(batch_size, seq_len, self.hidden_size, device=self.wrec_plastic.device)

        # Initialize the output tensor
        output = torch.zeros(batch_size, seq_len, self.output_size, device=self.wrec_plastic.device)

        # return a hidden state track, initialize the track tensor
        if return_dynamics:
            trajectories = torch.zeros(batch_size, seq_len, self.hidden_size, device=self.wrec_plastic.device)

        # Simulate loops
        # simulation loop
        for i in range(seq_len):
            if self.ML_RNN:
                # Use machine learning conventions for forward propagation
                rec_input = self.nonlinearity(
                    h.matmul(wrec.t())  # 循环连接
                    + input[:, i, :].matmul(self.wi)
                    + self.brec)
                # Note that if noise is added inside the nonlinearity, the amplitude should be adapted to the slope...
                # + np.sqrt(2. / self.dt) * self.noise_std * noise[:, i, :])
                # Update the hidden status
                h = ((1 - self.dt) * h
                     + self.dt * rec_input
                     + np.sqrt(self.dt) * self.noise_std * noise[:, i, :])
                # Compute the output
                out_i = self.readout_nonlinearity(h.matmul(self.wo))
            else:
                # Use the forward propagation mode of the biophysical convention
                rec_input = (
                        self.nonlinearity(h).matmul(wrec.t())
                        + input[:, i, :].matmul(self.wi)
                        + self.brec)
                # Update the hidden status
                h = ((1 - self.dt) * h
                     + self.dt * rec_input
                     + np.sqrt(self.dt) * self.noise_std * noise[:, i, :])
                # Compute the output
                out_i = self.readout_nonlinearity(h).matmul(self.wo)

            # Save the output of the current timestep
            output[:, i, :] = out_i

            if return_dynamics:
                trajectories[:, i, :] = h

        if not return_dynamics:
            return output
        else:
            return output, trajectories

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

def train(net, task, n_epochs, batch_size=32,learning_rate=1e-2, clip_gradient=None, cuda=True, rec_step=1,
          optimizer='adam', h_init=None, verbose=True):
    """
    Train a network
    :param net: nn.Module
    :param task: function; generates input, target, mask for a single batch
    :param n_epochs: int
    :param batch_size: int
    :param learning_rate: float
    :param clip_gradient: None or float, if not None the value at which gradient norm is clipped
    :param cuda: bool
    :param rec_step: int; record weights after these steps
    :return: res
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

    net.D_mask = net.D_mask.to(device)

    # Optimizer
    if optimizer == 'sgd':
        optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)
    elif optimizer == 'adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    else:
        raise Exception("Optimizer not known.")

    # Save initial weights
    wi_init = net.wi.detach().cpu().numpy().copy()
    wo_init = net.wo.detach().cpu().numpy().copy()
    brec_init = net.brec.detach().cpu().numpy().copy()
    wrec_init = (net.D_mask * torch.relu(net.wrec_plastic.data)).detach().cpu().numpy().copy()
    weights_init = [wi_init, wrec_init, wo_init, brec_init]
    # Record
    dim_rec = net.hidden_size
    dim_in = net.input_size
    dim_out = net.output_size
    n_rec_epochs = n_epochs // rec_step
    losses = np.zeros((n_epochs), dtype=np.float32)
    gradient_norm_sqs = np.zeros((n_epochs), dtype=np.float32)
    epochs = np.zeros((n_epochs))
    rec_epochs = np.zeros((n_rec_epochs))
    if net.train_wi:
        wis = np.zeros((n_rec_epochs, dim_in, dim_rec), dtype=np.float32)
    if net.train_wrec:
        wrecs = np.zeros((n_rec_epochs, dim_rec, dim_rec), dtype=np.float32)
    if net.train_wo:
        wos = np.zeros((n_rec_epochs, dim_rec, dim_out), dtype=np.float32)
    if net.train_brec:
        brecs = np.zeros((n_rec_epochs, dim_rec), dtype=np.float32)

    time0 = time.time()
    # gradients = {}
    if verbose:
        print("Training...")
    for i in range(n_epochs):
        # Save weights (before update)
        if i % rec_step == 0:
            k = i // rec_step
            rec_epochs[k] = i
            if net.train_wi:
                wis[k] = net.wi.cpu().detach().numpy()
            if net.train_wrec:
                wrecs[k] = (net.D_mask * torch.relu(net.wrec_plastic.data)).detach().cpu().numpy().copy()
            if net.train_wo:
                wos[k] = net.wo.cpu().detach().numpy()
            if net.train_brec:
                brecs[k] = net.brec.cpu().detach().numpy()
         # Generate batch
        _input, _target, _mask = task(batch_size)
        # Convert training data1 to pytorch tensors
        _input = torch.from_numpy(_input)
        _target = torch.from_numpy(_target)
        _mask = torch.from_numpy(_mask)
        # Allocate
        input = _input.to(device=device)
        target = _target.to(device=device)
        mask = _mask.to(device=device)

        optimizer.zero_grad()
        output = net(input, return_dynamics=False, h_init=h_init)
        loss = loss_mse(output, target, mask)
        #Gredient decent
        loss.backward()
        # Update weights
        optimizer.step()
        if i > 1000:
            wrec = (net.D_mask * torch.relu(net.wrec_plastic.data)).detach().cpu().numpy().copy()
            wrec_positive = wrec[wrec > 0]
            wrec_negative = -wrec[wrec < 0]
            params_pos = scipy.stats.lognorm.fit(wrec_positive, method='mle')
            params_neg = scipy.stats.lognorm.fit(wrec_negative, method='mle')
            learning_rate2 = 0.01 * i / n_epochs
            mu_e = np.log(params_pos[2])
            mu_i = np.log(params_neg[2])
            loc_e = params_pos[1]
            loc_i = params_neg[1]
            sigma_e = params_pos[0]
            sigma_i = params_neg[0]
            dsco_sgd = DScoSGD(net, mu_e, mu_i, loc_e, loc_i, sigma_e, sigma_i, learning_rate2)
            dsco_sgd.apply()

        # These 2 lines important to prevent memory leaks
        loss.detach_()
        output.detach_()
        # Save
        epochs[i] = i
        losses[i] = loss.item()
        with torch.no_grad():
            grad_norm = net.wrec_plastic.grad.abs().mean().item()
            print(f"Epoch {i}, Gradient Norm: {grad_norm},loss: {loss.item()}")

        if verbose:
            print("epoch %d / %d:  loss=%.6f, time=%.1f sec." % (i + 1, n_epochs, np.mean(losses), time.time() - time0),
                  end="\r")
    if verbose:
        print("\nDone. Training took %.1f sec." % (time.time() - time0))

    gradient_norms = np.sqrt(gradient_norm_sqs)
    # Final weights
    wi_last = net.wi.detach().cpu().numpy().copy()
    wrec_last = (net.D_mask * torch.relu(net.wrec_plastic.data)).detach().cpu().numpy().copy()
    wo_last = net.wo.detach().cpu().numpy().copy()
    brec_last = net.brec.detach().cpu().numpy().copy()
    weights_last = [wi_last, wrec_last, wo_last, brec_last]

    # Weights throughout training:
    weights_train = {}
    if net.train_wi:
        weights_train["wi"] = wis
    if net.train_wrec:
        weights_train["wrec"] = wrecs
    if net.train_wo:
        weights_train["wo"] = wos
    if net.train_brec:
        weights_train["brec"] = brecs

    res = [losses,weights_init, weights_last, weights_train, epochs, rec_epochs]
    return res

def run_training(task_specs):
    (file_name_prefix, n_epochs, gs, dims, task_params, task_generator) = task_specs

    # Task
    task = task_generator(dims, dt, **task_params)
    n_gs = len(gs)
    dim_in, dim_rec, dim_out = dims
    dim_out  = dim_out

    # Epochs
    rec_step = n_epochs // n_rec_epochs
    epochs = np.arange(n_epochs)
    rec_epochs = np.arange(0, n_epochs, rec_step)

    # Learning rate
    if optimizer == 'sgd':
        lr = lr0
    elif optimizer == 'adam':
        lr = lr0/dim_rec
        # For rank truncation
    ranks = np.arange(n_ranks)
    rank_max = ranks[-1] + 1

    # Weights
    wi_init_all = np.zeros((n_samples, n_gs, dim_in, dim_rec))
    wrec_init_all = np.zeros((n_samples, n_gs, dim_rec, dim_rec))
    wo_init_all = np.zeros((n_samples, n_gs, dim_rec, dim_out))
    brec_init_all = np.zeros((n_samples, n_gs, dim_rec))
    wi_last_all = np.zeros((n_samples, n_gs, dim_in, dim_rec))
    wrec_last_all = np.zeros((n_samples, n_gs, dim_rec, dim_rec))
    wo_last_all = np.zeros((n_samples, n_gs, dim_rec, dim_out))
    brec_last_all = np.zeros((n_samples, n_gs, dim_rec))
    if train_wi:
        wis_all = np.zeros((n_samples, n_gs, n_rec_epochs, dim_in, dim_rec))
    if train_wo:
        wos_all = np.zeros((n_samples, n_gs, n_rec_epochs, dim_rec, dim_out))
    if train_brec:
        brecs_all = np.zeros((n_samples, n_gs, n_rec_epochs, dim_rec))

    # Results
    losses_all = np.zeros((n_samples, n_gs, n_epochs))
    sv_dw_all = np.zeros((n_samples, n_gs, n_rec_epochs, dim_rec))
    loss_rr_all = np.zeros((n_samples, n_gs, n_ranks))
    norm_diff_rr_all = np.zeros((n_samples, n_gs, n_ranks))
    sv_all = np.zeros((3, n_samples, n_gs, dim_rec))

    for k in range(n_samples):
        print("Sample: ", k)
        time_t = 0
        for i, g in enumerate(gs):
            print("   ", i, g)

            if (not same_connectivity) or i == 0:
                # Connectivity
                # Initial internal connectivity
                # Initial internal connectivity
                wrec_0 = create_wrec_init(dim_rec)

                # Input and output vectors
                wio = np.random.normal(0, 1, (dim_rec, dim_in + dim_out))
                if orthogonalize_wio:
                    wio = np.linalg.qr(wio)[0]
                else:
                    wio /= np.linalg.norm(wio, axis=0)[None, :]
                # Make sure that the vecotrs are still normalized
                assert np.allclose(np.linalg.norm(wio, axis=0), 1), "Normalization gone wrong!"
                # Change normalization to the proper one
                wio *= np.sqrt(dim_rec)
                wi_init = wio[:, :dim_in].T.copy()
                wo_init = wio[:, dim_in:].copy() / dim_rec
                del wio

            wrec_init = g * wrec_0

            # Network
            net = BioRNN(dims, noise_std, dt, g=g,
                         wi_init=wi_init, wo_init=wo_init, wrec_init=wrec_init,
                      train_wi=train_wi, train_wrec=train_wrec, train_wo=train_wo, train_brec=train_brec,
                      nonlinearity=nonlinearity, readout_nonlinearity=readout_nonlinearity,
                      ML_RNN=ML_RNN,e_ratio=0.8,apply_dale=True
                      )

            # Train
            time0_t = time.time()
            res = train(net, task=task, n_epochs=n_epochs, batch_size=batch_size, learning_rate=lr,
                        clip_gradient=None,cuda=use_cuda, rec_step=rec_step, optimizer=optimizer,
                        verbose=False)
            torch.save(net.state_dict(), 'mante/' + "BioRNN_net.pt")
            losses, weights_init, weights_last, weights_train, _,_ = res
            # Weights
            wi_init, wrec_init, wo_init, brec_init = weights_init
            wi_last, wrec_last, wo_last, brec_last = weights_last
            dwrec_last = wrec_last - wrec_init
            wrecs = weights_train["wrec"]
            time_t += time.time() - time0_t

            # Compute SVs
            sv_dw = np.linalg.svd(wrecs - wrec_init, compute_uv=False)
            del wrecs
            # Reconstruct connectivty with only the largest rank
            u_last, s_last, vT_last = np.linalg.svd(dwrec_last)
            # Simulate for truncated dwrec
            loss_rr_i = np.zeros((n_ranks))
            norm_diff_rr_i = np.zeros((n_ranks))
            for j, rank in enumerate(ranks):
                if rank == 0:
                    dw_rr = 0
                else:
                    dw_rr = (u_last[:, :rank] * s_last[None, :rank]) @ vT_last[:rank]
                w_rr = wrec_init + dw_rr

                # Run network
                net_test = BioRNN(dims, noise_std, dt,
                               g=None, wi_init=wi_last, wo_init=wo_last, wrec_init=w_rr, brec_init=brec_last,
                               nonlinearity=nonlinearity, readout_nonlinearity=readout_nonlinearity,
                               ML_RNN=ML_RNN,e_ratio=0.8,apply_dale=True
                               )
                res_test = run_net(net_test, task, batch_size=batch_size_test)
                u, y, mask, z, loss = res_test

                # Save
                loss_rr_i[j] = loss
                norm_diff_rr_i[j] = np.linalg.norm(dw_rr - dwrec_last)

            # Save
            wi_init_all[k, i] = wi_init
            wrec_init_all[k, i] = wrec_init
            wo_init_all[k, i] = wo_init
            brec_init_all[k, i] = brec_init
            wi_last_all[k, i] = wi_last
            wrec_last_all[k, i] = wrec_last
            wo_last_all[k, i] = wo_last
            brec_last_all[k, i] = brec_last
            if train_wi:
                wis_all[k, i] = weights_train["wi"]
            if train_wo:
                wos_all[k, i] = weights_train["wo"]
            if train_brec:
                brecs_all[k, i] = weights_train["brec"]
            losses_all[k, i] = losses
            sv_dw_all[k, i] = sv_dw
            loss_rr_all[k, i] = loss_rr_i
            norm_diff_rr_all[k, i] = norm_diff_rr_i
        print("Learning took %.1f sec." % (time_t))
    # Compute EVs and SVs at the end of training
    sv_all[0] = np.linalg.svd(wrec_init_all, compute_uv=False)
    sv_all[1] = np.linalg.svd(wrec_last_all, compute_uv=False)
    sv_all[2] = np.linalg.svd(wrec_last_all - wrec_init_all, compute_uv=False)
    ###############################################################################
    # Save data
    to_be_dumped = {
        # Simulation parameters
        "dims": dims,
        "dt": dt,
        "gs": gs,
        "lr": lr,
        "noise_std": noise_std,
        "ML_RNN": ML_RNN,
        "nonlinearity": nonlinearity,
        "readout_nonlinearity": readout_nonlinearity,
        "n_epochs": n_epochs,
        "rec_step": rec_step,
        "epochs": epochs,
        "rec_epochs": rec_epochs,
        "ranks": ranks,
        "batch_size": batch_size,
        "batch_size_test": batch_size_test,
        "train_wi": train_wi,
        "train_wrec": train_wrec,
        "train_wo": train_wo,
        "train_brec": train_brec,
        # Task
        "task_params": task_params,
        # Weights
        "wi_init_all": wi_init_all,
        "wrec_init_all": wrec_init_all,
        "wo_init_all": wo_init_all,
        "brec_init_all": brec_init_all,
        "wi_last_all": wi_last_all,
        "wrec_last_all": wrec_last_all,
        "wo_last_all": wo_last_all,
        "brec_last_all": brec_last_all,
        # Results
        "losses_all": losses_all,
        "sv_dw_all": sv_dw_all,
        "loss_rr_all": loss_rr_all,
        "norm_diff_rr_all": norm_diff_rr_all,
        "sv_all": sv_all,
        # "gradients": gradients,
        # Computation time
        "time_t": time_t,
    }
    return to_be_dumped

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

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
            # Draw a QQ chart with positive values
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

use_cuda = torch.cuda.is_available()

optimizer = 'adam'
# Integration time step
dt = 0.5
# Training parameters
batch_size = 32
batch_size_test = 512
# Neural noise
noise_std = 0.
# Whether IO vectors are orthogonalized
orthogonalize_wio = False
# Same random connectivity for each g?
same_connectivity = False
# Reconstruction loss
n_ranks = 26
# Network architecture
ML_RNN = False
nonlinearity = 'tanh'
readout_nonlinearity = None
# Task names
tasks_file_name_prefix = ['mante']
# Number of epochs
tasks_n_epochs = np.array([3000])
# Number of kept weight matrices (we don't really need this for the analysis...)
n_rec_epochs = 200
# Learning rate
lr0 = 0.05
dim_rec = 200
# Simulate and analyze all three tasks for multiple samples.
n_samples = 1
# Values for g
tasks_gs = np.array([[1.0]])
# What to train
train_wi = True
train_wrec = True
train_wo = True
train_brec = False
apply_dale = True
mask_sparse = None
train_ = False

os.makedirs('mante/',exist_ok=True)
os.makedirs('Figures_mante/', exist_ok=True)

#Task setting and running
idx_task = 0
file_name_prefix = tasks_file_name_prefix[idx_task]
n_epochs = tasks_n_epochs[idx_task]
gs = tasks_gs[idx_task]

# Network parameters
dim_in = 2 * 2
dim_out = 1
dims = [dim_in, dim_rec, dim_out]

# Join
mante_params = {
    "choices": np.arange(dim_in//2),
    "fixation_duration":  3,
    "stimulus_duration":  20,
    "delay_duration":  5,
    "decision_duration":  20,
    "input_amp":  1.,
    "target_amp":  0.5,
    "context_amp":  1.,
    "rel_input_std":  0.05,
    "coherences":  np.array([-8, -4, -2, -1, 1, 2, 4, 8]) / 8.,
    "fixate": True,}

if train_:
    mante_specs = (file_name_prefix, n_epochs, gs, dims, mante_params, mante)
    params_biornn = run_training(task_specs=mante_specs)
    wrec = params_biornn['wrec_last_all'][0][0]
    np.save('mante/wrec', wrec)
    loss_origin = params_biornn['losses_all'][0][0]
    np.save('mante/loss_biornn', loss_origin)
else:
    wrec = np.load('mante/wrec.npy')
    loss_origin = np.load('mante/loss_biornn.npy')
    set_plot()

    # ================================================================================
    #  Task structure
    # ================================================================================
    net = BioRNN(dims, noise_std, dt, g=1,
                 wi_init=None, wo_init=None, wrec_init=None,
                 train_wi=train_wi, train_wrec=train_wrec, train_wo=train_wo, train_brec=train_brec,
                 nonlinearity=nonlinearity, readout_nonlinearity=readout_nonlinearity,
                 ML_RNN=ML_RNN, e_ratio=0.8, apply_dale=True
                 )
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("GPU is available.")
    else:
        device = torch.device("cpu")
        print("GPU is not available, using CPU.")
    net.load_state_dict(torch.load('mante/' + "BioRNN_net.pt",map_location=device))

    # =================================================================================
    # Network input and output
    # =================================================================================
    fig_width = 1.5 * 2.2     # width in inches
    fig_height = 1.5 * 2.0    # height in inches
    fig_size = [fig_width, fig_height]
    fig = plt.figure(figsize=fig_size)
    ax0, ax1, ax2 = fig.add_subplot(311), fig.add_subplot(312), fig.add_subplot(313)
    task_plt, ts_plt = mante(dims, dt, return_ts=True, test=True, **mante_params)
    res_plt = run_net(net, task_plt, batch_size=1)
    _input, _target, _mask, net_output, loss = res_plt
    dim_in = _input.shape[-1]
    dim_out = net_output.shape[-1]
    dim_sens = dim_in // 2
    x0f = 0.05
    x1f = 1.55
    colors = ["0.6", "0.1"]
    input11 = _input[0, :, 0]
    input12 = _input[0, :, 1]
    input21 = _input[0, :, 2]
    input22 = _input[0, :, 3]
    #
    output1 = _target[0, :, 0]
    mask1 = _mask[0, :, 0]
    # output1 = np.where(mask1 != 0, output1, np.nan)
    #
    net_output1 = net_output[0, :, 0]
    #
    # plt.plot(ts_plt, input21, label='input3', color='k', alpha=0.5)
    # plt.plot(ts_plt, input22, label='input4', color='k', alpha=0.9)
    # plt.legend(frameon=False)
    # plt.title(f'input1 {0}')
    # plt.show()
    #
    # plt.plot(ts_plt, input11, label='input1', color='k', alpha=0.5)
    # plt.plot(ts_plt, input12, label='input2', color='k', alpha=0.9)
    # plt.legend(frameon=False)
    # plt.title(f'input2 {0}')
    # plt.show()
    #
    # plt.plot(ts_plt, output1, label='output1', color='k', alpha=0.9)
    # plt.plot(ts_plt, net_output1, label='net_out1', color='k', alpha=0.5)
    # plt.title(f'output1 {0}')
    # plt.legend(frameon=False)
    # plt.show()

    for i in range(dim_sens):
        # Sensory input
        ax0.plot(ts_plt, _input[0, :, dim_sens + i], '-', lw=2, c=colors[i], label=r"$x_%d$" % (i + 1))
        # Context
        ax1.plot(ts_plt, _input[0, :, i], '-', lw=2, c=colors[i], label=r"$c_%d$" % (i + 1))
    for i in range(dim_out):
        # Output
        ax2.plot(ts_plt, net_output[0, :, i], '-', lw=2, c=colors[0], label=r"$y$")
        # Target
        m = np.bool_(_mask[0, :, i])
        z_hat = _target[0, :, i]
        z_hat = np.where(m, z_hat, np.nan)
        ax2.plot(ts_plt, z_hat, '-', lw=2, c=colors[1], label=r"$\hat{y}$")

    # Indicate task phases
    t0 = mante_params['fixation_duration']
    t1 = t0 + mante_params['stimulus_duration']
    t2 = t1 + mante_params['delay_duration']
    t3 = t2 + mante_params['decision_duration']
    for i, ti in enumerate([t0, t1, t2]):
        for ax in [ax0, ax1, ax2]:
            ax.axvline(ti, ls='--', lw=1, c='0.7', zorder=-1)

    ax1.set_yticks([0.0, 1.0])
    ax1.set_yticklabels([0.0, 1.0])
    ax2.set_ylim(-0.6, 0.6)
    ax2.set_yticks([-0.5, 0.])

    ax0.set_ylabel("signal", fontsize=12)
    ax1.set_ylabel("context", fontsize=12)
    ax2.set_ylabel("output", fontsize=12)
    ax2.set_xlabel("time", fontsize=12)

    #     y0, y1 = ax2.get_ylim()
    #     ypos = y0 - 0.4 * (y1 - y0)
    #     ax2.text((x0 + t0)/2, ypos, 'Fix', fontsize=fs, ha='center', va='bottom')
    #     ax2.text((t0 + t1)/2, ypos, 'Input', fontsize=fs, ha='center', va='bottom')
    #     ax2.text((t1 + t2)/2, ypos, 'Delay', fontsize=fs, ha='center', va='bottom')
    #     ax2.text((t2 + t3)/2, ypos, 'Decision', fontsize=fs, ha='center', va='bottom')
    #     ax2.set_xlabel("Trial time $t$", labelpad=15)

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
        ax.legend(loc=5, frameon=False, framealpha=1., labelspacing=0.1, handlelength=1., fontsize=10)
        if not i == 2:
            ax.set_xticklabels([])
    #     ax2.set_xticks([0, t_max_plt//2, t_max_plt])
    ax2.set_xticks([0, 25, 50])

    string = f'Mante_Input&Output.png'
    print(string)
    plt.savefig('Figures_mante/' + string, dpi=300, bbox_inches='tight')
    string = f'Mante_Input&Output.pdf'
    print(string)
    plt.savefig('Figures_mante/' + string, bbox_inches='tight')
    plt.show()

    # =================================================================================
    #   Network Training error
    # =================================================================================
    fig_width = 1.5 * 2.2  # width in inches
    fig_height = 1.5 * 2.0  # height in inches
    # only_epoch = 3000
    fig_size = [fig_width, fig_height]
    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(111)

    ax.plot(loss_origin, '-', c='0.35', lw=1.5)
    loss_array = np.array(loss_origin)
    only_epoch = len(loss_array)  # 1200
    loss_0 = loss_array[0]
    ax.set_xlim(-0.05*only_epoch, 1.05*only_epoch)
    ax.set_yticks([0, loss_0])
    ax.set_xticks([0, only_epoch//2, only_epoch])
    ax.set_yticklabels([0, r"$l_0$"], fontsize=12,usetex=True)
    ax.set_xlabel("iteration",  fontsize=14)
    ax.set_ylabel("loss", labelpad=0, fontsize=14)
    ax.axhline(0, ls='--', c='0.7', zorder=-1)
    ax.plot(loss_origin, '-', c='0.35', lw=1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    string = f'Mante_TrainingError.png'
    print(string)
    plt.savefig('Figures_mante/' + string, dpi=300, bbox_inches='tight')
    string = f'Mante_TrainingError.pdf'
    print(string)
    plt.savefig('Figures_mante/' + string, bbox_inches='tight')
    string = f'Mante_TrainingError.eps'
    print(string)
    plt.savefig('Figures_mante/' + string, bbox_inches='tight')
    plt.show()

    # =================================================================================
    #   Heat map + QQplot
    # =================================================================================
    fig_width = 1.5 * 2.2   # width in inches
    fig_height = 1.5 * 2.0    # height in inches
    fig_size = [fig_width, fig_height]
    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(111)

    abs_max = np.abs(wrec).max()
    linthresh = abs_max * 0.01
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
    x_e = (n_front - 1) / 2
    x_i = n_front + (n_back - 1) / 2
    y_e = ((n_front / 0.8) - 1) * 1.1
    y_i = ((n_front / 0.8) - 1) * 1.1
    # plt.axvline(x=160, color='k', linestyle='--',lw=2 )
    # plt.axhline(y=160, color='k', linestyle='--', alpha=0.5)

    plt.text(x_e, y_e, "excitatory", ha='center', fontsize=10, color=color_e)
    plt.text(x_i, y_i, "inhibitory", ha='center', fontsize=10, color=color_i)

    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.05, aspect=20)
    cbar.set_label('wrec(log scale)', fontsize=10, labelpad=-10)

    ticks = [-0.1, 0, 0.1]
    tick_labels = ['-0.1', '0', '0.1']
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(tick_labels)

    plt.xticks([])
    plt.yticks([])

    string = f'Mante_Hotmap.png'
    print(string)
    plt.savefig('Figures_mante/' + string, dpi=300, bbox_inches='tight')
    string = f'Mante_Hotmap.pdf'
    print(string)
    plt.savefig('Figures_mante/' + string, bbox_inches='tight')
    plt.show()

    # QQplot
    matrix_pos = wrec[wrec > 0]
    matrix_neg = wrec[wrec < 0]

    params_pos = scipy.stats.lognorm.fit(matrix_pos, method='mle')
    s_e, loc_e, scale_e = params_pos

    mu_e = np.log(scale_e)  # Corresponding to the μ of log-normal
    sigma_e = s_e  # Corresponding to the σ of log-normal

    params_neg = scipy.stats.lognorm.fit(-matrix_neg, method='mle')
    s_i, loc_i, scale_i = params_neg

    mu_i = np.log(scale_i)
    sigma_i = s_i

    n_column_e = int(wrec.shape[0] * 0.8)
    n_column_i = wrec.shape[0] - n_column_e

    s_pos, p_pos = scipy.stats.kstest(matrix_pos, 'lognorm', args=params_pos)
    s_neg, p_neg = scipy.stats.kstest(-matrix_neg, 'lognorm', args=params_neg)
    print('positive {lognorm}KS-test p-value & s_pos:', p_pos, s_pos)
    print('negative {lognorm}KS-test p-value & s_neg:', p_neg, s_neg)

    # Positive weight QQplot
    fig_width = 1.5 * 2.2    # width in inches
    fig_height = 1.5 * 2.0   # height in inches
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

    string = f'Mante_QQplot_E.png'
    print(string)
    plt.savefig('Figures_mante/' + string, dpi=300, bbox_inches='tight')
    string = f'Mante_QQplot_E.pdf'
    print(string)
    plt.savefig('Figures_mante/' + string, bbox_inches='tight')
    plt.show()

    # Negative weight QQplot
    fig_width = 1.5 * 2.2    # width in inches
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

    string = f'Mante_QQplot_I.png'
    print(string)
    plt.savefig('Figures_mante/' + string, dpi=300, bbox_inches='tight')
    string = f'Mante_QQplot_I.pdf'
    print(string)
    plt.savefig('Figures_mante/' + string, bbox_inches='tight')
    plt.show()