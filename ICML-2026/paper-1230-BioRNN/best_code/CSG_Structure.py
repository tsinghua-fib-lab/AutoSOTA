import random
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.legend_handler import HandlerLine2D
from matplotlib.ticker import FuncFormatter
from matplotlib.transforms import Bbox
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from sklearn.preprocessing import StandardScaler
from sympy import Line2D
import matplotlib.pyplot as plt
import torch
import funcs_Sphere as fs
import scipy
import torch.nn as nn
from matplotlib import cm
from scikit_posthocs import posthoc_dunn
import pandas as pd
import seaborn as sns
import os
from jinja2.nodes import Break
from sklearn.decomposition import NMF, PCA
import modules4 as md
import matplotlib.pyplot as plt
# import lib_rnns as lr
# import tools_MF as tm
from mpl_toolkits.mplot3d import Axes3D
from math import sqrt
import time as Time
import scipy.stats
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.stats import lognorm
from scipy import stats
from matplotlib.colors import LinearSegmentedColormap, SymLogNorm
from matplotlib import colors
from datetime import datetime
from scipy.stats import kruskal
from matplotlib.ticker import NullLocator

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

def create_inp_out2(trials, Nt, tss, amps, R_on, SR_on, just=-1, perc=0.2):
    n_ts = len(tss)  # Time interval sequence length
    time = np.arange(Nt)

    strt = -0.5
    inputt = np.zeros((trials, Nt, 2))
    outputt = strt * np.ones((trials, Nt, 1))
    maskt = np.zeros((trials, Nt, 1))

    r_inp = np.ones((trials, Nt))
    # r2_inp = np.ones((trials, Nt))
    s_inp = np.zeros((trials, Nt))

    if just == -1:  # all types of trials
        ct = np.random.randint(n_ts, size=trials)

    else:
        ct = just * np.ones(trials, dtype=np.int8)

    # Don't have the set cue in a set of inputs
    ct2 = np.random.rand(trials) < perc

    rnd = np.zeros(trials)
    if SR_on > 0:
        rnd = np.random.randint(-SR_on, SR_on, trials)

    for itr in range(trials):
        if ct2[itr]:
            maskt[itr, :, 0] = (time > R_on + 1 + rnd[itr]) * (time < np.max(tss) + R_on + 1 + rnd[itr])
        else:
            maskt[itr, :, 0] = (time > R_on + 1 + rnd[itr]) * (time < tss[ct[itr]] + R_on + 1 + rnd[itr])
            mask_aft = time >= tss[ct[itr]] + R_on + 1 + rnd[itr]
            s_inp[itr, time > R_on + rnd[itr]] = 10.  # 10.
            s_inp[itr, time > 1 + R_on + rnd[itr]] = 0.
            if sum(maskt[itr, :, 0]):
                outputt[itr, maskt[itr, :, 0] == 1., 0] = np.linspace(strt, -strt, int(sum(maskt[itr, :, 0])),
                                                                      endpoint=True)
                outputt[itr, mask_aft == 1, 0] = np.linspace(-strt, -strt, int(sum(mask_aft)), endpoint=True)

        # Include zero read-out in cost function
        if ct2[itr]:
            maskt[itr, :, 0] = (time < Nt)  # np.max(tss)+R_on+1+rnd[itr])
        else:
            maskt[itr, :, 0] = (time < Nt)  # tss[ct[itr]]+R_on+1+rnd[itr])
        if just == -1:
            r_inp[itr, :] = amps[ct[itr]] * r_inp[itr, :]

    if just > -1:
        r_inp = amps[just] * r_inp

    inputt[:, :, 0] = r_inp
    inputt[:, :, 1] = s_inp
    # inputt[:,:,2] = r2_inp
    # outputt = strt*np.ones((trials, Nt, 1))

    dtype = torch.FloatTensor
    inputt = torch.from_numpy(inputt).type(dtype)
    outputt = torch.from_numpy(outputt).type(dtype)
    maskt = torch.from_numpy(maskt).type(dtype)

    return (inputt, outputt, maskt, ct, ct2)

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
          cuda=True):
    """
    Train the neural network model
    :param net: The neural network model to be trained (nn.Module)
    :param _input: Enter the data tensor, shaped like (num_trials, num_timesteps, input_dim)
    :param _target: The object data tensor, shaped like (num_trials, num_timesteps, output_dim)
    :param _mask: Mask tensor, shaped by  (num_trials, num_timesteps, 1)
    :param n_epochs: Training epochs(int)
    :param lr: Learning rate (float)
    :param batch_size: Batch size (int)
    :param if_dscosgd: Whether to use the DScoSGD optimizer(bool)
    :param clip_gradient: Gradient clipping threshold, None means no clipping (None or float)
    :param cuda: Whether CUDA is used (bool)
    :return: List of losses during training
    """
    print("Training...")

    # Initialize the Adam optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    # Obtain sample size
    num_examples = _input.shape[0]

    # Initialize the list of loss and gradient norms
    losses = []
    gradient_norms = []

    # CUDA Device Management
    if cuda:
        if not torch.cuda.is_available():
            print("Warning: CUDA not available on this machine, switching to CPU")
            device = torch.device('cpu')
        else:
            device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # Move the model and data to the specified device
    net.to(device=device)
    input = _input.to(device=device)
    target = _target.to(device=device)
    mask = _mask.to(device=device)

    # Calculate and print the initial loss
    with torch.no_grad():
        initial_loss = loss_mse(net(input), target, mask)
        print(f"initial loss: {initial_loss.item()}")

    # Training Loop
    for epoch in range(n_epochs):
        begin = Time.time()

        # Batch training
        for i in range(num_examples // batch_size):
            # Clearing gradient
            optimizer.zero_grad()

            # Randomly sampled batch index
            random_batch_idx = random.sample(range(num_examples), batch_size)

            # Get the current batch data
            batch = input[random_batch_idx]

            # Forward propagation
            output = net(batch)

            # Calculate losses
            loss = loss_mse(output, target[random_batch_idx], mask[random_batch_idx])
            losses.append(loss.item())

            # backpropagation
            loss.backward()

            # Gradient cropping
            if clip_gradient is not None:
                torch.nn.utils.clip_grad_norm_(net.parameters(), clip_gradient)

            # Update parameters
            optimizer.step()

            # DScoSGD optimizer processing
            with torch.no_grad():
                if if_dscosgd and epoch > 60:
                    # Get and process the circular weights
                    wrec = (net.D_mask * torch.relu(net.wrec_plastic.data)).detach().cpu().numpy().copy()
                    wrec_positive = wrec[wrec > 0]
                    wrec_negative = -wrec[wrec < 0]

                    # Fit log-normal distributions
                    params_pos = scipy.stats.lognorm.fit(wrec_positive, method='mle')
                    params_neg = scipy.stats.lognorm.fit(wrec_negative, method='mle')

                    # Calculate the DScoSGD parameters
                    learning_rate2 = 0.1 * epoch/n_epochs
                    mu_e = np.log(params_pos[2])
                    mu_i = np.log(params_neg[2])
                    loc_e = params_pos[1]
                    loc_i = params_neg[1]
                    sigma_e = params_pos[0]
                    sigma_i = params_neg[0]

                    # Apply DScoSGD optimization
                    dsco_sgd = DScoSGD(net, mu_e, mu_i, loc_e, loc_i, sigma_e, sigma_i, learning_rate2)
                    dsco_sgd.apply()

            # Free up memory to prevent memory leaks
            loss.detach_()
            output.detach_()

            # Print the training progress
            if epoch != 0:
                print(f"epoch {15 * epoch + i}:  loss={loss.item()}  (took: {Time.time() - begin} s) ")
            else:
                print(f"epoch {i}:  loss={loss.item()}  (took: {Time.time() - begin} s)")

    # Returns to the list of training losses
    return losses

def train_with_reg(net, _input, _target, _mask, n_epochs, lr=1e-2, batch_size=32, clip_gradient=None, cuda=True,
                   l1_lambda=0.001):
    """
    Network training function with L1 regularization

    Parameters:
    Train the neural network model
    :param net: The neural network model to be trained (nn.Module)
    :param _input: Enter the data tensor, shaped like (num_trials, num_timesteps, input_dim)
    :param _target: The object data tensor, shaped like (num_trials, num_timesteps, output_dim)
    :param _mask: Mask tensor, shaped by  (num_trials, num_timesteps, 1)
    :param n_epochs: Training epochs(int)
    :param lr: Learning rate (float)
    :param batch_size: Batch size (int)
    :param if_dscosgd: Whether to use the DScoSGD optimizer(bool)
    :param clip_gradient: Gradient clipping threshold, None means no clipping (None or float)
    :param cuda: Whether CUDA is used (bool)
    :param l1_lambda: L1 regularization coefficient (float)
    :return: List of loss values
    """
    print("Training...")

    # Initialize the Adam optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    # Obtain the total number of samples
    num_examples = _input.shape[0]

    # Used to store loss and gradient norms during training
    losses = []
    gradient_norms = []

    # CUDA Device Management
    if cuda:
        if not torch.cuda.is_available():
            print("Warning: CUDA not available on this machine, switching to CPU")
            device = torch.device('cpu')
        else:
            device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # Move the model and data to the specified device
    net.to(device=device)
    input = _input.to(device=device)
    target = _target.to(device=device)
    mask = _mask.to(device=device)

    # Calculate and print the initial loss (no gradient calculations)
    with torch.no_grad():
        initial_loss = loss_mse(net(input), target, mask)
        print(f"initial loss: {initial_loss.item()}")

    # Start the training loop
    for epoch in range(n_epochs):
        begin = Time.time()

        # Process data by batch
        for i in range(num_examples // batch_size):
            # Clearing gradient
            optimizer.zero_grad()

            # A random sample of a batch
            random_batch_idx = random.sample(range(num_examples), batch_size)
            batch = input[random_batch_idx]

            # Forward propagation
            output = net(batch)

            # Calculate MSE losses
            mse_loss = loss_mse(output, target[random_batch_idx], mask[random_batch_idx])

            # Calculate L1 regularization terms (L1 penalty for cyclic weight wrec)
            l1_penalty = torch.sum(torch.abs(net.wrec))

            # Total loss = MSE loss + L1 regularization
            loss = mse_loss + l1_lambda * l1_penalty
            losses.append(loss.item())

            # backpropagation
            loss.backward()

            # Gradient clipping (if clip_gradient is set)
            if clip_gradient is not None:
                torch.nn.utils.clip_grad_norm_(net.parameters(), clip_gradient)

            # Update model parameters
            optimizer.step()

            # Free the computational graph to prevent memory leaks
            loss.detach_()
            output.detach_()

            # Print the training progress
            if epoch != 0:
                print(f"epoch {15 * epoch + i}:  loss_mes={mse_loss.item()}  loss_l1={l1_penalty.item()}  (took: {Time.time() - begin} s) ")
            else:
                print(f"epoch {i}:  loss_mes={mse_loss.item()}  loss_l1={l1_penalty.item()}  (took: {Time.time() - begin} s)")

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

class LowRankRNN(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, noise_std, alpha, rho=0.1, rank=1,
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
            trajectories = torch.zeros(batch_size, seq_len + 1, self.hidden_size, device=self.m.device)
            trajectories[:, 0, :] = h

        # simulation loop
        for i in range(seq_len):
            h = h + self.noise_std * noise[:, i, :] + self.alpha * (-h + r.matmul(self.wrec.t()) +
                                                                    input[:, i, :].matmul(self.wi_full))
            r = self.non_linearity(h)
            output[:, i, :] = r.matmul(self.wo_full)
            if return_dynamics:
                trajectories[:, i + 1, :] = h

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
            h = h + self.noise_std * noise[:, i, :] + self.alpha * (-h + r.wrec +
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

class EIRNN(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, noise_std, alpha=0.2, rho=0.1,
                 train_wi=False, train_wo=False, train_wrec=True, train_h0=False, wi_init=None,
                 wo_init=None, wrec_init=None, si_init=None, so_init=None, h0_init=None, e_ratio=0.8,
                 apply_dale=True):
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
        super(EIRNN, self).__init__()
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
        self.D = torch.cat([torch.ones(hidden_size, self.e_size), -torch.ones(hidden_size, self.i_size)],
                           1).float()
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

    def forward(self, input, return_dynamics=False, return_noise=False, apply_dale=True):
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
        elif return_dynamics == True and not return_noise:
            return output, trajectories
        else:
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

def gen_intervals(ts_final, N_steps, ts_min=0.4):
    '''
    Generates time intervals for training schedule, from faster to slower intervals

    Parameters
    ----------
    ts_final : np.array 2D
        Final time intervals to be trained on
    N_steps : int
        Number of intermediate training steps
    ts_min : double, optional
        Fastest interval, as a fraction of final interval. The default is 0.4.

    Returns
    -------
    Tss : 2D array, with the time intervals for intervals at each time step

    '''
    Tss = np.zeros((len(ts_final), N_steps))
    stp_pr = np.linspace(ts_min, 1.0, N_steps)
    for i in range(N_steps):
        Tss[:, i] = stp_pr[i] * ts_final
    return (Tss)

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

        # draw a QQ chart
        if plot_QQ:
            # Draw a QQ chart with positive values
            stats.probplot(matrix_pos, dist=stats.lognorm, sparams=params_pos, plot=plt)
            plt.title("QQ Plot for Positive Weights", fontsize=12)
            plt.xlabel("Theoretical Quantiles")
            plt.ylabel("Sample Quantiles")
            plt.show()

            # Draw a QQ chart with negative values (pay attention to negative values)
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

def give_vectors(sigma1, sigma2, s_m1, s_m2, s=1, hidden_units=1500, max_iter=100, bn2=0.5):
    bigSigma = np.zeros((5, 5))  # 2*rank+input
    bigSigma[0, 0] = s_m1
    bigSigma[1, 1] = s_m2
    bigSigma[2, 2] = 1.
    bigSigma[3, 3] = 1.
    bigSigma[4, 4] = s ** 2
    bigSigma[0, 2] = sigma1
    bigSigma[2, 0] = sigma1
    bigSigma[1, 3] = sigma2
    bigSigma[3, 1] = sigma2
    bigSigma[3, 4] = s * bn2
    bigSigma[4, 3] = s * bn2

    stop = False
    ite = 0
    while stop == False and ite < max_iter:
        if np.min(np.linalg.eigvals(bigSigma)) < 0:
            bigSigma[2, 2] = 1.1 * bigSigma[2, 2]
            bigSigma[3, 3] = 1.1 * bigSigma[3, 3]
            ite += 1
        else:
            bigSigma[2, 2] = 1.1 * bigSigma[2, 2]
            bigSigma[3, 3] = 1.1 * bigSigma[3, 3]
            stop = True

    mean = np.zeros(5)
    error0 = 10.
    for K in range(100):
        X = np.random.multivariate_normal(mean, bigSigma, hidden_units)

        empSig = np.dot(X.T, X) / hidden_units
        error = np.std(empSig - bigSigma)
        if error < error0:
            error0 = error
            X_save = X
    X = X_save
    empSig = np.dot(X.T, X) / hidden_units

    M = X[:, 0:2]
    N = X[:, 2:4]
    I = X[:, 4]
    return (M, N, I)

def run_FP_fs(M, N, T=180, dt=0.2, trajs=1):
    dims = np.shape(M)[1]
    time = np.arange(0, T, dt)

    for tr in range(trajs):
        ks = np.zeros((dims, len(time)))
        k0 = 0.1 * np.random.randn(dims)
        ks[:, 0] = k0
        for it, ti in enumerate(time[:-1]):
            ks[:, it + 1] = ks[:, it] + dt * (def_field_net(ks[:, it], M, N))[0]

    return (np.dot(M, ks[:, -1]), ks[:, -1])

def def_field_net(k0, M, N):
    sol = -k0
    dims = len(k0)
    for d in range(dims):
        sol[d] += np.mean(N[:, d] * np.tanh(np.dot(M, k0)))

    E = np.sqrt(np.sum(sol ** 2))
    return (sol, E)

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

repeat = 1 # number of example trained networks. Can go up to 20
rank = 2
dt = 10  # ms
tau = 100  # ms
alpha = dt / tau
std_noise_rec = np.sqrt(2 * alpha) * 0.1
input_size = 2
hidden_size = 1500
output_size = 1
# initial connectivity
sigma_mn = 0.85
trials_train = 500
trials_test = 100
Nt = 350  # 2001
Nt2 = 550
time = np.arange(Nt)
time2 = np.arange(Nt2)
R_on = 1000 // dt  # 500//dt
# %%
# =============================================================================
#   Initialize inputs, outputs and recurrent connectivity (same as training initialization)
# =============================================================================
sigma1 = 0.8
sigma2 = 0.8
s_m1 = 1.
s_m2 = 1.
Mnaive, Nnaive, Inaive = give_vectors(sigma1, sigma2, s_m1, s_m2, s=1, hidden_units=hidden_size, max_iter=100, bn2=0.5)
Is_naive = np.random.randn() * Mnaive[:, 0] + np.random.randn() * Mnaive[:, 1]
O_naive = np.random.randn() * Mnaive[:, 0] + np.random.randn() * Mnaive[:, 1]
O_naive = O_naive[:, np.newaxis]
I_naive = np.vstack((Inaive, Is_naive))
# =============================================================================
#   Input intervals and input amplitudes
# =============================================================================
tss = np.array((800, 1550))
tss2 = np.array((800, 1050, 1300, 1550))
gain = 2.
# tss4 = np.linspace(tss2[0], tss2[0] + gain * (tss2[-1] - tss2[0]), 32)
# tss4 = np.linspace(200, 2300, 32)
amps = np.linspace(0, 0.25, 4)
# amps4 = np.linspace(-0.2, 0.25 * gain, len(tss4))
# tss4 = np.linspace(tss2[0], tss2[0] + gain * (tss2[-1] - tss2[0]), 32)
# amps4 = np.linspace(0, 0.25 * gain, len(tss4))
tss4 = np.linspace(200, 2300, 32)
amps4 = np.linspace(-0.2, 0.25 * gain, len(tss4))
N_steps = 5
Tss = gen_intervals(tss, N_steps)
Tss2 = gen_intervals(tss2, N_steps)
Tss3 = gen_intervals(tss2, N_steps)
Tss4 = gen_intervals(tss4, N_steps)
# %%
train_ = False
time = np.arange(Nt)
# Parameters of task
SR_on = 60
factor = 1
dela = 120
# =============================================================================
#   Define colors
# =============================================================================
# Colors
cls2 = set_plot()
cls2[1, :] = cls2[2, :]
cls2[2, :] = cls2[4, :]
cls2[3, :] = cls2[5, :]
cls = np.zeros((7, 3))
cl11 = np.array((102, 153, 255)) / 255.
cl12 = np.array((53, 153, 53)) / 255.
cl21 = np.array((255, 204, 51)) / 255.
cl22 = np.array((204, 0, 0)) / 255.
cls[0, :] = 0.4 * np.ones((3,))
cls[1, :] = cl11
cls[2, :] = 0.5 * cl11 + 0.5 * cl12
cls[3, :] = cl12
cls[4, :] = cl21
cls[5, :] = 0.5 * cl21 + 0.5 * cl22
cls[6, :] = cl22
cls4 = np.zeros((21, 3))
cl11 = np.array((102, 153, 255)) / 255.
cl12 = np.array((53, 153, 53)) / 255.
cl21 = np.array((255, 204, 51)) / 255.
cl22 = np.array((204, 0, 0)) / 255.
# New colors
cl11 = np.array((71, 89, 156)) / 255.  # p.array((102, 153, 255))/255.
cl12 = np.array((53, 153, 53)) / 255.
cl21 = np.array((255, 204, 51)) / 255.
cl22 = np.array((203, 81, 71)) / 255.  # np.array((204, 0, 0))/255.
cls4[0, :] = 0.4 * np.ones((3,))
cls4[1 * 3, :] = cl11
cls4[2 * 3, :] = 0.5 * cl11 + 0.5 * cl12
cls4[3 * 3, :] = cl12
cls4[4 * 3, :] = cl21
cls4[5 * 3, :] = 0.5 * cl21 + 0.5 * cl22
cls4[6 * 3, :] = cl22
# New colors April 2021
cls[5, :] = '0.3'
cls[4, :] = '0.6'
cls[3, :] = cl11  # 0.4*np.ones((3,))
cls[2, :] = cl21  # 0.5*cl11+0.5*cl12
cls[1, :] = cl12
cls[0, :] = cl22

loss_delete_all = np.zeros((10, 20))
loss_delete_all2 = np.zeros((10, 20))

os.makedirs('Figures_CSG_Structure/', exist_ok=True)

for i in range(6):
    cls4[i * 3 + 1, :] = (2. / 3) * cls4[i * 3, :] + (1. / 3) * cls4[(i + 1) * 3, :]
    cls4[i * 3 + 2, :] = (1. / 3) * cls4[i * 3, :] + (2. / 3) * cls4[(i + 1) * 3, :]

cl_low = (190 / 255, 197 / 255, 213 / 255)
cl_full = (228 / 255, 107 / 255, 144 / 255)
cl_bio = (255 / 255, 127 / 255, 14 / 255)
color_e = (209 / 255, 87 / 255, 73 / 255)
color_i = (28 / 255, 92 / 255, 158 / 255)

# =============================================================================
#   Task input and output
# =============================================================================
_input, _output, _mask, _ct, _ct2 = create_inp_out2(1, Nt2, Tss3[:, 4] // dt, amps, R_on, 1, just=2, perc=0.)
inputt = _input.detach().cpu().numpy()
outputt = _output.detach().cpu().numpy()
fig_width = 1.5 * 2.2  # width in inches
fig_height = 1.5 * 2.0  # height in inches
fig_size = [fig_width, fig_height]
fig = plt.figure(figsize=fig_size)
ax0, ax1, ax2 = fig.add_subplot(311), fig.add_subplot(312), fig.add_subplot(313)
colors = ["0.3", "0.1"]

#  input
ax0.plot(np.arange(-100, Nt2) / R_on, np.concatenate((100*[0], inputt[0,:,0])), '-', lw=3, c=colors[0])
ax1.plot(np.arange(-100, Nt2) / R_on, np.concatenate((100*[0], inputt[0,:,1])), '-', lw=3, c=colors[0])
ax0.axvline((1 + R_on) / R_on, ls='--', lw=2, c='0.6', ymax=0.9, zorder=1)
ax0.axvline((1 + R_on + tss2[2] / dt) / R_on, ls='--', lw=2, c='0.6',ymax=0.9, zorder=1)
ax0.text((1 + R_on) / R_on, amps[2] * 1.1, r'$Set$', fontsize=12, ha='center', zorder=2, usetex=True)
ax0.text((1 + R_on + tss2[2] / dt) / R_on, amps[2] * 1.1, r'$Go$', fontsize=12, ha='center', zorder=2, usetex=True)

#  pulse
ax1.axvline((1 + R_on) / R_on, ls='--', lw=2, c='0.6', ymax=20, zorder=1)

#  output
ax2.plot(np.arange(-100, Nt2) / R_on, np.concatenate((100*[-0.5], outputt[0,:,0])), '-', lw=3, c=colors[1])
ax2.axvline((1 + R_on) / R_on, ls='--', lw=2, c='0.6', zorder=1)
ax2.axvline((1 + R_on + tss2[2] / dt) / R_on, ls='--', lw=2, c='0.6', zorder=1)
ax2.text(1.65, 0.55,
         r'$\leftarrow t_{\mathrm{out}} \rightarrow$',
         ha='center', va='center', fontsize=16, usetex=True)

for i, ax in enumerate([ax0, ax1, ax2]):
    ax.set_yticklabels([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xlim([-0.2, 4.1])
    if i != 2:
        ax.set_xticks([])

ax0.set_ylabel("input1", fontsize=12)
ax1.set_ylabel("input2", fontsize=12)
ax2.set_ylabel("output", fontsize=12)
ax2.set_xlabel("time(s)", fontsize=12)
ax2.set_xticks([0, 2, 4])

plt.tight_layout()
plt.subplots_adjust(hspace=0.5)

string = f'CSG_Input&Output.png'
print(string)
plt.savefig('Figures_CSG_Structure/' + string, dpi=300, bbox_inches='tight')
string = f'CSG_Input&Output.pdf'
print(string)
plt.savefig('Figures_CSG_Structure/' + string, bbox_inches='tight')
plt.show()

# =============================================================================
#   Load data from networks
# =============================================================================
T0_lr_all = []
T0_fr_all = []
set_plot()

for tr in range(repeat):
    A = np.load('TrainedNets/net_CSG' + str(tr) + '.npz')  # This is how the networks are initialized
    M = A['arr_0']
    N = A['arr_1']
    Is = A['arr_2']
    Wo = A['arr_3']

    corrWo = 0.7
    Wo = corrWo * Wo[:, np.newaxis] / hidden_size

    dtype = torch.FloatTensor
    mrec_i = M / np.sqrt(hidden_size)
    nrec_i = N / np.sqrt(hidden_size)
    mrec_I = torch.from_numpy(mrec_i).type(dtype)
    nrec_I = torch.from_numpy(nrec_i).type(dtype)
    inp_I = torch.from_numpy(Is.T).type(dtype)
    out_I = torch.from_numpy(Wo).type(dtype)

    mrec_naive_i = Mnaive / np.sqrt(hidden_size)
    nrec_naive_i = Nnaive / np.sqrt(hidden_size)
    mrec_naive_I = torch.from_numpy(mrec_i).type(dtype)
    nrec_naive_I = torch.from_numpy(nrec_i).type(dtype)
    inp_naive_I = torch.from_numpy(I_naive).type(dtype)
    out_naive_I = torch.from_numpy(O_naive / hidden_size).type(dtype)
    print('Repeat ' + str(tr))
    i = N_steps - 1
    # train only directly on longest intervals
    H0_MN, k0_ = run_FP_fs(M, N, T=180, dt=0.2, trajs=1)
    if k0_[0] > 0:
        H0_MN = -H0_MN
    dtype = torch.FloatTensor
    h0_MN = torch.from_numpy(H0_MN).type(dtype)

    net_low = OptimizedLowRankRNN(input_size, hidden_size, output_size, 0.1 * std_noise_rec, alpha,
                                  rank=rank, train_wi=True, train_wrec=True, train_wo=True, train_h0=True,
                                  wi_init=inp_I, wo_init=out_I, m_init=mrec_I, n_init=nrec_I, h0_init=h0_MN)

    net_fr = FullRankRNN(input_size, hidden_size, output_size, 0.1 * std_noise_rec, alpha,
                         train_wi=True, train_wrec=True, train_wo=True, train_h0=True)

    wrec_dscosgd = create_wrec_init(hidden_size)
    net_DScoSGD = BIORNN(input_size, hidden_size, output_size, 0.1 * std_noise_rec, alpha, wrec_init=wrec_dscosgd,
                         train_wi=True, train_wrec=True, train_wo=True, train_h0=True, e_ratio=0.8, apply_dale=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("GPU is available.")
    else:
        device = torch.device("cpu")
        print("GPU is not available, using CPU.")
    loss_low = np.load("CSG_NETS/" + str(tr) + "CSG_LowRank_Train_loss.npz")['arr_0']
    net_low.load_state_dict(
        torch.load("CSG_NETS/" + str(tr) + "CSG_LowRank_Train_net.pt", map_location=device))
    loss_full = np.load("CSG_NETS/" + str(tr) + "CSG_FullRank_Train_loss.npy")
    net_fr.load_state_dict(
        torch.load("CSG_NETS/" + str(tr) + "CSG_FullRank_Train_net.pt", map_location=device))
    loss_bio = np.load("CSG_NETS/" + str(tr) + "CSG_BioRNN_Train_loss.npz")['arr_0']
    net_DScoSGD.load_state_dict(
        torch.load("CSG_NETS/" + str(tr) + "CSG_BioRNN_Train_net.pt", map_location=device))

    wrec = (net_DScoSGD.D_mask * torch.relu(net_DScoSGD.wrec_plastic.data)).detach().cpu().numpy().copy()
    net_low.to(device)
    net_fr.to(device)
    net_DScoSGD.to(device)

# =================================================================================
#   wrec histogram
# =================================================================================
    fig_width = 1.5 * 2.6  # width in inches
    fig_height = 1.5 * 2.0  # height in inches
    fig_size = [fig_width, fig_height]
    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(111)

    color_e = (209 / 255, 87 / 255, 73 / 255)
    color_i = (28 / 255, 92 / 255, 158 / 255)
    matrix_pos = wrec[wrec > 0]
    matrix_neg = wrec[wrec < 0]

    sns.histplot(matrix_pos, bins=100, kde=True, stat='density', label='兴奋性', color=color_e, ax=ax,
                 alpha=0.6, line_kws={'linewidth': 2, 'linestyle': '--'})
    # Plotting negative data (histogram + KDE)
    sns.histplot(matrix_neg, bins=100, kde=True, stat='density', label='抑制性', color=color_i, ax=ax,
                 alpha=0.6, line_kws={'linewidth': 2, 'linestyle': '--'})

    # ax.hist(matrix_pos, bins=100, density=True, , alpha=0.5, label='excitatory weights', color=color_e)
    # ax.hist(matrix_neg, bins=100, density=True, cumulative=True, alpha=0.5, label='inhibitory weights', color=color_i)
    ax.set_xlim(-0.1, 0.1)

    ax.legend(loc=2, frameon=False, framealpha=1., labelspacing=0.1, handlelength=1., fontsize=10)
    ax.set_xlabel(r'$w$', fontsize=14, usetex=True)
    ax.set_ylabel(r'$p$(w)', fontsize=14, usetex=True)

    print(ax.get_position())

    # Create an embedded axis
    params_pos = scipy.stats.lognorm.fit(matrix_pos, method='mle')
    s_e, loc_e, scale_e = params_pos

    params_neg = scipy.stats.lognorm.fit(-matrix_neg, method='mle')
    s_i, loc_i, scale_i = params_neg

    # Positive weight QQ map
    s_pos, p_pos = scipy.stats.kstest(matrix_pos, 'lognorm', args=params_pos)
    s_neg, p_neg = scipy.stats.kstest(-matrix_neg, 'lognorm', args=params_neg)

    ax_inset1 = inset_axes(ax, width="35%", height="35%", loc='upper right', bbox_to_anchor=(0.01, 0.005, 1, 1), bbox_transform=ax.transAxes)
    osm, osr = stats.probplot(matrix_pos, dist=stats.lognorm, sparams=params_pos, plot=ax_inset1)
    points = ax_inset1.get_lines()[0]  # scatter
    line = ax_inset1.get_lines()[1]  # guides
    # Modify colors
    points.set_color(color_e)  # Scattering red
    points.set_marker('o')  # Maintain dot shape (optional)
    points.set_markersize(1.5)
    line.set_color('grey')  # The guides turn gray
    line.set_linewidth(0.5)

    ax_inset1.set_xlabel("theoretical quantiles", fontsize=8, labelpad=3)
    ax_inset1.set_ylabel("sample quantiles", fontsize=8, labelpad=3)
    ax_inset1.yaxis.set_label_position("left")

    ax_inset1.set_xticks([])
    ax_inset1.set_yticks([])

    # ax_inset1.yaxis.tick_right()
    # ax_inset1.tick_params(axis='both', labelsize=6)
    ax_inset1.set_title('')

    # Negative weight QQ chart
    ax_inset2 = inset_axes(ax, width="35%", height="35%", loc='lower right', bbox_to_anchor=(0.01, 0.1, 1, 1), bbox_transform=ax.transAxes)
    osm, osr = stats.probplot(-matrix_neg, dist=stats.lognorm, sparams=params_neg, plot=ax_inset2)
    points = ax_inset2.get_lines()[0]  # scatter
    line = ax_inset2.get_lines()[1]  # guides
    # Modify colors
    points.set_color(color_i)  # scattering Red
    points.set_marker('o')  # Maintain dot shape (optional)
    points.set_markersize(1.5)
    line.set_color('grey')  # The guides turn gray
    line.set_linewidth(0.5)

    ax_inset2.set_xlabel("theoretical quantiles", fontsize=8, labelpad=3)
    ax_inset2.set_ylabel("sample quantiles", fontsize=8, labelpad=3)
    ax_inset2.set_xticks([])
    ax_inset2.set_yticks([])

    # ax_inset2.tick_params(axis='both', labelsize=6)
    ax_inset2.set_title('')

    string = f'CSG_Weight_Distribution.png'
    print(string)
    plt.savefig('Figures_CSG_Structure/' + string, dpi=300, bbox_inches='tight')
    string = f'CSG_Weight_Distribution.pdf'
    print(string)
    plt.savefig('Figures_CSG_Structure/' + string,bbox_inches='tight')
    plt.show()

    # =================================================================================
    #   Network Training error
    # =================================================================================
    fig_width = 1.5 * 2.2  # width in inches
    fig_height = 1.5 * 2.0  # height in inches
    fig_size = [fig_width, fig_height]
    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(111)

    iterations_per_epoch = 15
    loss_array = np.array(loss_bio)
    # Reshape the data into a set of every 15 iterations
    loss_reshaped = loss_array.reshape(-1, iterations_per_epoch)
    # Calculate the average for each group
    loss_epoch_mean = np.mean(loss_reshaped, axis=1)[4:]
    # Number of epochs generated
    epochs = np.arange(len(loss_epoch_mean))

    # loss_epoch_mean = loss_bio[4:501]
    # epochs = np.arange(len(loss_epoch_mean))
    ax.plot(epochs, loss_epoch_mean, '-', c='0.35', lw=1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    loss_0 = loss_epoch_mean[0]
    total_epochs = len(loss_epoch_mean)
    ax.set_xlim(-0.05 * total_epochs, 1.05 * total_epochs)
    ax.set_yticks([loss_0])
    ax.set_yticklabels([r"$l_0$"], fontsize=12, usetex=True)
    # ax.set_xticks([250, 500])
    ax.set_xlabel(r"$epoch$", fontsize=14,usetex=True)
    ax.set_ylabel(r"$loss$", labelpad=0, fontsize=12)
    ax.axhline(0, ls='--', c='0.7', zorder=-1)
    # ============================== Insert an inline image ================================

    ax_inset1 = inset_axes(ax, width="50%", height="50%", loc="upper right")
    trials = 1
    for xx in range(len(Tss3[:, i])):
        input_tr, output_tr, mask_tr, ct_train, ct2_train = create_inp_out2(trials, Nt, Tss3[:, i] // dt, amps,
                                                                            R_on, 1, just=xx, perc=0.)
        # bio rnn
        outp_bio = net_DScoSGD.forward(input_tr, return_dynamics=False)
        outp_bio = outp_bio.detach().numpy()
        avg_outp0 = np.mean(outp_bio[:, :, 0], 0)

        ax_inset1.plot((time * dt - 1000) / 1000, avg_outp0, '-', color=cls[xx, :], lw=1, aa=True, zorder=10)
        # ax_inset1.plot((time * dt - 1000) / 1000, np.mean(output_tr[:, :, 0].detach().numpy(), 0), '-', color='grey', lw=1)
    ax_inset1.plot([], [], '-', color='k', lw=1, aa=True, label='heavy-tailed')
    ax_inset1.plot([0, 0], [-0.5, 0.5], '--', color='grey', lw=1, zorder=-2,alpha=0.7)

    ax_inset1.axhline(-0.5, ls='--', lw=1, c='k', zorder=-2, alpha=0.7)
    ax_inset1.axhline(0.5, ls='--', lw=1, c='k', zorder=-2, alpha=0.7)

    ax_inset1.spines['top'].set_visible(False)
    ax_inset1.spines['right'].set_visible(False)
    ax_inset1.yaxis.set_ticks_position('left')
    ax_inset1.xaxis.set_ticks_position('bottom')
    ax_inset1.set_xlabel('time after set(s)', fontsize=10)
    ax_inset1.set_ylabel('output', fontsize=10, labelpad=0)
    ax_inset1.set_ylim([-0.55, 0.55])
    ax_inset1.set_xlim([-0.2, 1.8])
    # ax_inset1.set_xticks(fontsize=10)
    ax_inset1.set_yticks([-0.5, 0, 0.5])
    # Set the font size of the tick label
    ax_inset1.tick_params(axis='x', labelsize=8)
    ax_inset1.tick_params(axis='y', labelsize=8)

    # ============================== Insert an inline image ================================
    string = f'CSG_TrainingError2.png'
    print(string)
    plt.savefig('Figures_CSG_Structure/' + string, dpi=300, bbox_inches='tight')
    string = f'CSG_TrainingError2.pdf'
    print(string)
    plt.savefig('Figures_CSG_Structure/' + string, bbox_inches='tight')
    plt.show()

    # =================================================================================
    #   Heat map
    # =================================================================================
    fig_width = 1.5 * 2.2  # width in inches
    fig_height = 1.5 * 2.0  # height in inches
    fig_size = [fig_width, fig_height]
    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(111)

    abs_max = np.abs(wrec).max()
    linthresh = abs_max * 1e-4  # Threshold for linear regions, adjustable
    n_front = int(wrec.shape[0] * 0.8)
    n_back = wrec.shape[0] - n_front
    # Use symmetrical logarithmic normalization
    im = ax.imshow(
        wrec,
        cmap='RdBu_r',  # Red-blue reverse, positive numbers are blue, negative numbers are red
        norm=SymLogNorm(linthresh=linthresh, linscale=1, vmin=-abs_max, vmax=abs_max),
        aspect='equal',
        interpolation='nearest'  # Avoid interpolation ambiguity
    )
    color_e = (209 / 255, 87 / 255, 73 / 255)
    color_i = (28 / 255, 92 / 255, 158 / 255)

    # plt.axvline(x=80, color='k', linestyle='--',lw=2 )
    # plt.axhline(y=160, color='k', linestyle='--', alpha=0.5)
    # Add tags
    x_e = (n_front - 1) / 2
    x_i = n_front + (n_back - 1) / 2
    y_e = ((n_front / 0.8) - 1) * 1.1
    y_i = ((n_front / 0.8) - 1) * 1.1

    plt.text(x_e, y_e, "excitatory", ha='center', fontsize=10, color=color_e)
    plt.text(x_i, y_i, "inhibitory", ha='center', fontsize=10, color=color_i)

    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.05, aspect=20)
    cbar.set_label(r'$w$(log scale)', fontsize=10, labelpad=-10)
    ticks = [-0.1, 0, 0.1]
    tick_labels = ['-0.1', '0', '0.1']
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(tick_labels)
    plt.xticks([])
    plt.yticks([])
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    string = f'CSG_Hotmap.png'
    print(string)
    plt.savefig('Figures_CSG_Structure/' + string, dpi=300, bbox_inches='tight')
    string = f'CSG_Hotmap.pdf'
    print(string)
    plt.savefig('Figures_CSG_Structure/' + string, bbox_inches='tight')
    plt.show()

    # =================================================================================
    #  QQplot
    # =================================================================================
    fig_width = 1.5 * 2.2  # width in inches
    fig_height = 1.5 * 2.0  # height in inches
    fig_size = [fig_width, fig_height]
    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(111)

    matrix_pos = wrec[wrec > 0]
    matrix_neg = wrec[wrec < 0]

    params_pos = scipy.stats.lognorm.fit(matrix_pos, method='mle')
    s_e, loc_e, scale_e = params_pos

    mu_e = np.log(scale_e)  # Corresponding to the μ of log-normal
    sigma_e = s_e  # Corresponding to the σ of log-normal

    params_neg = scipy.stats.lognorm.fit(-matrix_neg, method='mle')
    s_i, loc_i, scale_i = params_neg

    mu_i = np.log(scale_i)  # Corresponding to the μ of log-normal
    sigma_i = s_i  # Corresponding to the σ of log-normal

    n_column_e = int(wrec.shape[0] * 0.8)
    n_column_i = wrec.shape[0] - n_column_e

    s_pos, p_pos = scipy.stats.kstest(matrix_pos, 'lognorm', args=params_pos)
    s_neg, p_neg = scipy.stats.kstest(-matrix_neg, 'lognorm', args=params_neg)
    print('positive {lognorm} KS-test p-value & s_pos:', p_pos, s_pos)
    print('negative {lognorm} KS-test p-value & s_neg:', p_neg, s_neg)

    # positive weight QQplot
    fig_width = 1.5 * 2.2  # width in inches
    fig_height = 1.5 * 2.0  # height in inches
    fig_size = [fig_width, fig_height]
    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(111)
    osm, osr = stats.probplot(matrix_pos, dist=stats.lognorm, sparams=params_pos, plot=ax)
    points = ax.get_lines()[0]  # scatter
    line = ax.get_lines()[1]  # guides
    # Modify colors
    points.set_color(color_e)  # Scattering red
    points.set_marker('o')  # Maintain dot shape (optional)
    points.set_markersize(4)
    line.set_color('grey')  # The guides turn gray
    line.set_linewidth(1.5)

    ax.set_xlabel("theoretical quantiles", fontsize=14)
    ax.set_ylabel("sample quantiles", fontsize=14)
    ax.set_title('')
    info_text = (f'$\mu^E = {mu_e:.3f}$\n  $\sigma^E = {sigma_e:.3f}$')
    ax.annotate(info_text,
                xy=(0.05, 0.95),
                xycoords='axes fraction',
                fontsize=12,
                ha='left', va='top',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='gray'))

    ax.grid(False)

    string = f'CSG_QQplot_E.png'
    print(string)
    plt.savefig('Figures_CSG_Structure/' + string, dpi=300, bbox_inches='tight')
    string = f'CSG_QQplot_E.pdf'
    print(string)
    plt.savefig('Figures_CSG_Structure/' + string, bbox_inches='tight')
    plt.show()

    # negative weight QQplot
    fig_width = 1.5 * 2.2  # width in inches
    fig_height = 1.5 * 2.0  # height in inches
    fig_size = [fig_width, fig_height]
    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(111)
    osm, osr = stats.probplot(-matrix_neg, dist=stats.lognorm, sparams=params_neg, plot=ax)
    points = ax.get_lines()[0]  # scatter
    line = ax.get_lines()[1]  # guides
    # Modify colors
    points.set_color(color_i)  # Scattering red
    points.set_marker('o')  # Maintain dot shape (optional)
    points.set_markersize(4)
    line.set_color('grey')  # The guides turn gray
    line.set_linewidth(1.5)

    ax.set_xlabel("theoretical quantiles", fontsize=14)
    ax.set_ylabel("sample quantiles", fontsize=14)
    ax.set_title('')
    info_text = (
        f'$\mu^I = {mu_i:.3f}$\n  $\sigma^I = {sigma_i:.3f}$')
    ax.annotate(info_text,
                xy=(0.05, 0.95),
                xycoords='axes fraction',
                fontsize=12,
                ha='left', va='top',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='gray'))
    ax.grid(False)

    string = f'CSG_QQplot_I.png'
    print(string)
    plt.savefig('Figures_CSG_Structure/' + string, dpi=300, bbox_inches='tight')
    string = f'CSG_QQplot_I.pdf'
    print(string)
    plt.savefig('Figures_CSG_Structure/' + string, bbox_inches='tight')
    plt.show()
