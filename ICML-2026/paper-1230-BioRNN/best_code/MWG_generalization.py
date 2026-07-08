from matplotlib.pyplot import legend
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scikit_posthocs import posthoc_dunn
from sklearn.decomposition import PCA
from matplotlib.patches import Rectangle, ConnectionPatch
from sklearn.preprocessing import StandardScaler
import modules4 as md
from math import sqrt, floor
import random
import time as Time
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import funcs_Sphere as fs
from matplotlib import cm
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.gofplots import qqplot
from scipy import stats
import scipy
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.stats import lognorm, norm, kruskal
import pickle
import numpy as np
from warnings import warn
from scipy import stats
import torch.nn.functional as F
from torch.distributions import Beta
import pandas as pd
import os
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import colors
from scipy.stats import gaussian_kde
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA
# from mpl_toolkits.mplot3d import Axes3D
# from sklearn.decomposition import PCA
# from sklearn.preprocessing import StandardScaler
from datetime import datetime

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
    plt.rcParams['text.usetex'] = False
    # plt.rcParams['text.latex.preamble'] = r'\usepackage{mathptmx}'

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

def create_inp_out_MWG(trials, Nt, tss, R1_on, SR1_on, fact=1., just=-1, perc=0.1, perc1=0.1, delayF=0,
                       delay_min=20, delay_max=250, align_set=False, inp_size=2, inc_mask_pre=30, inc_mask_post=30):
    '''
    Inputs
    ------
    trials:     Number of trials
    Nt :        Number of time points
    tss :       Intervals between set and go
    R1_on:      Time of ready
    SR1_on:     Possible deviation of the onset of "Ready".
    fact:       Scaling factor for the sampled interval (dividing)
    just:       Not given: all intervals, otherwise, selected interval index
    perc:       Percentage of trials in which no inputs appear
    perc1:      Percentage of trials in which only the ready cue appears
    delayF:     Fixed delay (if not given, variable)
    delay_min:  Minimum delay
    delay_max:  Maximum delay
    noset:
    noready:
    align_set:

    Outputs
    -------
    inputt:
    outputt:
    maskt:
    ct: Interval index at every trial
    ct2: Trials without inputs
    ct3: Trials without Set inputs
    '''

    n_ts = len(tss)
    time = np.arange(Nt)

    tss_comp = np.round(tss / fact)  # Produced intervals
    strt = -0.5  # Initial readout value

    if inp_size == 2:
        inputt = np.zeros((trials, Nt, 2))
    else:
        inputt = np.zeros((trials, Nt, 3))
    outputt = strt * np.ones((trials, Nt, 1))
    maskt = np.zeros((trials, Nt, 1))
    interval = np.min(tss_comp) // 2  # Minimal interval to be produced
    # inc_mask = 30                  # Minimal numbe of time points

    s_inp_R = np.zeros((trials, Nt))  # Ready
    s_inp_S1 = np.zeros((trials, Nt))
    s_inp_S2 = np.zeros((trials, Nt))

    if delayF == 0:
        delayF = np.round(np.mean((delay_min, delay_max)))

    if just == -1:  # all types of trials
        ct = np.random.randint(n_ts, size=trials)
    else:
        ct = just * np.ones(trials, dtype=np.int8)

    # Don't have nor set nor ready cue in a set of inputs
    ct2 = np.random.rand(trials) < perc

    # Don't have a set cue
    ct3 = np.random.rand(trials) < perc1

    rnd = np.zeros(trials)
    if SR1_on > 0:
        rnd = np.random.randint(-SR1_on, SR1_on, trials)  # random deviation at Ready onset

    for itr in range(trials):
        redset = tss[ct[itr]]  # produced interval
        redset_comp = tss_comp[ct[itr]]  # measurement interval
        delay = np.random.randint(delay_min, delay_max)

        if not align_set:  # Align at Ready
            maskt[itr, :, 0] = (time > R1_on + 1 + rnd[itr] + redset_comp + delay) * (
                    time < redset_comp + redset + R1_on + 1 + rnd[itr] + delay)
            mask_aft = time >= redset_comp + redset + R1_on + 1 + rnd[itr] + delay

            # Create Ready
            s_inp_R[itr, time > R1_on + rnd[itr]] = 10.
            s_inp_R[itr, time > R1_on + rnd[itr] + 1] = 0.
            # Create End of Measurement
            s_inp_S1[itr, time > R1_on + rnd[itr] + redset_comp] = 10.
            s_inp_S1[itr, time > 1 + R1_on + rnd[itr] + redset_comp] = 0.
            # Create Set
            s_inp_S2[itr, time > R1_on + rnd[itr] + redset_comp + delay] = 10.
            s_inp_S2[itr, time > 1 + R1_on + rnd[itr] + redset_comp + delay] = 0.

            # Create output
            if sum(maskt[itr, :, 0]):
                outputt[itr, maskt[itr, :, 0] == 1., 0] = np.linspace(strt, -strt, int(sum(maskt[itr, :, 0])),
                                                                      endpoint=True)
                outputt[itr, mask_aft == 1, 0] = np.linspace(-strt, -strt, int(sum(mask_aft)), endpoint=True)

            maskt[itr, :, 0] = (time > redset_comp + R1_on + 1 + rnd[itr] - inc_mask_pre + delay) * (
                    time < redset_comp + redset + R1_on + 1 + rnd[itr] + inc_mask_post + delay)

        else:  # Align at Set
            fixT = R1_on + np.max(tss)  # Set time (last input)
            redset = tss[ct[itr]]  # produced
            redset_comp = tss_comp[ct[itr]]  # measurement

            maskt[itr, :, 0] = (time > 1 + fixT - rnd[itr]) * (
                    time < redset + fixT + 1 - rnd[itr])  # (time>1+fixT-inc_mask)*(time<redset+fixT+1+inc_mask)
            mask_aft = time >= redset + 1 + fixT - rnd[itr]

            s_inp_R[itr, time > fixT - redset_comp - rnd[itr] - delayF] = 10.
            s_inp_R[itr, time > fixT - redset_comp + 1 - rnd[itr] - delayF] = 0.

            s_inp_S1[itr, time > fixT - delayF - rnd[itr]] = 10.
            s_inp_S1[itr, time > 1 + fixT - delayF - rnd[itr]] = 0.

            s_inp_S2[itr, time > fixT - rnd[itr]] = 10.
            s_inp_S2[itr, time > 1 + fixT - rnd[itr]] = 0.

            if sum(maskt[itr, :, 0]):
                outputt[itr, maskt[itr, :, 0] == 1., 0] = np.linspace(strt, -strt, int(sum(maskt[itr, :, 0])),
                                                                      endpoint=True)
                outputt[itr, mask_aft == 1, 0] = np.linspace(-strt, -strt, int(sum(mask_aft)), endpoint=True)

            maskt[itr, :, 0] = (time > 1 + fixT - inc_mask_pre - rnd[itr]) * (
                    time < redset + fixT + 1 + inc_mask_post - rnd[itr])

        if ct2[itr] == True:
            s_inp_R[itr, :] = 0.
            s_inp_S1[itr, :] = 0.
            s_inp_S2[itr, :] = 0.
            maskt[itr, :, 0] = time < Nt
            outputt[itr, :, 0] = strt
        elif ct3[itr] == True:
            s_inp_S2[itr, :] = 0.
            outputt[itr, :, 0] = strt
            maskt[itr, :, 0] = time < Nt

    if inp_size == 2:
        inputt[:, :, 0] += s_inp_R
        inputt[:, :, 0] += s_inp_S1
        inputt[:, :, 1] += s_inp_S2
    else:
        inputt[:, :, 0] += s_inp_R
        inputt[:, :, 1] += s_inp_S1
        inputt[:, :, 2] += s_inp_S2

    dtype = torch.FloatTensor
    inputt = torch.from_numpy(inputt).type(dtype)
    outputt = torch.from_numpy(outputt).type(dtype)
    maskt = torch.from_numpy(maskt).type(dtype)

    return (inputt, outputt, maskt, ct, ct2, ct3)

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
                if if_dscosgd and epoch > 270:
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
                 plot_gradient=False, clip_gradient=None, keep_best=False, cuda=False, resample=False, save_loss=False):
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

class EIRNN(nn.Module):

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
        elif return_dynamics == True and not return_noise:
            return output, trajectories
        else:
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

def get_SVDweights_CSG(net_low, rank=3):
    M = net_low.cpu().m.detach().numpy()
    N = net_low.cpu().n.detach().numpy()

    J_pre = np.dot(M, N.T)
    u, s, v = np.linalg.svd(J_pre)
    M_pre = u[:, 0:rank]
    N_pre = np.diag(s[0:rank]).dot(v[0:rank, :]).T
    corr_pre = np.dot(M_pre.T, N_pre)
    I = net_low.wi.detach().numpy()
    O = net_low.wo.detach().numpy()
    return (M_pre, N_pre, corr_pre, I, O, J_pre)

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

def low_pass(x, ints=5):
    y = np.copy(x)
    for ix in range(len(x)):
        lmin = np.max((0, ix - ints))
        lmax = np.min((len(x), ix + ints))
        y[ix] = np.mean(x[lmin:lmax])
    return (y)

def plot_output_MWG(net_low_all, tss2, dt, rank=3, give_trajs=False,
                    fr=False, t0s=False, gener=False, tss_ref=0, dela=150, only_perf=False):
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("GPU is available.")
    else:
        device = torch.device("cpu")
        print("GPU is not available, using CPU.")
    if fr == False:
        M_pre, N_pre, corr_pre, I_pre, O_pre, J_pre = get_SVDweights_CSG(net_low_all, rank=rank)

    CLL = cls
    if gener == True:
        evenly_spaced_interval = np.linspace(0, 1, len(tss2) + 10)
        cls2 = [cm.viridis(x) for x in evenly_spaced_interval]
        CLL = cls2
    trials = 10
    Trajs = np.zeros((len(time), rank, len(tss2)))
    T0s_lr = []

    dtss = tss2[1] - tss2[0]
    for xx in range(len(tss2)):
        input_train, output_train, mask_train, ct_train, ct2_train, ct3_train = create_inp_out_MWG(trials, Nt,
                                                                                                   tss2 // dt,
                                                                                                   R_on + dela, 1,
                                                                                                   just=xx, perc=0.,
                                                                                                   perc1=0.,
                                                                                                   fact=factor,
                                                                                                   align_set=True,
                                                                                                   delayF=dela,
                                                                                                   inp_size=3)
        input_train = input_train.to(device)
        output_train = output_train.to(device)
        mask_train = mask_train.to(device)
        net_low_all.to(device)
        outp, traj = net_low_all.forward(input_train, return_dynamics=True)
        outp = outp.detach().cpu().numpy()
        if fr == False:
            traj = traj.detach().cpu().numpy()

            mtraj = np.mean(traj, 0)
            k1_traj = M_pre[:, 0].dot(mtraj.T) / (np.sqrt(hidden_size) * np.sum(M_pre[:, 0] ** 2))
            k2_traj = M_pre[:, 1].dot(mtraj.T) / (np.sqrt(hidden_size) * np.sum(M_pre[:, 1] ** 2))
            k3_traj = M_pre[:, 2].dot(mtraj.T) / (np.sqrt(hidden_size) * np.sum(M_pre[:, 2] ** 2))
            Trajs[:, 0, xx] = k1_traj[:-1]
            Trajs[:, 1, xx] = k2_traj[:-1]
            Trajs[:, 2, xx] = k3_traj[:-1]
        outp2 = np.copy(outp)

        outp3 = np.copy(outp2[:, 1:, 0])
        # outp3[np.diff(outp2[:,:,0])<0]=5.
        outp3[:, np.diff(low_pass(np.mean(outp2[:, :, 0], 0))) < 0] = 5.

        outp3[:, time[1:] * dt < 4000] = 5.
        outp2 = outp3
        if gener == False:
            tt0s = time[np.argmin(np.abs(outp2 - 0.35), 1)] * dt - 4000
        else:
            tt0s = time[np.argmin(np.abs(outp2 - 0.35), 1)] * dt - 4000 - dela * dt
        T0s_lr.append(tt0s)
        avg_outp0 = np.mean(outp[:, :, 0], 0)  # np.mean(outp3[:,:,0],0)


    if give_trajs == True:
        return (Trajs)
    if t0s == True:
        return (T0s_lr)
    else:
        return (fig, ax)

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

# =============================================================================
#   MWG task parameter setting
# =============================================================================
rank = 3
dt = 10  # ms
tau = 100  # ms
alpha = dt / tau
std_noise_rec = np.sqrt(2 * alpha) * 0.1

input_size = 3
hidden_size = 1000
output_size = 1

size_f = np.sqrt(10)
# initial connectivity
sigma_mn = 0.85

trials_train = 500
trials_test = 100

# %%
tss = np.array((800, 1550))
tss2 = np.array((800, 1050, 1300, 1550))
tss3 = np.linspace(500, 3000, 30)
N_steps = 5

Tss = gen_intervals(tss, N_steps)
Tss2 = gen_intervals(tss2, N_steps)

# %%
Nt = 1000  # 1000
time = np.arange(Nt)

# Parameters of task
R_on = 1000 // dt  # 500//dt
SR_on = 60
factor = 1
dela = 150
# 7 #20 number of examples
# Colors
cls2 = set_plot()
cls2[1, :] = cls2[2, :]
cls2[2, :] = cls2[4, :]
cls2[3, :] = cls2[5, :]

initial_h0 = False
only_perf = True
train_ = False

repeat = 10 # 1

T0s_lr2 = []
T0s_fr2 = []
Evs_all = []
Evs_fr_all = []
P_all = []
P_fr_all = []
P_all2 = []
P_fr_all2 = []
T0_EI_all = []
T0_DiscoSGD_all = []
E_neg, I_neg, wrec_norm = [], [], []
# =============================================================================
#   Color settings
# =============================================================================
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
# New colors
cl11 = np.array((71, 89, 156)) / 255.  # p.array((102, 153, 255))/255.
cl12 = np.array((53, 153, 53)) / 255.
cl21 = np.array((255, 204, 51)) / 255.
cl22 = np.array((203, 81, 71)) / 255.  # np.array((204, 0, 0))/255.
# New colors April 2021
cls[3, :] = cl11  # 0.4*np.ones((3,))
cls[2, :] = cl21  # 0.5*cl11+0.5*cl12
cls[1, :] = cl12
cls[0, :] = cl22

# =============================================================================
#   Test whether the server is connected
# =============================================================================
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name())
set_plot()

os.makedirs('Figures_MWG_Generalization/', exist_ok=True)
os.makedirs('MWG_NETS/', exist_ok=True)
os.makedirs('TrainedNets_Generalization/', exist_ok=True)

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU is available.")
else:
    device = torch.device("cpu")
    print("GPU is not available, using CPU.")

cl_full = (228 / 255, 107 / 255, 144 / 255)
cl_low = (190 / 255, 197 / 255, 213 / 255)
cl_ei = (216 / 255, 194 / 255, 255 / 255)
cl_bio = (255 / 255, 127 / 255, 14 / 255)
cl_full_reg = (55 / 255, 127 / 255, 149 / 255)

rank_low = []
rank_full = []
rank_l1 = []
rank_ei = []
rank_bio = []

for tr in range(repeat):
    begin_time = datetime.now()
    begin_time = begin_time.strftime("%Y-%m-%d %H:%M:%S")
    i = N_steps - 1
    A = np.load('TrainedNets/net_MWG' + str(1) + '.npz')  # A = np.load('net_MWG'+str(tR+1)+'.npz')
    M = A['arr_0']
    N = A['arr_1']
    Is = A['arr_2']
    Wo = A['arr_3']
    cond0 = A['arr_4']
    corrWo = 1
    Wo = Wo / hidden_size
    if len(np.shape(Wo)) == 1:
        Wo = Wo[:, np.newaxis]
    flow = M.dot(N.T).dot(np.tanh(cond0)) - cond0
    dtype = torch.FloatTensor
    mrec_i = M
    nrec_i = N
    mrec_I = torch.from_numpy(mrec_i).type(dtype)
    nrec_I = torch.from_numpy(nrec_i).type(dtype)
    Is2 = np.zeros((hidden_size, input_size))
    Is2[:, input_size - 1] = N[:, -2]
    inp_I = torch.from_numpy(Is2.T).type(dtype)
    out_I = torch.from_numpy(Wo).type(dtype)
    h0_i = torch.from_numpy(cond0).type(dtype)

    net_low = OptimizedLowRankRNN(input_size, hidden_size, output_size, 0.1 * std_noise_rec, alpha,
                                     rank=rank, train_wi=True, train_wrec=True, train_wo=True, train_h0=True,
                                     wo_init=out_I, m_init=mrec_I, n_init=nrec_I, h0_init = h0_i)
    wrec_ei_dscosgd = create_wrec_init(hidden_size)

    net_EI = EIRNN(input_size, hidden_size, output_size, 0.1 * std_noise_rec, alpha, wrec_init=wrec_ei_dscosgd,
                   train_wi=True, train_wrec=True, train_wo=True, train_h0=True, e_ratio=0.8, apply_dale=True)
    net_fr = FullRankRNN(input_size, hidden_size, output_size, 0.1 * std_noise_rec, alpha,
                            train_wi = True, train_wrec=True, train_wo = True, train_h0=True)
    net_fr_reg = FullRankRNN(input_size, hidden_size, output_size, 0.1 * std_noise_rec, alpha,
                             train_wi=True, train_wrec=True, train_wo=True, train_h0=True)
    wrec_dscosgd = create_wrec_init(hidden_size)
    net_DScoSGD = BIORNN(input_size, hidden_size, output_size, 0.1 * std_noise_rec, alpha, wrec_init=wrec_dscosgd,
                              train_wi=True, train_wrec=True, train_wo=True, train_h0=True, e_ratio=0.8, apply_dale=True)
    if train_ == True:
        set_seed(tr)
        input_train, output_train, mask_train, ct_train, ct2_train, ct3_train = create_inp_out_MWG(trials_train,
             Nt, tss2 // dt, R_on, SR_on, delay_min=25, perc=0.1, perc1=0.1, fact=factor, align_set=False,inp_size=3)
        # =============================================================================
        # Train Full rank + L1
        print('train full-rank-reg')
        loss_full_reg = train_with_reg(net_fr_reg, input_train, output_train[:, :, 0:1], mask_train[:, :, 0:1],
                          n_epochs=140, lr=1e-4, clip_gradient=1., cuda=True, l1_lambda=0.00005)
        np.save("TrainedNets_Generalization/" + str(tr) + "MWG_FullRankReg_Train_loss", loss_full_reg)
        torch.save(net_fr_reg.state_dict(), "TrainedNets_Generalization/" + str(tr) + "MWG_FullRankReg_Train_net.pt")
        # =============================================================================
        # Train EIRNN
        print('train EIRNN')
        loss_ei = train(net_EI, input_train, output_train[:, :, 0:1], mask_train[:, :, 0:1], n_epochs=340,
                        lr=2e-4, if_dscosgd=False, clip_gradient=1., cuda=True)
        np.save("TrainedNets_Generalization/" + str(tr) + "MWG_EIRNN_Train_loss", loss_ei)
        torch.save(net_EI.state_dict(), "TrainedNets_Generalization/" + str(tr) + "MWG_EIRNN_Train_net.pt")
        # # =============================================================================
        # Train low-rank network -both inputs and outputs
        print('train low-rank')
        loss_low = train(net_low, input_train, output_train[:, :, 0:1], mask_train[:, :, 0:1],
                            n_epochs=300, lr=1e-3, clip_gradient=1.0,  if_dscosgd=False, cuda=True)
        np.savez("MWG_NETS/" + str(tr) + "MWG_LowRank_Train_loss", loss_low)
        torch.save(net_low.state_dict(), "MWG_NETS/" + str(tr) + "MWG_LowRank_Train_net.pt")
        # # =============================================================================
        # Train Full rank
        print('train full-rank')
        loss_full = train(net_fr, input_train, output_train[:, :, 0:1], mask_train[:, :, 0:1],
                          n_epochs=70, lr=1e-4, clip_gradient=1., if_DScoSGD=False, cuda=True)
        np.save("MWG_NETS/" + str(tr) + "MWG_FullRank_Train_loss", loss_full)
        torch.save(net_fr.state_dict(), "MWG_NETS/" + str(tr) + "MWG_FullRank_Train_net.pt")
        # # =============================================================================
        print("train biornn")
        loss_dscosgd = train(net_DScoSGD, input_train, output_train[:, :, 0:1], mask_train[:, :, 0:1],
                             n_epochs=340, lr=2e-4, clip_gradient=1., if_DScoSGD=True, cuda=True)
        np.savez("MWG_NETS/" + str(tr) + "MWG_BioRNN_train_loss", loss_dscosgd)
        torch.save(net_DScoSGD.state_dict(), "MWG_NETS/" + str(tr) + "MWG_BioRNN_net.pt")
        # # =============================================================================
        end_time = datetime.now()
        end_time = end_time.strftime("%Y-%m-%d %H:%M:%S")
        print(f'The start time of training is:{begin_time}', '\n', f'The end of the training time is:{end_time}')
    else:
        # =============================================================================
        #   Analyze the network
        # =============================================================================
        net_low.load_state_dict(torch.load("MWG_NETS/" + str(tr) + "MWG_LowRank_Train_net.pt", map_location=device))
        net_fr.load_state_dict(torch.load("MWG_NETS/" + str(tr) + "MWG_FullRank_Train_net.pt", map_location=device))
        net_DScoSGD.load_state_dict(torch.load("MWG_NETS/" + str(tr) + "MWG_BioRNN_net.pt", map_location=device))
        net_EI.load_state_dict(
            torch.load("TrainedNets_Generalization/" + str(tr) + "MWG_EIRNN_Train_net.pt", map_location=device))
        net_fr_reg.load_state_dict(
            torch.load("TrainedNets_Generalization/" + str(tr) + "MWG_FullRankReg_Train_net.pt", map_location=device))

        wrec_low = net_low.m.matmul(net_low.n.t()).detach().cpu().numpy().copy()
        wrec_fr = net_fr.wrec.detach().cpu().numpy().copy()
        wrec_bio = (net_DScoSGD.D_mask * torch.relu(net_DScoSGD.wrec_plastic.data)).detach().cpu().numpy().copy()
        wrec_ei = (net_EI.D_mask * torch.relu(net_EI.wrec_plastic.data)).detach().cpu().numpy().copy()
        wrec_reg = net_fr_reg.wrec.detach().cpu().numpy().copy()

        cl_full = (228 / 255, 107 / 255, 144 / 255)
        cl_low = (190 / 255, 197 / 255, 213 / 255)
        cl_ei = (216 / 255, 194 / 255, 255 / 255)
        cl_bio = (255 / 255, 127 / 255, 14 / 255)
        cl_full_reg = (55 / 255, 127 / 255, 149 / 255)

        set_seed(3)
        t0s_low2 = plot_output_MWG(net_low, tss3, dt, fr=False, t0s=True, gener=True, only_perf=True,
                                            tss_ref=tss2, dela=100)
        t0s_fr2 = plot_output_MWG(net_fr, tss3, dt, fr=True, t0s=True, gener=True, only_perf=True,
                                           tss_ref=tss2, dela=100)
        t0s_dscosgd2 = plot_output_MWG(net_DScoSGD, tss3, dt, fr=True, t0s=True, gener=True,
                                                only_perf=True,tss_ref=tss2, dela=100)
        t0s_ei2 = plot_output_MWG(net_EI, tss3, dt, fr=True, t0s=True, gener=True,
                                                only_perf=True, tss_ref=tss2, dela=100)
        t0s_fr_reg2 = plot_output_MWG(net_fr_reg, tss3, dt, fr=True, t0s=True, gener=True, only_perf=True,
                                  tss_ref=tss2, dela=100)
        # =============================================================================
        #   Nets Comparison
        # =============================================================================
        fig_width = 1.5 * 2.2  # width in inches
        fig_height = 1.5 * 2.0  # height in inches
        fig_size = [fig_width, fig_height]
        fig = plt.figure(figsize=fig_size)
        ax = fig.add_subplot(111)

        plt.plot(tss3, tss3, '--k')
        for xx in range(len(tss3)):
            if xx == 0:
                plt.scatter(tss3[xx] * np.ones(len(t0s_fr2[xx])), t0s_fr2[xx] / 0.85, color=cl_full, label='unconstrained')
                # plt.scatter(tss3[xx] * np.ones(len(t0s_fr_reg2[xx])), t0s_fr_reg2[xx] / 0.85, color=cl_full_reg,
                #             label=r'$L_1$')
                plt.scatter(tss3[xx] * np.ones(len(t0s_low2[xx])), t0s_low2[xx] / 0.85, color=cl_low, label='low-rank')
                plt.scatter(tss3[xx] * np.ones(len(t0s_ei2[xx])), t0s_ei2[xx] / 0.85, color=cl_ei, label='E-I')
                plt.scatter(tss3[xx] * np.ones(len(t0s_dscosgd2[xx])), t0s_dscosgd2[xx] / 0.85, color=cl_bio,label='heavy-tailed')
            else:
                plt.scatter(tss3[xx] * np.ones(len(t0s_fr2[xx])), t0s_fr2[xx] / 0.85, color=cl_full)
                # plt.scatter(tss3[xx] * np.ones(len(t0s_fr_reg2[xx])), t0s_fr_reg2[xx] / 0.85, color=cl_full_reg)
                plt.scatter(tss3[xx] * np.ones(len(t0s_low2[xx])), t0s_low2[xx] / 0.85, color=cl_low)
                plt.scatter(tss3[xx] * np.ones(len(t0s_ei2[xx])), t0s_ei2[xx] / 0.85, color=cl_ei)
                plt.scatter(tss3[xx] * np.ones(len(t0s_dscosgd2[xx])), t0s_dscosgd2[xx] / 0.85, color=cl_bio)

        plt.scatter(tss2, tss2, s=60, c='k')
        ax.set_xlabel(r'$t_s$ (ms)')
        ax.set_ylabel(r'$t_p$ (ms)')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        plt.plot(tss2, tss2, '--k')
        # ax.set_xticks([1000, 2000, 3000])
        # ax.set_yticks([1000, 2000, 3000])
        plt.ylim([600, 2500])
        plt.xlim([600, 3100])
        plt.legend()
        # plt.title('Generalization Points Test', fontsize=10, loc='center')
        # plt.show()

        # =============================================================================
        #   Generalization: mu + std
        # =============================================================================
        fig_width = 1.5 * 2.2  # width in inches
        fig_height = 1.5 * 2.0  # height in inches
        fig_size = [fig_width, fig_height]
        fig = plt.figure(figsize=fig_size)
        ax = fig.add_subplot(111)

        T0s_fr = np.array(t0s_fr2) / 0.85
        T0s_fr_reg = np.array(t0s_fr_reg2) / 0.85
        T0s_lr = np.array(t0s_low2) / 0.85
        T0s_bio = np.array(t0s_dscosgd2) / 0.85
        T0s_ei = np.array(t0s_ei2) / 0.85

        lr_mean = np.squeeze(np.mean(T0s_lr, 1)) / 1000
        lr_std = np.squeeze(np.std(T0s_lr, 1)) / 1000

        fr_mean = np.squeeze(np.mean(T0s_fr, 1)) / 1000
        fr_std = np.squeeze(np.std(T0s_fr, 1)) / 1000

        fr_reg_mean = np.squeeze(np.mean(T0s_fr_reg, 1)) / 1000
        fr_reg_std = np.squeeze(np.std(T0s_fr_reg, 1)) / 1000

        bio_mean = np.squeeze(np.mean(T0s_bio, 1)) / 1000
        bio_std = np.squeeze(np.std(T0s_bio, 1)) / 1000

        ei_mean = np.squeeze(np.mean(T0s_ei, 1)) / 1000
        ei_std = np.squeeze(np.std(T0s_ei, 1)) / 1000

        t0 = tss3 / 1000

        ax.plot(t0, fr_mean, '-', lw=2, color=cl_full, label='unconstrained')
        ax.plot(t0, fr_reg_mean, '-', lw=2, color=cl_full_reg, label=r'$\ell^1$-regularization')
        ax.plot(t0, lr_mean, '-', lw=2, color=cl_low, label='low-rank')
        ax.plot(t0, ei_mean, '-', lw=2, color=cl_ei, label='E-I')
        ax.plot(t0, bio_mean, '-', lw=2, color=cl_bio, label='heavy-tailed')

        ax.fill_between(t0, fr_mean - fr_std, fr_mean + fr_std, color=cl_full, alpha=0.5)
        ax.fill_between(t0, fr_reg_mean - fr_reg_std, fr_reg_mean + fr_reg_std, color=cl_full_reg, alpha=0.5)
        ax.fill_between(t0, lr_mean - lr_std, lr_mean + lr_std, color=cl_low, alpha=0.5)
        ax.fill_between(t0, ei_mean - ei_std, ei_mean + ei_std, color=cl_ei, alpha=0.5)
        ax.fill_between(t0, bio_mean - bio_std, bio_mean + bio_std, color=cl_bio, alpha=0.5)

        ax.plot(t0, t0, '--k', lw=0.8)
        for xx in range(len(tss2)):
            plt.scatter(tss2[xx] / 1000, tss2[xx] / 1000, marker='s', s=80, color=cls[xx, :], edgecolor='k',
                        zorder=4)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        ax.set_ylim([0.5, 2.8])  # 原来的设置
        ax.set_xlim([0.5, 2.4])
        ax.set_xlabel('input interval (s)', fontsize=14)
        ax.set_ylabel('output interval (s)', fontsize=14)
        ax.legend(loc=2, frameon=True, framealpha=0, labelspacing=0.1, handlelength=1.)

        string = f'MWG_Generalization.png'
        print(string)
        plt.savefig('Figures_MWG_Generalization/' + string, dpi=300, bbox_inches='tight')
        string = f'MWG_Generalization.pdf'
        print(string)
        plt.savefig('Figures_MWG_Generalization/' + string, bbox_inches='tight')
        string = f'MWG_Generalization.eps'
        print(string)
        plt.savefig('Figures_MWG_Generalization/' + string, bbox_inches='tight')
        plt.show()

# ==================================================================================================
#   Kruskal-Wallis Test
# ==================================================================================================
gener_error_low = np.zeros((10, 32, 10))
gener_error_full = np.zeros((10, 32, 10))
gener_error_bio = np.zeros((10, 32, 10))
gener_error_ei = np.zeros((10, 32, 10))
gener_error_reg = np.zeros((10, 32, 10))

cl_full = (228 / 255, 107 / 255, 144 / 255)
cl_low = (190 / 255, 197 / 255, 213 / 255)
cl_ei = (216 / 255, 194 / 255, 255 / 255)
cl_bio = (255 / 255, 127 / 255, 14 / 255)
cl_full_reg = (55 / 255, 127 / 255, 149 / 255)

repeat = 10
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU is available.")
else:
    device = torch.device("cpu")
    print("GPU is not available, using CPU.")
for tr in range(repeat):
    A = np.load('TrainedNets/net_MWG' + str(1) + '.npz')  # A = np.load('net_MWG'+str(tR+1)+'.npz')
    M = A['arr_0']
    N = A['arr_1']
    Is = A['arr_2']
    Wo = A['arr_3']
    cond0 = A['arr_4']
    corrWo = 1
    Wo = Wo / hidden_size
    if len(np.shape(Wo)) == 1:
        Wo = Wo[:, np.newaxis]
    flow = M.dot(N.T).dot(np.tanh(cond0)) - cond0
    dtype = torch.FloatTensor
    mrec_i = M
    nrec_i = N
    mrec_I = torch.from_numpy(mrec_i).type(dtype)
    nrec_I = torch.from_numpy(nrec_i).type(dtype)
    Is2 = np.zeros((hidden_size, input_size))
    Is2[:, input_size - 1] = N[:, -2]
    inp_I = torch.from_numpy(Is2.T).type(dtype)
    out_I = torch.from_numpy(Wo).type(dtype)
    h0_i = torch.from_numpy(cond0).type(dtype)

    net_low = OptimizedLowRankRNN(input_size, hidden_size, output_size, 0.1 * std_noise_rec, alpha,
                                     rank=rank, train_wi=True, train_wrec=True, train_wo=True, train_h0=True,
                                     wo_init=out_I, m_init=mrec_I, n_init=nrec_I, h0_init = h0_i)
    wrec_ei_dscosgd = create_wrec_init(hidden_size)
    net_EI = EIRNN(input_size, hidden_size, output_size, 0.0 * std_noise_rec, alpha, wrec_init=wrec_ei_dscosgd,
                   train_wi=True, train_wrec=True, train_wo=True, train_h0=True, e_ratio=0.8, apply_dale=True)
    net_fr_reg = FullRankRNN(input_size, hidden_size, output_size, 0.0 * std_noise_rec, alpha,
                             train_wi=True, train_wrec=True, train_wo=True, train_h0=True)
    net_fr = FullRankRNN(input_size, hidden_size, output_size, 0.0 * std_noise_rec, alpha,
                         train_wi=True, train_wrec=True, train_wo=True, train_h0=True)
    wrec_dscosgd = create_wrec_init(hidden_size)
    net_DScoSGD = BIORNN(input_size, hidden_size, output_size, 0.0 * std_noise_rec, alpha, wrec_init=wrec_dscosgd,
                         train_wi=True, train_wrec=True, train_wo=True, train_h0=True, e_ratio=0.8, apply_dale=True)

    net_low.load_state_dict(torch.load("MWG_NETS/" + str(tr) + "MWG_LowRank_Train_net.pt", map_location=device))
    net_fr.load_state_dict(torch.load("MWG_NETS/" + str(tr) + "MWG_FullRank_Train_net.pt", map_location=device))
    net_DScoSGD.load_state_dict(torch.load("MWG_NETS/" + str(tr) + "MWG_BioRNN_net.pt", map_location=device))
    net_EI.load_state_dict(torch.load("TrainedNets_Generalization/" + str(tr) + "MWG_EIRNN_Train_net.pt", map_location=device))
    net_fr_reg.load_state_dict(
        torch.load("TrainedNets_Generalization/" + str(tr) + "MWG_FullRankReg_Train_net.pt", map_location=device))

    net_low.to(device)
    net_fr.to(device)
    net_DScoSGD.to(device)
    net_EI.to(device)
    net_fr_reg.to(device)

    test_trials = 10
    for xx in range(len(tss3)):
        input_tr, output_tr, mask_tr, ct_train, ct2_train, ct3_train = create_inp_out_MWG(test_trials, Nt,
                                               tss3 // dt, R_on + 100, 1, just=xx, perc=0., perc1=0.,
                                               fact=factor, align_set=True, delayF=100, inp_size=3)
        input_tr = input_tr.to(device)
        output_tr = output_tr.to(device)
        mask_tr = mask_tr.to(device)

        outp = net_low.forward(input_tr, return_dynamics=False)
        outp = outp.detach().cpu().numpy()
        t0s_lr = time[np.argmin(np.abs(outp - 0.35), 1)] * dt - 4000 - 100 * dt
        t0s_lr_diff = np.abs(t0s_lr / 0.85 - tss3[xx])

        outp = net_fr.forward(input_tr, return_dynamics=False)
        outp = outp.detach().cpu().numpy()
        t0s_fr = time[np.argmin(np.abs(outp - 0.35), 1)] * dt - 4000 - 100 * dt
        t0s_fr_diff = np.abs(t0s_fr / 0.85 - tss3[xx])

        outp = net_EI.forward(input_tr, return_dynamics=False)
        outp = outp.detach().cpu().numpy()
        t0s_ei = time[np.argmin(np.abs(outp - 0.35), 1)] * dt - 4000 - 100 * dt
        t0s_ei_diff = np.abs(t0s_ei / 0.85 - tss3[xx])

        outp = net_DScoSGD.forward(input_tr, return_dynamics=False)
        outp = outp.detach().cpu().numpy()
        t0s_bio = time[np.argmin(np.abs(outp - 0.35), 1)] * dt - 4000 - 100 * dt
        t0s_bio_diff = np.abs(t0s_bio / 0.85 - tss3[xx])

        outp = net_fr_reg.forward(input_tr, return_dynamics=False)
        outp = outp.detach().cpu().numpy()
        t0s_reg = time[np.argmin(np.abs(outp - 0.35), 1)] * dt - 4000 - 100 * dt
        t0s_reg_diff = np.abs(t0s_reg / 0.85 - tss3[xx])

        gener_error_full[tr, xx] = np.squeeze(t0s_fr_diff)
        gener_error_low[tr, xx] = np.squeeze(t0s_lr_diff)
        gener_error_ei[tr, xx] = np.squeeze(t0s_ei_diff)
        gener_error_bio[tr, xx] = np.squeeze(t0s_bio_diff)
        gener_error_reg[tr, xx] = np.squeeze(t0s_reg_diff)

gener_error_full= np.mean(np.mean(gener_error_full, 2), 1) / 1000
gener_error_low = np.mean(np.mean(gener_error_low, 2), 1) / 1000
gener_error_ei = np.mean(np.mean(gener_error_ei, 2), 1) / 1000
gener_error_bio = np.mean(np.mean(gener_error_bio, 2), 1) / 1000
gener_error_reg = np.mean(np.mean(gener_error_reg, 2), 1) / 1000

print(f"FullRankReg: {gener_error_reg}")
mean_full = np.mean(gener_error_full)
sem_full = np.std(gener_error_full) / np.sqrt(len(gener_error_bio))
mean_low = np.mean(gener_error_low)
sem_low = np.std(gener_error_low) / np.sqrt(len(gener_error_low))
mean_ei = np.mean(gener_error_ei)
sem_ei = np.std(gener_error_ei) / np.sqrt(len(gener_error_ei))
mean_bio = np.mean(gener_error_bio)
sem_bio = np.std(gener_error_bio) / np.sqrt(len(gener_error_bio))
mean_reg = np.mean(gener_error_reg)
sem_reg = np.std(gener_error_reg) / np.sqrt(len(gener_error_reg))

fig_width = 1.5 * 2.2 * 1.4  # width in inches
fig_height = 1.5 * 2.0 * 1.4  # height in inches
fig_size = [fig_width, fig_height]
fig = plt.figure(figsize=fig_size)
ax = fig.add_subplot(111)

for xx in range(10):
    if xx == 0:
        ax.scatter(1, gener_error_full[xx], color=cl_full, edgecolor='k', s=40, alpha=0.8, label='unconstrained')
        ax.scatter(2, gener_error_reg[xx], color=cl_full_reg, edgecolor='k', s=40, alpha=0.8, label=r'$\ell^1$-regularization')
        ax.scatter(3, gener_error_low[xx], color=cl_low, edgecolor='k', s=40, alpha=0.8, label='low-rank')
        ax.scatter(4, gener_error_ei[xx], color=cl_ei, edgecolor='k', s=40, alpha=0.8, label='E-I')
        ax.scatter(5, gener_error_bio[xx], color=cl_bio, edgecolor='k', s=40, alpha=0.8, label='heavy-tailed')
    else:
        ax.scatter(1, gener_error_full[xx], color=cl_full, edgecolor='k', s=40, alpha=0.8)
        ax.scatter(2, gener_error_reg[xx], color=cl_full_reg, edgecolor='k', s=40, alpha=0.8)
        ax.scatter(3, gener_error_low[xx], color=cl_low, edgecolor='k', s=40, alpha=0.8)
        ax.scatter(4, gener_error_ei[xx], color=cl_ei, edgecolor='k', s=40, alpha=0.8)
        ax.scatter(5, gener_error_bio[xx], color=cl_bio, edgecolor='k', s=40, alpha=0.8)

# Draw a horizontal line of the mean ± SEM

# full rank
ax.errorbar(1.2, mean_full, yerr=sem_full, fmt='o', capsize=6, color=cl_full, elinewidth=1.5, markersize=8,
            capthick=1.5, markeredgecolor='k')
# full rank reg
ax.errorbar(2.2, mean_reg, yerr=sem_reg, fmt='o', capsize=6, color=cl_full_reg, elinewidth=1.5, markersize=8, capthick=1.5,
            markeredgecolor='k')
# low rank
ax.errorbar(3.2, mean_low, yerr=sem_low, fmt='o', capsize=6, color=cl_low, elinewidth=1.5, markersize=8, capthick=1.5,
            markeredgecolor='k')
# E-I
ax.errorbar(4.2, mean_ei, yerr=sem_ei, fmt='o', capsize=6, color=cl_ei, elinewidth=1.5, markersize=8, capthick=1.5,
            markeredgecolor='k')
# bio rnn
ax.errorbar(4.75, mean_bio, yerr=sem_bio, fmt='o', capsize=6, color=cl_bio, elinewidth=1.5, markersize=8, capthick=1.5,
            markeredgecolor='k')


ax.set_xticks([1, 2, 3, 4, 5], ['unconstrained', r'$\ell^1$-regularization', 'low-rank', 'E-I', 'heavy-tailed'])
ax.set_xlim([0.5, 5.5])
# ax.set_yticks([0.1, 0.2, 0.3, 0.4])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
ax.set_ylabel(r'$ \epsilon $', rotation=0, fontsize=16, labelpad=10)
# plt.legend(frameon=True, framealpha=1., labelspacing=0.1, handlelength=1., bbox_to_anchor=(1.1, 1),loc='upper right')
plt.xticks(rotation = 25)
string = f'MWG_Kruskal_Wallis_test.png'
print(string)
plt.savefig('Figures_MWG_Generalization/' + string, dpi=300, bbox_inches='tight')
string = f'MWG_Kruskal_Wallis_test.pdf'
print(string)
plt.savefig('Figures_MWG_Generalization/' + string, bbox_inches='tight')
string = f'MWG_Kruskal_Wallis_test.eps'
print(string)
plt.savefig('Figures_MWG_Generalization/' + string, bbox_inches='tight')
plt.show()

# =========================================Intergroup tests=================================================
fig_width = 1.5 * 2.2 * 1.4  # width in inches
fig_height = 1.5 * 2.0 * 1.4 # height in inches
fig_size = [fig_width, fig_height]
fig = plt.figure(figsize=fig_size)
ax = fig.add_subplot(111)

statistic, p_value = kruskal(gener_error_full, gener_error_reg, gener_error_low, gener_error_ei, gener_error_bio)
print(f"Kruskal-Wallisstatistics: {statistic}")
print(f"p : {p_value}")
# Determine whether the difference is significant
alpha = 0.05  # Significance level
if p_value < alpha:
    print("Rejecting the null hypothesis, there were significant differences between different groups")
else:
    print("The null hypothesis could not be rejected, and there was no significant difference between the groups")

# Organize the data into tidy-forms
df = pd.DataFrame({
    'error': np.concatenate([gener_error_full,
                             gener_error_reg,
                             gener_error_low,
                             gener_error_ei,
                             gener_error_bio]),
    'model': ['unconstrained'] * len(gener_error_full) + \
             [r'$\ell^1$-regularization'] * len(gener_error_reg) + \
             ['low-rank'] * len(gener_error_low) + \
             ['E-I'] * len(gener_error_ei) + \
             ['heavy-tailed'] * len(gener_error_bio)
})

# Dunn test (Holm correction)
dunn = posthoc_dunn(df, val_col='error', group_col='model', p_adjust='holm')
# output_filename = 'Figures_paper_figure3/dunn_test_results.xlsx'
# dunn.to_excel(output_filename, index=True)
print(f"p:\n{dunn}")

ax = sns.boxplot(x='model', y='error', data=df,
                 palette={'unconstrained': cl_full,
                         r'$\ell^1$-regularization':cl_full_reg,
                          'low-rank': cl_low,
                          'E-I': cl_ei,
                          'heavy-tailed': cl_bio},
                 linewidth=1.2, width=0.55)

# Draw a salient horizontal line + asterisk
def add_sig_bar(ax, x1, x2, y, p):
    """Draw a horizontal line between the two sets of x coordinates and mark them with an asterisk/value"""
    ax.plot([x1, x1, x2, x2], [y, y + 0.02, y + 0.02, y], lw=1.5, color='k')
    if p < 0.001:
        txt = '***'
    elif p < 0.01:
        txt = '**'
    elif p < 0.05:
        txt = '*'
    else:
        txt = 'ns'
    ax.text((x1 + x2) / 2, y + 0.01, txt, ha='center', va='bottom', color='k')

# Take the p-value in the Dunn table
y_max = df['error'].max()
# add_sig_bar(ax, 0, 1, y_max*1.05, dunn.loc['unconstrained', 'low-rank'])
add_sig_bar(ax, 0, 4, y_max * 1.05, dunn.loc['unconstrained', 'heavy-tailed'])
add_sig_bar(ax, 1, 4, y_max * 1.15, dunn.loc[r'$\ell^1$-regularization', 'heavy-tailed'])
add_sig_bar(ax, 2, 4, y_max * 1.25, dunn.loc['low-rank', 'heavy-tailed'])
add_sig_bar(ax, 3, 4, y_max * 1.35, dunn.loc['E-I', 'heavy-tailed'])

ax.set_ylabel('', fontsize=14)
ax.set_xlabel('')

# ax.set_yticklabels([])
sns.despine()
ax.set_xticklabels(['unconstrained', r'$\ell^1$-regularization', 'low-rank', 'E-I', 'heavy-tailed'], rotation=25)
string = f'MWG_Kruskal_Wallis_grouptest.png'
print(string)
plt.savefig('Figures_MWG_Generalization/' + string, dpi=300, bbox_inches='tight')
string = f'MWG_Kruskal_Wallis_grouptes.pdf'
print(string)
plt.savefig('Figures_MWG_Generalization/' + string, bbox_inches='tight')
string = f'MWG_Kruskal_Wallis_grouptes.eps'
print(string)
plt.savefig('Figures_MWG_Generalization/' + string, bbox_inches='tight')
plt.show()
# =========================================Intergroup tests=================================================


# =========================================Fusion diagram=================================================
fig_width = 1.5 * 2.2 * 1.4  # width in inches
fig_height = 1.5 * 2.0 * 1.4 # height in inches
fig_size = [fig_width, fig_height]
fig = plt.figure(figsize=fig_size)
ax = fig.add_subplot(111)

statistic, p_value = kruskal(gener_error_full, gener_error_reg, gener_error_low, gener_error_ei, gener_error_bio)
print(f"Kruskal-Wallisstatistics: {statistic}")
print(f"p : {p_value}")
#Determine whether the difference is significant
alpha = 0.05  # Significance level
if p_value < alpha:
    print("Rejecting the null hypothesis, there were significant differences between different groups。")
else:
    print("The null hypothesis could not be rejected, and there was no significant difference between the groups")

# Organize the data into tidy-forms
df = pd.DataFrame({
    'error': np.concatenate([gener_error_full,
                             gener_error_reg,
                             gener_error_low,
                             gener_error_ei,
                             gener_error_bio]),
    'model': ['unconstrained'] * len(gener_error_full) + \
             [r'$\ell^1$-regularization'] * len(gener_error_reg) + \
             ['low-rank'] * len(gener_error_low) + \
             ['E-I'] * len(gener_error_ei) + \
             ['heavy-tailed'] * len(gener_error_bio)
})

# Dunn test(fdr_bh Calibration)
dunn = posthoc_dunn(df, val_col='error', group_col='model', p_adjust='fdr_bh')
print(f"p : \n{dunn}")  # See the specific p-value

# Draw a box line map
ax = sns.boxplot(x='model', y='error', data=df,
                 palette={'unconstrained': cl_full,
                         r'$\ell^1$-regularization':cl_full_reg,
                          'low-rank': cl_low,
                          'E-I': cl_ei,
                          'heavy-tailed': cl_bio},
                 linewidth=1.2, width=0.4)

# =============================================== Add a scatter plot ===============================================
# Add scatter points to the right of each box plot
# Scatter Position Offset (Right)
offset = 0.35

# unconstrained (x=0)
for xx in range(10):
    ax.scatter(0 + offset, gener_error_full[xx], color=cl_full, edgecolor='k', s=40, alpha=0.8)

# l1-regularization (x=1)
for xx in range(10):
    ax.scatter(1 + offset, gener_error_reg[xx], color=cl_full_reg, edgecolor='k', s=40, alpha=0.8)

# low-rank (x=2)
for xx in range(10):
    ax.scatter(2 + offset, gener_error_low[xx], color=cl_low, edgecolor='k', s=40, alpha=0.8)

# E-I (x=3)
for xx in range(10):
    ax.scatter(3 + offset, gener_error_ei[xx], color=cl_ei, edgecolor='k', s=40, alpha=0.8)

# heavy-tailed (x=4)
for xx in range(10):
    ax.scatter(4 + offset, gener_error_bio[xx], color=cl_bio, edgecolor='k', s=40, alpha=0.8)

# =============================================== Add a salience horizontal line ===============================================
# Draw a salient horizontal line + asterisk
def add_sig_bar(ax, x1, x2, y, p):
    """Draw a horizontal line between the two sets of x coordinates and mark them with an asterisk/value"""
    ax.plot([x1, x1, x2, x2], [y, y + 0.02, y + 0.02, y], lw=1.5, color='k')
    if p < 0.001:
        txt = '***'
    elif p < 0.01:
        txt = '**'
    elif p < 0.05:
        txt = '*'
    else:
        txt = 'ns'
    ax.text((x1 + x2) / 2, y + 0.01, txt, ha='center', va='bottom', color='k')

# Take the p-value in the Dunn table
y_max = df['error'].max()
add_sig_bar(ax, 0, 4, y_max * 1.05, dunn.loc['unconstrained', 'heavy-tailed'])
add_sig_bar(ax, 1, 4, y_max * 1.15, dunn.loc[r'$\ell^1$-regularization', 'heavy-tailed'])
add_sig_bar(ax, 2, 4, y_max * 1.25, dunn.loc['low-rank', 'heavy-tailed'])
add_sig_bar(ax, 3, 4, y_max * 1.35, dunn.loc['E-I', 'heavy-tailed'])

ax.set_ylabel(r'$d$', fontsize=14,rotation=0, labelpad=10)
ax.set_xlabel('')

# ax.set_yticklabels([])
sns.despine()
ax.set_xticklabels(['unconstrained', r'$\ell^1$-regularization', 'low-rank', 'E-I', 'heavy-tailed'], rotation=25)

string = f'MWG_combined_plot.png'
print(string)
plt.savefig('Figures_MWG_Generalization/' + string, dpi=300, bbox_inches='tight')
string = f'MWG_combined_plot.pdf'
print(string)
plt.savefig('Figures_MWG_Generalization/' + string, bbox_inches='tight')
string = f'MWG_combined_plot.eps'
print(string)
plt.savefig('Figures_MWG_Generalization/' + string, bbox_inches='tight')
plt.show()