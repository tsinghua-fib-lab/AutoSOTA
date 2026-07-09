# Tools to plot particles

# Libraries
import numpy as np
from matplotlib.colors import LogNorm
from scipy.stats import gaussian_kde

def plot_ramachandran_hist2d(ax, phi_data, psi_data, bins=128, weights=None):
    """Plot the 2D histogram of Ramachandran coordinates

    Args:
        * ax (matplotlib.axes.Axes): Ax to display
        * phi_data (numpy.Array of shape (n,)): Phi coordinates
        * psi_data (numpy.Array of shape (n,)): Psi coordinates
        * bins (int): Number of bins (default is 128)
        * weights (numpy.Array of shape (n,)): Weights of the samples (default is None)

    Returns:
        * ax (matplotlib.axes.Axes): Ax to display
    """
    ax.hist2d(phi_data.flatten(), psi_data.flatten(), bins=bins, weights=weights,
        norm=LogNorm(vmin=1e-4, vmax=1.0), range=[[-np.pi, np.pi], [-np.pi, np.pi]], density=True)
    ax.set_xticks(np.arange(-np.pi, np.pi + np.pi/2, step=(np.pi/2)))
    ax.set_xticklabels(['-π', '-π/2', '0', 'π/2', 'π'])
    ax.set_yticks(np.arange(-np.pi, np.pi + np.pi/2, step=(np.pi/2)))
    ax.set_yticklabels(['-π', '-π/2', '0', 'π/2', 'π'])
    ax.set_xlabel(r'$\phi$')
    ax.set_ylabel(r'$\psi$')
    return ax

def filter_nan(x, y):
    """Filter out NaNs from a couple of arrays"""
    is_nan = np.logical_or(np.isnan(x), np.isnan(y))
    not_nan = np.logical_not(is_nan)
    x, y = x[not_nan], y[not_nan]
    return x, y

def plot_phi_hist1d(ax, phi_data, bins=256, weights=None):
    """Plot the 1D histogram of the first Ramachandran coordinate

    Args:
        * ax (matplotlib.axes.Axes): Ax to display
        * phi_data (numpy.Array of shape (n,)): Phi coordinates
        * bins (int): Number of bins (default is 256)
        * weights (numpy.Array of shape (n,)): Weights of the samples (default is None)

    Returns:
        * ax (matplotlib.axes.Axes): Ax to display
    """
    h_phi, _ = np.histogram(phi_data.flatten(), bins, range=[-np.pi, np.pi],
        weights=weights, density=True)
    x = np.linspace(-np.pi, np.pi, bins)
    ax.plot(x, h_phi, linewidth=3)
    ax.tick_params(axis='both', labelsize=20)
    ax.set_xlabel(r'$\phi$', fontsize=24)
    ax.set_ylabel(r'$p(\phi)$', fontsize=24)
    ax.set_xticks(np.arange(-np.pi, np.pi+np.pi/2, step=(np.pi/2)), ['-π','-π/2','0','π/2','π'])
    return ax

def plot_psi_hist1d(ax, psi_data, bins=256, weights=None):
    """Plot the 1D histogram of the second Ramachandran coordinate

    Args:
        * ax (matplotlib.axes.Axes): Ax to display
        * psi_data (numpy.Array of shape (n,)): Phi coordinates
        * bins (int): Number of bins (default is 256)
        * weights (numpy.Array of shape (n,)): Weights of the samples (default is None)

    Returns:
        * ax (matplotlib.axes.Axes): Ax to display
    """
    h_psi, _ = np.histogram(psi_data.flatten(), bins, range=[-np.pi, np.pi],
        weights=weights, density=True)
    x = np.linspace(-np.pi, np.pi, bins)
    ax.plot(x, h_psi, linewidth=3)
    ax.tick_params(axis='both', labelsize=20)
    ax.set_xlabel(r'$\psi$', fontsize=24)
    ax.set_ylabel(r'$p(\psi)$', fontsize=24)
    ax.set_xticks(np.arange(-np.pi, np.pi+np.pi/2, step=(np.pi/2)), ['-π','-π/2','0','π/2','π'])
    return ax

def plot_phi_psi_train_pred_hist1d(ax, phi_data, psi_data, phi_test, psi_test, bins=256, weights_source=None):
    """Plot the 1D histogram of Ramachandran coordinates with two datasets"""
    # Filter out NaNs
    phi_data, psi_data = filter_nan(phi_data, psi_data)
    phi_test, psi_test = filter_nan(phi_test, psi_test)
    # Build and display the histograms
    ax[0] = plot_phi_hist1d(ax[0], phi_data, bins=bins, weights=weights_source)
    ax[0] = plot_phi_hist1d(ax[0], phi_test, bins=bins)
    ax[1] = plot_psi_hist1d(ax[1], psi_data, bins=bins, weights=weights_source)
    ax[1] = plot_psi_hist1d(ax[1], psi_test, bins=bins)
    return ax

def plot_energy_hist1d(ax, en, range_limits, label, weights=None, bins=128):
    """Plot the 1D energy histogram"""
    ax.hist(ax, bins=bins, alpha=0.5, range=range_limits, density=True, label=label, weights=weights)
    ax.set_xlabel("Energy  / $k_B T$", fontsize=45)
    ax.set_ylabel('Density', fontsize=30)
    return ax

def plot_energy_train_pred_hist1d(ax, energy_data, energy_test, weights=None, bins=128):
    """Plot the 1D energy histograms of different datasets"""
    range_limits = (energy_test.min() - 10, energy_test.max() + 100)
    ax = plot_energy_hist1d(ax, energy_test, range_limits, "MD", bins=bins)
    ax = plot_energy_hist1d(ax, energy_data, range_limits, "Model", bins=bins)
    if weights is not None:
        ax = plot_energy_hist1d(ax, energy_data, range_limits, "Model-reweighed", weights=weights)
    ax.legend(fontsize=30)
    return ax

def free_energy_proj(samples, weights=None, kBT=1.0, bw_method=0.18):
    """Compute the free energy projection using kernel density estimation"""
    grid = np.linspace(samples.min(), samples.max(), 100)
    fes = -kBT * gaussian_kde(samples, bw_method, weights).logpdf(grid)
    fes -= fes.min()
    return grid, fes

def plot_free_energy_projection(ax, angles, eps=1e-5, weights=None, **kwargs):
    """Compute and plot the free energy curves for different transformations and weightings."""
    # Generate transformed phi values for left and right wrapping.
    phi_right = angles.flatten().copy()
    phi_right[angles < 0] += 2 * np.pi
    phi_left = angles.flatten().copy()
    phi_left[angles > np.pi / 2] -= 2 * np.pi
    grid_left, fes_left = free_energy_proj(phi_left, weights=weights)
    grid_right, fes_right = free_energy_proj(phi_right, weights=weights)
    # Extract relevant portions of grid and free energy based on a middle cutoff.
    middle = 0
    idx_left = (grid_left >= -np.pi) & (grid_left < middle)
    idx_right = (grid_right <= np.pi) & (grid_right > middle)
    grid_left, fes_left  = grid_left[idx_left], fes_left[idx_left]
    grid_right, fes_right = grid_right[idx_right], fes_right[idx_right]
    # Display the results
    ax.plot(np.hstack([grid_left, grid_right]), np.hstack([fes_left, fes_right]), **kwargs)
    return ax