# MIT License
#
# Copyright (c) 2025 tummfm
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.



"""Create plots of instantaneous quantities."""
import matplotlib.pyplot as plt
import numpy as onp

from jax_md_mod.model import prior
from jax_md import partition

from .util import create_subplots


def topology_from_neighbor_list(neighbor, species) -> prior.Topology:
    """Creates a topology based on distance information."""

    if neighbor.format == partition.Dense:
        n_particles, max_neighbors = neighbor.idx.shape
        senders = onp.repeat(onp.arange(n_particles), max_neighbors)
        receivers = onp.ravel(neighbor.idx)
        bond_idx = onp.stack((senders, receivers), axis=1)
    else:
        raise NotImplementedError("Only dense neighbor lists are supported.")

    mask = onp.logical_and(
        bond_idx[:, 0] < species.size,
        bond_idx[:, 1] < species.size
    )
    bond_idx = bond_idx[mask, :]
    assert species.size == n_particles, "Species and neighbor list must have the same size."

    top = prior.Topology(n_particles, species, bond_idx)
    return top


def plot_molecule(position, topology):
    """Plot ADFs.

    Args:
        ax: Add to existing axis
        adfs: Adfs to add to the plot
        labels: Label for each adf
        angles: Angles :math:`\\phi` corresponding to the adfs
        xlabel: Label of x-axis
        create_legend: Create a legend with the provided labels
        reference_first: Whether first adf is special, e.g., the reference

    Returns:
        Returns figure and axis of the ADF plot.

    """
    fig = plt.figure(layout="constrained")
    ax = fig.add_subplot(2, 2, 1, projection='3d')

    # Plot the atoms and assign the correct color to the species
    ax.scatter(position[:, 0], position[:, 1], position[:, 2],
               c=topology.get_atom_species())

    for idx1, idx2 in topology.get_bonds()[0]:
        ax.plot(position[(idx1, idx2), 0], position[(idx1, idx2), 1],
                position[(idx1, idx2), 2], color="k")

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    centers = position.mean(axis=0)
    box_dim = onp.max(onp.max(position, axis=0) - onp.min(position, axis=0))
    rmin = centers - 0.5 * box_dim
    rmax = centers + 0.5 * box_dim

    ax.set_xlim3d([rmin[0], rmax[2]])
    ax.set_ylim3d([rmin[1], rmax[2]])
    ax.set_zlim3d([rmin[2], rmax[2]])

    ax.view_init(elev=20., azim=-35, roll=0)

    coordinate = ["X", "Y", "Z"]
    for plt_idx, (i, j) in enumerate([(0, 1), (1, 2), (2, 0)]):
        ax = fig.add_subplot(2, 2, plt_idx + 2)

        # Plot the atoms and assign the correct color to the species
        ax.scatter(position[:, i], position[:, j], c=topology.get_atom_species())

        for idx1, idx2 in topology.get_bonds()[0]:
            ax.plot(position[(idx1, idx2), i], position[(idx1, idx2), j],
                    color="k")

        ax.set_xlabel(coordinate[i])
        ax.set_ylabel(coordinate[j])

    return fig
