# Matplotlib generation
import numpy as np
# import matplotlib; matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
import matplotlib as mpl
from typing import List, Tuple, Dict, Optional, Any, Union, Callable
from scipy.interpolate import UnivariateSpline

def animate_gridworld(width: int, height: int, states: List[Tuple[int]], values: np.array=None, nsteps: int=1,
                      constraints=None):
    """
    Generate matplotlib animation for gridworld.

    :param width: Gridworld width.
    :param height: Gridworld height.
    :param states: State trajectory.
    :param values: Array of values (for illustration).
    :param nsteps: Number of frames per action.
    :param constraints: 3d arrays containing coordinates of lower left and upper rigth corner (in the last dimension).
    :return: Animation.
    """

    # generate plot
    fig, ax = plt.subplots()
    ax.grid(which='minor', color='black', linestyle='-', linewidth=2)
    ax.minorticks_on()
    ax.invert_yaxis()

    # Major ticks
    ax.set_xticks(np.arange(0, width, 1))
    ax.set_yticks(np.arange(0, height, 1))

    # Labels for major ticks
    ax.set_xticklabels(np.arange(0, width, 1))
    ax.set_yticklabels(np.arange(0, height, 1))

    # Minor ticks
    ax.set_xticks(np.arange(-.5, width, 1), minor=True)
    ax.set_yticks(np.arange(-.5, height, 1), minor=True)

    # Draw plot
    if values is None:
        values = np.ones(width * height)
    values = values.reshape(height, width)
    im = ax.imshow(values, origin='lower')
    plt.colorbar(im)
    circ = plt.Circle(states[0], 0.2, color='w')
    ax.add_patch(circ)

    # Draw constraints
    if constraints is not None:
        for i in range(len(constraints[:,0,0])):
            bottom_left = constraints[i, 0, :]
            width = constraints[i, 1, 0]
            height = constraints[i, 1, 1]
            rect = patches.Rectangle(bottom_left-0.5, width, height, linewidth=2, edgecolor='r', facecolor='none', zorder=10)
            ax.add_patch(rect)

    # Update function
    def update(state):
        xy = np.array(state)
        circ.set_center(xy)
        return circ

    # Generate animation states (nsteps many steps in between two consecutive states).
    prev_x = states[0][0]
    prev_y = states[0][1]
    xs = [states[0][0]]
    ys = [states[0][0]]
    for x, y in states[1:]:
        xs = xs + np.linspace(prev_x, x, nsteps+1).tolist()[1:]
        ys = ys + np.linspace(prev_y, y, nsteps+1).tolist()[1:]
        prev_x = x
        prev_y = y

    return animation.FuncAnimation(fig, update, list(zip(xs, ys)), interval=100, save_count=50)

def plot_gridworld(env, policy, title, width: int, height: int, values: np.array=None,
                      constraints=None, logging = True, Draw_policy = True):
    """
    Generate matplotlib animation for gridworld.

    :param width: Gridworld width.
    :param height: Gridworld height.
    :param states: State trajectory.
    :param values: Array of values (for illustration).
    :param nsteps: Number of frames per action.
    :param constraints: 3d arrays containing coordinates of lower left and upper rigth corner (in the last dimension).
    :param rollouts: Trajectories to be displayed.
    :return: Animation.
    """

    # generate plot
    fig, ax = plt.subplots()
    ax.grid(which='minor', color='grey', linestyle='-', linewidth=2)
    ax.minorticks_on()
    ax.invert_yaxis()

    # Major ticks
    ax.set_xticks(np.arange(0, width, 1))
    ax.set_yticks(np.arange(0, height, 1))

    # Labels for major ticks
    ax.set_xticklabels(np.arange(1, width+1, 1))
    ax.set_yticklabels(np.arange(1, height+1, 1))

    # Minor ticks
    ax.set_xticks(np.arange(-.5, width, 1), minor=True)
    ax.set_yticks(np.arange(-.5, height, 1), minor=True)

    # Draw plot
    if values is None:
        values = np.ones(width * height)
    values = values.reshape(height, width)
    im = ax.imshow(values, origin='lower', cmap=mpl.colormaps['RdYlGn'], alpha=0.5, vmin=-1.0, vmax=1.0)
    if logging:
        plt.colorbar(im)
       
    plt.title(title, fontsize=18)
    # Draw constraints
    if constraints is not None:
        for i in range(len(constraints[:,0,0])):
            bottom_left = constraints[i, 0, :]
            width = constraints[i, 1, 0]
            height = constraints[i, 1, 1]
            rect = patches.Rectangle(bottom_left-0.5, width, height, linewidth=2, edgecolor='r', facecolor='none', zorder=10, hatch='/')
            ax.add_patch(rect)

    # Update function
    def update(state):
        xy = np.array(state)
        circ.set_center(xy)
        return circ

    # Draw policy
    if Draw_policy:
        # print(Draw_policy)
        c = 0.28
        for s in range(env.n):
            if env.r[s,0] == 0:
                for a, p in enumerate(policy[s, :]):
                    x,y  = env.int2point(s)
                    dx, dy = env.actions[a]
                    head_with = 0.1 * p ** (1/2)
                    ax.arrow(x, y, dx*p*c, dy*p*c, head_width=head_with, color='black')

    return fig

