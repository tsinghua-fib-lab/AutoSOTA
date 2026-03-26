import jax 
import jax.numpy as jnp
import numpy as onp
import matplotlib.pyplot as plt


def plot_flow(flow, x_init, y_init, xlim = None, n_pts = 1000, color = 'r', alpha = 1, title = "", **kwargs):
    """
    flow: flow map y = phi(x_init, x)(y_init)
    """
    if xlim is None:
        x_min, x_max = plt.xlim()
    else:
        x_min, x_max = xlim
    x_pts = jnp.linspace(x_min, x_max, n_pts)
    assert len(x_init) == len(y_init), "Initial conditions must have the same length."
    y = jnp.permute_dims(jax.vmap(flow, in_axes = (None, 0, 0))(x_pts, x_init, y_init), (1,0))
    plt.plot(x_pts, y, '-', color = color, linewidth=1, alpha = alpha, **kwargs)
    plt.title(title)
    plt.xlim(x_min, x_max)

def plot_data(x_data, y_data, alpha = 0.1, title = "", **kwargs):
    plt.scatter(x_data, y_data, color = 'black', s = 12, alpha = alpha, **kwargs)
    plt.title(title)

def plot_velocity_grid(v_yt, xlim = None, ylim = None, n_pts = 20, color = 'black', alpha = 0.8, title = "", **kwargs):
    """
    xlim, ylim: tuple of (min, max) for x and y axis
    if not given will use plt.xlim() and plt.ylim()
    v_yt: v(y, t) function (i.e., lambda y, t: v(y, t, params))
    """
    if xlim is None:
        x_min, x_max = plt.xlim()
    else:
        x_min, x_max = xlim
    if ylim is None:
        y_min, y_max = plt.ylim()
    else:
        y_min, y_max = ylim
    
    X, Y = jnp.meshgrid(jnp.arange(x_min, x_max, (x_max-x_min)/n_pts), jnp.arange(y_min, y_max, (y_max - y_min)/n_pts))
    v_yt = jax.vmap(v_yt, in_axes = (0, 0))
    U = jnp.ones_like(X)
    V = v_yt(Y, X).squeeze()
    magnitude = jnp.sqrt(U**2 + V**2)
    U, V = U/magnitude, V/magnitude
    plt.title(title)
    plt.quiver(X, Y, U, V, color = color, alpha = alpha, angles = "xy", **kwargs)

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

def plot_velocity_points(v_yt, x_pts, y_pts, color = 'black', alpha = 0.8, title = "", **kwargs):
    """
    plot velocity at specific points.
    """
    U = jnp.ones_like(x_pts)
    V = v_yt(y_pts, x_pts).squeeze()
    magnitude = jnp.sqrt(U**2 + V**2)
    U, V = U/magnitude, V/magnitude
    plt.title(title)
    plt.quiver(x_pts, y_pts, U, V, color = color, alpha = alpha, angles = "xy", **kwargs)
