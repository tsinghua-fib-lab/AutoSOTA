"""
Defines the window functions which are applied to the output of each subdomain network

This module is used by decompositions.py
"""

import jax.nn
import jax.numpy as jnp


def sigmoid(xmin, xmax, wmin, wmax, x, tol=1e-8):
    "window function, for a SINGLE point with shape (xdim)"

    t = jnp.log((1-tol)/tol)
    mu_min, sd_min = xmin + wmin/2, wmin/(2*t)
    mu_max, sd_max = xmax - wmax/2, wmax/(2*t)
    ws = jax.nn.sigmoid((x-mu_min)/sd_min)*jax.nn.sigmoid((mu_max-x)/sd_max)

    # multiply kernels together
    w = jnp.prod(ws, axis=0, keepdims=True)

    return w


def cosine(xmin, xmax, x):
    "window function, for a SINGLE point with shape (xdim)"

    mu, sd = (xmin+xmax)/2, (xmax-xmin)/2
    ws = ((1+jnp.cos(jnp.pi*(x-mu)/sd))/2)**2
    ws = jnp.heaviside(x-xmin,1)*jnp.heaviside(xmax-x,1)*ws

    # multiply kernels together
    w = jnp.prod(ws, axis=0, keepdims=True)

    return w


def bump(xmin, xmax, x):
    "window function, for a SINGLE point with shape (xdim)"

    mu, sd = (xmin+xmax)/2, (xmax-xmin)/2
    ws = jnp.exp(3/(((x-mu)/sd)**2-1.001))/jnp.exp(-3)

    # multiply kernels together
    w = jnp.prod(ws, axis=0, keepdims=True)

    return w

def wendland(xmin, xmax, x):
    "window function, for a SINGLE point with shape (xdim)"

    mu, sd = (xmin+xmax)/2, (xmax-xmin)/2
    r = jnp.abs((x-mu)/sd)
    ws = ((1-r)**4)*(4*r+1)
    ws = jnp.heaviside(x-xmin,1)*jnp.heaviside(xmax-x,1)*ws

    # multiply kernels together
    w = jnp.prod(ws, axis=0, keepdims=True)

    return w

def rbf(xmin, xmax, x):
    "window function, for a SINGLE point with shape (xdim)"

    mu, sd = (xmin+xmax)/2, (xmax-xmin)/2
    w = jnp.exp(-0.5*jnp.sum(((x-mu)/(0.25*sd))**2, keepdims=True))

    return w



if __name__ == "__main__":

    from jax import vmap
    import numpy as np
    import matplotlib.pyplot as plt


    def run_windows(xmin, xmax, wmin, wmax):

        # get test points (2 corners)
        x_test = np.stack([xmin, xmin+wmin, xmax, xmax-wmax], axis=0)# (4, xd)

        # get plot grid
        n = 100
        xd = xmin.shape[0]
        xs = [np.linspace(mi, ma, n) for mi,ma in zip(xmin, xmax)]
        xxs = np.stack(np.meshgrid(*xs, indexing="ij"), 0)# (xd, nm)
        x = xxs.reshape((xd, n**xd)).T# (n, xd)

        # run window functions
        windows = [
                (sigmoid, (xmin, xmax, wmin, wmax)),
                (cosine, (xmin, xmax)),
                (bump, (xmin, xmax)),
                (wendland, (xmin, xmax)),
                (rbf, (xmin, xmax)),
                ]
        wws, w_tests = [],[]
        for window_fn, params in windows:

            w_fn = vmap(window_fn, in_axes=([None,]*len(params)+[0]))
            wws.append(w_fn(*params, x).reshape((n,)*xd))# (nm)
            w_tests.append(w_fn(*params, x_test))# (4)

        return windows, x_test, w_tests, xxs, wws

    ## 1D test
    xmin, xmax = np.array([-5.5]), np.array([4])
    wmin, wmax = np.array([3]), np.array([4])
    windows, x_test, w_tests, xxs, wws = run_windows(xmin, xmax, wmin, wmax)
    for ww, w_test, (window_fn, _) in zip(wws, w_tests, windows):
        plt.figure()
        plt.title(window_fn.__name__)
        plt.plot(xxs[0,:], ww[:])
        for t in x_test:
            plt.axvline(t[0], color="tab:grey", alpha=0.2)
        plt.axhline(0, color="tab:grey", alpha=0.2)
        plt.show()
        print(x_test.T)
        print(w_test.T, w_test.shape)

    ## 2D test
    xmin, xmax = np.array([-9,-8]), np.array([9,8])
    wmin, wmax = np.array([10,3]), np.array([4,7])
    windows, x_test, w_tests, xxs, wws = run_windows(xmin, xmax, wmin, wmax)
    for ww, w_test, (window_fn, _) in zip(wws, w_tests, windows):
        plt.figure()
        plt.title(window_fn.__name__)
        plt.imshow(ww.T,# transpose as jnp.meshgrid uses indexing="ij"
                   origin="lower", extent=(xmin[0], xmax[0], xmin[1], xmax[1]),
                   cmap="viridis")
        plt.colorbar()
        for t in x_test:
            plt.scatter(t[0], t[1], color="tab:red")
        plt.gca().set_aspect("equal")
        plt.show()
        print(x_test.T)
        print(w_test.T, w_test.shape)