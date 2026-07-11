"""
Defines domain decompositions used for FBPINNs

The key elements which need to be defined are:
    - the number of subdomains
    - the normalisation, unnormalisation and window functions for each subdomain
    - the inside_models and inside_points methods, which determine whether a batch
      of training points is inside a batch of subdomains (and vice-versa)
    - a plotting method used during training

Each decomposition class must inherit from the Decomposition base class.
Each decomposition class must define the NotImplemented methods.

This module is used by constants.py (and subsequently trainers.py)
"""

import jax.numpy as jnp
from jax import vmap
from jax.tree_util import tree_map
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.collections as mcoll

from fbpinns import windows
from fbpinns import networks
from fbpinns.decompositions_base import inside_models_batch, inside_points_batch
from fbpinns.util.jax_util import tree_index
from fbpinns.util.other import colors


class Decomposition:
    """Base decomposition class to be inherited by different decomposition classes.

    Note all methods in this class are jit compiled / used by JAX,
    so they must not include any side-effects!
    (A side-effect is any effect of a function that doesn’t appear in its output)
    This is why only static methods are defined.
    """

    # required methods

    @staticmethod
    def init_params(*args):
        """Initialise class parameters.
        Returns tuple of dicts ({k: pytree}, {k: pytree}) containing static and trainable parameters.
        The special key, k='subdomain', should be used to specify all subdomain parameters"""

        # below parameters need to be defined
        static_params = {
            "m":None,# total number of models (i.e. subdomains) in domain
            "xd":None,# dimensionality of x
            "subdomain":{
                "pou":None,# pou for each subdomain
                }
            }
        raise NotImplementedError

    @staticmethod
    def norm_fn(params, x):
        """"Applies norm function, for a SINGLE point with shape (xd,) and params for a SINGLE model"""
        raise NotImplementedError

    @staticmethod
    def unnorm_fn(params, u):
        """"Applies unnorm function, for a SINGLE point with shape (ud,) and params for a SINGLE model"""
        raise NotImplementedError

    @staticmethod
    def window_fn(params, x):
        """"Applies window function, for a SINGLE point with shape (xd,) and params for a SINGLE model"""
        raise NotImplementedError

    @staticmethod
    def inside_points(all_params, x_batch):
        """Returns ips, ims, inside_ims, where
        ips, ims = indices of all point-model pairs where point is inside model
        inside_ims = indicies of all models which have at least one point inside them
        """
        raise NotImplementedError

    @staticmethod
    def inside_models(all_params, x_batch, ims):
        """Returns inside_ips, d, where
        inside_ips = indicies of all points which are at least inside one model in ims
        d = average number of points inside each model in ims
        """
        raise NotImplementedError

    # helper methods

    @staticmethod
    def plot(all_params, active=None, create_fig=True):
        "Plots decomposition. Returns matplotlib figure"
        raise NotImplementedError






class RectangularDecompositionND(Decomposition):
    """ND hyperrectangular domain.
    Rectangular subdomains can be placed arbitrarily in domain."""

    @staticmethod
    def init_params(subdomain_xs, subdomain_ws, unnorm):
        """Creates hyperrectangular subdomains initialised on a regular grid
        with subdomain centers subdomain_xs and widths subdomain_ws.
        """

        # get dimensionality of DD
        nm = tuple([len(x) for x in subdomain_xs])# shape of rectangular DD grid
        m = np.prod(nm)
        xd = len(subdomain_xs)# number of input dimensions

        # get level params
        ps = RectangularDecompositionND._get_level_params(0, xd, subdomain_xs, subdomain_ws, unnorm)

        # set constants for rectangular scheduler
        xmins0, xmaxs0 = (ps[0]+ps[2]/2), (ps[1]-ps[3]/2)# center lines of overlapping regions

        params = tree_map(lambda x: jnp.array(x), ps)

        static_params = {
            "m":m,
            "xd":xd,
            "subdomain":{"params":params[:-1],# use special key for subdomain parameters
                         "pou":params[-1]},
            "xmins0":xmins0,
            "xmaxs0":xmaxs0,
            }

        return static_params, {}

    @staticmethod
    def _get_level_params(il, xd, subdomain_xs, subdomain_ws, unnorm):

        # get subdomain extents
        xs = np.stack(np.meshgrid(*subdomain_xs, indexing="ij"), 0)# (xd, nm)
        ws = np.stack(np.meshgrid(*subdomain_ws, indexing="ij"), 0)# (xd, nm)
        if xs.shape != ws.shape:
            raise ValueError("shape of subdomain_ws not same as subdomain_xs")
        xmins, xmaxs = xs - (ws/2), xs + (ws/2)

        # get subdomain overlap widths
        wmins, wmaxs = 0.5*(xmaxs-xmins), 0.5*(xmaxs-xmins)# default value for 1 subdomain (xd, nm)
        for i in range(xd):
            # fill in slices
            sl0, sl1 = [slice(None),]*xd, [slice(None),]*xd
            sl0[i] = slice(None, -1); sl1[i] = slice(1, None)
            sl0, sl1 = (i,) + tuple(sl0), (i,) + tuple(sl1)
            wmaxs[sl0] = wmins[sl1] = xmaxs[sl0] - xmins[sl1]
            # fill in edges
            sl0, sl1 = [slice(None),]*xd, [slice(None),]*xd
            sl0[i] = 0; sl1[i] = -1
            sl0, sl1 = (i,) + tuple(sl0), (i,) + tuple(sl1)
            wmins[sl0] = wmaxs[sl0]
            wmaxs[sl1] = wmins[sl1]
        if (wmins <= 0).any() or (wmaxs <= 0).any():
            raise ValueError("some subdomains are not overlapping!")

        # flatten arrays
        f = lambda x: (x.reshape(xd, -1)).T# (m, xd)
        xmins, xmaxs = f(xmins), f(xmaxs)# (m, xd)
        wmins, wmaxs = f(wmins), f(wmaxs)# (m, xd)
        s = (xmins.shape[0], 1)

        # get flag for whether to apply window or not
        if xmins.shape[0] == 1:# 1 subdomain case
            flags = np.zeros(s)
        else:
            flags = np.ones(s)

        # get unnorm parameters
        unnorms = np.concatenate([unnorm[0]*np.ones(s), unnorm[1]*np.ones(s)], axis=1)

        # get pou index
        # important note: each POU MUST cover the entire domain (POU boundary introduces discontinuities)
        pous = il*np.ones(s)

        return [xmins, xmaxs, wmins, wmaxs, flags, unnorms, pous]

    @staticmethod
    def norm_fn(params, x):
        params = params["static"]["decomposition"]["subdomain"]["params"]
        xmin, xmax = params[:2]
        mu, sd = (xmax+xmin)/2, (xmax-xmin)/2
        return networks.norm(mu, sd, x)

    @staticmethod
    def unnorm_fn(params, u):
        params = params["static"]["decomposition"]["subdomain"]["params"]
        mu, sd = params[5]
        return networks.unnorm(mu, sd, u)

    @staticmethod
    def window_fn(params, x):
        params = params["static"]["decomposition"]["subdomain"]["params"]
        return params[4]*windows.cosine(*params[:2], x)+(1-params[4])

    @staticmethod
    def inside_points(all_params, x_batch):
        m = all_params["static"]["decomposition"]["m"]
        ims = jnp.arange(m)
        batch_size = min(int(1e9/(4*ims.shape[0])), x_batch.shape[0])# limit GPU memory
        all_params = {"params": all_params["static"]["decomposition"]["subdomain"]["params"]}# filter out subdomain params
        return inside_points_batch(all_params, x_batch, ims, batch_size,
                                   RectangularDecompositionND._inside_rectangleND)

    @staticmethod
    def inside_models(all_params, x_batch, ims):
        batch_size = min(int(1e9/(4*ims.shape[0])), x_batch.shape[0])# limit GPU memory
        all_params = {"params": all_params["static"]["decomposition"]["subdomain"]["params"]}# filter out subdomain params
        return inside_models_batch(all_params, x_batch, ims, batch_size,
                                   RectangularDecompositionND._inside_rectangleND)

    @staticmethod
    def _inside_rectangleND(all_params, x_batch, ims):
        "Code for assessing if point is in ND hyperrectangle"

        ps = all_params["params"]
        x_batch = jnp.expand_dims(x_batch, 1)# (n,1,xd)
        xmins = jnp.expand_dims(ps[0][ims], 0)# (1,mc,xd)
        xmaxs = jnp.expand_dims(ps[1][ims], 0)# (1,mc,xd)
        inside = (x_batch >= xmins) & (x_batch <= xmaxs)# (n,mc,xd)
        inside = jnp.all(inside, -1)# (n,mc) keep as bool to reduce memory
        return inside

    # helper methods

    @staticmethod
    def plot(all_params, iaxes=[0,1], active=None, show_norm=False, show_window=False, create_fig=True):
        p = all_params["static"]["decomposition"]
        params = {"static":{"decomposition":{"subdomain":p["subdomain"]}}}
        m, xd = p["m"], p["xd"]

        if active is None:
            active = np.ones(m)

        # 1D plots
        if xd == 1:

            if create_fig: f = plt.figure(figsize=(8,4))
            else: f = plt.gcf()

            for im in range(m):
                # get domain params
                param = tree_index(params, im)
                xmin, xmax, wmin, wmax, *_ = param["static"]["decomposition"]["subdomain"]["params"]
                mu, sd = (xmax+xmin)/2, (xmax-xmin)/2

                # plot subdomain
                h = -0.1-0.1*(im%4)
                plt.hlines(h, xmin[0], xmax[0], colors=colors[im],
                           linewidth=5 if active[im] else 2,
                           alpha=1 if active[im] else 0.5,
                           linestyle=":" if active[im]==2 else "-")

                if active[im]:
                    # plot active norm
                    if show_norm:
                        plt.scatter(mu[0], h, color=colors[im], s=100)
                        plt.scatter(mu[0]+sd[0], h, color=colors[im], s=100, edgecolor="k")

                    # plot active window
                    x = np.linspace(xmin[0], xmax[0], 100).reshape((-1,1))
                    w = vmap(RectangularDecompositionND.window_fn, in_axes=(None,0))(param, x)
                    plt.plot(x[:,0], w[:,0], c=colors[im],
                             linestyle=":" if active[im]==2 else "-")

            # plot summed windows (expensive!)
            if show_window:
                xmins, xmaxs, wmins, wmaxs, *_ = params["static"]["decomposition"]["subdomain"]["params"]
                xmin, xmax = xmins.min(0), xmaxs.max(0)
                x = np.linspace(xmin[0], xmax[0], 1000).reshape((-1,1))
                ws = vmap(vmap(RectangularDecompositionND.window_fn, in_axes=(None,0)), in_axes=(0,None))(params, x)
                w = ws.sum(0)
                plt.plot(x[:,0], w[:,0], color="tab:grey", alpha=0.5)

        # 2D+ plots
        else:
            a,b = iaxes
            if create_fig: f = plt.figure(figsize=(8,8))
            else: f = plt.gcf()

            # get domain params
            xmins, xmaxs, wmins, wmaxs, *_ = params["static"]["decomposition"]["subdomain"]["params"]
            mus, sds = (xmaxs+xmins)/2, (xmaxs-xmins)/2

            # plot subdomains
            lines = np.empty((m, 4, 2, 2))# 0,1,2,3 counterclockwise from x axes
            lines[:,0,0,0] = lines[:,2,0,0] = lines[:,3,0,0] = lines[:,3,1,0] = xmins[:,a]
            lines[:,0,1,0] = lines[:,1,0,0] = lines[:,1,1,0] = lines[:,2,1,0] = xmaxs[:,a]
            lines[:,0,0,1] = lines[:,0,1,1] = lines[:,1,0,1] = lines[:,3,0,1] = xmins[:,b]
            lines[:,1,1,1] = lines[:,2,0,1] = lines[:,2,1,1] = lines[:,3,1,1] = xmaxs[:,b]
            lws = np.array([[3 if active[im] else 1]*4 for im in range(m)]).flatten()
            alphas = np.array([[1 if active[im] else 0.5]*4 for im in range(m)]).flatten()
            lss = np.array([[":" if active[im]==2 else "-"]*4 for im in range(m)]).flatten()
            cs = np.array([[colors[im]]*4 for im in range(m)]).flatten()
            lines = mcoll.LineCollection(lines.reshape((-1,2,2)),
                                         linewidths=lws,
                                         alpha=alphas,
                                         linestyles=lss,
                                         colors=cs)
            plt.gca().add_collection(lines)

            # plot active norms
            if show_norm:
                alphas = np.array([1 if active[im] else 0 for im in range(m)])
                cs = np.array([colors[im] for im in range(m)])
                plt.scatter(mus[:,a], mus[:,b], color=cs, alpha=alphas, s=100)
                plt.scatter(mus[:,a]+sds[:,a], mus[:,b]+sds[:,b], color=cs, alpha=alphas, s=100, edgecolor="k")

            # plot summed windows (expensive!)
            if show_window:
                xmin, xmax = xmins.min(0), xmaxs.max(0)
                x = np.tile(np.expand_dims((xmax+xmin)/2, 0), (150**2, 1))
                xs = [np.linspace(mi, ma, 150) for mi,ma in zip(xmin[np.array([a,b])],xmax[np.array([a,b])])]
                xxs = np.stack(np.meshgrid(*xs, indexing="ij"), 0)# (2, nm)
                x_ = xxs.reshape((2, 150**2)).T
                x[:,a] = x_[:,0]; x[:,b] = x_[:,1]
                ws = vmap(vmap(RectangularDecompositionND.window_fn, in_axes=(None,0)), in_axes=(0,None))(params, x)
                ww = ws.sum(0).reshape((150,150))
                plt.imshow(ww.T,
                           origin="lower", extent=(xmin[a], xmax[a], xmin[b], xmax[b]),
                           cmap="bwr", vmin=0, vmax=2, zorder=-99)

            # set axis limits / labels / aspect ratio
            xmin, xmax = xmins.min(0), xmaxs.max(0)
            mi, ma = xmin-0.05*(xmax-xmin), xmax+0.05*(xmax-xmin)
            plt.xlim(mi[a], ma[a]); plt.ylim(mi[b], ma[b])
            plt.xlabel(a); plt.ylabel(b)
            plt.gca().set_aspect("equal")

        return f


class MultilevelRectangularDecompositionND(RectangularDecompositionND):
    """ND hyperrectangular domain, with multiple DDs at different scales.
    Rectangular subdomains can be placed arbitrarily in domain."""

    def init_params(subdomain_xss, subdomain_wss, unnorm):
        """Creates multiscale hyperrectangular subdomains initialised on a regular grid
        with subdomain centers subdomain_xs and widths subdomain_ws.
        """

        # get dimensionality of DD
        nms = [tuple([len(x) for x in subdomain_xs]) for subdomain_xs in subdomain_xss]# shape of rectangular DD grid
        if False in [len(nm)==len(nms[0]) for nm in nms]:
            raise ValueError("subdomain_xss are not all the same dimensionality")
        m = sum([np.prod(nm) for nm in nms])
        xd = len(subdomain_xss[0])# number of input dimensions

        # get level params
        ps = [[] for _ in range(7)]
        for il,(subdomain_xs, subdomain_ws) in enumerate(zip(subdomain_xss, subdomain_wss)):
            ps_ = RectangularDecompositionND._get_level_params(il, xd, subdomain_xs, subdomain_ws, unnorm)
            for i,p_ in enumerate(ps_): ps[i].append(p_)
        ps = [np.concatenate(p) for p in ps]

        # set constants for rectangular scheduler
        xmins0, xmaxs0 = (ps[0]+ps[2]/2), (ps[1]-ps[3]/2)# center lines of overlapping regions

        params = tree_map(lambda x: jnp.array(x), ps)

        static_params = {
            "m":m,
            "xd":xd,
            "subdomain":{"params":params[:-1],# use special key for subdomain parameters
                         "pou":params[-1]},
            "xmins0":xmins0,
            "xmaxs0":xmaxs0,
            }

        return static_params, {}



if __name__ == "__main__":

    ## 1D test

    subdomain_xs = [np.linspace(-3,3,4),]
    subdomain_ws = [3*np.ones(4),]

    decomposition = RectangularDecompositionND
    ps_ = decomposition.init_params(subdomain_xs, subdomain_ws, (0,1))
    all_params = {"static":{"decomposition":ps_[0]}, "trainable":{"decomposition":ps_[1]}}
    m = all_params["static"]["decomposition"]["m"]
    active = np.ones(m)

    active[1] = 0
    active[2] = 2
    decomposition.plot(all_params, active=active, show_norm=True, show_window=True)
    x_batch = np.array([[-2],
                        [1],
                        [3],
                        [4.7]])
    for x in x_batch:
        plt.scatter(x[0], 0.5)
    plt.show()
    print(decomposition.inside_models(all_params, x_batch, np.arange(m)))
    print(decomposition.inside_points(all_params, x_batch))

    # single subdomain test
    subdomain_xs = [np.array([0])]
    subdomain_ws = [np.array([1])]

    decomposition = RectangularDecompositionND
    ps_ = decomposition.init_params(subdomain_xs, subdomain_ws, (0,1))
    all_params = {"static":{"decomposition":ps_[0]}, "trainable":{"decomposition":ps_[1]}}

    decomposition.plot(all_params)
    plt.show()



    ## 2D test

    subdomain_xs = [np.linspace(-3,3,4), np.linspace(-2,2,3)]
    subdomain_ws = [3*np.ones(4), 2.2*np.ones(3)]

    decomposition = RectangularDecompositionND
    ps_ = decomposition.init_params(subdomain_xs, subdomain_ws, (0,1))
    all_params = {"static":{"decomposition":ps_[0]}, "trainable":{"decomposition":ps_[1]}}
    m = all_params["static"]["decomposition"]["m"]
    active = np.ones(m)

    active[1] = 0
    active[2] = 2
    decomposition.plot(all_params, active=active, show_norm=True, show_window=True)
    x_batch = np.array([[-3.6,-4.2],
                        [1,2],
                        [3,4]])
    for x in x_batch:
        plt.scatter(x[0], x[1])
    plt.show()
    print(decomposition.inside_models(all_params, x_batch, np.arange(m)))
    print(decomposition.inside_points(all_params, x_batch))

    # single subdomain test
    subdomain_xs = [np.array([0]), np.array([0])]
    subdomain_ws = [np.array([1]), np.array([2])]

    decomposition = RectangularDecompositionND
    ps_ = decomposition.init_params(subdomain_xs, subdomain_ws, (0,1))
    all_params = {"static":{"decomposition":ps_[0]}, "trainable":{"decomposition":ps_[1]}}

    decomposition.plot(all_params)
    plt.show()



    ## 3D test

    subdomain_xs = [np.linspace(-3,3,4), np.linspace(-2,2,3), np.linspace(-1,1,2)]
    subdomain_ws = [3*np.ones(4), 2.2*np.ones(3), 2.5*np.ones(2)]

    decomposition = RectangularDecompositionND
    ps_ = decomposition.init_params(subdomain_xs, subdomain_ws, (0,1))
    all_params = {"static":{"decomposition":ps_[0]}, "trainable":{"decomposition":ps_[1]}}
    m = all_params["static"]["decomposition"]["m"]
    active = np.ones(m)

    active[1] = 0
    active[2] = 2
    decomposition.plot(all_params, iaxes=[0,1], active=active, show_norm=True, show_window=True)
    plt.show()
    decomposition.plot(all_params, iaxes=[1,2], active=active, show_norm=True, show_window=True)
    plt.show()

    # large number of subdomains test
    subdomain_xs = [np.linspace(-3,3,20), np.linspace(-2,2,20), np.linspace(-1,1,20)]
    subdomain_ws = [3*np.ones(20), 2.2*np.ones(20), 2.5*np.ones(20)]

    decomposition = RectangularDecompositionND
    ps_ = decomposition.init_params(subdomain_xs, subdomain_ws, (0,1))
    all_params = {"static":{"decomposition":ps_[0]}, "trainable":{"decomposition":ps_[1]}}

    decomposition.plot(all_params)
    plt.show()

    ## multiscale tests

    subdomain_xss = [[np.linspace(-3,3,4), np.linspace(-2,2,3)],
                     [np.linspace(-3,3,10), np.linspace(-2,2,10)],
                     ]
    subdomain_wss = [[3*np.ones(4), 2.2*np.ones(3)],
                     [1*np.ones(10), 1*np.ones(10)],
                     ]

    decomposition = MultilevelRectangularDecompositionND
    ps_ = decomposition.init_params(subdomain_xss, subdomain_wss, (0,1))
    all_params = {"static":{"decomposition":ps_[0]}, "trainable":{"decomposition":ps_[1]}}
    m = all_params["static"]["decomposition"]["m"]
    active = np.ones(m)

    decomposition.plot(all_params, active=active, show_norm=True, show_window=True)