"""
Defines plotting functions for 3D FBPINN / PINN problems

This module is used by plot_trainer.py (and subsequently trainers.py)
"""

import matplotlib.pyplot as plt

from fbpinns.plot_trainer_1D import _plot_setup, _to_numpy
from fbpinns.plot_trainer_2D import _plot_test_im

@_to_numpy
def plot_3D_FBPINN(x_batch_test, u_exact, u_test, us_test, ws_test, us_raw_test, x_batch, all_params, i, active, decomposition, n_test):

    xlim, ulim = _plot_setup(x_batch_test, u_exact)
    xlim0 = x_batch_test.min(0), x_batch_test.max(0)

    nt = n_test[-1]//10# slice across last dimension
    shape = (1+nt+1, 3)# nrows, ncols
    f = plt.figure(figsize=(8,8*shape[0]/3))

    # plot domain + x_batch
    for iplot, (a,b) in enumerate([[0,1],[0,2],[1,2]]):
        plt.subplot2grid(shape,(0,iplot))
        plt.title(f"[{i}] Domain decomposition")
        plt.scatter(x_batch[:,a], x_batch[:,b], alpha=0.5, color="k", s=1)
        decomposition.plot(all_params, active=active, create_fig=False, iaxes=[a,b])
        plt.xlim(xlim[0][a], xlim[1][a])
        plt.ylim(xlim[0][b], xlim[1][b])
        plt.gca().set_aspect("equal")

    # plot full solutions
    ulim = [ulim[0][0], ulim[1][0]]
    for it in range(nt):
        plt.subplot2grid(shape,(1+it,0))
        plt.title(f"[{i}] Full solution")
        _plot_test_im(u_test[:,0:1], xlim0, ulim, n_test, it=it*10)

        plt.subplot2grid(shape,(1+it,1))
        plt.title(f"[{i}] Ground truth")
        _plot_test_im(u_exact[:,0:1], xlim0, ulim, n_test, it=it*10)

        plt.subplot2grid(shape,(1+it,2))
        plt.title(f"[{i}] Difference")
        _plot_test_im(u_exact[:,0:1] - u_test[:,0:1], xlim0, ulim, n_test, it=it*10)

    # plot raw hist
    plt.subplot2grid(shape,(1+nt,0))
    plt.title(f"[{i}] Raw solutions")
    plt.hist(us_raw_test.flatten(), bins=100, label=f"{us_raw_test.min():.1f}, {us_raw_test.max():.1f}")
    plt.legend(loc=1)
    plt.xlim(-5,5)

    plt.tight_layout()

    return (("test",f),)

@_to_numpy
def plot_3D_PINN(x_batch_test, u_exact, u_test, u_raw_test, x_batch, all_params, i, n_test):
    
    xlim, ulim = _plot_setup(x_batch_test, u_exact)
    xlim0 = x_batch.min(0), x_batch.max(0)

    nt = n_test[-1]//10# slice across last dimension
    shape = (1+nt+1, 3)# nrows, ncols
    f = plt.figure(figsize=(8,8*shape[0]/3))

    # plot x_batch
    for iplot, (a,b) in enumerate([[0,1],[0,2],[1,2]]):
        plt.subplot2grid(shape,(0,iplot))
        plt.title(f"[{i}] Training points")
        plt.scatter(x_batch[:,a], x_batch[:,b], alpha=0.5, color="k", s=1)
        plt.xlim(xlim[0][a], xlim[1][a])
        plt.ylim(xlim[0][b], xlim[1][b])
        plt.gca().set_aspect("equal")

    # plot full solution
    ulim = [ulim[0][0], ulim[1][0]]
    for it in range(nt):
        plt.subplot2grid(shape,(1+it,0))
        plt.title(f"[{i}] Full solution")
        _plot_test_im(u_test[:,0:1], xlim0, ulim, n_test, it=it*10)

        plt.subplot2grid(shape,(1+it,1))
        plt.title(f"[{i}] Ground truth")
        _plot_test_im(u_exact[:,0:1], xlim0, ulim, n_test, it=it*10)

        plt.subplot2grid(shape,(1+it,2))
        plt.title(f"[{i}] Difference")
        _plot_test_im(u_exact[:,0:1] - u_test[:,0:1], xlim0, ulim, n_test, it=it*10)

    # plot raw hist
    plt.subplot2grid(shape,(1+nt,0))
    plt.title(f"[{i}] Raw solution")
    plt.hist(u_raw_test.flatten(), bins=100, label=f"{u_raw_test.min():.1f}, {u_raw_test.max():.1f}")
    plt.legend(loc=1)
    plt.xlim(-5,5)

    plt.tight_layout()

    return (("test",f),)