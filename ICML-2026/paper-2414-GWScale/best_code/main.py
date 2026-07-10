import logging
import time
import torch

logging.basicConfig(encoding='utf-8', level=logging.INFO)

import solvers
from utils.data.meshes.load import load_pointcloud_from_mesh
from utils.data.images.load import load_pointcloud_from_image

from utils.math.costs import euclidean, gaussian
from utils.math.functions import noise

from utils.gradients.gradient_cntgw import gradient_cntgw
from utils.gradients.gradient_quadraticgw import gradient_quadraticgw
from utils.gradients.gradient_kernelgw import gradient_kernelgw

from utils.viz.data import plot_pointcloud
from utils.viz.transports import plot_transfer
from utils.viz.misc import plot_gradients, plot_flow

from applications.barycenters import egw_barycenter
from applications.gradient_flows import gradient_flow

if __name__ == "__main__":
    ### 1 - Solvers comparison ###

    ARGS = {'eps': 1e-3,
        'numItermax': 100,
        'stop_criterion': 'energy',
        'stopThr': 1e-5,
        'SINK_ARGS': {'numItermax': 50, 'symmetrize': True}}

    N, M = 5000, 5000
    X = load_pointcloud_from_mesh('data/00049424_ferrari.ply', N=N)
    Y = load_pointcloud_from_mesh('data/muybridge_014_01.ply', N=N)
    
    cost = lambda u, v: euclidean(u, v, p=1)
        
    Cx = cost(X[:, None], X[None, :])
    Cy = cost(Y[:, None], Y[None, :])
    exact_solvers = [solvers.EntropicGW(Cx=Cx, Cy=Cy, **ARGS),
                     solvers.KernelGW(Cx=Cx, Cy=Cy, **ARGS)]

    approx_solvers = [solvers.CntGW(X, Y, cost, approx_dims=20, **ARGS),
                      solvers.LowRankGW(X, Y, cost, approx_dims=20, **ARGS),
                      solvers.MultiscaleCntGW(X, Y, cost, approx_dims=20, ratio=0.1, **ARGS), ]

    for solver in exact_solvers:
        start_time = time.time()
        solver.to('cuda')
        solver.solve(verbose=False)
        print(f"Solver: {solver.__class__.__name__} \t Time: {time.time() - start_time}")
        P = solver.transport_plan(lazy=False).cpu()
        solver.to('cpu')
        solver = None
        plot_transfer(X, Y, P, lazy=False)


    for solver in approx_solvers:
        start_time = time.time()
        solver.solve(verbose=False)
        print(f"Solver: {solver.__class__.__name__} \t Time: {time.time() - start_time}")
        P = solver.transport_plan(lazy=True)
        plot_transfer(X, Y, P, lazy=True)


    ### 2 - Gradients ###
    """N, M = 1500, 1500
    X = load_pointcloud_from_image('data/Sbis.png', N=N)
    Y = load_pointcloud_from_image('data/Cter.png', N=M)

    cost = lambda u, v: euclidean(u, v, p=1)

    ARGS = {'eps': 1e-3,
            'stop_criterion': 'energy',
            'stopThr': 5e-5,
            'sink_adaptive_schedule': True,
            'numItermax': 50,
            'sink_max_numItermax': 400,
            'SINK_ARGS': {'symmetrize': True, 'numItermax': 25, 'stopThr': 1e-6}
            }

    grad = gradient_cntgw(X, Y, cost, which='x', approx_dims=20, solver_kwargs=ARGS)
    grad_auto = gradient_cntgw(X, X, cost, which='x', approx_dims=20, solver_kwargs=ARGS)
    grad_unbiased = grad - grad_auto

    plot_gradients(X, grad_unbiased, scale_units='x', scale=0.03)"""

    ### 3 - Barycenters ###
    """N, M = 1500, 1500
    X = load_pointcloud_from_image('data/Sbis.png', N=N)
    Y = load_pointcloud_from_image('data/Cter.png', N=M)

    cost = lambda u, v: euclidean(u, v, p=1)

    ARGS = {'eps': 1e-3,
            'stop_criterion': 'energy',
            'stopThr': 5e-5,
            'sink_adaptive_schedule': True,
            'numItermax': 50,
            'sink_max_numItermax':400,
            'SINK_ARGS': {'symmetrize': True, 'numItermax': 25, 'stopThr': 1e-6}
            }

    bar = egw_barycenter(X, Y, cost, weight=0.5, approx_dims=30, iters=30, lbda=50, momentum=0.25, solver_kwargs=ARGS)
    plot_pointcloud(bar)"""

    ### 4 - Flow ###
    """N, M = 1500, 1500
    X = load_pointcloud_from_image('data/Sbis.png', N=N)
    Y = load_pointcloud_from_image('data/Cter.png', N=M)

    cost = lambda u, v: euclidean(u, v, p=1)

    ARGS = {'eps': 1e-3,
            'stop_criterion': 'energy',
            'stopThr': 5e-5,
            'sink_adaptive_schedule': True,
            'numItermax': 50,
            'sink_max_numItermax':400,
            'SINK_ARGS': {'symmetrize': True, 'numItermax': 25, 'stopThr': 1e-6}
            }

    X_t_list = gradient_flow(X, Y, cost, approx_dims=50, iters=200, lbda=15, momentum=0.9, solver_kwargs=ARGS,
                             record_step=10, plot=False)
    plot_flow(X_t_list, X)"""
