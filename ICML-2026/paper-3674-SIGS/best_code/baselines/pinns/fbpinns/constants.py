"""
Defines the Constants object which defines and stores a problem setup and all of its hyperparameters
for both FBPINNs and PINNs

This constants object should be passed to the appropriate trainer class defined in trainers.py

This module is used by trainers.py
"""

import socket

import numpy as np
import optax

from fbpinns import domains, problems, decompositions, networks, schedulers
from fbpinns.constants_base import ConstantsBase


# helper functions

def get_subdomain_ws(subdomain_xs, width):
    return [width*np.min(np.diff(x))*np.ones_like(x) for x in subdomain_xs]


# main constants class

class Constants(ConstantsBase):

    def __init__(self, **kwargs):
        "Defines global constants for model"

        # Define run
        self.run = "test"

        # Define domain
        self.domain = domains.RectangularDomainND
        self.domain_init_kwargs = dict(
            xmin=np.array([0.]),
            xmax=np.array([1.])
            )

        # Define problem
        self.problem = problems.HarmonicOscillator1D
        #self.problem = problems.HarmonicOscillator1DInverse
        self.problem_init_kwargs = dict(
            d=2,
            w0=20,
            )

        # Define domain decomposition
        subdomain_xs = [np.linspace(0,1,5)]
        subdomain_ws = get_subdomain_ws(subdomain_xs, 2.99)
        self.decomposition = decompositions.RectangularDecompositionND
        self.decomposition_init_kwargs = dict(
            subdomain_xs=subdomain_xs,
            subdomain_ws=subdomain_ws,
            unnorm=(0., 1.),
            )

        # Define neural network
        self.network = networks.FCN
        self.network_init_kwargs = dict(
            layer_sizes=[1, 32, 1],
            )

        # Define scheduler
        self.n_steps = 15000
        self.scheduler = schedulers.AllActiveSchedulerND
        self.scheduler_kwargs = dict()
        #self.scheduler = schedulers.PointSchedulerRectangularND
        #self.scheduler_kwargs = dict(
        #    point=np.array([0.]),
        #    )

        # Define optimisation parameters
        self.ns = ((60,),)# batch_shape for each training constraint
        self.n_test = (200,)# batch_shape for test data
        self.sampler = "grid"# one of ["grid", "uniform", "sobol", "halton"]
        self.optimiser = optax.adam
        self.optimiser_kwargs = dict(
            learning_rate=1e-3
            )
        self.seed = 0

        # Define summary output parameters
        self.summary_freq    = 1000# outputs train stats to command line
        self.test_freq       = 1000# outputs test stats to plot / file / command line
        self.model_save_freq = 10000
        self.show_figures = True# whether to show figures
        self.save_figures = False# whether to save figures
        self.clear_output = False# whether to clear ipython output periodically

        # other constants
        self.hostname = socket.gethostname().lower()

        # overwrite with input arguments
        for key in kwargs.keys(): self[key] = kwargs[key]# invokes __setitem__ in ConstantsBase



if __name__ == "__main__":

    c = Constants(seed=2)
    print(c)

    c.get_outdirs()
    c.save_constants_file()