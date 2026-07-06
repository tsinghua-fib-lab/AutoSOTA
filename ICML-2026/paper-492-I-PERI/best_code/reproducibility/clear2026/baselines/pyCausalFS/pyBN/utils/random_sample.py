"""
******************
Random Sample Code
******************

Generate a random sample dataset from a known Bayesian Network,
with or without evidence.

"""

__author__ = """Nicholas Cullen <ncullen.th@dartmouth.edu>"""

from pyBN.classes.factor import Factor
import numpy as np
from copy import deepcopy

def random_sample(bn, n=1000):
    """
    Take a random sample of "n" observations from a
    BayesNet object. This is essentially just the
    forward sample algorithm that returns the samples.

    Parameters
    ----------
    *bn* : a BayesNet object from which to sample

    *n* : an integer
        The number of observations to take

    *evidence* : a dictionary, key=rv & value=instantiation
        Evidence to pass in

    Returns
    -------
    *sample_dict* : a list of samples, where each sample
        is a list of values in bn.nodes() (topsort) order

    Notes
    -----

    """
    sample = np.empty((n,bn.num_nodes()),dtype=np.int)

    rv_map = dict([(rv,idx) for idx,rv in enumerate(bn.nodes())])
    factor_map = dict([(rv,Factor(bn,rv)) for rv in bn.nodes()])
    
    for i in range(n):
        for rv in bn.nodes():
            f = deepcopy(factor_map[rv])
            # reduce_factor by parent samples
            for p in bn.parents(rv):
                f.reduce_factor(p,bn.values(p)[sample[i][rv_map[p]]])
            choice_vals = bn.values(rv)
            choice_probs = f.cpt
            chosen_val = np.random.choice(choice_vals, p=choice_probs)
            sample[i][rv_map[rv]] = bn.values(rv).index(chosen_val)

    return sample





