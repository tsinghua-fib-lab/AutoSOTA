
__author__ = """N. Cullen <ncullen.th@dartmouth.edu>"""

from copy import copy
import numpy as np

from pyBN.classes.factor import Factor
from pyBN.classes.factorization import Factorization



def ve_map(bn,
            evidence={},
            target=None,
            prob=False):
    """
    Perform Max-Sum Variable Elimination over a BayesNet object
    for exact maximum a posteriori inference.

    This has been validated w/ and w/out evidence
    
    """
    _phi = Factorization(bn)

    order = copy(list(bn.nodes()))
    #### EVIDENCE PROCESSING ####
    for E, e in evidence.items():
        _phi -= (E,e)
        order.remove(E)

    #### MAX-PRODUCT ELIMINATE VAR ####
    for var in order:
        _phi //= var 
    
    #### TRACEBACK MAP ASSIGNMENT ####
    max_assignment = _phi.traceback_map()

    #### RETURN ####
    if prob:
        # multiply phi's together if there is evidence
        final_phi = _phi.consolidate()
        max_prob = round(final_phi.cpt[0],5)

        if target is not None:
            return max_prob, max_assignment[target]
        else:
            return max_prob, max_assignment
    else:
        if target is not None:
            return max_assignment[target]
        else:
            return max_assignment