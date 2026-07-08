"""Optimisation code"""

from typing import Callable
import cvxpy as cp

from bayesian_dro.Bayesian_DRO_continuous import LARGEST_X, SMALLEST_X


def get_kl_bdro_problem(
    decision_objective: Callable[[cp.Variable, cp.Parameter], cp.Expression],
    num_posterior_samples: int,
    num_likelihood_samples: int,
    dim: int = 1,
    is_portfolio: bool = False,
) -> cp.Problem:
    """Bayesian DRO as a cvxpy optimisaton problem.

    Args:
        decision_objective: A callable objective function implemented using cvxpy.
            The first argument should be a cvxpy variable.
            The second argument should be a cvxpy parameter representing the data samples.
            The return should be a cvxpy expression.
        num_posterior_samples: Number of posterior samples.
        num_likelihood_samples: Number of likelihood samples for each posterior sample.
        dim: Dimension of random variable xi.

    Returns:
        problem: A cvxpy Problem object

    Notes:
        We use an epigraph variable t to upper bound the objective function G(x, xi).
        That is, we add constraints G(x, xi[i]) <= t[i] for all i = 1,...,num_posterior_samples.

        The main optimisation trick is then to use the perspective of the log-sum-exp (LSE) function.
        Specifically, the perspective is l * LSE(t[i] / l), where l is the Lagrangian variable.
        As l -> 0, then l * LSE(t[i] / l) tends to max(t[i]).
    """
    # declare variables
    x = cp.Variable(dim, name="x")
    lam = [
        cp.Variable(1, name=f"lam_{i}", nonneg=True)
        for i in range(num_posterior_samples)
    ]
    t = cp.Variable((num_posterior_samples, num_likelihood_samples), name="t")

    # declare parameters
    epsilon_minus_constant = cp.Parameter(1, name="epsilon_minus_constant", nonneg=True)
    # TODO write up why we use list?
    xi = [
        cp.Parameter((num_likelihood_samples, dim), name=f"xi_{i}")
        for i in range(num_posterior_samples)
    ]

    # create the objective function for the Bayesian DRO problem
    # NOTE we pass the max function to f_recession because,
    # as lam -> 0, then lam * LSE(t[i] / lam) tends to max(t[i]).
    bdro_obj = cp.Minimize(
        (1.0 / num_posterior_samples)
        * cp.sum(
            [
                lam[i] @ (epsilon_minus_constant)
                + lam[i] * cp.log(1.0 / num_likelihood_samples)
                + cp.perspective(cp.log_sum_exp(t[i]), lam[i], f_recession=cp.max(t[i]))
                for i in range(num_posterior_samples)
            ]
        )
    )
    # add the decision objective as an epigraph constraint
    # examples of decision objectives are the newsvendor objective
    constraints = [
        x >= SMALLEST_X,
        # x <= LARGEST_X,   # NOTE this can cause some unexpected behaviour for large epsilon in newsvendor problem
    ] + [decision_objective(x, xi[i]) <= t[i] for i in range(num_posterior_samples)]

    # TODO this is a temporary fix for portfolio : this whole function should
    # really be a class that one can inherit from and add custom constraints
    if is_portfolio:
        constraints.append(cp.sum(x) == 1)

    return cp.Problem(bdro_obj, constraints)


class DRO_BAS_MMD():
    '''
    DRO-BAS problem with the MMD as a KDRO problem in CVXPY 
    '''
    def __init__(self, dim_decision, dim_data, loss_call): 
        '''
        dim_decision: the dimension of the decision variable (parameter to be optimized)
        dim_data: dimension of the data

        loss_call: A callable objective function implemented using cvxpy.
            The first argument should be a cvxpy variable.
            The second argument should be a cvxpy parameter representing the certifying points.
            The return should be a cvxpy expression.
        '''
        assert dim_decision > 0 

        self.dim_decision = dim_decision
        self.dim_data = dim_data
        self.loss_call = loss_call

    def get_portfolio_problem(self, n_sample, num_certify_samples):
        '''
        Get the optimisation problem in CVXPY for the portfolio problem
        
        Args:
        n_sample: Number of total samples.
        num_certify_samples: Number of certifying points for the discretisation of the constraint.

        Returns:
        problem: A cvxpy Problem object
        
        '''
        n_certify = num_certify_samples
        K = cp.Parameter((n_sample+n_certify, n_sample+n_certify), name="K")
        K_decomposed = cp.Parameter((n_sample+n_certify, n_sample+n_certify), name="K_decomposed")
        epsilon = cp.Parameter(1, name="epsilon", nonneg=True)
        Xobs = cp.Parameter((n_sample,self.dim_data), name="Xobs")
        Xcert = cp.Parameter((n_certify,self.dim_data), name="Xcert")
        
        # theta is the decision variable
        theta = cp.Variable(self.dim_decision, name="theta")

        # f0 = a bias term as part of the RKHS function. A scalar
        f0 = cp.Variable()

        # Beta is the vector of coefficients of the dual RKHS function.
        beta = cp.Variable(n_sample+n_certify)

        # function values at the kernel_points
        fvals = K @ beta

        # List of constraints for cvxpy
        constraints = []
        loss_call = self.loss_call
        # always certify the observations
        for i in range(n_sample):
            #FIXME when we merge with multivariate the Xobs.shape[1] should just be dim
            constraints += [loss_call(theta, Xobs[i,:].reshape((1, Xobs.shape[1]))) 
            <= f0 + fvals[i] ]

        # certify the certifying points
        for i in range(n_certify):
            #FIXME when we merge with multivariate the Xcert.shape[1] should just be dim
            xcert_i = Xcert[i,:].reshape((1, Xcert.shape[1]))
            constraints += [loss_call(theta, xcert_i) <= f0 +
            fvals[i+n_sample]]
        constraints += [theta >= 0, cp.sum(theta) == 1]
        
        emp = f0 + cp.sum(fvals[:n_sample]) / n_sample
        rkhs_norm = cp.norm(beta.T @ K_decomposed) # pass decomposed kernel directly
        reg_term = epsilon * rkhs_norm

        # objective function
        obj = emp + reg_term
        opt = cp.Problem(cp.Minimize(obj), constraints)
        
        return opt

    def get_newsvendor_problem(self, n_sample, num_certify_samples):
        '''
        Get the optimisation problem in CVXPY for the newsvendor problem
        
        Args:
        n_sample: Number of total samples.
        num_certify_samples: Number of certifying points for the discretisation of the constraint.

        Returns:
        problem: A cvxpy Problem object
        
        '''
        n_certify = num_certify_samples
        K = cp.Parameter((n_sample+n_certify, n_sample+n_certify), name="K")
        K_decomposed = cp.Parameter((n_sample+n_certify, n_sample+n_certify), name="K_decomposed")
        epsilon = cp.Parameter(1, name="epsilon", nonneg=True)
        Xobs = cp.Parameter((n_sample,self.dim_data), name="Xobs")
        Xcert = cp.Parameter((n_certify,self.dim_data), name="Xcert")
        
        # theta is the decision variable
        theta = cp.Variable(self.dim_decision, name="theta")

        # f0 = a bias term as part of the RKHS function. A scalar
        f0 = cp.Variable()

        # Beta is the vector of coefficients of the dual RKHS function.
        beta = cp.Variable(n_sample+n_certify)

        # function values at the kernel_points
        fvals = K @ beta

        # List of constraints for cvxpy
        constraints = []
        loss_call = self.loss_call
        # always certify the observations
        for i in range(n_sample):
            #FIXME when we merge with multivariate the Xobs.shape[1] should just be dim
            constraints += [loss_call(theta, Xobs[i,:].reshape((1, Xobs.shape[1]))) 
            <= f0 + fvals[i] ]

        # certify the certifying points
        for i in range(n_certify):
            #FIXME when we merge with multivariate the Xcert.shape[1] should just be dim
            xcert_i = Xcert[i,:].reshape((1, Xcert.shape[1]))
            constraints += [loss_call(theta, xcert_i) <= f0 +
            fvals[i+n_sample]]
        constraints += [theta >= SMALLEST_X]  #, theta <= LARGEST_X
        
        emp = f0 + cp.sum(fvals[:n_sample]) / n_sample
        rkhs_norm = cp.norm(beta.T @ K_decomposed) # pass decomposed kernel directly
        reg_term = epsilon * rkhs_norm

        # objective function
        obj = emp + reg_term
        opt = cp.Problem(cp.Minimize(obj), constraints)
        
        return opt
