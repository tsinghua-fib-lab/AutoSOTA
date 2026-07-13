# This file is licensed under the MIT License.
# See the LICENSE file in the project root for full license information.
#

import numpy as np
import numpy.typing as npt
import cvxpy as cp
from .utils import policy_iteration, compute_stationary_distribution, soft_policy_iteration, get_cvxpy_solvers, solve_with_fallback
from typing import Tuple, Optional
from .mdp_description import MDPDescription
from enum import Enum

class BoundType(Enum):
    BOUND_1 = 1
    """ New bound """
    BOUND_2 = 2
    """ Upper bound of BOUND_1 """


golden_ratio_sq =  ((1+ np.sqrt(5)) / 2) ** 2

class NewMDPDescription(MDPDescription):
    """Similar to the MDPDescription class, but used to compute the new
    upper bound and allocation Vectors.
    """

    V_greedy_k: npt.NDArray[np.float64]
    """ Value of the optimal policy for different values of k """
    Mk_V_greedy: npt.NDArray[np.float64]
    """ 2k-th Moment of the optimal policy for different values of k """
    
    
    def __init__(self, P: npt.NDArray[np.float64], R: npt.NDArray[np.float64], discount_factor: float, max_moment_order_k: int = 20, abs_tol: float = 1e-6):
        """Initialize the class

        Parameters
        ----------
        P : npt.NDArray[np.float64]
            Transition function, of shape |S|x|A|x|S| (state, action, next state)
        R : npt.NDArray[np.float64]
            Rewards, of shape |S|x|A|x|S| (state, action, next state)
        discount_factor : float
            discount factor in (0,1)
        max_moment_order_k : int
            max k value
        abs_tol : float, optional
            absolute tolerance for policy iteration, by default 1e-6
        """        
        super().__init__(P, R, discount_factor, abs_tol)
        self.max_moment_order_k = max_moment_order_k
        self.values_k = list(range(1, max_moment_order_k))
        self.num_values_k = len(self.values_k)
        
        self.V_greedy_k = np.zeros((self.num_values_k, P.shape[0], P.shape[0], P.shape[1]))
        self.Mk_V_greedy = np.zeros((self.num_values_k, P.shape[0], P.shape[1]))
        self.Mk_V_greedy_temp =np.zeros((self.num_values_k, P.shape[0], P.shape[1]))
        
        for id_k, k in enumerate(self.values_k):
            # Compute quantities of itnerest
            self.V_greedy_k[id_k] = (self.V_greedy[:, np.newaxis, np.newaxis] - self.avg_V_greedy[np.newaxis,:,:]) ** (2 * k)
            self.Mk_V_greedy_temp [id_k] = (
                P.reshape(self.dim_state * self.dim_action, -1) 
                * (self.V_greedy_k[id_k].reshape(self.dim_state, self.dim_state*self.dim_action).T)
            ).sum(-1).reshape(self.dim_state, self.dim_action) ** (2 ** (-k))
        self.Mk_V_greedy = self.Mk_V_greedy_temp.max(0) ** 2
        
    
    def evaluate_allocation(self, omega: npt.NDArray[np.float64], type: BoundType = BoundType.BOUND_1, navigation_constraints: bool = False) -> float:
        """Evaluate a given allocation

        Parameters
        ----------
        omega : npt.NDArray[np.float64]
            Allocation (of size |S|x|A|) to evaluate
        type : BoundType, optional
            Type of bound to use, by default BoundType.BOUND_1
        navigation_constraints : bool, optional
            If true, evaluates if the allocation satisfies the allocation constraints, by default False

        Returns
        -------
        float
            The value of the allocation

        Raises
        ------
        Exception
            If the bound type is not correct
        """        
        if navigation_constraints is True:
            checks = np.array([np.sum(omega[s]) - np.sum(np.multiply(self.P[:,:,s], omega)) for s in range(self.dim_state)])
            assert np.all(np.isclose(checks, 0)), "Allocation does not satisfy navigation constraints"
        ns, na = self.dim_state, self.dim_action
        
        if type.value == BoundType.BOUND_1.value:
            # Evaluate bound 1
            # max_{s,a\neq pi*(s)} [ ... + max_{s'} ....]
            obj_supp = []
            for s in range(ns):
                for a in range(na):
                    if a == self.pi_greedy[s]: continue
                    T1 = self.normalizer * (2 + 8 * golden_ratio_sq * self.Mk_V_greedy[s,a]) / (omega[s,a] * self.delta_sq[s,a])
                    
                    obj_supp_s = []
                    # Evaluate max_{s'}
                    for sp in range(ns):
                        C = 4 * max(1, 4 * golden_ratio_sq * self.Mk_V_greedy[sp, self.pi_greedy[sp]]) * ((1+self.discount_factor) ** 2)
                        T2 = self.normalizer * C/ (omega[sp, self.pi_greedy[sp]] * self.delta_sq[s,a] * ((1 - self.discount_factor) ** 2))
                        obj_supp_s.append(T2)
                    
                    obj_supp.append(T1 + np.max(obj_supp_s))

            objective = np.max(obj_supp)
            return objective / self.normalizer
        
        elif type.value == BoundType.BOUND_2.value:
            # Evaluate an upper bound of bound_1.
            # max_{s,a\neq pi*(s)} [...] + max_{s}[...]
            objective = 0
            
            Hstar = np.max([
                self.normalizer *  (2 + 8 * golden_ratio_sq * self.Mk_V_greedy[s, self.pi_greedy[s]]) / (self.delta_sq_min * ((1 - self.discount_factor) ** 2)) for s in range(ns)])
            
            obj_supp_1 = []
            obj_supp_2 = []
            for s in range(ns):
                obj_supp_2.append(Hstar / omega[s, self.pi_greedy[s]] )

                for a in range(na):
                    if a == self.pi_greedy[s]: continue
                    T1 = self.normalizer * (2 + 8 * golden_ratio_sq * self.Mk_V_greedy[s,a]) / (omega[s,a] * (self.delta_sq[s,a]))
                    obj_supp_1.append(T1)
                    

            return (np.max(obj_supp_1) + np.max(obj_supp_2)) / self.normalizer
        else:
            raise Exception(f'Type {type} not found')
        
    def compute_allocation(
            self,
            type: BoundType = BoundType.BOUND_1,
            navigation_constraints: bool = False
            ) -> Tuple[npt.NDArray[np.float64], float]:
        """Compute allocation

        Parameters
        ----------
        type : BoundType, optional
            Type of bound to use, by default BoundType.BOUND_1
        navigation_constraints : bool, optional
            If true, enables navigation constraints, by default False

        Returns
        -------
        Tuple[npt.NDArray[np.float64], float]
            A tuple that consists of 2 elements
                1. Allocation vector, of size |S|x|A|
                2. The value of this allocation vector

        Raises
        ------
        Exception
            Exception if the bound type is not found, or the computation is not feasible
        """        
        
        # Set up problem
        Mk = self.Mk_V_greedy
        Delta_sq = self.delta_sq
        Delta_sq_min = self.delta_sq_min
        pi_greedy = self.pi_greedy            
            
        
        ns, na = self.dim_state, self.dim_action
        omega = cp.Variable((ns, na), nonneg=True)
        sigma = cp.Variable(1, nonneg=True)
        
        # Constraints definition
        constraints = [cp.sum(omega) == 1, omega >= sigma, sigma >= 1e-15]
        if navigation_constraints:
            constraints.extend([cp.sum(omega[s]) == cp.sum(cp.multiply(self.P[:,:,s], omega)) for s in range(self.dim_state)])

        tol = 0#1e-14
        def bound_type_1():
            obj_supp = []
            for s in range(ns):
                for a in range(na):
                    if a == self.pi_greedy[s]: continue
                    T1 = self.normalizer * cp.inv_pos(omega[s,a]) * (2 + 8 * golden_ratio_sq * Mk[s,a]) / (Delta_sq[s,a]+tol)
                    
                    obj_supp_s = []
                    for sp in range(ns):
                        C = 4 * max(1, 4 * golden_ratio_sq * self.Mk_V_greedy[sp, pi_greedy[sp]]) * ((1+self.discount_factor) ** 2)
                        T2 = self.normalizer * cp.inv_pos(omega[sp, pi_greedy[sp]]) * C / (tol + Delta_sq[s,a] * ((1 - self.discount_factor) ** 2))

                        obj_supp_s.append(T2)
                    
                    obj_supp.append(T1 + cp.max(cp.hstack(obj_supp_s)))

            objective = cp.max(cp.hstack(obj_supp))
            return objective

        def bound_type_2():
            Hstar = np.max([(2 + 8 * golden_ratio_sq * Mk[s, pi_greedy[s]]) /  (tol+Delta_sq_min * ((1 - self.discount_factor) ** 2)) for s in range(ns)])
            
            obj_supp_1 = []
            obj_supp_2 = []
            for s in range(ns):
                obj_supp_2.append(self.normalizer * cp.inv_pos(omega[s, pi_greedy[s]]) * Hstar)
                
                for a in range(na):
                    if a == self.pi_greedy[s]: continue
                    T1 = self.normalizer * cp.inv_pos(omega[s,a] ) * (2 + 8 * golden_ratio_sq * Mk[s,a]) / (Delta_sq[s,a]+tol)
                    obj_supp_1.append(T1)

            return cp.max(cp.hstack(obj_supp_1)) + cp.max(cp.hstack(obj_supp_2))
        
        solvers = get_cvxpy_solvers()
        # Try 10 times to solve it. It's a generic value
        for _it in range(10):
            try:
                if type.value == BoundType.BOUND_1.value:
                    objective = cp.Minimize(bound_type_1())
                elif type.value == BoundType.BOUND_2.value:
                    objective = cp.Minimize(bound_type_2())
                else:
                    raise Exception(f'Type {type} not found')

                problem = cp.Problem(objective, constraints)
                result = solve_with_fallback(problem, solvers, verbose=False, warm_start=True)#, abstol=1e-10, reltol=1e-10, feastol=1e-10, max_iters=200)
                break
            except Exception as e:
                if _it == 9:
                    print('Cannot solve the UB!')
                    omega.value = np.ones((ns, na)) / (ns * na)
                    #raise Exception(f'Cannot solve the UB! {e}')
        return omega.value, self.evaluate_allocation(omega.value, type, navigation_constraints=False)

    @staticmethod
    def compute_mf_allocation(
            discount_factor: float,
            Q: npt.NDArray[np.float64],
            Mk: npt.NDArray[np.float64],
            num_visits: npt.NDArray[np.float64],
            type: BoundType = BoundType.BOUND_1,
            navigation_constraints: bool = False, 
            ) ->  npt.NDArray[np.float64]:
        """Compute allocation using an estimate of the Q-values and
        the M values

        Parameters
        ----------
        discount_factor : float
            discount factor
        Q : npt.NDArray[np.float64]
            Q-values of the greedy policy
        Mk : npt.NDArray[np.float64]
            M values of the greedy policy
        num_visits : npt.NDArray[np.float64]
            Number of visits of shape |S|x|A|x|S| (state, action, next_state)
        type : BoundType, optional
            type of bound to compute, by default BoundType.BOUND_1
        navigation_constraints : bool, optional
            if true, enables navigation constraints, by default False

        Returns
        -------
        npt.NDArray[np.float64]
            Allocation vector omega
        """      
        # Set up variables  
        ns, na = Q.shape
        pi_greedy = Q.argmax(1)
        Delta_sq = np.clip((Q.max(1)[:, np.newaxis] - Q) ** 2, a_min=1e-9, a_max=None)
        idxs_subopt_actions = np.array([[False if pi_greedy[s] == a else True for a in range(na)] for s in range(ns)])
        Delta_sq_subopt = Delta_sq[idxs_subopt_actions]
        Delta_sq_min =  Delta_sq_subopt.min()

        # Maximum likelihood of the transitions (with optimistic initialization)
        p_transitions = np.ones((ns, na, ns)) + num_visits
        P = p_transitions / p_transitions.sum(-1, keepdims=True)
        
        # Compute normalizer
        normalizer = (Delta_sq_min * (1 -  discount_factor) ** 3) / (ns * na)

        # Solve optimization problem using estimated values
        omega = cp.Variable((ns, na), nonneg=True)
        sigma = cp.Variable(1, nonneg=True)
        constraints = [cp.sum(omega) == 1, omega >= sigma, sigma >= 1e-15]
        
        if navigation_constraints:
            constraints.extend([cp.sum(omega[s]) == cp.sum(cp.multiply(P[:,:,s], omega)) for s in range(ns)])

        tol = 0#1e-14
        def bound_type_1():
            obj_supp = []
            for s in range(ns):
                for a in range(na):
                    if a == pi_greedy[s]: continue
                    T1 = normalizer * cp.inv_pos(omega[s,a]) * (2 + 8 * golden_ratio_sq * Mk[s,a]) / (Delta_sq[s,a]+tol)
                    
                    obj_supp_s = []
                    for sp in range(ns):
                        C = 4 * max(1, 4 * golden_ratio_sq * Mk[sp, pi_greedy[sp]]) * ((1+ discount_factor) ** 2)
                        T2 = normalizer * cp.inv_pos(omega[sp, pi_greedy[sp]]) * C / (tol + Delta_sq[s,a] * ((1 - discount_factor) ** 2))

                        obj_supp_s.append(T2)
                    
                    obj_supp.append(T1 + cp.max(cp.hstack(obj_supp_s)))

            objective = cp.max(cp.hstack(obj_supp))
            return objective

        def bound_type_2():
            Hstar = np.max([(2 + 8 * golden_ratio_sq * Mk[s, pi_greedy[s]]) /  (tol+Delta_sq_min * ((1 - discount_factor) ** 2)) for s in range(ns)])
            
            obj_supp_1 = []
            obj_supp_2 = []
            for s in range(ns):
                obj_supp_2.append(normalizer * cp.inv_pos(omega[s, pi_greedy[s]]) * Hstar)
                
                for a in range(na):
                    if a == pi_greedy[s]: continue
                    T1 = normalizer * cp.inv_pos(omega[s,a] ) * (2 + 8 * golden_ratio_sq * Mk[s,a]) / (Delta_sq[s,a]+tol)
                    obj_supp_1.append(T1)

            return cp.max(cp.hstack(obj_supp_1)) + cp.max(cp.hstack(obj_supp_2))
        
        
        solvers = get_cvxpy_solvers()
        for _it in range(10):
            try:
                if type.value == BoundType.BOUND_1.value.value:
                    objective = cp.Minimize(bound_type_1())
                elif type.value == BoundType.BOUND_2:
                    objective = cp.Minimize(bound_type_2())
                else:
                    raise Exception(f'Type {type} not found')

                problem = cp.Problem(objective, constraints)
                result = solve_with_fallback(problem, solvers, verbose=False, warm_start=True)#, abstol=1e-10, reltol=1e-10, feastol=1e-10, max_iters=200)
                break
            except Exception as e:
                if _it == 9:
                    raise Exception(f'Cannot solve the UB! {e}')
        return omega.value
