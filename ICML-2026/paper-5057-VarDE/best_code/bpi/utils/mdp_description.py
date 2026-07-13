# This file is licensed under the MIT License.
# See the LICENSE file in the project root for full license information.
#

import numpy as np
import numpy.typing as npt
import cvxpy as cp
from .utils import policy_iteration, project_omega, get_cvxpy_solvers, solve_with_fallback
from typing import Tuple

class MDPDescription(object):
    """Class used to compute the sample complexity of an MDP using 
        the bound from Aymen et al. 2021.
        It can be used to store useful information, such as
        the variance of an MDP, the span, etc...
    """  

    P: npt.NDArray[np.float64]
    """ Transition function """
    R: npt.NDArray[np.float64]
    """ Reward """ 
    discount_factor: float
    """ Discount factor """
    abs_tol: float
    """ Absolute value error, used to stop the policy iteration procedure """
    
    V_greedy: npt.NDArray[np.float64]
    """ Value of the optimal policy """
    pi_greedy: npt.NDArray[np.float64]
    """ Optimal policy """
    Q_greedy: npt.NDArray[np.float64]
    """ Q-values of the optimal policy """
    
    idxs_subopt_actions: npt.NDArray[np.bool_]
    """ Indexes of the suboptimal actions """
    delta_sq: npt.NDArray[np.float64]
    """ Gaps squared"""
    delta_sq_subopt: npt.NDArray[np.float64]
    """ Gaps squared of the suboptimal actions"""
    delta_sq_min: float
    """ Minimum gap """
    
    avg_V_greedy: npt.NDArray[np.float64]
    """ Average value of the optimal policy in the next state """
    var_V_greedy: npt.NDArray[np.float64]
    """ Variance value of the optimal policy in the next state """
    var_V_greedy_max: float
    """ Maximum variance """
    span_V_greedy: npt.NDArray[np.float64]
    """ Span of the optimal policy in the next state """
    span_V_greedy_max: float
    """ Maximum span """
    
    T1: npt.NDArray[np.float64]
    """ T1 term from Aymen et al. 2021 """
    T2: npt.NDArray[np.float64]
    """ T2 term from Aymen et al. 2021 """
    T3: npt.NDArray[np.float64]
    """ T3 term from Aymen et al. 2021 """
    T4: npt.NDArray[np.float64]
    """ T4 term from Aymen et al. 2021 """
    H: npt.NDArray[np.float64]
    """ Hsa term from Aymen et al. 2021 """
    Hstar: npt.NDArray[np.float64]
    """ H* term from Aymen et al. 2021 """ 
    
    normalizer: float
    """ Normalizing costant, used to simplify computations.(delta_min*(1-gamma)^3)/(|S|*|A|)"""
    
    
    def __init__(self, P: npt.NDArray[np.float64], R: npt.NDArray[np.float64], discount_factor: float, abs_tol: float = 1e-6):
        """Initialize the MDP and compute quantities of interest

        Parameters
        ----------
        P : npt.NDArray[np.float64]
            Transition function, of shape |S|x|A|x|S|
        R : npt.NDArray[np.float64]
            Reward function, of shape |S|x|A|x|S|. Values should be in [0,1]
        discount_factor : float
            Discount factor, in (0, 1)
        abs_tol : float, optional
            Absolute tolerance for policy iteration, by default 1e-6
        """        
        self.P = P
        self.R = R
        self.discount_factor = discount_factor
        self.abs_tol = abs_tol

        Rmax, Rmin = np.max(R), np.min(R)
        
        # Normalize rewards
        if Rmax > 1 or Rmin < 0:
            self.R = (self.R - Rmin) / (Rmax - Rmin)

        # Policy iteration
        V, pi, Q = policy_iteration(self.discount_factor, self.P, self.R, atol=self.abs_tol)
        self.V_greedy = np.array(V)
        self.pi_greedy = pi
        self.Q_greedy = np.array(Q)
        self.idxs_subopt_actions = np.array([
            [False if self.pi_greedy[s] == a else True for a in range(self.dim_action)] for s in range(self.dim_state)])

        # Compute Delta
        self.delta_sq = np.clip((self.V_greedy[:, np.newaxis] - self.Q_greedy) ** 2, a_min=1e-32, a_max=None)
        self.delta_sq_subopt = self.delta_sq[self.idxs_subopt_actions]
        self.delta_sq_min =  self.delta_sq_subopt.min()
        
        # Compute variance of V, VarMax and Span
        self.avg_V_greedy = self.P @ self.V_greedy
        self.var_V_greedy =  self.P @ (V ** 2) - (self.avg_V_greedy) ** 2
        self.var_V_greedy_max = np.max(self.var_V_greedy[~self.idxs_subopt_actions])
        
        self.span_V_greedy = np.maximum(np.max(self.V_greedy) - self.avg_V_greedy, self.avg_V_greedy- np.min(self.V_greedy))
        self.span_V_greedy_max = np.max(self.span_V_greedy[~self.idxs_subopt_actions])

        # Compute T terms
        self.T1 = np.zeros((self.dim_state, self.dim_action))
        T2_1 = np.zeros_like(self.T1)
        T2_2 = np.zeros_like(self.T1)
        self.T1[self.idxs_subopt_actions] = np.nan_to_num(2 / self.delta_sq_subopt, nan=0, posinf=0, neginf=0)
        T2_1[self.idxs_subopt_actions] = np.nan_to_num(16 * self.var_V_greedy[self.idxs_subopt_actions] / self.delta_sq_subopt, nan=0, posinf=0, neginf=0)
        T2_2[self.idxs_subopt_actions] = np.nan_to_num(6 * self.span_V_greedy[self.idxs_subopt_actions] ** (4/3) / self.delta_sq_subopt ** 2/3, nan=0, posinf=0, neginf=0)
        self.T2 = np.maximum(T2_1, T2_2)
        
        self.T3 = np.nan_to_num(2 / (self.delta_sq_min * ((1 -  discount_factor) ** 2)), nan=0, posinf=0, neginf=0)
        
        self.T4 = np.nan_to_num(min(
            max(
                27 / (self.delta_sq_min * (1 -  discount_factor) ** 3),
                8 / (self.delta_sq_min * ((1-discount_factor) ** 2.5 )),
                14 * (self.span_V_greedy_max/((self.delta_sq_min ** 2/3) * ((1 - discount_factor) ** (4/3))))
            ),
            max(
                16 * self.var_V_greedy_max /  (self.delta_sq_min * (1 - discount_factor)**2),
                6 * (self.span_V_greedy_max/ ((self.delta_sq_min ** 2/3) * ((1-discount_factor) ** (4/3))))
            )
        ), nan=0, posinf=0, neginf=0)
    
        # Compute H and Hstar
        self.H = self.T1 + self.T2
        self.Hstar = self.dim_state * (self.T3 + self.T4)
        
        self.normalizer = (self.delta_sq_min * (1 -  discount_factor) ** 3) / (self.dim_state * self.dim_action)
        
    def _optimal_generative_allocation(self) -> npt.NDArray[np.float64]:
        # Compute allocation vector for the generative case
        omega = np.copy(self.H)
        if np.isclose(self.H.sum(), 0, atol=0):
            omega = np.ones((self.dim_state, self.dim_action)) / (self.dim_state * self.dim_action)
        else:
            omega[~self.idxs_subopt_actions] = np.sqrt(self.H.sum() * self.Hstar) / self.dim_state
            omega = omega / omega.sum()
        return omega
    
    def _optimal_allocation_with_navigation_constraints(self, reg: float = 1e-1, max_trials: int = 20, rel_tol: float = 0.03) -> npt.NDArray[np.float64]:
        """Compute optimal allocation with navigation constraints

        Parameters
        ----------
        reg : float, optional
            regularizer, in (0,1), used to initialize the problem, by default 1e-1
        max_trials : int, optional
            Sometimes the solver computes an inaccurate solution.
            Computing the problem several times may help.
            max_trials limits the number of trials, by default 20
        rel_tol : float, optional
            Used to verify the stability of the solution.
            The relative difference between the result of the optimization problem, 
            and the evaluation of the allocation vector, should not exceed this value, by default 0.03

        Returns
        -------
        npt.NDArray[np.float64]
            Computed allocation vector

        Raises
        ------
        Exception
            Error if it is not possible to compute a stable solution
        """        
        # Use as initial point the omega from the generative setting projected on the feasible set
        # defined by the navigation constraints
        
        
        ns, na = self.dim_state, self.dim_action
        omega0 = self._optimal_generative_allocation()
        omega0 = reg * np.ones((ns, na)) / (ns * na) +  (1- reg) * omega0
        omega0_proj = project_omega(omega0, self.P, force_policy=False)
        
        
        for trial in range(20):
            H = self.H * self.normalizer / (10 ** float( trial))
            Hstar = (self.T3 + self.T4)  * self.normalizer / (10 ** float(trial))
            
            
            omega = cp.Variable((self.dim_state, self.dim_action))
            sigma = cp.Variable(1, nonneg=True)
            constraints = [cp.sum(omega) == 1, omega >= sigma]
            constraints.extend([cp.sum(omega[s]) == cp.sum(cp.multiply(self.P[:,:,s], omega)) for s in range(self.dim_state)])
            
            tol = 0
            objective = cp.max(
                cp.multiply(
                    H[self.idxs_subopt_actions].reshape(self.dim_state, self.dim_action-1),
                    cp.inv_pos(tol + cp.reshape(omega[self.idxs_subopt_actions], (self.dim_state, self.dim_action-1)))
                    )
                ) \
                + cp.max(cp.inv_pos(tol + omega[~self.idxs_subopt_actions]) * Hstar)
            

            objective = cp.Minimize(objective)
            problem = cp.Problem(objective, constraints)
            omega.value = omega0_proj
            
            # We need to do this because the solution may be incorrect sometimes.
            # The idea is to project the solution back onto the set of navigation constraints
            # and see if there is any difference with the solution returned by the solver.
            # In case of incorrect solution, we should see a large relative difference between the two
            # results
            _trial = 0
            solvers = get_cvxpy_solvers()
            tol = 1e-14
            while True:
                try:
                    result = solve_with_fallback(problem, solvers, verbose=False, warm_start=True)
                    omega = omega.value
                    
                    # Check solution: project and evaluate relative difference
                    omega_nav_constr_proj = project_omega(omega, self.P, force_policy=False)
                    v1,v2 = self.evaluate_allocation(omega), self.evaluate_allocation(omega_nav_constr_proj)
                    eps = np.abs(v1-v2)/v1
                    
                    if eps < rel_tol:
                        omega = np.clip(omega, 1e-10, 1)
                        omega = omega / omega.sum()
                        return omega
                except Exception as e:
                    break
                    #solver = np.random.choice([cp.MOSEK, cp.ECOS, cp.SCS])
                    # if _trial == max_trials:
                    #     import pdb
                    #     pdb.set_trace()
                    #     print(f'Cannot solve the UB! {e}')
                    #     return np.ones((ns, na)) / (ns * na)
                # _trial += 1
                # if _trial > max_trials:
                #     print('Impossible to compute a stable solution!')
                #     return np.ones((ns, na)) / (ns * na)
        if trial == 20:
            print('There was an error...')
            return np.ones((ns, na)) / (ns * na) 
        omega = np.clip(omega, 1e-10, 1)
        omega = omega / omega.sum()
        return omega

    def evaluate_allocation(self, omega: npt.NDArray[np.float64], navigation_constraints: bool = False) -> float:
        """Evaluate a given allocation

        Parameters
        ----------
        omega : npt.NDArray[np.float64]
            Allocation to evaluate
        navigation_constraints : bool, optional
            If true, checks that the allocation verifies the navigation constraints, by default False

        Returns
        -------
        float
            The value of the allocation
        """        
        if navigation_constraints is True:
            checks = np.abs(np.array([np.sum(omega[s]) - np.sum(np.multiply(self.P[:,:,s], omega)) for s in range(self.dim_state)])) 
            assert checks.max() < 1e-5, "Allocation does not satisfy navigation constraints"

        H = self.H * self.normalizer
        Hstar = (self.T3 + self.T4) * self.normalizer

        U = np.max(
            H[self.idxs_subopt_actions].reshape(self.dim_state, self.dim_action-1)/omega[self.idxs_subopt_actions].reshape(self.dim_state, self.dim_action-1)) \
                + np.max(Hstar/ (omega[~self.idxs_subopt_actions]))
        return U / self.normalizer

    def compute_allocation(self, navigation_constraints: bool = False) -> Tuple[npt.NDArray[np.float64], float]:
        """Compute allocation vector

        Parameters
        ----------
        navigation_constraints : bool, optional
            enable navigation constranits, by default False

        Returns
        -------
        Tuple[npt.NDArray[np.float64], float]
            - Omega, an allocation of shape |S|x|A|
            - Value of Omega
        """        
        if navigation_constraints is False:
            omega = self._optimal_generative_allocation()
        else:
            omega = self._optimal_allocation_with_navigation_constraints()
        return omega, self.evaluate_allocation(omega, navigation_constraints=False)
        
    @property
    def dim_state(self) -> int:
        """Number of states"""        
        return self.P.shape[0]
    
    @property
    def dim_action(self) -> int:
        """Number of actions"""
        return self.P.shape[1]
    
if __name__ == '__main__':
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
    ns, na = 5,3
    np.random.seed(2)
    discount_factor = 0.99
    
    # Just a couple of tests
    for i in range(10):
        while True:
            try:
                P = np.random.dirichlet(np.ones(ns), size=(ns, na))
                R = np.random.dirichlet(np.ones(ns), size=(ns, na))
                
                
                
                mdp = MDPDescription(P, R, discount_factor)
                omega_gen, omega_gen_val = mdp.compute_allocation()
                omega_nav_constr, omega_nav_constr_val = mdp.compute_allocation(True)
                
                

                omega_gen_proj = project_omega(omega_gen, P, tol=1e-3)
                omega_nav_constr_proj = project_omega(omega_nav_constr, P, tol=1e-3)
                print(f'{omega_gen_val} - {mdp.evaluate_allocation(omega_gen)}')
                print(f'{omega_nav_constr_val} - {mdp.evaluate_allocation(omega_nav_constr)}')
                print(mdp.evaluate_allocation(omega_gen_proj, True))
                print(mdp.evaluate_allocation(omega_nav_constr_proj, True))
                print('--------------------------')
                break
            except Exception:
                continue
        
    
