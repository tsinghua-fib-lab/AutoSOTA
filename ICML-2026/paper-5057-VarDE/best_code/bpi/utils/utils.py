# This file is licensed under the MIT License.
# See the LICENSE file in the project root for full license information.
#

import numpy as np
import cvxpy as cp
from cvxpy.constraints.constraint import Constraint
from numpy.typing import NDArray
from typing import Optional, Tuple, List, Callable
from scipy.linalg._fblas import dger, dgemm
from typing import NamedTuple

policy_evaluation_c = None
try:
    import pyximport
    _ = pyximport.install(
        setup_args={"include_dirs": np.get_include()},
        reload_support=True,
    )
    from .cutils import policy_evaluation as policy_evaluation_c
except Exception:
    # Optional Cython acceleration; fall back to pure Python if build/import fails.
    policy_evaluation_c = None


class Results(NamedTuple):
    step: int
    omega: NDArray[np.float64]
    greedy_policy: NDArray[np.int64]
    total_state_visits: NDArray[np.float64]
    last_visit: NDArray[np.float64]
    exp_visits: NDArray[np.float64]
    eval_greedy: NDArray[np.float64]
    elapsed_time: float


def policy_evaluation(
        gamma: float,
        P: NDArray[np.float64],
        R: NDArray[np.float64],
        pi: NDArray[np.int64],
        V0: Optional[NDArray[np.float64]] = None,
        atol: float = 1e-6) -> NDArray[np.float64]:
    """Policy evaluation

    Args:
        gamma (float): Discount factor
        P (NDArray[np.float64]): Transition function of shape (num_states, num_actions, num_states)
        R (NDArray[np.float64]): Reward function of shape (num_states, num_actions)
        pi (Optional[NDArray[np.int64]], optional): policy
        V0 (Optional[NDArray[np.float64]], optional): Initial value function. Defaults to None.
        atol (float): Absolute tolerance

    Returns:
        NDArray[np.float64]: Value function
    """
    
    NS, NA = P.shape[:2]
    # Initialize values
    if V0 is None:
        V0 = np.zeros(NS)
    
    V = V0.copy()
    while True:
        Delta = 0
        V_next = np.array([P[s, pi[s]] @ (R[s, pi[s]] + gamma * V) for s in range(NS)])
        
        Delta = np.max([Delta, np.abs(V_next - V).max()])
        V = V_next

        if Delta < atol:
            break
    return V
        

def policy_iteration(
        gamma: float,
        P: NDArray[np.float64],
        R: NDArray[np.float64],
        pi0: Optional[NDArray[np.int64]] = None,
        V0: Optional[NDArray[np.float64]] = None,
        atol: float = 1e-6) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Policy iteration

    Args:
        gamma (float): Discount factor
        P (NDArray[np.float64]): Transition function of shape (num_states, num_actions, num_states)
        R (NDArray[np.float64]): Reward function of shape (num_states, num_actions)
        pi0 (Optional[NDArray[np.int64]], optional): Initial policy. Defaults to None.
        V0 (Optional[NDArray[np.float64]], optional): Initial value function. Defaults to None.
        atol (float): Absolute tolerance

    Returns:
        NDArray[np.float64]: Optimal value function
        NDArray[np.float64]: Optimal policy
        NDArray[np.float64]: Optimal Q function
    """
    
    NS, NA = P.shape[:2]

    # Initialize values    
    V = V0 if V0 is not None else np.zeros(NS)
    pi = pi0 if pi0 is not None else np.random.binomial(1, p=0.5, size=(NS))
    next_pi = np.zeros_like(pi)
    policy_stable = False
    while not policy_stable:
        policy_stable = True
        V = policy_evaluation(gamma, P, R, pi, V, atol)
        #V = policy_evaluation_c(gamma, P, R[..., np.newaxis] if len(R.shape) == 2 else R, pi, atol)
        Q = [[P[s,a] @ (R[s,a] + gamma * V) for a in range(NA)] for s in range(NS)]
        next_pi = np.argmax(Q, axis=1)
        
        if np.any(next_pi != pi):
            policy_stable = False
        pi = next_pi

    return V, pi, Q

def soft_policy_iteration(
        gamma: float,
        theta: float,
        P: NDArray[np.float64],
        R: NDArray[np.float64],
        pi0: Optional[NDArray[np.float64]] = None,
        Q0: Optional[NDArray[np.float64]] = None,
        atol: float = 1e-6) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Policy iteration

    Args:
        gamma (float): Discount factor
        P (NDArray[np.float64]): Transition function of shape (num_states, num_actions, num_states)
        R (NDArray[np.float64]): Reward function of shape (num_states, num_actions)
        pi0 (Optional[NDArray[np.int64]], optional): Initial policy. Defaults to None.
        V0 (Optional[NDArray[np.float64]], optional): Initial value function. Defaults to None.
        atol (float): Absolute tolerance

    Returns:
        NDArray[np.float64]: Optimal value function
        NDArray[np.float64]: Optimal policy
        NDArray[np.float64]: Optimal Q function
    """
    
    NS, NA = P.shape[:2]

    # Initialize values    
    Q = Q0 if Q0 is not None else np.zeros((NS, NA))

    compute_V = lambda X: theta * np.log(np.exp((X-X.max())/theta).sum(-1)) +X.max()
    Vsoft = compute_V(Q)
    while True:
        Delta = 0
        Q_next = np.array([[P[s, a] @ (R[s, a] + gamma * Vsoft) for a in range(NA)] for s in range(NS)])
        #V_next = entropy_pi + (pi * Q_next).sum(-1)
        V_next = compute_V(Q_next)
        Delta = np.max([Delta, np.abs(V_next - Vsoft).max()])
        Q = Q_next
        Vsoft = V_next
        
        if Delta < atol:
            break
    
    pi = np.exp((Q - Q.max())/theta)
    pi = pi / pi.sum(-1)[:, np.newaxis]

    return Vsoft, pi, Q

def project_omega(
        x: NDArray[np.float64],
        P: NDArray[np.float64],
        force_policy: bool = True) -> NDArray[np.float64]:
    """Project omega using navigation constraints

    Parameters
    ----------
    x : NDArray[np.float64]
        Allocation vector to project
    P : NDArray[np.float64]
        Transition matrix (S,A,S)

    Returns
    -------
    NDArray[np.float64]
        The projected allocation vector
    """
    ns, na = P.shape[:2]
    omega = cp.Variable((ns, na), nonneg=True)
    constraints = [cp.sum(omega) == 1]
    constraints.extend([cp.sum(omega[s]) == cp.sum(cp.multiply(P[:,:,s], omega)) for s in range(ns)])
    if force_policy:
        constraints.append(omega >= 1e-15)
        constraints.extend(
            [omega[s,a] == x[s,a]/np.sum(x[s]) * cp.sum(omega[s]) for a in range(na) for s in range(ns)])
    else:
        constraints.append(omega >= 1e-4)
  
    objective = cp.Minimize(0.5 * cp.norm(x - omega, 2)**2)
    problem = cp.Problem(objective, constraints)
        
    solvers = get_cvxpy_solvers()
    res = solve_with_fallback(problem, solvers, verbose=False)
    #print(f'res: {res}')
    return omega.value

def compute_stationary_distribution(
        x: NDArray[np.float64],
        P: NDArray[np.float64]) -> NDArray[np.float64]:
    """Project omega using navigation constraints

    Parameters
    ----------
    x : NDArray[np.float64]
        Allocation vector to project
    P : NDArray[np.float64]
        Transition matrix (S,A,S)

    Returns
    -------
    NDArray[np.float64]
        The projected allocation vector
    """
    ns, na = P.shape[:2]
    omega = cp.Variable((ns, na), nonneg=True)
    constraints = [cp.sum(omega) == 1, omega >= 1e-15]
    constraints.extend([cp.sum(omega[s]) == cp.sum(cp.multiply(P[:,:,s], omega)) for s in range(ns)])
    constraints.extend([omega[s,a] == x[s,a]/np.sum(x[s]) * cp.sum(omega[s]) for a in range(na) for s in range(ns)])
  
    problem = cp.Problem(cp.Minimize(1), constraints)
    solvers = get_cvxpy_solvers()
    res = solve_with_fallback(problem, solvers, verbose=False)
    #print(f'res: {res}')
    return omega.value


def get_cvxpy_solvers(preferred: Optional[List[str]] = None) -> List[str]:
    if preferred is None:
        preferred = [cp.MOSEK, cp.ECOS, cp.SCS]
    installed = set(cp.installed_solvers())
    solvers = [s for s in preferred if s in installed]
    if not solvers:
        raise Exception(f'No supported CVXPY solvers installed. Found: {sorted(installed)}')
    return solvers


def solve_with_fallback(problem: cp.Problem, solvers: List[str], **kwargs):
    last_exc = None
    for solver in solvers:
        try:
            return problem.solve(solver=solver, **kwargs)
        except Exception as exc:
            last_exc = exc
    raise Exception(f'Cannot solve with available solvers {solvers}. Last error: {last_exc}')


def is_positive_definite(x: np.ndarray, atol: float = 1e-9) -> bool:
    """Check if a matrix is positive definite
    Args:
        x (np.ndarray): matrix
        atol (float, optional): absolute tolerance. Defaults to 1e-9.
    Returns:
        bool: Returns True if the matrix is positive definite
    """    
    return np.all(np.linalg.eigvals(x) > atol)

def is_symmetric(a: np.ndarray, rtol: float = 1e-05, atol: float = 0) -> bool:
    """Check if a matrix is symmetric
    Args:
        a (np.ndarray): matrix to check
        rtol (float, optional): relative tolerance. Defaults to 1e-05.
        atol (float, optional): absolute tolerance. Defaults to 1e-08.
    Returns:
        bool: returns True if the matrix is symmetric
    """    
    return np.allclose(a, a.T, rtol=rtol, atol=atol)

def mean_cov(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Returns mean and covariance of a matrix
    See https://groups.google.com/g/scipy-user/c/FpOU4pY8W2Y
    Args:
        X (np.ndarray): _description_
    Returns:
        Tuple[np.ndarray, np.ndarray]: (Mean,Covariance) tuple
    """   
    n, p = X.shape
    m = X.mean(axis=0)
    # covariance matrix with correction for rounding error
    # S = (cx'*cx - (scx'*scx/n))/(n-1)
    # Am Stat 1983, vol 37: 242-247.
    cx = X - m
    scx = cx.sum(axis=0)
    scx_op = dger(-1.0/n,scx,scx)
    S = dgemm(1.0, cx.T, cx.T, beta=1.0,
    c = scx_op, trans_a=0, trans_b=1, overwrite_c=1)
    S[:] *= 1.0/(n-1)
    return m, S.T

def unit_vector(vector: NDArray[np.float64]) -> NDArray[np.float64]:
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1: NDArray[np.float64], v2: NDArray[np.float64]) -> float:
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def gram_schmidt(vectors):
    basis = []
    for v in vectors:
        w = v - np.sum( np.dot(v,b)*b / np.dot(b,b)  for b in basis )
        if (w > 1e-10).any():  
            basis.append(w) #/np.linalg.norm(w))
    return np.array(basis)
