""""
Implementation of gradient-descent ascent methods for CMDPs (in occupancy measure and policy space)
"""
import numpy as np
from env.gridworld import *
from einops import einsum
from scipy.stats import entropy

# policy space updates

def NPG_update(env, policy, r, beta, eta, init_v, max_iters=50, tol=1e-5):
    """
    Entropy regularized NPG for softmax policy parametrization. For eta = (1-gamma) / beta this reduces to soft-policy iteration.

    :param env: Gridworld environment.
    :param policy: Current policy, shape (n,m).
    :param r: Reward, shape (n,m).
    :param beta: Regularization parameter.
    :param eta: Learning rate.
    :param init_v: Initial values, shape (n,).
    :param max_iters: Maximum number of Bellman iterations for soft values approximation.
    :param tol: Tolerance for soft values approximation.
    :return: policy, soft_v
    """

    v = env.approx_soft_v_eval(init_v, policy, beta, r=r, max_iters=max_iters, tol=tol, logging=False)
    q = r + env.gamma * einsum(env.P, v, 's a s_next, s_next -> s a')
    q_shifted = q - np.amax(q, axis=1, keepdims=True)
    unnormalized_policy = policy ** (1 - eta * beta / (1 - env.gamma)) * np.exp(eta * q_shifted / (1 - env.gamma))
    return unnormalized_policy / np.sum(unnormalized_policy, axis=1, keepdims=True), v

def xi_update(env, xi, policy, eta, init_v_c, max_iters=50, tol=1e-5, threshold=float('inf')):
    """
    Calculates the projected gradient descent step for the dual variable xi.

    :param env: Gridworld environment.
    :param xi: Dual variable, shape (k,).
    :param policy: Current policy, shape (n,m).
    :param eta: Learning rate.
    :param init_v_c: Initial constraint cost values, shape (n, k).
    :param max_iters: Maximum number of Bellman iterations for value approximation.
    :param tol: Tolerance for value approximation.
    :param threshold: Threshold for dual variable xi.
    :return: xi_next, constraint cost values, xi gradient = constraint values
    """

    v_c = env.approx_vector_cost_eval(init_v_c, env.Psi, policy, max_iters=max_iters, tol=tol, logging=False)
    grad_xi = env.b - einsum(env.nu0, v_c, 's, s k->k')
    return np.clip(xi - eta * grad_xi, 0.0, threshold), v_c, grad_xi


# policy space algorithm

def cmdp_gda(env, beta , eta_p, eta_xi , r = None, max_iters=1e4, tol=1e-3, mode='sim_gda', n_v_tot_eval_steps=10, n_v_c_eval_steps=10, logging = True, check_steps = 100):

    r = env.r if r is None else r

    v_tot = np.zeros(env.n)
    v_c = np.zeros((env.n, env.k))
    xi = np.zeros_like(env.b)
    policy = env.soft_v_greedy(v_tot, beta)
    it = 0
    n_converged = 0
    while it < max_iters:

        # primal update
        r_xi = r - np.einsum('jki,i->jk', env.Psi, xi)
        policy_next, v_tot = NPG_update(env, policy, r_xi, beta, eta_p, v_tot, max_iters=n_v_tot_eval_steps)

        if mode == 'sim_gda':
            # dual update
            xi, v_c, grad_xi = xi_update(env, xi, policy, eta_xi, v_c, max_iters=n_v_c_eval_steps)

        elif mode == 'alt_gda':
            # dual update
            xi, v_c, grad_xi = xi_update(env, xi, policy_next, eta_xi, v_c, max_iters=n_v_c_eval_steps)

        else:
            print('Non-existent GDA mode specified')
            break

        # early stopping: check policy convergence
        if tol > 0:
            policy_delta = float(np.max(np.abs(policy_next - policy)))
            if policy_delta < tol:
                n_converged += 1
                if n_converged >= 50:
                    if logging:
                        print('step: %d, converged (policy_delta=%e < tol=%e for 50 steps)' % (it, policy_delta, tol))
                    break
            else:
                n_converged = 0

        # logging
        if it % check_steps == 0:
            if logging:
                print('step: ', it, ', primal: ', '%0.5f' % (np.einsum('i,i', env.nu0, v_tot) + np.dot(xi, env.b)), ', dual: ', xi,
                      ', constraint-viol: ', '%0.5f' % np.sum(np.clip(-grad_xi, 0.0, float('inf'))))

        # update some variables
        policy = policy_next
        it += 1

    return policy, xi, v_tot

# occupancy measure updates
def lagrangian_occ(env, A, mu, r, xi, v, beta):
    policy = env.occ2policy(mu)
    H = entropy(policy, axis=1)
    grad_xi = env.b - einsum(mu, env.Psi, 's a, s a k -> k')/(1-env.gamma)
    grad_v = einsum(A, mu, 's a s_next, s a -> s_next') - (1-env.gamma) * env.nu0
    return einsum(mu, r, 's a, s a ->')/(1-env.gamma) - regularization(env, mu, beta) + xi.T @ grad_xi + v.T @ grad_v

def regularization(env, mu, beta):
    policy = env.occ2policy(mu)
    nu = np.sum(mu, axis=1)
    return -beta * np.sum(nu * entropy(policy, axis=1)) / (1-env.gamma)

def grad_causal_entropy_occ(env, mu, beta):
    policy = env.occ2policy(mu)
    if env.n > 1:
        return -beta * np.log(policy) / (1-env.gamma)
    else:
        return -beta * (np.log(policy) + 1) / (1-env.gamma)

def mu_grad_occ(env, A, mu, r, v, beta):
    return r / (1 - env.gamma) + einsum(v, A, 's_next, s a s_next -> s a') + grad_causal_entropy_occ(env, mu, beta)

def mu_update_occ(env, A, mu_last, mu, r, v, beta, eta):
    mu_next = mu_last * np.exp(eta * mu_grad_occ(env, A, mu, r, v, beta))
    mu_next = np.clip(mu_next, 1e-20, float('inf'))  # clipping to avoid division by zero
    return mu_next / np.sum(mu_next)

def xi_update_occ(env, xi_last, mu, eta, threshold=float('inf')):
    grad_xi = env.b - einsum(env.Psi, mu, 's a k, s a -> k') / (1 - env.gamma)
    return np.clip(xi_last - eta * grad_xi, 0.0, threshold), grad_xi

def v_update_occ(env, A, v_last, mu, eta):
    grad_v = einsum(A, mu, 's a s_next, s a -> s_next') - (1 - env.gamma) * env.nu0
    return v_last - eta * grad_v, grad_v

# occupancy measure space algorithm

def cmdp_gda_occ(env, beta , eta_mu, eta_xi, eta_v, r = None, max_iters=1e4, mode='sim_gda', logging = True, check_steps = 10000):

    r = env.r if r is None else r
    A = env.A_tensor()

    mu = np.ones((env.n, env.m)) / (env.n*env.m)
    v = np.zeros(env.n)
    xi = np.zeros_like(env.b)

    mu_avg = np.zeros_like(mu)
    v_avg = np.zeros_like(v)
    xi_avg = np.zeros_like(xi)

    it = 0
    while it < max_iters:

        # primal update (common for all modes)
        r_xi = r - np.einsum('jki,i->jk', env.Psi, xi)
        mu_int = mu_update_occ(env, A, mu, mu, r_xi, v, beta, eta_mu)

        if mode == 'sim_gda':
            # dual update
            xi_next, grad_xi = xi_update_occ(env, xi, mu, eta_xi)
            v_next, grad_v = v_update_occ(env, A, v, mu, eta_v)
            mu_next = mu_int

        elif mode == 'extragradient':
            # intermediate extrapolation step
            # dual update
            xi_int, _ = xi_update_occ(env, xi, mu, eta_xi)
            v_int, _ = v_update_occ(env, A, v, mu, eta_v)

            # update step
            # primal update
            r_xi_int = r - np.einsum('jki,i->jk', env.Psi, xi_int)
            mu_next = mu_update_occ(env, A, mu, mu_int, r_xi_int, v_int, beta, eta_mu)

            # dual update
            xi_next, grad_xi = xi_update_occ(env, xi, mu_int, eta_xi)
            v_next, grad_v = v_update_occ(env, A, v, mu_int, eta_v)

        elif mode == 'alt_gda':
            # dual update
            xi_next, grad_xi = xi_update_occ(env, xi, mu_int, eta_xi)
            v_next, grad_v = v_update_occ(env, A, v, mu_int, eta_v)
            mu_next = mu_int

        else:
            print('Non-existent GDA mode specified')
            break

        # logging
        if it % check_steps == 0:
            if logging:
                print('step: ', it, ', primal value: ', '%0.5f' % np.sum(mu*r / (1 - env.gamma)), ', mu_delta: ',
                      '%0.5f' % np.sum(np.abs(mu_next - mu))
                      , ', xi: ', xi, ', constraint-viol: ',
                      '%0.5f' % np.sum(np.clip(-grad_xi, 0.0, float('inf'))),
                      ', v_delta: ',
                      '%0.5f' % np.sum(np.abs(v_next - v)),
                      ', grad_v: ', '%0.5f' % np.sum(np.abs(grad_v)))

        # averaging
        mu_avg = (mu_avg * it + mu_next) / (it + 1)
        xi_avg = (xi_avg * it + xi_next) / (it + 1)
        v_avg = (v_avg * it + v_next) / (it + 1)

        # update some variables
        mu = mu_next
        xi = xi_next
        v = v_next
        it += 1

    return mu, xi, v, mu_avg, xi_avg, v_avg