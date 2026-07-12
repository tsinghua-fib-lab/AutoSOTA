import torch
import math
import pickle

from utils import is_pow_of_10, format_number, get_distributions, get_kernel_mat


def semi_discrete_objective(v, C, eps):
    batch_size = C.size(0)
    z = (v - C) / eps
    log_sum_exp = torch.logsumexp(z, dim=-1).mean() - math.log(batch_size)
    h_e = v.mean() - eps * log_sum_exp - eps
    return h_e


def v_conjugate(Ky_times_coeffs, Cxy_ts, eps):
    z = (Ky_times_coeffs - Cxy_ts) / eps
    v_c = torch.logsumexp(z, dim=1).mean() - math.log(len(Ky_times_coeffs))
    return -eps * v_c


def semi_dual_objective(Y_tr, Y_ts, Ky_times_coeffs, k_prev, k, Cxy_ts, coeffs, eps, sigma_sq):
    Ky_times_coeffs += get_kernel_mat(Y_ts, Y_tr[k_prev:k], sigma_sq) @ coeffs[k_prev:k]

    vk_avg = Ky_times_coeffs.mean()
    v_c_avg = v_conjugate(Ky_times_coeffs, Cxy_ts, eps)

    semidual_val = v_c_avg + vk_avg - eps
    return semidual_val, Ky_times_coeffs


def sgd_on_semi_discrete(seed, lr=5., n_iter=int(2e5), n_test=int(1e4), eps=0.01, eval_every=int(1e4)):
    torch.set_default_dtype(torch.float64)
    mu, nu = get_distributions()
    torch.manual_seed(42)
    X_test, Y = mu.sample((n_test,)), nu.sample((n_test,))

    torch.manual_seed(seed)
    v0 = torch.randn_like(Y)
    v_avg = v0.clone()  # for storing Polyak-Ruppert average
    v = v0.requires_grad_()

    C_test = torch.cdist(X_test.unsqueeze(1), Y.unsqueeze(1)) ** 2

    optimizer = torch.optim.SGD([v], lr=lr, maximize=True)
    iters = []
    obj_vals = []

    for k in range(1, n_iter + 1):
        optimizer.zero_grad()
        x = mu.sample((1,))
        C = (x.unsqueeze(1) - Y) ** 2
        obj_val = semi_discrete_objective(v, C, eps)

        obj_val.backward()
        optimizer.step()

        with torch.no_grad():
            v_avg.mul_((k - 1) / k).add_(v / k)
            # Save current value on test from time to time
            if is_pow_of_10(k) or k % eval_every == 0:
                obj_vals.append(semi_discrete_objective(v_avg, C_test, eps).item())
                iters.append(k)
                print(f'    seed {seed}: iter {format_number(k)}/{format_number(n_iter)}')

    fname = f"trajectories/ref_seed{seed}_lr{lr}.pickle"
    with open(fname, "wb") as file:
        pickle.dump((obj_vals, iters), file)


def kernel_sgd_on_dual(sigma_sq, lr, seed, n_iter=int(1e5)+1, n_test=int(1e4), eps=0.01, eval_every=int(1e4)):
    torch.set_default_dtype(torch.float64)
    mu, nu = get_distributions()
    torch.manual_seed(42)
    X_ts, Y_ts = mu.sample((n_test,)), nu.sample((n_test,))

    torch.manual_seed(seed)
    X_tr, Y_tr = mu.sample((n_iter,)), nu.sample((n_iter,))
    C_tr = (X_tr - Y_tr) ** 2
    Cxy_ts = torch.cdist(X_ts.unsqueeze(1), Y_ts.unsqueeze(1)) ** 2

    Ky_times_coeffs = torch.zeros_like(Y_ts)
    coeffs = torch.zeros(n_iter)
    obj_vals = []
    k_prev = 0
    iters = []

    coeffs[0] = lr * (1 - torch.exp((-C_tr[0]) / eps))  # Iteration k=0

    for k in range(1, n_iter):
        Kx_k = get_kernel_mat(X_tr[k:k+1], X_tr[:k], sigma_sq)
        Ky_k = get_kernel_mat(Y_tr[k:k+1], Y_tr[:k], sigma_sq)

        u_prev_at_yk = Kx_k @ coeffs[:k]
        v_prev_at_yk = Ky_k @ coeffs[:k]

        grad = 1 - torch.exp((u_prev_at_yk + v_prev_at_yk - C_tr[k]) / eps)
        coeffs[k] = lr * grad / math.sqrt(k + 1)

        if is_pow_of_10(k) or k % eval_every == 0:
            semidual_val, Ky_times_coeffs = semi_dual_objective(Y_tr, Y_ts, Ky_times_coeffs,
                                                                k_prev, k, Cxy_ts, coeffs, eps, sigma_sq)
            obj_vals.append(semidual_val.item())
            iters.append(k)
            k_prev = k
            print(f'    sigma^2={sigma_sq}, seed={seed}: iter {format_number(k)}/{format_number(n_iter)}')

    fname = f"trajectories/dual_seed{seed}_lr{lr}_sig{sigma_sq}.pickle"
    with open(fname, "wb") as file:
        pickle.dump((obj_vals, iters), file)


def kernel_sgd_on_approx_semidual(sigma_sq, lr, seed, rho, n_iter=int(1e5)+1, n_test=int(1e4), eps=0.01, eval_every=int(1e4)):
    torch.set_default_dtype(torch.float64)
    mu, nu = get_distributions()
    torch.manual_seed(42)
    X_ts, Y_ts = mu.sample((n_test,)), nu.sample((n_test,))

    torch.manual_seed(seed)
    X_tr, Y_tr = mu.sample((n_iter,)), nu.sample((n_iter,))
    C_tr = (X_tr - Y_tr) ** 2
    Cxy_ts = torch.cdist(X_ts.unsqueeze(1), Y_ts.unsqueeze(1)) ** 2
    Ky_times_coeffs = torch.zeros_like(Y_ts)

    coeffs = torch.zeros(n_iter)
    obj_vals = []
    k_prev = 0
    alpha = -C_tr[0] / eps
    log_rho = math.log(rho)
    iters = []

    for k in range(n_iter):
        if k > 0:
            Ky_k = get_kernel_mat(Y_tr[k:k+1], Y_tr[:k], sigma_sq)
            v_prev_at_yk = Ky_k @ coeffs[:k]
        else:
            v_prev_at_yk = torch.zeros(1)

        z = (v_prev_at_yk - C_tr[k]) / eps - alpha
        grad_v = 1 - torch.sigmoid(z + log_rho) / rho
        grad_alpha = -eps * grad_v.item()

        coeffs[k] = lr * grad_v / math.sqrt(k + 1)
        alpha += lr * grad_alpha / math.sqrt(k + 1)

        if is_pow_of_10(k) or k % eval_every == 0:
            obj_val, Ky_times_coeffs = semi_dual_objective(Y_tr, Y_ts, Ky_times_coeffs,
                                                           k_prev, k, Cxy_ts, coeffs, eps, sigma_sq)

            obj_vals.append(obj_val)
            iters.append(k)
            k_prev = k
            print(f'    rho={rho}, seed={seed}: iter {format_number(k)}/{format_number(n_iter)}')

    fname = f"trajectories/semidual_lr{lr}_sig{sigma_sq}_rho{rho}_seed{seed}.pickle"
    with open(fname, "wb") as file:
        pickle.dump((obj_vals, iters), file)
