from functools import partial
import numpy as np
import torch
import matplotlib.pyplot as plt
from ddsketch import DDSketch
from scipy.optimize import fsolve, newton, newton_krylov, brentq
from tdigest import TDigest
from .class_CP_QQ import calc_matrix_M, sum_hypergeo, Multi_Boucle


def calibrate_lac(scores, targets, alpha=0.1, return_dist=False):
    assert scores.size(0) == targets.size(0)
    assert targets.size(0)
    n = torch.tensor(targets.size(0))
    assert n

    score_dist = torch.take_along_dim(1 - scores, targets.unsqueeze(1), 1).flatten()
    assert (
        0 <= torch.ceil((n + 1) * (1 - alpha)) / n <= 1
    ), f"{alpha=} {n=} {torch.ceil((n+1)*(1-alpha))/n=}"
    qhat = torch.quantile(
        score_dist, torch.ceil((n + 1) * (1 - alpha)) / n, interpolation="higher"
    )
    return (qhat, score_dist) if return_dist else qhat


def inference_lac(scores, qhat, allow_empty_sets=False):
    n = scores.size(0)

    elements_mask = scores >= (1 - qhat)

    if not allow_empty_sets:
        elements_mask[torch.arange(n), scores.argmax(1)] = True

    return elements_mask


def calibrate_aps(scores, targets, alpha=0.1, return_dist=False):
    # n = scores.shape[0]
    # cal_smx = scores.numpy()
    # cal_labels = targets.numpy()
    # # Get scores. calib_X.shape[0] == calib_Y.shape[0] == n
    # cal_pi = cal_smx.argsort(1)[:,::-1]; cal_srt = np.take_along_axis(cal_smx,cal_pi,axis=1).cumsum(axis=1)
    # cal_scores = np.take_along_axis(cal_srt,cal_pi.argsort(axis=1),axis=1)[range(n),cal_labels]
    # # Get the score quantile
    # qhat = np.quantile(cal_scores, np.ceil((n+1)*(1-alpha))/n, interpolation='higher')
    # return qhat

    assert scores.size(0) == targets.size(0)
    n = torch.tensor(targets.size(0))
    assert n

    sorted_index = torch.argsort(scores, descending=True)
    sorted_scores_cumsum = torch.take_along_dim(scores, sorted_index, dim=1).cumsum(
        dim=1
    )
    score_dist = torch.take_along_dim(sorted_scores_cumsum, sorted_index.argsort(1), 1)[
        torch.arange(n), targets
    ]
    assert (
        0 <= torch.ceil((n + 1) * (1 - alpha)) / n <= 1
    ), f"{alpha=} {n=} {torch.ceil((n+1)*(1-alpha))/n=}"
    qhat = torch.quantile(
        score_dist, torch.ceil((n + 1) * (1 - alpha)) / n, interpolation="higher"
    )
    assert 0 < qhat <= 1, f"{qhat=:.4f}"
    return (qhat, score_dist) if return_dist else qhat


def inference_aps(scores, qhat, allow_empty_sets=False):
    sorted_index = scores.argsort(1, descending=True)
    sorted_scores_cumsum = torch.take_along_dim(scores, sorted_index, dim=1).cumsum(1)
    elements_mask = sorted_scores_cumsum <= qhat

    if not allow_empty_sets:
        elements_mask[:, 0] = True

    prediction_sets = torch.take_along_dim(elements_mask, sorted_index.argsort(1), 1)
    return prediction_sets


def calibrate_raps(
    scores, targets, alpha=0.1, k_reg=1, lam_reg=0.01, return_dist=False
):
    # RAPS regularization parameters (larger lam_reg and smaller k_reg leads to smaller sets)
    assert scores.size(0) == targets.size(0)
    n = torch.tensor(targets.size(0))
    assert n
    num_classes = scores.shape[1]
    assert num_classes and 0 < k_reg <= num_classes
    reg = torch.cat(
        [torch.zeros(k_reg), torch.tensor([lam_reg]).repeat(num_classes - k_reg)]
    ).unsqueeze(0)

    sorted_index = torch.argsort(scores, descending=True)
    reg_sorted_scores = reg + torch.take_along_dim(scores, sorted_index, dim=1)
    score_dist = torch.take_along_dim(
        reg_sorted_scores.cumsum(1), sorted_index.argsort(1), 1
    )[torch.arange(n), targets]
    score_dist -= (
        torch.rand(n)
        * reg_sorted_scores[
            torch.arange(n), torch.where(sorted_index == targets[:, None])[1]
        ]
    )
    assert (
        0 <= torch.ceil((n + 1) * (1 - alpha)) / n <= 1
    ), f"{alpha=} {n=} {torch.ceil((n+1)*(1-alpha))/n=}"
    qhat = torch.quantile(
        score_dist, torch.ceil((n + 1) * (1 - alpha)) / n, interpolation="higher"
    )
    assert 0 < qhat <= 1, f"{qhat=:.4f}"
    # qhat = torch.minimum(torch.tensor(1.0), qhat)
    return (qhat.item(), score_dist) if return_dist else qhat.item()


def inference_raps(
    scores,
    qhat,
    allow_empty_sets=False,
    k_reg=1,
    lam_reg=0.01,
):
    num_classes = scores.shape[1]
    reg = torch.cat(
        [torch.zeros(k_reg), torch.tensor([lam_reg]).repeat(num_classes - k_reg)]
    ).unsqueeze(0)

    # scores = torch.tensor(scores)
    # qhat = torch.tensor(qhat)
    sorted_index = scores.argsort(1, descending=True)
    sorted_scores = torch.take_along_dim(scores, sorted_index, dim=1)
    reg_sorted_scores = sorted_scores + reg
    elements_mask = (
        reg_sorted_scores.cumsum(dim=1)
        - torch.rand(*reg_sorted_scores.shape) * reg_sorted_scores
    ) <= qhat

    if not allow_empty_sets:
        elements_mask[:, 0] = True

    prediction_sets = torch.take_along_dim(
        elements_mask, sorted_index.argsort(axis=1), axis=1
    )
    return prediction_sets


def get_coverage(psets, targets, precision=None):
    psets = psets.clone().detach()
    targets = targets.clone().detach()
    n = psets.shape[0]
    coverage = psets[torch.arange(n), targets].float().mean().item()
    if precision is not None:
        coverage = round(coverage, precision)
    return coverage


def get_size(psets, precision=1):
    psets = psets.clone().detach()
    size = psets.sum(1).float().mean().item()
    if precision is not None:
        size = round(size, precision)
    return size


def get_coverage_by_class(psets, targets, num_classes):
    psets = psets.clone().detach()
    targets = targets.clone().detach()
    results = {}
    for c in range(num_classes):
        index = targets == c
        psets_c = psets[index]
        targets_c = targets[index]
        results[c] = get_coverage(psets_c, targets_c)
    return results


def get_efficiency_by_class(psets, targets, num_classes):
    psets = psets.clone().detach()
    targets = targets.clone().detach()
    sizes = psets.sum(1)
    results = {}
    for c in range(num_classes):
        index = targets == c
        psets_c = psets[index]
        results[c] = get_size(psets_c)
    return results


def get_coverage_size_over_alphas(
    cal_scores,
    cal_targets,
    test_scores,
    test_targets,
    alphas,
    method="lac",
    allow_empty_sets=False,
    k_reg=1,
    lam_reg=0.01,
    decentral=False,
    client_index_map=None,
    precision=4,
    quantile_method: str = 'tdigest',
):
    n = test_targets.shape[0]
    coverage_results, size_results = {}, {}
    q1_coverage, q1_size = {}, {}
    q2_coverage, q2_size = {}, {}
    q3_coverage, q3_size = {}, {}
    q4_coverage, q4_size = {}, {}

    for alpha in alphas:
        if method == "lac":
            if decentral and client_index_map is not None:
                qhat = get_distributed_quantile(
                    cal_scores,
                    cal_targets,
                    alpha=alpha,
                    method="lac",
                    client_index_map=client_index_map,
                    quantile_method=quantile_method,
                )
            else:
                qhat = calibrate_lac(cal_scores, cal_targets, alpha=alpha)
            psets = inference_lac(test_scores, qhat, allow_empty_sets=allow_empty_sets)
        elif method == "aps":
            if decentral and client_index_map is not None:
                qhat = get_distributed_quantile(
                    cal_scores,
                    cal_targets,
                    alpha=alpha,
                    method="aps",
                    client_index_map=client_index_map,
                    quantile_method=quantile_method,
                )
            else:
                qhat = calibrate_aps(cal_scores, cal_targets, alpha=alpha)
            psets = inference_aps(test_scores, qhat, allow_empty_sets=allow_empty_sets)
        elif method == "raps":
            if decentral and client_index_map is not None:
                qhat = get_distributed_quantile(
                    cal_scores,
                    cal_targets,
                    alpha=alpha,
                    method="raps",
                    client_index_map=client_index_map,
                    k_reg=k_reg, lam_reg=lam_reg,
                    quantile_method=quantile_method,
                )
            else:
                qhat = calibrate_raps(
                    cal_scores, cal_targets, alpha=alpha, k_reg=k_reg, lam_reg=lam_reg
                )
            psets = inference_raps(
                test_scores,
                qhat,
                allow_empty_sets=allow_empty_sets,
                k_reg=k_reg,
                lam_reg=lam_reg,
            )
        elif method == "naive_lac":
            psets = inference_lac(
                test_scores, 1 - alpha, allow_empty_sets=allow_empty_sets
            )
        elif method == "naive_aps":
            psets = inference_aps(
                test_scores, 1 - alpha, allow_empty_sets=allow_empty_sets
            )
        else:
            raise ValueError()
            
        t = round(1-alpha, 2)

        coverage_results[t] = get_coverage(psets, test_targets, precision=precision)
        size_results[t] = get_size(psets)
        
        s = psets.sum(1).float()
        q1 = torch.quantile(s, 0.25, interpolation='higher')
        q2 = torch.quantile(s, 0.5, interpolation='higher')
        q3 = torch.quantile(s, 0.75, interpolation='higher')
        # m = torch.median(s)
        q1_index = torch.nonzero(s <= q1).flatten()
        q2_index = torch.nonzero((q1 < s) & (s <= q2)).flatten()
        q3_index = torch.nonzero((q2 < s) & (s <= q3)).flatten()
        q4_index = torch.nonzero(q3 < s).flatten()
        t = round(1-alpha, 2)
        q1_coverage[t] = get_coverage(psets[q1_index], test_targets[q1_index])
        q2_coverage[t] = get_coverage(psets[q2_index], test_targets[q2_index])
        q3_coverage[t] = get_coverage(psets[q3_index], test_targets[q3_index])
        q4_coverage[t] = get_coverage(psets[q4_index], test_targets[q4_index])
        q1_size[t] = get_size(psets[q1_index])
        q2_size[t] = get_size(psets[q2_index])
        q3_size[t] = get_size(psets[q3_index])
        q4_size[t] = get_size(psets[q4_index])

    return dict(
        coverage=coverage_results, size=size_results,
        q1_coverage=q1_coverage, q1_size=q1_size,
        q2_coverage=q2_coverage, q2_size=q2_size,
        q3_coverage=q3_coverage, q3_size=q3_size,
        q4_coverage=q4_coverage, q4_size=q4_size,
    )


def get_distributed_quantile(
    cal_scores,
    cal_targets,
    alpha: float,
    method="lac",
    k_reg=None,
    lam_reg=None,
    client_index_map: dict = None,
    quantile_method: str = 'tdigest',
):
    # choose score function
    if method == "lac":
        score_func = calibrate_lac
    elif method == "aps":
        score_func = calibrate_aps
    elif method == "raps":
        score_func = partial(calibrate_raps, k_reg=k_reg, lam_reg=lam_reg)
    else:
        raise ValueError(f"{method} score function not implemented")

    if quantile_method == 'tdigest':
        digest = TDigest()
    elif quantile_method == 'ddsketch':
        sketch = DDSketch()
    elif quantile_method == 'mean':
        mean_q = 0
    else:
        raise ValueError(f'{quantile_method} not supported')
        
    # N = 0
    # lambdas = {}
    # for k, (client, index) in enumerate(client_index_map.items()):
    #     scores = cal_scores[index]
    #     targets = cal_targets[index]
    #     n = targets.shape[0]
    #     N += n
    #     lambdas[k] = n + 1
        
    # assert np.allclose(sum(lambdas.values()), 10), sum(lambdas.values())
        
    for k, (client, index) in enumerate(client_index_map.items()):
        scores = cal_scores[index]
        targets = cal_targets[index]

        assert len(scores), len(targets)

        q, score_dist = score_func(scores, targets, alpha=alpha, return_dist=True)

        # add client scores and merge with global to estimate quantile
        if quantile_method == 'tdigest':
            client_digest = TDigest()
            client_digest.batch_update(score_dist.numpy())
            # client_digest.batch_update(score_dist.numpy(), w=lambdas[k])
            digest = digest + client_digest
        elif quantile_method == 'ddsketch':
            decentral_sketch = DDSketch()
            client_sketch = DDSketch()
            for score in score_dist.tolist():
                client_sketch.add(score)
                # client_sketch.add(score, weight=lambdas[k])
            sketch.merge(client_sketch)
        elif quantile_method == 'mean':
            mean_q += q

    N = cal_scores.shape[0]
    K = len(client_index_map)
    t = np.ceil((N + K) * (1 - alpha)) / N
    if quantile_method == 'tdigest':
        q_hat = digest.percentile(round(100*t))
    elif quantile_method == 'ddsketch':
        q_hat = sketch.get_quantile_value(t)
    elif quantile_method == 'mean':
        q_hat = mean_q / len(client_index_map)

    return q_hat


def get_decentralized_quantile(
        cal_scores,
        cal_targets,
        alpha: float,
        method="lac",
        k_reg=None,
        lam_reg=None,
        gossip=10,
        W=None,
        G=None,
        R=10,
        kappa=None,
        q0=None,
        mu=None,
        epsilon_0=None,
        qlevel=None,
        opt_method='ADMM',
        client_index_map: dict = None,
        iid_flag=False,
        quantile_method: str = 'tdigest',
):
    # choose score function
    if method == "lac":
        score_func = calibrate_lac
    elif method == "aps":
        score_func = calibrate_aps
    elif method == "raps":
        score_func = partial(calibrate_raps, k_reg=k_reg, lam_reg=lam_reg)
    else:
        raise ValueError(f"{method} score function not implemented")

    N = cal_scores.shape[0]
    K = len(client_index_map)

    # Quantization Set
    num_clients = len(client_index_map)
    local_scores = {}

    if iid_flag:
        # IID case
        permutation = np.random.permutation(len(client_index_map['client_0']))
        cal_scores = cal_scores[permutation]
        cal_targets = cal_targets[permutation]
        alpha_tilde = 1 - np.ceil((1-alpha)*(N+1))/N
    else:
        alpha_tilde = 1 - np.ceil((1-alpha)*(N+K))/N

    num_points = 0
    for k, (client, index) in enumerate(client_index_map.items()):
        if iid_flag:
            # IID case
            num_points = np.floor(len(index) / num_clients)
            scores = cal_scores[np.int32(k*num_points):np.int32((k+1)*num_points)]
            targets = cal_targets[np.int32(k*num_points):np.int32((k+1)*num_points)]
        else:
            scores = cal_scores[index]
            targets = cal_targets[index]
            num_points = max(num_points, len(scores))

        assert len(scores), len(targets)

        q, score_dist = score_func(scores, targets, alpha=alpha, return_dist=True)

        # Store the fraction of scores
        local_scores[k] = np.array(score_dist)

    # Gossip GD for pinball loss
    # R = 50
    T = int(gossip / R)
    eta = 5e-4

    # Graph parameter
    eigenvalues = np.sort(np.abs(np.linalg.eigvals(W)))
    p = 1 - np.abs(eigenvalues[-2])
    beta = np.linalg.norm(W - 1/K * (np.ones(shape=(K, 1)) @ np.ones(shape=(1, K))), 2)
    tilde_beta = np.sqrt(2) * np.power(1 - np.sqrt(1 - beta), R)
    edges = list(G.edges())
    E = len(edges)
    # Construct A_1 and A_2
    A_1 = np.zeros((E, K))
    A_2 = np.zeros((E, K))
    for q, (i, j) in enumerate(edges):
        A_1[q, i] = 1
        A_2[q, j] = 1
    A = np.vstack([A_1, A_2])
    B = np.vstack([-np.eye(E), -np.eye(E)])
    M_plus = A_1.transpose() + A_2.transpose()
    M_minus = A_1.transpose() - A_2.transpose()
    L_plus = 1/2 * M_plus @ M_plus.transpose()
    L_minus = 1/2 * M_minus @ M_minus.transpose()
    W_ADMM = 1/2 * (L_plus + L_minus)

    U, S, Vt = np.linalg.svd(M_plus)
    sigma_max_M = S[0]
    U, S, Vt = np.linalg.svd(M_minus)
    non_zero_singular_values = S[S > 1e-10]
    sigma_min_M = np.min(non_zero_singular_values)
    kappa_G = sigma_max_M / sigma_min_M

    # Gossip parameter
    eta_MG = (1-np.sqrt(1-beta**2))/(1+np.sqrt(1+beta**2))
    M_r = np.eye(K)
    M_r_1 = np.eye(K)
    for r in range(R):
        M_bar = (1+eta_MG) * W @ M_r - eta_MG * M_r_1
        M_r_1 = M_r
        M_r = M_bar

    # Smooth paramters
    kappa = 2000 if kappa is None else kappa
    mu = 2000 if mu is None else mu
    nl = num_points
    L = nl * kappa / 4 + mu

    # Mean of quantile
    q_mean = np.mean([np.quantile(list(np.array([local_scores[k]]).reshape(-1, )), 1-alpha_tilde) for k in range(K)])

    # Regularization parameter s0
    q0 = q_mean if q0 is None else q0

    # Quantile of quantile, this is too slow
    # QQ_M = calc_matrix_M(int(K), int(nl), .0, mid=False)
    # mm = list(np.ravel(QQ_M))
    # vv = min(i for i in mm if i > (1 - alpha))
    # kk = int(np.where(QQ_M == vv)[0])
    # ll = int(np.where(QQ_M == vv)[1])
    # if q0 is None:
    #     q0 = sorted(list([sorted(list(np.array([local_scores[k]]).reshape(-1, )))[ll-1] for k in range(K)]))[kk-1]
    # else:
    #     q0 = q0

    # ADMM parameters
    kappa_f = L/mu
    mu_ADMM = np.power((1 + (kappa_G ** 2)/(2*kappa_f**2) - (kappa_G/(2*kappa_f)*np.sqrt(((kappa_G ** 2)/(kappa_f ** 2)) + 4))), -1)
    c = 1/2 * (2*np.power(mu_ADMM, 1/2)*L) / (sigma_max_M * sigma_min_M)
    mat_G = np.block([[c * np.eye(E), np.zeros((E, E))], [np.zeros((E, E)), (1/c) * np.eye(E)]])
    delta = np.min([ (mu_ADMM-1)*(sigma_min_M ** 2)/(mu_ADMM * sigma_max_M ** 2), mu/((c/4)*(sigma_max_M**2) + (mu_ADMM/c)*(L**2)*np.power(sigma_min_M, -2)) ])

    # For TEST
    q_true = np.quantile(np.concatenate([v.flatten() for v in local_scores.values()]), 1-alpha_tilde)
    err = np.zeros(shape=(T, ))

    if opt_method == 'ADMM':
        X0 = np.zeros(shape=(K,))
        a_admm = np.zeros((K,))
        for t in range(T):
            X_t = X0.astype(np.float64)
            a_t = a_admm

            for k in range(K):
                N_k = list(G.neighbors(k))
                local_score = local_scores[k]
                temp_f = lambda x: (diff_smooth_pinball(x, alpha_tilde, kappa, mu, q0, local_score) + a_t[k] +
                                    2 * c * len(N_k) * x - c * (len(N_k)*X_t[k] + np.sum([X_t[j] for j in N_k])))
                try:
                    X0[k] = brentq(temp_f, -200, 200, xtol=1e-6, maxiter=1000)
                except Exception:
                    X0[k] = fsolve(temp_f, X_t[k])

            if qlevel is not None:
                if qlevel == 8:
                    X0 = X0.astype(np.int8)
                elif qlevel == 16:
                    X0 = X0.astype(np.float16)
                elif qlevel == 32:
                    X0 = X0.astype(np.float32)

            for k in range(K):
                N_k = list(G.neighbors(k))
                a_admm[k] = a_t[k] + c * (len(N_k) * X0[k] - np.sum([X0[j] for j in N_k]))

            err[t] = 10 * np.log10(1/K * np.linalg.norm(np.array([X0[k] - q_true for k in range(K)])) ** 2 / np.abs(q_true) ** 2)
            if np.linalg.norm(X0 - X_t) < 1e-6:
                # end_T = t
                break

        # # PLOT GADMM under pinball loss Convergence
        # plt.plot(err)
        # plt.ylabel('NMSE(dB)')
        # plt.show()

        beta = np.linalg.pinv(M_minus) @ a_admm
        u_star = np.block([np.mean(X0) * np.ones((E,)), beta])
        init_error = u_star.transpose() @ mat_G @ u_star
        epsilon_T = 1/K * 1 / mu * np.power(1/(1+delta), T-2) * init_error

        if epsilon_0 is None:
            # epsilon_0 = np.abs(q0 - np.mean(X0))
            epsilon_0 = 0.1    # 1e-3, 1e-4, 1e-5 for softmax, 0.1 for logits score
        else:
            epsilon_0 = epsilon_0

        tilde_epsilon_0 = np.sqrt((2 * N * np.log(2)) / (mu * kappa) + epsilon_0 ** 2)

        q_hat = np.mean(X0) + np.sqrt(epsilon_T) + tilde_epsilon_0
        X0 = X0 + np.sqrt(epsilon_T) + tilde_epsilon_0

    if opt_method == 'DSGD':
        # Mutiple-Gossip DGD iteration
        X0 = np.zeros(shape=(K, ))
        b2 = 55
        for t in range(T):
            # eta = b / (t + a)
            # Gradient
            g = np.zeros(shape=(K, ))

            # Local GD
            for k in range(K):
                local_score = local_scores[k]
                q = X0[k]
                g[k] = diff_smooth_pinball(q, alpha, kappa, mu, q0, local_score) - 1/K * (1-alpha)
                # g[k] = - np.mean((local_score > X0[k]) - alpha) - 1/K * (1-alpha)
            X0 = X0 - eta * g
            # X0 = X0 @ W
            if qlevel is not None:
                if qlevel == 8:
                    X0 = X0.astype(np.int8)
                elif qlevel == 16:
                    X0 = X0.astype(np.float16)
                elif qlevel == 32:
                    X0 = X0.astype(np.float32)
            X0 = M_bar @ X0

            err[t] = 10 * np.log10(1/K * np.linalg.norm(np.array([X0[k] - q_true for k in range(K)])) ** 2 / np.abs(q_true) ** 2)
            # b = np.max([b, np.mean([(g[k] - np.mean(g)) ** 2 for k in range(K)])])

        c = ((18 * L ** 2) / mu) + (12 * K * L)
        c2 = (c * eta ** 2 * tilde_beta ** 2 * b2) / ((1-tilde_beta)**2)
        # c2 = 0
        nabla_f0_2 = ((1/K * diff_smooth_pinball(0, alpha, kappa, mu, q0, np.array([local_scores[k] for k in range(K)]).reshape(-1, ))) ** 2)
        # epsilon_T2 = (1/(mu*L) + (2/(mu ** 2))) * nabla_f0_2 * np.exp(-mu * eta * T / 4) + (2/L + 4/mu)*c2
        epsilon_T2 = (1 / (mu ** 2)) * nabla_f0_2 * np.exp(-mu * eta * T / 4) + (2 / mu) * c2

        epsilon_0 = np.abs(q0 - np.mean(X0))
        epsilon_T = np.sqrt(epsilon_T2) + epsilon_0
        # print(epsilon_T)

        q_hat = np.mean(X0) + epsilon_T
        X0 = X0 + epsilon_T

        # PLOT Gossip GD under pinball loss Convergence
        plt.plot(err)
        plt.ylabel('NMSE(dB)')
        plt.show()

    return q_hat, X0


def get_decentralized_coverage_size_over_alphas(
        cal_scores,
        cal_targets,
        test_scores,
        test_targets,
        alphas,
        method="lac",
        allow_empty_sets=False,
        k_reg=1,
        lam_reg=0.01,
        decentral=False,
        client_index_map=None,
        precision=4,
        gossip=10,
        R=10,
        W=None,
        G=None,
        q0_list=None,
        epsilon_0=None,
        iid_flag=False,
        quantile_method: str = 'tdigest',
):
    n = test_targets.shape[0]
    coverage_results, size_results = {}, {}
    q1_coverage, q1_size = {}, {}
    q2_coverage, q2_size = {}, {}
    q3_coverage, q3_size = {}, {}
    q4_coverage, q4_size = {}, {}

    for alpha_idx, alpha in enumerate(alphas):
        if method == "lac":
            if decentral and client_index_map is not None:
                q0 = q0_list[alpha_idx] if q0_list is not None else None
                qhat, qhat_K = get_decentralized_quantile(
                    cal_scores,
                    cal_targets,
                    alpha=alpha,
                    method="lac",
                    client_index_map=client_index_map,
                    quantile_method=quantile_method,
                    gossip=gossip,
                    W=W,
                    R=R,
                    G=G,
                    q0=q0,
                    epsilon_0=epsilon_0,
                    iid_flag=iid_flag,
                )
            else:
                # For Split Centralized CP
                qhat = calibrate_lac(cal_scores, cal_targets, alpha=alpha)

            # if decentral:
            #     K = len(client_index_map)
            #     N = len(cal_scores)
            #
            #     qhat_index = np.random.randint(K, size=n)
            #
            #     qhat_n = qhat_K[qhat_index]
            #
            #     # elements_mask = quantized_test_scores >= (1 - qhat)
            #
            #     elements_mask = torch.zeros_like(test_scores)
            #     for i in range(n):
            #         elements_mask[i, :] = test_scores[i, :] >= (1 - qhat_n[i])
            #
            #     if not allow_empty_sets:
            #         elements_mask[torch.arange(n), test_scores.argmax(1)] = True
            #
            #     psets = elements_mask
            # else:
            psets = inference_lac(test_scores, qhat, allow_empty_sets=allow_empty_sets)

        elif method == "aps":
            if decentral and client_index_map is not None:
                qhat, _ = get_decentralized_quantile(
                    cal_scores,
                    cal_targets,
                    alpha=alpha,
                    method="aps",
                    client_index_map=client_index_map,
                    quantile_method=quantile_method,
                    gossip=gossip,
                    W=W,
                    R=R,
                    G=G,
                    iid_flag=iid_flag,
                )
            else:
                qhat = calibrate_aps(cal_scores, cal_targets, alpha=alpha)
            psets = inference_aps(test_scores, qhat, allow_empty_sets=allow_empty_sets)
        elif method == "raps":
            if decentral and client_index_map is not None:
                qhat, _ = get_decentralized_quantile(
                    cal_scores,
                    cal_targets,
                    alpha=alpha,
                    method="raps",
                    client_index_map=client_index_map,
                    k_reg=k_reg, lam_reg=lam_reg,
                    quantile_method=quantile_method,
                    gossip=gossip,
                    W=W,
                    R=R,
                    G=G,
                    iid_flag=iid_flag,
                )
            else:
                qhat = calibrate_raps(
                    cal_scores, cal_targets, alpha=alpha, k_reg=k_reg, lam_reg=lam_reg
                )
            psets = inference_raps(
                test_scores,
                qhat,
                allow_empty_sets=allow_empty_sets,
                k_reg=k_reg,
                lam_reg=lam_reg,
            )
        elif method == "naive_lac":
            psets = inference_lac(
                test_scores, 1 - alpha, allow_empty_sets=allow_empty_sets
            )
        elif method == "naive_aps":
            psets = inference_aps(
                test_scores, 1 - alpha, allow_empty_sets=allow_empty_sets
            )
        else:
            raise ValueError()

        t = round(1 - alpha, 2)

        coverage_results[t] = get_coverage(psets, test_targets, precision=precision)
        size_results[t] = get_size(psets)

        s = psets.sum(1).float()
        q1 = torch.quantile(s, 0.25, interpolation='higher')
        q2 = torch.quantile(s, 0.5, interpolation='higher')
        q3 = torch.quantile(s, 0.75, interpolation='higher')
        # m = torch.median(s)
        q1_index = torch.nonzero(s <= q1).flatten()
        q2_index = torch.nonzero((q1 < s) & (s <= q2)).flatten()
        q3_index = torch.nonzero((q2 < s) & (s <= q3)).flatten()
        q4_index = torch.nonzero(q3 < s).flatten()
        t = round(1 - alpha, 2)
        q1_coverage[t] = get_coverage(psets[q1_index], test_targets[q1_index])
        q2_coverage[t] = get_coverage(psets[q2_index], test_targets[q2_index])
        q3_coverage[t] = get_coverage(psets[q3_index], test_targets[q3_index])
        q4_coverage[t] = get_coverage(psets[q4_index], test_targets[q4_index])
        q1_size[t] = get_size(psets[q1_index])
        q2_size[t] = get_size(psets[q2_index])
        q3_size[t] = get_size(psets[q3_index])
        q4_size[t] = get_size(psets[q4_index])

    return dict(
        coverage=coverage_results, size=size_results,
        q1_coverage=q1_coverage, q1_size=q1_size,
        q2_coverage=q2_coverage, q2_size=q2_size,
        q3_coverage=q3_coverage, q3_size=q3_size,
        q4_coverage=q4_coverage, q4_size=q4_size,
    )


def get_decentralized_coverage_size_over_gossip(
        cal_scores,
        cal_targets,
        test_scores,
        test_targets,
        gossips,
        alpha=0.1,
        method="lac",
        allow_empty_sets=False,
        k_reg=1,
        lam_reg=0.01,
        decentral=False,
        client_index_map=None,
        precision=4,
        W=None,
        G=None,
        R=1,
        qlevels=None,
        kappa=None,
        q0=None,
        mu=None,
        epsilon_0=None,
        iid_flag=False,
        quantile_method: str = 'tdigest',
):
    n = test_targets.shape[0]
    coverage_results, size_results = {}, {}
    q1_coverage, q1_size = {}, {}
    q2_coverage, q2_size = {}, {}
    q3_coverage, q3_size = {}, {}
    q4_coverage, q4_size = {}, {}

    for ii, gossip in enumerate(gossips):
        gossip = int(gossip)
        if qlevels is not None:
            qlevel = qlevels[ii]
        else:
            qlevel = None
        if method == "lac":
            if decentral and client_index_map is not None:
                qhat, qhat_K = get_decentralized_quantile(
                    cal_scores,
                    cal_targets,
                    alpha=alpha,
                    method="lac",
                    client_index_map=client_index_map,
                    quantile_method=quantile_method,
                    gossip=gossip,
                    W=W,
                    R=R,
                    G=G,
                    q0=q0,
                    kappa=kappa,
                    mu=mu,
                    epsilon_0=epsilon_0,
                    qlevel=qlevel,
                    iid_flag=iid_flag,
                )
            else:
                # For Split Centralized CP
                qhat = calibrate_lac(cal_scores, cal_targets, alpha=alpha)

            # if decentral:
            #     K = len(client_index_map)
            #     N = len(cal_scores)
            #
            #     qhat_index = np.random.randint(K, size=n)
            #
            #     qhat_n = qhat_K[qhat_index]
            #
            #     # elements_mask = quantized_test_scores >= (1 - qhat)
            #
            #     elements_mask = torch.zeros_like(test_scores)
            #     for i in range(n):
            #         elements_mask[i, :] = test_scores[i, :] >= (1 - qhat_n[i])
            #
            #     if not allow_empty_sets:
            #         elements_mask[torch.arange(n), test_scores.argmax(1)] = True
            #
            #     psets = elements_mask
            # else:
            psets = inference_lac(test_scores, qhat, allow_empty_sets=allow_empty_sets)

        elif method == "aps":
            if decentral and client_index_map is not None:
                qhat, _ = get_decentralized_quantile(
                    cal_scores,
                    cal_targets,
                    alpha=alpha,
                    method="aps",
                    client_index_map=client_index_map,
                    quantile_method=quantile_method,
                    gossip=gossip,
                    W=W,
                    R=R,
                    G=G,
                    iid_flag=iid_flag,
                )
            else:
                qhat = calibrate_aps(cal_scores, cal_targets, alpha=alpha)
            psets = inference_aps(test_scores, qhat, allow_empty_sets=allow_empty_sets)
        elif method == "raps":
            if decentral and client_index_map is not None:
                qhat, _ = get_decentralized_quantile(
                    cal_scores,
                    cal_targets,
                    alpha=alpha,
                    method="raps",
                    client_index_map=client_index_map,
                    k_reg=k_reg, lam_reg=lam_reg,
                    quantile_method=quantile_method,
                    gossip=gossip,
                    W=W,
                    R=R,
                    G=G,
                    iid_flag=iid_flag,
                )
            else:
                qhat = calibrate_raps(
                    cal_scores, cal_targets, alpha=alpha, k_reg=k_reg, lam_reg=lam_reg
                )
            psets = inference_raps(
                test_scores,
                qhat,
                allow_empty_sets=allow_empty_sets,
                k_reg=k_reg,
                lam_reg=lam_reg,
            )
        elif method == "naive_lac":
            psets = inference_lac(
                test_scores, 1 - alpha, allow_empty_sets=allow_empty_sets
            )
        elif method == "naive_aps":
            psets = inference_aps(
                test_scores, 1 - alpha, allow_empty_sets=allow_empty_sets
            )
        else:
            raise ValueError()

        t = round(gossip, 1)

        coverage_results[t] = get_coverage(psets, test_targets, precision=precision)
        size_results[t] = get_size(psets)

        s = psets.sum(1).float()
        q1 = torch.quantile(s, 0.25, interpolation='higher')
        q2 = torch.quantile(s, 0.5, interpolation='higher')
        q3 = torch.quantile(s, 0.75, interpolation='higher')
        # m = torch.median(s)
        q1_index = torch.nonzero(s <= q1).flatten()
        q2_index = torch.nonzero((q1 < s) & (s <= q2)).flatten()
        q3_index = torch.nonzero((q2 < s) & (s <= q3)).flatten()
        q4_index = torch.nonzero(q3 < s).flatten()
        # t = round(1 - alpha, 2)
        q1_coverage[t] = get_coverage(psets[q1_index], test_targets[q1_index])
        q2_coverage[t] = get_coverage(psets[q2_index], test_targets[q2_index])
        q3_coverage[t] = get_coverage(psets[q3_index], test_targets[q3_index])
        q4_coverage[t] = get_coverage(psets[q4_index], test_targets[q4_index])
        q1_size[t] = get_size(psets[q1_index])
        q2_size[t] = get_size(psets[q2_index])
        q3_size[t] = get_size(psets[q3_index])
        q4_size[t] = get_size(psets[q4_index])

    return dict(
        coverage=coverage_results, size=size_results,
        q1_coverage=q1_coverage, q1_size=q1_size,
        q2_coverage=q2_coverage, q2_size=q2_size,
        q3_coverage=q3_coverage, q3_size=q3_size,
        q4_coverage=q4_coverage, q4_size=q4_size,
    )


def smooth_pinball(q, alpha, kappa, mu, q0, V_list):
    m = lambda x, kappa: x + 1/kappa * np.log(1 + np.maximum(np.minimum(np.exp(-kappa * x), 1e30), -1e30))
    rho = (1 - alpha) * np.sum(m(V_list - q, kappa)) + alpha * np.sum(m(q - V_list, kappa)) + (mu/2) * ((q - q0) ** 2)
    return rho

# def smooth_pinball(q, alpha, kappa, mu, q0, V_list):
#     def stable_log_exp(x, kappa):
#         # Use the log-sum-exp trick for numerical stability
#         if x < 0:
#             return -x + np.log1p(np.exp(kappa * x))
#         else:
#             return np.log1p(np.exp(-kappa * x))
#
#     m = lambda x, kappa: x + 1/kappa * np.vectorize(stable_log_exp)(-x, kappa)
#     rho = (1 - alpha) * np.sum(m(V_list - q, kappa)) + alpha * np.sum(m(q - V_list, kappa)) + (mu/2) * ((q - q0) ** 2)
#     return rho


def diff_smooth_pinball(q, alpha, kappa, mu, q0, V_list):
    dm = lambda x, k: 1 - 1 / (1 + np.minimum(np.exp(k * x), 1e30))
    drho = (1 - alpha) * -1 * np.sum(dm(V_list - q, kappa)) + alpha * np.sum(dm(q - V_list, kappa)) + mu * (q - q0)
    return drho
