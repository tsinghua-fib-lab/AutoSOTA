from functools import partial
import numpy as np
import torch
from ddsketch import DDSketch
from tdigest import TDigest
import matplotlib.pyplot as plt


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
        quantization_M=1000,
        W=None,
        qlevel=None,
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

    N = cal_scores.shape[0]
    K = len(client_index_map)
    M = quantization_M

    # Quantization Set
    # Qset = np.linspace(0, 1, M + 1)
    Qset = np.linspace(torch.min(cal_scores), torch.max(cal_scores), M + 1)

    # Quantized score set of devices
    num_clients = len(client_index_map)
    local_scores = np.zeros(shape=(M + 1, num_clients))
    weight_list = np.zeros(shape=(K, ))

    for k, (client, index) in enumerate(client_index_map.items()):
        # Non-IID case
        scores = cal_scores[index]
        targets = cal_targets[index]

        assert len(scores), len(targets)

        # weight_list[k] = (len(scores) + 1)/ (N + K)

        q, score_dist = score_func(scores, targets, alpha=alpha, return_dist=True)

        # Quantization
        q_score_dist = quantize_array(score_dist, Qset)
        # Count occurrences of each quantization level
        counts = np.zeros(len(Qset))
        for i, level in enumerate(Qset):
            counts[i] = np.sum(q_score_dist == level)
        count_score_dist = counts

        # Store the fraction of scores
        # local_scores[:, k] = K * weight_list[k] * fraction_score_dist
        local_scores[:, k] = K/N * count_score_dist

    # Gossip
    X0 = local_scores
    # X_bar = np.mean(local_scores, 1)
    gamma = 1
    # err = np.zeros(shape=(gossip, ))
    for t in range(gossip):
        if qlevel is not None:
            if qlevel == 8:
                X0 = X0.astype(np.int8)
            elif qlevel == 16:
                X0 = X0.astype(np.float16)
            elif qlevel == 32:
                X0 = X0.astype(np.float32)
        local_scores = X0 + gamma * X0 @ (W - np.eye(num_clients))
        X0 = local_scores
        # err[t] = 10*np.log10(1/num_clients * np.sum([np.linalg.norm(X0[:, k] - X_bar) ** 2 for k in range(num_clients)]))

    # plt.plot(np.array(range(gossip)), err, linewidth=2)
    # plt.title('Gossip Convergence')
    # plt.xlabel('Gossip iteration')
    # plt.ylabel(r'$\frac{1}{K} \sum_{k\in[K]}\| x_k^{(t)} - \bar{x} \|^2$ (dB)')
    # plt.show()
    # plt.bar(np.array(range(quantization_M)), X_bar)
    # plt.title('Empirical Distribution of Quantized Scores')
    # plt.show()

    return local_scores


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
        quantization_M=1000,
        W=None,
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
                p = get_decentralized_quantile(
                    cal_scores,
                    cal_targets,
                    alpha=alpha,
                    method="lac",
                    client_index_map=client_index_map,
                    quantile_method=quantile_method,
                    gossip=gossip,
                    quantization_M=quantization_M,
                    W=W,
                )
            else:
                # For Split Centralized CP
                # qhat = calibrate_lac(cal_scores, cal_targets, alpha=alpha)

                # For Quantized Split Centralized CP
                K = len(client_index_map)
                N = len(cal_scores)
                M = quantization_M
                # Qset = np.linspace(0, 1, M + 1)
                Qset = np.linspace(torch.min(cal_scores), torch.max(cal_scores), M + 1)
                score_dist = torch.take_along_dim(1 - cal_scores, cal_targets.unsqueeze(1), 1).flatten()
                q_score_dist = quantize_array(score_dist, Qset)
                # Count occurrences of each quantization level
                counts = np.zeros(len(Qset))
                for i, level in enumerate(Qset):
                    counts[i] = np.sum(q_score_dist == level)
                fraction_score_dist = counts / len(q_score_dist)
                p_ = np.zeros(shape=(M + 1,))
                p_[-1] = 1
                p_bar_plus = N / (N + K) * fraction_score_dist + K / (N + K) * p_
                m_alpha = M
                for m in range(1, M + 1 + 1):
                    temp_sum = np.sum(p_bar_plus[: m])
                    if temp_sum >= 1 - alpha:
                        m_alpha = m - 1
                        break
                    else:
                        continue
                qhat = Qset[m_alpha]

            if decentral:
                K = len(client_index_map)
                N = len(cal_scores)
                M = quantization_M
                T = gossip
                alpha_tilde = 1 - np.ceil((1-alpha)*(N+K))/N

                # Qset = np.linspace(0, 1, M + 1)
                Qset = np.linspace(torch.min(cal_scores), torch.max(cal_scores), M + 1)
                quantized_test_scores = torch.from_numpy(quantize_array(test_scores, Qset))

                n_list = np.zeros(shape=(K, ))
                for k, (client, index) in enumerate(client_index_map.items()):
                    # Non-IID case
                    scores = cal_scores[index]
                    targets = cal_targets[index]

                    assert len(scores), len(targets)

                    n_list[k] = len(scores)

                eigenvalues = np.sort(np.abs(np.linalg.eigvals(W)))
                gamma = 1
                rho = 1 - np.abs(eigenvalues[-2])
                epsilon = ((K*np.sqrt(2*K*M) * np.power(1-gamma*rho, T))/N) * np.sqrt(np.max([n_list[k] ** 2 + np.mean(n_list ** 2) for k in range(K)]))

                qhat_K = torch.zeros(size=(K,))
                for k in range(K):
                    pk = p[:, k]
                    m_alpha = M
                    for m in range(1, M + 1 + 1):
                        temp_sum = np.sum(pk[: m])
                        if temp_sum >= 1 - alpha_tilde + epsilon:
                            m_alpha = m - 1
                            break
                        else:
                            continue
                    qhat_K[k] = Qset[m_alpha]

                qhat_index = np.random.randint(K, size=n)

                qhat_n = qhat_K[qhat_index]

                # elements_mask = quantized_test_scores >= (1 - qhat)

                elements_mask = torch.zeros_like(quantized_test_scores)
                for i in range(n):
                    elements_mask[i, :] = quantized_test_scores[i, :] >= (1 - qhat_n[i])

                if not allow_empty_sets:
                    elements_mask[torch.arange(n), quantized_test_scores.argmax(1)] = True

                psets = elements_mask
            else:
                psets = inference_lac(test_scores, qhat, allow_empty_sets=allow_empty_sets)

        elif method == "aps":
            qhat = calibrate_aps(cal_scores, cal_targets, alpha=alpha)
            psets = inference_aps(test_scores, qhat, allow_empty_sets=allow_empty_sets)
        elif method == "raps":
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
        quantization_M=1000,
        W=None,
        qlevels=None,
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
                p = get_decentralized_quantile(
                    cal_scores,
                    cal_targets,
                    alpha=alpha,
                    method="lac",
                    client_index_map=client_index_map,
                    quantile_method=quantile_method,
                    gossip=gossip,
                    quantization_M=quantization_M,
                    W=W,
                    qlevel=qlevel,
                )
            else:
                qhat = calibrate_lac(cal_scores, cal_targets, alpha=alpha)

            # psets = inference_lac(test_scores, qhat, allow_empty_sets=allow_empty_sets)

            if decentral:
                K = len(client_index_map)
                N = len(cal_scores)
                M = quantization_M
                T = gossip
                alpha_tilde = 1 - np.ceil((1-alpha)*(N+K))/N

                # Qset = np.linspace(0, 1, M + 1)
                Qset = np.linspace(torch.min(cal_scores), torch.max(cal_scores), M + 1)
                quantized_test_scores = torch.from_numpy(quantize_array(test_scores, Qset))

                n_list = np.zeros(shape=(K, ))
                for k, (client, index) in enumerate(client_index_map.items()):
                    # Non-IID case
                    scores = cal_scores[index]
                    targets = cal_targets[index]

                    assert len(scores), len(targets)

                    n_list[k] = len(scores)

                eigenvalues = np.sort(np.abs(np.linalg.eigvals(W)))
                gamma = 1
                rho = 1 - np.abs(eigenvalues[-2])
                epsilon = ((K*np.sqrt(2*K*M) * np.power(1-gamma*rho, T))/N) * np.sqrt(np.max([n_list[k] ** 2 + np.mean(n_list ** 2) for k in range(K)]))

                qhat_K = torch.zeros(size=(K,))
                for k in range(K):
                    pk = p[:, k]
                    m_alpha = M
                    for m in range(1, M + 1 + 1):
                        temp_sum = np.sum(pk[: m])
                        if temp_sum >= 1 - alpha_tilde + epsilon:
                            m_alpha = m - 1
                            break
                        else:
                            continue
                    qhat_K[k] = Qset[m_alpha]

                qhat_index = np.random.randint(K, size=n)

                qhat_n = qhat_K[qhat_index]

                # elements_mask = quantized_test_scores >= (1 - qhat)

                elements_mask = torch.zeros_like(quantized_test_scores)
                for i in range(n):
                    elements_mask[i, :] = quantized_test_scores[i, :] >= (1 - qhat_n[i])

                if not allow_empty_sets:
                    elements_mask[torch.arange(n), quantized_test_scores.argmax(1)] = True

                psets = elements_mask
            else:
                psets = inference_lac(test_scores, qhat, allow_empty_sets=allow_empty_sets)
        elif method == "aps":
            qhat = calibrate_aps(cal_scores, cal_targets, alpha=alpha)
            psets = inference_aps(test_scores, qhat, allow_empty_sets=allow_empty_sets)
        elif method == "raps":
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


def get_decentralized_coverage_size_over_quantization_level(
        cal_scores,
        cal_targets,
        test_scores,
        test_targets,
        gossip=100,
        alpha=0.1,
        method="lac",
        allow_empty_sets=False,
        k_reg=1,
        lam_reg=0.01,
        decentral=False,
        client_index_map=None,
        precision=4,
        quantization_Ms=None,
        W=None,
        quantile_method: str = 'tdigest',
):
    n = test_targets.shape[0]
    coverage_results, size_results = {}, {}
    q1_coverage, q1_size = {}, {}
    q2_coverage, q2_size = {}, {}
    q3_coverage, q3_size = {}, {}
    q4_coverage, q4_size = {}, {}

    for quantization_M in quantization_Ms:
        M = int(quantization_M)
        if method == "lac":
            if decentral and client_index_map is not None:
                p = get_decentralized_quantile(
                    cal_scores,
                    cal_targets,
                    alpha=alpha,
                    method="lac",
                    client_index_map=client_index_map,
                    quantile_method=quantile_method,
                    gossip=gossip,
                    quantization_M=M,
                    W=W,
                )
            else:
                qhat = calibrate_lac(cal_scores, cal_targets, alpha=alpha)

            if decentral:
                K = len(client_index_map)
                N = len(cal_scores)
                M = int(quantization_M)
                T = gossip
                alpha_tilde = 1 - np.ceil((1-alpha)*(N+K))/N

                # Qset = np.linspace(0, 1, int(M) + 1)
                Qset = np.linspace(torch.min(cal_scores), torch.max(cal_scores), M + 1)
                quantized_test_scores = torch.from_numpy(quantize_array(test_scores, Qset))

                n_list = np.zeros(shape=(K, ))
                for k, (client, index) in enumerate(client_index_map.items()):
                    # Non-IID case
                    scores = cal_scores[index]
                    targets = cal_targets[index]

                    assert len(scores), len(targets)

                    n_list[k] = len(scores)

                eigenvalues = np.sort(np.abs(np.linalg.eigvals(W)))
                gamma = 1
                rho = 1 - np.abs(eigenvalues[-2])
                epsilon = ((K*np.sqrt(2*K*M) * np.power(1-gamma*rho, T))/N) * np.sqrt(np.max([n_list[k] ** 2 + np.mean(n_list ** 2) for k in range(K)]))

                qhat_K = torch.zeros(size=(K,))
                for k in range(K):
                    pk = p[:, k]
                    m_alpha = M
                    for m in range(1, M + 1 + 1):
                        temp_sum = np.sum(pk[: m])
                        if temp_sum >= 1 - alpha_tilde + epsilon:
                            m_alpha = m - 1
                            break
                        else:
                            continue
                    qhat_K[k] = Qset[m_alpha]

                qhat_index = np.random.randint(K, size=n)

                qhat_n = qhat_K[qhat_index]

                # elements_mask = quantized_test_scores >= (1 - qhat)

                elements_mask = torch.zeros_like(quantized_test_scores)
                for i in range(n):
                    elements_mask[i, :] = quantized_test_scores[i, :] >= (1 - qhat_n[i])

                if not allow_empty_sets:
                    elements_mask[torch.arange(n), quantized_test_scores.argmax(1)] = True

                psets = elements_mask
            else:
                psets = inference_lac(test_scores, qhat, allow_empty_sets=allow_empty_sets)

        elif method == "aps":
            qhat = calibrate_aps(cal_scores, cal_targets, alpha=alpha)
            psets = inference_aps(test_scores, qhat, allow_empty_sets=allow_empty_sets)
        elif method == "raps":
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

        t = round(quantization_M, 1)

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



def quantize_array(s, levels):
    """
    Quantize the input array s according to the given quantization rule.

    Args:
    s (numpy.ndarray): Input array, values should be in the range [0, 1]
    levels (list): List of quantization levels S_1, S_2, ..., S_M

    Returns:
    numpy.ndarray: Quantized array
    """
    # Create a mask for the special case s in [S_0, S_1]
    mask_first = s <= levels[1]

    # Initialize the output array
    quantized = np.zeros_like(s)

    # Set values for the special case
    quantized[mask_first] = levels[1]

    # Quantize the remaining values
    for m in range(2, len(levels)):
        mask = (levels[m - 1] < s) & (s <= levels[m])
        quantized[mask] = levels[m]

    return quantized
