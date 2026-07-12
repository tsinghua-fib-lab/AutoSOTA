import numpy as np
import torch

# ==================================================================================================================== #
#                                                  compute order                                                       #
# ==================================================================================================================== #
#   --- functions for ternary order
def compute_ternary_order_tau(base_rank, ref_rank, tau=1e-4):
    if base_rank - ref_rank > tau:
        order = 0
    elif abs(base_rank - ref_rank) <= tau:
        order = 2  ################################################## CHECK LATER!!!!
    elif base_rank - ref_rank < -tau:
        order = 1
    else:
        raise ValueError(f'order relation is wrong. (base,ref,tau): {base_rank, ref_rank, tau}')
    return order


def compute_ternary_order_fixed_base(base_rank, ref_ranks, tau=1e-4):
    # base score is fixed.
    num_scores = len(ref_ranks)
    orders = np.zeros((num_scores,), dtype=np.int32)
    for idx in range(num_scores):
        orders[idx] = compute_ternary_order_tau(base_rank, ref_ranks[idx], tau)
    return orders


def compute_ternary_order_fixed_ref(ref_rank, base_ranks, tau=1e-4):
    # ref score is fixed.
    num_scores = len(base_ranks)
    orders = np.zeros((num_scores,), dtype=np.int32)
    for idx in range(num_scores):
        orders[idx] = compute_ternary_order_tau(base_ranks[idx], ref_rank, tau)
    return orders


#   --- functions for binary order
def compute_binary_order(base_rank, ref_rank):
    if base_rank > ref_rank:
        order = 0
    elif base_rank < ref_rank:
        order = 1
    elif base_rank == ref_rank:
        order = 2
    else:
        raise ValueError(f'order relation is wrong. (base,ref,tau): {base_rank, ref_rank}')
    return order


def compute_binary_order_fixed_base(base_rank, ref_ranks):
    # base score is fixed.
    num_scores = len(ref_ranks)
    orders = np.zeros((num_scores,), dtype=np.int32)
    for idx in range(num_scores):
        orders[idx] = compute_binary_order(base_rank, ref_ranks[idx])
    return orders


def compute_binary_order_fixed_ref(ref_rank, base_ranks):
    # ref score is fixed.
    num_scores = len(base_ranks)
    orders = np.zeros((num_scores,), dtype=np.int32)
    for idx in range(num_scores):
        orders[idx] = compute_binary_order(base_ranks[idx], ref_rank)
    return orders

# ==================================================================================================================== #
#                                             estimation method (saaty, voting)                                        #
# ==================================================================================================================== #
def one_step_voting_ternary(orders, ranks, rank_levels, tau=1e-4):
    num_refs = len(orders)
    votes = np.zeros_like(rank_levels, dtype=np.int32)

    for idx in range(num_refs):
        # compute where to vote
        order = orders[idx]
        if order == 0:
            try:
                min_idx = np.argwhere(rank_levels > (ranks[idx] + tau))[0, 0]
            except:
                min_idx = -1
            max_idx = len(rank_levels)
        elif order == 2:          ############ CHECK LATER!!!!!
            try:
                min_idx = np.argwhere(rank_levels >= (ranks[idx] - tau))[0, 0]
            except:
                min_idx = -1

            max_idx = np.argwhere(rank_levels <= (ranks[idx] + tau))[-1, 0] + 1

        elif order == 1:  ############ CHECK LATER!!!!!
            min_idx = 0
            try:
                max_idx = np.argwhere(rank_levels < (ranks[idx] - tau))[-1, 0] + 1
            except:
                max_idx = 0
        else:
            raise ValueError(f'order value is out of range: {order}')

        # voting
        votes[min_idx:max_idx] += 1
    winners = np.argwhere(votes == np.amax(votes))
    # elected_index = winners[(len(winners)/2)][0]  # take the middle value when multiple winners exist.
    elected_index = winners[0][0]  # take the min value when multiple winners exist.
    return rank_levels[elected_index], votes


def one_step_voting_binary(orders, ranks, rank_levels, tau=0.0):
    num_refs = len(orders)
    votes = np.zeros_like(rank_levels, dtype=np.int32)
    votes_for_sum = np.zeros_like(rank_levels, dtype=np.float32)

    for idx in range(num_refs):
        # compute where to vote
        order = orders[idx]
        if order == 0:
            try:
                min_idx = np.argwhere(rank_levels >= (ranks[idx] + tau))[0, 0]
            except:
                min_idx = -1
            max_idx = len(rank_levels)

        elif order == 1:
            min_idx = 0
            try:
                max_idx = np.argwhere(rank_levels < (ranks[idx] - tau))[-1, 0] + 1
            except:
                max_idx = 0
        else:
            raise ValueError(f'order value is out of range: {order}')

        # voting
        votes[min_idx:max_idx] += 1
        # votes_for_sum[min_idx:max_idx] += 1/(len(rank_levels) - min_idx)
    winners = np.argwhere(votes == np.amax(votes))
    elected_index = winners[int((len(winners)/2))][0]  # take the middle value when multiple winners exist.
    # elected_index = winners[0][0]  # take the min value when multiple winners exist.
    return rank_levels[elected_index], votes


def soft_voting_ternary(probs, ranks, rank_levels, tau=1e-4):
    num_refs = len(probs)
    rank_levels = rank_levels.astype(np.float32)
    p_x = np.zeros_like(rank_levels)

    for i_ref, ref_score in enumerate(ranks):
        cond_p_per_levels = np.zeros((len(rank_levels), 3))
        cond_probs = _conditional_probs_uniform_ternary(ref_score, tau, rank_levels)
        order_per_levels = compute_ternary_order_fixed_ref(ref_score, rank_levels, tau)
        for i_level, order in enumerate(order_per_levels):
            if order == 1:
                cond_p_per_levels[i_level, order] = cond_probs[order]
            else:
                cond_p_per_levels[i_level, order] = cond_probs[order]
        p_x += np.matmul(cond_p_per_levels, probs[i_ref])
    p_x = p_x / num_refs
    max_idx = np.argmax(p_x)
    # plt.scatter(score_levels, p_x)
    # plt.xticks(score_levels)
    # plt.grid()

    # for binary classification : summation method
    low_scores = np.squeeze(np.argwhere(rank_levels < 5.0))
    high_scores = np.squeeze(np.argwhere(rank_levels >= 5.0))
    if np.sum(p_x[low_scores]) <= np.sum(p_x[high_scores]):
        pred_by_sum = 0
    else:
        pred_by_sum = 1

    return rank_levels[max_idx], np.sum(rank_levels * p_x), pred_by_sum


def soft_voting_binary(probs, ranks, rank_levels, tau=0.0):
    num_refs = len(probs)
    rank_levels = rank_levels.astype(np.float32)
    p_x = np.zeros_like(rank_levels)

    for i_ref, ref_score in enumerate(ranks):
        cond_p_per_levels = np.zeros((len(rank_levels), 2))
        cond_probs = _conditional_probs_uniform_binary(ref_score, rank_levels, tau=tau)
        order_per_levels = compute_binary_order_fixed_ref(ref_score, rank_levels)
        for i_level, order in enumerate(order_per_levels):
            cond_p_per_levels[i_level, order] = cond_probs[order]
        p_x += np.matmul(cond_p_per_levels, probs[i_ref])

    # normalize the sum of probs to be 1.0
    p_x = p_x / num_refs
    max_idx = np.argmax(p_x)
    # plt.scatter(score_levels, p_x)
    # plt.xticks(score_levels)
    # plt.grid()

    # for binary classification : summation method
    low_scores = np.squeeze(np.argwhere(rank_levels < 5.0))
    high_scores = np.squeeze(np.argwhere(rank_levels >= 5.0))
    if np.sum(p_x[low_scores]) <= np.sum(p_x[high_scores]):
        pred_by_sum = 0
    else:
        pred_by_sum = 1

    return rank_levels[max_idx], np.sum(rank_levels * p_x), pred_by_sum


def MAP_rule_binary(probs, ranks, rank_levels, tau=0.0):
    num_refs = len(probs)
    rank_levels = rank_levels.astype(np.float32)
    p_x = np.zeros_like(rank_levels)

    for i_ref, ref_score in enumerate(ranks):
        cond_p_per_levels = np.zeros((len(rank_levels), 2))
        cond_probs = _conditional_probs_uniform_binary(ref_score, rank_levels, tau=tau)
        for i_level, i_rank in enumerate(rank_levels):
            if ref_score == i_rank:
                cond_p_per_levels[i_level, 0] = cond_probs[0]*(1/2)
                cond_p_per_levels[i_level, 1] = cond_probs[1]*(1/2)
            elif ref_score > i_rank:
                cond_p_per_levels[i_level, 1] = cond_probs[1]
            elif ref_score < i_rank:
                cond_p_per_levels[i_level, 0] = cond_probs[0]

        p_x += np.matmul(cond_p_per_levels, probs[i_ref])

    # normalize the sum of probs to be 1.0
    p_x = p_x / num_refs
    winners = np.argwhere(p_x == np.amax(p_x))
    max_idx = winners[int((len(winners) / 2))][0]
    # max_idx = np.argmax(p_x)
    # plt.scatter(score_levels, p_x)
    # plt.xticks(score_levels)
    # plt.grid()
    #
    # # for binary classification : summation method
    # low_scores = np.squeeze(np.argwhere(rank_levels < 5.0))
    # high_scores = np.squeeze(np.argwhere(rank_levels >= 5.0))
    # if np.sum(p_x[low_scores]) <= np.sum(p_x[high_scores]):
    #     pred_by_sum = 0
    # else:
    #     pred_by_sum = 1

    # return rank_levels[max_idx], np.sum(rank_levels * p_x), pred_by_sum
    return rank_levels[max_idx], np.sum(rank_levels * p_x)

### for debug


def MC_and_MAP_rule(orders, probs, ranks, rank_levels, gt, tau=0.0, is_debug=False):
    # MC rule
    num_refs = len(orders)
    votes = np.zeros_like(rank_levels, dtype=np.int32)
    votes_for_sum = np.zeros_like(rank_levels, dtype=np.float32)

    for idx in range(num_refs):
        # compute where to vote
        order = orders[idx]
        if order == 0:
            try:
                min_idx = np.argwhere(rank_levels > (ranks[idx] + tau))[0, 0]
            except:
                min_idx = -1
            max_idx = len(rank_levels)

        elif order == 1:
            min_idx = 0
            try:
                max_idx = np.argwhere(rank_levels < (ranks[idx] - tau))[-1, 0] + 1
            except:
                max_idx = 0
        elif order == 2:
            min_idx = np.argwhere(rank_levels==ranks[idx])
            max_idx = np.argwhere(rank_levels==ranks[idx])
        else:
            raise ValueError(f'order value is out of range: {order}')

        # voting
        if order == 2:
            votes[min_idx] += 1
        else:
            votes[min_idx:max_idx] += 1
            # eq_idx = np.argwhere(rank_levels == ranks[idx]).flatten()[0]
            # votes[eq_idx] += 0.5
        # votes_for_sum[min_idx:max_idx] += 1/(len(rank_levels) - min_idx)
    winners = np.argwhere(votes == np.amax(votes))
    elected_index = winners[int((len(winners) / 2))][0]  # take the middle value when multiple winners exist.
    # elected_index = winners[0][0]  # take the min value when multiple winners exist.
    MC_estimation = rank_levels[elected_index]

    # MAP rule
    num_refs = len(probs)
    rank_levels = rank_levels.astype(np.float32)
    p_x = np.zeros_like(rank_levels)

    for i_ref, ref_score in enumerate(ranks):
        cond_p_per_levels = np.zeros((len(rank_levels), 2))
        cond_probs = _conditional_probs_uniform_binary(ref_score, rank_levels, tau=tau)
        for i_level, i_rank in enumerate(rank_levels):
            if ref_score == i_rank:
                cond_p_per_levels[i_level, 0] = cond_probs[0] * (1 / 2)
                cond_p_per_levels[i_level, 1] = cond_probs[1] * (1 / 2)
            elif ref_score > i_rank:
                cond_p_per_levels[i_level, 1] = cond_probs[1]
            elif ref_score < i_rank:
                cond_p_per_levels[i_level, 0] = cond_probs[0]

        p_x += np.matmul(cond_p_per_levels, probs[i_ref])

    # normalize the sum of probs to be 1.0
    p_x = p_x / num_refs
    winners = np.argwhere(p_x == np.amax(p_x))
    max_idx = winners[int((len(winners) / 2))][0]
    MAP_estimation = rank_levels[max_idx]

    # MAP_estimation = np.sum(rank_levels * p_x)

    # window_size = 5
    # window_idx = np.argmax(np.convolve(p_x, np.ones(window_size), 'valid'))
    # p_x_in_window = p_x[window_idx: window_idx+window_size]
    # p_x_in_window = p_x_in_window / np.sum(p_x_in_window)
    # ranks_in_window = rank_levels[window_idx: window_idx+window_size]
    # MAP_estimation2 = np.sum(ranks_in_window * p_x_in_window)

    if abs(abs(MAP_estimation-gt) - abs(MC_estimation-gt)) > 5 and is_debug:
        print(f'MAP:{MAP_estimation},  MC:{MC_estimation},  GT:{gt}')
        print(f'MAP estimation error : {abs(MAP_estimation - gt)}')
        print(f'MC estimation error : {abs(MC_estimation - gt)}')

    return MC_estimation, MAP_estimation


def _conditional_probs_uniform_ternary(ref_rank, rank_levels, tau=1e-4, assertion=False):
    n_high = len(np.argwhere(rank_levels > (ref_rank + tau)))
    n_similar = len(np.argwhere(np.logical_and((ref_rank - tau) <= rank_levels, rank_levels <= (ref_rank + tau))))
    n_low = len(np.argwhere(rank_levels < (ref_rank - tau)))

    if assertion:
        assert((n_high + n_similar + n_low) == len(rank_levels))

    cond_probs = np.zeros((3,))
    for idx, n_levels in enumerate([n_high, n_similar, n_low]):
        if n_levels < 1:   # to prevent dividing by zero
            continue
        cond_probs[idx] = 1/n_levels
    return cond_probs


def _conditional_probs_uniform_binary(ref_rank, rank_levels, tau=0.0):
    n_high = len(np.argwhere(rank_levels > (ref_rank + tau))) + 0.5
    n_low = len(np.argwhere(rank_levels < (ref_rank - tau))) + 0.5
    # assert((n_high + n_low) == len(rank_levels))

    cond_probs = np.zeros((2,))
    for idx, n_levels in enumerate([n_high, n_low]):
        # if n_levels < 1:
        #     continue
        cond_probs[idx] = 1/n_levels
    return cond_probs


def find_kNN(queries, samples, k=1, metric='L2'):
    """
    :param queries: BxNxC
    :param samples: BxMxC
    :param metric:
    :return:
    """
    if len(queries.shape) == 2:
        queries = queries.view(1, queries.shape[0], queries.shape[1])
    if len(samples.shape) == 2:
        samples = samples.view(1, samples.shape[0], samples.shape[1])

    if metric == 'L2':
        dist_mat = -torch.cdist(queries, samples)  # BxNxM

    elif metric == 'cosine':
        # queries = torch.nn.functional.normalize(queries, dim=-1)
        # samples = torch.nn.functional.normalize(samples, dim=-1)

        dist_mat = torch.matmul(queries, samples.transpose(2,1))

    vals, inds = torch.topk(dist_mat, k, dim=-1)
    return vals, inds
