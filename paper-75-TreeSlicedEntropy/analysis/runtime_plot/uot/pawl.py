"""
This file contains a numba oriented implementation of the partial OT in 1D.
The main functions to be used outside this file are:

* `partial_ot_1d(x, y, max_iter, p=1)` where `x` and `y` are numpy arrays of 
  shape (n, ) and (m, ) and `max_iter` is the number of pairs one wants to 
  include in the solution and `p` indicates which partial Wasserstein distance 
  should be computed. Note however that all partial solutions of size
  lower than `max_iter` can be retrieved from the output of this function
  (see docs of the function for more details)
* `partial_ot_1d_elbow(x, y, p=1)` where `x` and `y` are numpy arrays of 
  shape (n, ) and (m, ). In this alternative implementation, the number of 
  pairs to be included in the solution is inferred using the elbow method on
  the series of costs of partial solutions (see docs of the function for 
  more details).
"""

import numpy as np
import heapq
from kneed import KneeLocator
import torch
from numba import njit
from numba.typed import Dict
from numba.core import types

int64 = types.int64

@njit(cache=True, fastmath=True)
def bisect_left(arr, x):
    """Similar to bisect.bisect_left(), from the built-in library."""
    M = len(arr)
    for i in range(M):
        if arr[i] >= x:
            return i
    return M

@njit(cache=True, fastmath=True)
def insert_new_chain(chains_starting_at, chains_ending_at, candidate_chain):
    """Insert the `candidate_chain` into the already known chains stored in 
    `chains_starting_at` and `chains_ending_at`.
    `chains_starting_at` and `chains_ending_at` are modified in-place and 
    the chain in which the candidate is inserted is returned.

    Examples
    --------
    >>> chains_starting_at = Dict()
    >>> chains_starting_at[2] = 4
    >>> chains_starting_at[8] = 9
    >>> chains_ending_at = Dict()
    >>> chains_ending_at[4] = 2
    >>> chains_ending_at[9] = 8
    >>> insert_new_chain(chains_starting_at, chains_ending_at, [5, 7])
    (2, 9)
    >>> chains_starting_at
    DictType[int64,int64]<iv=None>({2: 9})
    >>> 
    >>> chains_starting_at = Dict()
    >>> chains_starting_at[2] = 4
    >>> chains_starting_at[8] = 9
    >>> chains_ending_at = Dict()
    >>> chains_ending_at[4] = 2
    >>> chains_ending_at[9] = 8
    >>> insert_new_chain(chains_starting_at, chains_ending_at, [11, 12])
    (11, 12)
    >>> chains_starting_at
    DictType[int64,int64]<iv=None>({2: 4, 8: 9, 11: 12})
    >>> 
    >>> chains_starting_at = Dict()
    >>> chains_starting_at[2] = 4
    >>> chains_starting_at[8] = 9
    >>> chains_ending_at = Dict()
    >>> chains_ending_at[4] = 2
    >>> chains_ending_at[9] = 8
    >>> insert_new_chain(chains_starting_at, chains_ending_at, [5, 6])
    (2, 6)
    >>> chains_starting_at
    DictType[int64,int64]<iv=None>({2: 6, 8: 9})
    >>> 
    >>> chains_starting_at = Dict()
    >>> chains_starting_at[2] = 4
    >>> chains_starting_at[8] = 9
    >>> chains_ending_at = Dict()
    >>> chains_ending_at[4] = 2
    >>> chains_ending_at[9] = 8
    >>> insert_new_chain(chains_starting_at, chains_ending_at, [6, 7])
    (6, 9)
    >>> chains_starting_at
    DictType[int64,int64]<iv=None>({2: 4, 6: 9})
    """
    i, j = candidate_chain
    if i - 1  in chains_ending_at:
        i = chains_ending_at[i - 1]
    if j + 1 in chains_starting_at:
        j = chains_starting_at[j + 1]
    if i in chains_starting_at:
        del chains_ending_at[chains_starting_at[i]]
    chains_starting_at[i] = j
    if j in chains_ending_at:
        del chains_starting_at[chains_ending_at[j]]
    chains_ending_at[j] = i
    return i, j

@njit(cache=True, fastmath=True)
def compute_cost_for_chain(idx_start, idx_end, chain_costs_cumsum):
    """Compute the associated cost for a chain (set of contiguous points
    included in the solution) ranging from `idx_start` to `idx_end` (both included).
    """
    return chain_costs_cumsum[idx_end] - chain_costs_cumsum.get(idx_start - 1, 0)

@njit(cache=True, fastmath=True)
def precompute_chain_costs_cumsum(minimal_chain_ending_at, n):
    """For each position `i` at which a chain could end,
    Compute (using dynamic programming and the costs of minimal chains that have been precomputed)
    the cost of the largest chain ending at `i`.

    This is useful because this can be used later, to compute the cost of any chain in O(1)
    (cf. `compute_cost_for_chain`).
    """
    chain_costs_cumsum = Dict.empty(key_type=types.int64, value_type=types.float64)
    for i in range(n):
        # If there is a minimal chain ending there
        # (if not, this cannot be the end of a chain)
        if i in minimal_chain_ending_at:
            start, additional_cost = minimal_chain_ending_at[i]
            chain_costs_cumsum[i] = chain_costs_cumsum.get(start - 1, 0) + additional_cost
    return chain_costs_cumsum


@njit(cache=True, fastmath=True)
def get_cost_w1(diff_cum_sum, idx_start, idx_end):
    if idx_start == 0:
        return diff_cum_sum[idx_end] # - 0
    else:
        return diff_cum_sum[idx_end]   - diff_cum_sum[idx_start - 1]


@njit(cache=True, fastmath=True)
def get_cost_wp(sorted_z, sorted_distrib_indicator, idx_start, idx_end, p):
    """TODO

    Examples
    --------
    >>> sorted_z = np.array([1., 2., 3., 4., 5., 6., 11., 12.])
    >>> sorted_distrib_indicator = np.array([0, 0, 1, 1, 0, 0, 1, 1])
    >>> get_cost_wp(sorted_z, sorted_distrib_indicator, 0, 3, p=2)
    8.0
    >>> get_cost_wp(sorted_z, sorted_distrib_indicator, 4, 7, p=2)
    72.0
    """
    subset_z = sorted_z[idx_start:idx_end+1]
    subset_indicator = sorted_distrib_indicator[idx_start:idx_end+1]
    subset_x = subset_z[subset_indicator == 0]
    subset_y = subset_z[subset_indicator == 1]
    return np.sum(np.abs(subset_x - subset_y) ** p)


@njit(cache=True, fastmath=True)
def compute_costs(sorted_z, diff_cum_sum, diff_ranks, sorted_distrib_indicator, p=1):
    """For each element in sorted `z`, compute its minimal chain (cf note below).
    Then compute the cost for each minimal chain and sort all minimal chains in increasing 
    cost order.

    Note: the "minimal chain" of a point x_i in x is the minimal set of adjacent 
    points (starting at x_i and extending to the right) that one should 
    take to get a balanced set (ie. a set in which we have as many 
    elements from x as elements from y)

    Examples
    --------
    >>> x = np.array([1., 2., 5., 6.])
    >>> y = np.array([3., 4., 11., 12.])
    >>> ind_x, ind_y, indices_sort_xy, sorted_distrib_indicator = preprocess(x, y)
    >>> sorted_z = np.concatenate((x[ind_x], y[ind_y]))[indices_sort_xy]
    >>> diff_cum_sum = compute_cumulative_sum_differences(x, y, indices_sort_xy, sorted_distrib_indicator)
    >>> ranks_xy, diff_ranks = compute_rank_differences(indices_sort_xy, sorted_distrib_indicator)
    >>> costs, chain_costs_cumsum = compute_costs(sorted_z, diff_cum_sum, diff_ranks, sorted_distrib_indicator, p=1)
    >>> list(costs)
    [(1.0, 1, 2), (1.0, 3, 4), (5.0, 5, 6)]
    """
    l_costs = [(0.0, 0, 0) for _ in range(0)]
    minimal_chain_ending_at = {}
    last_pos_for_rank_x = Dict.empty(key_type=int64, value_type=int64)
    last_pos_for_rank_y = Dict.empty(key_type=int64, value_type=int64)
    n = len(diff_ranks)
    for idx_end in range(n):
        # For each item in either distrib, find the scope of the smallest
        # "minimal chain" that would end at that point and extend on the left, 
        # if one exists, and store the cost of this "minimal chain" by relying 
        # on differences of cumulative sums
        cur_rank = diff_ranks[idx_end]
        idx_start = -1
        if sorted_distrib_indicator[idx_end] == 0:
            target_rank = cur_rank - 1
            if target_rank in last_pos_for_rank_y:
                idx_start = last_pos_for_rank_y[target_rank]
            last_pos_for_rank_x[cur_rank] = idx_end
        else:
            target_rank = cur_rank + 1
            if target_rank in last_pos_for_rank_x:
                idx_start = last_pos_for_rank_x[target_rank]
            last_pos_for_rank_y[cur_rank] = idx_end
        if idx_start != -1:
            if p == 1:
                cost = get_cost_w1(diff_cum_sum, idx_start, idx_end)
            else:
                cost = get_cost_wp(sorted_z, sorted_distrib_indicator, idx_start, idx_end, p)
            if idx_end == idx_start + 1:
                heapq.heappush(l_costs, (abs(cost), idx_start, idx_end))
            assert idx_end not in minimal_chain_ending_at
            minimal_chain_ending_at[idx_end] = (idx_start, abs(cost))
    return l_costs, precompute_chain_costs_cumsum(minimal_chain_ending_at, n)

@njit(cache=True, fastmath=True)
def arg_insert_in_sorted(sorted_x, sorted_y):
    """Returns the index (similar to argsort) to be used to sort 
    the concatenation of `sorted_x` and `sorted_y` 
    (supposed to be 1d arrays).

    `sorted_x` and `sorted_y` are supposed to be sorted and their
    order cannot be changed in the resulting array (which is important
    for our ranking based algo in case of ex-aequos).

    Examples
    --------
    >>> x = np.array([-2., -1., 1.])
    >>> y = np.array([-3., 0., 1.5, 4.])
    >>> arg_insert_in_sorted(x, y)
    array([3, 0, 1, 4, 2, 5, 6])
    >>> x = np.array([-2., -1., -1.])
    >>> y = np.array([-3., 0., 1.5, 4.])
    >>> arg_insert_in_sorted(x, y)
    array([3, 0, 1, 2, 4, 5, 6])
    """
    n_x = sorted_x.shape[0]
    n_y = sorted_y.shape[0]
    arr_out = np.zeros((n_x + n_y), dtype=np.int64)
    idx_x = 0
    idx_y = 0
    while idx_x + idx_y < n_x + n_y:
        if idx_x == n_x:
            arr_out[idx_x + idx_y] = n_x + idx_y
            idx_y += 1
        elif idx_y == n_y:
            arr_out[idx_x + idx_y] = idx_x
            idx_x += 1
        else:
            if sorted_x[idx_x] <= sorted_y[idx_y]:
                arr_out[idx_x + idx_y] = idx_x
                idx_x += 1
            else:
                arr_out[idx_x + idx_y] = n_x + idx_y
                idx_y += 1
    return arr_out


@njit(cache=True, fastmath=True)
def preprocess(x, y):
    """Given two 1d distributions `x` and `y`:
    
    1. `indices_sort_x` sorts `x` (ie. `x[indices_sort_x]` is sorted) and
       `indices_sort_y` sorts `y` (ie. `y[indices_sort_y]` is sorted)
    
    2. stack them into a single distrib such that:
    
    * the new distrib is sorted with sort indices (wrt a stack of sorted x and sorted y) `indices_sort_xy`
    * `sorted_distrib_indicator` is a vector of zeros and ones where 0 means 
        "this point comes from x" and 1 means "this point comes from y"
    """
    indices_sort_x = np.argsort(x)
    indices_sort_y = np.argsort(y)
    indices_sort_xy = arg_insert_in_sorted(x[indices_sort_x], y[indices_sort_y])

    idx = np.concatenate((np.zeros(x.shape[0], dtype=np.int64), np.ones(y.shape[0], dtype=np.int64)))
    sorted_distrib_indicator = idx[indices_sort_xy]

    return indices_sort_x, indices_sort_y, indices_sort_xy, sorted_distrib_indicator

@njit(cache=True, fastmath=True)
def generate_solution_using_marginal_costs(costs, ranks_xy, chain_costs_cumsum, max_iter, sorted_distrib_indicator):
    """Generate a solution from a sorted list of minimal chain costs.
    See the note in `compute_costs` docs for a definition of minimal chains.

    The solution is a pair of lists. The first list contains the indices from `sorted_x`
    that are in the active set, and the second one contains the indices from `sorted_y`
    that are in the active set. 
    The third returned element is a list of marginal costs induced by each step:
    `list_marginal_costs[i]` is the marginal cost induced by the `i`-th step of the algorithm, such
    that `np.cumsum(list_marginal_costs[:i])` gives all intermediate costs up to step `i`.

    Examples
    --------
    >>> x = np.array([1., 2., 5., 6.])
    >>> y = np.array([3., 4., 11., 12.])
    >>> ind_x, ind_y, indices_sort_xy, sorted_distrib_indicator = preprocess(x, y)
    >>> sorted_z = np.concatenate((x[ind_x], y[ind_y]))[indices_sort_xy]
    >>> diff_cum_sum = compute_cumulative_sum_differences(x, y, indices_sort_xy, sorted_distrib_indicator)
    >>> ranks_xy, diff_ranks = compute_rank_differences(indices_sort_xy, sorted_distrib_indicator)
    >>> costs, chain_costs_cumsum = compute_costs(sorted_z, diff_cum_sum, diff_ranks, sorted_distrib_indicator, p=1)
    >>> generate_solution_using_marginal_costs(costs, ranks_xy, chain_costs_cumsum, 3, sorted_distrib_indicator)
    (array([1, 2, 3]), array([0, 1, 2]), [1.0, 1.0, 5.0])
    """
    active_set = set()
    chains_starting_at = Dict.empty(key_type=int64, value_type=int64)
    chains_ending_at = Dict.empty(key_type=int64, value_type=int64)
    list_marginal_costs = []
    list_active_set_inserts = []
    n = len(sorted_distrib_indicator)
    while len(costs) > 0 and max_iter > len(active_set) // 2:
        c, i, j = heapq.heappop(costs)
        if i in active_set or j in active_set:
            continue
        new_chain = None
        # Case 1: j == i + 1 => "Simple" insert
        if j == i + 1:
            new_chain = insert_new_chain(chains_starting_at, chains_ending_at, [i, j])
        # Case 2: insert a chain that contains a chain
        elif j - 1 in chains_ending_at:
            del chains_starting_at[i + 1]
            del chains_ending_at[j - 1]
            new_chain = insert_new_chain(chains_starting_at, chains_ending_at, [i, j])
        # There should be no "Case 3"
        else:
            raise ValueError
        active_set.update({i, j})
        list_active_set_inserts.append((i, j))
        list_marginal_costs.append(c)
        
        # We now need to update the candidate chains wrt the chain we have just created
        p_s, p_e = new_chain
        if p_s == 0 or p_e == n - 1:
            continue
        if sorted_distrib_indicator[p_s - 1] != sorted_distrib_indicator[p_e + 1]:
            # Insert (p_s - 1, p_e + 1) as a new candidate chain with marginal cost
            marginal_cost = (compute_cost_for_chain(p_s - 1, p_e + 1, chain_costs_cumsum)
                             - compute_cost_for_chain(p_s, p_e, chain_costs_cumsum))
            heapq.heappush(costs, (marginal_cost, p_s - 1, p_e + 1))

    # Generate index arrays in the order of insertion in the active set
    indices_sorted_x = np.array([ranks_xy[i] 
                                 if sorted_distrib_indicator[i] == 0 else ranks_xy[j] 
                                 for i, j in list_active_set_inserts])
    indices_sorted_y = np.array([ranks_xy[i] 
                                 if sorted_distrib_indicator[i] == 1 else ranks_xy[j] 
                                 for i, j in list_active_set_inserts])

    return (indices_sorted_x, indices_sorted_y, list_marginal_costs)

@njit(cache=True, fastmath=True)
def compute_cumulative_sum_differences(x_sorted, y_sorted, indices_sort_xy, sorted_distrib_indicator):
    """Computes difference between cumulative sums for both distribs.

    The cumulative sum vector for a sorted x is:

        cumsum_x = [x_0, x_0 + x_1, ..., x_0 + ... + x_n]

    This vector is then extend to reach a length of 2*n by repeating 
    values at places that correspond to an y item.
    In other words, if the order of x and y elements on the real 
    line is something like x-y-y-x..., then the extended vector is 
    (note the repetitions):

        cumsum_x = [x_0, x_0, x_0, x_0 + x_1, ..., x_0 + ... + x_n]

    Overall, this function returns `cumsum_x - cumsum_y` where `cumsum_x`
    and `cumsum_y` are the extended versions.

    Examples
    --------
    >>> x = np.array([-1., 3., 4.])
    >>> y = np.array([1., 2., 5.])
    >>> _, _, indices_sort_xy, sorted_distrib_indicator = preprocess(x, y)
    >>> sorted_distrib_indicator
    array([0, 1, 1, 0, 0, 1])
    >>> compute_cumulative_sum_differences(x, y, indices_sort_xy, sorted_distrib_indicator)  # [-1, -1, -1, 2, 6, 6] - [0, 1, 3, 3, 3, 8]
    array([-1., -2., -4., -1.,  3., -2.])
    """
    cum_sum_xs = np.cumsum(x_sorted)
    cum_sum_ys = np.cumsum(y_sorted)
    cum_sum = np.concatenate((cum_sum_xs, cum_sum_ys))
    cum_sum = cum_sum[indices_sort_xy]

    cum_sum_x = _insert_constant_values_float(cum_sum, 0, sorted_distrib_indicator)
    cum_sum_y = _insert_constant_values_float(cum_sum, 1, sorted_distrib_indicator)

    return cum_sum_x - cum_sum_y

@njit(cache=True, fastmath=True)
def insert_constant_values_int(arr, distrib_index, sorted_distrib_indicator):
    """Takes `arr` as input. For each position `i` in `arr`, 
    if `self.sorted_distrib_indicator[i]==distrib_index`,
    the value from `arr` is copied, 
    otherwise, the previous value that was copied from `arr` is repeated.

    Examples
    --------
    >>> arr = np.array([1, -1, -1, 2, 3, -1])
    >>> sorted_distrib_indicator = np.array([0, 1, 1, 0, 0, 1])
    >>> insert_constant_values_int(arr, 0, sorted_distrib_indicator)
    array([1, 1, 1, 2, 3, 3])
    """
    arr_insert = np.copy(arr)
    for i in range(len(arr)):
        if sorted_distrib_indicator[i]!=distrib_index:
            if i == 0:
                arr_insert[i] = 0
            else:
                arr_insert[i] = arr_insert[i-1]
    return arr_insert

@njit(cache=True, fastmath=True)
def _insert_constant_values_float(arr, distrib_index, sorted_distrib_indicator):
    """Takes `arr` as input. For each position `i` in `arr`, 
    if `self.sorted_distrib_indicator[i]==distrib_index`,
    the value from `arr` is copied, 
    otherwise, the previous value that was copied from `arr` is repeated.

    Examples
    --------
    >>> arr = np.array([1., -1., -1., 2., 3., -1.])
    >>> sorted_distrib_indicator = np.array([0, 1, 1, 0, 0, 1])
    >>> insert_constant_values_int(arr, 0, sorted_distrib_indicator)
    array([1., 1., 1., 2., 3., 3.])
    """
    arr_insert = np.copy(arr)
    for i in range(len(arr)):
        if sorted_distrib_indicator[i]!=distrib_index:
            if i == 0:
                arr_insert[i] = 0.
            else:
                arr_insert[i] = arr_insert[i-1]
    return arr_insert


@njit(cache=True, fastmath=True)
def compute_rank_differences(indices_sort_xy, sorted_distrib_indicator):
    """Precompute important rank-related quantities for better minimal chain extraction.
    
    Two quantities are returned:

    * `ranks_xy` is an array that gathers ranks of the elements in 
        their original distrib, eg. if the distrib indicator is
        [0, 1, 1, 0, 0, 1], then `rank_xy` will be:
        [0, 0, 1, 1, 2, 2]
    * `diff_ranks` is computed from `ranks_xy_x_cum` and `ranks_xy_y_cum`.
        For the example above, we would have:
        
        ranks_xy_x_cum = [1, 1, 1, 2, 3, 3]
        ranks_xy_y_cum = [0, 1, 2, 2, 2, 3]

        And `diff_ranks` is just `ranks_xy_x_cum - ranks_xy_y_cum`.

    Examples
    --------
    >>> x = np.array([-2., 2., 3.])
    >>> y = np.array([-1., 1., 5.])
    >>> _, _, indices_sort_xy, sorted_distrib_indicator = preprocess(x, y)
    >>> ranks_xy, diff_ranks = compute_rank_differences(indices_sort_xy, sorted_distrib_indicator)
    >>> ranks_xy
    array([0, 0, 1, 1, 2, 2])
    >>> diff_ranks
    array([ 1,  0, -1,  0,  1,  0])
    """
    n_x = np.sum(sorted_distrib_indicator == 0)
    n_y = np.sum(sorted_distrib_indicator == 1)
    ranks_x, ranks_y = np.arange(n_x), np.arange(n_y)
    ranks_xy = np.concatenate((ranks_x, ranks_y))
    ranks_xy = ranks_xy[indices_sort_xy]

    ranks_xy_x = ranks_xy.copy()
    ranks_xy_x[sorted_distrib_indicator==1] = 0
    ranks_xy_x[sorted_distrib_indicator==0] += 1
    ranks_xy_x_cum = insert_constant_values_int(ranks_xy_x, 0, sorted_distrib_indicator)

    ranks_xy_y = ranks_xy.copy()
    ranks_xy_y[sorted_distrib_indicator==0] = 0
    ranks_xy_y[sorted_distrib_indicator==1] += 1
    ranks_xy_y_cum = insert_constant_values_int(ranks_xy_y, 1, sorted_distrib_indicator)

    diff_ranks = ranks_xy_x_cum - ranks_xy_y_cum

    return ranks_xy, diff_ranks

@njit(cache=True, fastmath=True)
def partial_ot_1d(x, y, max_iter, p=1):
    """Main routine for the partial OT problem in 1D.
    
    Does:
    
    1. Preprocessing of the distribs (sorted & co)
    2. Precomputations (ranks, cumulative sums)
    3. Extraction of minimal chains
    4. Generate and return solution

    Note that the indices in `indices_x` and `indices_y` are ordered wrt their order of
    appearance in the solution such that `indices_x[:10]` (resp y) is the set of indices
    from x (resp. y) for the partial problem of size 10.

    Arguments
    ---------
    x : np.ndarray of shape (n, )
        First distrib to be considered (weights are considered uniform)
    y : np.ndarray of shape (m, )
        Second distrib to be considered (weights are considered uniform)
    max_iter : int
        Number of iterations of the algorithm, which is equal to the number of pairs
        in the returned solution.
    p : int (default: 1)
        Order of the partial Wasserstein distance to be computed (p-Wasserstein, or $W_p^p$)

    Returns
    -------
    indices_x : np.ndarray of shape (min(n, m, max_iter), )
        Indices of elements from the x distribution to be included in the partial solutions
        Order of appearance in this array indicates order of inclusion in the solution
    indices_y : np.ndarray of shape (min(n, m, max_iter), )
        Indices of elements from the x distribution to be included in the partial solutions
        Order of appearance in this array indicates order of inclusion in the solution
    list_marginal_costs : list of length min(n, m, max_iter)
        List of marginal costs associated to the intermediate partial problems
        `np.cumsum(list_marginal_costs)` gives the corresponding total costs for intermediate partial problems

    Examples
    --------
    >>> x = np.array([5., -2., 4.])
    >>> y = np.array([-1., 1., 3.])
    >>> partial_ot_1d(x, y, max_iter=2)
    (array([1, 2]), array([0, 2]), [1.0, 1.0])
    """
    # Sort distribs and keep track of their original indices
    indices_sort_x, indices_sort_y, indices_sort_xy, sorted_distrib_indicator = preprocess(x, y)
    sorted_z = np.concatenate((x[indices_sort_x], y[indices_sort_y]))[indices_sort_xy]

    # Precompute useful quantities
    diff_cum_sum = compute_cumulative_sum_differences(x[indices_sort_x], 
                                                      y[indices_sort_y], 
                                                      indices_sort_xy,
                                                      sorted_distrib_indicator)
    ranks_xy, diff_ranks = compute_rank_differences(indices_sort_xy, sorted_distrib_indicator)

    # Compute costs for "minimal chains"
    costs, chain_costs_cumsum = compute_costs(sorted_z, diff_cum_sum, diff_ranks, sorted_distrib_indicator, p=p)

    # Generate solution from sorted costs
    sol_indices_x_sorted, sol_indices_y_sorted, sol_costs = generate_solution_using_marginal_costs(costs, 
                                                                                                   ranks_xy, 
                                                                                                   chain_costs_cumsum, 
                                                                                                   max_iter, 
                                                                                                   sorted_distrib_indicator)

    # Convert back into indices in original `x` and `y` distribs
    indices_x = indices_sort_x[sol_indices_x_sorted]
    indices_y = indices_sort_y[sol_indices_y_sorted]
    marginal_costs = sol_costs
    return indices_x, indices_y, marginal_costs

def random_slice(nproj, dn):
    theta=torch.randn((nproj,dn))
    theta=torch.stack([th/torch.sqrt((th**2).sum()) for th in theta]).cpu()
    return theta

def pawl(X,Y,n_proj,k=-1):
    n,dn=X.shape
    m,_=Y.shape
    theta=random_slice(n_proj, dn).T
    if k==-1:
        k = min(n, m)
    min_cost = np.inf
    X_line = torch.matmul(X, theta)
    Y_line = torch.matmul(Y, theta)
    cost = torch.zeros(n_proj)
    for i in range(theta.shape[1]):
        x_proj, y_proj = X_line[:,i].detach().numpy(), Y_line[:,i].detach().numpy()
        x_proj, y_proj = np.array(x_proj, dtype=np.float64), np.array(y_proj, dtype=np.float64)
        ind_x, ind_y, _ = partial_ot_1d(x_proj, y_proj, k)
        sorted_ind_x, sorted_ind_y = np.sort(ind_x), np.sort(ind_y)
        X_s, Y_s = X[sorted_ind_x], Y[sorted_ind_y]
        cost[i] = torch.mean(torch.abs(X_s - Y_s))
    return torch.mean(cost) / k

def __main__():
    np.random.seed(0)
    x = torch.randn(30, 2)
    y = torch.randn(40, 2)
    print(pawl(x, y, 10))

if __name__ == "__main__":
    __main__()