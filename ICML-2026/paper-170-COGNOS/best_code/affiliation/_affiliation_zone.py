#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import multiprocessing
from functools import partial

from affiliation._integral_interval import interval_intersection

def t_start(j, Js = [(1,2),(3,4),(5,6)], Trange = (1,10)):
    """
    Helper for `E_gt_func`
    
    :param j: index from 0 to len(Js) (included) on which to get the start
    :param Js: ground truth events, as a list of couples
    :param Trange: range of the series where Js is included
    :return: generalized start such that the middle of t_start and t_stop 
    always gives the affiliation zone
    """
    b = max(Trange)
    n = len(Js)
    if j == n:
        return(2*b - t_stop(n-1, Js, Trange))
    else:
        return(Js[j][0])

def t_stop(j, Js = [(1,2),(3,4),(5,6)], Trange = (1,10)):
    """
    Helper for `E_gt_func`
    
    :param j: index from 0 to len(Js) (included) on which to get the stop
    :param Js: ground truth events, as a list of couples
    :param Trange: range of the series where Js is included
    :return: generalized stop such that the middle of t_start and t_stop 
    always gives the affiliation zone
    """
    if j == -1:
        a = min(Trange)
        return(2*a - t_start(0, Js, Trange))
    else:
        return(Js[j][1])

def E_gt_func(j, Js, Trange):
    """
    Get the affiliation zone of element j of the ground truth
    
    :param j: index from 0 to len(Js) (excluded) on which to get the zone
    :param Js: ground truth events, as a list of couples
    :param Trange: range of the series where Js is included, can 
    be (-math.inf, math.inf) for distance measures
    :return: affiliation zone of element j of the ground truth represented
    as a couple
    """
    range_left = (t_stop(j-1, Js, Trange) + t_start(j, Js, Trange))/2
    range_right = (t_stop(j, Js, Trange) + t_start(j+1, Js, Trange))/2
    return((range_left, range_right))

def get_all_E_gt_func(Js, Trange):
    """
    Get the affiliation partition from the ground truth point of view
    
    :param Js: ground truth events, as a list of couples
    :param Trange: range of the series where Js is included, can 
    be (-math.inf, math.inf) for distance measures
    :return: affiliation partition of the events
    """
    # E_gt is the limit of affiliation/attraction for each ground truth event
    E_gt = [E_gt_func(j, Js, Trange) for j in range(len(Js))]
    return(E_gt)


def _process_one_e_gt_zone_task(e_gt_item, Is_list):
    """
    Processes a single E_gt zone (e_gt_item) against all intervals in Is_list.
    This function mimics the inner loop logic of the original affiliation_partition.

    Note: The original code's filtering logic (discarded_idx_before, etc.)
    was effectively unused because `Is_j = [x for x, y in zip(Is, kept_index)]`
    doesn't actually filter `Is`. This parallel version faithfully reproduces that behavior.
    """

    # Original (redundant) filtering logic in the loop:
    # discarded_idx_before = [I[1] < e_gt_item[0] for I in Is_list]
    # discarded_idx_after = [I[0] > e_gt_item[1] for I in Is_list]
    # kept_index = [not(a or b) for a, b in zip(discarded_idx_before, discarded_idx_after)]
    # Is_j = [x for x, y in zip(Is_list, kept_index)] # This makes Is_j a copy of Is_list

    # So, we directly apply interval_intersection to all Is in Is_list
    # with the current e_gt_item.
    return [interval_intersection(I, e_gt_item) for I in Is_list]
#
#
# # ----------------------------------------------------
# # 并行化的 affiliation_partition 函数
# # ----------------------------------------------------
# def affiliation_partition(Is, E_gt, num_processes=None):
#     """
#     Cut the events into the affiliation zones in parallel.
#
#     :param Is: events as a list of couples
#     :param E_gt: range of the affiliation zones
#     :param num_processes: Number of processes to use. Defaults to CPU count.
#     :return: a list of list of intervals (each interval represented by either
#     a couple or None for empty interval). The outer list is indexed by each
#     affiliation zone of `E_gt`. The inner list is indexed by the events of `Is`.
#     """
#     if num_processes is None:
#         num_processes = 24
#
#     print(f"并行处理中，使用 {num_processes} 个进程...")
#
#     # 使用 functools.partial 预设 Is_list 参数
#     # 这样 _process_one_e_gt_zone_task 函数就只剩下一个需要 map 的参数 (e_gt_item)
#     worker_func = partial(_process_one_e_gt_zone_task, Is_list=Is)
#
#     with multiprocessing.Pool(processes=num_processes) as pool:
#         # pool.map 会将 E_gt 中的每个元素作为第一个参数传递给 worker_func，
#         # 并并行执行，然后收集结果。
#         results = pool.map(worker_func, E_gt)
#
#     return results


def affiliation_partition(Is = [(1,1.5),(2,5),(5,6),(8,9)], E_gt = [(1,2.5),(2.5,4.5),(4.5,10)]):
    """
    Cut the events into the affiliation zones
    The presentation given here is from the ground truth point of view,
    but it is also used in the reversed direction in the main function.

    :param Is: events as a list of couples
    :param E_gt: range of the affiliation zones
    :return: a list of list of intervals (each interval represented by either
    a couple or None for empty interval). The outer list is indexed by each
    affiliation zone of `E_gt`. The inner list is indexed by the events of `Is`.
    """
    out = [None] * len(E_gt)
    for j in range(len(E_gt)):
        E_gt_j = E_gt[j]
        discarded_idx_before = [I[1] < E_gt_j[0] for I in Is]  # end point of predicted I is before the begin of E
        discarded_idx_after = [I[0] > E_gt_j[1] for I in Is] # start of predicted I is after the end of E
        kept_index = [not(a or b) for a, b in zip(discarded_idx_before, discarded_idx_after)]
        Is_j = [x for x, y in zip(Is, kept_index)]
        out[j] = [interval_intersection(I, E_gt[j]) for I in Is_j]
    return(out)
