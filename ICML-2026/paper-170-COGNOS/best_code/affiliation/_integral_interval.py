#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math
from affiliation.generics import _sum_wo_nan
"""
In order to shorten the length of the variables,
the general convention in this file is to let:
    - I for a predicted event (start, stop),
    - Is for a list of predicted events,
    - J for a ground truth event,
    - Js for a list of ground truth events.
"""

def interval_length(J = (1,2)):
    """
    Length of an interval
    
    :param J: couple representating the start and stop of an interval, or None
    :return: length of the interval, and 0 for a None interval
    """
    if J is None:
        return(0)
    return(J[1] - J[0])

def sum_interval_lengths(Is = [(1,2),(3,4),(5,6)]):
    """
    Sum of length of the intervals
    
    :param Is: list of intervals represented by starts and stops
    :return: sum of the interval length
    """
    return(sum([interval_length(I) for I in Is]))

def interval_intersection(I = (1, 3), J = (2, 4)): 
    """
    Intersection between two intervals I and J
    I and J should be either empty or represent a positive interval (no point)
    
    :param I: an interval represented by start and stop
    :param J: a second interval of the same form
    :return: an interval representing the start and stop of the intersection (or None if empty)
    """
    if I is None:
        return(None)
    if J is None:
        return(None)
        
    I_inter_J = (max(I[0], J[0]), min(I[1], J[1]))
    if I_inter_J[0] >= I_inter_J[1]:
        return(None)
    else:
        return(I_inter_J)

def interval_subset(I = (1, 3), J = (0, 6)):
    """
    Checks whether I is a subset of J
    
    :param I: an non empty interval represented by start and stop
    :param J: a second non empty interval of the same form
    :return: True if I is a subset of J
    """
    if (I[0] >= J[0]) and (I[1] <= J[1]):
        return True
    else:
        return False

def cut_into_three_func(I, J):
    """
    Cut an interval I into a partition of 3 subsets:
        the elements before J,
        the elements belonging to J,
        and the elements after J
    
    :param I: an interval represented by start and stop, or None for an empty one
    :param J: a non empty interval
    :return: a triplet of three intervals, each represented by either (start, stop) or None
    """
    if I is None:
        return((None, None, None))
    
    I_inter_J = interval_intersection(I, J)
    if I == I_inter_J:
        I_before = None
        I_after = None
    elif I[1] <= J[0]:
        I_before = I
        I_after = None
    elif I[0] >= J[1]:
        I_before = None
        I_after = I
    elif (I[0] <= J[0]) and (I[1] >= J[1]):
        I_before = (I[0], I_inter_J[0])
        I_after = (I_inter_J[1], I[1])
    elif I[0] <= J[0]:
        I_before = (I[0], I_inter_J[0])
        I_after = None
    elif I[1] >= J[1]:
        I_before = None
        I_after = (I_inter_J[1], I[1])
    else:
        raise ValueError('unexpected unconsidered case')
    return(I_before, I_inter_J, I_after)
  
def get_pivot_j(I, J):
    """
    Get the single point of J that is the closest to I, called 'pivot' here,
    with the requirement that I should be outside J
    
    :param I: a non empty interval (start, stop)
    :param J: another non empty interval, with empty intersection with I
    :return: the element j of J that is the closest to I
    """
    if interval_intersection(I, J) is not None:
        raise ValueError('I and J should have a void intersection')

    j_pivot = None # j_pivot is a border of J
    if max(I) <= min(J):
        j_pivot = min(J)
    elif min(I) >= max(J):
        j_pivot = max(J)
    else:
        raise ValueError('I should be outside J')
    return(j_pivot)

def integral_mini_interval(I, J):
    """
    In the specific case where interval I is located outside J,
    integral of distance from x to J over the interval x \in I.
    This is the *integral* i.e. the sum.
    It's not the mean (not divided by the length of I yet)
    
    :param I: a interval (start, stop), or None
    :param J: a non empty interval, with empty intersection with I
    :return: the integral of distances d(x, J) over x \in I
    """
    if I is None:
        return(0)

    j_pivot = get_pivot_j(I, J)
    a = min(I)
    b = max(I)
    return((b-a)*abs((j_pivot - (a+b)/2)))

def integral_interval_distance(I, J):
    """
    For any non empty intervals I, J, compute the
    integral of distance from x to J over the interval x \in I.
    This is the *integral* i.e. the sum. 
    It's not the mean (not divided by the length of I yet)
    The interval I can intersect J or not
    
    :param I: a interval (start, stop), or None
    :param J: a non empty interval
    :return: the integral of distances d(x, J) over x \in I
    """
    # I and J are single intervals (not generic sets)
    # I is a predicted interval in the range of affiliation of J
    
    def f(I_cut):
        return(integral_mini_interval(I_cut, J))
    # If I_middle is fully included into J, it is
    # the distance to J is always 0
    def f0(I_middle):
        return(0)

    cut_into_three = cut_into_three_func(I, J)
    # Distance for now, not the mean:
    # Distance left: Between cut_into_three[0] and the point min(J)
    d_left = f(cut_into_three[0])
    # Distance middle: Between cut_into_three[1] = I inter J, and J
    d_middle = f0(cut_into_three[1])
    # Distance right: Between cut_into_three[2] and the point max(J)
    d_right = f(cut_into_three[2])
    # It's an integral so summable
    return(d_left + d_middle + d_right)

def integral_mini_interval_P_CDFmethod__min_piece(I, J, E):
    """
    Helper of `integral_mini_interval_Pprecision_CDFmethod`
    In the specific case where interval I is located outside J,
    compute the integral $\int_{d_min}^{d_max} \min(m, x) dx$, with:
    - m the smallest distance from J to E,
    - d_min the smallest distance d(x, J) from x \in I to J
    - d_max the largest distance d(x, J) from x \in I to J
    
    :param I: a single predicted interval, a non empty interval (start, stop)
    :param J: ground truth interval, a non empty interval, with empty intersection with I
    :param E: the affiliation/influence zone for J, represented as a couple (start, stop)
    :return: the integral $\int_{d_min}^{d_max} \min(m, x) dx$
    """
    if interval_intersection(I, J) is not None:
        raise ValueError('I and J should have a void intersection')
    if not interval_subset(J, E):
        raise ValueError('J should be included in E')
    if not interval_subset(I, E):
        raise ValueError('I should be included in E')

    e_min = min(E)
    j_min = min(J)
    j_max = max(J)
    e_max = max(E)
    i_min = min(I)
    i_max = max(I)
  
    d_min = max(i_min - j_max, j_min - i_max)
    d_max = max(i_max - j_max, j_min - i_min)
    m = min(j_min - e_min, e_max - j_max)
    A = min(d_max, m)**2 - min(d_min, m)**2
    B = max(d_max, m) - max(d_min, m)
    C = (1/2)*A + m*B
    return(C)

def integral_mini_interval_Pprecision_CDFmethod(I, J, E):
    """
    Integral of the probability of distances over the interval I.
    In the specific case where interval I is located outside J,
    compute the integral $\int_{x \in I} Fbar(dist(x,J)) dx$.
    This is the *integral* i.e. the sum (not the mean)
    
    :param I: a single predicted interval, a non empty interval (start, stop)
    :param J: ground truth interval, a non empty interval, with empty intersection with I
    :param E: the affiliation/influence zone for J, represented as a couple (start, stop)
    :return: the integral $\int_{x \in I} Fbar(dist(x,J)) dx$
    """
    integral_min_piece = integral_mini_interval_P_CDFmethod__min_piece(I, J, E)
  
    e_min = min(E)
    j_min = min(J)
    j_max = max(J)
    e_max = max(E)
    i_min = min(I)
    i_max = max(I)
    d_min = max(i_min - j_max, j_min - i_max)
    d_max = max(i_max - j_max, j_min - i_min)
    integral_linear_piece = (1/2)*(d_max**2 - d_min**2)
    integral_remaining_piece = (j_max - j_min)*(i_max - i_min)
    
    DeltaI = i_max - i_min
    DeltaE = e_max - e_min
    
    output = DeltaI - (1/DeltaE)*(integral_min_piece + integral_linear_piece + integral_remaining_piece)
    return(output)

def integral_interval_probaCDF_precision(I, J, E):
    """
    Integral of the probability of distances over the interval I.
    Compute the integral $\int_{x \in I} Fbar(dist(x,J)) dx$.
    This is the *integral* i.e. the sum (not the mean)
    
    :param I: a single (non empty) predicted interval in the zone of affiliation of J
    :param J: ground truth interval
    :param E: affiliation/influence zone for J
    :return: the integral $\int_{x \in I} Fbar(dist(x,J)) dx$
    """
    # I and J are single intervals (not generic sets)
    def f(I_cut):
        if I_cut is None:
            return(0)
        else:
            return(integral_mini_interval_Pprecision_CDFmethod(I_cut, J, E))
            
    # If I_middle is fully included into J, it is
    # integral of 1 on the interval I_middle, so it's |I_middle|
    def f0(I_middle):
        if I_middle is None:
            return(0)
        else:
            return(max(I_middle) - min(I_middle))
    
    cut_into_three = cut_into_three_func(I, J)
    # Distance for now, not the mean:
    # Distance left: Between cut_into_three[0] and the point min(J)
    d_left = f(cut_into_three[0])
    # Distance middle: Between cut_into_three[1] = I inter J, and J
    d_middle = f0(cut_into_three[1])
    # Distance right: Between cut_into_three[2] and the point max(J)
    d_right = f(cut_into_three[2])
    # It's an integral so summable
    return(d_left + d_middle + d_right)

def cut_J_based_on_mean_func(J, e_mean):
    """
    Helper function for the recall.
    Partition J into two intervals: before and after e_mean
    (e_mean represents the center element of E the zone of affiliation)
    
    :param J: ground truth interval
    :param e_mean: a float number (center value of E)
    :return: a couple partitionning J into (J_before, J_after)
    """
    if J is None:
        J_before = None
        J_after = None
    elif e_mean >= max(J):
        J_before = J
        J_after = None
    elif e_mean <= min(J):
        J_before = None
        J_after = J
    else: # e_mean is across J
        J_before = (min(J), e_mean)
        J_after = (e_mean, max(J))
        
    return((J_before, J_after))

def integral_mini_interval_Precall_CDFmethod(I, J, E):
    """
    Integral of the probability of distances over the interval J.
    In the specific case where interval J is located outside I,
    compute the integral $\int_{y \in J} Fbar_y(dist(y,I)) dy$.
    This is the *integral* i.e. the sum (not the mean)
    
    :param I: a single (non empty) predicted interval
    :param J: ground truth (non empty) interval, with empty intersection with I
    :param E: the affiliation/influence zone for J, represented as a couple (start, stop)
    :return: the integral $\int_{y \in J} Fbar_y(dist(y,I)) dy$
    """
    # The interval J should be located outside I 
    # (so it's either the left piece or the right piece w.r.t I)
    i_pivot = get_pivot_j(J, I)
    e_min = min(E)
    e_max = max(E)
    e_mean = (e_min + e_max) / 2
    
    # If i_pivot is outside E (it's possible), then
    # the distance is worst that any random element within E,
    # so we set the recall to 0
    if i_pivot <= min(E):
        return(0)
    elif i_pivot >= max(E):
        return(0)
    # Otherwise, we have at least i_pivot in E and so d < M so min(d,M)=d
    
    cut_J_based_on_e_mean = cut_J_based_on_mean_func(J, e_mean)
    J_before = cut_J_based_on_e_mean[0]
    J_after = cut_J_based_on_e_mean[1]
  
    iemin_mean = (e_min + i_pivot)/2
    cut_Jbefore_based_on_iemin_mean = cut_J_based_on_mean_func(J_before, iemin_mean)
    J_before_closeE = cut_Jbefore_based_on_iemin_mean[0] # before e_mean and closer to e_min than i_pivot ~ J_before_before
    J_before_closeI = cut_Jbefore_based_on_iemin_mean[1] # before e_mean and closer to i_pivot than e_min ~ J_before_after
    
    iemax_mean = (e_max + i_pivot)/2
    cut_Jafter_based_on_iemax_mean = cut_J_based_on_mean_func(J_after, iemax_mean)
    J_after_closeI = cut_Jafter_based_on_iemax_mean[0] # after e_mean and closer to i_pivot than e_max ~ J_after_before
    J_after_closeE = cut_Jafter_based_on_iemax_mean[1] # after e_mean and closer to e_max than i_pivot ~ J_after_after
    
    if J_before_closeE is not None:
        j_before_before_min = min(J_before_closeE) # == min(J)
        j_before_before_max = max(J_before_closeE)
    else:
        j_before_before_min = math.nan
        j_before_before_max = math.nan
  
    if J_before_closeI is not None:
        j_before_after_min = min(J_before_closeI) # == j_before_before_max if existing
        j_before_after_max = max(J_before_closeI) # == max(J_before)
    else:
        j_before_after_min = math.nan
        j_before_after_max = math.nan
   
    if J_after_closeI is not None:
        j_after_before_min = min(J_after_closeI) # == min(J_after)
        j_after_before_max = max(J_after_closeI) 
    else:
        j_after_before_min = math.nan
        j_after_before_max = math.nan
    
    if J_after_closeE is not None:
        j_after_after_min = min(J_after_closeE) # == j_after_before_max if existing
        j_after_after_max = max(J_after_closeE) # == max(J)
    else:
        j_after_after_min = math.nan
        j_after_after_max = math.nan
  
    # <-- J_before_closeE --> <-- J_before_closeI --> <-- J_after_closeI --> <-- J_after_closeE -->
    # j_bb_min       j_bb_max j_ba_min       j_ba_max j_ab_min      j_ab_max j_aa_min      j_aa_max
    # (with `b` for before and `a` for after in the previous variable names)
    
    #                                          vs e_mean  m = min(t-e_min, e_max-t)  d=|i_pivot-t|   min(d,m)                            \int min(d,m)dt   \int d dt        \int_(min(d,m)+d)dt                                    \int_{t \in J}(min(d,m)+d)dt
    # Case J_before_closeE & i_pivot after J   before     t-e_min                    i_pivot-t       min(i_pivot-t,t-e_min) = t-e_min    t^2/2-e_min*t     i_pivot*t-t^2/2  t^2/2-e_min*t+i_pivot*t-t^2/2 = (i_pivot-e_min)*t      (i_pivot-e_min)*tB - (i_pivot-e_min)*tA = (i_pivot-e_min)*(tB-tA)
    # Case J_before_closeI & i_pivot after J   before     t-e_min                    i_pivot-t       min(i_pivot-t,t-e_min) = i_pivot-t  i_pivot*t-t^2/2   i_pivot*t-t^2/2  i_pivot*t-t^2/2+i_pivot*t-t^2/2 = 2*i_pivot*t-t^2      2*i_pivot*tB-tB^2 - 2*i_pivot*tA + tA^2 = 2*i_pivot*(tB-tA) - (tB^2 - tA^2)
    # Case J_after_closeI & i_pivot after J    after      e_max-t                    i_pivot-t       min(i_pivot-t,e_max-t) = i_pivot-t  i_pivot*t-t^2/2   i_pivot*t-t^2/2  i_pivot*t-t^2/2+i_pivot*t-t^2/2 = 2*i_pivot*t-t^2      2*i_pivot*tB-tB^2 - 2*i_pivot*tA + tA^2 = 2*i_pivot*(tB-tA) - (tB^2 - tA^2)
    # Case J_after_closeE & i_pivot after J    after      e_max-t                    i_pivot-t       min(i_pivot-t,e_max-t) = e_max-t    e_max*t-t^2/2     i_pivot*t-t^2/2  e_max*t-t^2/2+i_pivot*t-t^2/2 = (e_max+i_pivot)*t-t^2  (e_max+i_pivot)*tB-tB^2 - (e_max+i_pivot)*tA + tA^2 = (e_max+i_pivot)*(tB-tA) - (tB^2 - tA^2)
    #
    # Case J_before_closeE & i_pivot before J  before     t-e_min                    t-i_pivot       min(t-i_pivot,t-e_min) = t-e_min    t^2/2-e_min*t     t^2/2-i_pivot*t  t^2/2-e_min*t+t^2/2-i_pivot*t = t^2-(e_min+i_pivot)*t  tB^2-(e_min+i_pivot)*tB - tA^2 + (e_min+i_pivot)*tA = (tB^2 - tA^2) - (e_min+i_pivot)*(tB-tA)
    # Case J_before_closeI & i_pivot before J  before     t-e_min                    t-i_pivot       min(t-i_pivot,t-e_min) = t-i_pivot  t^2/2-i_pivot*t   t^2/2-i_pivot*t  t^2/2-i_pivot*t+t^2/2-i_pivot*t = t^2-2*i_pivot*t      tB^2-2*i_pivot*tB - tA^2 + 2*i_pivot*tA = (tB^2 - tA^2) - 2*i_pivot*(tB-tA)
    # Case J_after_closeI & i_pivot before J   after      e_max-t                    t-i_pivot       min(t-i_pivot,e_max-t) = t-i_pivot  t^2/2-i_pivot*t   t^2/2-i_pivot*t  t^2/2-i_pivot*t+t^2/2-i_pivot*t = t^2-2*i_pivot*t      tB^2-2*i_pivot*tB - tA^2 + 2*i_pivot*tA = (tB^2 - tA^2) - 2*i_pivot*(tB-tA)
    # Case J_after_closeE & i_pivot before J   after      e_max-t                    t-i_pivot       min(t-i_pivot,e_max-t) = e_max-t    e_max*t-t^2/2     t^2/2-i_pivot*t  e_max*t-t^2/2+t^2/2-i_pivot*t = (e_max-i_pivot)*t      (e_max-i_pivot)*tB - (e_max-i_pivot)*tA = (e_max-i_pivot)*(tB-tA)
    
    if i_pivot >= max(J):
        part1_before_closeE = (i_pivot-e_min)*(j_before_before_max - j_before_before_min) # (i_pivot-e_min)*(tB-tA) # j_before_before_max - j_before_before_min
        part2_before_closeI = 2*i_pivot*(j_before_after_max-j_before_after_min) - (j_before_after_max**2 - j_before_after_min**2) # 2*i_pivot*(tB-tA) - (tB^2 - tA^2) # j_before_after_max - j_before_after_min
        part3_after_closeI = 2*i_pivot*(j_after_before_max-j_after_before_min) - (j_after_before_max**2 - j_after_before_min**2) # 2*i_pivot*(tB-tA) - (tB^2 - tA^2) # j_after_before_max - j_after_before_min  
        part4_after_closeE = (e_max+i_pivot)*(j_after_after_max-j_after_after_min) - (j_after_after_max**2 - j_after_after_min**2) # (e_max+i_pivot)*(tB-tA) - (tB^2 - tA^2) # j_after_after_max - j_after_after_min
        out_parts = [part1_before_closeE, part2_before_closeI, part3_after_closeI, part4_after_closeE]
    elif i_pivot <= min(J):
        part1_before_closeE = (j_before_before_max**2 - j_before_before_min**2) - (e_min+i_pivot)*(j_before_before_max-j_before_before_min) # (tB^2 - tA^2) - (e_min+i_pivot)*(tB-tA) # j_before_before_max - j_before_before_min
        part2_before_closeI = (j_before_after_max**2 - j_before_after_min**2) - 2*i_pivot*(j_before_after_max-j_before_after_min) # (tB^2 - tA^2) - 2*i_pivot*(tB-tA) # j_before_after_max - j_before_after_min
        part3_after_closeI = (j_after_before_max**2 - j_after_before_min**2) - 2*i_pivot*(j_after_before_max - j_after_before_min) # (tB^2 - tA^2) - 2*i_pivot*(tB-tA) # j_after_before_max - j_after_before_min
        part4_after_closeE = (e_max-i_pivot)*(j_after_after_max - j_after_after_min) # (e_max-i_pivot)*(tB-tA) # j_after_after_max - j_after_after_min
        out_parts = [part1_before_closeE, part2_before_closeI, part3_after_closeI, part4_after_closeE]
    else:
        raise ValueError('The i_pivot should be outside J')
    
    out_integral_min_dm_plus_d = _sum_wo_nan(out_parts) # integral on all J, i.e. sum of the disjoint parts

    # We have for each point t of J:
    # \bar{F}_{t, recall}(d) = 1 - (1/|E|) * (min(d,m) + d)
    # Since t is a single-point here, and we are in the case where i_pivot is inside E.
    # The integral is then given by:
    # C = \int_{t \in J} \bar{F}_{t, recall}(D(t)) dt
    #   = \int_{t \in J} 1 - (1/|E|) * (min(d,m) + d) dt
    #   = |J| - (1/|E|) * [\int_{t \in J} (min(d,m) + d) dt]
    #   = |J| - (1/|E|) * out_integral_min_dm_plus_d    
    DeltaJ = max(J) - min(J)
    DeltaE = max(E) - min(E)
    C = DeltaJ - (1/DeltaE) * out_integral_min_dm_plus_d
    
    return(C)

def integral_interval_probaCDF_recall(I, J, E):
    """
    Integral of the probability of distances over the interval J.
    Compute the integral $\int_{y \in J} Fbar_y(dist(y,I)) dy$.
    This is the *integral* i.e. the sum (not the mean)

    :param I: a single (non empty) predicted interval
    :param J: ground truth (non empty) interval
    :param E: the affiliation/influence zone for J
    :return: the integral $\int_{y \in J} Fbar_y(dist(y,I)) dy$
    """
    # I and J are single intervals (not generic sets)
    # E is the outside affiliation interval of J (even for recall!)
    # (in particular J \subset E)
    #
    # J is the portion of the ground truth affiliated to I
    # I is a predicted interval (can be outside E possibly since it's recall)
    def f(J_cut):
        if J_cut is None:
            return(0)
        else:
            return integral_mini_interval_Precall_CDFmethod(I, J_cut, E)

    # If J_middle is fully included into I, it is
    # integral of 1 on the interval J_middle, so it's |J_middle|
    def f0(J_middle):
        if J_middle is None:
            return(0)
        else:
            return(max(J_middle) - min(J_middle))
    
    cut_into_three = cut_into_three_func(J, I) # it's J that we cut into 3, depending on the position w.r.t I
    # since we integrate over J this time.
    #
    # Distance for now, not the mean:
    # Distance left: Between cut_into_three[0] and the point min(I)
    d_left = f(cut_into_three[0])
    # Distance middle: Between cut_into_three[1] = J inter I, and I
    d_middle = f0(cut_into_three[1])
    # Distance right: Between cut_into_three[2] and the point max(I)
    d_right = f(cut_into_three[2])
    # It's an integral so summable
    return(d_left + d_middle + d_right)
