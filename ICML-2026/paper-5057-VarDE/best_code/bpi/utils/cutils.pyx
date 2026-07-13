# This file is licensed under the MIT License.
# See the LICENSE file in the project root for full license information.
#

#! /usr/bin/env python
# -*- coding: utf-8 -*-

# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False

import numpy as numpy
cimport numpy as np

cpdef np.ndarray[double, ndim=1, mode='c']  policy_evaluation(gamma: double, P: double[:,:,:], R: double[:,:,:], pi: long[::1], atol: double = 1e-6):
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
    
    cdef long NS = P.shape[0]
    cdef long s = 0
    cdef np.ndarray[double, ndim=1, mode="c"] V = numpy.zeros(NS)
    cdef np.ndarray[double, ndim=1, mode="c"] V_next = numpy.zeros(NS)
    cdef double Delta = 0

    while True:
        Delta = 0
        for s in range(NS):
            V_next[s] = P[s, pi[s]] @ (R[s, pi[s]] + gamma * V)

        Delta = numpy.max([Delta, numpy.abs(V_next - V).max()])
        V = V_next.copy()
        
        if Delta < atol:
            break
    return V 


cpdef np.ndarray[long, ndim=1, mode='c']policy_iteration(gamma: double, P: double[:,:,:], R: double[:,:,:], atol: double = 1e-6):
    """Policy evaluation

    Args:
        gamma (float): Discount factor
        P (NDArray[np.float64]): Transition function of shape (num_states, num_actions, num_states)
        R (NDArray[np.float64]): Reward function of shape (num_states, num_actions)
        V0 (Optional[NDArray[np.float64]], optional): Initial value function. Defaults to None.
        atol (float): Absolute tolerance

    Returns:
        NDArray[np.int64]: greedy policy
    """
    
    cdef long NS = P.shape[0]
    cdef long NA = P.shape[1]
    cdef long s = 0
    cdef long a = 0
    cdef np.ndarray[long, ndim=1, mode="c"] pi = numpy.zeros(NS, dtype=long)
    cdef np.ndarray[long, ndim=1, mode="c"] next_pi = numpy.zeros(NS, dtype=long)
    cdef np.ndarray[double, ndim=1, mode="c"] V = numpy.zeros(NS)
    cdef np.ndarray[double, ndim=2, mode="c"] Q = numpy.zeros((NS, NA))
    cdef double Delta = 0
    cdef long policy_stable = 0

    while policy_stable == 0:
        policy_stable = 1
        V = policy_evaluation(gamma, P, R, pi, atol)
        for s in range(NS):
            for a in range(NA):
                Q[s,a] = P[s,a] @ (R[s,a] + gamma * V)
        next_pi = numpy.argmax(Q, axis=1)

        if not numpy.array_equal(next_pi, pi):
            policy_stable = 0
        pi = next_pi


    return pi
      