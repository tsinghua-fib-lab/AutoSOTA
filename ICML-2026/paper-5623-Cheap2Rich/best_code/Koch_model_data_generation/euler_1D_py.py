#!/usr/bin/env python
# encoding: utf-8
r"""
Riemann solvers for the Euler equations

This module contains Riemann solvers for the Euler equations which have the
form (in 1d):

.. math::
    q_t + f(q)_x = 0

where

.. math::
    q(x,t) = \left [ \begin{array}{c} \rho \\ \rho u \\ E \end{array} \right ],

the flux function is

.. math::
    f(q) = \left [ \begin{array}{c} \rho u \\ \rho u^2 + p \\ u(E+p) \end{array}\right ].

and :math:`\rho` is the density, :math:`u` the velocity, :math:`E` is the
energy and :math:`p` is the pressure.

Unless otherwise noted, the ideal gas equation of state is used:

.. math::
    E = (\gamma - 1) \left (E - \frac{1}{2}\rho u^2 \right)
"""

from __future__ import absolute_import
import numpy as np
from six.moves import range

num_eqn = 3

def euler_rq1D(q_l,q_r,aux_l,aux_r,problem_data):
    r"""
    HLL euler solver ::

        W_1 = Q_hat - Q_l    s_1 = min(u_l-c_l,u_l+c_l,lambda_roe_1,lambda_roe_2)
        W_2 = Q_r - Q_hat    s_2 = max(u_r-c_r,u_r+c_r,lambda_roe_1,lambda_roe_2)

        Q_hat = ( f(q_r) - f(q_l) - s_2 * q_r + s_1 * q_l ) / (s_1 - s_2)
    """
    
    # Number of equations is actually 4 in this scemario for a progress variable
    num_eqn = 4

    # Problem dimensions
    num_rp = q_l.shape[1]
    num_waves = 2

    # Return values
    wave = np.empty( (num_eqn, num_waves, num_rp) )
    s = np.empty( (num_waves, num_rp) )
    amdq = np.zeros( (num_eqn, num_rp) )
    apdq = np.zeros( (num_eqn, num_rp) )

    # Solver parameters
    gamma1 = problem_data['gamma1']
    area = (aux_l[0,:]+aux_r[0,:])/2.0

    # Calculate Roe averages, right and left speeds
    u, a, _, pl, pr = roe_averages_rq1d(q_l,q_r,area,problem_data)
    H_r = (q_r[2,:] / area + pr) / (q_r[0,:] / area)
    H_l = (q_l[2,:] / area + pl) / (q_l[0,:] / area)
    u_r = q_r[1,:] / q_r[0,:]
    u_l = q_l[1,:] / q_l[0,:]
    a_r = np.sqrt(gamma1 * (H_r - 0.5 * u_r**2))
    a_l = np.sqrt(gamma1 * (H_l - 0.5 * u_l**2))

    # Compute Einfeldt speeds
    s_index = np.empty((4,num_rp))
    s_index[0,:] = u + a
    s_index[1,:] = u - a
    s_index[2,:] = u_l + a_l
    s_index[3,:] = u_l - a_l
    s[0,:]  = np.min(s_index,axis=0)
    s_index[2,:] = u_r + a_r
    s_index[3,:] = u_r - a_r
    s[1,:] = np.max(s_index,axis=0)

    # Compute middle state
    q_hat = np.empty((num_eqn,num_rp))
    q_hat[0,:] = (q_r[1,:] - q_l[1,:] -
                    s[1,:] * q_r[0,:] + s[0,:] * q_l[0,:]) / (s[0,:] - s[1,:])
    q_hat[1,:] = (q_r[1,:]**2/q_r[0,:] + pr * area -
                    (q_l[1,:]**2/q_l[0,:] + pl * area) -
                    s[1,:] * q_r[1,:] + s[0,:] * q_l[1,:]) / (s[0,:] - s[1,:])
    q_hat[2,:] = ((q_r[2,:] + pr * area)*q_r[1,:]/q_r[0,:] - 
                    (q_l[2,:] + pl * area)*q_l[1,:]/q_l[0,:] -
                    s[1,:] * q_r[2,:] + s[0,:] * q_l[2,:]) / (s[0,:] - s[1,:])
    q_hat[3,:] = ((q_r[3,:]/q_r[0,:])*q_r[1,:] - (q_l[3,:]/q_l[0,:])*q_l[1,:] -
                    s[1,:] * q_r[3,:] + s[0,:] * q_l[3,:]) / (s[0,:] - s[1,:])

    # Compute each family of waves
    wave[:,0,:] = q_hat - q_l
    wave[:,1,:] = q_r - q_hat

    # Compute variations
    s_index = np.zeros((2,num_rp))
    for m in range(num_eqn):
        for mw in range(num_waves):
            s_index[0,:] = s[mw,:]
            amdq[m,:] += np.min(s_index,axis=0) * wave[m,mw,:]
            apdq[m,:] += np.max(s_index,axis=0) * wave[m,mw,:]

    return wave, s, amdq, apdq

def roe_averages_rq1d(q_l,q_r,area,problem_data):
    # Solver parameters
    gamma1 = problem_data['gamma1']
    hv = problem_data['hv']

    # Calculate Roe averages
    rhsqrtl = np.sqrt(q_l[0,...] / area)
    rhsqrtr = np.sqrt(q_r[0,...] / area)
    pl = gamma1 * (q_l[2,...] - (0.5 * (q_l[1,...]**2) / q_l[0,...]) - 
         hv*q_l[3,...]) / area
    pr = gamma1 * (q_r[2,...] - (0.5 * (q_r[1,...]**2) / q_r[0,...]) - 
         hv*q_r[3,...]) / area
    rhsq2 = rhsqrtl + rhsqrtr
    u = ((q_l[1,...] / area) / rhsqrtl + (q_r[1,...] / area) / rhsqrtr) / rhsq2
    enthalpy = ((q_l[2,...] / area + pl) / rhsqrtl + (q_r[2,...] / area + pr) / 
         rhsqrtr) / rhsq2
    a = np.sqrt(gamma1 * (enthalpy - 0.5 * u**2))

    return u, a, enthalpy, pl, pr

def roe_averages_rq1d_alt(q_l,q_r,area,problem_data):
    # Solver parameters
    gamma1 = problem_data['gamma1']
    hv = problem_data['hv']

    # Calculate Roe averages
    rhsqrtl = np.sqrt(q_l[0,...] / area)
    rhsqrtr = np.sqrt(q_r[0,...] / area)
    pl = gamma1 * (q_l[2,...] - (0.5 * (q_l[1,...]**2) / q_l[0,...])) / area
    pr = gamma1 * (q_r[2,...] - (0.5 * (q_r[1,...]**2) / q_r[0,...])) / area
    rhsq2 = rhsqrtl + rhsqrtr
    u = ((q_l[1,...] / area) / rhsqrtl + (q_r[1,...] / area) / rhsqrtr) / rhsq2
    enthalpy = ((q_l[2,...] / area + pl) / rhsqrtl + (q_r[2,...] / area + pr) / 
         rhsqrtr) / rhsq2
    a = np.sqrt(gamma1 * (enthalpy - 0.5 * u**2))

    return u, a, enthalpy, pl, pr

def euler_hllc_rq1D(q_l,q_r,aux_l,aux_r,problem_data):
    r"""
    HLLC Euler solver ::

        W_1 = q_hat_l - q_l      s_1 = min(u_l-c_l,u_l+c_l,lambda_roe_1,lambda_roe_2)
        W_2 = q_hat_r - q_hat_l  s_2 = s_m
        W_3 = q_r - q_hat_r      s_3 = max(u_r-c_r,u_r+c_r,lambda_roe_1,lambda_roe_2)
        s_m = (p_r - p_l + rho_l*u_l*(s_l - u_l) - rho_r*u_r*(s_r - u_r))\
          / (rho_l*(s_l-u_l) - rho_r*(s_r - u_r))

    left middle state::

        q_hat_l[0,:] = rho_l*(s_l - u_l)/(s_l - s_m)
        q_hat_l[1,:] = rho_l*(s_l - u_l)/(s_l - s_m)*s_m
        q_hat_l[2,:] = rho_l*(s_l - u_l)/(s_l - s_m)\
                *(E_l/rho_l + (s_m - u_l)*(s_m + p_l/(rho_l*(s_l - u_l))))

    right middle state::

        q_hat_r[0,:] = rho_r*(s_r - u_r)/(s_r - s_m)
        q_hat_r[1,:] = rho_r*(s_r - u_r)/(s_r - s_m)*s_m
        q_hat_r[2,:] = rho_r*(s_r - u_r)/(s_r - s_m)\
                *(E_r/rho_r + (s_m - u_r)*(s_m + p_r/(rho_r*(s_r - u_r))))

    *problem_data* should contain:

        - *gamma*: (float) Ratio of specific heat capacities
        - *gamma1*: (float) :math:`\gamma - 1`

    :Version 1.0 (2015-11-18)
    """

    # Problem dimensions
    num_rp = q_l.shape[1]
    num_waves = 3
    num_eqn = 4

    # Return values
    wave = np.empty( (num_eqn, num_waves, num_rp) )
    s = np.empty( (num_waves, num_rp) )
    amdq = np.zeros( (num_eqn, num_rp) )
    apdq = np.zeros( (num_eqn, num_rp) )
    
    # Solver parameters
    gamma1 = problem_data['gamma1']
    hv = problem_data['hv']
    area = (aux_l[0,:]+aux_r[0,:])/2.0

    # Calculate Roe averages, right and left speeds
    u, a, _, p_l, p_r = roe_averages_rq1d(q_l,q_r,area,problem_data)
    rho_r = q_r[0,:] / area
    rho_l = q_l[0,:] / area
    E_r = q_r[2,:] / area
    E_l = q_l[2,:] / area
    H_r = (E_r + p_r) / rho_r
    H_l = (E_l + p_l) / rho_l
    u_r = q_r[1,:] / q_r[0,:]
    u_l = q_l[1,:] / q_l[0,:]
    a_r = np.sqrt(gamma1 * (H_r - 0.5 * u_r**2))
    a_l = np.sqrt(gamma1 * (H_l - 0.5 * u_l**2))
    z_r = q_r[3,:] / q_r[0,:]
    z_l = q_l[3,:] / q_l[0,:]

    # Compute Einfeldt speeds
    s_index = np.empty((4,num_rp))
    s_index[0,:] = u + a
    s_index[1,:] = u - a
    s_index[2,:] = u_l + a_l
    s_index[3,:] = u_l - a_l
    s[0,:]  = np.min(s_index,axis=0)
    s_index[2,:] = u_r + a_r
    s_index[3,:] = u_r - a_r
    s[2,:] = np.max(s_index,axis=0)

    # left and right speeds
    s_l = s[0,:]
    s_r = s[2,:]
    
    # middle speed
    s_m = np.empty((num_rp))
    s_m[:] = (p_r - p_l + rho_l*u_l*(s_l - u_l) - rho_r*u_r*(s_r - u_r))\
          / (rho_l*(s_l-u_l) - rho_r*(s_r - u_r))
    s[1,:] = s_m
    
    # left middle states
    q_hat_l = np.empty((num_eqn,num_rp))
    q_hat_l[0,:] = area * rho_l*(s_l - u_l)/(s_l - s_m)
    q_hat_l[1,:] = area * rho_l*(s_l - u_l)/(s_l - s_m)*s_m
    q_hat_l[2,:] = area * rho_l*(s_l - u_l)/(s_l - s_m)\
                *(E_l/rho_l + (s_m - u_l)*(s_m + p_l/(rho_l*(s_l - u_l))))
    q_hat_l[3,:] = q_hat_l[0,:] * z_l
    
    # right middle state
    q_hat_r = np.empty((num_eqn,num_rp))
    q_hat_r[0,:] = area * rho_r*(s_r - u_r)/(s_r - s_m)
    q_hat_r[1,:] = area * rho_r*(s_r - u_r)/(s_r - s_m)*s_m
    q_hat_r[2,:] = area * rho_r*(s_r - u_r)/(s_r - s_m)\
                *(E_r/rho_r + (s_m - u_r)*(s_m + p_r/(rho_r*(s_r - u_r))))
    q_hat_r[3,:] = q_hat_r[0,:] * z_r

    # Compute each family of waves
    wave[:,0,:] = q_hat_l - q_l
    wave[:,1,:] = q_hat_r - q_hat_l
    wave[:,2,:] = q_r - q_hat_r
    
    # Compute variations
    s_index = np.zeros((2,num_rp))
    for m in range(num_eqn):
        for mw in range(num_waves):
            s_index[0,:] = s[mw,:]
            amdq[m,:] += np.min(s_index,axis=0) * wave[m,mw,:]
            apdq[m,:] += np.max(s_index,axis=0) * wave[m,mw,:]
            
    return wave, s, amdq, apdq

def euler_hllc_rq1D_counterProp(q_l,q_r,aux_l,aux_r,problem_data):
    r"""
    HLLC Euler solver ::

        W_1 = q_hat_l - q_l      s_1 = min(u_l-c_l,u_l+c_l,lambda_roe_1,lambda_roe_2)
        W_2 = q_hat_r - q_hat_l  s_2 = s_m
        W_3 = q_r - q_hat_r      s_3 = max(u_r-c_r,u_r+c_r,lambda_roe_1,lambda_roe_2)
        s_m = (p_r - p_l + rho_l*u_l*(s_l - u_l) - rho_r*u_r*(s_r - u_r))\
          / (rho_l*(s_l-u_l) - rho_r*(s_r - u_r))

    left middle state::

        q_hat_l[0,:] = rho_l*(s_l - u_l)/(s_l - s_m)
        q_hat_l[1,:] = rho_l*(s_l - u_l)/(s_l - s_m)*s_m
        q_hat_l[2,:] = rho_l*(s_l - u_l)/(s_l - s_m)\
                *(E_l/rho_l + (s_m - u_l)*(s_m + p_l/(rho_l*(s_l - u_l))))

    right middle state::

        q_hat_r[0,:] = rho_r*(s_r - u_r)/(s_r - s_m)
        q_hat_r[1,:] = rho_r*(s_r - u_r)/(s_r - s_m)*s_m
        q_hat_r[2,:] = rho_r*(s_r - u_r)/(s_r - s_m)\
                *(E_r/rho_r + (s_m - u_r)*(s_m + p_r/(rho_r*(s_r - u_r))))

    *problem_data* should contain:

        - *gamma*: (float) Ratio of specific heat capacities
        - *gamma1*: (float) :math:`\gamma - 1`

    :Version 1.0 (2015-11-18)
    """

    # Problem dimensions
    num_rp = q_l.shape[1]
    num_waves = 3
    num_eqn = 4

    # Return values
    wave = np.empty( (num_eqn, num_waves, num_rp) )
    s = np.empty( (num_waves, num_rp) )
    amdq = np.zeros( (num_eqn, num_rp) )
    apdq = np.zeros( (num_eqn, num_rp) )
    
    # Solver parameters
    gamma1 = problem_data['gamma1']
    hv = problem_data['hv']
    area = (aux_l[0,:]+aux_r[0,:])/2.0

    # Calculate Roe averages, right and left speeds
    u, a, _, p_l, p_r = roe_averages_rq1d_alt(q_l,q_r,area,problem_data)
    rho_r = q_r[0,:] / area
    rho_l = q_l[0,:] / area
    E_r = q_r[2,:] / area
    E_l = q_l[2,:] / area
    H_r = (E_r + p_r) / rho_r
    H_l = (E_l + p_l) / rho_l
    u_r = q_r[1,:] / q_r[0,:]
    u_l = q_l[1,:] / q_l[0,:]
    a_r = np.sqrt(gamma1 * (H_r - 0.5 * u_r**2))
    a_l = np.sqrt(gamma1 * (H_l - 0.5 * u_l**2))
    z_r = q_r[3,:] / q_r[0,:]
    z_l = q_l[3,:] / q_l[0,:]

    # Compute Einfeldt speeds
    s_index = np.empty((4,num_rp))
    s_index[0,:] = u + a
    s_index[1,:] = u - a
    s_index[2,:] = u_l + a_l
    s_index[3,:] = u_l - a_l
    s[0,:]  = np.min(s_index,axis=0)
    s_index[2,:] = u_r + a_r
    s_index[3,:] = u_r - a_r
    s[2,:] = np.max(s_index,axis=0)

    # left and right speeds
    s_l = s[0,:]
    s_r = s[2,:]
    
    # middle speed
    s_m = np.empty((num_rp))
    s_m[:] = (p_r - p_l + rho_l*u_l*(s_l - u_l) - rho_r*u_r*(s_r - u_r))\
          / (rho_l*(s_l-u_l) - rho_r*(s_r - u_r))
    s[1,:] = s_m
    
    # left middle states
    q_hat_l = np.empty((num_eqn,num_rp))
    q_hat_l[0,:] = area * rho_l*(s_l - u_l)/(s_l - s_m)
    q_hat_l[1,:] = area * rho_l*(s_l - u_l)/(s_l - s_m)*s_m
    q_hat_l[2,:] = area * rho_l*(s_l - u_l)/(s_l - s_m)\
                *(E_l/rho_l + (s_m - u_l)*(s_m + p_l/(rho_l*(s_l - u_l))))
    q_hat_l[3,:] = q_hat_l[0,:] * z_l
    
    # right middle state
    q_hat_r = np.empty((num_eqn,num_rp))
    q_hat_r[0,:] = area * rho_r*(s_r - u_r)/(s_r - s_m)
    q_hat_r[1,:] = area * rho_r*(s_r - u_r)/(s_r - s_m)*s_m
    q_hat_r[2,:] = area * rho_r*(s_r - u_r)/(s_r - s_m)\
                *(E_r/rho_r + (s_m - u_r)*(s_m + p_r/(rho_r*(s_r - u_r))))
    q_hat_r[3,:] = q_hat_r[0,:] * z_r

    # Compute each family of waves
    wave[:,0,:] = q_hat_l - q_l
    wave[:,1,:] = q_hat_r - q_hat_l
    wave[:,2,:] = q_r - q_hat_r
    
    # Compute variations
    s_index = np.zeros((2,num_rp))
    for m in range(num_eqn):
        for mw in range(num_waves):
            s_index[0,:] = s[mw,:]
            amdq[m,:] += np.min(s_index,axis=0) * wave[m,mw,:]
            apdq[m,:] += np.max(s_index,axis=0) * wave[m,mw,:]
            
    return wave, s, amdq, apdq

def euler_roe_1D(q_l,q_r,aux_l,aux_r,problem_data):
    r"""
    Roe Euler solver in 1d

    *aug_global* should contain -
     - *gamma* - (float) Ratio of the heat capacities
     - *gamma1* - (float) :math:`1 - \gamma`
     - *efix* - (bool) Whether to use an entropy fix or not

    See :ref:`pyclaw_rp` for more details.

    :Version: 1.0 (2009-6-26)
    """

    # Problem dimensions
    num_rp = q_l.shape[1]
    num_waves = 3

    # Return values
    wave = np.empty( (num_eqn, num_waves, num_rp) )
    s = np.empty( (num_waves, num_rp) )
    amdq = np.zeros( (num_eqn, num_rp) )
    apdq = np.zeros( (num_eqn, num_rp) )

    # Solver parameters
    gamma1 = problem_data['gamma1']

    # Calculate Roe averages
    u, a, enthalpy = roe_averages(q_l,q_r,problem_data)[0:3]

    # Find eigenvector coefficients
    delta = q_r - q_l
    a2 = gamma1 / a**2 * ((enthalpy -u**2)*delta[0,...] + u*delta[1,...] - delta[2,...])
    a3 = (delta[1,...] + (a-u) * delta[0,...] - a*a2) / (2.0*a)
    a1 = delta[0,...] - a2 - a3

    # Compute the waves
    wave[0,0,...] = a1
    wave[1,0,...] = a1 * (u-a)
    wave[2,0,...] = a1 * (enthalpy - u*a)
    s[0,...] = u - a

    wave[0,1,...] = a2
    wave[1,1,...] = a2 * u
    wave[2,1,...] = a2 * 0.5 * u**2
    s[1,...] = u

    wave[0,2,...] = a3
    wave[1,2,...] = a3 * (u+a)
    wave[2,2,...] = a3 * (enthalpy + u*a)
    s[2,...] = u + a

    # Entropy fix
    if problem_data['efix']:
        raise NotImplementedError("Entropy fix has not been implemented!")
    else:
        # Godunov update
        s_index = np.zeros((2,num_rp))
        for m in range(num_eqn):
            for mw in range(num_waves):
                s_index[0,:] = s[mw,:]
                amdq[m,:] += np.min(s_index,axis=0) * wave[m,mw,:]
                apdq[m,:] += np.max(s_index,axis=0) * wave[m,mw,:]

    return wave,s,amdq,apdq

def euler_hll_1D(q_l,q_r,aux_l,aux_r,problem_data):
    r"""
    HLL euler solver ::

        W_1 = Q_hat - Q_l    s_1 = min(u_l-c_l,u_l+c_l,lambda_roe_1,lambda_roe_2)
        W_2 = Q_r - Q_hat    s_2 = max(u_r-c_r,u_r+c_r,lambda_roe_1,lambda_roe_2)

        Q_hat = ( f(q_r) - f(q_l) - s_2 * q_r + s_1 * q_l ) / (s_1 - s_2)

    *problem_data* should contain:
     - *gamma* - (float) Ratio of the heat capacities
     - *gamma1* - (float) :math:`1 - \gamma`

    :Version: 1.0 (2014-03-04)
    """

    # Problem dimensions
    num_rp = q_l.shape[1]
    num_waves = 2

    # Return values
    wave = np.empty( (num_eqn, num_waves, num_rp) )
    s = np.empty( (num_waves, num_rp) )
    amdq = np.zeros( (num_eqn, num_rp) )
    apdq = np.zeros( (num_eqn, num_rp) )

    # Solver parameters
    gamma1 = problem_data['gamma1']

    # Calculate Roe averages, right and left speeds
    u, a, _, pl, pr = roe_averages(q_l,q_r,problem_data)
    H_r = (q_r[2,:] + pr) / q_r[0,:]
    H_l = (q_l[2,:] + pl) / q_l[0,:]
    u_r = q_r[1,:] / q_r[0,:]
    u_l = q_l[1,:] / q_l[0,:]
    a_r = np.sqrt(gamma1 * (H_r - 0.5 * u_r**2))
    a_l = np.sqrt(gamma1 * (H_l - 0.5 * u_l**2))

    # Compute Einfeldt speeds
    s_index = np.empty((4,num_rp))
    s_index[0,:] = u + a
    s_index[1,:] = u - a
    s_index[2,:] = u_l + a_l
    s_index[3,:] = u_l - a_l
    s[0,:]  = np.min(s_index,axis=0)
    s_index[2,:] = u_r + a_r
    s_index[3,:] = u_r - a_r
    s[1,:] = np.max(s_index,axis=0)

    # Compute middle state
    q_hat = np.empty((num_eqn,num_rp))
    q_hat[0,:] = (q_r[1,:] - q_l[1,:] -
                    s[1,:] * q_r[0,:] + s[0,:] * q_l[0,:]) / (s[0,:] - s[1,:])
    q_hat[1,:] = (q_r[1,:]**2/q_r[0,:] + pr - (q_l[1,:]**2/q_l[0,:] + pl) -
                    s[1,:] * q_r[1,:] + s[0,:] * q_l[1,:]) / (s[0,:] - s[1,:])
    q_hat[2,:] = ((q_r[2,:] + pr)*q_r[1,:]/q_r[0,:] - (q_l[2,:] + pl)*q_l[1,:]/q_l[0,:] -
                    s[1,:] * q_r[2,:] + s[0,:] * q_l[2,:]) / (s[0,:] - s[1,:])

    # Compute each family of waves
    wave[:,0,:] = q_hat - q_l
    wave[:,1,:] = q_r - q_hat

    # Compute variations
    s_index = np.zeros((2,num_rp))
    for m in range(num_eqn):
        for mw in range(num_waves):
            s_index[0,:] = s[mw,:]
            amdq[m,:] += np.min(s_index,axis=0) * wave[m,mw,:]
            apdq[m,:] += np.max(s_index,axis=0) * wave[m,mw,:]

    return wave, s, amdq, apdq

def euler_hllc_1D(q_l,q_r,aux_l,aux_r,problem_data):
    r"""
    HLLC Euler solver ::

        W_1 = q_hat_l - q_l      s_1 = min(u_l-c_l,u_l+c_l,lambda_roe_1,lambda_roe_2)
        W_2 = q_hat_r - q_hat_l  s_2 = s_m
        W_3 = q_r - q_hat_r      s_3 = max(u_r-c_r,u_r+c_r,lambda_roe_1,lambda_roe_2)
        s_m = (p_r - p_l + rho_l*u_l*(s_l - u_l) - rho_r*u_r*(s_r - u_r))\
          / (rho_l*(s_l-u_l) - rho_r*(s_r - u_r))

    left middle state::

        q_hat_l[0,:] = rho_l*(s_l - u_l)/(s_l - s_m)
        q_hat_l[1,:] = rho_l*(s_l - u_l)/(s_l - s_m)*s_m
        q_hat_l[2,:] = rho_l*(s_l - u_l)/(s_l - s_m)\
                *(E_l/rho_l + (s_m - u_l)*(s_m + p_l/(rho_l*(s_l - u_l))))

    right middle state::

        q_hat_r[0,:] = rho_r*(s_r - u_r)/(s_r - s_m)
        q_hat_r[1,:] = rho_r*(s_r - u_r)/(s_r - s_m)*s_m
        q_hat_r[2,:] = rho_r*(s_r - u_r)/(s_r - s_m)\
                *(E_r/rho_r + (s_m - u_r)*(s_m + p_r/(rho_r*(s_r - u_r))))

    *problem_data* should contain:

        - *gamma*: (float) Ratio of specific heat capacities
        - *gamma1*: (float) :math:`\gamma - 1`

    :Version 1.0 (2015-11-18)
    """

    # Problem dimensions
    num_rp = q_l.shape[1]
    num_waves = 3

    # Return values
    wave = np.empty( (num_eqn, num_waves, num_rp) )
    s = np.empty( (num_waves, num_rp) )
    amdq = np.zeros( (num_eqn, num_rp) )
    apdq = np.zeros( (num_eqn, num_rp) )
    
    # Solver parameters
    gamma1 = problem_data['gamma1']
    
    # Calculate Roe averages, right and left speeds
    u, a, _, p_l, p_r = roe_averages(q_l,q_r,problem_data)
    rho_r = q_r[0,:]
    rho_l = q_l[0,:]
    E_r = q_r[2,:]
    E_l = q_l[2,:]
    H_r = (E_r + p_r) / rho_r
    H_l = (E_l + p_l) / rho_l
    u_r = q_r[1,:] / rho_r
    u_l = q_l[1,:] / rho_l
    a_r = np.sqrt(gamma1 * (H_r - 0.5 * u_r**2))
    a_l = np.sqrt(gamma1 * (H_l - 0.5 * u_l**2))

    # Compute Einfeldt speeds
    s_index = np.empty((4,num_rp))
    s_index[0,:] = u + a
    s_index[1,:] = u - a
    s_index[2,:] = u_l + a_l
    s_index[3,:] = u_l - a_l
    s[0,:]  = np.min(s_index,axis=0)
    s_index[2,:] = u_r + a_r
    s_index[3,:] = u_r - a_r
    s[2,:] = np.max(s_index,axis=0)

    # left and right speeds
    s_l = s[0,:]
    s_r = s[2,:]
    
    # middle speed
    s_m = np.empty((num_rp))
    s_m[:] = (p_r - p_l + rho_l*u_l*(s_l - u_l) - rho_r*u_r*(s_r - u_r))\
          / (rho_l*(s_l-u_l) - rho_r*(s_r - u_r))
    s[1,:] = s_m
    
    # left middle states
    q_hat_l = np.empty((num_eqn,num_rp))
    q_hat_l[0,:] = rho_l*(s_l - u_l)/(s_l - s_m)
    q_hat_l[1,:] = rho_l*(s_l - u_l)/(s_l - s_m)*s_m
    q_hat_l[2,:] = rho_l*(s_l - u_l)/(s_l - s_m)\
                *(E_l/rho_l + (s_m - u_l)*(s_m + p_l/(rho_l*(s_l - u_l))))
    
    # right middle state
    q_hat_r = np.empty((num_eqn,num_rp))
    q_hat_r[0,:] = rho_r*(s_r - u_r)/(s_r - s_m)
    q_hat_r[1,:] = rho_r*(s_r - u_r)/(s_r - s_m)*s_m
    q_hat_r[2,:] = rho_r*(s_r - u_r)/(s_r - s_m)\
                *(E_r/rho_r + (s_m - u_r)*(s_m + p_r/(rho_r*(s_r - u_r))))

    # Compute each family of waves
    wave[:,0,:] = q_hat_l - q_l
    wave[:,1,:] = q_hat_r - q_hat_l
    wave[:,2,:] = q_r - q_hat_r
    
    # Compute variations
    s_index = np.zeros((2,num_rp))
    for m in range(num_eqn):
        for mw in range(num_waves):
            s_index[0,:] = s[mw,:]
            amdq[m,:] += np.min(s_index,axis=0) * wave[m,mw,:]
            apdq[m,:] += np.max(s_index,axis=0) * wave[m,mw,:]
            
    return wave, s, amdq, apdq

def euler_exact_1D(q_l,q_r,aux_l,aux_r,problem_data):
    r"""
    Exact euler Riemann solver
    
    .. warning::
        This solver has not been implemented.
    
    """
    raise NotImplementedError("The exact Riemann solver has not been implemented.")

def roe_averages(q_l,q_r,problem_data):
    # Solver parameters
    gamma1 = problem_data['gamma1']

    # Calculate Roe averages
    rhsqrtl = np.sqrt(q_l[0,...])
    rhsqrtr = np.sqrt(q_r[0,...])
    pl = gamma1 * (q_l[2,...] - 0.5 * (q_l[1,...]**2) / q_l[0,...])
    pr = gamma1 * (q_r[2,...] - 0.5 * (q_r[1,...]**2) / q_r[0,...])
    rhsq2 = rhsqrtl + rhsqrtr
    u = (q_l[1,...] / rhsqrtl + q_r[1,...] / rhsqrtr) / rhsq2
    enthalpy = ((q_l[2,...] + pl) / rhsqrtl + (q_r[2,...] + pr) / rhsqrtr) / rhsq2
    a = np.sqrt(gamma1 * (enthalpy - 0.5 * u**2))

    return u, a, enthalpy, pl, pr

def roe_averages_r1d(q_l,q_r,problem_data):
    # Solver parameters
    gamma1 = problem_data['gamma1']
    hv = problem_data['hv']

    # Calculate Roe averages
    rhsqrtl = np.sqrt(q_l[0,...])
    rhsqrtr = np.sqrt(q_r[0,...])
    pl = gamma1 * (q_l[2,...] - 0.5 * (q_l[1,...]**2) / q_l[0,...] - hv*q_l[3,...])
    pr = gamma1 * (q_r[2,...] - 0.5 * (q_r[1,...]**2) / q_r[0,...] - hv*q_r[3,...])
    rhsq2 = rhsqrtl + rhsqrtr
    u = (q_l[1,...] / rhsqrtl + q_r[1,...] / rhsqrtr) / rhsq2
    enthalpy = ((q_l[2,...] + pl) / rhsqrtl + (q_r[2,...] + pr) / rhsqrtr) / rhsq2
    a = np.sqrt(gamma1 * (enthalpy - 0.5 * u**2))

    return u, a, enthalpy, pl, pr