# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
# https://github.com/tensorflow/privacy/blob/master/tensorflow_privacy/privacy/analysis/gdp_accountant.py
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,  software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,  either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
r"""Implements privacy accounting for Gaussian Differential Privacy.

Applies the Dual and Central Limit Theorem (CLT) to estimate privacy budget of
an iterated subsampled Gaussian Mechanism (by either uniform or Poisson
subsampling).
"""

import numpy as np
import math
from scipy import optimize
from scipy import stats


def compute_mu_uniform(epoch, noise_multi, n, batch_size):
  """Compute mu from uniform subsampling."""

  t = epoch * n / batch_size
  c = batch_size * np.sqrt(t) / n
  return np.sqrt(2) * c * np.sqrt(
      np.exp(noise_multi**(-2)) * stats.norm.cdf(1.5 / noise_multi) +
      3 * stats.norm.cdf(-0.5 / noise_multi) - 2)


def compute_mu_poisson(epoch, noise_multi, n, batch_size):
  """Compute mu from Poisson subsampling."""

  t = epoch * n / batch_size
  return np.sqrt(np.exp(noise_multi**(-2)) - 1) * np.sqrt(t) * batch_size / n


def delta_eps_mu(eps, mu):
  """Compute dual between mu-GDP and (epsilon, delta)-DP."""
  return stats.norm.cdf(-eps / mu + mu /
                        2) - np.exp(eps) * stats.norm.cdf(-eps / mu - mu / 2)


def eps_from_mu(mu, delta):
  """Compute epsilon from mu given delta via inverse dual."""

  def f(x):
    """Reversely solve dual by matching delta."""
    return delta_eps_mu(x, mu) - delta

  return optimize.root_scalar(f, bracket=[0, 500], method='brentq').root


def compute_eps_uniform(epoch, noise_multi, n, batch_size, delta):
  """Compute epsilon given delta from inverse dual of uniform subsampling."""

  return eps_from_mu(
      compute_mu_uniform(epoch, noise_multi, n, batch_size), delta)


def compute_eps_poisson(epoch, noise_multi, n, batch_size, delta):
  """Compute epsilon given delta from inverse dual of Poisson subsampling."""

  return eps_from_mu(
      compute_mu_poisson(epoch, noise_multi, n, batch_size), delta)

from math import exp, sqrt
from scipy.special import erf

def calibrateAnalyticGaussianMechanism(epsilon, delta, GS, tol = 1.e-12):
    """ Calibrate a Gaussian perturbation for differential privacy using the analytic Gaussian mechanism of [Balle and Wang, ICML'18]

    Arguments:
    epsilon : target epsilon (epsilon > 0)
    delta : target delta (0 < delta < 1)
    GS : upper bound on L2 global sensitivity (GS >= 0)
    tol : error tolerance for binary search (tol > 0)

    Output:
    sigma : standard deviation of Gaussian noise needed to achieve (epsilon,delta)-DP under global sensitivity GS
    """

    def Phi(t):
        return 0.5*(1.0 + erf(float(t)/sqrt(2.0)))

    def caseA(epsilon,s):
        return Phi(sqrt(epsilon*s)) - exp(epsilon)*Phi(-sqrt(epsilon*(s+2.0)))

    def caseB(epsilon,s):
        return Phi(-sqrt(epsilon*s)) - exp(epsilon)*Phi(-sqrt(epsilon*(s+2.0)))

    def doubling_trick(predicate_stop, s_inf, s_sup):
        while(not predicate_stop(s_sup)):
            s_inf = s_sup
            s_sup = 2.0*s_inf
        return s_inf, s_sup

    def binary_search(predicate_stop, predicate_left, s_inf, s_sup):
        s_mid = s_inf + (s_sup-s_inf)/2.0
        while(not predicate_stop(s_mid)):
            if (predicate_left(s_mid)):
                s_sup = s_mid
            else:
                s_inf = s_mid
            s_mid = s_inf + (s_sup-s_inf)/2.0
        return s_mid

    delta_thr = caseA(epsilon, 0.0)

    if (delta == delta_thr):
        alpha = 1.0

    else:
        if (delta > delta_thr):
            predicate_stop_DT = lambda s : caseA(epsilon, s) >= delta
            function_s_to_delta = lambda s : caseA(epsilon, s)
            predicate_left_BS = lambda s : function_s_to_delta(s) > delta
            function_s_to_alpha = lambda s : sqrt(1.0 + s/2.0) - sqrt(s/2.0)

        else:
            predicate_stop_DT = lambda s : caseB(epsilon, s) <= delta
            function_s_to_delta = lambda s : caseB(epsilon, s)
            predicate_left_BS = lambda s : function_s_to_delta(s) < delta
            function_s_to_alpha = lambda s : sqrt(1.0 + s/2.0) + sqrt(s/2.0)

        predicate_stop_BS = lambda s : abs(function_s_to_delta(s) - delta) <= tol

        s_inf, s_sup = doubling_trick(predicate_stop_DT, 0.0, 1.0)
        s_final = binary_search(predicate_stop_BS, predicate_left_BS, s_inf, s_sup)
        alpha = function_s_to_alpha(s_final)
        
    sigma = alpha*GS/sqrt(2.0*epsilon)

    return sigma

def advanced_composition(epsilon, delta, k, tol=1e-6, max_iter=100):
    """Return ε₀ satisfying tight advanced composition equality."""
    A = math.sqrt(2 * k * math.log(1 / delta))

    def f(e0):
        return A * e0 + k * e0 * (math.exp(e0) - 1) - epsilon

    lo, hi = 0.0, epsilon
    for _ in range(max_iter):
        mid = (lo + hi) / 2
        if f(mid) > 0:
            hi = mid
        else:
            lo = mid
        if hi - lo < tol:
            break
    return (lo + hi) / 2
