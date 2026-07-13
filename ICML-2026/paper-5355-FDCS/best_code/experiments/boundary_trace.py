"""Boundary-tracing routines for the experiments in

Etam Benger and Katrina Ligett (2026), "Fair decisions from calibrated
scores: Achieving optimal classification while satisfying sufficiency,"
ICML 2026.
"""

import numpy as np
from dataclasses import dataclass


# -----------------------------------------------------------------------------
# Score distributions
# -----------------------------------------------------------------------------

def _preprocess_scores(s, w, s_tol=1e-6, w_tol=1e-6):
  s = np.asarray(s, dtype=float)
  w = np.asarray(w, dtype=float)

  if s.shape != w.shape:
    raise ValueError('s and w must have the same shape')
  if s.ndim != 1:
    raise ValueError('s and w must be 1-dimensional')
  if len(s) < 2:
    raise ValueError('s and w must have at least 2 elements')
  if not np.all((0 <= s) & (s <= 1)):
    raise ValueError('all scores must be in [0, 1]')
  if not np.all(w >= 0):
    raise ValueError('all weights must be nonnegative')
  if np.all(w <= w_tol):
    raise ValueError('all weights are below tolerance')
  if not np.isclose(w.sum(), 1, atol=w_tol):
    raise ValueError('weights must sum to 1')

  s_input = s.copy()
  idx = np.full(len(s), -1, dtype=int)

  order = np.argsort(-s)
  s = s[order]
  w = w[order]

  s_new, w_new = [], []

  for i, si, wi in zip(order, s, w):
    if wi < w_tol:
      continue

    if s_new and np.isclose(si, s_new[-1], atol=s_tol):
      w_new[-1] += wi
      idx[i] = len(s_new) - 1
    else:
      s_new.append(si)
      w_new.append(wi)
      idx[i] = len(s_new) - 1

  s_new = np.array(s_new)
  w_new = np.array(w_new)
  w_new = w_new / w_new.sum()

  if len(s_new) < 2:
    raise ValueError('s must have at least 2 distinct values with nonnegligible weight')

  # Discarded tiny-weight bins do not affect the optimization, but this keeps
  # selection rules well-defined in the original input order.
  for i in np.where(idx < 0)[0]:
    idx[i] = np.argmin(np.abs(s_new - s_input[i]))

  return s_new, w_new, idx


class GroupScoreDistribution:
  def __init__(self, scores, weights, name=None, s_tol=1e-6, w_tol=1e-6):
    self.name = name
    self.s_input = np.asarray(scores, dtype=float)
    self.w_input = np.asarray(weights, dtype=float)

    self.s, self.w, self.idx = _preprocess_scores(
        self.s_input, self.w_input, s_tol=s_tol, w_tol=w_tol)

    self.m = len(self.s)
    self.pi = np.sum(self.s * self.w)

    self.muk = np.cumsum(self.w)
    ws_cumsum = np.cumsum(self.w * self.s)
    self.pk = ws_cumsum / self.muk
    self.qk = np.r_[
        (self.pi - ws_cumsum[:-1]) / (1 - self.muk[:-1]),
        self.s[-1]]
    self.c = ws_cumsum - self.muk * self.s
  
  def __repr__(self):
    name = f'  name    = {self.name},\n' if self.name is not None else ''
    return (
        'GroupScoreDistribution(\n'
        f'{name}'
        f'  scores  = {np.array2string(self.s_input, precision=4, separator=", ")},\n'
        f'  weights = {np.array2string(self.w_input, precision=4, separator=", ")},\n'
        f'  pi      = {self.pi:.4f},\n'
        f'  m       = {self.m}\n'
        ')')

  def q_boundary(self, p, j):
    return (p * (self.pi - self.c[j]) - self.pi * self.s[j]) / (
        p - self.s[j] - self.c[j])

  def boundary(self, res=1e-3):
    p = np.array([self.s[0]])
    q = np.array([self.pi])

    for j in range(1, self.m):
      pR = self.pk[j - 1]
      pL = self.pk[j]

      n = max(1, int(np.ceil((pR - pL) / res)))
      p_ = np.linspace(pR, pL, n, endpoint=False)
      q_ = self.q_boundary(p_, j)

      p = np.r_[p, p_]
      q = np.r_[q, q_]

    p = np.r_[p, self.pi]
    q = np.r_[q, self.s[-1]]

    return p, q
  
  def _boundary_params(self, p, q):
    mu = np.clip((self.pi - q) / (p - q), 1e-6, 1 - 1e-6)

    k = np.where(self.muk >= mu)[0][0]
    ptop = self.s[k] + self.c[k] / mu
    eta = (p - self.pi) / (ptop - self.pi)

    return mu, k, eta

  def is_on_boundary(self, p, q):
    return np.isclose(self._boundary_params(p, q)[2], 1)

  def selection_rule(self, p, q, input_order=True):
    mu, k, eta = self._boundary_params(p, q)

    t = np.zeros(self.m)
    t[:k] = 1
    t[k] = 1 - (self.muk[k] - mu) / self.w[k]

    if not np.isclose(eta, 1):
      t = (1 - eta) * mu + eta * t

    if input_order:
      t = t[self.idx]

    return t


# -----------------------------------------------------------------------------
# Results
# -----------------------------------------------------------------------------

@dataclass
class Optimum:
  value: float
  p: float
  q: float


@dataclass
class TraceResult:
  dist0: GroupScoreDistribution
  dist1: GroupScoreDistribution
  prob_a1: float
  pi_agg: float
  p: np.ndarray
  q: np.ndarray
  max_acc: Optimum
  min_dsep: Optimum

  @property
  def dists(self):
    return [self.dist0, self.dist1]

  def __repr__(self):
    dist0_name = self.dist0.name if self.dist0.name is not None else 'dist0'
    dist1_name = self.dist1.name if self.dist1.name is not None else 'dist1'
    return (
        'TraceResult(\n'
        f'  A=0: {dist0_name},\n'
        f'  A=1: {dist1_name},\n'
        f'  prob_a1 = {self.prob_a1:.4f},\n'
        f'  pi_agg  = {self.pi_agg:.4f},\n'
        f'  max_acc:  value = {self.max_acc.value:.4f},\n'
        f'            p = {self.max_acc.p:.4f}, q = {self.max_acc.q:.4f}\n'
        f'  min_dsep: value = {self.min_dsep.value:.4f},\n'
        f'            p = {self.min_dsep.p:.4f}, q = {self.min_dsep.q:.4f}\n'
        ')')


# -----------------------------------------------------------------------------
# Intersection boundary tracing
# -----------------------------------------------------------------------------

def _solve_quadratic(a, b, c):
  if np.isclose(a, 0):
    if np.isclose(b, 0):
      return []
    return [-c / b]

  d = b**2 - 4 * a * c

  if np.isclose(d, 0):
    return [-b / (2 * a)]
  if d < 0:
    return []

  return [(-b + np.sqrt(d)) / (2 * a),
          (-b - np.sqrt(d)) / (2 * a)]


def _right_endpoints(pL, pR, roots, tol):
  roots = np.sort(np.asarray(roots, dtype=float))
  roots = roots[(roots > pL) & (roots < pR)]

  pts = np.r_[pL, roots, pR]
  pts = pts[np.r_[False, np.diff(pts) > tol]][:-1]

  return np.r_[pts, pR]


def _update_max_acc(dist, pi_agg, pL, pR, j, best, tol):
  D = dist.c[j] - dist.pi + pi_agg
  E = dist.s[j] * (dist.pi - pi_agg) - dist.c[j] * pi_agg

  F = D * (1 - 2 * (dist.pi + dist.s[j])) - 2 * E
  G = 4 * D * dist.pi * dist.s[j] + 2 * E
  H = (2 * E - D) * dist.pi * dist.s[j] - E * (dist.pi + dist.s[j])

  roots = _solve_quadratic(F, G, H)

  for p in _right_endpoints(pL, pR, roots, tol):
    acc = 1 - pi_agg + (2 * p - 1) * (D * p + E) / (
        (p - dist.pi) * (p - dist.s[j]))
    if acc > best.value:
      best = Optimum(acc, p, dist.q_boundary(p, j))

  return best


def _update_min_dsep(dist, pi_agg, K, pL, pR, j, best, tol):
  A = dist.pi - pi_agg - dist.c[j]
  B = pi_agg * (2 * dist.c[j] - dist.pi + dist.s[j] + 1) - dist.s[j] * dist.pi
  C = pi_agg * (dist.s[j] * dist.pi - dist.s[j] - dist.c[j])

  D = -A * (dist.pi + dist.s[j]) - B
  E = 2 * A * dist.pi * dist.s[j] - 2 * C
  F = B * dist.pi * dist.s[j] + C * (dist.pi + dist.s[j])

  roots = _solve_quadratic(D, E, F)

  for p in _right_endpoints(pL, pR, roots, tol):
    dsep = K * (A * p**2 + B * p + C) / (
        (p - dist.pi) * (p - dist.s[j]))
    if dsep < best.value:
      best = Optimum(dsep, p, dist.q_boundary(p, j))

  return best


# Algorithm 2 in the paper.
def compute_pmax_qmin(dist0, dist1):
  dists = [dist0, dist1]
  pi = np.array([dist0.pi, dist1.pi])

  a = np.argmax(pi)
  b = 1 - a

  k = np.where(dists[a].qk[:-1] <= pi[b])[0][0]
  if k == 0:
    p_max = min(dists[b].s[0], dists[a].s[0])
  else:
    p_max = min(
        dists[b].s[0],
        (dists[a].s[k] * (pi[a] - pi[b]) - pi[b] * dists[a].c[k])
        / (pi[a] - dists[a].c[k] - pi[b]))

  k = np.where(dists[b].pk[1:] <= pi[a])[0][0] + 1
  if k == dists[b].m - 1:
    q_min = max(dists[a].s[-1], dists[b].s[-1])
  else:
    q_min = max(
        dists[a].s[-1],
        (pi[b] * (pi[a] - dists[b].s[k]) - dists[b].c[k] * pi[a])
        / (pi[a] - dists[b].c[k] - dists[b].s[k]))

  return p_max, q_min


# Algorithm 1 in the paper.
def trace_intersection(dist0, dist1, prob_a1, tol=1e-6, res=1e-3):
  dists = [dist0, dist1]
  pi = np.array([dist0.pi, dist1.pi])
  pi_agg = (1 - prob_a1) * pi[0] + prob_a1 * pi[1]

  a = np.argmax(pi)
  b = 1 - a

  if not ((pi[b] > dists[a].s[-1]) and (pi[a] < dists[b].s[0])):
    raise ValueError('there is no nontrivial intersection boundary')

  p_max, q_min = compute_pmax_qmin(dist0, dist1)

  acc_pmax = 1 - pi_agg + (2 * p_max - 1) * (pi_agg - pi[b]) / (
      p_max - pi[b])
  acc_qmin = 1 - pi_agg + (2 * pi[a] - 1) * (pi_agg - q_min) / (
      pi[a] - q_min)

  if acc_pmax >= acc_qmin:
    max_acc = Optimum(acc_pmax, p_max, pi[b])
  else:
    max_acc = Optimum(acc_qmin, pi[a], q_min)

  K = 2 * prob_a1 * (1 - prob_a1) * np.abs(pi[0] - pi[1]) / (
      pi_agg * (1 - pi_agg))

  dsep_pmax = K * (pi_agg * (1 - p_max) + pi[b] * (p_max - pi_agg)) / (
      p_max - pi[b])
  dsep_qmin = K * (pi_agg * (1 - pi[a]) + q_min * (pi[a] - pi_agg)) / (
      pi[a] - q_min)

  if dsep_pmax <= dsep_qmin:
    min_dsep = Optimum(dsep_pmax, p_max, pi[b])
  else:
    min_dsep = Optimum(dsep_qmin, pi[a], q_min)

  all_p = np.array([max(pi)])
  all_q = np.array([q_min])
  pL = pi[a]

  while pL < p_max - tol:
    k = np.where(dist0.pk[:-1] > pL)[0][-1] + 1
    l = np.where(dist1.pk[:-1] > pL)[0][-1] + 1
    pR = min(dist0.pk[k - 1], dist1.pk[l - 1], p_max)

    A0, A1 = dist0.pi - dist0.c[k], dist1.pi - dist1.c[l]
    B0, B1 = dist0.s[k] + dist0.c[k], dist1.s[l] + dist1.c[l]
    C0, C1 = dist0.s[k] * dist0.pi, dist1.s[l] * dist1.pi

    A = A0 - A1
    B = C1 - C0 + B0 * A1 - B1 * A0
    C = B1 * C0 - B0 * C1

    roots = _solve_quadratic(A, B, C)

    for pR in _right_endpoints(pL, pR, roots, tol):
      p = (pL + pR) / 2
      a = 0 if A * p**2 + B * p + C > 0 else 1
      j = k if a == 0 else l

      max_acc = _update_max_acc(dists[a], pi_agg, pL, pR, j, max_acc, tol)
      min_dsep = _update_min_dsep(dists[a], pi_agg, K, pL, pR, j, min_dsep, tol)

      n = max(1, int(np.ceil(abs(pR - pL) / res)))
      p_ = np.linspace(pL, pR, n + 1, endpoint=True)[1:]
      q_ = dists[a].q_boundary(p_, j)

      all_p = np.r_[all_p, p_]
      all_q = np.r_[all_q, q_]

      pL = pR

  all_p = np.r_[all_p, p_max]
  all_q = np.r_[all_q, min(pi)]

  return TraceResult(
      dist0=dist0,
      dist1=dist1,
      prob_a1=prob_a1,
      pi_agg=pi_agg,
      p=all_p,
      q=all_q,
      max_acc=max_acc,
      min_dsep=min_dsep)
