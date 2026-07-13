from typing import Dict
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from env import Bandit
from utils import Welford

# Uniform Sampler Baseline Agent
class UniformSampler:
	"""
	Baseline: Uniform sampler for best-arm identification.
	Samples arms in a round-robin (uniform) fashion for T pulls,
	then returns the index of the arm with the largest empirical mean.
	"""
	def __init__(self, bandit: Bandit, T: int):
		self.env = bandit
		self.T = T
		K = bandit.K
		self.trackers = [Welford() for _ in range(K)]
		self.N = np.zeros(K, dtype=int)

		self.n_history = []
		self.rec_history = []

	def run(self) -> int:
		K = self.env.K
		t = 0
		while t < self.T:
			arm_indexes = list(range(K))
			random.shuffle(arm_indexes)
			for i in arm_indexes:
				if t >= self.T:
					break
				r = self.env.pull(i)
				self.trackers[i].update(r)
				self.N[i] += 1
				self.n_history.append(self.N.copy())
				self.rec_history.append(np.argmax([tr.mean for tr in self.trackers]))
				t += 1

		# Return the arm with the highest empirical mean
		return int(np.argmax([tr.mean for tr in self.trackers]))
	
# Upper Confidence Bound Exploration Baseline Agent
class UCBE:
	"""
	Upper Confidence Bound Exploration (UCBE) baseline for best-arm identification.
	Selects arms by maximizing mean + sqrt( a / n ).
	"""
	def __init__(self, bandit: Bandit, T: int, warm_start: int = 1, a: float = 1.0):
		self.env = bandit
		self.T = int(T)
		self.warm_start = int(warm_start)
		self.a = float(a)
		self.n_history = []
		self.rec_history = []

		K = self.env.K
		self.trackers = [Welford() for _ in range(K)]
		self.N = np.zeros(K, dtype=int)

		# Warm-start: pull each arm `warm_start` times (counts toward budget)
		for _ in range(self.warm_start):
			arm_indices = list(range(K))
			random.shuffle(arm_indices)
			for i in arm_indices:
				if self.N.sum() >= self.T:
					break
				r = self.env.pull(i)
				self.trackers[i].update(r)
				self.N[i] += 1
				self.n_history.append(self.N.copy())
				self.rec_history.append(np.argmax([tr.mean for tr in self.trackers]))

	def run(self) -> int:
		K = self.env.K
		# Remaining budget after warm-start pulls
		steps = max(self.T - int(self.N.sum()), 0)

		for _ in range(steps):
			# If any arm has zero pulls, ensure it's sampled
			zeros = np.where(self.N == 0)[0]
			if zeros.size > 0:
				i = int(zeros[0])
			else:
				# compute UCBE score for each arm
				means = np.array([tr.mean for tr in self.trackers])
				bonus = np.sqrt(self.a / np.maximum(self.N, 1))
				scores = means + bonus
				i = int(np.argmax(scores))

			r = self.env.pull(i)
			self.trackers[i].update(r)
			self.N[i] += 1
			self.n_history.append(self.N.copy())
			self.rec_history.append(np.argmax([tr.mean for tr in self.trackers]))

		return int(np.argmax([tr.mean for tr in self.trackers]))
	
	def plot_diagnostics(self):
		K = self.env.K
		n_arr = np.array(self.n_history)
		plt.figure(figsize=(8, 6))
		for k in range(min(K, n_arr.shape[1] if n_arr.ndim > 1 else 1)):
			plt.plot(n_arr[:, k], label=f'Arm {k}')
		plt.title('Number of Pulls per Arm (UCBE)')
		plt.xlabel('Time step')
		plt.legend()
		plt.grid(True, linestyle='--', alpha=0.4)
		plt.tight_layout()
		plt.show()

# Successive Rejects Baseline Agent
class SuccessiveRejects:
	"""
	Baseline: Successive Rejects (Audibert & Bubeck, 2010)
	Fixed-budget best-arm identification for stochastic bandits.
	"""
	def __init__(self, bandit: Bandit, T: int, warm_start: int = 1):
		self.env = bandit
		self.T = int(T)
		self.warm_start = max(1, int(warm_start)) # at least one pull per arm
		K = self.env.K
		self.trackers = [Welford() for _ in range(K)]
		self.N = np.zeros(K, dtype=int)
		self.n_history = []
		self.rec_history = []

		# Optional warm-start: pull each arm warm_start times
		for _ in range(self.warm_start):
			arm_indices = list(range(K))
			random.shuffle(arm_indices)
			for i in arm_indices:
				if self.N.sum() >= self.T:
					break
				r = self.env.pull(i)
				self.trackers[i].update(r)
				self.N[i] += 1
				self.n_history.append(self.N.copy())
				self.rec_history.append(np.argmax([tr.mean for tr in self.trackers]))

	def run(self) -> int:
		K = self.env.K

		# If budget exhausted or trivial K
		if K == 1 or self.N.sum() >= self.T:
			return int(np.argmax([tr.mean for tr in self.trackers]))

		# Effective remaining budget after initial pulls
		T_remaining = max(self.T - int(self.N.sum()), 0)

		# Harmonic-like factor
		BK = 0.5 + np.sum(1.0 / np.arange(2, K + 1))
		# Cumulative per-arm allocation schedule n_r (r = 1..K-1), n_0 = 0
		n = [0]
		for r in range(1, K):
			nr = int(np.ceil((T_remaining / BK) / (K + 1 - r)))
			n.append(nr)

		active = list(range(K))

		# Phases r = 1..K-1
		for r in range(1, K):
			if len(active) <= 1:
				break

			m_r = max(n[r] - n[r - 1], 0)  # additional pulls per active arm this phase

			# Allocate pulls respecting the remaining global budget
			for _ in range(m_r):
				active_shuffled = active.copy()
				random.shuffle(active_shuffled)
				for arm in active_shuffled:
					if self.N.sum() >= self.T:
						# Budget exhausted: pick best-so-far
						return int(np.argmax([tr.mean for tr in self.trackers]))
					rwd = self.env.pull(arm)
					self.trackers[arm].update(rwd)
					self.N[arm] += 1

					self.n_history.append(self.N.copy())
					self.rec_history.append(np.argmax([tr.mean for tr in self.trackers]))

			# Eliminate the empirically worst arm among active
			means_active = [self.trackers[a].mean for a in active]
			worst_pos = int(np.argmin(means_active))
			del active[worst_pos]

		# One arm remains
		if len(active) == 1:
			return int(active[0])

		# Fallback: return best empirical mean
		return int(np.argmax([tr.mean for tr in self.trackers]))

# Sequential Halving Baseline Agent (SH)
class SH:
	"""
	Sequential Halving (Karnin et al., ICML'13) baseline for best-arm identification.
	"""
	def __init__(self, bandit: Bandit, T: int, warm_start: int = 1):
		self.env = bandit
		self.T = int(T)
		self.warm_start = max(1, int(warm_start))
		self.n_history = []
		self.rec_history = []

		K = self.env.K
		self.trackers = [Welford() for _ in range(K)]
		self.N = np.zeros(K, dtype=int)

		self.active = list(range(K))
		self.phase_N = np.zeros(K, dtype=int)
		self.phase_S = np.zeros(K, dtype=float)

		log2K = int(math.ceil(math.log2(K))) if K > 1 else 1
		self.p = np.array([1.0 / (k * log2K) for k in range(1, K + 1)], dtype=float)

		# Warm-start: pull each arm warm_start times (counts toward budget)
		for _ in range(self.warm_start):
			arm_indices = list(range(K))
			random.shuffle(arm_indices)
			for i in arm_indices:
				if self.N.sum() >= self.T:
					break
				r = self.env.pull(i)
				self.trackers[i].update(r)
				self.N[i] += 1
				self.n_history.append(self.N.copy())
				self.rec_history.append(np.argmax([tr.mean for tr in self.trackers]))

	def _current_sums(self) -> np.ndarray:
		return np.array([tr.mean * n for tr, n in zip(self.trackers, self.N)], dtype=float)

	def _decision(self) -> int:
		if len(self.active) == 0:
			return int(np.argmax([tr.mean for tr in self.trackers]))
		if len(self.active) == 1:
			return int(self.active[0])

		current_sums = self._current_sums()
		phase_sums = current_sums - self.phase_S
		phase_counts = self.N - self.phase_N
		phase_means = []
		for arm in self.active:
			denom = phase_counts[arm]
			if denom <= 0:
				phase_means.append(-np.inf)
			else:
				phase_means.append(phase_sums[arm] / denom)
		best_idx = int(np.argmax(phase_means))
		return int(self.active[best_idx])

	def run(self) -> int:
		K = self.env.K
		if K == 1 or self.N.sum() >= self.T:
			return self._decision()

		steps = max(self.T - int(self.N.sum()), 0)

		for _ in range(steps):
			j = len(self.active)
			if j > 2:
				phase_counts = self.N - self.phase_N
				active_counts = phase_counts[self.active]
				min_phase = int(np.min(active_counts)) if active_counts.size > 0 else 0
				threshold = int(math.floor(self.p[j - 1] * self.T))
				if min_phase >= threshold and min_phase > 0:
					current_sums = self._current_sums()
					phase_sums = current_sums - self.phase_S
					phase_means = []
					for arm in self.active:
						denom = phase_counts[arm]
						if denom <= 0:
							phase_means.append(-np.inf)
						else:
							phase_means.append(phase_sums[arm] / denom)
					order = np.argsort(phase_means)
					keep_size = int(math.ceil(j / 2))
					keep_idx = order[-keep_size:]
					self.active = [self.active[idx] for idx in keep_idx]
					self.phase_N = self.N.copy()
					self.phase_S = current_sums

			if len(self.active) == 0:
				return int(np.argmax([tr.mean for tr in self.trackers]))

			phase_counts = self.N - self.phase_N
			active_counts = [phase_counts[a] for a in self.active]
			arm = int(self.active[int(np.argmin(active_counts))])

			r = self.env.pull(arm)
			self.trackers[arm].update(r)
			self.N[arm] += 1
			self.n_history.append(self.N.copy())
			self.rec_history.append(np.argmax([tr.mean for tr in self.trackers]))

		return self._decision()

# Continuous Rejects Baseline Agent (CR)
class ContinuousRejects:
	"""
	Continuous Rejects (CR) baseline for best-arm identification.
	mode='A' for aggressive (average) and mode='C' for conservative (min).
	"""
	def __init__(self, bandit: Bandit, T: int, mode: str = 'A', warm_start: int = 1):
		self.env = bandit
		self.T = int(T)
		self.mode = str(mode).upper()
		if self.mode not in ['A', 'C']:
			raise ValueError("mode must be 'A' or 'C'")
		self.warm_start = max(1, int(warm_start))
		self.n_history = []
		self.rec_history = []

		K = self.env.K
		self.trackers = [Welford() for _ in range(K)]
		self.N = np.zeros(K, dtype=int)
		self.active = list(range(K))

		# Warm-start: pull each arm warm_start times (counts toward budget)
		for _ in range(self.warm_start):
			arm_indices = list(range(K))
			random.shuffle(arm_indices)
			for i in arm_indices:
				if self.N.sum() >= self.T:
					break
				r = self.env.pull(i)
				self.trackers[i].update(r)
				self.N[i] += 1
				self.n_history.append(self.N.copy())
				self.rec_history.append(np.argmax([tr.mean for tr in self.trackers]))

	@staticmethod
	def _logbar(j: int) -> float:
		if j <= 1:
			return 0.5
		return 0.5 + sum(1.0 / k for k in range(2, j + 1))

	@staticmethod
	def _G(beta: float) -> float:
		if beta <= 0:
			return float('inf')
		return 1.0 / math.sqrt(beta) - 1.0

	def _decision(self) -> int:
		if len(self.active) == 0:
			return int(np.argmax([tr.mean for tr in self.trackers]))
		means = np.array([tr.mean for tr in self.trackers], dtype=float)
		active_means = [means[a] for a in self.active]
		return int(self.active[int(np.argmax(active_means))])

	def _next_sample(self) -> int:
		if len(self.active) == 0:
			return int(np.argmax([tr.mean for tr in self.trackers]))

		K = self.env.K
		active_set = set(self.active)
		notC = [i for i in range(K) if i not in active_set]
		j = len(self.active)

		means = np.array([tr.mean for tr in self.trackers], dtype=float)
		l = self.active[int(np.argmin([means[k] for k in self.active]))]

		total_active = int(self.N[self.active].sum()) if self.active else 0
		total_not_active = int(self.N[notC].sum()) if notC else 0
		denom = float(self.T - total_not_active)
		beta = self._logbar(j) * float(total_active) / denom if denom > 0 else float('inf')

		if j > 2:
			elim_ok = (len(notC) == 0)
			if not elim_ok and len(notC) > 0:
				max_notC = int(np.max(self.N[notC]))
				elim_ok = self.N[l] > max_notC

			if elim_ok:
				other_means = [means[k] for k in self.active if k != l]
				if len(other_means) > 0:
					if self.mode == 'A':
						gap = float(np.mean(other_means) - means[l])
					else:
						gap = float(np.min(other_means) - means[l])
					if gap > self._G(beta):
						self.active = [a for a in self.active if a != l]

		if len(self.active) == 0:
			return int(np.argmax(means))

		counts_active = [self.N[a] for a in self.active]
		return int(self.active[int(np.argmin(counts_active))])

	def run(self) -> int:
		K = self.env.K
		if K == 1 or self.N.sum() >= self.T:
			return self._decision()

		steps = max(self.T - int(self.N.sum()), 0)

		for _ in range(steps):
			arm = self._next_sample()
			r = self.env.pull(arm)
			self.trackers[arm].update(r)
			self.N[arm] += 1
			self.n_history.append(self.N.copy())
			self.rec_history.append(np.argmax([tr.mean for tr in self.trackers]))

		return self._decision()

# Convenience wrappers for CR-A / CR-C
class CRA(ContinuousRejects):
	"""
	Continuous Rejects with aggressive rate (CR-A).
	"""
	def __init__(self, bandit: Bandit, T: int, warm_start: int = 1):
		super().__init__(bandit=bandit, T=T, mode='A', warm_start=warm_start)

class CRC(ContinuousRejects):
	"""
	Continuous Rejects with conservative rate (CR-C).
	"""
	def __init__(self, bandit: Bandit, T: int, warm_start: int = 1):
		super().__init__(bandit=bandit, T=T, mode='C', warm_start=warm_start)

# UGapE Baseline Agent
class UGapE:
	"""
	UGapE (fixed-budget) baseline for best-arm identification.
	Selects between the current best candidate and its closest competitor
	using confidence-based gap indices.
	"""
	def __init__(self, bandit: Bandit, T: int, warm_start: int = 1, a: float = 1.0):
		self.env = bandit
		self.T = int(T)
		self.warm_start = max(1, int(warm_start))
		self.a = float(a)
		self.n_history = []
		self.rec_history = []

		K = self.env.K
		self.trackers = [Welford() for _ in range(K)]
		self.N = np.zeros(K, dtype=int)

		# Warm-start: pull each arm warm_start times (counts toward budget)
		for _ in range(self.warm_start):
			arm_indices = list(range(K))
			random.shuffle(arm_indices)
			for i in arm_indices:
				if self.N.sum() >= self.T:
					break
				r = self.env.pull(i)
				self.trackers[i].update(r)
				self.N[i] += 1
				self.n_history.append(self.N.copy())
				self.rec_history.append(np.argmax([tr.mean for tr in self.trackers]))

	def _beta(self) -> np.ndarray:
		return np.sqrt(self.a / np.maximum(self.N, 1))

	def run(self) -> int:
		K = self.env.K
		if K == 1 or self.N.sum() >= self.T:
			return int(np.argmax([tr.mean for tr in self.trackers]))

		steps = max(self.T - int(self.N.sum()), 0)

		for _ in range(steps):
			means = np.array([tr.mean for tr in self.trackers])
			beta = self._beta()
			U = means + beta
			L = means - beta

			max_u_idx = int(np.argmax(U))
			max_u = U[max_u_idx]
			second_max_u = np.max(np.delete(U, max_u_idx)) if K > 1 else max_u

			B = np.empty(K)
			for k in range(K):
				best_other = max_u if k != max_u_idx else second_max_u
				B[k] = best_other - L[k]

			J = int(np.argmin(B))
			U_masked = U.copy()
			U_masked[J] = -np.inf
			I = int(np.argmax(U_masked)) if K > 1 else J

			if beta[J] >= beta[I]:
				arm = J
			else:
				arm = I

			r = self.env.pull(arm)
			self.trackers[arm].update(r)
			self.N[arm] += 1
			self.n_history.append(self.N.copy())
			self.rec_history.append(np.argmax([tr.mean for tr in self.trackers]))

		return int(np.argmax([tr.mean for tr in self.trackers]))
