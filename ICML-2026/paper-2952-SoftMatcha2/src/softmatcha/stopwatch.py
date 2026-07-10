"""
Stopwatch module including some useful global objects.

Example:
	>>> from softmatcha import stopwatch
	>>> n = 0
	>>> for i in range(100)
			with stopwatch.timers["sum"]:
				n += i
			m = 1
			for j in range(i):
				with stopwatch.timers["factorial"]:
					m *= j + 1
	>>> print(stopwatch.timers.elapsed_time)
"""

from __future__ import annotations

import contextlib
import time
from collections import defaultdict
from typing import Any, Callable, TypeVar


class Stopwatch:
	"""Stopwatch class to measure the elapsed time.

	Example:
		>>> stopwatch = Stopwatch()
		>>> for i in range(10):
				with stopwatch:
					time.sleep(1)
		>>> print(f"{stopwatch.elapsed_time:.3f}")
		10.000
		>>> print(f"{stopwatch.ncalls}")
		10
	"""

	def __init__(self) -> None:
		self.reset()

	def reset(self) -> None:
		"""Reset the stopwatch."""
		self._acc_time: float = 0.0
		self._acc_ncalls: int = 0
		self._start: float = 0.0

	def __enter__(self) -> None:
		self._start = time.perf_counter()

	def __exit__(self, *args) -> None:
		self._acc_time += time.perf_counter() - self._start
		self._acc_ncalls += 1

	@property
	def elpased_time(self) -> float:
		"""Return the total elapsed time."""
		return self._acc_time

	@property
	def ncalls(self) -> int:
		"""Return the number of calls."""
		return self._acc_ncalls


T_IN = TypeVar("T_IN")
T_OUT = TypeVar("T_OUT")


class StopwatchDict(defaultdict[str, Stopwatch | contextlib.nullcontext]):
	"""A dictionary of the :class:`Stopwatch` class.

	Example:
		>>> stopwatches = StopwatchDict()
		>>> for i in range(10):
				with stopwatches["A"]:
					time.sleep(1)
			for i in range(3):
				with stopwatches["B"]:
					time.sleep(1)
		>>> print(f"{stopwatches.total}")
		{"A": 10.000, "B": 3.000}
	"""

	def __init__(self) -> None:
		super().__init__(Stopwatch)

	def reset(self, profile: bool = False) -> None:
		"""Reset all stopwatches."""
		if profile:
			self.default_factory = Stopwatch
			for t in self.values():
				if isinstance(t, Stopwatch):
					t.reset()
		else:
			self.default_factory = contextlib.nullcontext
			for k in list(self.keys()):
				del self[k]

	def __call__(self, key: str, generator: bool = False) -> Callable[[Callable], Any]:
		"""Decorator"""

		def _measure(func: Callable) -> Callable:
			if generator:
				def _wrap(*args, **kwargs):
					g = func(*args, **kwargs)
					try:
						while True:
							with timers[key]:
								value = next(g)
							yield value
					except StopIteration:
						return
			else:
				def _wrap(*args, **kwargs):
					with timers[key]:
						return func(*args, **kwargs)

			return _wrap

		return _measure

	@property
	def elapsed_time(self) -> dict[str, float]:
		"""Return the total elapsed time."""
		return {k: round(v.elpased_time, 5) for k, v in self.items() if isinstance(v, Stopwatch)}

	@property
	def ncalls(self) -> dict[str, int]:
		"""Return the number of calls."""
		return {k: v.ncalls for k, v in self.items() if isinstance(v, Stopwatch)}


timers = StopwatchDict()
