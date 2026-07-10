from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from softmatcha.typing import NDArrayF32


@dataclass
class Pattern:
	"""Pattern class stores the lexical and semantic pattern information.

	tokens (list[int]): A sequence of tokens.
	embeddings (NDArrayF32): D-dimensional token embeddings.
	thresholds (NDArrayF32): Thresholds for matched scores.
	  The range of matched scores are `[0,1]`, and if `score > threshold`,
	  the text token is regarded as matched the pattern token.
	"""

	tokens: list[int]
	embeddings: NDArrayF32
	thresholds: NDArrayF32

	def __len__(self) -> int:
		return len(self.tokens)

	@classmethod
	def build(
		cls, tokens: list[int], embeddings: NDArrayF32, thresholds: list[float]
	) -> Pattern:
		"""Build a pattern class.

		Args:
			tokens (list[int]): A sequence of tokens.
			embeddings (NDArrayF32): D-dimensional token embeddings.
			thresholds (list[float]): Thresholds for matched scores.
			  The range of matched scores are `[0,1]`, and if `score > threshold`,
			  the text token is regarded as matched the pattern token.

		Returns:
			Pattern: This class.
		"""
		return cls(
			tokens,
			embeddings,
			np.array(thresholds, dtype=np.float32),
		)
