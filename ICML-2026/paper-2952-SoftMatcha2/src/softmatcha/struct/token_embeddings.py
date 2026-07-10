from dataclasses import dataclass

from softmatcha.typing import NDArrayF32


@dataclass
class TokenEmbeddings:
		"""TokenEmbeddings stores a sequence of tokens and their embedding vectors.

		tokens (list[int]): A sequence of tokens.
		embeddings (NDArrayF32): D-dimensional embedding vectors of tokens of shape `(L, D)`,
			where `L` indicates the sequence length.
		"""

		tokens: list[int]
		embeddings: NDArrayF32

		def __len__(self) -> int:
				return len(self.tokens)
