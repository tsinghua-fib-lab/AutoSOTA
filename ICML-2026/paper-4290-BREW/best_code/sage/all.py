"""
SageMath compatibility shim for BREW watermarking.

Provides the minimal Sage API used by watermark/BREW/BREW.py:
  - GF(2)        : binary finite field
  - codes.BCHCode: BCH code construction
  - vector       : vector constructor

Uses galois for finite field arithmetic and BCH encoding/decoding.
"""

import galois
import numpy as np

# ---------------------------------------------------------------------------
# Expose the Sage-level names that BREW imports via "from sage.all import *"
# ---------------------------------------------------------------------------
__all__ = ["GF", "codes", "vector"]


# ===================================================================
# GF(q) - finite field constructor
# ===================================================================
def GF(q):
    """Return the finite (Galois) field of order q."""
    return galois.GF(q)


# ===================================================================
# Sage-like vector
# ===================================================================
class _SageVector:
    """Minimal Sage-like vector wrapper around a galois FieldArray or list."""

    __slots__ = ("_field", "_array")

    def __init__(self, field, data):
        self._field = field
        if isinstance(data, _SageVector):
            self._array = data._array.copy()
        elif isinstance(data, galois.FieldArray):
            self._array = data.copy()
        elif isinstance(data, np.ndarray):
            self._array = field(data)
        else:
            self._array = field(list(data))

    def __iter__(self):
        return iter(int(x) for x in self._array)

    def __len__(self):
        return len(self._array)

    def __getitem__(self, idx):
        val = self._array[idx]
        return int(val) if isinstance(val, galois.FieldArray) else val

    def __repr__(self):
        return "(" + ", ".join(str(int(b)) for b in self._array) + ")"

    def __eq__(self, other):
        if isinstance(other, _SageVector):
            return np.array_equal(self._array, other._array)
        return False

    def __hash__(self):
        return hash(tuple(int(b) for b in self._array))

    def list(self):
        return [int(b) for b in self._array]

    @property
    def field(self):
        return self._field

    @property
    def array(self):
        return self._array


def vector(field, data):
    """Sage-compatible vector constructor."""
    return _SageVector(field, data)


# ===================================================================
# Ambient space (GF(2)^n)
# ===================================================================
class _AmbientSpace:
    """Wraps GF(2)^n so that calling A(v) casts v to the ambient space."""

    __slots__ = ("_field", "_n")

    def __init__(self, field, n):
        self._field = field
        self._n = n

    def __call__(self, vec):
        """Cast a vector into this ambient space."""
        if isinstance(vec, _SageVector):
            data = list(vec)
        elif isinstance(vec, galois.FieldArray):
            data = [int(x) for x in vec]
        else:
            data = list(vec)
        if len(data) < self._n:
            data = data + [0] * (self._n - len(data))
        elif len(data) > self._n:
            data = data[:self._n]
        return _SageVector(self._field, data)

    def __repr__(self):
        return f"AmbientSpace(GF(2)^{self._n})"


# ===================================================================
# DecodingError
# ===================================================================
class DecodingError(Exception):
    """Raised when BCH decoding fails (errors exceed correction radius)."""
    pass


# ===================================================================
# BCH decoder wrapper
# ===================================================================
class _BCHDecoder:
    """Wraps galois BCH decoder for Sage-compatible decode_to_code()."""

    __slots__ = ("_bch", "_field")

    def __init__(self, bch_code, field):
        self._bch = bch_code
        self._field = field

    def decode_to_code(self, received_vector):
        """
        Decode a received vector to the nearest codeword.
        Raises DecodingError if errors exceed correction capability.
        """
        if isinstance(received_vector, _SageVector):
            bits = [int(b) for b in received_vector]
        elif isinstance(received_vector, galois.FieldArray):
            bits = [int(b) for b in received_vector]
        else:
            bits = list(received_vector)

        rx = self._field(bits)

        try:
            decoded_msg = self._bch.decode(rx)
            recoded = self._bch.encode(decoded_msg, output="codeword")
            return _SageVector(self._field, recoded)
        except Exception:
            raise DecodingError(
                "Decoding failed because the number of errors exceeded the decoding radius"
            )


# ===================================================================
# BCH Code wrapper (Sage-compatible)
# ===================================================================
class _BCHCode:
    """
    Sage-compatible BCH code wrapper.

    Usage matches codes.BCHCode(GF(2), 31, 15).
    """

    __slots__ = ("_bch", "_field", "_n", "_k")

    def __init__(self, field, n, designed_distance):
        d = int(designed_distance)
        try:
            self._bch = galois.BCH(n, d=d, field=field)
        except Exception:
            raise ValueError(
                f"Cannot construct BCH code with n={n}, d={d}"
            )
        self._field = field
        self._n = self._bch.n
        self._k = self._bch.k

    def dimension(self):
        return self._k

    def encode(self, message_vector):
        """Encode a message vector to a codeword."""
        if isinstance(message_vector, _SageVector):
            msg_bits = [int(b) for b in message_vector]
        elif isinstance(message_vector, galois.FieldArray):
            msg_bits = [int(b) for b in message_vector]
        else:
            msg_bits = list(message_vector)

        msg_array = self._field(msg_bits)
        codeword_array = self._bch.encode(msg_array, output="codeword")
        return _SageVector(self._field, codeword_array)

    def ambient_space(self):
        return _AmbientSpace(self._field, self._n)

    def decoder(self):
        return _BCHDecoder(self._bch, self._field)

    def __repr__(self):
        return f"BCHCode(n={self._n}, k={self._k})"


# ===================================================================
# codes module (Sage-compatible)
# ===================================================================
class _CodesModule:
    """Sage-compatible codes module with BCHCode."""

    @staticmethod
    def BCHCode(field, length, designed_distance):
        return _BCHCode(field, length, designed_distance)


codes = _CodesModule()
