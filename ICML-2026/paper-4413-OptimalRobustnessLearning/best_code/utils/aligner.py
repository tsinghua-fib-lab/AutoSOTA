from abc import ABC, abstractmethod
import numpy as np

class Aligner(ABC):
    def __init__(self, cache_line_size):
        self._cache_line_size = cache_line_size

    @abstractmethod
    def get_aligned_addr(self, address):
        pass

class ShiftAligner(Aligner):
    def __init__(self, cache_line_size=64):
        def is_pow_of_two(x):
            return (x & (x - 1)) == 0
        if not is_pow_of_two(cache_line_size):
            raise ValueError("ShiftAligner: Cache line size ({}) must be a power of two.".format(cache_line_size))

        super().__init__(cache_line_size)
        self._cache_line_bits = int(np.log2(self._cache_line_size))
    
    def get_aligned_addr(self, address):
        return address >> self._cache_line_bits

class NormalAligner(Aligner):
    def __init__(self, cache_line_size=1):
        super().__init__(cache_line_size)
    
    def get_aligned_addr(self, address):
        return address