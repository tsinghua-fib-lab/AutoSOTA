from abc import ABC, abstractmethod
import numpy as np

class HashFunction(ABC):
    def __init__(self, num_buckets):
        self._num_buckets = num_buckets

    @abstractmethod
    def get_bucket_index(self, aligned_address, pc):
        pass

class ShiftHashFunction(HashFunction):
    def __init__(self, num_buckets):
        if num_buckets & (num_buckets - 1) != 0:
            raise ValueError("Cache line size must be power of two.")

        super().__init__(num_buckets)
        self._set_bits = int(np.log2(num_buckets))
    
    def get_bucket_index(self, aligned_address, pc):
        return aligned_address & ((1 << self._set_bits) - 1)


class BrightKiteHashFunction(HashFunction):
    def __init__(self, num_buckets):
        if num_buckets != 100:
            raise ValueError("BrightKiteHashFunction: num_buckets must be 100")
        super().__init__(num_buckets)
    
    def get_bucket_index(self, aligned_address, pc):
        pc = int(pc)
        if pc < 0 or pc > 100:
            raise ValueError("BrightKiteHashFunction: Invalid data")
        return pc

class CitiHashFunction(HashFunction):
    def __init__(self, num_buckets):
        if num_buckets != 12:
            raise ValueError("CitiHashFunction: num_buckets must be 12")
        super().__init__(num_buckets)
    
    def get_bucket_index(self, aligned_address, pc):
        pc = int(pc)
        if pc < 0 or pc > 12:
            raise ValueError("CitiHashFunction: Invalid data")
        return pc