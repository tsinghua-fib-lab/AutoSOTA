import json
import math
from itertools import chain

import numpy as np
from numba import njit


@njit
def _hamming_fast(block1, block2):
    assert len(block1) == len(block2)
    distance = 0
    for b1, b2 in zip(block1, block2):
        distance += b1 ^ b2
    return distance

@njit
def _block_edit_distances_fast(X, Y, m):
    n = len(X)
    k = len(Y)

    dp = [j * m for j in range(k + 1)]
    
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i * m
        
        for j in range(1, k + 1):
            temp = dp[j]
            
            cost_del = dp[j] + m          
            cost_ins = dp[j-1] + m     
            cost_sub = prev + _hamming_fast(X[i-1], Y[j-1])
            
            dp[j] = min(cost_del, cost_ins, cost_sub)
            prev = temp
    
    return dp


class AdaptiveBitSequenceAlignment:
    def __init__(self, args):
        self._block_size = args.watermark_block_size

        try:
            with open(f"./watermark/ber_map/ber_map_{self._block_size}.json", "r") as f:
                self._ber_map = json.load(f)
                self._ber_means = {int(k): v["mean"] for k, v in self._ber_map.items()}
                self._ber_stds = {int(k): v["std"] for k, v in self._ber_map.items()}
                self._ber_mean_outbound = min(self._ber_means.values())
                self._ber_std_outbound = min(self._ber_stds.values())
        except Exception as e:
            raise FileNotFoundError(f"BER map file not found for block size {self._block_size}")

        try:
            with open(f"./watermark/ber_map/ber_map_1.json", "r") as f:
                self._ber_map_1 = json.load(f)
                self._ber_means_1 = {int(k): v["mean"] for k, v in self._ber_map_1.items()}
                self._ber_stds_1 = {int(k): v["std"] for k, v in self._ber_map_1.items()}
                self._ber_mean_outbound_1 = min(self._ber_means_1.values())
                self._ber_std_outbound_1 = min(self._ber_stds_1.values())
        except Exception as e:
            raise FileNotFoundError(f"BER map file not found for block size 1")

    def _block_edit_distances(self, X, Y, m):
        X, Y = np.array(X), np.array(Y)
        return _block_edit_distances_fast(X, Y, m)

    def block_edit_rate(self, X, Y):
        distance = self._block_edit_distances(X, Y)[-1]
        max_length = max(len(X), len(Y)) * self._block_size
        return distance / max_length if max_length > 0 else 0.0
    
    def compute_score(self, 
                      extracted_bit_sequence, 
                      secret_bit_sequence, 
                      lower_ratio=0.5, 
                      upper_ratio=1.5,
                      criterion="block_edit_distance"):
        # If no bits were extracted (e.g. text too short), return a default low score
        if not extracted_bit_sequence:
            return 0.0
        
        if criterion == "exact_match":
            num_matches = 0
            min_len = 0
            for e_block, s_block in zip(extracted_bit_sequence, secret_bit_sequence):
                num_matches += sum(eb == sb for eb, sb in zip(e_block, s_block))
                min_len += len(e_block)
            p0 = 0.5
            p_hat = num_matches / min_len if min_len > 0 else 0.0
            standard_error = np.sqrt(p0 * (1 - p0) / min_len) if min_len > 0 else 0.0
            z_score = (p_hat - p0) / standard_error if standard_error > 0 else 0.0
            return z_score

        
        if criterion == "block_edit_distance":
            num_blocks = len(extracted_bit_sequence)
            lower_num_blocks = max(1, math.ceil(lower_ratio * num_blocks))
            upper_num_blocks = math.ceil(upper_ratio * num_blocks)
            beds = self._block_edit_distances(
                extracted_bit_sequence,
                secret_bit_sequence[:upper_num_blocks],
                self._block_size
            )
        elif criterion == "edit_distance":
            num_blocks = len(extracted_bit_sequence) * self._block_size
            lower_num_blocks = max(1, math.ceil(lower_ratio * num_blocks))
            upper_num_blocks = math.ceil(upper_ratio * num_blocks)
            beds = self._block_edit_distances(
                [[b] for b in list(chain.from_iterable(extracted_bit_sequence))],
                [[b] for b in list(chain.from_iterable(secret_bit_sequence[:upper_num_blocks]))],
                1
            )
        else:
            raise ValueError(f"Unsupported criterion: {criterion}")
        
        max_z_score = -float('inf')
        for possible_num_blocks in range(lower_num_blocks, upper_num_blocks + 1):
            dp_id = possible_num_blocks
            bed = beds[dp_id]

            max_num_blocks = max(num_blocks, possible_num_blocks)
            ber = bed / (max_num_blocks * self._block_size) if max_num_blocks > 0 else 0.0
            
            if criterion == "block_edit_distance":
                mean = self._ber_means.get(max_num_blocks, self._ber_mean_outbound)
                std = self._ber_stds.get(max_num_blocks, self._ber_std_outbound)
            elif criterion == "edit_distance":
                mean = self._ber_means_1.get(max_num_blocks, self._ber_mean_outbound_1)
                std = self._ber_stds_1.get(max_num_blocks, self._ber_std_outbound_1)

            if std > 1e-10:
                z_score = (mean - ber) / std
            else:
                z_score = 0.0
            
            if z_score > max_z_score:
                max_z_score = z_score
        
        return max_z_score


if __name__ == "__main__":
    class MockArgs:
        def __init__(self):
            self.watermark_block_size = 4

    ABSA = AdaptiveBitSequenceAlignment(MockArgs())

    X = [[0, 0, 1, 0], [1, 0, 1, 1], [0, 1, 0, 1], [1, 0, 1, 0]]
    Y = [[0, 0, 1, 0], [1, 0, 1, 1], [0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 1, 1, 0]]

    print(ABSA.compute_score(X, Y, lower_ratio=0.5, upper_ratio=1.5))