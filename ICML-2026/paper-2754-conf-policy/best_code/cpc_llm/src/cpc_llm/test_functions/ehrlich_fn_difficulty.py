from holo.test_functions.closed_form import Ehrlich

import json
import numpy as np
import pandas as pd
import s3fs
import torch
from tqdm import tqdm
from typing import List
from functools import reduce


class DMP:
    def __init__(self, transition_matrix: torch.Tensor):
        self.transition_matrix = transition_matrix
        self.can_transition = (self.transition_matrix > 0).double().numpy()
        self.n = self.transition_matrix.shape[0]
        # seq len -> last state -> count of num. seqs of that len ending in that state
        self.memos = {1: np.array([1 for _ in range(self.n)])}
        self.cache = {}
        self.matrix_mult_cache = {1: self.can_transition.T}

    def can_transition_matrix_power(self, power):
        if power in self.matrix_mult_cache:
            return self.matrix_mult_cache[power]
        # find max power
        for p in range(max(self.matrix_mult_cache.keys()) + 1, power + 1):
            self.matrix_mult_cache[p] = (
                self.matrix_mult_cache[p - 1] @ self.can_transition.T
            )
        return self.matrix_mult_cache[power]

    def num_sequences(self, seq_len, end_with=None, start_with=None):
        if seq_len not in self.memos:
            # find largest value in cache that is populated
            for i in range(max(self.memos.keys()) + 1, seq_len + 1):
                prev_num_seqs = self.memos[i - 1]
                num_seqs = self.can_transition.T @ prev_num_seqs
                self.memos[i] = num_seqs

        if end_with is None and start_with is None:
            return np.sum(self.memos[seq_len])
        elif start_with is None and end_with is not None:
            return self.memos[seq_len][end_with]
        else:
            # start_with is not None
            if (seq_len, start_with, end_with) in self.cache:
                return self.cache[(seq_len, start_with, end_with)]
            prod_vec = (
                self.can_transition_matrix_power(seq_len - 1)
                @ np.eye(self.n)[start_with]
            )
            # fill in cache
            for end in range(self.n):
                self.cache[(seq_len, start_with, end)] = prod_vec[end]
            self.cache[(seq_len, start_with, None)] = np.sum(prod_vec)
            return self.cache[(seq_len, start_with, end_with)]


# for each possible placement of motifs, find the number of possible sequences in each gap


def is_feasible(seq: List[int], transition_matrix: np.ndarray):
    # checks if the sequence is feasible according to transition matrix, can have None values
    for i in range(1, len(seq)):
        if seq[i] is None or seq[i - 1] is None:
            continue
        if transition_matrix[seq[i - 1]][seq[i]] == 0:
            return False
    return True


def motif_placement_helper(
    current_seq: List[int],
    motif: List[int],
    spacing: List[int],
    transition_matrix: np.ndarray,
):
    # check each possible starting position of the motif
    motif_len = sum(spacing) + 1
    possible_placements = []

    for start_idx in range(0, len(current_seq) - motif_len + 1):
        curr_placement = current_seq.copy()
        curr_idx = start_idx
        if (
            curr_placement[curr_idx] is not None
            and curr_placement[curr_idx] != motif[0]
        ):
            continue
        curr_placement[curr_idx] = motif[0]
        motif_placed = True
        for item, item_spacing in zip(motif[1:], spacing):
            curr_idx += item_spacing
            if (
                curr_placement[curr_idx] is not None
                and curr_placement[curr_idx] != item
            ):
                motif_placed = False
                break
            curr_placement[curr_idx] = item
        if not motif_placed:
            continue
        if not is_feasible(curr_placement, transition_matrix):
            continue
        possible_placements.append(curr_placement)
    return possible_placements


def fill_in_motif_gaps(curr_seq: List[int], dmp: DMP):
    """Return the number of ways to fill in the gaps in a given seq."""
    # first find gaps
    gap_idxs = []
    gap_start = 0 if curr_seq[0] is None else None
    for i in range(1, len(curr_seq)):
        curr = curr_seq[i]
        if curr is None:
            if gap_start is None:
                gap_start = i
        else:
            if gap_start is not None:
                # close prev gap
                gap_idxs.append((gap_start, i - 1))
                gap_start = None
    if gap_start is not None:
        # add last gap
        gap_idxs.append((gap_start, len(curr_seq) - 1))
    # for each gap, count the number of ways we can fill it
    num_placements_per_gap = []
    for gap_start, gap_end in gap_idxs:
        start_with = None if gap_start == 0 else curr_seq[gap_start - 1]
        end_with = None if gap_end == len(curr_seq) - 1 else curr_seq[gap_end + 1]
        inc = 0
        if start_with is not None:
            inc += 1
        if end_with is not None:
            inc += 1
        num_gap_placements = dmp.num_sequences(
            gap_end - gap_start + 1 + inc, start_with=start_with, end_with=end_with
        )
        num_placements_per_gap.append(num_gap_placements)
    total = reduce(lambda x, y: x * y, num_placements_per_gap)
    return total


# get the total number of solutions (w/ regret 0) for a particular set of motifs, spacings,
# and transition matrix
def get_possible_motif_placements(
    seq_len,
    motifs,
    spacings,
    dmp: DMP,
):
    curr_placements = [[None for _ in range(seq_len)]]
    np_tm = dmp.transition_matrix.numpy()
    for motif, spacing in tqdm(
        zip(motifs, spacings), desc="Counting motif placements..."
    ):
        new_placements = []
        for placement in tqdm(
            curr_placements, desc="expanding previous motif placements..."
        ):
            new_placements.extend(
                motif_placement_helper(placement, motif, spacing, np_tm)
            )
        curr_placements = new_placements

    # now for each particular motif placement, figure out how many possibilities there are for
    # non-motif indexes
    total = 0
    for placement in tqdm(
        curr_placements, desc="Counting ways to fill in motif gaps..."
    ):
        total += fill_in_motif_gaps(placement, dmp)
    return total


def main():
    ehrlich_dirs = [
        # "s3://prescient-data-dev/sandbox/chena78/pipelines/ga_ehrlich_s32_d32_m4_l4_steps30_p1000_pm0.005/c4_k4_l32_n1000_pm0.005_steps30",
        "s3://prescient-data-dev/sandbox/chena78/pipelines/ga_ehrlich_s8_d128_m8_l8_steps30_p1000_pm0.005/c8_k8_l128_n1000_pm0.005_steps30",
        # "s3://prescient-data-dev/sandbox/chena78/pipelines/ga_ehrlich_s32_d128_m4_l4_steps30_p1000_pm0.005/c4_k4_l128_n1000_pm0.005_steps30",
        # "s3://prescient-data-dev/sandbox/chena78/pipelines/ga_ehrlich_s32_d128_m8_l8_steps30_p1000_pm0.005/c8_k8_l128_n1000_pm0.005_steps30",
    ]
    s3 = s3fs.S3FileSystem()
    for dir in ehrlich_dirs:
        fp = f"{dir}/ehrlich.jsonl"
        df = pd.read_json(fp, orient="records", lines=True)
        row = df.iloc[0]
        print(row)
        test_fn = Ehrlich(
            num_states=row["num_states"],
            dim=row["dim"],
            num_motifs=row["num_motifs"],
            motif_length=row["motif_length"],
            quantization=row["quantization"],
            noise_std=row["noise_std"],
            negate=row["negate"],
            random_seed=int(row["random_seed"]),
        )
        motifs = [t.tolist() for t in test_fn.motifs]
        spacings = [t.tolist() for t in test_fn.spacings]
        dmp = DMP(test_fn.transition_matrix)
        num_sols = get_possible_motif_placements(test_fn.dim, motifs, spacings, dmp)
        num_feasible_seqs = dmp.num_sequences(row["dim"])
        total_seqs = int(test_fn.transition_matrix.shape[0]) ** int(row["dim"])
        pc = num_sols / total_seqs
        print(
            f"Proportion of 0-regret solutions: {num_sols} / {total_seqs} ({pc * 100}%)"
        )
        output_dict = {
            "num_optimal_solutions": int(num_sols),
            "num_feasible_sequences": int(num_feasible_seqs),
            "total_possible_sequences": total_seqs,
            "proportion_optimal_solutions": pc,
            "proportion_feasible_sequences": num_feasible_seqs / total_seqs,
            "proportion_optimal_solutions_of_feasible": num_sols / num_feasible_seqs,
        }
        with s3.open(f"{dir}/difficulty.json", "w") as f:
            f.write(json.dumps(output_dict))


main()
