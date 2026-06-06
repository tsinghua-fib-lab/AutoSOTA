import math
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import floyd_warshall, connected_components
import numpy as np
import bisect
from typing import List, Tuple
from quickselect.hoare import nth_smallest, select

# -------------------------------------------------------------------
# Set-Based Datasets
# -------------------------------------------------------------------
#Given a set of numbers X and a target integer T, the problem is to decide if there exists a subset Y⊆X 
#such that the sum of the elements in Y equals T. An NP-complete problem
def set_subset_sum_decision(x, T):
    possible_sums = {0}
    for val in x:
        possible_sums |= {s + val for s in possible_sums}
    return 1 if T in possible_sums else 0

class SubsetSumDecisionDataset(Dataset):
    """
    Dataset for subset-sum decision with a dynamic target generation strategy to guarantee
    a balanced distribution of positive and negative samples,
    even for large lengths and small value ranges.
    """
    def __init__(
        self,
        n_samples: int = 1000,
        length_range: tuple[int,int] = (10,10),
        value_range: tuple[int,int] = (-2,2),
        noise_prob: float = 0.0,
        adversarial_range: tuple[int,int] = (10,30),
        seed: int = 42,
        **kwargs
    ):
        super().__init__()
        random.seed(seed)
        np.random.seed(seed)

        self.n_samples = n_samples
        self.length_range = length_range
        self.value_range = value_range
        self.noise_prob = noise_prob
        self.adversarial_range = adversarial_range
        
        self.data = []
        for _ in range(n_samples):
            n = random.randint(*self.length_range)
            x = [random.randint(*self.value_range) for _ in range(n)]
            x_sorted = sorted(x)
            # Calculate all possible sums for this specific set `x`.
            possible_sums = {0}
            for val in x_sorted:
                possible_sums |= {s + val for s in possible_sums}

            # With 50% probability, create a solvable instance (y=1).
            if random.random() < 0.5:
                # Pick a random achievable sum as the target.
                T = random.choice(list(possible_sums))
                y = 1
            # With 50% probability, create an unsolvable instance (y=0).
            else:
                # Find a target `T` that is guaranteed NOT to be in possible_sums.
                # A robust way is to find a "hole" in the range of achievable sums.
                min_achievable = min(possible_sums)
                max_achievable = max(possible_sums)
                # Create a set of all integers in the achievable range.
                full_range_set = set(range(min_achievable, max_achievable + 1))
                # Find which integers are NOT possible to sum to.
                unachievable_sums = list(full_range_set - possible_sums)
                
                if unachievable_sums:
                    # If there's a "hole", pick a random one. This creates a hard negative.
                    T = random.choice(unachievable_sums)
                else:
                    # If all sums in the range are possible, pick a target just outside it.
                    T = max_achievable + random.randint(1, 5)
                y = 0
            
            y_t = torch.tensor(y, dtype=torch.float)
            x_t_data_list = [[xi_val, T] for xi_val in x_sorted]
            x_t = torch.tensor(x_t_data_list, dtype=torch.float)

            if self.noise_prob > 0:
                for i in range(x_t.shape[0]):
                    if random.random() < self.noise_prob:
                        x_t[i, 0] += random.randint(*self.adversarial_range)
            self.data.append((x_t, y_t))

    def __len__(self):
        return self.n_samples
    def __getitem__(self, idx):
        return self.data[idx]



def set_max_subset_sum(x):
    """Maximum achievable subset-sum (empty set allowed)."""
    possible = {0}
    for v in x:
        possible |= {s + v for s in possible}
    return max(possible)

def max_subset_mask(x_sorted):
    """
    Return a 0/1 mask (len(x)) marking one maximum-sum subset.

    NB:  With no knapsack-style constraint, the optimal strategy
    is simply “take every non-negative item”.  A full DP traceback
    is overkill and expensive for larger ranges.
    """
    # pick all positive values; if all are negative, choose the
    # empty set (mask of zeros) because the empty sum (0) beats any
    # negative total.
    return torch.tensor([1 if v > 0 else 0 for v in x_sorted],
                        dtype=torch.long)

class MaxSubsetSumDataset(Dataset):
    """
    Args
    ----
    task               (str): 'classification' (default) or 'regression'
    *remaining args*        : unchanged from your original definition
    """
    def __init__(
        self,
        n_samples: int = 1000,
        length_range: tuple[int,int] = (10,10),
        value_range:  tuple[int,int] = (-2,2),
        noise_prob:   float = 0.0,
        adversarial_range: tuple[int,int] = (10,30),
        seed: int = 42,
        classification: bool = True,
        **kwargs
    ):
        super().__init__()
        random.seed(seed)
        self.n_samples, self.classification = n_samples, classification
        self.length_range, self.value_range = length_range, value_range
        self.noise_prob, self.adv_range = noise_prob, adversarial_range

        self.data = []
        for _ in range(n_samples):
            n = random.randint(*self.length_range)
            x = [random.randint(*self.value_range) for _ in range(n)]
            if self.noise_prob:
                x = [xi + random.randint(*self.adv_range)
                     if random.random() < self.noise_prob else xi
                     for xi in x]

            x_sorted = sorted(x)                       # keep order stable
            x_t = torch.tensor(x_sorted, dtype=torch.float).unsqueeze(-1)

            if self.classification: # classification
                y_t = max_subset_mask(x_sorted)         # (n,) long
            else:                                      #regression
                y_val = set_max_subset_sum(x_sorted)
                y_t = torch.tensor([y_val], dtype=torch.float)

            self.data.append((x_t, y_t))

    def __len__(self):      return self.n_samples
    def __getitem__(self,i):return self.data[i]


#0/1 Knapsack Problem (Optimization Version). Given a set of items, each with a specific value and weight, 
#the goal is to select a subset of items whose total weight does not exceed a given capacity C, 
# such that the total value of the selected items is maximized. 
# Each item can either be included exactly once (1) or not at all (0). NP-hard
# 0/1 Knapsack solver with optional item-selection reconstruction
# ----------------------------------------------------------------------
def set_knapsack_01(
    items: List[Tuple[int, int]],      # (value, weight)
    capacity: int,
    return_items: bool = False
):
    """
    Solve 0-1 knapsack with DP.  
    Parameters
    ----------
    items : list of (value, weight)
    capacity : int
    return_items : bool, optional
        If True, also return a list[int] mask of chosen items
        (same ordering as `items`).
    """
    n = len(items)
    # DP table: rows = items processed, cols = capacity
    dp = [[0]*(capacity+1) for _ in range(n+1)]

    # Build DP table
    for i, (val, wt) in enumerate(items, start=1):
        for w in range(capacity+1):
            if wt > w:
                dp[i][w] = dp[i-1][w]
            else:
                dp[i][w] = max(dp[i-1][w], dp[i-1][w-wt] + val)

    best_val = dp[n][capacity]

    if not return_items:
        return best_val

    # ------------------------------------------------------------------
    # Reconstruct chosen item mask
    # ------------------------------------------------------------------
    chosen = [0]*n
    w = capacity
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i-1][w]:
            chosen[i-1] = 1
            w -= items[i-1][1]           # subtract weight of chosen item

    return best_val, chosen


# ----------------------------------------------------------------------
# Dataset generator (regression OR classification)
# ----------------------------------------------------------------------
class KnapsackDataset(Dataset):
    """
    Generate synthetic 0-1 knapsack instances.

    Each item token = [value, weight, capacity]          (input_dim = 3)

    * Regression mode   -> label = optimal total value   (scalar)
    * Classification    -> label = 0/1 mask of items     (length = n)
    """
    def __init__(
        self,
        n_samples: int = 1000,
        length_range: Tuple[int, int] = (10, 10),
        value_range:  Tuple[int, int] = (1, 10),
        weight_range: Tuple[int, int] = (1, 10),
        target_range: Tuple[int, int] = (1, 30),
        noise_prob: float = 0.0,
        adversarial_range: Tuple[int, int] = (10, 30),
        classification: bool = True,
        seed: int = 42,
        **kwargs
    ):
        super().__init__()
        random.seed(seed)

        self.n_samples         = n_samples
        self.length_range      = length_range
        self.value_range       = value_range
        self.weight_range      = weight_range
        self.target_range      = target_range
        self.noise_prob        = noise_prob
        self.adversarial_range = adversarial_range
        self.classification    = classification

        self.data = []
        for _ in range(n_samples):
            n_items = random.randint(*self.length_range)
            capacity = random.randint(*self.target_range)

            items = [
                (random.randint(*self.value_range),
                 random.randint(*self.weight_range))
                for _ in range(n_items)
            ]

            items_sorted = sorted(items, key=lambda x: x[0])
            if classification:
                opt_val, mask = set_knapsack_01(
                    items_sorted, capacity, return_items=True
                )
                y = torch.tensor(mask, dtype=torch.float32)
            else:
                opt_val = set_knapsack_01(items_sorted, capacity)
                y = torch.tensor(opt_val, dtype=torch.float32)

            x_features = [[v, w, capacity] for v, w in items_sorted]
            x = torch.tensor(x_features, dtype=torch.float32)

            #feature noise
            if self.noise_prob > 0.0:
                for i in range(x.shape[0]):
                    if random.random() < self.noise_prob:
                        x[i, 0] += random.randint(*self.adversarial_range)
                        x[i, 1] += random.randint(*self.adversarial_range)
                        x[i, 1] = max(x[i, 1], 1.0)

            self.data.append((x, y))

    def __len__(self): return self.n_samples
    def __getitem__(self, idx): return self.data[idx]

#An optimization problem where you aim to maximize the total value of items placed into a knapsack with a limited capacity, 
#but unlike the 0/1 version, you are allowed to take fractions of items. polynomial time using a greedy approach.
# ----- fractional knapsack -----
def set_fractional_knapsack(
    items: List[Tuple[int, int]],   # (value, weight)
    capacity: int,
    return_items: bool = False
):
    """
    Greedy fractional-knapsack solver.

    Parameters
    ----------
    items : list of (value, weight)             # original order!
    capacity : int
    return_items : bool
        * False  -> return optimal total value (float)
        * True   -> return (optimal value, list[float] fractions_selected)

    The fraction vector is aligned to the *input order* of `items`.
    """
    # Keep original indices so we can map fractions back
    indexed_items = list(enumerate(items))
    # Greedy by value/weight ratio
    sorted_by_ratio = sorted(
        indexed_items,
        key=lambda x: x[1][0] / x[1][1],
        reverse=True,
    )

    n = len(items)
    fractions = [0.0] * n
    remaining = capacity
    total_value = 0.0

    for idx, (val, wt) in sorted_by_ratio:
        if remaining <= 0:
            break
        if wt <= remaining:         # take whole item
            fractions[idx] = 1.0
            total_value += val
            remaining -= wt
        else:                       # take fraction
            frac = remaining / wt
            fractions[idx] = frac
            total_value += val * frac
            remaining = 0

    if return_items:
        return total_value, fractions
    return total_value


class FractionalKnapsackDataset(Dataset):
    """
    Synthetic data for the *fractional* knapsack.
      · Input token:  [value_i, weight_i, capacity]
      · Output:
          - regression: optimal total value          (scalar)
          - classification: per-item fraction taken  (len = n_items)
    """
    def __init__(
        self,
        n_samples: int = 1000,
        length_range: Tuple[int, int] = (10, 10),
        value_range:  Tuple[int, int] = (10, 20),
        weight_range: Tuple[int, int] = (1, 5),
        target_range: Tuple[int, int] = (1, 5),
        noise_prob: float = 0.0,
        adversarial_range: Tuple[int, int] = (10, 30),
        classification: bool = True,
        seed: int = 42,
        **kwargs
    ):
        super().__init__()
        random.seed(seed)

        self.n_samples      = n_samples
        self.length_range   = length_range
        self.value_range    = value_range
        self.weight_range   = weight_range
        self.target_range   = target_range
        self.noise_prob     = noise_prob
        self.adversarial_range = adversarial_range
        self.classification = classification

        self.data = []
        for _ in range(n_samples):
            # ----------------- generate instance -----------------
            n_items  = random.randint(*self.length_range)
            capacity = random.randint(*self.target_range)
            items = [
                (random.randint(*self.value_range),
                 random.randint(*self.weight_range))
                for _ in range(n_items)
            ]

            # Stable order so label aligns with feature seq
            items_sorted = sorted(items, key=lambda x: x[0])  # same trick as before

            # ---------- solve knapsack & build label ----------
            if classification:
                opt_val, fractions = set_fractional_knapsack(
                    items_sorted, capacity, return_items=True
                )
                y = torch.tensor(fractions, dtype=torch.float32)
            else:
                opt_val = set_fractional_knapsack(items_sorted, capacity)
                y = torch.log1p(torch.tensor(opt_val, dtype=torch.float32))

            # ------------- assemble feature tensor -------------
            x = torch.tensor(
                [[v, w, capacity] for v, w in items_sorted],
                dtype=torch.float32,
            )

            # Optional noise injection (unchanged)
            if self.noise_prob > 0.0:
                for i in range(x.shape[0]):
                    if random.random() < self.noise_prob:
                        x[i, 0] += random.randint(*self.adversarial_range)
                        x[i, 1] += random.randint(*self.adversarial_range)
                        x[i, 1] = max(x[i, 1], 1.0)

            self.data.append((x, y))

    def __len__(self):  return self.n_samples
    def __getitem__(self, idx):  return self.data[idx]


# 0/1 Minimum Coin Change Problem, Weakly NP-hard.
# ----- min coin -----
def set_min_coin_change(
    coins: List[int],
    target: int,
    return_items: bool = False
):
    """
    0/1 Minimum-Coin Change (each coin may be used at most once).

    Parameters
    ----------
    coins : list[int]           # original order is preserved for mask
    target : int
    return_items : bool
        False -> int   (min #coins, or math.inf if impossible)
        True  -> (min #coins, list[int] chosen_mask)

    chosen_mask[i] = 1 if coin i is used in an optimal solution.
    If no solution exists, mask will be all-zeros and min = math.inf.
    """
    n = len(coins)
    # dp[i][t] = min coins to reach t using first i coins
    dp  = [[math.inf]*(target+1) for _ in range(n+1)]
    keep = [[False]*(target+1)   for _ in range(n+1)]

    dp[0][0] = 0                     # base case

    for i, c in enumerate(coins, start=1):
        for t in range(target+1):
            # option 1 : skip coin i-1
            best = dp[i-1][t]
            use  = False

            # option 2 : take coin i-1  (if it fits and improves)
            if c <= t and dp[i-1][t-c] + 1 < best:
                best = dp[i-1][t-c] + 1
                use  = True

            dp[i][t]   = best
            keep[i][t] = use

    best_num = dp[n][target]

    if not return_items:
        return best_num if best_num != math.inf else 0

    # ------------------- reconstruct mask -------------------
    mask = [0]*n
    t = target
    for i in range(n, 0, -1):
        if keep[i][t]:
            mask[i-1] = 1
            t -= coins[i-1]
    return best_num, mask


class MinCoinChangeDataset(Dataset):
    """
    Synthetic 0/1 Minimum-Coin Change data.

      ▸ Input token:  [coin_value, target]           (input_dim = 2)
      ▸ Output:
            • regression:   minimum #coins  (scalar)
            • classification: 0/1 mask      (len = n_coins)
    """
    def __init__(
        self,
        n_samples: int = 1000,
        length_range: Tuple[int, int] = (10, 10),
        value_range:  Tuple[int, int] = (1, 10),
        target_range: Tuple[int, int] = (1, 20),
        noise_prob: float = 0.0,
        adversarial_range: Tuple[int, int] = (1, 30),
        classification: bool = True,
        seed: int = 42,
        **kwargs
    ):
        super().__init__()
        random.seed(seed)
        np.random.seed(seed)

        self.n_samples = n_samples
        self.length_range = length_range
        self.value_range = value_range
        self.target_range = target_range
        self.noise_prob = noise_prob
        self.adversarial_range = adversarial_range
        self.classification = classification

        self.data = []
        for _ in range(n_samples):
            n_coins = random.randint(*self.length_range)
            coins = [random.randint(*self.value_range) for _ in range(n_coins)]
            coins_sorted = sorted(coins)

            possible_sums = {0}
            for c in coins_sorted:
                possible_sums.update({s + c for s in possible_sums})

            # 2. Decide if we're aiming for a solvable (y=1) or unsolvable (y=0) instance.
            aim_for_solvable = random.random() < 0.5
            T = -1
            if aim_for_solvable:
                valid_solvable_targets = [s for s in possible_sums if self.target_range[0] <= s <= self.target_range[1]]
                if valid_solvable_targets:
                    T = random.choice(valid_solvable_targets)
                else:
                    aim_for_solvable = False # Give up and try to find an unsolvable one.

            if not aim_for_solvable:
                all_possible_targets_in_range = set(range(self.target_range[0], self.target_range[1] + 1))
                valid_unsolvable_targets = list(all_possible_targets_in_range - possible_sums)
                if valid_unsolvable_targets:
                    T = random.choice(valid_unsolvable_targets)
                else:
                    # Give up: if no unsolvable targets exist, we must pick a solvable one.
                    # This can happen if the range is small and fully covered by possible_sums.
                    valid_solvable_targets = [s for s in possible_sums if self.target_range[0] <= s <= self.target_range[1]]
                    if valid_solvable_targets:
                        T = random.choice(valid_solvable_targets)
                    else: # Fallback if range is impossible to hit
                        T = self.target_range[0]

            if self.classification:
                _, mask = set_min_coin_change(coins_sorted, T, return_items=True)
                y = torch.tensor(mask, dtype=torch.float32)
            else:
                min_cnt = set_min_coin_change(coins_sorted, T)
                y_val = 0 if min_cnt == math.inf else min_cnt
                y = torch.tensor(y_val, dtype=torch.float32)

            x = torch.tensor(
                [[c, T] for c in coins_sorted],
                dtype=torch.float32
            )

            if self.noise_prob > 0.0:
                for i in range(x.shape[0]):
                    if random.random() < self.noise_prob:
                        x[i, 0] += random.randint(*self.adversarial_range)
                        x[i, 0] = max(x[i, 0], 1.0)

            self.data.append((x, y))

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.data[idx]




class QuickselectDataset(Dataset):
    def __init__(
        self,
        n_samples: int = 1000,
        length_range: tuple[int,int] = (10,10),     # n = number of elements in xs
        value_range: tuple[int,int]  = (1,10),      # Range for values in xs
        max_k: int = 8,
        normalize: bool = True,
        noise_prob: float = 0.0,                    # Probability of adding noise
        adversarial_range = (1,5),  # Range of noise values
        classification: bool = True,                # True for classification, False for regression
        seed: int = 42,
        **kwargs  # Catches other arguments like k if passed from a config
    ):
        super().__init__()
        random.seed(seed)

        self.n_samples         = n_samples
        self.length_range      = length_range
        self.value_range       = value_range
        self.noise_prob        = noise_prob
        self.normalize = normalize
        self.max_k = max_k
        self.adversarial_range = adversarial_range
        self.classification    = classification

        self.data = []
        for _ in range(n_samples):
            n = random.randint(*self.length_range)
            if n < 2: 
                continue
            xs_original_unsorted = [random.randint(*self.value_range) for _ in range(n)]

            # 3) Define k.
            upper_bound_for_k = min(self.max_k, n)
            k = random.randint(2, upper_bound_for_k)
            y_val = nth_smallest(xs_original_unsorted.copy(), k-1)


            #Nomalization (like CLRS30)
            x_features = []
            min_val, max_val = min(xs_original_unsorted), max(xs_original_unsorted)
            val_range = max_val - min_val if max_val > min_val else 1.0

            for val in xs_original_unsorted:
                if self.normalize:
                    norm_val = (val - min_val) / val_range
                    #Normalize k to be a relative rank between 0 and 1 (CLRS30)
                    norm_k = (k - 1) / (n - 1) if n > 1 else 0.0
                    x_features.append([norm_val, norm_k])
                else:
                    # Original, non-normalized features
                    x_features.append([float(val), float(k)])
            
            x_t = torch.tensor(x_features, dtype=torch.float)
            
            if self.noise_prob > 0:
                processed_rows = [
                    (row_tensor + random.random()) 
                    if random.random() < self.noise_prob 
                    else row_tensor 
                    for row_tensor in x_t
                ]
                if processed_rows:
                    x_t = torch.stack(processed_rows)
                
            if not self.classification:
                # Regression target: the actual k-th smallest value.
                y_t = torch.tensor(y_val, dtype=torch.float)
            else:
                #Classification mask: identify ALL positions of the k-th smallest value
                mask = [0.0] * n # Use float for consistency
                for i in range(n):
                    if xs_original_unsorted[i] == y_val:
                        mask[i] = 1.0 # Mark ALL occurrences
                y_t = torch.tensor(mask, dtype=torch.float)

            self.data.append((x_t, y_t))

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.data[idx]


def balanced_partition_subset(x_sorted):
    """
    Return a bit-vector (list[int]) marking one subset that attains the
    minimum |sum(S) – sum(Ŝ)|.  We pick the lexicographically-smallest
    solution to make the labels unique and reproducible.
    """
    n, total = len(x_sorted), sum(x_sorted)
    # DP over sums up to total//2.  parent[s] = (prev_s, idx) records the path.
    reach = [False]*(total//2 + 1)
    parent = [None]*(total//2 + 1)
    reach[0] = True
    for idx, val in enumerate(x_sorted):
        # iterate backwards so each item is used at most once
        for s in range(total//2, val-1, -1):
            if reach[s-val] and not reach[s]:
                reach[s] = True
                parent[s] = (s-val, idx)
    # choose the achievable sum with smallest diff = |total-2*s|
    best_s = min((s for s in range(total//2+1) if reach[s]),
                 key=lambda s: abs(total - 2*s))
    # reconstruct subset indices
    chosen = [0]*n
    s = best_s
    while s != 0:
        s_prev, idx = parent[s]
        chosen[idx] = 1          # mark element idx as in subset S
        s = s_prev
    return chosen, abs(total - 2*best_s)


class BalancedPartitionDataset(Dataset):
    """
    Balanced-Partition data for either regression (diff) or
    classification (0/1 membership per element).
    """
    def __init__(
        self,
        n_samples: int = 1000,
        length_range: tuple[int,int] = (10,10),
        value_range:  tuple[int,int] = (1,10),
        noise_prob:   float = 0.0,
        adversarial_range: tuple[int,int] = (10,30),
        classification: bool = True,
        seed: int = 42,
        **kwargs
    ):
        super().__init__()
        random.seed(seed)

        self.n_samples = n_samples
        self.cls = classification
        self.data = []

        for _ in range(n_samples):
            n = random.randint(*length_range)
            x = [random.randint(*value_range) for _ in range(n)]
            x_sorted = sorted(x)

            labels, diff = balanced_partition_subset(x_sorted)

            x_t = torch.tensor(x_sorted, dtype=torch.float32).unsqueeze(-1)
            if classification:
                y_t = torch.tensor(labels, dtype=torch.float32)   # shape (n,)
            else:
                y_t = torch.tensor(diff, dtype=torch.float32)     # scalar

            # input noise
            if noise_prob > 0:
                for i in range(x_t.shape[0]):
                    if random.random() < noise_prob:
                        x_t[i, 0] += random.randint(*adversarial_range)

            self.data.append((x_t, y_t))

    def __len__(self): return self.n_samples
    def __getitem__(self, idx): return self.data[idx]



def first_fit_decreasing_assign(sizes_sorted, capacity):
    """
    Greedy First-Fit Decreasing that also returns the bin assignment
    for every item in *the given order*.

    Parameters
    ----------
    sizes_sorted : list[int]
        Item sizes **already sorted in the order you will feed to the model**.
        (Here we rely on ascending order for consistency with the original code.)
    capacity : int
        Capacity of each bin.

    Returns
    -------
    assignments : list[int]        # same length as sizes_sorted
        The bin index (0-based) each item ends up in.
    num_bins : int
        Total bins used.
    """
    loads      = []          # current used capacity of each bin
    assignment = []

    for sz in sizes_sorted:
        for bidx, load in enumerate(loads):
            if load + sz <= capacity:
                loads[bidx] += sz
                assignment.append(bidx)
                break
        else:                # no existing bin fits → open new one
            loads.append(sz)
            assignment.append(len(loads) - 1)

    return assignment, len(loads)


#class BinPackingDataset(Dataset):
#    """
#    Bin-Packing data for either:
#      • classification  → per-item bin index (multi-class)
#      • regression      → scalar #bins (original behaviour)
#
#    Each token = [size_i, capacity].  Capacity is repeated so every item
#    sees the global constraint.
#    """
#    def __init__(
#        self,
#        n_samples: int                 = 1000,
#        length_range: tuple[int,int]   = (10, 10),
#        value_range:  tuple[int,int]   = (1, 10),
#        target_range: tuple[int,int]   = (10, 30),
#        noise_prob:   float            = 0.0,
#        adversarial_range: tuple[int,int] = (10, 30),
#        classification: bool           = True,
#        seed: int                      = 42,
#        **kwargs
#    ):
#        super().__init__()
#        random.seed(seed)
#
#        self.n_samples   = n_samples
#        self.cls         = classification
#        self.data        = []
#
#        for _ in range(n_samples):
#            # 1) sample sequence and capacity
#            n        = random.randint(*length_range)
#            capacity = random.randint(*target_range)
#            sizes    = [random.randint(*value_range) for _ in range(n)]
#            
#            # 3) clamp sizes (item must fit!) and sort for determinism
#            sizes = [min(max(1, sz), capacity) for sz in sizes]
#            sizes_sorted = sorted(sizes)               # ascending → deterministic
#
#            # 4) run FFD + capture assignments
#            labels, bin_cnt = first_fit_decreasing_assign(sizes_sorted, capacity)
#
#            # 5) build tensors
#            x_t = torch.tensor([[sz, capacity] for sz in sizes_sorted],
#                               dtype=torch.float32)
#
#            if self.cls:
#                # token-wise integer class labels
#                y_t = torch.tensor(labels, dtype=torch.long)      # shape (n,)
#            else:
#                y_t = torch.tensor(bin_cnt, dtype=torch.float32)  # scalar
#
#            if noise_prob > 0:
#                for i in range(x_t.shape[0]):
#                    if random.random() < noise_prob:
#                        # Add noise only to the item size (the first element of the feature vector)
#                        noise_val = random.randint(*adversarial_range)
#                        x_t[i, 0] += noise_val
#                        # After adding noise, we should still clamp it to be valid
#                        x_t[i, 0] = torch.clamp(x_t[i, 0], min=1, max=capacity)
#
#            self.data.append((x_t, y_t))
#
#    def __len__(self):
#        return self.n_samples
#
#    def __getitem__(self, idx):
#        return self.data[idx]

class BinPackingDataset(Dataset):
    """
    Bin-Packing data for either:
      • classification  → per-item bin 0/1 classification
      • regression      → scalar #bins (original behaviour)

    Each token = [size_i, capacity, position_in_sequence].  Capacity is repeated so every item
    sees the global constraint.
    """
    def __init__(
        self,
        n_samples: int               = 1000,
        length_range: tuple[int,int]   = (8, 8),
        value_range:  tuple[int,int]   = (1, 10),
        target_range: tuple[int,int]   = (10, 30),
        noise_prob:   float             = 0.0,
        adversarial_range: tuple[int,int] = (10, 30),
        classification: bool             = True,
        seed: int                        = 42,
        **kwargs
    ):
        super().__init__()
        random.seed(seed)

        self.n_samples    = n_samples
        self.cls          = classification
        self.data         = []

        for _ in range(n_samples):
            n        = random.randint(*length_range)
            capacity = random.randint(*target_range)
            sizes    = [random.randint(*value_range) for _ in range(n)]

            sizes_clean = [min(max(1, sz), capacity) for sz in sizes]
            # Using descending sort for the more challenging FFD heuristic
            sizes_sorted = sorted(sizes_clean, reverse=True)

            # Get the multi-class bin assignments from the solver
            multi_class_labels, bin_cnt = first_fit_decreasing_assign(sizes_sorted, capacity)

            # Convert multi-class labels to binary
            if self.cls:
                # The task is to predict if an item starts a new bin.
                binary_labels = []
                max_bin_index_seen = -1
                for bin_index in multi_class_labels:
                    if bin_index > max_bin_index_seen:
                        # This item started a new bin.
                        binary_labels.append(1.0)
                        max_bin_index_seen = bin_index
                    else:
                        # This item fit into an existing bin.
                        binary_labels.append(0.0)
                y_t = torch.tensor(binary_labels, dtype=torch.float32)
            else:
                # Regression task remains the same: predict total bins used.
                y_t = torch.tensor(bin_cnt, dtype=torch.float32)

            x_features = []
            n = len(sizes_sorted)
            for i, sz in enumerate(sizes_sorted):
                # Normalize the index to be a float between 0.0 and 1.0
                normalized_position = i / (n - 1) if n > 1 else 0.0
                x_features.append([sz, capacity, normalized_position])
            x_t = torch.tensor(x_features, dtype=torch.float32)

            # Noise logic remains the same
            if noise_prob > 0:
                for i in range(x_t.shape[0]):
                    if random.random() < noise_prob:
                        noise_val = random.randint(*adversarial_range)
                        x_t[i, 0] += noise_val
                        x_t[i, 0] = torch.clamp(x_t[i, 0], min=1, max=capacity)

            self.data.append((x_t, y_t))

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.data[idx]


# ----- Longest Increasing Subsequence (LIS) Length ----- (regression)
def calculate_lis_length(nums):
    if not nums:
        return 0
    tails = []
    for num in nums:
        if not tails or num > tails[-1]:
            tails.append(num)
        else:
            tails[bisect.bisect_left(tails, num)] = num
    return len(tails)

class LISDataset(Dataset):
    """
    Dataset for the Longest Increasing Subsequence (LIS) Length problem.
      - Input: sequence of numbers x_i
      - Label: the length of the LIS of x

    Each token in x_t is a 2d-vector [value_i, positional index] by default.

    Outputs:
      x_t: Tensor of shape (n, input_dim) - input sequence
      y_t: Tensor of shape ()            - scalar LIS length (regression target)
    """
    def __init__(
        self,
        n_samples: int = 1000,
        length_range: tuple[int,int] = (8,8),
        value_range: tuple[int,int] = (1,10),
        noise_prob: float = 0.0,
        adversarial_range: tuple[int,int] = (10,30),
        seed: int = 42,
        **kwargs
    ):
        super().__init__()
        random.seed(seed)

        self.n_samples         = n_samples
        self.length_range      = length_range
        self.value_range       = value_range
        self.noise_prob        = noise_prob
        self.adversarial_range = adversarial_range

        self.data = []
        for _ in range(n_samples):
            n = random.randint(*self.length_range)
            if n == 0:
                continue
            sequence = [random.randint(*self.value_range) for _ in range(n)]
            if self.noise_prob > 0:
                noisy_sequence_temp = []
                for val in sequence:
                    if random.random() < self.noise_prob:
                        noisy_sequence_temp.append(val + random.randint(*self.adversarial_range))
                    else:
                        noisy_sequence_temp.append(val)
                sequence = noisy_sequence_temp
            y_val = calculate_lis_length(sequence)
            # Shape: (n, 2)
            x_t_list = [[float(val)] for val in sequence]
            temp_xt_list = []
            for i, val in enumerate(sequence):
                normalized_idx = np.exp(i / (n - 1)) if n > 1 else 0.0
                temp_xt_list.append([float(val), normalized_idx])
            x_t_list = temp_xt_list

            x_t = torch.tensor(x_t_list, dtype=torch.float)
            y_t = torch.tensor(float(y_val), dtype=torch.float)

            self.data.append((x_t, y_t))

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.data[idx]
    
# ----- convex hull -----
def compute_convex_hull(points):
    pts = sorted(points)
    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and ((lower[-1][0] - lower[-2][0]) * (p[1] - lower[-2][1]) -
                                     (lower[-1][1] - lower[-2][1]) * (p[0] - lower[-2][0])) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and ((upper[-1][0] - upper[-2][0]) * (p[1] - upper[-2][1]) -
                                     (upper[-1][1] - upper[-2][1]) * (p[0] - upper[-2][0])) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]

class ConvexHullDataset(Dataset):
    """
    Dataset for the Convex Hull classification problem.

    Each example is a random set of 2D points; the label is 1 if the point
    lies on the set's convex hull, else 0.

    Args:
      n_samples        (int): number of examples
      length_range  ((int,int)): min/max number of points
      value_range((int,int)): min/max for x and y coordinates
      noise_prob      (float): chance to jitter each coordinate
      adversarial_range((int,int)): jitter range if noise applies
      seed             (int): RNG seed
      **kwargs
    """
    def __init__(
        self,
        n_samples: int = 1000,
        length_range: tuple[int,int] = (10, 10),
        value_range: tuple[int,int] = (0, 10),
        noise_prob: float = 0.0,
        adversarial_range: tuple[int,int] = (1, 3),
        seed: int = 42,
        **kwargs
    ):
        super().__init__()
        random.seed(seed)

        self.n_samples        = n_samples
        self.length_range     = length_range
        self.value_range      = value_range
        self.noise_prob       = noise_prob
        self.adversarial_range= adversarial_range

        self.data = []
        for _ in range(n_samples):
            n = random.randint(*self.length_range)
            pts = []
            for _ in range(n):
                x = random.randint(*self.value_range)
                y = random.randint(*self.value_range)
                pts.append((x, y))
            
            hull_pts = set(compute_convex_hull(pts))
            mask = [1 if p in hull_pts else 0 for p in pts] #label each point: 1 if on hull, else 0
            
            x_t = torch.tensor(pts, dtype=torch.float)
            y_t = torch.tensor(mask, dtype=torch.float)

            if self.noise_prob > 0:
                for i in range(x_t.shape[0]):
                    if random.random() < self.noise_prob:
                        x_t[i, 0] += random.randint(*self.adversarial_range)
                        x_t[i, 1] += random.randint(*self.adversarial_range)
                            
            self.data.append((x_t, y_t))
    def __len__(self): return self.n_samples
    def __getitem__(self, idx): return self.data[idx]
    
# ----- three sum decision -----
def three_sum_decision(xs, target=0):
    """
    Checks if any three *distinct* elements (by index) in list 'xs' sum to 'target'.
    Args:
        xs (list): A list of numbers.
        target (int/float): The target sum.
    Returns:
        int: 1 if three distinct elements sum to target, 0 otherwise.
    """
    n = len(xs)
    if n < 3:
        return 0 # Cannot choose 3 distinct elements
    xs_sorted = sorted(xs) # O(n log n) - necessary for two-pointer approach
    # O(n^2) part
    for i in range(n - 2):
        if i > 0 and xs_sorted[i] == xs_sorted[i-1]:
             continue
        left, right = i + 1, n - 1 # Pointers for the remaining part
        needed_sum = target - xs_sorted[i]
        while left < right:
            current_two_sum = xs_sorted[left] + xs_sorted[right]
            if current_two_sum == needed_sum:
                # Found a triplet (xs_sorted[i], xs_sorted[left], xs_sorted[right])
                # Indices i, left, right are guaranteed distinct by construction
                return 1
            elif current_two_sum < needed_sum:
                left += 1 # Need a larger sum, move left pointer right
            else: # current_two_sum > needed_sum
                right -= 1 # Need a smaller sum, move right pointer left
    # If loops complete without finding a triplet
    return 0

class ThreeSumDecisionDataset(Dataset):
    """
    Dataset for the 3-SUM decision problem: given a list x and a target T,
    label = 1 if any three distinct elements of x sum to T, else 0.
    Uses the canonical T=0 by default.

    Each token is a 2-vector [value_i, T], so your model should use input_dim=2.

    Args:
      n_samples         (int): number of examples
      length_range ((int,int)): min/max list length n
      value_range  ((int,int)): sampling range for each x_i (can be negative)
      target_range ((int,int)): sampling range for T (default is fixed T=0)
      noise_prob        (float): probability to perturb each x_i
      adversarial_range ((int,int)): magnitude of noise if applied
      seed               (int): RNG seed
      **kwargs: catch-all for unused params
    """
    def __init__(
        self,
        n_samples: int = 1000,
        length_range: tuple[int,int] = (8, 8), # List length n
        value_range: tuple[int,int]  = (-10, 10), # Value range for elements
        target_range: tuple[int,int] = (0, 0),   # Target T range (fixed T=0 default)
        noise_prob: float = 0.0,
        adversarial_range: tuple[int,int] = (100, 200), # Noise magnitude
        seed: int = 42,
        **kwargs
    ):
        super().__init__()
        random.seed(seed)

        self.n_samples        = n_samples
        self.length_range     = length_range
        self.value_range      = value_range
        self.target_range     = target_range
        self.noise_prob       = noise_prob
        self.adversarial_range= adversarial_range

        self.data = []
        for _ in range(n_samples):
            n = random.randint(*self.length_range)
            xs = [random.randint(*self.value_range) for _ in range(n)]
            
            T = random.randint(*self.target_range)
            y_val = three_sum_decision(xs, T)
            
            xs_sorted = sorted(xs)
            
            x_t_list_of_pairs = [[val, T] for val in xs_sorted]
            x_t = torch.tensor(x_t_list_of_pairs, dtype=torch.float)
            
            y_t = torch.tensor(y_val, dtype=torch.long)
            
            if self.noise_prob > 0:
                for i in range(x_t.shape[0]):
                    if random.random() < self.noise_prob:
                        x_t[i, 0] += random.randint(*self.adversarial_range)
            
            self.data.append((x_t, y_t))

    def __len__(self):
        return self.n_samples
    def __getitem__(self, idx):
        return self.data[idx]



class FloydWarshallDataset(Dataset):
    """
    Floyd-Warshall dataset that generates sparse, weighted graphs and uses the
    predecessor matrix for classification.
    """
    def __init__(
        self,
        n_samples: int = 1000,
        length_range: tuple[int, int] = (4, 4),
        p_range: tuple[float, float] = (0.5, 0.9),
        value_range: tuple[float, float] = (0.0, 0.2),
        noise_prob: float = 0.0,
        adversarial_range: tuple[float, float] = (0.1, 0.5),
        classification: bool = True,
        seed: int = 42,
        **kwargs
    ):
        """
        Args:
            classification (bool):
                - True: Predict the predecessor matrix (Pi). Multi-class classification.
                - False: Predict the shortest path distances (D). Regression.
        """
        super().__init__()
        random.seed(seed)
        np.random.seed(seed)

        self.n_samples = n_samples
        self.length_range = length_range
        self.p_range = p_range
        self.weight_range = value_range
        self.noise_prob = noise_prob
        self.adversarial_range = adversarial_range
        self.classification = classification

        self.data = []
        for _ in range(n_samples):
            n = random.randint(*self.length_range)
            if n <= 0: continue
            
            p_sample = random.uniform(*self.p_range)
            W = self._generate_er_graph(n, p_sample, self.weight_range)
            
            W_features = np.copy(W)
            W_features[np.isinf(W_features)] = 0.0

            graph = csr_matrix(W)
            # --- MINIMAL CHANGE: Get predecessors from floyd_warshall ---
            D, Pi = floyd_warshall(csgraph=graph, directed=False, return_predecessors=True)
            # ---

            # --- MINIMAL CHANGE: Use Pi for classification target ---
            if self.classification:
                # The predecessor for unreachable nodes is -9999.
                # We map this to a valid class index, e.g., n.
                # The classes will be {0, 1, ..., n-1} for predecessors,
                # and class {n} for "no predecessor".
                Pi[Pi == -9999] = n 
                y_t = torch.tensor(Pi.flatten(), dtype=torch.long)
            else:
                # Regression task remains the same: predict distances
                large_val_for_inf = n * self.weight_range[1] + 1
                D[np.isinf(D)] = large_val_for_inf
                y_t = torch.tensor(D.flatten(), dtype=torch.float)
            # --- END OF CHANGES ---

            # Prepare input tensor x_t (this part is unchanged)
            flat_W = W_features.flatten()
            features = [torch.tensor(flat_W, dtype=torch.float).unsqueeze(-1)]
            indices = np.arange(n * n)
            norm_i = (indices // n) / (n - 1) if n > 1 else np.zeros(n * n)
            norm_j = (indices % n) / (n - 1) if n > 1 else np.zeros(n * n)
            features.append(torch.tensor(norm_i, dtype=torch.float).unsqueeze(-1))
            features.append(torch.tensor(norm_j, dtype=torch.float).unsqueeze(-1))
            x_t = torch.cat(features, dim=-1)

            # Noise addition (unchanged)
            if self.noise_prob > 0:
                for k_idx in range(n * n):
                    if (k_idx // n) != (k_idx % n) and random.random() < self.noise_prob:
                        jitter_val = random.uniform(*self.adversarial_range)
                        x_t[k_idx, 0] = torch.clamp(x_t[k_idx, 0] + jitter_val, min=0.0)
            
            self.data.append((x_t, y_t))

    def _generate_er_graph(self, n: int, p: float, weight_range: tuple[float, float]) -> np.ndarray:
        """
        Generates a weighted, undirected Erdos-Renyi random graph with float weights.
        The logic to choose between int/float is removed as it's no longer needed.
        """
        low, high = weight_range
        adj = np.random.binomial(1, p, size=(n, n))
        adj = adj * adj.T
        weights = np.random.uniform(low=low, high=high, size=(n, n))
        symmetric_weights = np.sqrt((weights * weights.T) + 1e-6)
        
        W = np.full((n, n), np.inf, dtype=float)
        W[adj == 1] = symmetric_weights[adj == 1]
        np.fill_diagonal(W, 0.0)
        return W

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.data[idx]







# ----- strongly connected components classification -----
class SCCDataset(Dataset):
    """
    Each example is a directed graph generated by a random community model.
    The task is to predict, for every pair of nodes (i, j), if they belong
    to the same strongly connected component.
    """
    def __init__(
        self,
        n_samples: int = 1000,
        length_range: tuple[int, int] = (4, 4), # Node count range
        k_range: tuple[int, int] = (1, 3),        # Community count range
        value_range: float = 0.6,                           # Intra-community edge probability
        eps: float = 0.05,                        # Inter-community edge probability
        noise_prob: float = 0.0,
        seed: int = 42,
        **kwargs
    ):
        """
        Args:
            n_samples (int): Number of graphs to generate.
            length_range (tuple): Min/max number of nodes n.
            k_range (tuple): Min/max number of communities to partition nodes into.
            p (float): Probability of edges WITHIN a community.
            eps (float): Probability of edges BETWEEN communities.
            noise_prob (float): Probability to flip any edge in the input features.
            seed (int): RNG seed for reproducibility.
        """
        super().__init__()
        self.n_samples = n_samples
        self.length_range = length_range
        self.k_range = k_range
        self.p = value_range
        self.eps = eps
        self.noise_prob = noise_prob
        
        # Set seeds for reproducibility
        random.seed(seed)
        np.random.seed(seed)

        self.data = []
        for _ in range(n_samples):
            # 1. Determine graph properties for this sample
            n = random.randint(*self.length_range)
            # Ensure k is not greater than n
            k = random.randint(self.k_range[0], max(self.k_range[0], n // 2))

            # 2. Generate the graph using the community model
            A = self._generate_community_graph(n, k, self.p, self.eps)

            # 3. Calculate the ground-truth SCCs (the labels)
            graph = csr_matrix(A)
            # The connected_components function with these parameters finds SCCs
            _, labels = connected_components(
                csgraph=graph, directed=True, return_labels=True, connection='strong'
            )
            # Create an (n, n) matrix where M[i, j] is 1 if i and j are in the same component
            same_cc = (labels[:, None] == labels[None, :]).astype(float)
            y_t = torch.tensor(same_cc.flatten(), dtype=torch.float)

            # 4. Create the input features for the model
            flat_A = A.flatten()
            features = [torch.tensor(flat_A, dtype=torch.float).unsqueeze(-1)]
            
            # Add normalized position indices as features
            indices = np.arange(n * n)
            norm_i = (indices // n) / (n - 1) if n > 1 else np.zeros(n * n)
            norm_j = (indices % n) / (n - 1) if n > 1 else np.zeros(n * n)
            features.append(torch.tensor(norm_i, dtype=torch.float).unsqueeze(-1))
            features.append(torch.tensor(norm_j, dtype=torch.float).unsqueeze(-1))
            x_t = torch.cat(features, dim=-1)

            # 5. Apply noise to the input features
            if self.noise_prob > 0:
                # Create a mask for edges to flip (excluding the diagonal)
                noise_mask = (np.random.rand(n, n) < self.noise_prob).flatten()
                diag_mask = (np.arange(n*n) // n == np.arange(n*n) % n)
                noise_mask[diag_mask] = False # Don't flip self-loops
                
                # Flip the edge existence feature where the mask is true
                x_t[noise_mask, 0] = 1.0 - x_t[noise_mask, 0]

            self.data.append((x_t, y_t))

    def _generate_community_graph(self, n: int, k: int, p: float, eps: float) -> np.ndarray:
        """
        Generates a random directed graph with community structure.

        Args:
            n (int): Total number of nodes.
            k (int): The number of communities to partition the nodes into.
            p (float): The probability of an edge between two nodes in the SAME community.
            eps (float): The probability of an edge between two nodes in DIFFERENT communities.

        Returns:
            np.ndarray: An (n, n) adjacency matrix.
        """
        A = np.zeros((n, n), dtype=float)
        nodes = np.arange(n)
        np.random.shuffle(nodes)
        
        # Partition nodes into k communities
        communities = np.array_split(nodes, k)
        node_to_community_id = np.zeros(n, dtype=int)
        for i, community in enumerate(communities):
            for node in community:
                node_to_community_id[node] = i

        # Iterate over every possible directed edge (u, v)
        for u in range(n):
            for v in range(n):
                if u == v:
                    continue
                
                # Check if u and v are in the same community
                if node_to_community_id[u] == node_to_community_id[v]:
                    # Intra-community edge probability
                    if random.random() < p:
                        A[u, v] = 1.0
                else:
                    # Inter-community edge probability
                    if random.random() < eps:
                        A[u, v] = 1.0
        return A

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.data[idx]
