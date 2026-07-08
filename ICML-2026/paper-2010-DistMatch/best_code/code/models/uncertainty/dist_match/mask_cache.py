import os
import math
from typing import Optional

from tqdm import tqdm

import numpy as np

from torch.multiprocessing import Pool, set_start_method

try:
    set_start_method("spawn")
except RuntimeError:
    pass


class MaskCacheManager:
    def __init__(
        self,
        matcher: callable,
        path: Optional[str] = None,
        batch_size: int | None = None,
        split_feature: bool = True,
    ):
        self.path = path
        self.matcher = matcher
        self.batch_size = batch_size
        self.split_feature = split_feature

    def _compute_batched(
        self,
        x: np.ndarray,
        data: np.ndarray,
    ):
        n_samples = len(data)
        n_batches = math.ceil(n_samples / self.batch_size)
        result = []
        
        for i in range(n_batches):
            start = i * self.batch_size
            end = min((i + 1) * self.batch_size, n_samples)
            sample1 = np.tile(x, (end - start, *([1] * x.ndim)))
            sample2 = data[start:end]
            res = self.matcher(sample1, sample2)
            result.append(res)

        return np.concatenate(result)

    def _compute_per_feature(
        self, data: tuple[np.ndarray, np.ndarray]
    ) -> np.ndarray:
        src_data, tgt_data = data
        n_src_samples = src_data.shape[0]
        n_tgt_samples = tgt_data.shape[0]

        are_equal = np.array_equal(src_data, tgt_data)

        match_mask = np.empty((n_src_samples, n_tgt_samples))
        match_mask[...] = np.nan

        for i in tqdm(range(n_src_samples)):
            if self.batch_size is not None:
                computed_mask = self._compute_batched(src_data[i, :], tgt_data)
                match_mask[i, :] = computed_mask
            else:
                tgt_len = min(i + 1, n_tgt_samples) if are_equal else n_tgt_samples
                for j in range(tgt_len):
                    if not np.isnan(match_mask[i, j]):
                        continue
                    i_term = src_data[i, :]
                    j_term = tgt_data[j, :]

                    match = self.matcher(i_term, j_term)
                    match_mask[i, j] = match
                    if are_equal:
                        match_mask[j, i] = match

        return match_mask

    def compute(self, src_data: np.ndarray, tgt_data: np.ndarray):
        assert src_data.shape[-1] == tgt_data.shape[-1]

        *_, n_feats = src_data.shape
        
        if self.split_feature:
            feat_data = [(src_data[..., i], tgt_data[..., i]) for i in range(n_feats)]

            pool = Pool(n_feats)
            feat_masks = pool.map(self._compute_per_feature, feat_data)

            match_mask = np.stack(feat_masks)
        else:
            match_mask = self._compute_per_feature((src_data, tgt_data))

        match_mask = np.ascontiguousarray(match_mask)

        print("Resulting match_mask shape:", match_mask.shape)

        return match_mask

    def put(self, src_data: np.ndarray, tgt_data: np.ndarray) -> np.ndarray:
        data = self.compute(src_data, tgt_data)
        np.save(self.path, data)

        return data

    def get(self, src_data: np.ndarray, tgt_data: np.ndarray) -> np.ndarray | None:
        if os.path.exists(self.path):
            return np.load(self.path)

        mask = self.put(src_data, tgt_data)
        return mask
