from collections import defaultdict


class InferenceCache:
    def __init__(self) -> None:
        self._cache = defaultdict(dict)

    def _key(self, ts_id, step):
        return f"{ts_id}_step_{step}"

    def clear(self):
        self._cache = defaultdict(dict)

    def is_cached(self, ts_id, overall_step):
        return self._key(ts_id, overall_step) in self._cache

    def get_cached(self, ts_id, overall_step, alpha):
        return self._cache[self._key(ts_id, overall_step)][alpha]

    def cache_result(self, ts_id, overall_step, alpha, pred_result):
        self._cache[self._key(ts_id, overall_step)][alpha] = pred_result
