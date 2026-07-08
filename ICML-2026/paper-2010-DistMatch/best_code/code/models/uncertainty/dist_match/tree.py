import math
import pickle
from functools import cached_property, partial
from typing import Callable, Any, Optional, List, Sequence, Tuple

from tqdm import tqdm

import numpy as np

from sklearn.linear_model import QuantileRegressor
from sklearn_quantile import RandomForestQuantileRegressor
import xgboost as xgb

# from doubt import QuantileRegressionForest as QRF


MatcherFnType = Callable[[np.ndarray, np.ndarray], np.ndarray | bool | float]


class Node:
    def __init__(
        self,
        split_value: Tuple[np.ndarray, int, int] | None = None,
        values: Tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
        left: Optional["Node"] = None,
        right: Optional["Node"] = None,
        parent: Optional["Node"] = None,
    ):
        self._split_value = split_value
        self._values = values

        self.left = left
        self.right = right
        self.parent = parent

    def get_all_parents(self) -> List["Node"]:
        node = self
        parents = []
        while node is not None and node.parent is not None:
            parent = node.parent
            parents.append(parent)
            node = parent
        return parents

    def get_split_value(self) -> Tuple[np.ndarray, int, int] | None:
        return self._split_value

    def get_values(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        return self._values

    def set_split_value(self, data: np.ndarray, dim: int, idx: int):
        self._split_value = (data, dim, idx)

    def set_values(self, xs: np.ndarray, ys: np.ndarray, ids: np.ndarray):
        self._values = (xs, ys, ids)

    def reset_split_value(self):
        self._split_value = None

    def reset_values(self):
        self._values = None


class DistMatchTree:
    def __init__(
        self,
        matcher: MatcherFnType,
        feature_dim: int,
        quantiles: np.ndarray,
        match_mask: np.ndarray | None = None,
        verbose: bool = False,
        min_samples_per_node: int = 0,
        n_recent_kept: int | None = None,
        batch_size: int | None = None,
        split_on_predict: bool = False,
        keep_relevant: bool = False,
        relevance_matcher: Optional[MatcherFnType] = None,
        quantile_regressor: str = 'qrf'
    ):
        self.matcher = matcher
        self.feature_dim = feature_dim
        self.quantiles = quantiles
        self.match_mask = match_mask
        self.verbose = verbose
        self.min_samples_per_node = min_samples_per_node
        self.batch_size = batch_size
        self.n_recent_kept = n_recent_kept
        self.split_on_predict = split_on_predict
        self.keep_relevant = keep_relevant
        self.relevance_matcher = (
            relevance_matcher if relevance_matcher is not None else matcher
        )
        self.quantile_regressor = quantile_regressor

        self.root = None

    def compare(
        self,
        sample1: np.ndarray,
        sample2: np.ndarray,
        matcher: MatcherFnType | None = None,
    ) -> np.ndarray:
        matcher = matcher if matcher is not None else self.matcher
        return matcher(sample1, sample2)

    def compare_batched(
        self,
        x: np.ndarray,
        data: np.ndarray,
        matcher: MatcherFnType | None = None,
    ):
        n_samples = len(data)
        n_batches = math.ceil(n_samples / self.batch_size)
        result = []

        for i in range(n_batches):
            start = i * self.batch_size
            end = min((i + 1) * self.batch_size, n_samples)
            sample1 = np.tile(x, (end - start, *([1] * x.ndim)))
            sample2 = data[start:end]
            result.append(self.compare(sample1, sample2, matcher))

        return np.concatenate(result)

    def compare_to_many(
        self,
        x: np.ndarray,
        data: np.ndarray,
        matcher: MatcherFnType | None = None,
        match_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        if match_mask is None:
            match_mask = np.empty(len(data))
            match_mask[...] = np.nan

        nan_ids = np.argwhere(np.isnan(match_mask)).ravel()

        if self.batch_size is None:
            for idx in nan_ids:
                match_mask[idx] = self.compare(x, data[idx], matcher)
        elif len(nan_ids) > 0:
            match_mask[nan_ids] = self.compare_batched(x, data[nan_ids], matcher)

        return match_mask

    def _greedy_search(
        self, x: np.ndarray, subset_ids: np.ndarray, match_mask: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        subset = x[subset_ids]
        n_items = len(subset)

        max_split = 0
        node_id, node_r_ids, node_l_ids = None, None, None
        p_bar = enumerate(subset)
        if self.verbose:
            p_bar = tqdm(p_bar, total=n_items)

        for idx, datum in p_bar:
            datum_mask = match_mask[idx, :]
            datum_mask = np.where(~np.isnan(datum_mask), match_mask[:, idx], datum_mask)

            r_mask = self.compare_to_many(
                datum, subset, self.matcher, datum_mask
            ).astype(bool)

            r_ids = subset_ids[r_mask]
            l_ids = subset_ids[~r_mask]
            # new_split = max(len(r_ids), len(l_ids))
            new_split = len(r_ids)
            if not (
                new_split >= self.min_samples_per_node
                and (n_items - new_split) >= self.min_samples_per_node
            ):
                continue

            if new_split > max_split:
                max_split = new_split
                node_id = subset_ids[idx]
                # Do not include selected node id
                r_ids = np.setdiff1d(r_ids, np.array([subset_ids[idx]]))
                node_r_ids = r_ids
                node_l_ids = l_ids

            match_mask[idx, :] = r_mask
            match_mask[:, idx] = r_mask

            if max_split == n_items:
                break

            if self.verbose:
                p_bar.set_description(f"Max Split: {max_split}")

        return node_id, node_r_ids, node_l_ids, match_mask

    def fit(self, x: np.ndarray, y: np.ndarray, preserve_match_mask: bool = False):
        self.invalidate_cached_properties()

        n_items = len(x)
        assert n_items == len(y)

        self.root = Node()

        q = [(self.root, np.arange(0, len(x)))]
        n_dims = x.shape[self.feature_dim]

        if not preserve_match_mask:
            # Initialize match mask
            self.match_mask = np.empty((n_dims, n_items, n_items))
            self.match_mask[...] = np.nan

        if self.verbose:
            print(
                f"Ratio of missing predictions: {np.isnan(self.match_mask).sum() / np.prod(self.match_mask.shape)}",
            )

        while len(q) != 0:
            cur_node, subset_ids = q.pop()
            dim = self._sample_dim(n_dims)

            features = self._take_feature_dim(x, dim)
            match_mask = self.match_mask[dim][np.ix_(subset_ids, subset_ids)]

            node_id, node_r_ids, node_l_ids, match_mask = self._greedy_search(
                features, subset_ids, match_mask
            )

            self.match_mask[np.ix_([dim], subset_ids, subset_ids)] = match_mask[
                None, ...
            ]

            if node_l_ids is None or len(node_l_ids) == 0:
                parent_ids = np.array(
                    [parent._split_value[2] for parent in cur_node.get_all_parents()]
                )
                val_ids = np.concatenate([subset_ids, parent_ids]).astype(int)

                cur_node.right = cur_node.left = None
                cur_node.reset_split_value()
                cur_node.set_values(x[val_ids], y[val_ids], val_ids)
                continue

            cur_node.set_split_value(features[node_id], dim, node_id)
            cur_node.left = Node(parent=cur_node)
            cur_node.right = Node(parent=cur_node)
            q.append((cur_node.left, node_l_ids))
            q.append((cur_node.right, node_r_ids))

    def _choose_tighthest_quantiles(self, quantiles: np.ndarray):
        n_split = len(quantiles) // 2
        low_quantiles, high_quantiles = quantiles[:n_split], quantiles[n_split:]
        width = (high_quantiles - low_quantiles).flatten()
        min_width_id = np.argmin(width)
        return np.array([low_quantiles[min_width_id], high_quantiles[min_width_id]])

    def _compute_qrf_quantiles(self, xs: np.ndarray, ys: np.ndarray, x: np.ndarray):
        quantiles = None
        if self.quantile_regressor == 'qrf':
            quantiles = (
                RandomForestQuantileRegressor(
                    n_estimators=10,
                    max_depth=2,
                    criterion="squared_error",
                    q=self.quantiles,
                )
                .fit(xs, ys)
                .predict(x)
                .squeeze()
            )
        elif self.quantile_regressor == 'linear':
            quantiles = np.array([
                QuantileRegressor(quantile=quantile).fit(xs, ys).predict(x)
                for quantile in self.quantiles[1:-1]
            ]).squeeze()
        elif self.quantile_regressor == 'xgboost':
            Xy = xgb.QuantileDMatrix(xs, ys)
            booster = xgb.train(
                {
                    "objective": "reg:quantileerror",
                    "tree_method": "hist",
                    "quantile_alpha": self.quantiles,
                    # Let's try not to overfit.
                    "learning_rate": 0.04,
                    "max_depth": 2,
                },
                Xy,
                num_boost_round=10,
            )
            quantiles = booster.inplace_predict(x)[0]
        else:
            raise Exception(f"Quantile regressor needs to be one of (\"qrf\", \"linear\", \"xgboost\"), {self.quantile_regressor} is provided")

        return self._choose_tighthest_quantiles(quantiles)

    def _compute_qrf_quantiles_from_node(self, x: np.ndarray, node: Node) -> np.ndarray:
        xs, ys, _ = node.get_values()

        n_samples = xs.shape[0]
        quantiles = self._compute_qrf_quantiles(
            xs.reshape(n_samples, -1), ys.reshape(n_samples), x.reshape(1, -1)
        )
        return quantiles

    def predict_single_node(self, x: np.ndarray) -> Node:
        node = self.root

        while node is not None and node.get_split_value() is not None:
            node_val, dim, _ = node.get_split_value()
            features = self._take_feature_dim(x, dim)
            node = node.right if self.compare(features, node_val) else node.left

        return node

    def predict_single(self, x: np.ndarray) -> tuple[Node, np.ndarray]:
        node = self.predict_single_node(x)

        quantiles = self._compute_qrf_quantiles_from_node(x, node)

        return node, quantiles

    def predict(self, x: np.ndarray) -> List[Node]:
        return np.stack([self.predict_single(datum)[1] for idx, datum in enumerate(x)])

    def _val_splits_set(
        self, x: np.ndarray, test_set: np.ndarray
    ) -> Tuple[bool, np.ndarray]:
        r_mask = self.compare_to_many(x, test_set, self.matcher).astype(bool)

        new_split = r_mask.sum()
        n_items = len(test_set)

        needs_split = (
            new_split > self.min_samples_per_node
            and (n_items - new_split) > self.min_samples_per_node
        )

        return needs_split, r_mask

    def _cut_leaf_node_values(self, node: Node, weights: Optional[np.ndarray] = None):
        weights = weights if weights is not None else node.get_values()[2]
        ids_to_keep = np.argsort(weights)
        ids_to_keep = ids_to_keep[-self.n_recent_kept :]

        xs, ys, ids = node._values
        xs = xs[ids_to_keep]
        ys = ys[ids_to_keep]
        ids = ids[ids_to_keep]
        node._values = xs, ys, ids
        return node

    def _sample_dim(self, n_features: int) -> int:
        return np.random.randint(0, n_features)
    
    def _take_feature_dim(self, value: np.ndarray, dim: int) -> np.ndarray:
        return np.take(value, dim, axis=self.feature_dim)

    def predict_single_with_update(
        self,
        x: np.ndarray,
        y: np.ndarray,
        idx: int,
    ) -> Node:
        node, quantiles = self.predict_single(x)

        dim = self._sample_dim(np.expand_dims(x, 0).shape[self.feature_dim])

        test_x = self._take_feature_dim(x, dim)
        features = self._take_feature_dim(node.get_values()[0], dim)

        needs_split, split_mask = (
            self._val_splits_set(test_x, features)
            if self.split_on_predict
            else (False, None)
        )
        updated_nodes = {}
        ext_vals = (x[None, ...], y[None, ...], [idx])

        if needs_split:
            self.invalidate_cached_properties()
            r_values = [
                np.concatenate((prev_vals[split_mask], new_vals))
                for prev_vals, new_vals in zip(node.get_values(), ext_vals)
            ]
            l_values = [vals[~split_mask] for vals in node._values]
            node.reset_values()
            node.set_split_value(test_x, dim, idx)
            node.right = Node(values=r_values, parent=node)
            node.left = Node(values=l_values, parent=node)

            updated_nodes[node.right] = True
            updated_nodes[node.left] = False
            quantiles = self._compute_qrf_quantiles_from_node(x, node.right)
        else:
            updated_vals = [
                np.concatenate((prev_vals, new_vals))
                for prev_vals, new_vals in zip(node.get_values(), ext_vals)
            ]
            node.set_values(*updated_vals)
            updated_nodes[node] = True

        if self.n_recent_kept is not None:
            for (node, relevance_weighted) in updated_nodes.items():
                if len(node.get_values()[0]) <= self.n_recent_kept:
                    continue

                weights = None
                if self.keep_relevant and relevance_weighted:
                    features = self._take_feature_dim(node.get_values()[0], dim)
                    weights = self.compare_to_many(test_x, features, self.relevance_matcher).astype(bool)
                self._cut_leaf_node_values(node, weights)

        return quantiles

    def predict_with_update(
        self,
        x: np.ndarray,
        y: np.ndarray,
        ids: np.ndarray,
    ) -> List[Node]:
        return np.stack(
            [
                self.predict_single_with_update(xi, yi, test_id)
                for xi, yi, test_id in zip(x, y, ids)
            ]
        )

    def reset_updates(self, idx: int):
        for node in self.leaf_nodes:
            xs, ys, ids = node.get_values()
            train_map = ids < idx
            node.set_values(xs[train_map], ys[train_map], ids[train_map])

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["match_mask"]
        del state["matcher"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def invalidate_cached_properties(self):
        if hasattr(self, "depth"):
            del self.depth
        if hasattr(self, "leaf_nodes"):
            del self.leaf_nodes

    @cached_property
    def leaf_nodes(self) -> Sequence[Node]:
        nodes = []
        if self.root is None:
            return nodes

        q = [self.root]
        while len(q) > 0:
            node = q.pop()
            if node._split_value is None:
                nodes.append(node)
                continue

            q.append(node.right)
            q.append(node.left)

        return nodes

    @cached_property
    def depth(self):
        if self.root is None:
            return 0

        max_depth = 0
        q = [(self.root, 0)]
        while len(q) > 0:
            node, depth = q.pop()
            if depth > max_depth:
                max_depth = depth

            if node.right is not None:
                q.append((node.right, depth + 1))

            if node.left is not None:
                q.append((node.left, depth + 1))
        return max_depth


class DistMatchQRF:
    def __init__(
        self,
        n_trees: int,
        matcher: MatcherFnType,
        alpha: float = 0.1,
        feature_dim: int = -1,
        match_mask: np.ndarray | None = None,
        bagging_ratio: float = 0.8,
        n_quantile_bins: int = 5,
        verbose: bool = False,
        batch_size: int | None = None,
        min_samples_per_node: int = 0,
        n_recent_kept: int | None = None,
        split_on_predict: bool = False,
        keep_relevant: bool = False,
        relevance_matcher: Optional[MatcherFnType] = None,
        quantile_regressor: str = 'qrf',
    ):
        assert feature_dim < 0, "Feature dimension must be negative"
        self.feature_dim = feature_dim
        self.verbose = verbose
        self._alpha = alpha
        self._n_quantile_bins = n_quantile_bins

        self.trees: List[DistMatchTree] = [
            DistMatchTree(
                matcher=matcher,
                quantiles=self._get_quantiles(),
                feature_dim=self.feature_dim,
                match_mask=match_mask,
                verbose=verbose,
                batch_size=batch_size,
                min_samples_per_node=min_samples_per_node,
                n_recent_kept=n_recent_kept,
                split_on_predict=split_on_predict,
                keep_relevant=keep_relevant,
                relevance_matcher=relevance_matcher,
                quantile_regressor=quantile_regressor,
            )
            for _ in range(n_trees)
        ]
        self.bagging_ratio = bagging_ratio
        self.match_mask = match_mask

    def _get_quantiles(self):
        high_quantiles = np.linspace(
            start=0, stop=self._alpha, num=self._n_quantile_bins
        )
        return np.concatenate([high_quantiles, 1 - self._alpha + high_quantiles])

    def set_alpha(self, alpha: float, n_quantile_bins: int):
        self._alpha = alpha
        self._n_quantile_bins = n_quantile_bins
        for tree in self.trees:
            tree.quantiles = self._get_quantiles()

    def reset_updates(self, idx: int):
        for tree in self.trees:
            tree.reset_updates(idx)

    def fit_one_tree(self, tree: DistMatchTree, x: np.ndarray, y: np.ndarray):
        n_dims = x.shape[self.feature_dim]
        n_samples = len(x)

        all_dims = range(n_dims)
        all_ids = np.asarray(range(n_samples))
        quantile = int(n_samples * self.bagging_ratio)

        ids = np.random.choice(all_ids, quantile).astype(int)

        mask_ids = np.ix_(all_dims, ids, ids)
        tree.match_mask = self.match_mask[mask_ids]
        tree.fit(x[ids], y[ids], preserve_match_mask=True)

        return tree, mask_ids

    def fit(self, x: np.ndarray, y: np.ndarray, preserve_match_mask: bool = False):
        print("Fitting trees ...")
        n_dims = x.shape[self.feature_dim]
        n_samples = len(x)

        if not preserve_match_mask or self.match_mask is None:
            self.match_mask = np.empty((n_dims, n_samples, n_samples))
            self.match_mask[...] = np.nan

        tree_fn = partial(self.fit_one_tree, x=x, y=y)

        for tree_id, tree in enumerate(self.trees):
            tree, tree_mask_ids = tree_fn(tree)
            self.match_mask[tree_mask_ids] = tree.match_mask
            self.trees[tree_id] = tree

    def predict_from_trees(
        self, *args: np.ndarray, tree_method: str = "predict"
    ) -> List[np.ndarray]:
        values = np.stack(
            [getattr(tree, tree_method)(*args) for tree in self.trees]
        )  # (n_trees, n_samples, 2)

        mean_values = values.mean(axis=0)

        return mean_values

    def predict(self, x: np.ndarray) -> np.ndarray:
        all_values = self.predict_from_trees(x, tree_method="predict")
        return all_values

    def predict_with_update(
        self, x: np.ndarray, y: np.ndarray, test_ids: Sequence
    ) -> np.ndarray:
        all_values = self.predict_from_trees(
            x, y, test_ids, tree_method="predict_with_update"
        )
        return all_values

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["match_mask"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def save(self, filename: str):
        with open(filename, "wb") as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, filename: str) -> "DistMatchQRF":
        with open(filename, "rb") as file:
            return pickle.load(file)

    def load_trees(self, filename: str):
        qrf = DistMatchQRF.load(filename)
        n_trees = len(self.trees)
        assert len(qrf.trees) == n_trees
        for i in range(n_trees):
            self.trees[i].invalidate_cached_properties()
            self.trees[i].root = qrf.trees[i].root
