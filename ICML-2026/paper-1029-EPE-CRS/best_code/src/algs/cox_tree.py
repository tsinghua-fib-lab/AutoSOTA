import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
from utils.subgroup import get_best_boxes, tree_to_bounding_boxes, epe_metric


class CoxTree:
    def __init__(self, criterion, max_depth=None, min_samples_split=2, min_samples_leaf=1, max_splits_per_feature=None):
        """
        Parameters:
          criterion : function
              A function with signature criterion(X_node, y_node) that returns a scalar impurity.
          subgp_features : list of indices or None
              Splits only performed on these features. If None, all features considered.
          max_depth : int or None
              Maximum depth of the tree. If None, then no maximum.
          min_samples_split : int
              Minimum number of samples required to attempt a split.
          min_samples_leaf : int
              Minimum number of samples required in a leaf node.
          max_splits_per_feature : int or None
              Number of threshold values to check per feature when splitting a node in the tree.
              Thresholds are roughly spaced among max_splits_per_feature quantiles of the unique values for that feature.
              If None, checks all split values.
        """
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_splits_per_feature = max_splits_per_feature

        # Following scikit-learn's internal representation
        self.children_left = []   # left child indices
        self.children_right = []  # right child indices
        self.feature = []         # feature used for split at each node (-2 indicates a leaf)
        self.threshold = []       # threshold value for the split (arbitrary for leaves)
        self.value = []           # prediction value at each node (mean target)
        self.node_count = 0
        self.n_features = 0

    def fit(self, X_adjust, X_subgp, y):
        """
        Fit the regression tree to X and y.
        """
        X_adjust = np.asarray(X_adjust)
        X_subgp = np.asarray(X_subgp)
        y = np.asarray(y)
        # Clear any previous tree structure
        self.children_left = []
        self.children_right = []
        self.feature = []
        self.threshold = []
        self.value = []
        self.node_count = 0
        self.n_features = X_subgp.shape[1]

        # Build tree recursively, starting at depth 0.
        self._build_tree(X_adjust, X_subgp, y, depth=0)
        # Convert lists to numpy arrays for scikit-learn–like API
        self.children_left = np.array(self.children_left, dtype=np.int32)
        self.children_right = np.array(self.children_right, dtype=np.int32)
        self.feature = np.array(self.feature, dtype=np.int32)
        self.threshold = np.array(self.threshold, dtype=np.float64)
        self.value = np.array(self.value, dtype=np.float64).reshape(-1, 1)
        return self

    def _build_tree(self, X_adjust, X_subgp, y, depth):
        """
        Recursively build the tree. Returns the index of the current node.
        """
        node_index = len(self.value)  # current node's index will be the next available index

        # Increase node_count for every new node created.
        self.node_count += 1

        # Evaluate impurity of the current node and store it.
        impurity_parent = self.criterion(X_adjust, y)
        self.value.append(impurity_parent)
        # Default: mark as a leaf node (feature=-2, threshold=-2)
        self.feature.append(-2)
        self.threshold.append(-2.0)
        self.children_left.append(-1)
        self.children_right.append(-1)

        n_samples, n_features = X_subgp.shape

        # Check stopping conditions
        if depth == self.max_depth or n_samples < self.min_samples_split:
            return node_index

        
        best_impurity = np.inf
        best_feature = None
        best_threshold = None
        best_splits = None

        # Try each feature for splitting.
        for feat in range(n_features):
            X_subgp_feat = X_subgp[:, feat]
            # sort unique values and consider midpoints between them as candidate thresholds
            unique_values = np.unique(X_subgp_feat)
            if unique_values.size == 1:
                continue  # cannot split on constant feature

            # Select candidate thresholds based on max_splits_per_feature
            if self.max_splits_per_feature is not None and unique_values.size > self.max_splits_per_feature:
                quantiles = np.linspace(0, 100, self.max_splits_per_feature + 2)[1:-1]  # Exclude min/max
                candidates = np.percentile(unique_values, quantiles)
            else:
                candidates = (unique_values[:-1] + unique_values[1:]) / 2.0  # Midpoints between unique values
                
            def evaluate(t):
                # Split the data
                left_mask = X_subgp_feat <= t
                right_mask = X_subgp_feat > t
                if left_mask.sum() < self.min_samples_leaf or right_mask.sum() < self.min_samples_leaf:
                    return (np.inf, t)  # do not allow splits that violate min_samples_leaf

                X_adjust_left, X_subgp_left, y_left = X_adjust[left_mask], X_subgp[left_mask], y[left_mask]
                X_adjust_right, X_subgp_right, y_right = X_adjust[right_mask], X_adjust[right_mask], y[right_mask]
                # Compute impurity for children nodes
                try:
                    impurity_left = self.criterion(X_adjust_left, y_left)
                    impurity_right = self.criterion(X_adjust_right, y_right)
                    n_left = X_adjust_left.shape[0]
                    n_right = X_adjust_right.shape[0]
                    impurity_split = (n_left * impurity_left + n_right * impurity_right) / n_samples
                except:
                    impurity_split = np.inf  # If an error occurs, treat as invalid split
                return (impurity_split, t)

            results = Parallel(n_jobs=-1)(
                delayed(evaluate)(t) for t in tqdm(candidates)
            )

            best_feat_impurity = min(results, key=lambda x: x[0])
            if best_feat_impurity[0] < best_impurity:
                impurity_split, t = best_feat_impurity
                best_impurity = impurity_split
                best_feature = feat
                best_threshold = t
                best_splits = (X_subgp_feat <= t, X_subgp_feat > t)

        # If no valid split was found that improves impurity, return as a leaf.
        if best_feature is None or best_impurity >= impurity_parent:
            return node_index

        # Record the split information in the current node.
        self.feature[node_index] = best_feature
        self.threshold[node_index] = best_threshold

        # Build left and right subtrees recursively.
        left_mask, right_mask = best_splits

        # Remember current node index; new node indices will be added to the lists.
        left_child = self._build_tree(X_adjust[left_mask], X_subgp[left_mask], y[left_mask], depth + 1)
        right_child = self._build_tree(X_adjust[right_mask], X_subgp[right_mask], y[right_mask], depth + 1)

        # Update children information for current node.
        self.children_left[node_index] = left_child
        self.children_right[node_index] = right_child

        return node_index



def cox_tree_job(X_adjust, X_subgp, Y, B, num_subgroups, max_depth, min_samples_leaf, max_splits_per_feature):
    ct = CoxTree(epe_metric,
                 max_depth=max_depth, 
                 min_samples_split=2*min_samples_leaf, 
                 min_samples_leaf=min_samples_leaf, 
                 max_splits_per_feature=max_splits_per_feature)
    ct.fit(X_adjust, X_subgp, Y)
    results = []
    # for depth in range(1, max_depth + 1):
    #     boxes = tree_to_bounding_boxes(ct, depth, B)
    #     results.append(get_best_boxes(X, Y, boxes, num_subgroups, epe_metric))
    
    for depth in range(1, max_depth + 1):
        boxes = tree_to_bounding_boxes(ct, depth, B)

        # get_best_boxes returns a list of len max_depth
        # Each entry in boxes_and_betas is a list of len num_subgroups
        # Each entry in these sub-lists is a tuple (box, beta)
        boxes_and_betas = get_best_boxes(X_adjust, X_subgp, Y, boxes, num_subgroups, epe_metric)
        for subgroup_id in range(num_subgroups):
            R, beta = boxes_and_betas[subgroup_id]
            results.append({
                'subgroup_id': subgroup_id,
                'R': R,
                'beta': beta,
                'max_depth': depth,
                'min_samples_leaf': min_samples_leaf,
                'max_splits_per_feature': max_splits_per_feature
            })
    return results