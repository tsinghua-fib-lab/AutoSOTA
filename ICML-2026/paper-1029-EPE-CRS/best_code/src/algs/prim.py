import numpy as np
from utils.subgroup import *

class PRIM:
    def __init__(self, metric, B_subgp, alpha=0.05, min_support=0.1):
        """
        PRIM algorithm for subgroup discovery.

        Parameters
        ----------
        metric : function
            Metric to optimize (takes X_adjust, Y)
        X_adjust_full : array
            Full adjustment features (needed to track indices during peeling)
        B_subgp : array, shape (2, n_subgp_features)
            Bounding box for subgroup space
        alpha : float
            Peeling fraction
        min_support : float
            Minimum support as fraction of total data
        """
        self.B_subgp = B_subgp
        self.alpha = alpha
        self.min_support = min_support
        self.boxes = []
        self.metric = metric

    def _peel(self, X_adjust_current, X_subgp_current, Y_current, current_metric_val):
        """
        Peel one step by removing alpha fraction from one side.

        Parameters
        ----------
        X_adjust_current : array
            Current adjustment features in box
        X_subgp_current : array
            Current subgroup features in box
        Y_current : array
            Current outcomes in box
        current_indices : array
            Indices mapping current data to full X_adjust

        Returns
        -------
        best_mask : bool array or None
            Mask for points to keep
        """
        best_mask = None
        best_mean = current_metric_val
        n_subgp_features = X_subgp_current.shape[1]

        for j in range(n_subgp_features):
            for direction in ['up', 'down']:
                if direction == 'up':
                    threshold = np.percentile(X_subgp_current[:, j], 100 * (1 - self.alpha))
                    mask = X_subgp_current[:, j] <= threshold
                else:
                    threshold = np.percentile(X_subgp_current[:, j], 100 * self.alpha)
                    mask = X_subgp_current[:, j] >= threshold

                remaining_X_adjust = X_adjust_current[mask]
                remaining_Y = Y_current[mask]

                try:
                    current_mean = self.metric(remaining_X_adjust, remaining_Y)
                except:
                    current_mean = -np.inf

                if current_mean > best_mean:
                    best_mean = current_mean
                    best_mask = mask

        return best_mask, best_mean

    def _paste(self, X_adjust_full, X_subgp_full, Y_full, R):
        """
        Paste one step by expanding region.

        Parameters
        ----------
        X_adjust_full : array
            Full adjustment features
        X_subgp_full : array
            Full subgroup features
        Y_full : array
            Full outcomes
        R : array, shape (2, n_subgp_features)
            Current region in subgroup space

        Returns
        -------
        best_R : array
            Best expanded region
        """
        box = in_region(X_subgp_full, R)
        current_X_adjust = X_adjust_full[box]
        current_Y = Y_full[box]

        try:
            best_mean = self.metric(current_X_adjust, current_Y)
        except:
            best_mean = -np.inf
        best_R = R

        n_subgp_features = X_subgp_full.shape[1]

        for j in range(n_subgp_features):
            for direction in ['up', 'down']:
                if direction == 'up':
                    current_quantile = np.sum(X_subgp_full[:, j] <= R[1, j]) / len(X_subgp_full)
                    threshold = np.percentile(X_subgp_full[:, j], min(100, 100 * (current_quantile + self.alpha)))
                    new_R = R.copy()
                    new_R[1, j] = threshold
                else:
                    current_quantile = np.sum(X_subgp_full[:, j] <= R[0, j]) / len(X_subgp_full)
                    threshold = np.percentile(X_subgp_full[:, j], max(0, 100 * (current_quantile - self.alpha)))
                    new_R = R.copy()
                    new_R[0, j] = threshold

                mask = in_region(X_subgp_full, new_R)
                pasted_X_adjust = X_adjust_full[mask]
                pasted_Y = Y_full[mask]

                try:
                    current_mean = self.metric(pasted_X_adjust, pasted_Y)
                except:
                    current_mean = -np.inf

                if current_mean > best_mean:
                    best_mean = current_mean
                    best_R = new_R

        return best_R

    def fit(self, X_adjust, X_subgp, Y):
        """
        Fit PRIM to find one subgroup.

        Parameters
        ----------
        X_adjust : array
            Adjustment features
        X_subgp : array
            Subgroup features
        Y : array
            Outcomes
        """
        current_X_adjust = X_adjust.copy()
        current_X_subgp = X_subgp.copy()
        current_Y = Y.copy()
        current_metric_val = -np.inf
        support = self.min_support * len(X_adjust)

        # Peeling steps
        print('Start peeling.')

        while len(current_X_adjust) >= support:
            mask, current_metric_val = self._peel(current_X_adjust, current_X_subgp, current_Y, current_metric_val)

            if mask is None:
                break
            else:
                current_X_adjust = current_X_adjust[mask]
                current_X_subgp = current_X_subgp[mask]
                current_Y = current_Y[mask]

        # Get bounding box of peeled region in subgroup space
        R = bounding_box(current_X_subgp)
        print(f'Start pasting from box: {R}')

        # Pasting steps
        print('Start pasting.')
        while True:
            new_R = self._paste(X_adjust, X_subgp, Y, R)
            if np.linalg.norm(new_R - R) < 1e-6:
                break
            else:
                R = new_R
            print(f'Current box: {R}')

        # Store the box (no expansion needed - R is already in subgroup space)
        self.boxes.append(R)

    def get_boxes(self):
        return self.boxes


def prim_job(X_adjust, X_subgp, Y, B_subgp, num_subgroups, peeling_frac, min_support_size):
    """
    PRIM job for subgroup discovery.

    Parameters
    ----------
    X_adjust : array
        Adjustment features
    X_subgp : array
        Subgroup features
    Y : array
        Outcomes
    B_subgp : array, shape (2, n_subgp_features)
        Bounding box for subgroup space
    num_subgroups : int
        Number of subgroups to find
    peeling_frac : float
        Peeling fraction (alpha)
    min_support_size : float
        Minimum support as fraction

    Returns
    -------
    results : list of dict
        Results for each subgroup
    """
    results = []
    X_adjust_remaining = X_adjust.copy()
    X_subgp_remaining = X_subgp.copy()
    Y_remaining = Y.copy()

    pr = PRIM(epe_metric, B_subgp, alpha=peeling_frac, min_support=min_support_size)

    for subgroup_id in range(num_subgroups):
        pr.fit(X_adjust_remaining, X_subgp_remaining, Y_remaining)
        R = pr.boxes[-1]

        # Find points in this region (in remaining data)
        selected_pts = in_region(X_subgp_remaining, R)

        try:
            beta = fit_cox(X_adjust_remaining[selected_pts], Y_remaining[selected_pts])
            results.append({
                'subgroup_id': subgroup_id,
                'R': R.copy(),
                'beta': beta.copy(),
                'peeling_frac': peeling_frac,
                'min_support_size': min_support_size
            })
        except:
            print(f"Failed to fit Cox model for subgroup {subgroup_id}, "
                  f"peeling frac {peeling_frac}, min supp size {min_support_size}.")
            results.append({
                'subgroup_id': subgroup_id,
                'R': R.copy(),
                'beta': np.full(X_adjust.shape[1], np.nan),
                'peeling_frac': peeling_frac,
                'min_support_size': min_support_size
            })

        # Remove selected points from remaining data
        X_adjust_remaining = X_adjust_remaining[~selected_pts]
        X_subgp_remaining = X_subgp_remaining[~selected_pts]
        Y_remaining = Y_remaining[~selected_pts]

    return results