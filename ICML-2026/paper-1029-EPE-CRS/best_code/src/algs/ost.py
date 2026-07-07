import numpy as np
import pandas as pd
from utils.subgroup import get_best_boxes, epe_metric


def osst_tree_to_bounding_boxes(source, B_subgp):
    """
    Extract leaf bounding boxes from an OSST TreeSurvivalRegressor source dict.

    Parameters
    ----------
    source : dict
        Root node dict from model.tree.source (OSST recursive dict representation).
        Internal nodes: {'feature': int, 'reference': float, 'relation': str,
                         'true': node, 'false': node}
        Leaf nodes: {'prediction': ...}
        For numerical features OSST uses '>=' splits only.
    B_subgp : ndarray, shape (2, d)
        Bounding box for the subgroup feature space.

    Returns
    -------
    boxes : list of ndarray, each shape (2, d)
        Bounding box for each leaf.
    """
    boxes = []

    def _recurse(node, box):
        if "prediction" in node:  # leaf node
            boxes.append(box.copy())
            return

        feat_idx = node["feature"]
        threshold = node["reference"]
        relation = node["relation"]

        true_box = box.copy()
        false_box = box.copy()

        if relation == ">=":
            # true branch: feature >= threshold → tighten lower bound
            true_box[0, feat_idx] = max(true_box[0, feat_idx], threshold)
            # false branch: feature < threshold → tighten upper bound
            false_box[1, feat_idx] = min(false_box[1, feat_idx], threshold)
        # For "==": categorical split; can't represent as axis-aligned interval → leave bounds unchanged

        _recurse(node["true"], true_box)
        _recurse(node["false"], false_box)

    _recurse(source, B_subgp.copy())
    return boxes


def ost_job(X_adjust, X_subgp, Y, B, num_subgroups, regularization, minimum_captured_points, depth_budget):
    from osst.model.osst import OSST

    feature_names = list(range(X_subgp.shape[1]))
    X_subgp_df = pd.DataFrame(X_subgp, columns=feature_names)
    event = np.array(Y['failure'], dtype=int)
    times = np.array(Y['time'], dtype=float)

    config = {
        "regularization": regularization,
        "depth_budget": depth_budget,
        "minimum_captured_points": minimum_captured_points,
        "verbose": False,
        "time_limit": 600,
    }
    model = OSST(config)
    model.fit(X_subgp_df, event, times)

    boxes = osst_tree_to_bounding_boxes(model.tree.source, B)
    boxes_and_betas = get_best_boxes(X_adjust, X_subgp, Y, boxes, num_subgroups, epe_metric)

    results = []
    for subgroup_id in range(num_subgroups):
        R, beta = boxes_and_betas[subgroup_id]
        results.append({
            'subgroup_id': subgroup_id,
            'R': R,
            'beta': beta,
            'regularization': regularization,
            'minimum_captured_points': minimum_captured_points,
            'depth_budget': depth_budget,
        })
    return results
