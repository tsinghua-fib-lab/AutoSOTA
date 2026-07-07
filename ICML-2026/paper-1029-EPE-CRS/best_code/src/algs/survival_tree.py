import numpy as np
from sksurv.tree import SurvivalTree
from utils.subgroup import get_best_boxes, tree_to_bounding_boxes, epe_metric



def survival_tree_job(X_adjust, X_subgp, Y, B, num_subgroups, max_depth, min_samples_leaf):
    st = SurvivalTree(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features=None
    )
    st.fit(X_subgp, Y)
    results = []
    for depth in range(1, max_depth + 1):
        boxes = tree_to_bounding_boxes(st.tree_, depth, B)

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
                'min_samples_leaf': min_samples_leaf
            })
    return results