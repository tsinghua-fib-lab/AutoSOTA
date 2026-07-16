# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False

"""Cython implementation of interventional CII computation."""

cimport numpy as cnp
cimport scipy.special.cython_special as cscipy
from libcpp.vector cimport vector
from libcpp.unordered_set cimport unordered_set
from cython.operator cimport dereference, preincrement
cimport cython

ctypedef cnp.float64_t DTYPE_t
ctypedef cnp.int32_t INT_t
ctypedef cnp.uint64_t UINT64_t

# C++ class to represent a node in the stack
cdef cppclass StackNode:
    int node_id
    unordered_set[int] A_set
    unordered_set[int] B_set
    unordered_set[int] NB_set
    
    StackNode():
        pass

# C++ unordered_set helper functions
cdef inline unordered_set[int] set_intersection(unordered_set[int] a, unordered_set[int] b):
    """Compute intersection of two sets."""
    cdef unordered_set[int] result
    cdef int elem
    cdef unordered_set[int].iterator it = a.begin()
    while it != a.end():
        elem = dereference(it)
        if b.count(elem) > 0:
            result.insert(elem)
        preincrement(it)
    return result

cdef inline unordered_set[int] set_union(unordered_set[int] a, unordered_set[int] b):
    """Compute union of two sets."""
    cdef unordered_set[int] result = a
    cdef int elem
    cdef unordered_set[int].iterator it = b.begin()
    while it != b.end():
        elem = dereference(it)
        result.insert(elem)
        preincrement(it)
    return result

cdef inline unordered_set[int] set_difference(unordered_set[int] a, unordered_set[int] b):
    """Compute set difference a - b."""
    cdef unordered_set[int] result
    cdef int elem
    cdef unordered_set[int].iterator it = a.begin()
    while it != a.end():
        elem = dereference(it)
        if b.count(elem) == 0:
            result.insert(elem)
        preincrement(it)
    return result



cdef inline double shapley_based_weight(
    unordered_set[int] A_set,
    unordered_set[int] B_set,
    unordered_set[int] N_set,
    unordered_set[int] U_set,
    int n_features
):
    """Compute Shapley-based weight for given sets."""
    cdef unordered_set[int] B_intersect_U = set_intersection(B_set, U_set)
    cdef unordered_set[int] B_union_U = set_union(B_set, U_set)
    cdef unordered_set[int] N_diff_B_union_U = set_difference(N_set, B_union_U)
    
    cdef int a = <int>A_set.size() - <int>B_intersect_U.size()
    cdef int b = <int>N_diff_B_union_U.size()
    
    if a + b < 0:
        return 0.0
    
    cdef double binom_val = cscipy.binom(a + b, b)
    if binom_val == 0.0:
        return 0.0
    
    return 1.0 / ((a + b + 1) * binom_val)


cdef void generate_powerset_and_update(
    dict interactions_dict,
    double const_prediction,
    unordered_set[int] A_set,
    unordered_set[int] B_set,
    unordered_set[int] NB_set,
    unordered_set[int] N_set,
    int max_order,
    int n_features
):
    """Generate powerset of A ∪ NB and update interaction values."""
    cdef unordered_set[int] A_union_NB = set_union(A_set, NB_set)
    cdef unordered_set[int] U_set
    cdef double weight
    cdef int sign
    cdef list U_list
    cdef tuple U_tuple
    cdef int elem
    cdef unordered_set[int].iterator it
    cdef list elements_list = []
    
    # Collect and sort the candidate indices once
    it = A_union_NB.begin()
    while it != A_union_NB.end():
        elements_list.append(dereference(it))
        preincrement(it)
    elements_list.sort()
    
    cdef vector[int] elements_vec
    for elem in elements_list:
        elements_vec.push_back(<int>elem)
    
    cdef int n_elements = <int>elements_vec.size()
    if n_elements == 0:
        return
    cdef int max_subset = 1 << n_elements
    cdef int subset_mask, i
    cdef int subset_size
    cdef int nb_count
    # Iterate through all non-empty subsets
    for subset_mask in range(1, max_subset):
        U_set.clear()
        U_list = []
        subset_size = 0
        nb_count = 0
        for i in range(n_elements):
            if subset_mask & (1 << i):
                elem = elements_vec[i]
                U_set.insert(elem)
                U_list.append(elem)
                subset_size += 1
                if subset_size > max_order:
                    break
                if NB_set.count(elem) > 0:
                    nb_count += 1
        
        # Check max_order constraint
        if subset_size == 0 or subset_size > max_order:
            continue
        
        # Compute sign based on parity of intersection with NB
        sign = 1 if (nb_count % 2 == 0) else -1
        
        # Compute weight
        weight = sign * shapley_based_weight(A_set, B_set, N_set, U_set, n_features)
        
        # Convert ordered list to tuple for dictionary key
        U_tuple = tuple(U_list)
        
        # Update dictionary
        if U_tuple in interactions_dict:
            interactions_dict[U_tuple] += weight * const_prediction
        else:
            interactions_dict[U_tuple] = weight * const_prediction


@cython.boundscheck(False)
@cython.wraparound(False)
def compute_interventional_cii_values_cython(
    cnp.ndarray[DTYPE_t, ndim=1] x,
    cnp.ndarray[DTYPE_t, ndim=2] reference_data,
    cnp.ndarray[INT_t, ndim=1] children_left,
    cnp.ndarray[INT_t, ndim=1] children_right,
    cnp.ndarray[INT_t, ndim=1] features,
    cnp.ndarray[DTYPE_t, ndim=1] thresholds,
    cnp.ndarray[DTYPE_t, ndim=1] values,
    int max_order,
    str decision_type="<="
):
    """Compute interventional CII values using Cython.
    
    Args:
        x: Instance to explain (1D array).
        reference_data: Background dataset (2D array).
        children_left: Left children indices.
        children_right: Right children indices.
        features: Feature indices for each node.
        thresholds: Threshold values for each node.
        values: Leaf node values.
        max_order: Maximum interaction order.
        decision_type: Decision comparison type ("<" or "<=").
    
    Returns:
        Dictionary mapping interaction tuples to their values.
    """
    cdef int n_features = x.shape[0]
    cdef int n_references = reference_data.shape[0]
    cdef int n_nodes = children_left.shape[0]
    cdef double D = <double>n_references
    
    # Initialize result dictionary
    interactions_dict = {}
    
    # Initialize N set with all features (0 to n_features-1)
    cdef unordered_set[int] N_set
    cdef int i
    for i in range(n_features):
        N_set.insert(i)
    
    # Determine decision function
    cdef bint use_strict_less = (decision_type == "<")
    
    # Stack for DFS traversal using C++ vector
    cdef vector[StackNode] stack
    cdef StackNode current
    cdef StackNode node_to_push
    
    cdef int r_idx, node_id, feature_index, child_node_x, child_node_ref
    cdef bint is_inner_node, same_child
    cdef unordered_set[int] A_set, B_set
    cdef double const_coalition
    cdef DTYPE_t x_val, ref_val, threshold
    cdef bint feature_in_B, feature_in_A
    
    # Iterate over reference points
    for r_idx in range(n_references):
        # Initialize stack with root node
        stack.clear()
        node_to_push.node_id = 0
        node_to_push.A_set.clear()
        node_to_push.B_set = N_set
        node_to_push.NB_set.clear()
        stack.push_back(node_to_push)
        
        # DFS traversal
        while not stack.empty():
                # Pop from stack
                current = stack.back()
                stack.pop_back()
                node_id = current.node_id
                A_set = current.A_set
                B_set = current.B_set
                
                # Check if inner node
                is_inner_node = (children_left[node_id] != children_right[node_id])
                
                if is_inner_node:
                    # Inner node processing
                    feature_index = features[node_id]
                    threshold = thresholds[node_id]
                    x_val = x[feature_index]
                    ref_val = reference_data[r_idx, feature_index]
                    
                    # Determine child nodes
                    if use_strict_less:
                        child_node_x = children_left[node_id] if x_val < threshold else children_right[node_id]
                        child_node_ref = children_left[node_id] if ref_val < threshold else children_right[node_id]
                    else:
                        child_node_x = children_left[node_id] if x_val <= threshold else children_right[node_id]
                        child_node_ref = children_left[node_id] if ref_val <= threshold else children_right[node_id]
                    
                    same_child = (child_node_x == child_node_ref)
                    
                    feature_in_B = (B_set.count(feature_index) > 0)
                    feature_in_A = (A_set.count(feature_index) > 0)

                    if same_child:
                        # Both go to same child
                        node_to_push.node_id = child_node_x
                        node_to_push.A_set = A_set
                        node_to_push.B_set = B_set
                        node_to_push.NB_set = current.NB_set
                        stack.push_back(node_to_push)
                    else:
                        # Paths diverge
                        # Check if feature in B
                        if feature_in_B:
                            node_to_push.node_id = child_node_x
                            node_to_push.A_set = A_set
                            node_to_push.A_set.insert(feature_index)
                            node_to_push.B_set = B_set
                            node_to_push.NB_set = current.NB_set
                            stack.push_back(node_to_push)
                        
                        # Check if feature not in A
                        if not feature_in_A:
                            node_to_push.node_id = child_node_ref
                            node_to_push.A_set = A_set
                            node_to_push.B_set = B_set
                            node_to_push.B_set.erase(feature_index)
                            node_to_push.NB_set = current.NB_set
                            if feature_in_B:
                                node_to_push.NB_set.insert(feature_index)
                            stack.push_back(node_to_push)
                else:
                    # Leaf node - update interactions
                    const_coalition = values[node_id] / D
                    
                    generate_powerset_and_update(
                        interactions_dict,
                        const_coalition,
                        A_set,
                        B_set,
                        current.NB_set,
                        N_set,
                        max_order,
                        n_features
                    )
    
    return interactions_dict


@cython.boundscheck(False)
@cython.wraparound(False)
def compute_interventional_cii_multi_tree(
    cnp.ndarray[DTYPE_t, ndim=1] x,
    cnp.ndarray[DTYPE_t, ndim=2] reference_data,
    list children_left_list,
    list children_right_list,
    list features_list,
    list thresholds_list,
    list values_list,
    int max_order,
    str decision_type="<="
):
    """Compute interventional CII values for multiple trees.
    
    Args:
        x: Instance to explain.
        reference_data: Background dataset.
        children_left_list: List of left children arrays for each tree.
        children_right_list: List of right children arrays for each tree.
        features_list: List of feature arrays for each tree.
        thresholds_list: List of threshold arrays for each tree.
        values_list: List of value arrays for each tree.
        max_order: Maximum interaction order.
        decision_type: Decision comparison type.
    
    Returns:
        Dictionary with aggregated CII values across all trees.
    """
    cdef int n_trees = len(children_left_list)
    cdef int tree_idx
    cdef dict tree_interactions
    cdef dict aggregated_interactions = {}
    cdef tuple key
    cdef double value
    
    # Process each tree
    for tree_idx in range(n_trees):
        tree_interactions = compute_interventional_cii_values_cython(
            x,
            reference_data,
            children_left_list[tree_idx],
            children_right_list[tree_idx],
            features_list[tree_idx],
            thresholds_list[tree_idx],
            values_list[tree_idx],
            max_order,
            decision_type
        )
        
        # Aggregate results
        for key, value in tree_interactions.items():
            if key in aggregated_interactions:
                aggregated_interactions[key] += value
            else:
                aggregated_interactions[key] = value
    
    return aggregated_interactions
