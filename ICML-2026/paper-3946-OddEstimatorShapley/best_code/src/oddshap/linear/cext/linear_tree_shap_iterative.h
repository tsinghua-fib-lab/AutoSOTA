#ifndef LINEAR_TREE_SHAP_ITERATIVE_H
#define LINEAR_TREE_SHAP_ITERATIVE_H

#include <stdio.h>
#include <cmath>
#include <algorithm>
#include <cstring>


typedef double tfloat;

struct Tree {
    tfloat *weights;
    tfloat *leaf_predictions;
    tfloat *thresholds;
    int *parents;
    int *edge_heights;
    int *features;
    int *children_left;
    int *children_right;
    int  max_depth;
    int  num_nodes;

    Tree(tfloat *weights, tfloat *leaf_predictions, tfloat *thresholds,
    		 int *parents, int *edge_heights,
		 int *features, int *children_left, int *children_right, int max_depth, int num_nodes):
        weights(weights), leaf_predictions(leaf_predictions), thresholds(thresholds),
        parents(parents), edge_heights(edge_heights),
	    features(features), children_left(children_left),
	children_right(children_right), max_depth(max_depth), num_nodes(num_nodes){};

    bool is_internal(int pos)const {
        return children_left[pos] >= 0;
    }
};


#if defined(_WIN32) || defined(WIN32)
#include <malloc.h>
#elif defined(__MVS__)
#include <stdlib.h>
#else
#include <alloca.h>
#endif

using namespace std;

// Stack frame structure for iterative traversal
struct StackFrame {
    int node;
    int feature;
    int depth;
    int stage;  // 0=enter, 1=after_left, 2=after_right, 3=final
};

// Inline psi function (same as recursive version)
inline tfloat psi_iterative(tfloat *e, const tfloat *offset, const tfloat *Base, tfloat q, const tfloat *n, int d)
{
    tfloat res = 0.;
    for (int i = 0; i < d; i++)
    {
        res += e[i] * offset[i] / (Base[i] + q) * n[i];
    }
    // printf("psi_iterative result: %f\n", res);
    return res / d;
}

// Iterative inference function
inline void inference_v2_iterative(
    const Tree &tree,
    const tfloat *Base,
    const tfloat *Offset,
    const tfloat *Norm,
    const tfloat *x,
    bool *activation,
    tfloat *value,
    tfloat *C,
    tfloat *E)
{
    // Allocate stack with sufficient capacity
    const int stack_capacity = tree.max_depth * 5 + 10;
    StackFrame *stack = (StackFrame *)alloca(stack_capacity * sizeof(StackFrame));
    int stack_top = 0;

    // Push initial frame
    stack[stack_top++] = {0, -1, 0, 0};

    // State variables
    tfloat s, q;
    int parent, left, right, offset_degree;
    tfloat *current_e;
    tfloat *child_e;
    tfloat *current_c;
    tfloat *prev_c;
    const tfloat *current_offset;
    const tfloat *current_norm;

    while (stack_top > 0) {
        StackFrame current = stack[--stack_top];
        int node = current.node;
        int feature = current.feature;
        int depth = current.depth;
        int stage = current.stage;

        parent = tree.parents[node];
        left = tree.children_left[node];
        right = tree.children_right[node];

        current_e = E + depth * tree.max_depth;
        child_e = E + (depth + 1) * tree.max_depth;
        current_c = C + depth * tree.max_depth;

        if (stage == 0) {  // Enter node
            s = 0.0;
            if (parent >= 0) {
                activation[node] = activation[node] & activation[parent];
                if (activation[parent]) {
                    s = 1.0 / tree.weights[parent];
                }
            }

            q = 0.0;
            if (feature >= 0) {
                if (activation[node]) {
                    q = 1.0 / tree.weights[node];
                }

                prev_c = C + (depth - 1) * tree.max_depth;
                for (int i = 0; i < tree.max_depth; i++) {
                    current_c[i] = prev_c[i] * (Base[i] + q);
                }

                if (parent >= 0) {
                    for (int i = 0; i < tree.max_depth; i++) {
                        current_c[i] = current_c[i] / (Base[i] + s);
                    }
                }
            }

            if (left >= 0) {  // Internal node
                // Set activation for children
                // printf("Internal node %d: feature = %d, threshold = %f, x[feature] = %f\n",
                //        node, tree.features[node], tree.thresholds[node], x[tree.features[node]]);

                if (x[tree.features[node]] <= tree.thresholds[node]) {
                    activation[left] = true;
                    activation[right] = false;
                } else {
                    activation[left] = false;
                    activation[right] = true;
                }

                // Push frames in reverse order (LIFO stack)
                // Order: stage3 -> stage2 -> right -> stage1 -> left
                stack[stack_top++] = {node, feature, depth, 3};
                stack[stack_top++] = {node, feature, depth, 2};
                stack[stack_top++] = {right, tree.features[node], depth + 1, 0};
                stack[stack_top++] = {node, feature, depth, 1};
                stack[stack_top++] = {left, tree.features[node], depth + 1, 0};
                // printf("Pushed children of node %d onto stack\n", node);
                // Print the current stack
                for (int i = 0; i < stack_top; i++) {
                    // printf("Stack[%d]: node=%d, feature=%d, depth=%d, stage=%d\n",
                    //        i, stack[i].node, stack[i].feature, stack[i].depth, stack[i].stage);
                }
            } else {  // Leaf node
                for (int i = 0; i < tree.max_depth; i++) {
                    current_e[i] = current_c[i] * tree.leaf_predictions[node];
                    // printf("Leaf node %d: prediction = %f\n", node, tree.leaf_predictions[node]);
                    // printf("E[%d][%d] = %f\n", depth, i, current_e[i]);
                }

                // Process feature contribution immediately for leaf nodes
                if (feature >= 0) {
                    if (!(parent >= 0 && !activation[parent])) {
                        q = 0.0;
                        if (activation[node]) {
                            q = 1.0 / tree.weights[node];
                        }

                        current_norm = Norm + tree.edge_heights[node] * tree.max_depth;
                        value[feature] += (q - 1.0) * psi_iterative(
                            current_e,
                            Offset,
                            Base,
                            q,
                            current_norm,
                            tree.edge_heights[node]
                        );

                        if (parent >= 0) {
                            s = 0.0;
                            if (activation[parent]) {
                                s = 1.0 / tree.weights[parent];
                            }

                            offset_degree = tree.edge_heights[parent] - tree.edge_heights[node];
                            current_norm = Norm + tree.edge_heights[parent] * tree.max_depth;
                            current_offset = Offset + offset_degree * tree.max_depth;
                            value[feature] -= (s - 1.0) * psi_iterative(
                                current_e,
                                current_offset,
                                Base,
                                s,
                                current_norm,
                                tree.edge_heights[parent]
                            );
                        }
                    }
                }
            }

        } else if (stage == 1) {  // After left child
            current_offset = Offset + (tree.edge_heights[node] - tree.edge_heights[left]) * tree.max_depth;
            for (int i = 0; i < tree.max_depth; i++) {
                current_e[i] = child_e[i] * current_offset[i];
            }
            // printf("After left child of node %d: current_e = [%f, %f, %f]\n", node, current_e[0], current_e[1], current_e[2]);

        } else if (stage == 2) {  // After right child
            current_offset = Offset + (tree.edge_heights[node] - tree.edge_heights[right]) * tree.max_depth;
            for (int i = 0; i < tree.max_depth; i++) {
                current_e[i] += child_e[i] * current_offset[i];
            }
            // printf("After right child of node %d: current_e = [%f, %f, %f]\n", node, current_e[0], current_e[1], current_e[2]);
        } else if (stage == 3) {  // Final processing for internal nodes
            if (feature >= 0) {
                if (parent >= 0 && !activation[parent]) {
                    continue;
                }

                q = 0.0;
                if (activation[node]) {
                    q = 1.0 / tree.weights[node];
                }

                current_norm = Norm + tree.edge_heights[node] * tree.max_depth;
                // printf("Processing internal node %d for feature %d with q=%f\n", node, feature, q);
                // printf("Current norm at node %d: [%f, %f, %f]\n", node,
                //        current_norm[0], current_norm[1], current_norm[2]);
                // printf("Value[%d] before update: %f\n", feature, value[feature]);
                value[feature] += (q - 1.0) * psi_iterative(
                    current_e,
                    Offset,
                    Base,
                    q,
                    current_norm,
                    tree.edge_heights[node]
                );
                // printf("Value[%d] after node %d: %f\n", feature, node, value[feature]);

                if (parent >= 0) {
                    s = 0.0;
                    if (activation[parent]) {
                        s = 1.0 / tree.weights[parent];
                    }

                    offset_degree = tree.edge_heights[parent] - tree.edge_heights[node];
                    current_norm = Norm + tree.edge_heights[parent] * tree.max_depth;
                    current_offset = Offset + offset_degree * tree.max_depth;
                    value[feature] -= (s - 1.0) * psi_iterative(
                        current_e,
                        current_offset,
                        Base,
                        s,
                        current_norm,
                        tree.edge_heights[parent]
                    );
                    // printf("Value[%d] after parent %d: %f\n", feature, parent, value[feature]);
                }
            }
            // printf("Finished processing internal node %d\n", node);
        }
    }
}

// Main entry point for iterative version
inline void linear_tree_shap_iterative(
    const Tree &tree,
    const tfloat *Base,
    const tfloat *Offset,
    const tfloat *Norm,
    const tfloat* X,
    int n_row,
    int n_col,
    tfloat * out)
{
    int size = (tree.max_depth + 1) * tree.max_depth;

    // Allocate working buffers once
    tfloat *C = new tfloat[size];
    std::fill_n(C, size, 1.);
    tfloat *E = new tfloat[size];
    bool *activation = new bool[tree.num_nodes];

    // Process all rows
    for (int i = 0; i < n_row; i++)
    {
        const tfloat *x = X + i*n_col;
        tfloat *value = out + i*n_col;
        inference_v2_iterative(tree, Base, Offset, Norm, x, activation, value, C, E);
    }

    delete[] C;
    delete[] E;
    delete[] activation;
}

#endif // LINEAR_TREE_SHAP_ITERATIVE_H
