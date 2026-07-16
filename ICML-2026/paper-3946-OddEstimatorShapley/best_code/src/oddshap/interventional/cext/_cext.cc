#define NPY_NO_DEPRECATED_API NPY_1_19_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>
#include "interventional.h"

// Module methods (file scope, not inside any class)
static PyObject *_cext_interventional_iterative(PyObject *self, PyObject *args, PyObject *kwargs);
static PyObject *_cext_interventional_iterative_bitmask(PyObject *self, PyObject *args, PyObject *kwargs);
static PyObject *_cext_interventional_iterative_direct(PyObject *self, PyObject *args, PyObject *kwargs);

static PyMethodDef module_methods[] = {
    {"interventional_iterative", (PyCFunction)_cext_interventional_iterative, METH_VARARGS | METH_KEYWORDS, "C++ iterative implementation of Interventional SHAP."},
    {"interventional_iterative_bitmask", (PyCFunction)_cext_interventional_iterative_bitmask, METH_VARARGS | METH_KEYWORDS, "Old bitmask-based implementation (32-feature limit, for benchmarking)."},
    {"interventional_iterative_direct", (PyCFunction)_cext_interventional_iterative_direct, METH_VARARGS | METH_KEYWORDS, "Direct array indexing implementation (unlimited features, O(1) access)."},
    {NULL, NULL, 0, NULL}
};

#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "_cext",
    "This module provides an interface for a interventional Tree SHAP implementation.",
    -1,
    module_methods,
    NULL,
    NULL,
    NULL,
    NULL
};
#endif

#if PY_MAJOR_VERSION >= 3
PyMODINIT_FUNC PyInit__cext(void)
#else
PyMODINIT_FUNC init_cext(void)
#endif
{
    #if PY_MAJOR_VERSION >= 3
        PyObject *module = PyModule_Create(&moduledef);
        if (!module)
            return NULL;
    #else
        PyObject *module = Py_InitModule("_cext", module_methods);
        if (!module) return;
    #endif

    /* Load `numpy` functionality. */
    import_array();

    #if PY_MAJOR_VERSION >= 3
        return module;
    #endif
}

static PyObject *_cext_interventional_iterative(PyObject *self, PyObject *args, PyObject *kwargs)
{
    PyObject *leaf_predictions_obj;
    PyObject *thresholds_obj;
    PyObject *features_obj;
    PyObject *children_left_obj;
    PyObject *children_right_obj;
    PyObject *X_point_obj;
    PyObject *X_reference_obj;
    PyObject *out_contribs_obj;
    int max_order = -1;  // Default: no limit

    /* Parse the input tuple with optional keyword argument */
    /* Now expects lists of tree arrays for multi-tree support */
    static char *kwlist[] = {(char*)"X_point", (char*)"X_reference", (char*)"children_left", 
                             (char*)"children_right", (char*)"features", (char*)"thresholds", 
                             (char*)"leaf_predictions", (char*)"out_contribs", 
                             (char*)"max_order", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(
            args, kwargs, "OOOOOOOO|i", kwlist,
            &X_point_obj,
            &X_reference_obj,
            &children_left_obj,
            &children_right_obj,
            &features_obj,
            &thresholds_obj,
            &leaf_predictions_obj,
            &out_contribs_obj,
            &max_order))
        return NULL;
    
    // Check if inputs are lists (multi-tree) or single arrays
    int is_multi_tree = PyList_Check(children_left_obj);
    int n_trees = is_multi_tree ? PyList_Size(children_left_obj) : 1;
    
    // Parse X arrays (same for all trees)
    PyArrayObject *X_array = (PyArrayObject *)PyArray_FROM_OTF(X_point_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *X_reference_array = (PyArrayObject *)PyArray_FROM_OTF(X_reference_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *out_contribs_array = (PyArrayObject *)PyArray_FROM_OTF(out_contribs_obj, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY2);
    
    if (X_array == NULL || X_reference_array == NULL || out_contribs_array == NULL) {
        Py_XDECREF(X_array);
        Py_XDECREF(X_reference_array);
        PyArray_ResolveWritebackIfCopy(out_contribs_array);
        Py_XDECREF(out_contribs_array);
        return NULL;
    }
    
    tfloat *X_point = (tfloat *)PyArray_DATA(X_array);
    tfloat *X_ref = (tfloat *)PyArray_DATA(X_reference_array);
    tfloat *out_contribs = (tfloat *)PyArray_DATA(out_contribs_array);
    
    const unsigned row_x = PyArray_DIM(X_reference_array, 0);
    const unsigned col_x = PyArray_DIM(X_reference_array, 1);
    
    // Prepare result dictionary (will accumulate results from all trees)
    PyObject* result_dict = PyDict_New();
    
    // Loop through all trees
    for (int tree_idx = 0; tree_idx < n_trees; tree_idx++) {
        // Get tree arrays for this tree
        PyObject *current_children_left_obj = is_multi_tree ? PyList_GetItem(children_left_obj, tree_idx) : children_left_obj;
        PyObject *current_children_right_obj = is_multi_tree ? PyList_GetItem(children_right_obj, tree_idx) : children_right_obj;
        PyObject *current_features_obj = is_multi_tree ? PyList_GetItem(features_obj, tree_idx) : features_obj;
        PyObject *current_thresholds_obj = is_multi_tree ? PyList_GetItem(thresholds_obj, tree_idx) : thresholds_obj;
        PyObject *current_leaf_predictions_obj = is_multi_tree ? PyList_GetItem(leaf_predictions_obj, tree_idx) : leaf_predictions_obj;
        
        /* Interpret the input objects as numpy arrays. */
        PyArrayObject *leaf_predictions_array = (PyArrayObject *)PyArray_FROM_OTF(current_leaf_predictions_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
        PyArrayObject *thresholds_array = (PyArrayObject *)PyArray_FROM_OTF(current_thresholds_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
        PyArrayObject *features_array = (PyArrayObject *)PyArray_FROM_OTF(current_features_obj, NPY_INT, NPY_ARRAY_IN_ARRAY);
        PyArrayObject *children_left_array = (PyArrayObject *)PyArray_FROM_OTF(current_children_left_obj, NPY_INT, NPY_ARRAY_IN_ARRAY);
        PyArrayObject *children_right_array = (PyArrayObject *)PyArray_FROM_OTF(current_children_right_obj, NPY_INT, NPY_ARRAY_IN_ARRAY);

        /* If that didn't work, throw an exception. */
        if (children_left_array == NULL || children_right_array == NULL ||
            features_array == NULL || leaf_predictions_array == NULL ||
            thresholds_array == NULL)
        {
            Py_XDECREF(children_left_array);
            Py_XDECREF(children_right_array);
            Py_XDECREF(features_array);
            Py_XDECREF(leaf_predictions_array);
            Py_XDECREF(thresholds_array);
            Py_XDECREF(X_array);
            Py_XDECREF(X_reference_array);
            PyArray_ResolveWritebackIfCopy(out_contribs_array);
            Py_XDECREF(out_contribs_array);
            Py_DECREF(result_dict);
            return NULL;
        }

        // Get pointers to the data as C-types
        tfloat *leaf_predictions = (tfloat *)PyArray_DATA(leaf_predictions_array);
        tfloat *thresholds = (tfloat *)PyArray_DATA(thresholds_array);
        int *features = (int *)PyArray_DATA(features_array);
        int *children_left = (int *)PyArray_DATA(children_left_array);
        int *children_right = (int *)PyArray_DATA(children_right_array);

        // Create tree structure
        Tree tree = Tree(
            leaf_predictions,
            thresholds,
            features,
            children_left,
            children_right
        );

        /// Call the new sparse version that returns a hash map
        SparseSubsetMap* sparse_map = compute_values_interventional_sparse(tree,
                                                                            X_point, X_ref,
                                                                            row_x, col_x,
                                                                            max_order);

        // Extract entries from sparse map
        int num_entries;
        int* feature_lists;
        int* list_lengths = new int[sparse_map->get_size()];
        tfloat* values;
        
        sparse_map->extract_entries(&feature_lists, list_lengths, &values, num_entries);
        
        // Accumulate results into dictionary
        int feature_offset = 0;
        for (int i = 0; i < num_entries; i++) {
            int subset_size = list_lengths[i];
            
            // Create tuple of feature indices for this subset
            PyObject* feature_tuple = PyTuple_New(subset_size);
            for (int j = 0; j < subset_size; j++) {
                PyTuple_SetItem(feature_tuple, j, PyLong_FromLong(feature_lists[feature_offset + j]));
            }
            feature_offset += subset_size;
            
            // Check if key already exists and add to it, or create new entry
            PyObject* existing_value = PyDict_GetItem(result_dict, feature_tuple);
            if (existing_value != NULL) {
                // Key exists, add to existing value
                double current_val = PyFloat_AsDouble(existing_value);
                PyObject* new_value = PyFloat_FromDouble(current_val + values[i]);
                PyDict_SetItem(result_dict, feature_tuple, new_value);
                Py_DECREF(new_value);
            } else {
                // Key doesn't exist, create new entry
                PyObject* new_value = PyFloat_FromDouble(values[i]);
                PyDict_SetItem(result_dict, feature_tuple, new_value);
                Py_DECREF(new_value);
            }
            
            Py_DECREF(feature_tuple);
        }
        
        // Cleanup C++ allocations for this tree
        delete[] list_lengths;
        if (feature_lists) delete[] feature_lists;
        if (values) delete[] values;
        delete sparse_map;
        
        // Clean up arrays for this tree
        Py_XDECREF(children_left_array);
        Py_XDECREF(children_right_array);
        Py_XDECREF(features_array);
        Py_XDECREF(leaf_predictions_array);
        Py_XDECREF(thresholds_array);
    }
    
    // clean up the created python objects (we DECREF the arrays we created)
    Py_XDECREF(X_array);
    Py_XDECREF(X_reference_array);
    Py_XDECREF(out_contribs_array);

    /* Return dictionary mapping feature tuples to values */
    return result_dict;
}

// Old bitmask-based wrapper (32-feature limit, writes directly to values array)
static PyObject *_cext_interventional_iterative_bitmask(PyObject *self, PyObject *args, PyObject *kwargs)
{
    PyObject *leaf_predictions_obj;
    PyObject *thresholds_obj;
    PyObject *features_obj;
    PyObject *children_left_obj;
    PyObject *children_right_obj;
    PyObject *X_point_obj;
    PyObject *X_reference_obj;
    PyObject *out_contribs_obj;
    int max_order = -1;  // Default: no limit

    /* Parse the input tuple with optional keyword argument */
    static char *kwlist[] = {(char*)"X_point", (char*)"X_reference", (char*)"children_left", 
                             (char*)"children_right", (char*)"features", (char*)"thresholds", 
                             (char*)"leaf_predictions", (char*)"out_contribs", 
                             (char*)"max_order", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(
            args, kwargs, "OOOOOOOO|i", kwlist,
            &X_point_obj,
            &X_reference_obj,
            &children_left_obj,
            &children_right_obj,
            &features_obj,
            &thresholds_obj,
            &leaf_predictions_obj,
            &out_contribs_obj,
            &max_order))
        return NULL;
    
    // Parse arrays
    PyArrayObject *X_array = (PyArrayObject *)PyArray_FROM_OTF(X_point_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *X_reference_array = (PyArrayObject *)PyArray_FROM_OTF(X_reference_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *out_contribs_array = (PyArrayObject *)PyArray_FROM_OTF(out_contribs_obj, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY2);
    PyArrayObject *leaf_predictions_array = (PyArrayObject *)PyArray_FROM_OTF(leaf_predictions_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *thresholds_array = (PyArrayObject *)PyArray_FROM_OTF(thresholds_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *features_array = (PyArrayObject *)PyArray_FROM_OTF(features_obj, NPY_INT, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *children_left_array = (PyArrayObject *)PyArray_FROM_OTF(children_left_obj, NPY_INT, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *children_right_array = (PyArrayObject *)PyArray_FROM_OTF(children_right_obj, NPY_INT, NPY_ARRAY_IN_ARRAY);
    
    if (X_array == NULL || X_reference_array == NULL || out_contribs_array == NULL ||
        children_left_array == NULL || children_right_array == NULL ||
        features_array == NULL || leaf_predictions_array == NULL ||
        thresholds_array == NULL) {
        Py_XDECREF(X_array);
        Py_XDECREF(X_reference_array);
        PyArray_ResolveWritebackIfCopy(out_contribs_array);
        Py_XDECREF(out_contribs_array);
        Py_XDECREF(children_left_array);
        Py_XDECREF(children_right_array);
        Py_XDECREF(features_array);
        Py_XDECREF(leaf_predictions_array);
        Py_XDECREF(thresholds_array);
        return NULL;
    }
    
    tfloat *X_point = (tfloat *)PyArray_DATA(X_array);
    tfloat *X_ref = (tfloat *)PyArray_DATA(X_reference_array);
    tfloat *out_contribs = (tfloat *)PyArray_DATA(out_contribs_array);
    tfloat *leaf_predictions = (tfloat *)PyArray_DATA(leaf_predictions_array);
    tfloat *thresholds = (tfloat *)PyArray_DATA(thresholds_array);
    int *features = (int *)PyArray_DATA(features_array);
    int *children_left = (int *)PyArray_DATA(children_left_array);
    int *children_right = (int *)PyArray_DATA(children_right_array);
    
    const unsigned row_x = PyArray_DIM(X_reference_array, 0);
    const unsigned col_x = PyArray_DIM(X_reference_array, 1);
    
    if (col_x > 32) {
        PyErr_SetString(PyExc_ValueError, "interventional_iterative_bitmask only supports up to 32 features!");
        Py_XDECREF(X_array);
        Py_XDECREF(X_reference_array);
        PyArray_ResolveWritebackIfCopy(out_contribs_array);
        Py_XDECREF(out_contribs_array);
        Py_XDECREF(children_left_array);
        Py_XDECREF(children_right_array);
        Py_XDECREF(features_array);
        Py_XDECREF(leaf_predictions_array);
        Py_XDECREF(thresholds_array);
        return NULL;
    }
    
    // Create tree structure
    Tree tree = Tree(
        leaf_predictions,
        thresholds,
        features,
        children_left,
        children_right
    );
    
    // Call old bitmask-based function (writes directly to out_contribs array)
    compute_values_interventional_bitmask(tree, X_point, X_ref, out_contribs,
                                          row_x, col_x, max_order);
    
    // Clean up arrays
    Py_XDECREF(X_array);
    Py_XDECREF(X_reference_array);
    Py_XDECREF(children_left_array);
    Py_XDECREF(children_right_array);
    Py_XDECREF(features_array);
    Py_XDECREF(leaf_predictions_array);
    Py_XDECREF(thresholds_array);
    
    // Resolve writeback and return the array
    PyArray_ResolveWritebackIfCopy(out_contribs_array);
    Py_XDECREF(out_contribs_array);
    
    Py_RETURN_NONE;
}

// Direct array indexing wrapper (unlimited features, O(1) access)
static PyObject *_cext_interventional_iterative_direct(PyObject *self, PyObject *args, PyObject *kwargs)
{
    PyObject *leaf_predictions_obj;
    PyObject *thresholds_obj;
    PyObject *features_obj;
    PyObject *children_left_obj;
    PyObject *children_right_obj;
    PyObject *X_point_obj;
    PyObject *X_reference_obj;
    int max_order = -1;  // Default: no limit

    /* Parse the input tuple with optional keyword argument */
    static char *kwlist[] = {(char*)"X_point", (char*)"X_reference", (char*)"children_left", 
                             (char*)"children_right", (char*)"features", (char*)"thresholds", 
                             (char*)"leaf_predictions", (char*)"max_order", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(
            args, kwargs, "OOOOOOO|i", kwlist,
            &X_point_obj,
            &X_reference_obj,
            &children_left_obj,
            &children_right_obj,
            &features_obj,
            &thresholds_obj,
            &leaf_predictions_obj,
            &max_order))
        return NULL;
    
    // Parse arrays
    PyArrayObject *X_array = (PyArrayObject *)PyArray_FROM_OTF(X_point_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *X_reference_array = (PyArrayObject *)PyArray_FROM_OTF(X_reference_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *leaf_predictions_array = (PyArrayObject *)PyArray_FROM_OTF(leaf_predictions_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *thresholds_array = (PyArrayObject *)PyArray_FROM_OTF(thresholds_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *features_array = (PyArrayObject *)PyArray_FROM_OTF(features_obj, NPY_INT, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *children_left_array = (PyArrayObject *)PyArray_FROM_OTF(children_left_obj, NPY_INT, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *children_right_array = (PyArrayObject *)PyArray_FROM_OTF(children_right_obj, NPY_INT, NPY_ARRAY_IN_ARRAY);
    
    if (X_array == NULL || X_reference_array == NULL ||
        children_left_array == NULL || children_right_array == NULL ||
        features_array == NULL || leaf_predictions_array == NULL ||
        thresholds_array == NULL) {
        Py_XDECREF(X_array);
        Py_XDECREF(X_reference_array);
        Py_XDECREF(children_left_array);
        Py_XDECREF(children_right_array);
        Py_XDECREF(features_array);
        Py_XDECREF(leaf_predictions_array);
        Py_XDECREF(thresholds_array);
        return NULL;
    }
    
    tfloat *X_point = (tfloat *)PyArray_DATA(X_array);
    tfloat *X_ref = (tfloat *)PyArray_DATA(X_reference_array);
    tfloat *leaf_predictions = (tfloat *)PyArray_DATA(leaf_predictions_array);
    tfloat *thresholds = (tfloat *)PyArray_DATA(thresholds_array);
    int *features = (int *)PyArray_DATA(features_array);
    int *children_left = (int *)PyArray_DATA(children_left_array);
    int *children_right = (int *)PyArray_DATA(children_right_array);
    
    const unsigned row_x = PyArray_DIM(X_reference_array, 0);
    const unsigned col_x = PyArray_DIM(X_reference_array, 1);
    
    // Compute required array size using combinatorial formula
    int array_size = compute_values_array_size(col_x, max_order);
    
    // Allocate numpy array for output
    npy_intp dims[1] = {array_size};
    PyArrayObject *out_array = (PyArrayObject *)PyArray_ZEROS(1, dims, NPY_DOUBLE, 0);
    if (out_array == NULL) {
        Py_XDECREF(X_array);
        Py_XDECREF(X_reference_array);
        Py_XDECREF(children_left_array);
        Py_XDECREF(children_right_array);
        Py_XDECREF(features_array);
        Py_XDECREF(leaf_predictions_array);
        Py_XDECREF(thresholds_array);
        return NULL;
    }
    tfloat *out_values = (tfloat *)PyArray_DATA(out_array);
    
    // Create tree structure
    Tree tree = Tree(
        leaf_predictions,
        thresholds,
        features,
        children_left,
        children_right
    );
    
    // Call direct array indexing function
    compute_values_interventional_direct(tree, X_point, X_ref, out_values,
                                         row_x, col_x, max_order);
    
    // Clean up input arrays
    Py_XDECREF(X_array);
    Py_XDECREF(X_reference_array);
    Py_XDECREF(children_left_array);
    Py_XDECREF(children_right_array);
    Py_XDECREF(features_array);
    Py_XDECREF(leaf_predictions_array);
    Py_XDECREF(thresholds_array);
    
    // Return the values array (transfer ownership to Python)
    return PyArray_Return(out_array);
}
