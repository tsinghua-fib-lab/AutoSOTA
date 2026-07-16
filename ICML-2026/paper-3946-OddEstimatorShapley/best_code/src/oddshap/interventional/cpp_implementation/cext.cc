#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <algorithm>
#include <cstdint>
#include <exception>
#include <limits>
#include <string>
#include <vector>

#include <numpy/arrayobject.h>

#include "conversion.hpp"
#include "interactions.hpp"

using interventional_cpp::FeatureIndex;
using interventional_cpp::InteractionKey;
using interventional_cpp::InteractionMap;

static bool convert_python_index(PyObject* obj, FeatureIndex& out, const char* context) {
    unsigned long long value = PyLong_AsUnsignedLongLong(obj);
    if (value == static_cast<unsigned long long>(-1) && PyErr_Occurred()) {
        return false;
    }
    if (value > std::numeric_limits<FeatureIndex>::max()) {
        PyErr_Format(PyExc_OverflowError, "%s is too large for FeatureIndex", context);
        return false;
    }
    out = static_cast<FeatureIndex>(value);
    return true;
}

static bool convert_iterable_to_vector(PyObject* iterable, std::vector<FeatureIndex>& out, const char* name) {
    PyObject* iterator = PyObject_GetIter(iterable);
    if (iterator == nullptr) {
        PyErr_Format(PyExc_TypeError, "Argument '%s' must be an iterable of non-negative integers", name);
        return false;
    }

    PyObject* item = nullptr;
    while ((item = PyIter_Next(iterator)) != nullptr) {
        FeatureIndex converted_value = 0;
        const bool success = convert_python_index(item, converted_value, name);
        Py_DECREF(item);
        if (!success) {
            Py_DECREF(iterator);
            return false;
        }
        out.push_back(converted_value);
    }
    Py_DECREF(iterator);

    if (PyErr_Occurred()) {
        return false;
    }

    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return true;
}

static bool convert_A_B_NB_interval_to_tuple(
    PyObject* iteratable,
    std::vector<interventional_cpp::IntervalTuple>& out
) {
    PyObject* iterator = PyObject_GetIter(iteratable);
    if (iterator == nullptr) {
        PyErr_SetString(PyExc_TypeError, "A_B_NB_Intervals must be an iterable of tuples");
        return false;
    }
    PyObject* item = nullptr;
    while ((item = PyIter_Next(iterator)) != nullptr) {
        // Each item is a list [A, B, NB, const_values]
        PyObject* iterator2 = PyObject_GetIter(item);
        Py_DECREF(item);
        if (iterator2 == nullptr) {
            PyErr_SetString(PyExc_TypeError, "Each interval entry must be an iterable");
            Py_DECREF(iterator);
            return false;
        }
        PyObject* sub_item = nullptr;
        std::vector<FeatureIndex> A, B, NB;
        double interval_value = 0.0;
        int index = 0;
        while ((sub_item = PyIter_Next(iterator2)) != nullptr) {
            if (index == 0) {
                if (!convert_iterable_to_vector(sub_item, A, "A in interval")) {
                    Py_DECREF(sub_item);
                    Py_DECREF(iterator2);
                    Py_DECREF(iterator);
                    return false;
                }
            } else if (index == 1) {
                if (!convert_iterable_to_vector(sub_item, B, "B in interval")) {
                    Py_DECREF(sub_item);
                    Py_DECREF(iterator2);
                    Py_DECREF(iterator);
                    return false;
                }
            } else if (index == 2) {
                if (!convert_iterable_to_vector(sub_item, NB, "NB in interval")) {
                    Py_DECREF(sub_item);
                    Py_DECREF(iterator2);
                    Py_DECREF(iterator);
                    return false;
                }
            } else if (index == 3) {
                interval_value = PyFloat_AsDouble(sub_item);
                if (PyErr_Occurred()) {
                    Py_DECREF(sub_item);
                    Py_DECREF(iterator2);
                    Py_DECREF(iterator);
                    return false;
                }
            } else {
                PyErr_SetString(PyExc_ValueError, "Each interval entry must have exactly four elements");
                Py_DECREF(sub_item);
                Py_DECREF(iterator2);
                Py_DECREF(iterator);
                return false;
            }
            Py_DECREF(sub_item);
            ++index;
        }
        Py_DECREF(iterator2);
        if (index != 4) {
            PyErr_SetString(PyExc_ValueError, "Each interval entry must have exactly four elements");
            Py_DECREF(iterator);
            return false;
        }
        out.emplace_back(A, B, NB, interval_value);
    }
    Py_DECREF(iterator);

    if (PyErr_Occurred()) {
        return false;
    }

    return true;

}

static bool convert_python_interaction_dict(PyObject* mapping, InteractionMap& out) {
    if (!PyDict_Check(mapping)) {
        PyErr_SetString(PyExc_TypeError, "interaction_to_values must be a dict");
        return false;
    }

    PyObject* key = nullptr;
    PyObject* value = nullptr;
    Py_ssize_t pos = 0;

    while (PyDict_Next(mapping, &pos, &key, &value)) {
        if (!PyTuple_Check(key)) {
            PyErr_SetString(PyExc_TypeError, "interaction_to_values keys must be tuples of feature indices");
            return false;
        }

        const Py_ssize_t tuple_len = PyTuple_GET_SIZE(key);
        InteractionKey interaction_key;
        interaction_key.reserve(static_cast<std::size_t>(tuple_len));
        for (Py_ssize_t i = 0; i < tuple_len; ++i) {
            PyObject* element = PyTuple_GET_ITEM(key, i);
            FeatureIndex idx = 0;
            if (!convert_python_index(element, idx, "interaction key element")) {
                return false;
            }
            interaction_key.push_back(idx);
        }
        std::sort(interaction_key.begin(), interaction_key.end());
        interaction_key.erase(std::unique(interaction_key.begin(), interaction_key.end()), interaction_key.end());

        const double numeric_value = PyFloat_AsDouble(value);
        if (PyErr_Occurred()) {
            return false;
        }

        const auto insert_result = out.emplace(interaction_key, numeric_value);
        if (!insert_result.second) {
            insert_result.first->second = numeric_value;
        }
    }

    return true;
}

static bool populate_python_dict_from_map(const InteractionMap& source, PyObject* target) {
    if (!PyDict_Check(target)) {
        PyErr_SetString(PyExc_TypeError, "target must be a dict");
        return false;
    }

    PyDict_Clear(target);
    for (const auto& entry : source) {
        const InteractionKey& key = entry.first;
        PyObject* key_tuple = PyTuple_New(static_cast<Py_ssize_t>(key.size()));
        if (key_tuple == nullptr) {
            return false;
        }

        for (Py_ssize_t i = 0; i < static_cast<Py_ssize_t>(key.size()); ++i) {
            PyObject* index_obj = PyLong_FromUnsignedLong(static_cast<unsigned long>(key[static_cast<std::size_t>(i)]));
            if (index_obj == nullptr) {
                Py_DECREF(key_tuple);
                return false;
            }
            PyTuple_SET_ITEM(key_tuple, i, index_obj);  // Steals reference.
        }

        PyObject* value_obj = PyFloat_FromDouble(entry.second);
        if (value_obj == nullptr) {
            Py_DECREF(key_tuple);
            return false;
        }

        if (PyDict_SetItem(target, key_tuple, value_obj) < 0) {
            Py_DECREF(key_tuple);
            Py_DECREF(value_obj);
            return false;
        }

        Py_DECREF(key_tuple);
        Py_DECREF(value_obj);
    }

    return true;
}

static PyObject* boundary_interventional_update(PyObject* /*self*/, PyObject* args, PyObject* kwargs) {
    PyObject* interaction_obj = nullptr;
    double const_coalition = 0.0;
    PyObject* A_obj = nullptr;
    PyObject* B_obj = nullptr;
    PyObject* NB_obj = nullptr;
    int max_order = 0;
    const char* weight_func_cstr = nullptr;

    static const char* kwlist[] = {
        "interaction_to_values",
        "const_coalition",
        "A",
        "B",
        "NB",
        "max_order",
        "weight_func",
        nullptr
    };

    if (!PyArg_ParseTupleAndKeywords(
            args,
            kwargs,
            "OdOOOis",
            const_cast<char**>(kwlist),
            &interaction_obj,
            &const_coalition,
            &A_obj,
            &B_obj,
            &NB_obj,
            &max_order,
            &weight_func_cstr)) {
        return nullptr;
    }

    if (max_order < 0) {
        PyErr_SetString(PyExc_ValueError, "max_order must be non-negative");
        return nullptr;
    }

    InteractionMap interaction_map;
    interaction_map.reserve(static_cast<std::size_t>(PyDict_Size(interaction_obj)));
    if (!convert_python_interaction_dict(interaction_obj, interaction_map)) {
        return nullptr;
    }

    std::vector<FeatureIndex> set_A;
    if (!convert_iterable_to_vector(A_obj, set_A, "A")) {
        return nullptr;
    }

    std::vector<FeatureIndex> set_B;
    if (!convert_iterable_to_vector(B_obj, set_B, "B")) {
        return nullptr;
    }

    std::vector<FeatureIndex> set_NB;
    if (!convert_iterable_to_vector(NB_obj, set_NB, "NB")) {
        return nullptr;
    }

    std::string weight_func;
    if (weight_func_cstr != nullptr) {
        weight_func = weight_func_cstr;
    }

    interventional_cpp::update_interaction_values(
        interaction_map,
        const_coalition,
        set_A,
        set_B,
        set_NB,
        max_order,
        weight_func
    );

    if (!populate_python_dict_from_map(interaction_map, interaction_obj)) {
        return nullptr;
    }

    Py_INCREF(interaction_obj);
    return interaction_obj;
}

static PyObject* boundary_interventional_update_batch(PyObject* /*self*/, PyObject* args, PyObject* kwargs) {
    PyObject* interaction_obj = nullptr;
    PyObject* intervals_obj = nullptr;
    int max_order = 0;
    const char* weight_func_cstr = nullptr;

    static const char* kwlist[] = {
        "interaction_to_values",
        "A_B_NB_Intervals",
        "max_order",
        "weight_func",
        nullptr
    };

    if (!PyArg_ParseTupleAndKeywords(
            args,
            kwargs,
            "OOis",
            const_cast<char**>(kwlist),
            &interaction_obj,
            &intervals_obj,
            &max_order,
            &weight_func_cstr)) {
        return nullptr;
    }

    if (max_order < 0) {
        PyErr_SetString(PyExc_ValueError, "max_order must be non-negative");
        return nullptr;
    }

    InteractionMap interaction_map;
    interaction_map.reserve(static_cast<std::size_t>(PyDict_Size(interaction_obj)));
    if (!convert_python_interaction_dict(interaction_obj, interaction_map)) {
        return nullptr;
    }

    std::string weight_func;
    if (weight_func_cstr != nullptr) {
        weight_func = weight_func_cstr;
    }

    PyObject* iterator = PyObject_GetIter(intervals_obj);
    if (iterator == nullptr) {
        PyErr_SetString(PyExc_TypeError, "A_B_NB_Intervals must be an iterable of tuples");
        return nullptr;
    }

    std::vector<interventional_cpp::IntervalTuple> intervals;
    if (!convert_A_B_NB_interval_to_tuple(intervals_obj, intervals)) {
        Py_DECREF(iterator);
        return nullptr;
    }
    Py_DECREF(iterator);

    for (const auto& interval : intervals) {
        const std::vector<FeatureIndex>& A = std::get<0>(interval);
        const std::vector<FeatureIndex>& B = std::get<1>(interval);
        const std::vector<FeatureIndex>& NB = std::get<2>(interval);
        const double const_coalition =  std::get<3>(interval);

        interventional_cpp::update_interaction_values(
            interaction_map,
            const_coalition,
            A,
            B,
            NB,
            max_order,
            weight_func
        );
    }

    if (!populate_python_dict_from_map(interaction_map, interaction_obj)) {
        return nullptr;
    }

    Py_INCREF(interaction_obj);
    return interaction_obj;
}


static PyObject* convert_forest_to_matrix(PyObject* /*self*/, PyObject* args, PyObject* kwargs)
{
    PyObject* children_left_obj = nullptr;
    PyObject* children_right_obj = nullptr;
    PyObject* features_obj = nullptr;
    PyObject* values_obj = nullptr;
    Py_ssize_t feature_hint_param = 0;

    static const char *kwlist[] = {
        "children_left",
        "children_right",
        "features",
        "values",
        "n_features",
        nullptr};

    if (!PyArg_ParseTupleAndKeywords(
            args,
            kwargs,
            "OOOO|n",
            const_cast<char **>(kwlist),
            &children_left_obj,
            &children_right_obj,
            &features_obj,
            &values_obj,
            &feature_hint_param))
    {
        return nullptr;
    }

    PyArrayObject* children_left_array = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(children_left_obj, NPY_INT32, NPY_ARRAY_IN_ARRAY | NPY_ARRAY_FORCECAST));
    PyArrayObject* children_right_array = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(children_right_obj, NPY_INT32, NPY_ARRAY_IN_ARRAY | NPY_ARRAY_FORCECAST));
    PyArrayObject *features_array = reinterpret_cast<PyArrayObject *>(
        PyArray_FROM_OTF(features_obj, NPY_INT32, NPY_ARRAY_IN_ARRAY | NPY_ARRAY_FORCECAST));
    PyArrayObject* values_array = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(values_obj, NPY_FLOAT64, NPY_ARRAY_IN_ARRAY));

    if (children_left_array == nullptr || children_right_array == nullptr ||
        features_array == nullptr || values_array == nullptr)
    {
        Py_XDECREF(children_left_array);
        Py_XDECREF(children_right_array);
        Py_XDECREF(features_array);
        Py_XDECREF(values_array);
        return nullptr;
    }

    PyArrayObject* children_left_contig = PyArray_GETCONTIGUOUS(children_left_array);
    PyArrayObject* children_right_contig = PyArray_GETCONTIGUOUS(children_right_array);
    PyArrayObject* features_contig = PyArray_GETCONTIGUOUS(features_array);
    PyArrayObject* values_contig = PyArray_GETCONTIGUOUS(values_array);

    Py_DECREF(children_left_array);
    Py_DECREF(children_right_array);
    Py_DECREF(features_array);
    Py_DECREF(values_array);

    if (children_left_contig == nullptr || children_right_contig == nullptr ||
        features_contig == nullptr || values_contig == nullptr)
    {
        Py_XDECREF(children_left_contig);
        Py_XDECREF(children_right_contig);
        Py_XDECREF(features_contig);
        Py_XDECREF(values_contig);
        return nullptr;
    }

    children_left_array = children_left_contig;
    children_right_array = children_right_contig;
    features_array = features_contig;
    values_array = values_contig;

    const int ndim = PyArray_NDIM(children_left_array);
    if (PyArray_NDIM(children_right_array) != ndim ||
        PyArray_NDIM(features_array) != ndim ||
        PyArray_NDIM(values_array) != ndim)
    {
        PyErr_SetString(PyExc_ValueError, "All tree arrays must have matching dimensionality");
        Py_DECREF(children_left_array);
        Py_DECREF(children_right_array);
        Py_DECREF(features_array);
        Py_DECREF(values_array);
        return nullptr;
    }

    if (ndim != 1 && ndim != 2)
    {
        PyErr_SetString(PyExc_ValueError, "Tree arrays must be one- or two-dimensional");
        Py_DECREF(children_left_array);
        Py_DECREF(children_right_array);
        Py_DECREF(features_array);
        Py_DECREF(values_array);
        return nullptr;
    }

    for (int axis = 0; axis < ndim; ++axis)
    {
        if (PyArray_DIM(children_right_array, axis) != PyArray_DIM(children_left_array, axis) ||
            PyArray_DIM(features_array, axis) != PyArray_DIM(children_left_array, axis) ||
            PyArray_DIM(values_array, axis) != PyArray_DIM(children_left_array, axis))
        {
            PyErr_SetString(PyExc_ValueError, "Tree arrays must share the same shape");
            Py_DECREF(children_left_array);
            Py_DECREF(children_right_array);
            Py_DECREF(features_array);
            Py_DECREF(values_array);
            return nullptr;
        }
    }

    std::vector<conversion::TreeArrays> tree_arrays;
    std::size_t feature_count_hint = feature_hint_param < 0 ? 0 : static_cast<std::size_t>(feature_hint_param);
    if (ndim == 1)
    {
        const npy_intp node_count = PyArray_DIM(children_left_array, 0);
        if (node_count <= 0)
        {
            PyErr_SetString(PyExc_ValueError, "Tree arrays must contain at least one node");
            Py_DECREF(children_left_array);
            Py_DECREF(children_right_array);
            Py_DECREF(features_array);
            Py_DECREF(values_array);
            return nullptr;
        }

        tree_arrays.push_back({
            static_cast<const int*>(PyArray_DATA(children_left_array)),
            static_cast<const int*>(PyArray_DATA(children_right_array)),
            static_cast<const int*>(PyArray_DATA(features_array)),
            static_cast<const double*>(PyArray_DATA(values_array)),
            static_cast<std::size_t>(node_count)
        });
        //feature_count_hint = std::max(feature_count_hint, static_cast<std::size_t>(node_count));
    }
    else
    {
        const npy_intp tree_count = PyArray_DIM(children_left_array, 0);
        const npy_intp node_count = PyArray_DIM(children_left_array, 1);

        if (tree_count <= 0 || node_count <= 0)
        {
            PyErr_SetString(PyExc_ValueError, "Tree arrays must contain at least one tree and one node");
            Py_DECREF(children_left_array);
            Py_DECREF(children_right_array);
            Py_DECREF(features_array);
            Py_DECREF(values_array);
            return nullptr;
        }

        tree_arrays.reserve(static_cast<std::size_t>(tree_count));
        //feature_count_hint = std::max(feature_count_hint, static_cast<std::size_t>(node_count));

        for (npy_intp tree_idx = 0; tree_idx < tree_count; ++tree_idx)
        {
            const int* left_ptr = reinterpret_cast<const int*>(PyArray_GETPTR2(children_left_array, tree_idx, 0));
            const int* right_ptr = reinterpret_cast<const int*>(PyArray_GETPTR2(children_right_array, tree_idx, 0));
            const int* feature_ptr = reinterpret_cast<const int*>(PyArray_GETPTR2(features_array, tree_idx, 0));
            const double* value_ptr = reinterpret_cast<const double*>(PyArray_GETPTR2(values_array, tree_idx, 0));

            if (left_ptr == nullptr || right_ptr == nullptr || feature_ptr == nullptr || value_ptr == nullptr)
            {
                PyErr_SetString(PyExc_RuntimeError, "Failed to access tree row data");
                Py_DECREF(children_left_array);
                Py_DECREF(children_right_array);
                Py_DECREF(features_array);
                Py_DECREF(values_array);
                return nullptr;
            }

            tree_arrays.push_back({
                left_ptr,
                right_ptr,
                feature_ptr,
                value_ptr,
                static_cast<std::size_t>(node_count)
            });
        }
    }

    conversion::LeafMatrix leaf_matrix;
    try
    {
        leaf_matrix = conversion::forest_to_leaf_matrix(
            tree_arrays,
            feature_count_hint);
    }
    catch (const std::out_of_range& error)
    {
        Py_DECREF(children_left_array);
        Py_DECREF(children_right_array);
        Py_DECREF(features_array);
        Py_DECREF(values_array);
        PyErr_SetString(PyExc_IndexError, error.what());
        return nullptr;
    }
    catch (const std::exception& error)
    {
        Py_DECREF(children_left_array);
        Py_DECREF(children_right_array);
        Py_DECREF(features_array);
        Py_DECREF(values_array);
        PyErr_SetString(PyExc_ValueError, error.what());
        return nullptr;
    }

    if (leaf_matrix.feature_paths.size() != leaf_matrix.leaf_values.size())
    {
        Py_DECREF(children_left_array);
        Py_DECREF(children_right_array);
        Py_DECREF(features_array);
        Py_DECREF(values_array);
        PyErr_SetString(PyExc_RuntimeError, "Internal error: mismatch between paths and leaf values");
        return nullptr;
    }

    if (leaf_matrix.feature_paths.empty())
    {
        Py_DECREF(children_left_array);
        Py_DECREF(children_right_array);
        Py_DECREF(features_array);
        Py_DECREF(values_array);
        PyErr_SetString(PyExc_ValueError, "Tree traversal produced no leaves");
        return nullptr;
    }

    const std::size_t n_features = leaf_matrix.feature_paths.front().size();
    for (const auto& path : leaf_matrix.feature_paths)
    {
        if (path.size() != n_features)
        {
            Py_DECREF(children_left_array);
            Py_DECREF(children_right_array);
            Py_DECREF(features_array);
            Py_DECREF(values_array);
            PyErr_SetString(PyExc_RuntimeError, "Internal error: inconsistent feature path length");
            return nullptr;
        }
    }

    PyObject* result = PyList_New(static_cast<Py_ssize_t>(leaf_matrix.feature_paths.size()));
    if (result == nullptr)
    {
        Py_DECREF(children_left_array);
        Py_DECREF(children_right_array);
        Py_DECREF(features_array);
        Py_DECREF(values_array);
        return nullptr;
    }


    for (std::size_t row_idx = 0; row_idx < leaf_matrix.feature_paths.size(); ++row_idx)
    {
        PyObject* row = PyList_New(static_cast<Py_ssize_t>(n_features + 1));
        if (row == nullptr)
        {
            Py_DECREF(result);
            Py_DECREF(children_left_array);
            Py_DECREF(children_right_array);
            Py_DECREF(features_array);
            Py_DECREF(values_array);
            return nullptr;
        }
        
        for (std::size_t col = 0; col < n_features; ++col)
        {
            PyObject* value_obj = PyLong_FromLong(static_cast<long>(leaf_matrix.feature_paths[row_idx][col]));
            if (value_obj == nullptr)
            {
                Py_DECREF(row);
                Py_DECREF(result);
                Py_DECREF(children_left_array);
                Py_DECREF(children_right_array);
                Py_DECREF(features_array);
                Py_DECREF(values_array);
                return nullptr;
            }
            PyList_SET_ITEM(row, static_cast<Py_ssize_t>(col), value_obj);
        }

        PyObject* leaf_value_obj = PyFloat_FromDouble(leaf_matrix.leaf_values[row_idx]);
        if (leaf_value_obj == nullptr)
        {
            Py_DECREF(row);
            Py_DECREF(result);
            Py_DECREF(children_left_array);
            Py_DECREF(children_right_array);
            Py_DECREF(features_array);
            Py_DECREF(values_array);
            return nullptr;
        }
        PyList_SET_ITEM(row, static_cast<Py_ssize_t>(n_features), leaf_value_obj);
        PyList_SET_ITEM(result, static_cast<Py_ssize_t>(row_idx), row);
    }

    Py_DECREF(children_left_array);
    Py_DECREF(children_right_array);
    Py_DECREF(features_array);
    Py_DECREF(values_array);

    return result;
}

static PyMethodDef module_methods[] = {
    {"interventional_update",
     reinterpret_cast<PyCFunction>(boundary_interventional_update),
     METH_VARARGS | METH_KEYWORDS,
     "Convert Python data structures to C++ containers and invoke the interventional updater."},
    {"interventional_update_batch",
     reinterpret_cast<PyCFunction>(boundary_interventional_update_batch),
     METH_VARARGS | METH_KEYWORDS,
     "Batch update over precomputed (A, B, NB, value) intervals."},
    {"convert_forest_to_matrix",
     reinterpret_cast<PyCFunction>(convert_forest_to_matrix),
     METH_VARARGS | METH_KEYWORDS,
     "Construct a Matrix representation of the Leafs of the tree."},
    {nullptr, nullptr, 0, nullptr}};

static struct PyModuleDef module_definition = {
    PyModuleDef_HEAD_INIT,
    "cpp_implementation",
    "Boundary layer that marshals Python objects into fast C++ data structures for interventional updates.",
    -1,
    module_methods
};

PyMODINIT_FUNC PyInit_cpp_implementation(void) {
    PyObject* module = PyModule_Create(&module_definition);
    if (module == nullptr) {
        return nullptr;
    }

    import_array();
    if (PyErr_Occurred()) {
        Py_DECREF(module);
        return nullptr;
    }

    return module;
}
