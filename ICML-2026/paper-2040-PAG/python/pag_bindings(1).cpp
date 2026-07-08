#include "pag_index.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace {

using FloatArray =
    py::array_t<float, py::array::c_style | py::array::forcecast>;
using LabelArray =
    py::array_t<pag::Label, py::array::c_style | py::array::forcecast>;

void validate_vector_dim(const pag::Index &index, size_t dim,
                         const char *name) {
  if (!index.is_loaded()) {
    throw std::logic_error("PAG index is not loaded or built");
  }
  if (dim != index.dimension()) {
    throw std::invalid_argument(std::string(name) +
                                " dimension does not match the index");
  }
}

std::pair<size_t, size_t> query_shape(const py::buffer_info &info) {
  if (info.ndim == 1) {
    return {1, static_cast<size_t>(info.shape[0])};
  }
  if (info.ndim == 2) {
    return {static_cast<size_t>(info.shape[0]),
            static_cast<size_t>(info.shape[1])};
  }
  throw std::invalid_argument("queries must be a 1D or 2D float32 array");
}

py::tuple search_impl(const pag::Index &index, FloatArray queries,
                      const pag::SearchOptions &options) {
  if (options.top_k == 0) {
    throw std::invalid_argument("top_k must be positive");
  }

  py::buffer_info info = queries.request();
  const auto [query_count, dim] = query_shape(info);
  validate_vector_dim(index, dim, "query");

  const float *query_data = static_cast<const float *>(info.ptr);
  const size_t top_k = options.top_k;
  std::vector<int64_t> ids(query_count * top_k, -1);
  std::vector<float> distances(query_count * top_k,
                               std::numeric_limits<float>::infinity());

  std::vector<std::vector<pag::SearchResult>> batch_results;
  {
    py::gil_scoped_release release;
    batch_results = index.search_batch(query_data, query_count, options);
  }

  for (size_t i = 0; i < query_count; ++i) {
    const size_t result_count = std::min(batch_results[i].size(), top_k);
    for (size_t j = 0; j < result_count; ++j) {
      ids[i * top_k + j] = static_cast<int64_t>(batch_results[i][j].id);
      distances[i * top_k + j] = batch_results[i][j].distance;
    }
  }

  py::array_t<int64_t> id_array({query_count, top_k});
  py::array_t<float> distance_array({query_count, top_k});
  std::memcpy(id_array.mutable_data(), ids.data(),
              ids.size() * sizeof(int64_t));
  std::memcpy(distance_array.mutable_data(), distances.data(),
              distances.size() * sizeof(float));
  return py::make_tuple(id_array, distance_array);
}

void build_impl(pag::Index &index, FloatArray vectors,
                const pag::BuildOptions &options) {
  py::buffer_info info = vectors.request();
  if (info.ndim != 2) {
    throw std::invalid_argument("vectors must be a 2D float32 array");
  }
  const size_t count = static_cast<size_t>(info.shape[0]);
  const size_t dim = static_cast<size_t>(info.shape[1]);
  const float *data = static_cast<const float *>(info.ptr);
  py::gil_scoped_release release;
  index.build(data, count, dim, options);
}

pag::Label add_impl(pag::Index &index, FloatArray vector) {
  py::buffer_info info = vector.request();
  if (info.ndim != 1) {
    throw std::invalid_argument("vector must be a 1D float32 array");
  }
  validate_vector_dim(index, static_cast<size_t>(info.shape[0]), "vector");
  const float *data = static_cast<const float *>(info.ptr);
  py::gil_scoped_release release;
  return index.add(data);
}

void insert_impl(pag::Index &index, FloatArray vector, pag::Label label) {
  py::buffer_info info = vector.request();
  if (info.ndim != 1) {
    throw std::invalid_argument("vector must be a 1D float32 array");
  }
  validate_vector_dim(index, static_cast<size_t>(info.shape[0]), "vector");
  const float *data = static_cast<const float *>(info.ptr);
  py::gil_scoped_release release;
  index.insert(data, label);
}

py::array_t<pag::Label> add_batch_impl(pag::Index &index, FloatArray vectors) {
  py::buffer_info info = vectors.request();
  if (info.ndim != 2) {
    throw std::invalid_argument("vectors must be a 2D float32 array");
  }
  const size_t count = static_cast<size_t>(info.shape[0]);
  const size_t dim = static_cast<size_t>(info.shape[1]);
  validate_vector_dim(index, dim, "vectors");
  const float *data = static_cast<const float *>(info.ptr);

  std::vector<pag::Label> labels;
  {
    py::gil_scoped_release release;
    labels = index.add_batch(data, count);
  }

  py::array_t<pag::Label> label_array({count});
  std::memcpy(label_array.mutable_data(), labels.data(),
              labels.size() * sizeof(pag::Label));
  return label_array;
}

void insert_batch_impl(pag::Index &index, FloatArray vectors,
                       LabelArray labels) {
  py::buffer_info vector_info = vectors.request();
  if (vector_info.ndim != 2) {
    throw std::invalid_argument("vectors must be a 2D float32 array");
  }
  const size_t count = static_cast<size_t>(vector_info.shape[0]);
  const size_t dim = static_cast<size_t>(vector_info.shape[1]);
  validate_vector_dim(index, dim, "vectors");

  py::buffer_info label_info = labels.request();
  if (label_info.ndim != 1 ||
      static_cast<size_t>(label_info.shape[0]) != count) {
    throw std::invalid_argument("labels must be a 1D array matching vectors");
  }

  const float *data = static_cast<const float *>(vector_info.ptr);
  const pag::Label *label_data =
      static_cast<const pag::Label *>(label_info.ptr);
  py::gil_scoped_release release;
  index.insert_batch(data, label_data, count);
}

void load_impl(pag::Index &index, const pag::LoadOptions &options) {
  py::gil_scoped_release release;
  index.load(options);
}

void save_impl(pag::Index &index) {
  py::gil_scoped_release release;
  index.save();
}

} // namespace

PYBIND11_MODULE(_pag, m) {
  m.doc() = "Projection-Augmented Graph approximate nearest neighbor search";

  py::enum_<pag::Metric>(m, "Metric")
      .value("L2", pag::Metric::L2)
      .value("Cosine", pag::Metric::Cosine)
      .value("MaximumInnerProduct", pag::Metric::MaximumInnerProduct);

  py::enum_<pag::IndexMode>(m, "IndexMode")
      .value("Static", pag::IndexMode::Static)
      .value("Online", pag::IndexMode::Online);

  py::class_<pag::BuildOptions>(m, "BuildOptions")
      .def(py::init<>())
      .def_readwrite("index_path", &pag::BuildOptions::index_path)
      .def_readwrite("metric", &pag::BuildOptions::metric)
      .def_readwrite("mode", &pag::BuildOptions::mode)
      .def_readwrite("max_search_k", &pag::BuildOptions::max_search_k)
      .def_readwrite("max_elements", &pag::BuildOptions::max_elements)
      .def_readwrite("ef_construction", &pag::BuildOptions::ef_construction)
      .def_readwrite("target_degree", &pag::BuildOptions::target_degree)
      .def_readwrite("projection_levels",
                     &pag::BuildOptions::projection_levels);

  py::class_<pag::LoadOptions>(m, "LoadOptions")
      .def(py::init<>())
      .def_readwrite("index_path", &pag::LoadOptions::index_path)
      .def_readwrite("metric", &pag::LoadOptions::metric)
      .def_readwrite("max_elements", &pag::LoadOptions::max_elements);

  py::class_<pag::SearchOptions>(m, "SearchOptions")
      .def(py::init<>())
      .def_readwrite("top_k", &pag::SearchOptions::top_k)
      .def_readwrite("ef_search", &pag::SearchOptions::ef_search);

  py::class_<pag::Index>(m, "Index")
      .def(py::init<>())
      .def("build", &build_impl, py::arg("vectors"), py::arg("options"))
      .def("load", &load_impl, py::arg("options"))
      .def("save", &save_impl)
      .def("search",
           [](const pag::Index &index, FloatArray queries, size_t top_k,
              size_t ef_search) {
             pag::SearchOptions options;
             options.top_k = top_k;
             options.ef_search = ef_search;
             return search_impl(index, queries, options);
           },
           py::arg("queries"), py::arg("top_k") = 10,
           py::arg("ef_search") = 0)
      .def("search_with_options", &search_impl, py::arg("queries"),
           py::arg("options"))
      .def("add", &add_impl, py::arg("vector"))
      .def("insert", &insert_impl, py::arg("vector"), py::arg("label"))
      .def("add_batch", &add_batch_impl, py::arg("vectors"))
      .def("insert_batch", &insert_batch_impl, py::arg("vectors"),
           py::arg("labels"))
      .def_property_readonly("is_loaded", &pag::Index::is_loaded)
      .def_property_readonly("dimension", &pag::Index::dimension)
      .def_property_readonly("metric", &pag::Index::metric);
}
