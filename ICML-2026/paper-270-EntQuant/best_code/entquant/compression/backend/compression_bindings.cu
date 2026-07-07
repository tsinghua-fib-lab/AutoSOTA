// compression_bindings.cu
#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "compression_backend.cuh"

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "CUDA-accelerated compression backend using nvcomp ANS";

    py::class_<CompressionBackend>(m, "CompressionBackend")
        .def(py::init<int>(), 
             py::arg("chunk_size") = 1 << 18)
        
        .def("compress", 
             &CompressionBackend::compress,
             py::arg("input"),
             "Compress input tensor, returns compressed data tensor")
        
        .def("decompress",
             &CompressionBackend::decompress,
             py::arg("key"),
             py::arg("compressed"),
             py::arg("output_buffer"),
             "Decompress with lazy config caching (first call per key reads header)")
        
        .def("clear_cache",
             py::overload_cast<>(&CompressionBackend::clear_cache),
             "Release all device resources and clear cached configs")
        
        .def("clear_cache",
             py::overload_cast<int>(&CompressionBackend::clear_cache),
             py::arg("device_id"),
             "Release resources for a specific device")
        
        .def("synchronize",
             py::overload_cast<>(&CompressionBackend::synchronize),
             "Block CPU until all device streams complete")
        
        .def("synchronize",
             py::overload_cast<int>(&CompressionBackend::synchronize),
             py::arg("device_id"),
             "Block CPU until specific device stream completes")
        
        .def_property_readonly("chunk_size",
             &CompressionBackend::get_chunk_size);
}