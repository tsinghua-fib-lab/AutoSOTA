find_package(Python3 COMPONENTS Interpreter REQUIRED)

# make sure to search torch with each invocation, as the paths will change
if(SKBUILD)
    message(STATUS "Building using SKBUILD: Resetting torch directories")
    unset(C10_CUDA_LIBRARY CACHE)
    unset(TORCH_LIBRARY CACHE)
    unset(kineto_LIBRARY CACHE)
endif()

execute_process(
        COMMAND "${Python3_EXECUTABLE}" "-c" "import torch;print(torch.utils.cmake_prefix_path)"
        OUTPUT_VARIABLE PT_CMAKE_PREFIX
        COMMAND_ECHO STDOUT
        OUTPUT_STRIP_TRAILING_WHITESPACE
        COMMAND_ERROR_IS_FATAL ANY
)

# cache CUDA_ARCHITECTURES, which seems to be reset by Torch
set(TMP_STORE_CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
set(TMP_STORE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")

set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH};${PT_CMAKE_PREFIX})
find_package(Torch REQUIRED CONFIG)

set(CMAKE_CUDA_ARCHITECTURES ${TMP_STORE_CUDA_ARCHITECTURES})
message(WARNING "Pytorch messes with global CMAKE_CUDA_FLAGS. Changing ${CMAKE_CUDA_FLAGS} back to ${TMP_STORE_CUDA_FLAGS}")
set(CMAKE_CUDA_FLAGS ${TMP_STORE_CUDA_FLAGS})

# this shared library isn't linked with the default `torch` target, but it is required because
# the symbol _ZN8pybind116detail11type_casterIN2at6TensorEvE4loadENS_6handleEb is needed.
cmake_path(REPLACE_FILENAME TORCH_LIBRARY libtorch_python.so OUTPUT_VARIABLE LIBTORCH_PYTHON)
target_link_libraries(torch INTERFACE ${LIBTORCH_PYTHON})
