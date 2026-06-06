cmake_minimum_required(VERSION 3.15)
project(litegs_cuda_project LANGUAGES CXX CUDA)

# 设置 C++ 和 CUDA 的标准版本
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(SOURCES
    binning.cu
    compact.cu
    cuda_errchk.cpp
    ext_cuda.cpp
    raster.cu
    transform.cu
)

add_library(litegs_fused MODULE ${SOURCES})
if(MSVC)
    # /MD or /MDd
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
    target_link_options(litegs_fused PRIVATE "/NODEFAULTLIB:LIBCMT")
endif()
set_target_properties(litegs_fused PROPERTIES SUFFIX ".pyd" PREFIX "")
target_compile_definitions(litegs_fused PRIVATE TORCH_EXTENSION_NAME=litegs_fused)
target_compile_options(litegs_fused PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
    --expt-relaxed-constexpr
>)

#link
find_package(Torch REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_library(TORCH_PYTHON_LIBRARY torch_python PATH "${TORCH_INSTALL_PREFIX}/lib")
target_link_libraries(litegs_fused "${TORCH_LIBRARIES}" "${TORCH_PYTHON_LIBRARY}" Python3::Python)
target_include_directories(litegs_fused PRIVATE ${TORCH_INCLUDE_DIRS} ${Python3_INCLUDE_DIRS})

# -G
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    message(STATUS "Debug mode enabled: adding CUDA debug options")
    target_compile_options(litegs_fused PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
        -G -lineinfo
    >)
endif()

# CUDA ARCH
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "52;60;61;70;75" CACHE STRING "CUDA architectures")
endif()