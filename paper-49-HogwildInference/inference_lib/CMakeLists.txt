cmake_minimum_required(VERSION 3.22)
project(HogwildCache CUDA CXX)
include(FetchContent)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES "80;86;89")

find_package(CUDAToolkit REQUIRED)

option(HOGDILD_TORCH_BINDINGS "Build pytorch bindings" OFF)
option(HOGDILD_BENCHMARKS "Build the benchmark program" ON)
option(HOGWILD_TESTS "Build the test program" ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "native")
endif()


add_subdirectory(src/tpl)

option(HOGDILD_ENABLE_FLOAT "build with float32 kernel support" ON)
option(HOGDILD_ENABLE_HALF "build with float16 kernel support" ON)
option(HOGDILD_ENABLE_BFLOAT "build with bfloat16 kernel support" ON)

add_library(hogwild-kernels STATIC ${HOGWILD_KERNEL_INSTANTIATIONS})
target_include_directories(hogwild-kernels PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_options(hogwild-kernels PRIVATE -expt-relaxed-constexpr -lineinfo)

if(HOGDILD_ENABLE_FLOAT)
    target_compile_definitions(hogwild-kernels PUBLIC HOGDILD_ENABLE_FLOAT=1)
endif ()

if(HOGDILD_ENABLE_HALF)
    target_compile_definitions(hogwild-kernels PUBLIC HOGDILD_ENABLE_HALF=1)
endif ()

if(HOGDILD_ENABLE_BFLOAT)
    target_compile_definitions(hogwild-kernels PUBLIC HOGDILD_ENABLE_BFLOAT=1)
endif ()


if(HOGDILD_BENCHMARKS OR HOGWILD_TESTS)
    FetchContent_Declare(
            cli11
            QUIET
            GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
            GIT_TAG v2.4.2
    )
    FetchContent_MakeAvailable(cli11)
endif ()

if(HOGDILD_TORCH_BINDINGS)
    include(cmake/pytorch.cmake)
    find_package(Python3 COMPONENTS Development.SABIModule REQUIRED)
    Python3_add_library(hogatt MODULE src/binding.cu WITH_SOABI USE_SABI 3.10)
    if(SKBUILD)
        install(TARGETS hogatt DESTINATION ${SKBUILD_PLATLIB_DIR}/hogwild)
    endif ()
    target_link_libraries(hogatt PRIVATE torch)
    target_compile_options(hogatt PUBLIC -expt-relaxed-constexpr -lineinfo)
    target_include_directories(hogatt PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
endif ()

if(HOGDILD_BENCHMARKS)
    add_executable(bench src/benchmark.cpp)
    target_include_directories(bench PUBLIC src)
    target_link_libraries(bench PRIVATE CLI11::CLI11 CUDA::cudart hogwild-kernels)
endif ()

if(HOGWILD_TESTS)
    add_executable(test src/test.cpp)
    target_include_directories(test PUBLIC src)
    target_link_libraries(test PRIVATE CLI11::CLI11 CUDA::cudart hogwild-kernels)
endif ()
