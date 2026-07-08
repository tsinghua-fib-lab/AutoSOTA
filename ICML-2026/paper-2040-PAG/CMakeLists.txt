cmake_minimum_required(VERSION 3.15)
project(PAG VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(PAG_BUILD_CLI "Build the PAG command-line tool" ON)
option(PAG_BUILD_EXAMPLES "Build PAG examples" OFF)
option(PAG_BUILD_PYTHON "Build Python bindings" OFF)
option(PAG_USE_AVX512 "Build the AVX-512 optimized PAG backend" ON)
option(PAG_NATIVE_ARCH "Compile for the local CPU with -march=native" OFF)
option(PAG_ENABLE_WARNINGS "Enable compiler warning flags" OFF)

set(PAG_OPTIMIZATION_LEVEL "-O3" CACHE STRING "Optimization flag used for PAG targets")
set_property(CACHE PAG_OPTIMIZATION_LEVEL PROPERTY STRINGS "-O2" "-O3" "-Ofast")

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(PAGCompilerOptions)

find_package(OpenMP REQUIRED)

add_library(pag_core STATIC pag.cpp)
add_library(PAG::pag_core ALIAS pag_core)
set_target_properties(pag_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(
  pag_core
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_link_libraries(pag_core PUBLIC OpenMP::OpenMP_CXX)
pag_add_common_options(pag_core)
pag_enable_avx512(pag_core)

if(PAG_BUILD_CLI)
  add_executable(PAG main.cpp)
  target_link_libraries(PAG PRIVATE pag_core)
  pag_add_common_options(PAG)
endif()

if(PAG_BUILD_EXAMPLES)
  add_executable(pag_api_basic examples/api_basic.cpp)
  target_link_libraries(pag_api_basic PRIVATE pag_core)
  pag_add_common_options(pag_api_basic)
endif()

if(PAG_BUILD_PYTHON)
  find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
  find_package(pybind11 CONFIG REQUIRED)
  pybind11_add_module(_pag MODULE python/pag_bindings.cpp)
  target_link_libraries(_pag PRIVATE pag_core)
  target_include_directories(_pag PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  pag_add_common_options(_pag)
  install(TARGETS _pag LIBRARY DESTINATION pag)
endif()

include(PAGInstall)

message(STATUS "PAG_BUILD_CLI=${PAG_BUILD_CLI}")
message(STATUS "PAG_BUILD_EXAMPLES=${PAG_BUILD_EXAMPLES}")
message(STATUS "PAG_BUILD_PYTHON=${PAG_BUILD_PYTHON}")
message(STATUS "PAG_USE_AVX512=${PAG_USE_AVX512}")
message(STATUS "PAG_NATIVE_ARCH=${PAG_NATIVE_ARCH}")
message(STATUS "PAG_ENABLE_WARNINGS=${PAG_ENABLE_WARNINGS}")
message(STATUS "PAG_OPTIMIZATION_LEVEL=${PAG_OPTIMIZATION_LEVEL}")
