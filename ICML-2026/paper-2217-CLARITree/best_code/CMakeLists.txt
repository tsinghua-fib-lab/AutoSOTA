cmake_minimum_required(VERSION 3.18)

project(claritree VERSION 0.1.0 LANGUAGES CXX)
option(CLARITREE_ENABLE_IPO "Enable IPO/LTO for release builds" OFF)

# ---- C++ standard ----
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_compile_options($<$<CONFIG:Release>:-O2>)
if(DEFINED ENV{ARCH})
  message(STATUS "Using extra ARCH flags: $ENV{ARCH}")
  add_compile_options("$ENV{ARCH}")
endif()

# ---- Dependencies ----
find_package(Eigen3 REQUIRED)                 # Eigen3::Eigen

# ---- Core library ----
add_library(sr_core
  src/clari_tree.cpp
  src/clari_tree_const.cpp
)

target_include_directories(sr_core PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(sr_core
  PUBLIC
    Eigen3::Eigen
)


target_compile_definitions(sr_core PRIVATE $<$<CONFIG:Debug>:DEBUG=1>)
if (MSVC)
  target_compile_options(sr_core PRIVATE $<$<CONFIG:Debug>:/bigobj>)
endif()

# IPO/LTO
include(CheckIPOSupported)
check_ipo_supported(RESULT ipo_supported OUTPUT ipo_msg)
if (ipo_supported AND CLARITREE_ENABLE_IPO)
  set_property(TARGET sr_core PROPERTY INTERPROCEDURAL_OPTIMIZATION $<$<CONFIG:Release>:ON>)
endif()


set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 REQUIRED CONFIG)      # pybind11::module / pybind11_add_module


pybind11_add_module(sr_py  MODULE
  src/bindings.cpp
)
target_link_libraries(sr_py  PRIVATE sr_core)
target_include_directories(sr_py PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)


set_target_properties(sr_py  PROPERTIES
  OUTPUT_NAME "_core"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/clari_tree"
)
