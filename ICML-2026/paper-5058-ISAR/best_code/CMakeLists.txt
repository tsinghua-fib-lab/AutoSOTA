cmake_minimum_required(VERSION 3.22)
project(ISAR CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE INTERNAL "")

# If no build type is specified, set Release as the default
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'Release' as none was specified.")
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# project
add_subdirectory(src)


#targets
add_executable(ISAR
   src/main.cpp
   src/graph.cpp
   src/mwu.cpp
   src/bounds.cpp)
target_include_directories(ISAR
   PRIVATE
   ${PROJECT_SOURCE_DIR}/include)
