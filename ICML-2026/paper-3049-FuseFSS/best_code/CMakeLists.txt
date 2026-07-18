cmake_minimum_required(VERSION 3.24)
project(FuseFSS LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(DEFINED SUF_ENABLE_CUDA AND NOT DEFINED FUSEFSS_ENABLE_CUDA)
  set(FUSEFSS_ENABLE_CUDA "${SUF_ENABLE_CUDA}" CACHE BOOL "Build CUDA backend")
endif()
if(DEFINED SUF_ENABLE_TESTS AND NOT DEFINED FUSEFSS_ENABLE_TESTS)
  set(FUSEFSS_ENABLE_TESTS "${SUF_ENABLE_TESTS}" CACHE BOOL "Build tests")
endif()
if(DEFINED SUF_ENABLE_BENCH AND NOT DEFINED FUSEFSS_ENABLE_BENCH)
  set(FUSEFSS_ENABLE_BENCH "${SUF_ENABLE_BENCH}" CACHE BOOL "Build benchmarks")
endif()

option(FUSEFSS_ENABLE_CUDA "Build CUDA backend" ON)
option(FUSEFSS_ENABLE_TESTS "Build tests" ON)
option(FUSEFSS_ENABLE_BENCH "Build benchmarks" ON)

add_library(fusefss_core INTERFACE)
target_include_directories(fusefss_core INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
add_library(suf_core ALIAS fusefss_core)

add_library(fusefss_compile STATIC
  src/masked_compile.cpp
)
target_link_libraries(fusefss_compile PUBLIC fusefss_core)
add_library(suf_compile ALIAS fusefss_compile)

if(FUSEFSS_ENABLE_CUDA)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  find_package(CUDAToolkit REQUIRED)
  add_library(fusefss_cuda STATIC
    src/cuda/pfss_kernels.cu
    src/cuda/gpu_backend.cu
    src/cuda/dpf_kernels.cu
    src/cuda/dcf_kernels.cu
    src/cuda/dmpf_kernels.cu
    src/cuda/interval_lut_kernels.cu
    src/secure_program.cpp
    src/dmpf.cpp
    src/interval_lut.cpp
  )
  target_include_directories(fusefss_cuda PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(fusefss_cuda PUBLIC fusefss_core fusefss_compile CUDA::cudart)
  target_compile_definitions(fusefss_cuda PUBLIC SUF_HAVE_CUDA=1 FUSEFSS_HAVE_CUDA=1)
  set_target_properties(fusefss_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  add_library(suf_cuda ALIAS fusefss_cuda)
  if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.0)
    target_compile_options(fusefss_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>)
  endif()
endif()

if(FUSEFSS_ENABLE_BENCH AND FUSEFSS_ENABLE_CUDA)
  add_executable(bench_fusefss_gpu src/bench/bench_fusefss_gpu.cpp)
  target_link_libraries(bench_fusefss_gpu PRIVATE fusefss_core fusefss_compile $<$<BOOL:${FUSEFSS_ENABLE_CUDA}>:fusefss_cuda>)
  target_compile_definitions(bench_fusefss_gpu PRIVATE FUSEFSS_ENABLE_BENCH=1 SUF_ENABLE_BENCH=1)

  add_executable(bench_fusefss_model src/bench/bench_fusefss_model.cpp)
  target_link_libraries(bench_fusefss_model PRIVATE fusefss_core fusefss_compile $<$<BOOL:${FUSEFSS_ENABLE_CUDA}>:fusefss_cuda>)
  target_compile_definitions(bench_fusefss_model PRIVATE FUSEFSS_ENABLE_BENCH=1 SUF_ENABLE_BENCH=1)

  add_executable(bench_fusefss_security_canary src/bench/bench_fusefss_security_canary.cpp)
  target_link_libraries(bench_fusefss_security_canary PRIVATE fusefss_core fusefss_compile $<$<BOOL:${FUSEFSS_ENABLE_CUDA}>:fusefss_cuda>)
  target_compile_definitions(bench_fusefss_security_canary PRIVATE FUSEFSS_ENABLE_BENCH=1 SUF_ENABLE_BENCH=1)

  add_executable(bench_sigmoid src/bench/bench_sigmoid.cpp)
  target_link_libraries(bench_sigmoid PRIVATE fusefss_core fusefss_compile $<$<BOOL:${FUSEFSS_ENABLE_CUDA}>:fusefss_cuda>)
  target_compile_definitions(bench_sigmoid PRIVATE FUSEFSS_ENABLE_BENCH=1 SUF_ENABLE_BENCH=1)
elseif(FUSEFSS_ENABLE_BENCH)
  message(STATUS "Skipping FuseFSS benchmarks because FUSEFSS_ENABLE_CUDA=OFF")
endif()

if(FUSEFSS_ENABLE_TESTS)
  enable_testing()
  add_executable(test_fusefss_ref_eval src/tests/test_fusefss_ref_eval.cpp)
  target_link_libraries(test_fusefss_ref_eval PRIVATE fusefss_core)
  add_test(NAME test_fusefss_ref_eval COMMAND test_fusefss_ref_eval)

  add_executable(test_operator_frontend src/tests/test_operator_frontend.cpp)
  target_link_libraries(test_operator_frontend PRIVATE fusefss_core fusefss_compile)
  add_test(NAME test_operator_frontend COMMAND test_operator_frontend)

  if(FUSEFSS_ENABLE_CUDA)
    add_executable(test_fusefss_gpu_smoke src/tests/test_fusefss_gpu_smoke.cpp)
    target_link_libraries(test_fusefss_gpu_smoke PRIVATE fusefss_core fusefss_cuda)
    add_test(NAME test_fusefss_gpu_smoke COMMAND test_fusefss_gpu_smoke)

    add_executable(test_dpf_gpu src/tests/test_dpf_gpu.cpp)
    target_link_libraries(test_dpf_gpu PRIVATE fusefss_core fusefss_cuda)
    add_test(NAME test_dpf_gpu COMMAND test_dpf_gpu)

    add_executable(test_interval_lut_sil2 src/tests/test_interval_lut_sil2.cpp)
    target_link_libraries(test_interval_lut_sil2 PRIVATE fusefss_core fusefss_cuda)
    add_test(NAME test_interval_lut_sil2 COMMAND test_interval_lut_sil2)

    add_executable(test_secure_interval_lut src/tests/test_secure_interval_lut.cpp)
    target_link_libraries(test_secure_interval_lut PRIVATE fusefss_core fusefss_cuda)
    add_test(NAME test_secure_interval_lut COMMAND test_secure_interval_lut)
  endif()
endif()
