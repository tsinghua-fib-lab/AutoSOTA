cmake_minimum_required(VERSION 3.18)

add_definitions(-DFLASHTP_TORCHSCRIPT)

add_library(sptp_exp_opt_large SHARED
    sptp_exp_opt_large.cpp
    torch_register.cpp
    bwd_bwd_sptp_linear_v2_shared_exp.cu
    bwd_bwd_sptp_linear_v2_shared_exp_double.cu
    bwd_sptp_linear_shared_exp.cu
    bwd_sptp_linear_shared_exp_double.cu
    fused_e3nn_exp_opt_large.py
    fwd_sptp_linear_v2_shared_exp.cu
    fwd_sptp_linear_v2_shared_exp_double.cu
)

target_compile_options(sptp_exp_opt_large PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                       -fPIC
                       >)

#set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --fast-math -O3")

target_include_directories(sptp_exp_opt_large PRIVATE ${Python3_INCLUDE_DIRS})
target_link_libraries(sptp_exp_opt_large PRIVATE ${Python3_LIBRARIES})

target_compile_features(sptp_exp_opt_large PRIVATE cxx_std_14)
target_link_libraries(sptp_exp_opt_large PRIVATE "${TORCH_LIBRARIES}")

target_include_directories(sptp_exp_opt_large PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}   # for "sptp.hpp"
)
