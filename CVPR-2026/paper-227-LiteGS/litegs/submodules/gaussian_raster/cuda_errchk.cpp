#include"cuda_errchk.h"
#include<stdio.h>

void cuda_error_check(const char* file, const char* function)
{
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        C10_CUDA_CHECK(err);
    }
}

void cuda_error_check_stage(const char* file, const char* function, const char* stage)
{
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA failure at %s.%s [%s]: %s\n", file, function, stage, cudaGetErrorString(err));
        C10_CUDA_CHECK(err);
    }
}
