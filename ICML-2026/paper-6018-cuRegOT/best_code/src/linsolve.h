#pragma once

#include <iostream>
#include <cmath>

// CUDA headers
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// cuDSS header
#include <cudss.h>

class SparseCholeskySolver
{
private:
    // CUDA stream
    cudaStream_t  m_stream;
    // cuDSS data structures
    cudssHandle_t m_handle;
    cudssConfig_t m_config;
    cudssData_t   m_data;
    cudssMatrix_t m_A;
    cudssMatrix_t m_b;
    cudssMatrix_t m_x;

public:
    SparseCholeskySolver();
    ~SparseCholeskySolver();

    // Get CUDA stream
    cudaStream_t get_cuda_stream() { return m_stream; }

    // Matrix creation
    void set_A(
        double* d_values, int* d_colind, int* d_rowptr,
        int n, size_t nnz
    );
    void set_b(double* d_rhs, int n, int nrhs = 1);
    void set_x(double* d_sol, int n, int nrhs = 1);

    // reorder() and symfac() are mainly for debugging purposes
    // analyze() is basically a combination of reorder() and symfac()
    void reorder();
    void symfac();

    // Symbolic analysis
    void analyze();
    // Factorization
    void factorize();
    // Solve
    void solve();
};
