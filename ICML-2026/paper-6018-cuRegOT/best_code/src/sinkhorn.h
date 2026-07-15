#pragma once

#include <cuda_runtime.h>

// Main solver functions
void cuda_sinkhorn_bcd(
    const double* M, const double* a, const double* b, double* P,
    double reg, int max_iter, double tol, int n, int m, int* niter,
    const double* x0 = nullptr, double* dual = nullptr,
    bool input_on_device = false, bool output_on_device = false
);

void cuda_sinkhorn_splr(
    const double* M, const double* a, const double* b, double* P,
    double reg, int max_iter, double tol, int n, int m, int* niter,
    double density_max, double shift_max,
    int sparsity_pattern_cycle, int candidate_sinkhorn_iter, int verbose,
    const double* x0 = nullptr, double* dual = nullptr,
    bool input_on_device = false, bool output_on_device = false
);



// Helper function to compute optimal beta given alpha
void compute_optimal_beta(
    const double* d_M, const double* d_alpha, const double* d_logb,
    double* d_beta, double reg, int n, int m,
    cudaStream_t stream = cudaStreamPerThread
);

// Helper function to compute optimal alpha given beta
void compute_optimal_alpha(
    const double* d_M, const double* d_beta, const double* d_loga,
    double* d_alpha, double reg, int n, int m,
    cudaStream_t stream = cudaStreamPerThread
);

// Helper function to compute final transport plan P
void compute_transport_plan(
    const double* d_M, const double* d_alpha, const double* d_beta,
    double* d_P, double reg, int n, int m
);
