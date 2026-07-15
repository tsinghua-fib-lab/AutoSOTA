#pragma once

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
    int sparsity_pattern_cycle, int candidate_sinkhorn_iter, int warm_start_iter, int verbose,
    const double* x0 = nullptr, double* dual = nullptr,
    bool input_on_device = false, bool output_on_device = false
);
