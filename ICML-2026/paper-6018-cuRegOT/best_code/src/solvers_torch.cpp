// Only compile this file when PyTorch is available
#if defined(TORCH_BUILD)

#include <iostream>
#include <vector>
#include <algorithm>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

#include "solvers.h"

namespace py = pybind11;

// PyTorch interface for sinkhorn_bcd function
py::dict torch_sinkhorn_bcd(
    torch::Tensor M,
    torch::Tensor a,
    torch::Tensor b,
    double reg,
    double tol,
    int max_iter,
    int verbose,
    const py::kwargs& kwargs
)
{
    // Check input dimensions
    if (M.dim() != 2)
    {
        throw std::runtime_error("M must be a 2D tensor");
    }
    if (a.dim() != 1 || b.dim() != 1)
    {
        throw std::runtime_error("a and b must be 1D tensors");
    }

    int n = M.size(0);
    int m = M.size(1);

    if (a.size(0) != n || b.size(0) != m)
    {
        throw std::runtime_error("Shape mismatch: M is [n x m], a should be length n, b should be length m");
    }

    // Ensure that inputs are on GPU with a double type
    // Zero-copy if already satisfying the conditions
    auto src_options = M.options();
    auto src_device = M.device();
    // If M is already on GPU, preserve its device ID
    // Otherwise use the default device
    auto options = torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCUDA);
    if (src_device.is_cuda())
    {
        options = options.device(torch::kCUDA, src_device.index());
    }
    M = M.to(options);
    a = a.to(options);
    b = b.to(options);

    // Set active device
    const at::cuda::CUDAGuard device_guard(options.device());

    // Ensure contiguous and get data pointers
    M = M.contiguous();
    a = a.contiguous();
    b = b.contiguous();
    // Pointing to device memory
    const double* d_M_ptr = M.data_ptr<double>();
    const double* d_a_ptr = a.data_ptr<double>();
    const double* d_b_ptr = b.data_ptr<double>();

    // Handle kwargs for initial values (x0)
    const double* d_x0_ptr = nullptr;
    torch::Tensor x0_storage;

    if (kwargs.contains("x0"))
    {
        torch::Tensor x0 = py::cast<torch::Tensor>(kwargs["x0"]);
        if (x0.dim() != 1 || x0.size(0) != n + m)
        {
            throw std::runtime_error("x0 must be a 1D tensor of length n + m");
        }

        // Shallow copy x0 to x0_storage if the device, type, and layout of x0 are compatible
        // Otherwise make a conversion
        x0_storage = x0.to(options).contiguous();
        d_x0_ptr = x0_storage.data_ptr<double>();
    }

    // Create output tensors that are double and on device
    torch::Tensor P = torch::empty({n, m}, options);
    torch::Tensor dual = torch::empty(n + m, options);
    // Pointing to device memory
    double* d_P_ptr = P.data_ptr<double>();
    double* d_dual_ptr = dual.data_ptr<double>();

    // Call CUDA function for BCD algorithm
    int niter = 0;
    cuda_sinkhorn_bcd(
        d_M_ptr, d_a_ptr, d_b_ptr, d_P_ptr,
        reg, max_iter, tol, n, m, &niter,
        d_x0_ptr, d_dual_ptr,
        true, true
    );

    // Convert P and dual when necessary using the input tensor options
    P = P.to(src_options);
    dual = dual.to(src_options);

    // Create result dictionary
    py::dict result;
    result["niter"] = niter;
    result["plan"] = P;
    result["dual"] = dual;

    if (verbose > 0)
    {
        std::cout << "torch_sinkhorn_bcd (CUDA implementation) completed" << std::endl;
        std::cout << "Input shape: [" << n << " x " << m << "]" << std::endl;
        std::cout << "Regularization parameter: " << reg << std::endl;
        std::cout << "Tolerance: " << tol << std::endl;
        std::cout << "Max iterations: " << max_iter << std::endl;
        std::cout << "Actual iterations: " << niter << std::endl;
        if (niter == max_iter)
        {
            std::cout << "Warning: Maximum iterations reached without convergence" << std::endl;
        }
    }

    return result;
}

// PyTorch interface for sinkhorn_splr function
py::dict torch_sinkhorn_splr(
    torch::Tensor M,
    torch::Tensor a,
    torch::Tensor b,
    double reg,
    double tol,
    int max_iter,
    int verbose,
    const py::kwargs& kwargs
)
{
    // Check input dimensions
    if (M.dim() != 2)
    {
        throw std::runtime_error("M must be a 2D tensor");
    }
    if (a.dim() != 1 || b.dim() != 1)
    {
        throw std::runtime_error("a and b must be 1D tensors");
    }

    int n = M.size(0);
    int m = M.size(1);

    if (a.size(0) != n || b.size(0) != m)
    {
        throw std::runtime_error("Shape mismatch: M is [n x m], a should be length n, b should be length m");
    }

    // Ensure that inputs are on GPU with a double type
    // Zero-copy if already satisfying the conditions
    auto src_options = M.options();
    auto src_device = M.device();
    // If M is already on GPU, preserve its device ID
    // Otherwise use the default device
    auto options = torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCUDA);
    if (src_device.is_cuda())
    {
        options = options.device(torch::kCUDA, src_device.index());
    }
    M = M.to(options);
    a = a.to(options);
    b = b.to(options);

    // Set active device
    const at::cuda::CUDAGuard device_guard(options.device());

    // Ensure contiguous and get data pointers
    M = M.contiguous();
    a = a.contiguous();
    b = b.contiguous();
    // Pointing to device memory
    const double* d_M_ptr = M.data_ptr<double>();
    const double* d_a_ptr = a.data_ptr<double>();
    const double* d_b_ptr = b.data_ptr<double>();

    // Handle kwargs for initial values (x0)
    const double* d_x0_ptr = nullptr;
    torch::Tensor x0_storage;

    if (kwargs.contains("x0"))
    {
        torch::Tensor x0 = py::cast<torch::Tensor>(kwargs["x0"]);
        if (x0.dim() != 1 || x0.size(0) != n + m)
        {
            throw std::runtime_error("x0 must be a 1D tensor of length n + m");
        }

        // Shallow copy x0 to x0_storage if the device, type, and layout of x0 are compatible
        // Otherwise make a conversion
        x0_storage = x0.to(options).contiguous();
        d_x0_ptr = x0_storage.data_ptr<double>();
    }

    // Get density_max from kwargs
    // Default to 10 / min(n, m)
    double density_max = 10.0 / std::min(n, m);
    density_max = std::min(density_max, 1.0);
    if (kwargs.contains("density"))
    {
        density_max = py::cast<double>(kwargs["density"]);
        density_max = std::min(density_max, 1.0);
        density_max = std::max(density_max, 0.0);
    }

    // Get shift_max from kwargs
    double shift_max = 0.001;
    if (kwargs.contains("shift"))
    {
        shift_max = py::cast<double>(kwargs["shift"]);
        shift_max = std::max(shift_max, 0.0);
    }

    // Get sparsity_pattern_cycle from kwargs
    int sparsity_pattern_cycle = 30;
    if (kwargs.contains("sparsity_pattern_cycle"))
    {
        sparsity_pattern_cycle = py::cast<int>(kwargs["sparsity_pattern_cycle"]);
    }

    // Get candidate_sinkhorn_iter from kwargs
    int candidate_sinkhorn_iter = 3;
    if (kwargs.contains("candidate_sinkhorn_iter"))
    {
        candidate_sinkhorn_iter = py::cast<int>(kwargs["candidate_sinkhorn_iter"]);
    }

    // Create output tensors that are double and on device
    torch::Tensor P = torch::empty({n, m}, options);
    torch::Tensor dual = torch::empty(n + m, options);
    // Pointing to device memory
    double* d_P_ptr = P.data_ptr<double>();
    double* d_dual_ptr = dual.data_ptr<double>();

    // Call CUDA function for SPLR algorithm
    int niter = 0;
    cuda_sinkhorn_splr(
        d_M_ptr, d_a_ptr, d_b_ptr, d_P_ptr,
        reg, max_iter, tol, n, m, &niter,
        density_max, shift_max,
        sparsity_pattern_cycle, candidate_sinkhorn_iter, 0, verbose,
        d_x0_ptr, d_dual_ptr,
        true, true
    );

    // Convert P and dual when necessary using the input tensor options
    P = P.to(src_options);
    dual = dual.to(src_options);

    // Create result dictionary
    py::dict result;
    result["niter"] = niter;
    result["plan"] = P;
    result["dual"] = dual;

    if (verbose > 0)
    {
        std::cout << "torch_sinkhorn_splr (CUDA implementation) completed" << std::endl;
        std::cout << "Input shape: [" << n << " x " << m << "]" << std::endl;
        std::cout << "Regularization parameter: " << reg << std::endl;
        std::cout << "Tolerance: " << tol << std::endl;
        std::cout << "Max iterations: " << max_iter << std::endl;
        std::cout << "Actual iterations: " << niter << std::endl;
        if (niter == max_iter)
        {
            std::cout << "Warning: Maximum iterations reached without convergence" << std::endl;
        }
    }

    return result;
}


#endif  // #if defined(TORCH_BUILD)
