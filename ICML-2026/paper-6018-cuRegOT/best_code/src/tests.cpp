#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;

// Defined in sinkhorn_splr_kernel.cu
void T_computation_sparsify_host(
    int nrun,
    const double* alpha,
    const double* beta,
    const double* M,
    const double* a,
    const double* b,
    double reg, double shift,
    int n, int m, int K,
    double* Trowsums, double* Tcolsums,
    double* objfn, double* grad, double* gnorm,
    double* Tvalues, int* Tflatind,
    double* csr_val, int* csr_rowptr, int* csr_colind
);

// Defined in linsolve.cu
void sparse_cholesky_solve_host(
    const double* values,
    const int* colind,
    const int* rowptr,
    const double* rhs,
    double* x,
    int n,
    int nnz
);

// Python interface for T_computation_sparsify_host function
py::dict test_T_computation_sparsify(
    py::array_t<double> alpha,
    py::array_t<double> beta,
    py::array_t<double> M,
    py::array_t<double> a,
    py::array_t<double> b,
    double reg,
    double shift,
    int K,
    int nrun
)
{
    // Get input array info
    py::buffer_info alpha_buf = alpha.request();
    py::buffer_info beta_buf = beta.request();
    py::buffer_info M_buf = M.request();
    py::buffer_info a_buf = a.request();
    py::buffer_info b_buf = b.request();

    if (alpha_buf.ndim != 1 || beta_buf.ndim != 1 || a_buf.ndim != 1 || b_buf.ndim != 1)
    {
        throw std::runtime_error("alpha, beta, a, and b must be 1D arrays");
    }
    if (M_buf.ndim != 2)
    {
        throw std::runtime_error("M must be a 2D array");
    }

    int n = alpha_buf.shape[0];
    int m = beta_buf.shape[0];

    if (M_buf.shape[0] != n || M_buf.shape[1] != m)
    {
        throw std::runtime_error("Shape mismatch: M should be [n x m], where n=len(alpha) and m=len(beta)");
    }
    if (a_buf.shape[0] != n || b_buf.shape[0] != m)
    {
        throw std::runtime_error("Shape mismatch: a should be [n], and b should be [m], where n=len(alpha) and m=len(beta)");
    }

    // Bound check for K
    int Te = n * (m - 1);
    int Ks = std::max(K, 1);
    Ks = std::min(Ks, Te);

    // Size of Hsl
    size_t Hsize = n + m - 1;

    // Number of nonzero elements in Hsl
    size_t nnz = Ks + Hsize;

    // Get raw pointers
    const double* alpha_ptr = static_cast<const double*>(alpha_buf.ptr);
    const double* beta_ptr = static_cast<const double*>(beta_buf.ptr);
    const double* M_ptr = static_cast<const double*>(M_buf.ptr);
    const double* a_ptr = static_cast<const double*>(a_buf.ptr);
    const double* b_ptr = static_cast<const double*>(b_buf.ptr);

    // Create output arrays for T computation
    py::array_t<double> Trowsums = py::array_t<double>(n);
    py::array_t<double> Tcolsums = py::array_t<double>(m);
    py::array_t<double> grad = py::array_t<double>(n + m - 1);
    py::array_t<double> Tvalues = py::array_t<double>(Te);
    py::array_t<int> Tflatind = py::array_t<int>(Te);

    // Create CSR output arrays
    py::array_t<double> val = py::array_t<double>(nnz);
    py::array_t<int> rowptr = py::array_t<int>(Hsize + 1);
    py::array_t<int> colind = py::array_t<int>(nnz);

    py::buffer_info Trowsums_buf = Trowsums.request();
    py::buffer_info Tcolsums_buf = Tcolsums.request();
    py::buffer_info grad_buf = grad.request();
    py::buffer_info Tvalues_buf = Tvalues.request();
    py::buffer_info Tflatind_buf = Tflatind.request();
    py::buffer_info val_buf = val.request();
    py::buffer_info rowptr_buf = rowptr.request();
    py::buffer_info colind_buf = colind.request();

    double* Trowsums_ptr = static_cast<double*>(Trowsums_buf.ptr);
    double* Tcolsums_ptr = static_cast<double*>(Tcolsums_buf.ptr);
    double* grad_ptr = static_cast<double*>(grad_buf.ptr);
    double* Tvalues_ptr = static_cast<double*>(Tvalues_buf.ptr);
    int* Tflatind_ptr = static_cast<int*>(Tflatind_buf.ptr);
    double* val_ptr = static_cast<double*>(val_buf.ptr);
    int* rowptr_ptr = static_cast<int*>(rowptr_buf.ptr);
    int* colind_ptr = static_cast<int*>(colind_buf.ptr);

    // Call CUDA function
    double objfn, gnorm;
    T_computation_sparsify_host(
        nrun,
        alpha_ptr, beta_ptr, M_ptr, a_ptr, b_ptr, reg, shift, n, m, Ks,
        Trowsums_ptr, Tcolsums_ptr, &objfn, grad_ptr, &gnorm, Tvalues_ptr, Tflatind_ptr,
        val_ptr, rowptr_ptr, colind_ptr
    );

    // Create result dictionary
    py::dict result;
    result["Trowsums"] = Trowsums;
    result["Tcolsums"] = Tcolsums;
    result["objfn"] = objfn;
    result["grad"] = grad;
    result["gnorm"] = gnorm;
    result["Tvalues"] = Tvalues;
    result["Tflatind"] = Tflatind;

    // Add CSR format results
    result["csr_val"] = val;
    result["csr_rowptr"] = rowptr;
    result["csr_colind"] = colind;
    result["K_actual"] = Ks;

    return result;
}

// Python interface for sparse Cholesky solver
py::array_t<double> test_sparse_cholesky_solve(
    py::array_t<double> values,
    py::array_t<int> colind,
    py::array_t<int> rowptr,
    py::array_t<double> rhs
)
{
    // Get input array info
    py::buffer_info values_buf = values.request();
    py::buffer_info colind_buf = colind.request();
    py::buffer_info rowptr_buf = rowptr.request();
    py::buffer_info rhs_buf = rhs.request();

    if (values_buf.ndim != 1 || colind_buf.ndim != 1 || rowptr_buf.ndim != 1 || rhs_buf.ndim != 1)
    {
        throw std::runtime_error("All inputs must be 1D arrays");
    }

    int nnz = values_buf.shape[0];
    int n_rhs = rhs_buf.shape[0];
    int n = rowptr_buf.shape[0] - 1;  // rowptr has length n+1

    if (colind_buf.shape[0] != nnz)
    {
        throw std::runtime_error("values and colind must have the same length");
    }

    if (n_rhs != n)
    {
        throw std::runtime_error("rhs length must equal matrix dimension (rowptr length - 1)");
    }

    // Get raw pointers
    const double* values_ptr = static_cast<const double*>(values_buf.ptr);
    const int* colind_ptr = static_cast<const int*>(colind_buf.ptr);
    const int* rowptr_ptr = static_cast<const int*>(rowptr_buf.ptr);
    const double* rhs_ptr = static_cast<const double*>(rhs_buf.ptr);

    // Create output array for solution
    py::array_t<double> x = py::array_t<double>(n);
    py::buffer_info x_buf = x.request();
    double* x_ptr = static_cast<double*>(x_buf.ptr);

    // Call CUDA function
    sparse_cholesky_solve_host(values_ptr, colind_ptr, rowptr_ptr, rhs_ptr, x_ptr, n, nnz);

    return x;
}
