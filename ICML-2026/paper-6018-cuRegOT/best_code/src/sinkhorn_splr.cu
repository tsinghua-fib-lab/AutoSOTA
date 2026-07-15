#include <iostream>
#include <cmath>
#include <thread>
#include <atomic>

// CUDA headers
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Annotate code blocks for profiling
#include <nvtx3/nvtx3.hpp>

// Utility functions
#include "utils.h"
#include "timer.h"
#include "sinkhorn.h"

// Linear solver
#include "linsolve.h"

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// Define block dimension
#define BLOCK_DIM 256
#define MAX_NUM_BLOCK_1D 256

// Helper function to transform the gamma=(alpha, beta) vector to gamma'=(alpha', beta_t', 0)
// alpha += beta[m-1], beta -= beta[m-1]
// From sinkhorn_splr_kernel.cu
void shift_gamma(double* d_gamma, int n, int m, cudaStream_t stream = cudaStreamPerThread);

// Helper functions to compute objective function value objfn,
// gradient grad, and sparsified Hessian in CSR form
// From sinkhorn_splr_kernel.cu
void launch_T_objfn_grad(
    const double* d_gamma,
    const double* d_M,
    const double* d_ab,
    double reg,
    int n, int m,
    bool write_values_and_indices,
    bool compute_grad_norm,
    double* d_grad,
    double& objfn, double& gnorm,
    double* d_work, int* d_iwork,
    double* h_pinned,
    cudaStream_t stream = cudaStreamPerThread
);
void launch_sphess(
    const double* d_gamma,
    const double* d_M,
    const double* d_work,
    int* d_iwork,
    double reg, double shift,
    int n, int m, int K,
    double* d_Hvalues, int* d_Hflatind, int* d_Hcolind, int* d_Hrowptr,
    bool fixed_indices = false,
    cudaStream_t stream = cudaStreamPerThread
);

// Helper function for line search computation
void launch_line_search_computation(
    const double* d_gamma_prev,
    const double* d_direc,
    double step,
    const double* d_M,
    const double* d_ab,
    double reg,
    int n, int m,
    double* d_gamma,
    double* d_grad,
    double& objfn,
    double& dg,
    double* d_work, int* d_iwork, double* h_pinned,
    bool write_values_and_indices = true,
    cudaStream_t stream = cudaStreamPerThread
);

// Helper function to compute low-rank vectors y and s
// From sinkhorn_splr_kernel_low_rank.cu
void launch_low_rank(
    const double* d_grad,
    const double* d_grad_prev,
    const double* d_gamma,
    const double* d_gamma_prev,
    double* d_y,
    double* d_s,
    double& ys,
    double& yy,
    int size,
    double* h_pinned,
    cudaStream_t stream = cudaStreamPerThread
);

// Helper function to compute search direction with low-rank terms
// From sinkhorn_splr_kernel_low_rank.cu
void launch_low_rank_search_direc(
    double* d_direc,
    const double* d_invA_y,
    const double* d_g,
    const double* d_y,
    const double* d_s,
    double ys,
    double reg,
    int size,
    double* h_pinned,
    cudaStream_t stream = cudaStreamPerThread
);

// Class for the SPLR solver
class SPLRSolver
{
private:
    // Problem dimensions
    const int     m_n;
    const int     m_m;
    const size_t  m_Me;
    const size_t  m_Te;
    const size_t  m_Hsize;
    const size_t  m_Kmax;
    // Regularization parameter
    const double  m_reg;
    // Input matrices and vectors on device
    const double* d_M;
    double*       d_M_storage;
    double*       d_ab;
    double*       d_logab;
    // Dual variables on device
    double*       d_gamma;
    double*       d_gamma_prev;
    // Sinkhorn BCD iterates
    cudaStream_t  m_sinkhorn_stream;
    std::thread   m_sinkhorn_thread;
    std::atomic<bool> m_sinkhorn_stop_flag;
    double*       d_gamma_bcd;
    double*       d_grad_bcd;
    double        m_objfn_bcd;
    double        m_gnorm_bcd;
    // Pointer aliases, d_gamma = (d_alpha, d_beta)
    double*       d_alpha;
    double*       d_beta;
    // Gradient
    double*       d_grad;
    double*       d_grad_prev;
    // Search direction and low-rank vectors
    double*       d_linsolve_rhs;  // [grad, y]
    double*       d_linsolve_sol;  // [direc, invA_y]
    double*       d_direc;
    double*       d_y;
    double*       d_s;
    double*       d_invA_y;
    // Sparsified Hessian in CSR representation
    double*       d_Hvalues;
    int*          d_Hflatind;
    int*          d_Hcolind;
    int*          d_Hrowptr;
    // Working space
    double*       d_work;
    int*          d_iwork;
    double*       h_pinned;
    // Sparse Cholesky solver
    SparseCholeskySolver m_linsolver;

public:
    // Constructor
    SPLRSolver(
        const double* M, const double* a, const double* b,
        double reg, int n, int m, size_t Kmax,
        bool input_on_device = false
    ):
        m_n(n), m_m(m), m_Me(n * m), m_Te(n * (m - 1)), m_Hsize(n + m - 1), m_Kmax(Kmax), m_reg(reg),
        d_M_storage(nullptr), m_sinkhorn_stop_flag(true)
    {
        nvtx3::scoped_range r{"SPLRSolver"};

        // If M is already on the device, then directly assign M to d_M
        // Otherwise, allocate device memory and copy from the host pointer
        double *d_M_storage = nullptr;
        if (input_on_device)
        {
            d_M = M;
        }
        else
        {
            // Allocate device memory
            CUDA_CHECK(cudaMalloc(&d_M_storage, m_Me * sizeof(double)));
            // Copy input to device
            CUDA_CHECK(cudaMemcpy(d_M_storage, M, m_Me * sizeof(double), cudaMemcpyHostToDevice));
            // Set input data pointers
            d_M = d_M_storage;
        }

        // Allocate device memory
        CUDA_CHECK(cudaMalloc(&d_ab, (m_n + m_m) * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_logab, (m_n + m_m) * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_gamma, (m_n + m_m) * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_gamma_prev, (m_n + m_m) * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_gamma_bcd, (m_n + m_m) * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_grad_bcd, m_Hsize * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_grad_prev, m_Hsize * sizeof(double)));
        // In computing the search direction, we need to solve
        // A * direc = grad or A * [direc, invA_y] = [grad, y],
        // depending on whether the low-rank term is used
        // Therefore, we should make the memory blocks [direc, invA_y] and [grad, y] contiguous
        CUDA_CHECK(cudaMalloc(&d_linsolve_rhs, 2 * m_Hsize * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_linsolve_sol, 2 * m_Hsize * sizeof(double)));
        // CUDA_CHECK(cudaMalloc(&d_grad, m_Hsize * sizeof(double)));
        // CUDA_CHECK(cudaMalloc(&d_direc, m_Hsize * sizeof(double)));
        // CUDA_CHECK(cudaMalloc(&d_y, m_Hsize * sizeof(double)));
        // CUDA_CHECK(cudaMalloc(&d_invA_y, m_Hsize * sizeof(double)));
        d_grad = d_linsolve_rhs;
        d_direc = d_linsolve_sol;
        d_y = d_linsolve_rhs + m_Hsize;
        d_invA_y = d_linsolve_sol + m_Hsize;
        CUDA_CHECK(cudaMalloc(&d_s, m_Hsize * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_Hvalues, (Kmax + m_Hsize) * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_Hflatind, (Kmax + m_Hsize) * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_Hcolind, (Kmax + m_Hsize) * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_Hrowptr, (m_Hsize + 1) * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_work, (m_n + m_m + m_Te) * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_iwork, std::max(m_Te, m_Hsize) * sizeof(int)));

        // Pinned memory
        CUDA_CHECK(cudaHostAlloc(&h_pinned, (4 * MAX_NUM_BLOCK_1D) * sizeof(double), cudaHostAllocMapped));

        // Pointer aliases
        d_alpha = d_gamma;
        d_beta = d_gamma + m_n;

        // Copy a and b to d_ab
        CUDA_CHECK(cudaMemcpy(
            d_ab, a, m_n * sizeof(double),
            input_on_device ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice
        ));
        CUDA_CHECK(cudaMemcpy(
            d_ab + m_n, b, m_m * sizeof(double),
            input_on_device ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice
        ));

        // Compute log(a) and log(b)
        compute_log_vector_cuda(d_ab, d_logab, m_n + m_m);

        // Set d_grad_prev to zero
        CUDA_CHECK(cudaMemsetAsync(d_grad_prev, 0, m_Hsize * sizeof(double), cudaStreamPerThread));

        // Create CUDA stream for Sinkhorn iterations
        CUDA_CHECK(cudaStreamCreate(&m_sinkhorn_stream));
    }

    // Destructor
    ~SPLRSolver()
    {
        nvtx3::scoped_range r{"~SPLRSolver"};

        // Free device memory
        if (d_M_storage != nullptr)
        {
            CUDA_CHECK(cudaFree(d_M_storage));
        }
        CUDA_CHECK(cudaFree(d_ab));
        CUDA_CHECK(cudaFree(d_logab));
        CUDA_CHECK(cudaFree(d_gamma));
        CUDA_CHECK(cudaFree(d_gamma_prev));
        CUDA_CHECK(cudaFree(d_gamma_bcd));
        CUDA_CHECK(cudaFree(d_grad_bcd));
        CUDA_CHECK(cudaFree(d_grad_prev));
        CUDA_CHECK(cudaFree(d_linsolve_rhs));
        CUDA_CHECK(cudaFree(d_linsolve_sol));
        // CUDA_CHECK(cudaFree(d_grad));
        // CUDA_CHECK(cudaFree(d_direc));
        // CUDA_CHECK(cudaFree(d_y));
        // CUDA_CHECK(cudaFree(d_invA_y));
        CUDA_CHECK(cudaFree(d_s));
        CUDA_CHECK(cudaFree(d_Hvalues));
        CUDA_CHECK(cudaFree(d_Hflatind));
        CUDA_CHECK(cudaFree(d_Hcolind));
        CUDA_CHECK(cudaFree(d_Hrowptr));
        CUDA_CHECK(cudaFree(d_work));
        CUDA_CHECK(cudaFree(d_iwork));

        // Free pinned memory
        CUDA_CHECK(cudaFreeHost(h_pinned));

        // Destroy CUDA stream for Sinkhorn iterations
        CUDA_CHECK(cudaStreamDestroy(m_sinkhorn_stream));
    }

    // Sinkhorn iteration to update alpha
    void update_alpha()
    {
        // Get pointer for log(a)
        const double* d_loga = d_logab;

        // Optimal alpha given beta
        // d_alpha = d_gamma
        // d_beta = d_gamma + n
        compute_optimal_alpha(d_M, d_beta, d_loga, d_alpha, m_reg, m_n, m_m, cudaStreamPerThread);
    }

    // Sinkhorn iteration to update beta
    void update_beta()
    {
        // Get pointer for log(b)
        const double* d_logb = d_logab + m_n;

        // Optimal beta given alpha
        // d_alpha = d_gamma
        // d_beta = d_gamma + n
        compute_optimal_beta(d_M, d_alpha, d_logb, d_beta, m_reg, m_n, m_m, cudaStreamPerThread);
    }

    // Initialize dual variables
    void init_dual(const double* x0, bool input_on_device = false)
    {
        nvtx3::scoped_range r{"init_dual"};

        // Initialize dual variable gamma
        if (x0 != nullptr)
        {
            // Use provided initial values: x0 contains [alpha (n elements), beta (m elements)]
            // But note that we force beta[m-1]=0, so we do a shifting
            // alpha += beta[m-1], beta -= beta[m-1]

            // Copy x0 to d_gamma
            CUDA_CHECK(cudaMemcpy(
                d_gamma, x0, (m_n + m_m) * sizeof(double),
                input_on_device ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice
            ));
            // Shift d_gamma
            shift_gamma(d_gamma, m_n, m_m, cudaStreamPerThread);
        }
        else
        {
            // If no initial values are provided, first set beta to zero,
            // and then compute alpha using BCD iteration
            CUDA_CHECK(cudaMemsetAsync(d_beta, 0, m_m * sizeof(double), cudaStreamPerThread));
            update_alpha();
        }

        // Initialize dual variable in previous iteration
        CUDA_CHECK(cudaMemsetAsync(d_gamma_prev, 0, (m_n + m_m) * sizeof(double), cudaStreamPerThread));
    }

    // Compute objective function value, gradient, and sparsified Hessian
    void dual_T_objfn_grad(
        double& objfn, double& gnorm,
        bool compute_gnorm = true,
        bool fixed_indices = false
    )
    {
        nvtx3::scoped_range r{"dual_T_objfn_grad"};

        // launch computation
        launch_T_objfn_grad(
            d_gamma, d_M, d_ab,
            m_reg, m_n, m_m,
            !fixed_indices,
            compute_gnorm,
            d_grad, objfn, gnorm,
            d_work, d_iwork, h_pinned,
            cudaStreamPerThread
        );
    }

    size_t dual_sphess(double density, double shift, bool fixed_indices = false)
    {
        nvtx3::scoped_range r{"dual_sphess"};

        // Make sure density is within (0, 1)
        density = std::min(density, 1.0);
        density = std::max(density, 0.0);

        // Keep K elements in T_t
        size_t K = static_cast<size_t>(density * m_Te);
        K = std::min(K, m_Kmax);
        K = std::max(K, size_t(1));

        // launch computation
        launch_sphess(
            d_gamma, d_M, d_work, d_iwork,
            m_reg, shift, m_n, m_m, K,
            d_Hvalues, d_Hflatind, d_Hcolind, d_Hrowptr,
            fixed_indices,
            cudaStreamPerThread
        );

        // Return number of nonzeros in sparsified Hessian
        size_t nnz = K + m_Hsize;
        return nnz;
    }

    // Get current gradient norm
    double grad_norm() const
    {
        nvtx3::scoped_range r{"grad_norm"};

        return compute_l2_norm_cuda(d_grad, h_pinned, m_Hsize, cudaStreamPerThread);
    }

    // Compute low-rank vectors
    void compute_low_rank(double& ys, double& yy)
    {
        nvtx3::scoped_range r{"compute_low_rank"};

        // y = grad - grad_prev
        // s = gamma - gamma_prev
        // ys = y's
        // yy = y'y
        launch_low_rank(
            d_grad, d_grad_prev, d_gamma, d_gamma_prev,
            d_y, d_s, ys, yy,
            m_Hsize,
            h_pinned,
            cudaStreamPerThread
        );
    }

    // Compute Sinkhorn iterate
    // This will be parallel to compute_search_direc()
private:
    // Some helper functions
    //
    // Compute objective function value and gradient on the Sinkhorn iterate
    void post_sinkhorn_iterate()
    {
        // Shift d_gamma_bcd such that the last element is zero
        shift_gamma(d_gamma_bcd, m_n, m_m, m_sinkhorn_stream);

        // Compute the objective function value and gradient of the Sinkhorn iterate
        launch_T_objfn_grad(
            d_gamma_bcd, d_M, d_ab,
            m_reg, m_n, m_m,
            false,  // write_values_and_indices = false means not writing to d_Tvalues and d_Tflatind
            true,   // compute gradient norm
            d_grad_bcd, m_objfn_bcd, m_gnorm_bcd,
            d_work, d_iwork, h_pinned,
            m_sinkhorn_stream
        );
    }
    // Sinkhorn loops that listen to the stopping flag
    void sinkhorn_loop_signal()
    {
        const double* d_loga = d_logab;
        const double* d_logb = d_logab + m_n;
        double* d_alpha_bcd = d_gamma_bcd;
        double* d_beta_bcd = d_gamma_bcd + m_n;

        int i = 0;
        while (!m_sinkhorn_stop_flag.load() && i < 1000)
        {
            compute_optimal_beta(d_M, d_alpha_bcd, d_logb, d_beta_bcd, m_reg, m_n, m_m, m_sinkhorn_stream);
            compute_optimal_alpha(d_M, d_beta_bcd, d_loga, d_alpha_bcd, m_reg, m_n, m_m, m_sinkhorn_stream);
            cudaStreamSynchronize(m_sinkhorn_stream);
            i++;
        }

        // Compute objective function value and gradient on the Sinkhorn iterate
        post_sinkhorn_iterate();
    }
    // Sinkhorn loops with a fixed number of iterations
    void sinkhorn_loop_fixed(int niter)
    {
        const double* d_loga = d_logab;
        const double* d_logb = d_logab + m_n;
        double* d_alpha_bcd = d_gamma_bcd;
        double* d_beta_bcd = d_gamma_bcd + m_n;

        // Sinkhorn iterations
        for (int i = 0; i < niter; i++)
        {
            compute_optimal_beta(d_M, d_alpha_bcd, d_logb, d_beta_bcd, m_reg, m_n, m_m, m_sinkhorn_stream);
            compute_optimal_alpha(d_M, d_beta_bcd, d_loga, d_alpha_bcd, m_reg, m_n, m_m, m_sinkhorn_stream);
        }

        // Compute objective function value and gradient on the Sinkhorn iterate
        post_sinkhorn_iterate();
    }
public:
    // Main function
    // niter > 0  =>  run a fixed number of Sinkhorn iterations
    // niter < 0  =>  automatically fill GPU idle time
    // niter = 0  =>  no Sinkhorn iterations
    void compute_sinkhorn_iterate(int niter = 3)
    {
        nvtx3::scoped_range r{"compute_sinkhorn_iterate"};

        const double* d_loga = d_logab;
        double* d_alpha_bcd = d_gamma_bcd;

        // Read beta from d_gamma and write BCD-computed alpha to d_gamma_bcd
        compute_optimal_alpha(d_M, d_beta, d_loga, d_alpha_bcd, m_reg, m_n, m_m, m_sinkhorn_stream);

        // If niter < 0, we create a new thread to launch CUDA kernels, and the thread will finish
        // after m_sinkhorn_stop_flag is set to true in compute_search_direc()
        //
        // If niter > 0, we create a new thread to run a fixed number of Sinkhorn iterations
        // The thread will finish at save_history()
        if (niter < 0)
        {
            // Launch Sinkhorn iteration loop that listens to the stopping flag
            m_sinkhorn_stop_flag.store(false);
            m_sinkhorn_thread = std::thread(&SPLRSolver::sinkhorn_loop_signal, this);
        }
        else
        {
             // Launch Sinkhorn iteration loop with a fixed number of iterations
            m_sinkhorn_thread = std::thread(&SPLRSolver::sinkhorn_loop_fixed, this, niter);
        }
    }

    // Get objective function value and gradient norm of the Sinkhorn iterate
    void objfn_gnorm_bcd(double& objfn, double& gnorm) const
    {
        nvtx3::scoped_range r{"objfn_gnorm_bcd"};

        objfn = m_objfn_bcd;
        gnorm = m_gnorm_bcd;
    }

    // Compute search direction (with low-rank term)
    void compute_search_direc(size_t nnz, double ys, bool low_rank = true, bool analyze_pattern = true, int verbose = 0)
    {
        nvtx3::scoped_range r{"compute_search_direc"};

        // Solve (H + UCV) * d = -g
        // U = [u, v], C = diag(a, b), V = U'
        // u = y, v = H * s
        // a = 1 / u's, b = -1 / v's

        // BFGS update rule
        // https://en.wikipedia.org/wiki/Broyden%E2%80%93Fletcher%E2%80%93Goldfarb%E2%80%93Shanno_algorithm

        // inv(H + UCV) = inv(H) + (y's + y'inv(H)y)(ss') - inv(H)ys' + sy'inv(H)
        //                         ----------------------   ---------------------   
        //                                 (y's)^2                   y's
        //
        // Therefore,
        // d = -inv(H + UCV)g = -inv(H)g - (y's + y'inv(H)y)(s'g)s + (s'g)inv(H)y + sy'inv(H)g
        //                                 -----------------------   -------------------------
        //                                         (y's)^2                      y's
        //
        // Let d0 = -inv(H)g
        // If no low-rank term is used, then directly return d0

        // Note that we have actually stored A = reg * H,
        // so we first compute
        // x = inv(A)g + (y's / reg + y'inv(A)y)(s'g)s - (s'g)inv(A)y + sy'inv(A)g
        //               -----------------------------   -------------------------
        //                          (y's)^2                         y's
        // and then do the scaling d <- -reg * x
        
        // direc = invA_g = inv(A) * g;
        Timer timer(verbose >= 3);
        timer.tic();
        cudaStream_t stream = m_linsolver.get_cuda_stream();
        // If low_rank = true, then we need to solve two vectors,
        //     A * [direc, invA_y] = [grad, y]
        // Otherwise we only solve A * direc = grad
        // Since the memory layout is d_linsolve_rhs = [d_grad, d_y] and
        // d_linsolve_sol = [d_direc, d_invA_y], we just need to specify
        // how many rhs to solve
        m_linsolver.set_A(d_Hvalues, d_Hcolind, d_Hrowptr, m_Hsize, nnz);
        m_linsolver.set_b(d_linsolve_rhs, m_Hsize, low_rank ? 2 : 1);
        m_linsolver.set_x(d_linsolve_sol, m_Hsize, low_rank ? 2 : 1);

        if (analyze_pattern)
        {
            // Scope for NVTX3 annotation
            {
                nvtx3::scoped_range reorder{"reorder"};

                // Reordering stage mainly runs on CPU
                m_linsolver.reorder();
                timer.toc("reorder");
            }

            // After the reordering part, we can stop the thread running
            // sinkhorn_loop_signal(), since the factorization step later will run on GPU
            //
            // But we do not want to stop sinkhorn_loop_fixed() until save_history(),
            // so we query m_sinkhorn_stop_flag here. This flag is only set to false for 
            // sinkhorn_loop_signal()
            if (!m_sinkhorn_stop_flag.load())
            {
                m_sinkhorn_stop_flag.store(true);
                if (m_sinkhorn_thread.joinable())
                {
                    m_sinkhorn_thread.join();
                }
            }

            // Scope for NVTX3 annotation
            {
                nvtx3::scoped_range sym_fac{"sym_fac"};
                m_linsolver.symfac();
                timer.toc("symfac");
            }
        }
        m_linsolver.factorize();
        timer.toc("factorize");
        m_linsolver.solve();
        timer.toc("solve");

        // After calling solve(), d_direc will contain the desired value
        // If low_rank = true, d_invA_y will also be available

        if (low_rank)
        {
            // invA_y = inv(A) * y;

            // BFGS rule
            // sg = s'g, ys = y's
            // yinvAy = y'(invA_y), yinvAg = y'(invA_g) = y'(direc)
            // sg_ys = sg / ys
            // direc += ((1 / reg + yinvAy / ys) * sg_ys - yinvAg / ys) * s - sg_ys * invA_y
            launch_low_rank_search_direc(d_direc, d_invA_y, d_grad, d_y, d_s, ys, m_reg, m_Hsize, h_pinned, stream);
        }
        timer.toc("low_rank");

        // Scaling d <- -reg * x
        compute_scaling_inplace_cuda(-m_reg, d_direc, m_Hsize, stream);
        timer.toc("scaling");

        if (verbose >= 3)
        {
            std::cout << "[search_direc_timing]--------------------------------------" << std::endl;
            std::cout << "║ reorder = " << timer["reorder"] << ", symfac = " << timer["symfac"] << std::endl;
            std::cout << "║ factorize = " << timer["factorize"] << ", solve = " << timer["solve"] << std::endl;
            std::cout << "║ low_rank = " << timer["low_rank"] << ", scaling = " << timer["scaling"] << std::endl;
            std::cout << "===========================================================" << std::endl;
        }
    }

    // Save d_gamma to d_gamma_prev, and d_grad to d_grad_prev
    void save_history()
    {
        nvtx3::scoped_range r{"save_history"};

        CUDA_CHECK(cudaMemcpy(d_gamma_prev, d_gamma, m_Hsize * sizeof(double), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(d_grad_prev, d_grad, m_Hsize * sizeof(double), cudaMemcpyDeviceToDevice));

        // Finish the thread in charge of Sinkhorn iterations
        if (m_sinkhorn_thread.joinable())
        {
            m_sinkhorn_thread.join();
        }
    }

    // More-Thuente line search with Wolfe conditions
    double line_search_wolfe(
        double init_step, double cur_obj, double& new_obj, bool& recompute_fg,
        bool fixed_indices = false,
        double c1 = 1e-4, double c2 = 0.9, int max_iter = 20
    )
    {
        using std::abs;
        nvtx3::scoped_range r{"line_search_wolfe"};

        // We assume d_gamma has been copied to d_gamma_prev,
        // so new point is computed as
        //     d_gamma = d_gamma_prev + step * d_direc
        // d_gamma_prev and d_direc are read-only during line search,
        // and d_gamma and d_grad will be overwritten

        // Typically the objective function value (f) and gradient (g)
        // have been computed on the new point when line search exits,
        // but there are cases that a different step is returned
        // In such cases, we flag recompute_fg = true
        recompute_fg = false;

        // Initial step size
        constexpr double step_min = 1e-10, step_max = 2.0;
        init_step = (std::max)(init_step, step_min);
        init_step = (std::min)(init_step, step_max);
        double step = init_step;
        double fx = cur_obj, dg = compute_dot_prod_cuda(d_grad_prev, d_direc, h_pinned, m_Hsize, cudaStreamPerThread);

        // Save the function value at the current x
        const double fx_init = cur_obj;
        // Projection of gradient on the search direction
        const double dg_init = dg;
        // Make sure d points to a descent direction
        if (dg_init > 0.0)
        {
            recompute_fg = true;
            return init_step;
        }

        // Tolerance for convergence test
        // Sufficient decrease
        const double test_decr = c1 * dg_init;
        // Curvature
        const double test_curv = -c2 * dg_init;

        // The bracketing interval
        constexpr double Inf = std::numeric_limits<double>::infinity();
        double I_lo = 0.0, I_hi = Inf;
        double fI_lo = 0.0, fI_hi = Inf;
        double gI_lo = (1.0 - c1) * dg_init, gI_hi = Inf;
        double psiI_lo = fI_lo;
        double fx_lo = fx_init, dg_lo = dg_init;

        // Status variables
        bool bracketed = false;
        bool f_is_psi = true;
        bool use_step_min_safeguard = (step_min > 0.0);
        double I_width = Inf;
        double I_width_prev = Inf;
        int I_shrink_fail_count = 0;

        // Constants
        constexpr double delta_max = 1.1;
        constexpr double delta_min = 7.0 / 12;
        constexpr double shrink = 0.66;
        int iter;
        for (iter = 0; iter < max_iter; iter++)
        {
            // Evaluate the current step size
            // 1. gamma = gamma_prev + step * direc
            // 2. Compute f and g
            // 3. Compute <g, direc>
            launch_line_search_computation(
                d_gamma_prev, d_direc, step,
                d_M, d_ab, m_reg, m_n, m_m,
                d_gamma, d_grad, fx, dg,
                d_work, d_iwork, h_pinned,
                !fixed_indices,
                cudaStreamPerThread
            );
            new_obj = fx;

            // phi(step) = f(xp + step * drt) = fx
            // phi'(step) = g(xp + step * drt)^T d = dg
            // psi(step) = f(xp + step * drt) - f(xp) - step * test_decr
            // psi'(step) = dg - test_decr
            const double psit = fx - fx_init - step * test_decr;
            const double dpsit = dg - test_decr;

            // Convergence test
            if (psit <= 0.0 && abs(dg) <= test_curv)
            {
                return step;
            }

            // Test whether step hits the boundaries and satisfies the exit conditions
            if (step <= step_min && (psit > 0.0 || dpsit >= 0.0))
            {
                return step;
            }
            if (step >= step_max && (psit <= 0.0 && dpsit < 0.0))
            {
                return step;
            }

            // Check and update the status of f_is_psi
            // f is initially set to psi, and is changed to phi in
            // subsequent iterations if psi(step) <= 0 and phi'(step) >= 0
            //
            // NOTE: empirically we find that using psi is usually better,
            //       so for now we do not follow the implementation of Moré and Thuente (1994)
            /*
            if (f_is_psi && (psit <= 0.0 && dg >= 0.0))
            {
                f_is_psi = false;
            }
            */
            const double ft = f_is_psi ? psit : fx;
            const double gt = f_is_psi ? dpsit : dg;

            // Check and update the status of use_step_min_safeguard
            // We impose a safeguarding rule that guarantees testing
            // step_min if psi(alpha_k) > 0 or psi'(alpha_k) >= 0
            // holds from the beginning
            if (use_step_min_safeguard && (psit <= 0.0 && dpsit < 0.0))
            {
                use_step_min_safeguard = false;
            }

            // Update new step
            double new_step;
            const bool in_case_2 = (psit <= psiI_lo) && (dpsit * (I_lo - step) > 0.0);
            if (in_case_2)
            {
                // For Case 2, we apply the safeguarding rule
                // newat = min(at + delta * (at - al), amax), delta in [1.1, 4]
                new_step = (std::min)(step_max, step + delta_max * (step - I_lo));
            }
            else
            {
                // For Case 1 and Case 3, use information of f and g to select new step
                new_step = step_selection(I_lo, I_hi, step, fI_lo, fI_hi, ft, gI_lo, gI_hi, gt);
                // Force new step in [step_min, step_max]
                new_step = (std::max)(new_step, step_min);
                new_step = (std::min)(new_step, step_max);

                // Apply safeguarding rule related to step_min when necessary:
                //     step+ in [alpha_min, max{delta_min * step, alpha_min}]
                //
                // If use_step_min_safeguard = true, then new_step cannot be obtained
                // from Case 2, since in Case 2 we have
                //     psi(alpha_k) <= 0 and psi'(alpha_k) < 0
                if (use_step_min_safeguard)
                {
                    const double lower = step_min;
                    const double upper = (std::max)(step_min, delta_min * step);
                    new_step = (std::max)(new_step, lower);
                    new_step = (std::min)(new_step, upper);
                }
            }

            // Update bracketing interval
            if (psit > psiI_lo)
            {
                // Case 1: psi(step) > psi(I_lo)
                I_hi = step;
                fI_hi = ft;
                gI_hi = gt;
            }
            else if (in_case_2)
            {
                // Case 2: psi(step) <= psi(I_lo), psi'(step)(I_lo - step) > 0
                I_lo = step;
                fI_lo = ft;
                gI_lo = gt;
                psiI_lo = psit;
                fx_lo = fx;
                dg_lo = dg;
            }
            else
            {
                // Case 3: psi(step) <= psi(I_lo), psi'(step)(I_lo - step) <= 0
                I_hi = I_lo;
                fI_hi = fI_lo;
                gI_hi = gI_lo;

                I_lo = step;
                fI_lo = ft;
                gI_lo = gt;
                psiI_lo = psit;
                fx_lo = fx;
                dg_lo = dg;
            }

            // Check and update the status of bracketed
            // bracketed is true if we have entered Case 1 or Case 3,
            // and I is contained in [step_min, step_max]
            if ((!bracketed) && (!in_case_2))
            {
                const double I_left = (std::min)(I_lo, I_hi);
                const double I_right = (std::max)(I_lo, I_hi);
                bracketed = (I_left >= step_min && I_right <= step_max);
            }

            // If bracketed, enforce sufficient interval shrink; if not shrinking enough, use bisection
            if (bracketed)
            {
                I_width_prev = I_width;
                I_width = abs(I_hi - I_lo);
                // Test interval shrinkage
                if (I_width_prev < Inf && I_width > shrink * I_width_prev)
                {
                    I_shrink_fail_count += 1;
                }
                else
                {
                    I_shrink_fail_count = 0;
                }
                // If interval fails to shrink enough twice, select new_step using bisection
                if (I_shrink_fail_count >= 2)
                {
                    new_step = (I_lo + I_hi) / 2.0;
                    I_shrink_fail_count = 0;
                }
            }

            // Set the new_step
            step = new_step;
        }

        // If we have used up all line search iterations, then the strong Wolfe condition
        // is not met. We choose not to raise an exception, but to return the best
        // step size so far

        // First test whether the last step is better than I_lo
        // If yes, return the last step
        const double psit = fx - fx_init - step * test_decr;
        if (psit <= psiI_lo)
            return step;

        // If not, then the best step size so far is I_lo, but it needs to be positive
        if (I_lo <= 0.0)
        {
            recompute_fg = true;
            return init_step;
        }

        // Return everything with _lo
        recompute_fg = true;
        step = I_lo;
        fx = fx_lo;
        dg = dg_lo;
        return step;
    }

    // Update iterate, d_gamma = d_gamma_prev + alpha * d_direc
    void update_gamma(double alpha)
    {
        nvtx3::scoped_range r{"update_gamma"};

        compute_axpy_cuda(d_direc, d_gamma_prev, alpha, d_gamma, m_Hsize, cudaStreamPerThread);
    }

    // Update gamma using the Sinkhorn iterate
    void update_gamma_sinkhorn()
    {
        nvtx3::scoped_range r{"update_gamma_sinkhorn"};

        CUDA_CHECK(cudaMemcpy(d_gamma, d_gamma_bcd, m_Hsize * sizeof(double), cudaMemcpyDeviceToDevice));
    }

    // Output results to host -- transport plan and dual variables
    void output_result(double* P, double* dual, bool output_on_device = false)
    {
        nvtx3::scoped_range r{"output_result"};

        // Copy (alpha, beta) to dual
        if (dual != nullptr)
        {
            CUDA_CHECK(cudaMemcpy(
                dual, d_gamma, (m_n + m_m) * sizeof(double),
                output_on_device ? cudaMemcpyDeviceToDevice : cudaMemcpyDeviceToHost
            ));
        }

        // In case P is nullptr
        if (P == nullptr)
        {
            return;
        }

        // Compute final transport plan
        // If P is on device, directly write to P
        // Otherwise, since d_work is no longer used,
        // and it has at least n*m elements, we can use d_work
        // to hold transport plan
        double* d_P = output_on_device ? P : d_work;
        compute_transport_plan(d_M, d_alpha, d_beta, d_P, m_reg, m_n, m_m);

        // Copy result back to host if output_on_device = false
        if (!output_on_device)
        {
            CUDA_CHECK(cudaMemcpy(P, d_P, m_Me * sizeof(double), cudaMemcpyDeviceToHost));
        }
    }
};


// CUDA implementation of SPLR algorithm for entropic-regularized OT
// input_on_device = true means that M, a, b, and x0 (if not nullptr) are device pointers
// output_on_device = true means that P and dual (if not nullptr) are device pointers
void cuda_sinkhorn_splr(
    const double* M, const double* a, const double* b, double* P,
    double reg, int max_iter, double tol, int n, int m, int* niter,
    double density_max, double shift_max,
    int sparsity_pattern_cycle, int candidate_sinkhorn_iter, int warm_start_iter, int verbose,
    const double* x0, double* dual,
    bool input_on_device, bool output_on_device
)
{
    // Algorithmic parameters
    // density
    density_max = std::min(density_max, 1.0);
    density_max = std::max(density_max, 0.0);
    const double density_min = 0.01 * density_max;
    double density = 0.5 * density_max;
    // shift
    double shift = shift_max;
    // Kmax -- maximum number of nonzero elements in sparsified T_t
    size_t Kmax = static_cast<size_t>(density_max * n * (m - 1));
    Kmax = std::max(Kmax, size_t(1));
    // sparsity_pattern_cycle -- cycle length of reusing sparsity pattern
    // const int sparsity_pattern_cycle = 30;
    // candidate_sinkhorn_iter -- number of Sinkhorn iterations used to generate candidate iterate
    //                            to potentially accelerate convergence
    // const int candidate_sinkhorn_iter = 3;

    // Create solver object
    SPLRSolver solver(M, a, b, reg, n, m, Kmax, input_on_device);

    // Initialize dual variables
    solver.init_dual(x0, input_on_device);

    // BCD warm-start: run a few Sinkhorn iterations before the main SPLR loop
    // to get a better initial dual solution
    if (warm_start_iter > 0) {
        for (int i = 0; i < warm_start_iter; i++) {
            solver.update_beta();
            solver.update_alpha();
        }
        cudaDeviceSynchronize();
    }

    // Initial objective function value (f), gradient (g),
    // and sparsified Hessian (sphess)
    double objfn, gnorm;
    // Only compute f and g
    constexpr bool compute_gnorm = true;
    solver.dual_T_objfn_grad(objfn, gnorm, compute_gnorm, false);
    // ||grad|| will be used for updating shift
    double gnorm_init = gnorm;
    shift = std::min(gnorm, shift_max);
    // Then continue computing sphess by
    // Note: shift should be applied to Hessian, but what we compute is Hsl = H * reg
    // Therefore, here we multiply shift by reg before adding to Hsl
    size_t nnz = solver.dual_sphess(density, shift * reg, false);

    // Main iteration
    // Initial step size
    double alpha = 1.0;
    // Timer
    Timer timer_inner(verbose >= 2);
    for (int iter = 0; iter < max_iter; iter++)
    {
        if (verbose >= 1)
        {
            std::cout << "iter = " << iter << ", objval = " << objfn <<
                ", ||grad|| = " << gnorm << std::endl;
        }

        // Start timing
        timer_inner.tic();

        // Convergence test
        // Also exit if objective function value is not finite
        if ((gnorm < tol) || (!std::isfinite(objfn)))
            break;

        // Compute y = grad - grad_prev and s = gamma - gamma_prev
        // ys = y's, yy = y'y
        double ys, yy;
        solver.compute_low_rank(ys, yy);
        timer_inner.toc("low_rank");

        // Compute search direction
        // Sparse Cholesky decomposition and solving + BFGS update rule
        //
        // We do not do low-rank update in the first iteration
        // When <y, s> is too small, don't use low-rank update
        constexpr double eps = 1e-6;  // Or use std::numeric_limits<double>::epsilon();
        const bool low_rank = (iter > 0) && (ys > (eps * yy));
        // Sparse Cholesky decomposition has three main stages:
        // sparsity pattern analysis, factorization, and solving
        //
        // Pattern analysis runs on CPU, and may be time-consuming,
        // so we can fix the pattern for a few iterations and cyclically update it
        //
        // Below are flags to indicate whether we need to analyze or update the sparsity pattern
        // in the current iteration (basically we update the pattern every `sparsity_pattern_cycle` iterations
        // and do the sparsity analysis in the next iteration)
        const bool analyze_pattern = (iter % sparsity_pattern_cycle == 0);
        const bool update_pattern = (iter % sparsity_pattern_cycle == (sparsity_pattern_cycle - 1));

        // When we need to analyze the sparsity pattern in this iteration,
        // we also compute the Sinkhorn iterate, since the majority of the pattern analysis
        // runs on CPU, and we can let GPU work to fill this idle time. In this way,
        // we do not increase the wall time, but can obtain some useful information about the
        // next move
        //
        // The Sinkhorn iterate is a candidate for the next move. For example, we can
        // compare the objective function value and gradient of both the Sinkhorn iterate
        // and the quasi-Newton iterate, and then decide which one is the next move
        if (analyze_pattern && (candidate_sinkhorn_iter != 0))
        {
            // This part will run roughly at the same time as the analysis stage of
            // the sparse Cholesky decomposition
            solver.compute_sinkhorn_iterate(candidate_sinkhorn_iter);
        }
        solver.compute_search_direc(nnz, ys, low_rank, analyze_pattern, verbose);
        timer_inner.toc("search_direc");

        // Line search will overwrite d_gamma and d_grad, so
        // save d_gamma to d_gamma_prev, and d_grad to d_grad_prev
        solver.save_history();

        // Let Sinkhorn and quasi-Newton computation synchronize here
        // CUDA_CHECK(cudaDeviceSynchronize());
        // In fact, the cudaMemcpy() function in solver.save_history() will implicitly synchronize,
        // so we do not need to explicitly call cudaDeviceSynchronize() here

        // Wolfe Line Search
        // Will overwrite d_gamma and d_grad
        // If the updated recompute_fg is false, then the overwritten d_gamma
        // is the new point, and objfn and d_grad contain the corresponding
        // objective function value and gradient, respectively
        // Otherwise, we need to recompute d_gamma, objfn, and d_grad
        bool recompute_fg = true;
        alpha = solver.line_search_wolfe(
            std::min(1.0, 1.5 * alpha), objfn, objfn, recompute_fg, !update_pattern
        );
        timer_inner.toc("line_search");

        // Recompute the new point if needed, and compute the new gradient norm
        const double gnorm_pre = gnorm;
        if (recompute_fg)
        {
            // d_gamma = d_gamma_prev + alpha * direc
            solver.update_gamma(alpha);
            // Compute f and g on new point d_gamma
            // Also compute gradient norm
            solver.dual_T_objfn_grad(objfn, gnorm, true, !update_pattern);        }
        else
        {
            // f and g have been computed in line search,
            // so we only compute gradient norm here
            gnorm = solver.grad_norm();
        }
        timer_inner.toc("grad");

        // If analyze_pattern is true, it means that we have also computed the Sinkhorn iterate
        // If it is better than the quasi-Newton direction (both objfn and gnorm are smaller),
        // then we overwrite gamma with the Sinkhorn iterate
        if (analyze_pattern && (candidate_sinkhorn_iter != 0))
        {
            double objfn_bcd, gnorm_bcd;
            solver.objfn_gnorm_bcd(objfn_bcd, gnorm_bcd);
            // const bool use_sinkhorn_iter = ((objfn_bcd < objfn) && (gnorm_bcd < gnorm));
            const bool use_sinkhorn_iter = ((objfn_bcd < objfn) && (gnorm_bcd < 10 * gnorm));

            if (verbose >= 2)
            {
                std::cout << "[sinkhorn] ------------------------------------------------" << std::endl;
                std::cout << "║ objfn = " << objfn << ", objfn_bcd = " << objfn_bcd << std::endl;
                std::cout << "║ gnorm = " << gnorm << ", gnorm_bcd = " << gnorm_bcd << std::endl;
                std::cout << "║ use_sinkhorn_iter = " << use_sinkhorn_iter << std::endl;
                std::cout << "===========================================================" << std::endl;
            }

            if (use_sinkhorn_iter)
            {
                // Overwrite d_gamma with the Sinkhorn iterate
                solver.update_gamma_sinkhorn();
                // Recompute objfn and gnorm
                solver.dual_T_objfn_grad(objfn, gnorm, true, !update_pattern);
            }
        }

        // Adjust density according to gnorm change
        if (update_pattern)
        {
            const bool bad_move = (gnorm_pre < gnorm_init) && (gnorm > gnorm_pre);
            density *= (bad_move ? 1.1 : 0.99);
            density = std::min(density_max, std::max(density_min, density));
        }

        // Compute sphess
        shift = std::min(gnorm, shift_max);
        nnz = solver.dual_sphess(density, shift * reg, !update_pattern);
        timer_inner.toc("sphess");

        if (verbose >= 2)
        {
            std::cout << "[lowrank]--------------------------------------------------" << std::endl;
            std::cout << "║ ys = " << ys << ", yy = " << yy << std::endl;
            std::cout << "║ low_rank = " << low_rank << ", analyze = " << analyze_pattern <<
                ", update = " << update_pattern << std::endl;
            std::cout << "===========================================================" << std::endl;
            std::cout << "[timing]---------------------------------------------------" << std::endl;
            std::cout << "║ low_rank = " << timer_inner["low_rank"] <<
                ", search_direc = " << timer_inner["search_direc"] << std::endl;
            std::cout << "║ line_search = " << timer_inner["line_search"] << std::endl;
            std::cout << "║ grad = " << timer_inner["grad"] <<
                ", sphess = " << timer_inner["sphess"] << std::endl;
            std::cout << "===========================================================" << std::endl << std::endl;
        }

        *niter = iter + 1;
    }

    // Final Sinkhorn iteration
    // solver.update_beta();
    // solver.update_alpha();

    // Compute final transport plan and output results to host
    solver.output_result(P, dual, output_on_device);
    CUDA_CHECK(cudaDeviceSynchronize());
}
