"""
    Benchmark script for Sparse GP-FVM

Profiles each step of the GP-FVM solver to identify bottlenecks.
"""

using GPFiniteVolume
using FunctionalGPs, GaussianMarkovRandomFields
using LinearAlgebra, SparseArrays
using Kronecker
using SparseConnectivityTracer, SparseMatrixColorings
using Printf

import GaussianMarkovRandomFields: mean, std, precision_matrix
import DifferentiationInterface as DI

include("problem.jl")

function benchmark_sparse_fvm(;
    N::Int = 100,
    n_timesteps::Int = 50,
    ρ::Float64 = 2.0,
    lengthscale_factor::Float64 = 3.0,
    smoothness::Int = 2,
    run_ekf::Bool = true,
    run_joint_solve::Bool = true
)
    println("="^70)
    println("Benchmark: Sparse GP-FVM")
    println("="^70)
    println("N = $N, n_timesteps = $n_timesteps, ρ = $ρ")
    println()

    timings = Dict{String, Float64}()

    # Problem setup
    problem = BurgersProblem(
        x_min = 0.0, x_max = 1.0, T_end = 0.3, ν = 0.01,
        u_left = 0.0, u_right = 0.0
    )
    ic = InitialCondition(:sine, 42)
    ν = problem.ν

    # =========================================================================
    # Step 1: Spatial setup
    # =========================================================================
    t0 = time()

    endpoints = range(problem.x_min, problem.x_max, length=N)
    intervals = intervals_from_endpoints(collect(endpoints))
    N_int = length(intervals)
    Δx = endpoints[2] - endpoints[1]
    n_t = n_timesteps
    Δt = problem.T_end / n_t

    timings["1. Spatial setup"] = time() - t0

    # =========================================================================
    # Step 2: Build sparse spatial precision
    # =========================================================================
    t0 = time()

    lengthscale = lengthscale_factor * Δx
    k = HalfIntegerMaternKernel(smoothness, [lengthscale])
    approx = sparse_precision([
        :f => EvaluationFunctional(endpoints),
        :f_dx => EvaluationFunctional(endpoints) ∘ PartialDerivative((1,)),
        :f_int => VectorizedLebesgueIntegral(intervals)
    ], k; ρ=ρ, ordering=:integrals_coarsest)

    Q_space = approx.Q
    N_space = size(Q_space, 1)

    timings["2. Sparse precision"] = time() - t0

    println("   N_space = $N_space, nnz(Q_space) = $(nnz(Q_space))")

    # =========================================================================
    # Step 3: Build initial state GMRF
    # =========================================================================
    t0 = time()

    Q_state0 = [Q_space spzeros(size(Q_space)...);
                spzeros(size(Q_space)...) Q_space]
    N_state = size(Q_state0, 1)
    x0 = GMRF(zeros(N_state), Q_state0)

    full_state_layout = layout((
        f = N, f_dx = N, f_int = N_int,
        df_dt = N, df_dx_dt = N, df_int_dt = N_int
    ))

    timings["3. Initial GMRF"] = time() - t0

    # =========================================================================
    # Step 4: Apply initial conditions
    # =========================================================================
    t0 = time()

    ys = [ic(x) for x in endpoints]
    ys_dx = [evaluate_dx(ic, x) for x in endpoints]
    ys_int = [evaluate_int(ic, endpoints[i], endpoints[i+1]) for i in 1:N_int]

    x0_ic = prescribe_indices(x0, indices(full_state_layout, :f), ys)
    x0_ic = prescribe_indices(x0_ic, indices(full_state_layout, :f_dx), ys_dx)
    x0_ic = prescribe_indices(x0_ic, indices(full_state_layout, :f_int), ys_int)

    timings["4. Apply IC"] = time() - t0

    # =========================================================================
    # Step 5: FVM constraint at t=0
    # =========================================================================
    t0 = time()

    function f_fvm_t0(x)
        x_s = State(x, full_state_layout)
        dint_dt = x_s.df_int_dt
        u_left = x_s.f[1:end-1]
        u_right = x_s.f[2:end]
        ux_left = x_s.f_dx[1:end-1]
        ux_right = x_s.f_dx[2:end]
        F_left = 0.5 * u_left.^2 - ν * ux_left
        F_right = 0.5 * u_right.^2 - ν * ux_right
        return dint_dt + (F_right - F_left)
    end

    fvm_model_0 = NonlinearLeastSquaresModel(f_fvm_t0, length(x0_ic))
    y_fvm_0 = zeros(N_int)
    lik_fvm_0 = fvm_model_0(y_fvm_0; σ=0.0001)

    timings["5a. FVM t=0 model setup"] = time() - t0

    t0 = time()
    x_fvm_0 = gaussian_approximation(x0_ic, lik_fvm_0)
    timings["5b. FVM t=0 solve"] = time() - t0

    # =========================================================================
    # Step 6: Time stepping setup
    # =========================================================================
    t0 = time()

    sde = IWPSDE(1.0)
    A_t, Σ_noise_t = discretize_vanloan(sde, Δt)
    Q_noise_t = inv(Σ_noise_t)
    Q_noise_t = 0.5 * (Q_noise_t + Q_noise_t')

    A = kronecker(A_t, sparse(I, N_space, N_space))
    Q_noise = kronecker(Q_noise_t, Q_space)

    timings["6. Time stepping setup"] = time() - t0

    # =========================================================================
    # Step 7: Build joint spacetime GMRF
    # =========================================================================
    t0 = time()

    ssm = ConstantLinearGaussianSSM(x_fvm_0, A, Q_noise)
    x_joint = GPFiniteVolume.joint_gmrf(ssm, n_t)

    timings["7. Joint GMRF"] = time() - t0

    joint_size = length(mean(x_joint))
    println("   Joint size = $joint_size")

    # =========================================================================
    # Step 8: Boundary conditions
    # =========================================================================
    t0 = time()

    timest = TimeStack(mean(x_joint), full_state_layout)
    bc_left = problem.u_left
    bc_right = problem.u_right
    ys_bc = repeat([bc_left, bc_right], n_t - 1)
    bc_indices = absindices(timest, :f, [1, N], 2:n_t)
    x_joint_bc = prescribe_indices(x_joint, bc_indices, ys_bc)

    timings["8. Boundary conditions"] = time() - t0

    # =========================================================================
    # Step 9: EKF sweep initialization
    # =========================================================================
    if run_ekf
        t0 = time()

        Q_noise_sparse = sparse(collect(Q_noise))
        means = Vector{Vector{Float64}}(undef, n_t)
        means[1] = mean(x_fvm_0)

        ekf_step_times = Float64[]

        for t in 2:n_t
            t_step = time()

            μ_prior = collect(A * means[t-1])
            x_prior = GMRF(μ_prior, Q_noise_sparse)

            A_bc_mat = zeros(2, length(μ_prior))
            A_bc_mat[1, 1] = 1.0
            A_bc_mat[2, N] = 1.0
            x_bc = ConstrainedGMRF(x_prior, A_bc_mat, [bc_left, bc_right])

            function f_fvm_block(x)
                x_c = State(x, full_state_layout)
                x_p = State(means[t-1], full_state_layout)
                F_c = 0.5 * x_c.f.^2 - ν * x_c.f_dx
                F_p = 0.5 * x_p.f.^2 - ν * x_p.f_dx
                dF_c = F_c[2:end] - F_c[1:end-1]
                dF_p = F_p[2:end] - F_p[1:end-1]
                return x_c.df_int_dt + 0.5 * (dF_c + dF_p)
            end

            model = NonlinearLeastSquaresModel(f_fvm_block, length(x_bc))
            lik = model(zeros(N_int); σ=0.0001)
            x_sol = gaussian_approximation(x_bc, lik; verbose=false)
            means[t] = mean(x_sol)

            push!(ekf_step_times, time() - t_step)
        end

        init_vec = vcat(means...)
        timings["9. EKF sweep (total)"] = time() - t0

        avg_step = sum(ekf_step_times) / length(ekf_step_times)
        println("   EKF: $(n_t-1) steps, avg $(round(avg_step*1000, digits=2)) ms/step")
    else
        init_vec = mean(x_joint_bc)
        timings["9. EKF sweep (skipped)"] = 0.0
    end

    # =========================================================================
    # Step 10: Setup joint FVM constraint
    # =========================================================================
    t0 = time()

    f_left_curr = absindices(timest, :f, 1:(N-1), 2:n_t)
    f_right_curr = absindices(timest, :f, 2:N, 2:n_t)
    f_dx_left_curr = absindices(timest, :f_dx, 1:(N-1), 2:n_t)
    f_dx_right_curr = absindices(timest, :f_dx, 2:N, 2:n_t)
    df_int_dt_curr = absindices(timest, :df_int_dt, 1:(N-1), 2:n_t)

    f_left_prev = absindices(timest, :f, 1:(N-1), 1:(n_t-1))
    f_right_prev = absindices(timest, :f, 2:N, 1:(n_t-1))
    f_dx_left_prev = absindices(timest, :f_dx, 1:(N-1), 1:(n_t-1))
    f_dx_right_prev = absindices(timest, :f_dx, 2:N, 1:(n_t-1))

    function f_fvm_crank_nicolson(x)
        dint_dt = x[df_int_dt_curr]
        F_L_curr = 0.5 * x[f_left_curr].^2 - ν * x[f_dx_left_curr]
        F_R_curr = 0.5 * x[f_right_curr].^2 - ν * x[f_dx_right_curr]
        F_L_prev = 0.5 * x[f_left_prev].^2 - ν * x[f_dx_left_prev]
        F_R_prev = 0.5 * x[f_right_prev].^2 - ν * x[f_dx_right_prev]
        net_flux_avg = 0.5 * ((F_R_curr - F_L_curr) + (F_R_prev - F_L_prev))
        return dint_dt + net_flux_avg
    end

    fvm_model = NonlinearLeastSquaresModel(f_fvm_crank_nicolson, length(x_joint_bc))
    y_fvm = zeros(length(f_left_curr))
    lik_fvm = fvm_model(y_fvm; σ=0.0001)

    timings["10a. Joint FVM model setup"] = time() - t0

    t0 = time()

    A_constr = zeros(1, length(x_joint_bc))
    A_constr[1] = 1.0
    e = [mean(x_joint_bc)[1]]

    timings["10b1. A_constr setup"] = time() - t0

    t0 = time()
    Q_joint = precision_matrix(x_joint_bc)
    timings["10b2. precision_matrix(x_joint_bc)"] = time() - t0

    println("   nnz(Q_joint) = $(nnz(Q_joint.data))")
    println("   type(x_joint_bc) = $(typeof(x_joint_bc))")

    t0 = time()
    x_init = GMRF(init_vec, Q_joint)
    timings["10b3. GMRF(init_vec, Q_joint)"] = time() - t0

    t0 = time()
    x_constr = ConstrainedGMRF(x_init, A_constr, e)
    timings["10b4. ConstrainedGMRF"] = time() - t0

    # =========================================================================
    # Step 11: Joint solve
    # =========================================================================
    if run_joint_solve
        # First, let's profile what happens in one Jacobian evaluation
        println("\n   Profiling Jacobian computation...")

        t0 = time()
        J = DI.jacobian(lik_fvm.f, lik_fvm.jac_backend, init_vec)
        t_jac = time() - t0
        println("   Jacobian size: $(size(J))")
        println("   Jacobian nnz: $(nnz(J)) ($(round(100*nnz(J)/prod(size(J)), digits=2))% fill)")
        println("   Jacobian time: $(round(t_jac, digits=3)) s")

        t0 = time()
        H = J' * Diagonal(lik_fvm.inv_σ²) * J
        t_hess = time() - t0
        println("   J'DJ time: $(round(t_hess, digits=3)) s")
        println("   H nnz: $(nnz(H)) ($(round(100*nnz(H)/prod(size(H)), digits=2))% fill)")

        timings["11a. Single Jacobian"] = t_jac
        timings["11b. Single J'DJ"] = t_hess

        println("\n   Running full solve...")
        t0 = time()
        x_solution = gaussian_approximation(x_constr, lik_fvm; verbose=true)
        timings["11c. Full solve"] = time() - t0
    else
        timings["11. Joint solve (skipped)"] = 0.0
    end

    # =========================================================================
    # Summary
    # =========================================================================
    println()
    println("="^70)
    println("Timing Summary")
    println("="^70)

    total = 0.0
    for (name, t) in sort(collect(timings), by=x->x[1])
        @printf("  %-35s %8.3f s\n", name, t)
        total += t
    end
    println("-"^50)
    @printf("  %-35s %8.3f s\n", "TOTAL", total)

    return timings
end

# Run with default settings
if abspath(PROGRAM_FILE) == @__FILE__
    # Warmup
    println("Warmup run...")
    benchmark_sparse_fvm(N=30, n_timesteps=10, run_ekf=false, run_joint_solve=false)
    println()

    # Actual benchmark
    println("\n" * "="^70)
    println("MAIN BENCHMARK")
    println("="^70 * "\n")

    benchmark_sparse_fvm(N=100, n_timesteps=50)
end
