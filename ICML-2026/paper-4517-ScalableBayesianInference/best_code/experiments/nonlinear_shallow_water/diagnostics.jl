"""
Memory and sparsity diagnostics for nonlinear SWE solver.

Run with:
    julia --project=. experiments/nonlinear_shallow_water/diagnostics.jl --nx 15 --nsteps 5
"""

using GPFiniteVolume
using FunctionalGPs, GaussianMarkovRandomFields
using LinearAlgebra, SparseArrays
using Kronecker: kronecker
using Unitful
using SparseConnectivityTracer, SparseMatrixColorings
using Zygote
import DifferentiationInterface as DI

import GaussianMarkovRandomFields: mean, std, precision_matrix, linear_condition
import FunctionalGPs: ⊗
using GaussianMarkovRandomFields: NonlinearLeastSquaresModel

include("problem.jl")
include("functionals.jl")
include("constraints.jl")
include("solver.jl")

# ------------------------------------------------------------------------------
# Memory utilities
# ------------------------------------------------------------------------------

"""Format bytes as human-readable string."""
function format_bytes(bytes)
    if bytes < 1024
        return "$(bytes) B"
    elseif bytes < 1024^2
        return "$(round(bytes/1024, digits=1)) KB"
    elseif bytes < 1024^3
        return "$(round(bytes/1024^2, digits=1)) MB"
    else
        return "$(round(bytes/1024^3, digits=2)) GB"
    end
end

"""Estimate memory for sparse matrix (CSC format)."""
function sparse_memory(A::SparseMatrixCSC)
    # CSC format: colptr (n+1 Int), rowval (nnz Int), nzval (nnz Float64)
    n = size(A, 2)
    nz = nnz(A)
    return (n + 1) * sizeof(Int) + nz * sizeof(Int) + nz * sizeof(eltype(A))
end
sparse_memory(A::Symmetric) = sparse_memory(parent(A))

"""Get nnz handling Symmetric wrapper."""
get_nnz(A::SparseMatrixCSC) = nnz(A)
get_nnz(A::Symmetric) = nnz(parent(A))

"""Unwrap Symmetric if needed."""
unwrap(A::Symmetric) = parent(A)
unwrap(A) = A

"""Estimate memory for dense matrix."""
dense_memory(A) = length(A) * sizeof(eltype(A))

"""Estimate Cholesky factor fill-in ratio."""
function cholesky_fillin(Q)
    try
        F = cholesky(Symmetric(unwrap(Q)))
        L = sparse(F.L)
        return nnz(L) / get_nnz(Q)
    catch e
        println("  Cholesky failed: $e")
        return NaN
    end
end

# ------------------------------------------------------------------------------
# Diagnostic run
# ------------------------------------------------------------------------------

function run_diagnostics(;
        N_x = 15,
        N_y = 15,
        n_timesteps = 5,
        Δt = 1.0u"minute",
        lengthscale = 10.0u"km",
        flux_lengthscale = 10.0u"km",
        smoothness = 2,
        ρ = 2.0,
        σ_fvm = 1e-4,
        σ_flux = 1e-4,
        prob = NonlinearSWEProblem(),
        units = SolveUnits()
    )

    println("=" ^ 70)
    println("MEMORY DIAGNOSTICS: Nonlinear SWE Solver")
    println("=" ^ 70)
    println("\nConfiguration:")
    println("  Grid: $(N_x) × $(N_y)")
    println("  Timesteps: $n_timesteps")
    println("  ρ (sparsity): $ρ")

    # Convert units
    p = to_solve_params(prob, units)
    L = units.length
    T = units.time
    Δt_solve = ustrip(T, uconvert(T, Δt))
    lengthscale_solve = ustrip(L, uconvert(L, lengthscale))
    flux_lengthscale_solve = ustrip(L, uconvert(L, flux_lengthscale))

    # -------------------------------------------------------------------------
    # Stage 1: Grid and functionals
    # -------------------------------------------------------------------------
    println("\n" * "=" ^ 70)
    println("STAGE 1: Grid and Functionals")
    println("=" ^ 70)

    xs, ys, x_intervals, y_intervals = setup_2d_grid(N_x, N_y, p.L_x, p.L_y)
    ginfo = GridInfo(xs, ys)

    println("\nGrid info:")
    println("  Grid points: $(ginfo.N_grid)")
    println("  Cells: $(ginfo.N_cells)")
    println("  Vertical faces: $(ginfo.n_vert_faces)")
    println("  Horizontal faces: $(ginfo.n_horiz_faces)")

    funcs = build_functionals(xs, ys, x_intervals, y_intervals)

    k_primary = HalfIntegerMaternKernel(smoothness, [lengthscale_solve]) ⊗
                HalfIntegerMaternKernel(smoothness, [lengthscale_solve])
    k_flux = HalfIntegerMaternKernel(smoothness, [flux_lengthscale_solve]) ⊗
             HalfIntegerMaternKernel(smoothness, [flux_lengthscale_solve])

    prec_info = build_all_precisions(funcs, k_primary, k_flux, ρ; verbose=false)

    N_space = prec_info.N_space
    println("\nSpatial state dimension: $N_space")
    println("Full IWP state (with derivs): $(2 * N_space)")
    println("Total spacetime dimension: $(2 * N_space * n_timesteps)")

    # -------------------------------------------------------------------------
    # Stage 2: Spatial precision analysis
    # -------------------------------------------------------------------------
    println("\n" * "=" ^ 70)
    println("STAGE 2: Spatial Precision Matrix")
    println("=" ^ 70)

    Q_space = prec_info.Q_space
    println("\nQ_space (spatial precision):")
    println("  Size: $(size(Q_space))")
    println("  nnz: $(get_nnz(Q_space))")
    println("  Fill ratio: $(round(get_nnz(Q_space) / prod(size(Q_space)) * 100, digits=3))%")
    println("  Memory: $(format_bytes(sparse_memory(Q_space)))")

    fillin = cholesky_fillin(Q_space)
    println("  Cholesky fill-in ratio: $(round(fillin, digits=2))×")

    # -------------------------------------------------------------------------
    # Stage 3: Joint spacetime GMRF
    # -------------------------------------------------------------------------
    println("\n" * "=" ^ 70)
    println("STAGE 3: Joint Spacetime GMRF")
    println("=" ^ 70)

    # Build IWP discretization
    sde = IWPSDE(1.0)
    A_t, Σ_noise_t = discretize_vanloan(sde, Δt_solve)
    Q_noise_t = inv(Σ_noise_t)
    Q_noise_t = 0.5 * (Q_noise_t + Q_noise_t')

    # Initial state
    Q_state0 = blockdiag(prec_info.Q_space, prec_info.Q_space)
    x0 = GMRF(zeros(2 * N_space), Symmetric(Q_state0))

    # Kronecker products
    A = kronecker(A_t, sparse(I, N_space, N_space))
    Q_noise = kronecker(Q_noise_t, prec_info.Q_space)

    # Build joint GMRF
    full_state_layout = build_full_state_layout(ginfo)

    # Apply ICs first
    A_ic, y_ic = build_ic_observations(p, funcs, ginfo, full_state_layout)
    n_ic = length(y_ic)
    Q_ϵ_ic = (1.0 / 1e-6^2) * sparse(I, n_ic, n_ic)
    x0_ic = linear_condition(x0; A=A_ic, Q_ϵ=Q_ϵ_ic, y=y_ic)

    ssm = ConstantLinearGaussianSSM(x0_ic, A, Q_noise)
    x_joint = GPFiniteVolume.joint_gmrf(ssm, n_timesteps)

    Q_joint = precision_matrix(x_joint)
    N_total = size(Q_joint, 1)

    println("\nQ_joint (spacetime precision):")
    println("  Size: $(size(Q_joint))")
    println("  nnz: $(get_nnz(Q_joint))")
    println("  Fill ratio: $(round(get_nnz(Q_joint) / prod(size(Q_joint)) * 100, digits=4))%")
    println("  Memory: $(format_bytes(sparse_memory(Q_joint)))")

    # Check block structure
    block_size = 2 * N_space
    Q_joint_raw = unwrap(Q_joint)
    println("\n  Block structure ($(n_timesteps) blocks of size $(block_size)):")
    for t in 1:min(3, n_timesteps)
        r1, r2 = (t-1)*block_size+1, t*block_size
        block = Q_joint_raw[r1:r2, r1:r2]
        println("    Block $t diagonal: nnz = $(nnz(block))")
        if t < n_timesteps
            c1, c2 = t*block_size+1, (t+1)*block_size
            offdiag = Q_joint_raw[r1:r2, c1:c2]
            println("    Block $t→$(t+1) off-diag: nnz = $(nnz(offdiag))")
        end
    end

    println("\n  Attempting Cholesky on Q_joint...")
    GC.gc()
    mem_before = Sys.free_memory()
    t_chol = @elapsed fillin_joint = cholesky_fillin(Q_joint)
    mem_after = Sys.free_memory()
    println("  Cholesky fill-in ratio: $(round(fillin_joint, digits=2))×")
    println("  Cholesky time: $(round(t_chol, digits=2))s")
    println("  Memory used: ~$(format_bytes(mem_before - mem_after))")

    # -------------------------------------------------------------------------
    # Stage 4: FVM constraint matrix
    # -------------------------------------------------------------------------
    println("\n" * "=" ^ 70)
    println("STAGE 4: FVM Constraint Matrix")
    println("=" ^ 70)

    timest = TimeStack(mean(x_joint), full_state_layout)
    fvm_time_range = 1:(n_timesteps - 1)
    A_fvm, b_fvm = build_fvm_constraint_matrix(timest, ginfo, p, fvm_time_range; crank_nicolson=true)

    println("\nA_fvm (FVM constraint matrix):")
    println("  Size: $(size(A_fvm)) (constraints × state)")
    println("  nnz: $(nnz(A_fvm))")
    println("  nnz per row: $(round(nnz(A_fvm) / size(A_fvm, 1), digits=1))")
    println("  Memory: $(format_bytes(sparse_memory(A_fvm)))")

    # A_fvm' * Q_ε * A_fvm contribution
    Q_ε_fvm = (1.0 / σ_fvm^2) * sparse(I, size(A_fvm, 1), size(A_fvm, 1))
    println("\n  Computing A_fvm' * Q_ε * A_fvm...")
    t_ata = @elapsed AtQA_fvm = A_fvm' * Q_ε_fvm * A_fvm
    println("  Result: nnz = $(nnz(AtQA_fvm))")
    println("  Time: $(round(t_ata, digits=3))s")

    # -------------------------------------------------------------------------
    # Stage 5: Flux constraint Jacobian
    # -------------------------------------------------------------------------
    println("\n" * "=" ^ 70)
    println("STAGE 5: Nonlinear Flux Constraint Jacobian")
    println("=" ^ 70)

    flux_time_range = 1:n_timesteps
    flux_idx = build_flux_constraint_indices(timest, ginfo, flux_time_range)
    n_flux = 8 * length(flux_idx.h_vert_face)

    println("\nFlux constraints:")
    println("  Number of constraints: $n_flux")
    println("  State dimension: $N_total")

    # Build residual function
    residual_fn = x -> flux_residual(x, flux_idx, p.g)

    # Test residual
    x_test = mean(x_joint)
    println("\n  Testing residual function...")
    t_res = @elapsed r_test = residual_fn(x_test)
    println("  Residual size: $(length(r_test))")
    println("  Residual time: $(round(t_res * 1000, digits=2))ms")
    println("  Residual norm: $(round(norm(r_test), sigdigits=3))")

    # Compute sparse Jacobian
    println("\n  Computing sparse Jacobian...")
    jac_backend = GaussianMarkovRandomFields.default_sparse_jacobian_backend()

    GC.gc()
    mem_before = Sys.free_memory()
    t_jac = @elapsed J = DI.jacobian(residual_fn, jac_backend, x_test)
    mem_after = Sys.free_memory()

    println("  Jacobian size: $(size(J))")
    println("  Jacobian nnz: $(nnz(J))")
    println("  nnz per row: $(round(nnz(J) / size(J, 1), digits=2))")
    println("  Jacobian memory: $(format_bytes(sparse_memory(J)))")
    println("  Jacobian compute time: $(round(t_jac, digits=2))s")
    println("  Memory used: ~$(format_bytes(mem_before - mem_after))")

    # J' * J analysis
    println("\n  Computing J' * J...")
    GC.gc()
    mem_before = Sys.free_memory()
    t_jtj = @elapsed JtJ = J' * J
    mem_after = Sys.free_memory()

    println("  J'J size: $(size(JtJ))")
    println("  J'J nnz: $(nnz(JtJ))")
    println("  J'J fill ratio: $(round(nnz(JtJ) / prod(size(JtJ)) * 100, digits=4))%")
    println("  J'J memory: $(format_bytes(sparse_memory(JtJ)))")
    println("  J'J compute time: $(round(t_jtj, digits=2))s")
    println("  Memory used: ~$(format_bytes(mem_before - mem_after))")

    # -------------------------------------------------------------------------
    # Stage 6: Combined precision analysis
    # -------------------------------------------------------------------------
    println("\n" * "=" ^ 70)
    println("STAGE 6: Combined Precision Matrix")
    println("=" ^ 70)

    # After FVM conditioning
    println("\n  Applying FVM constraints via linear_condition...")
    GC.gc()
    mem_before = Sys.free_memory()
    t_fvm_cond = @elapsed x_joint_fvm = linear_condition(x_joint; A=A_fvm, Q_ϵ=Q_ε_fvm, y=b_fvm)
    mem_after = Sys.free_memory()

    Q_after_fvm = precision_matrix(x_joint_fvm)
    println("  Q after FVM: nnz = $(get_nnz(Q_after_fvm))")
    println("  Time: $(round(t_fvm_cond, digits=2))s")
    println("  Memory used: ~$(format_bytes(mem_before - mem_after))")

    # Combined with J'J (simulating one Newton step)
    println("\n  Simulating Newton step: Q_combined = Q_fvm + J'J/σ²...")
    inv_σ²_flux = 1.0 / σ_flux^2
    GC.gc()
    mem_before = Sys.free_memory()
    t_combine = @elapsed Q_combined = unwrap(Q_after_fvm) + inv_σ²_flux * JtJ
    mem_after = Sys.free_memory()

    println("  Q_combined nnz: $(get_nnz(Q_combined))")
    println("  Q_combined memory: $(format_bytes(sparse_memory(Q_combined)))")
    println("  Time: $(round(t_combine, digits=3))s")

    # Cholesky of combined
    println("\n  Attempting Cholesky on Q_combined...")
    GC.gc()
    mem_before = Sys.free_memory()
    t_chol_combined = @elapsed fillin_combined = cholesky_fillin(Q_combined)
    mem_after = Sys.free_memory()

    println("  Cholesky fill-in ratio: $(round(fillin_combined, digits=2))×")
    println("  Cholesky time: $(round(t_chol_combined, digits=2))s")
    println("  Memory used: ~$(format_bytes(mem_before - mem_after))")

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    println("\n" * "=" ^ 70)
    println("SUMMARY")
    println("=" ^ 70)

    println("\nKey metrics:")
    println("  State dimension: $N_total")
    println("  Flux constraints: $n_flux")
    println("  Q_joint nnz: $(get_nnz(Q_joint)) → Cholesky $(round(fillin_joint, digits=1))×")
    println("  Q_combined nnz: $(get_nnz(Q_combined)) → Cholesky $(round(fillin_combined, digits=1))×")

    # Estimate for larger grid
    scale_factor = (21/N_x)^2 * (9/n_timesteps)
    println("\nProjected for 21×21 × 9 timesteps ($(round(scale_factor, digits=1))× larger):")
    println("  State dimension: ~$(round(Int, N_total * scale_factor))")
    println("  Q_combined nnz: ~$(round(Int, get_nnz(Q_combined) * scale_factor))")

    # Cholesky memory scales roughly as nnz(L)
    # nnz(L) ≈ fillin_combined * nnz(Q_combined)
    estimated_L_nnz = fillin_combined * get_nnz(Q_combined) * scale_factor
    println("  Estimated Cholesky nnz: ~$(round(Int, estimated_L_nnz))")
    println("  Estimated Cholesky memory: ~$(format_bytes(round(Int, estimated_L_nnz * 8)))")

    return (
        N_total = N_total,
        n_flux = n_flux,
        Q_joint_nnz = get_nnz(Q_joint),
        Q_combined_nnz = get_nnz(Q_combined),
        J_nnz = nnz(J),
        JtJ_nnz = nnz(JtJ),
        fillin_joint = fillin_joint,
        fillin_combined = fillin_combined,
    )
end

# ------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------

using ArgParse

function parse_commandline()
    s = ArgParseSettings(description = "Memory diagnostics for nonlinear SWE solver")

    @add_arg_table! s begin
        "--nx"
            help = "Grid points in x"
            arg_type = Int
            default = 15
        "--ny"
            help = "Grid points in y"
            arg_type = Int
            default = 15
        "--nsteps"
            help = "Number of time steps"
            arg_type = Int
            default = 5
        "--rho"
            help = "Sparse Cholesky threshold"
            arg_type = Float64
            default = 2.0
    end

    return parse_args(s)
end

if abspath(PROGRAM_FILE) == @__FILE__
    args = parse_commandline()
    run_diagnostics(
        N_x = args["nx"],
        N_y = args["ny"],
        n_timesteps = args["nsteps"],
        ρ = args["rho"]
    )
end
