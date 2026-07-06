using Random
using Statistics
using LinearAlgebra
using Distributions
using StatsBase
using JLD2, FileIO

include("AliasSampler.jl")

struct Params
    y::Array{Float64, 1}
    x::Array{Float64, 2}
    data_size::Int64
    dim::Int64
    c::Float64
    df::Int64
    beta::Float64
    M::Array{Float64, 1}
    L::Float64
    lam::Float64
    rho::Array{Float64, 1}
    rho_sum::Float64
    Alias::AliasSampler
    stepsize::Float64
    nsamples::Int64
end

struct ExperimentConfig
    experiment::String
    data_size::Int64
    dim::Int64
    c::Float64
    df::Int64
    lam_const::Float64
    beta::Float64
    stepsize_list::Vector{Float64}
    theta_init_dim::Int64
    include_sgld::Bool
end

function generate_data(dim::Int64, data_size::Int64)
    x = randn(dim, data_size)
    theta_true = ones(dim)
    y = vec(x' * theta_true .+ randn(data_size))
    return y, x, theta_true
end

function Params(y::Array{Float64, 1}, x::Array{Float64, 2}, dim::Int64, c::Float64, df::Int64, beta::Float64, lam_const::Float64, stepsize::Float64, nsamples::Int64)
    data_size = size(y)[1]
    x_2norm = sqrt.(vec(sum(x.^2, dims=1)))
    y_abs = abs.(y)
    M = zeros(data_size)
    for i=1:data_size
        M[i] = beta * (df+1)/2 * log(1 + (1/df) * ((y_abs[i] + x_2norm[i] * c)^2))
    end
    L = sum(M)
    lam = lam_const * (L^2)
    rho = lam * M ./ L .+ M
    rho_sum = sum(rho)
    weights = Weights(rho ./ rho_sum)
    Alias = AliasSampler(weights)
    return Params(y, x, data_size, dim, c, df, beta, M, L, lam, rho, rho_sum, Alias, stepsize, nsamples)
end

function barker_proposal(self::Params, theta_cur::Array{Float64, 1}, grad::Array{Float64, 1})
    theta_prime = zeros(self.dim)
    for j=1:self.dim
        eta = randn() * self.stepsize
        p = 1.0 / (1.0 + exp(-eta * grad[j]))
        theta_prime[j] = theta_cur[j] + 2.0 * (rand(Binomial(1, p)) - 0.5) * eta
    end
    return theta_prime
end

function rw_proposal(self::Params, theta_cur::Array{Float64, 1})
    theta_prime = zeros(self.dim)
    for j=1:self.dim
        theta_prime[j] = theta_cur[j] + self.stepsize * randn()
    end
    return theta_prime
end

function mala_proposal(self::Params, theta_cur::Array{Float64, 1}, grad::Array{Float64, 1})
    theta_prime = zeros(self.dim)
    for j=1:self.dim
        theta_prime[j] = theta_cur[j] + self.stepsize^2 * grad[j] + sqrt(2*self.stepsize^2) * randn()
    end
    return theta_prime
end

function sgld_proposal(self::Params, theta_cur::Array{Float64, 1}, grad::Array{Float64, 1}, stepsize::Float64)
    theta_prime = zeros(self.dim)
    for j=1:self.dim
        theta_prime[j] = theta_cur[j] + stepsize^2 * grad[j] + sqrt(2*stepsize^2) * randn()
    end
    return theta_prime
end

function get_phi(theta::Array{Float64,1}, y_i::Float64, x_i::Array{Float64, 1}, df::Int64, beta::Float64, M_i::Float64)
    phi_i = beta * -(df+1)/2 * log(1 + (1/df)*(y_i - theta' * x_i)^2) + M_i
    return phi_i
end

function grad_barker(theta::Array{Float64,1}, y_i::Float64, x_i::Array{Float64, 1}, s_i::Int64, df::Int64, phi_i::Float64, lam::Float64, beta::Float64, M_i::Float64, L::Float64)
    err = y_i - theta' * x_i
    grad_phi = beta * -(df + 1)/2 * 1/(1 + err^2/df) * 2*err/df * -x_i
    grad = grad_phi * s_i / (lam * M_i / L + phi_i)
    return grad
end

function grad_mala(theta::Array{Float64,1}, y_i::Float64, x_i::Array{Float64, 1}, df::Int64, beta::Float64)
    err = y_i - theta' * x_i
    grad = beta * -(df + 1)/2 * 1/(1 + err^2/df) * 2*err/df * -x_i
    return grad
end

function leapfrog(self::Params, theta::Array{Float64,1}, p::Array{Float64,1}, grad::Array{Float64,1})
    p += self.stepsize * 0.5 * grad
    theta += self.stepsize * p
    p += self.stepsize * 0.5 * grad
    return theta, p
end

function quick_poisson(self::Params, theta_cur::Array{Float64,1})
    s = zeros(Int64, self.data_size)
    B = rand(Poisson(self.rho_sum))
    for b=1:B
        idx = rand(self.Alias)
        M_idx = self.M[idx]
        phi_idx = get_phi(theta_cur, self.y[idx], self.x[:, idx], self.df, self.beta, M_idx)
        add_prob = (self.lam * M_idx + self.L * phi_idx) / (self.lam * M_idx + self.L * M_idx)
        if rand() <= add_prob
            s[idx] += 1
        end
    end
    return s
end

function get_experiment_config(experiment::String, target_rate::Float64)
    if experiment == "robust_reg_10d_n100000"
        # original zip setup
        if target_rate == 0.55
            stepsize_list = [0.18, 0.38, 0.17, 0.465, 0.36, 0.07, 0.46]
        elseif target_rate == 0.4
            stepsize_list = [0.26, 0.42, 0.25, 0.55, 0.41, 0.088, 0.58]
        elseif target_rate == 0.25
            stepsize_list = [0.36, 0.488, 0.35, 0.68, 0.47, 0.11, 0.7]
        else
            error("Unsupported target_rate: $target_rate")
        end
        return ExperimentConfig(experiment, 100000, 10, 15.0, 4, 0.01, 1e-4, stepsize_list, 10, true)
    elseif experiment == "robust_reg_50d_n100000"
        # from TruncLin-50d-All.jl; keep only hmc5 and no SGLD
        if target_rate == 0.55
            stepsize_list = [0.024, 0.077, 0.02, 0.084, 0.0755, 0.0094, 0.085]
        elseif target_rate == 0.4
            stepsize_list = [0.035, 0.085, 0.03, 0.098, 0.085, 0.0115, 0.1]
        elseif target_rate == 0.25
            stepsize_list = [0.05, 0.097, 0.04, 0.115, 0.094, 0.0142, 0.118]
        else
            error("Unsupported target_rate: $target_rate")
        end
        return ExperimentConfig(experiment, 100000, 50, 30.0, 4, 0.0001, 0.001, stepsize_list, 50, false)
    elseif experiment == "robust_reg_10d_n200000"
        # from TruncLin-10d-All.jl for new setting; keep zip logic with single hmc and SGLD
        if target_rate == 0.55
            stepsize_list = [0.118, 0.228, 0.105, 0.288, 0.230, 0.044, 0.288]
        elseif target_rate == 0.4
            stepsize_list = [0.168, 0.260, 0.158, 0.340, 0.260, 0.054, 0.347]
        elseif target_rate == 0.25
            stepsize_list = [0.217, 0.291, 0.214, 0.428, 0.291, 0.064, 0.428]
        else
            error("Unsupported target_rate: $target_rate")
        end
        return ExperimentConfig(experiment, 200000, 10, 15.0, 4, 0.01, 1e-4, stepsize_list, 10, true)
    else
        error("Unsupported experiment: $experiment")
    end
end

function default_outdir(experiment::String)
    return joinpath("results", experiment)
end

function build_output_path(outdir::String, experiment::String, data_size::Int64, c::Float64, df::Int64, lam_const::Float64, target_rate::Float64, beta::Float64, round::Int64)
    filename = "$(experiment)-data_size$(data_size)-c$(c)-df$(df)-lam_const$(lam_const)-target_rate$(target_rate)-beta$(beta)-round$(round).jld2"
    return joinpath(outdir, filename)
end
