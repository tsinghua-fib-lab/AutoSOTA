using Random
using Statistics
using LinearAlgebra
using Distributions
using StatsBase
using JLD2, FileIO

include("AliasSampler.jl")

struct Params
    y::Array{Float64, 2}
    prec_y::Array{Float64, 2}
    data_size::Int64
    dim::Int64
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

function generate_data(dim::Int64, data_size::Int64)
    cov_y = collect(Diagonal(LinRange(1, 0.05, dim)))
    y = rand(MvNormal(zeros(dim), cov_y), data_size)
    return y, cov_y
end

function Params(y::Array{Float64,2}, prec_y::Array{Float64,2}, dim::Int64, beta::Float64, lam_const::Float64, stepsize::Float64, nsamples::Int64)
    data_size = size(y)[2]
    lam_max = maximum(eigen(prec_y).values)
    M = vec(0.5 * beta * lam_max * sum((abs.(y) .+ 3.0).^2, dims=1))
    L = sum(M)
    lam = lam_const * (L^2)
    rho = lam * M ./ L .+ M
    rho_sum = sum(rho)
    weights = Weights(rho ./ rho_sum)
    Alias = AliasSampler(weights)
    return Params(y, prec_y, data_size, dim, beta, M, L, lam, rho, rho_sum, Alias, stepsize, nsamples)
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
        theta_prime[j] = theta_cur[j] + self.stepsize^2 * grad[j] + sqrt(2*self.stepsize^2) * randn()
    end
    return theta_prime
end

function get_phi(theta::Array{Float64,1}, y_i::Array{Float64,1}, beta::Float64, prec_y::Array{Float64,2}, M_i::Float64)
    phi_i = -0.5 * beta * (theta - y_i)' * prec_y * (theta - y_i) + M_i
    return phi_i
end

function grad_barker(theta::Array{Float64,1}, y_i::Array{Float64,1}, s_i::Int64, phi_i::Float64, lam::Float64, beta::Float64, prec_y::Array{Float64,2}, M_i::Float64, L::Float64)
    grad = -beta * prec_y * (theta - y_i) * s_i / (lam * M_i / L + phi_i)
    return grad
end

function grad_mala(theta::Array{Float64,1}, y_i::Array{Float64,1}, beta::Float64, prec_y::Array{Float64,2})
    grad = -beta * prec_y * (theta .- y_i)
    return grad
end

function quick_poisson(self::Params, theta_cur::Array{Float64,1})
    s = zeros(Int64, self.data_size)
    B = rand(Poisson(self.rho_sum))
    for b=1:B
        idx = rand(self.Alias)
        M_idx = self.M[idx]
        phi_idx = get_phi(theta_cur, self.y[:,idx], self.beta, self.prec_y, M_idx)
        add_prob = (self.lam * M_idx + self.L * phi_idx) / (self.lam * M_idx + self.L * M_idx)
        if rand() <= add_prob
            s[idx] += 1
        end
    end
    return s
end

function get_stepsize_list(target_rate::Float64)
    if target_rate == 0.55
        return [0.15, 0.27, 0.14, 0.4, 0.275, 0.4]
    elseif target_rate == 0.4
        return [0.2, 0.31, 0.2, 0.5, 0.315, 0.5]
    elseif target_rate == 0.25
        return [0.29, 0.36, 0.3, 0.6, 0.365, 0.6]
    else
        error("Unsupported target_rate: $target_rate")
    end
end

function get_steps_list()
    return [5000, 2500, 50000, 20000, 20000, 5000, 100000]
end

function build_output_path(outdir::String, data_size::Int64, lam_const::Float64, target_rate::Float64, beta::Float64, round::Int64)
    filename = "20d-all-data_size$(data_size)-lam_const$(lam_const)-target_rate$(target_rate)-beta$(beta)-round$(round).jld2"
    return joinpath(outdir, filename)
end

function compute_theta_true(y::Array{Float64,2}, cov_y::Array{Float64,2}, beta::Float64; save_theta_true::Bool=false)
    theta_true_untrunc = rand(MvNormal(vec(mean(y, dims=2)), cov_y./(size(y)[2]*beta)), 1000000)
    theta_true_list = Vector{Vector{Float64}}()

    for i in 1:1000000
        if all(-3 .<= theta_true_untrunc[:, i] .<= 3)
            push!(theta_true_list, theta_true_untrunc[:, i])
        end
    end

    theta_true_mat = hcat(theta_true_list...)
    mean_true = mean(theta_true_mat, dims=2)
    cov_true = diag(cov(permutedims(theta_true_mat)))

    if save_theta_true
        return theta_true_mat, mean_true, cov_true
    else
        return nothing, mean_true, cov_true
    end
end