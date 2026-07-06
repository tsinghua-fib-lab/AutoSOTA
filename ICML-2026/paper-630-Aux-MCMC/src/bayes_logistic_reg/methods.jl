using Statistics
using LinearAlgebra
using Distributions
using StatsBase

# =========================
# Dispatch
# =========================

function dispatch_method(args::Dict,
                         train_x::Array{Float64,2},
                         train_y::Vector{Int},
                         test_x::Array{Float64,2},
                         test_y::Vector{Int})
    method = args["method"]

    if method == "mh"
        return run_mh(args, train_x, train_y, test_x, test_y)
    elseif method == "mala"
        return run_mala(args, train_x, train_y, test_x, test_y)
    elseif method == "hmc"
        return run_hmc(args, train_x, train_y, test_x, test_y)
    elseif method == "tuna_mh"
        return run_tuna_mh(args, train_x, train_y, test_x, test_y)
    elseif method == "tuna_sgld"
        return run_tuna_sgld(args, train_x, train_y, test_x, test_y)
    elseif method == "barker"
        return run_barker(args, train_x, train_y, test_x, test_y)
    else
        error("Unknown method: $method")
    end
end

# =========================
# Shared helpers
# =========================

function l2dist(x::Vector{Float64}, y::Vector{Float64})
    @assert length(x) == length(y)
    acc = 0.0
    for i in eachindex(x)
        acc += (x[i] - y[i])^2
    end
    return sqrt(acc)
end

function logistic_logtarget(X::Array{Float64,2},
                            y::Vector{Int},
                            T::Float64,
                            theta::Vector{Float64},
                            theta_prime::Vector{Float64},
                            idx::Int)
    Xi_dot_theta = 0.0
    Xi_dot_theta_prime = 0.0
    for j in eachindex(theta)
        Xi_dot_theta += X[j, idx] * theta[j]
        Xi_dot_theta_prime += X[j, idx] * theta_prime[j]
    end

    predict = sigmoid(Xi_dot_theta)
    predict_prime = sigmoid(Xi_dot_theta_prime)
    yi = y[idx]

    logl = logH(predict, yi) / T
    logl_prime = logH(predict_prime, yi) / T
    return logl, logl_prime
end

# =========================
# MH
# =========================

struct MHSampler
    X::Array{Float64,2}
    y::Vector{Int}
    stepsize::Float64
    data_size::Int
    pca_dim::Int
    T::Float64
    theta_prime::Vector{Float64}
end

function MHSampler(X::Array{Float64,2}, y::Vector{Int}, stepsize::Float64, pca_dim::Int, T::Float64)
    theta_prime = zeros(pca_dim)
    return MHSampler(X, y, stepsize, size(X, 2), pca_dim, T, theta_prime)
end

function mh_proposal!(self::MHSampler, theta::Vector{Float64})
    for i in eachindex(theta)
        self.theta_prime[i] = theta[i] + self.stepsize * randn()
    end
    return self.theta_prime
end

stand_mh(u::Float64) = exp(u)

function mh_next!(self::MHSampler, theta::Vector{Float64})
    sig = 0
    theta_prime = mh_proposal!(self, theta)
    logmh = 0.0
    for i in 1:self.data_size
        ll_old, ll_new = logistic_logtarget(self.X, self.y, self.T, theta, theta_prime, i)
        logmh += ll_new - ll_old
    end
    acc_prob = stand_mh(logmh)
    if rand() < acc_prob
        theta .= theta_prime
        sig = 1
    end
    return theta, sig
end

function mh_train(self::MHSampler, nsamples::Int, burnin::Int, test_x::Array{Float64,2}, test_y::Vector{Int})
    theta = kaiming_unif_init(self.pca_dim)
    acc_init = test_accuracy(repeat(theta, 1, 2), test_x, test_y)

    samples = zeros(self.pca_dim, nsamples)
    succ = 0.0
    iters = nsamples + burnin
    interval = 100
    total_runtime = 0.0

    for i in 1:iters
        runtime = @elapsed begin
            theta, sig = mh_next!(self, theta)
            succ += sig
            if i > burnin
                samples[:, i - burnin] .= theta
            end
        end
        total_runtime += runtime
    end

    acc, acc_time, datause = finalize_trajectory(
        samples, test_x, test_y, interval, total_runtime, k -> k * self.data_size
    )
    acc = [acc_init; acc]
    acc_time = [0.0; acc_time]

    avg_accept_prob = succ / iters
    return Dict(
        "samples" => samples,
        "avg_accept_prob" => avg_accept_prob,
        "acc" => acc,
        "acc_time" => acc_time,
        "datause" => datause,
    )
end

function run_mh(args::Dict, X::Array{Float64,2}, y::Vector{Int}, test_x::Array{Float64,2}, test_y::Vector{Int})
    sampler = MHSampler(X, y, args["stepsize"], args["pca_dim"], args["T"])
    return mh_train(sampler, args["nsamples"], args["burnin"], test_x, test_y)
end

# =========================
# MALA
# =========================

struct MALASampler
    X::Array{Float64,2}
    y::Vector{Int}
    stepsize::Float64
    data_size::Int
    pca_dim::Int
    T::Float64
    theta_prime::Vector{Float64}
end

function MALASampler(X::Array{Float64,2}, y::Vector{Int}, stepsize::Float64, pca_dim::Int, T::Float64)
    theta_prime = zeros(pca_dim)
    return MALASampler(X, y, stepsize, size(X, 2), pca_dim, T, theta_prime)
end

function mala_proposal!(self::MALASampler, theta::Vector{Float64})
    sigmoid_term = sigmoid.(self.X' * theta) .- self.y
    grad = self.X * sigmoid_term / self.T
    proposal_mean = theta - 0.5 * self.stepsize^2 * (2 .* grad ./ norm(grad))
    self.theta_prime .= proposal_mean .+ self.stepsize .* randn(self.pca_dim)

    sigmoid_term_prime = sigmoid.(self.X' * self.theta_prime) .- self.y
    grad_prime = self.X * sigmoid_term_prime / self.T
    proposal_mean_prime = self.theta_prime - 0.5 * self.stepsize^2 * (2 .* grad_prime ./ norm(grad_prime))

    return self.theta_prime, proposal_mean, proposal_mean_prime
end

function mala_logp_proposal(self::MALASampler, theta::Vector{Float64}, prop_mean::Vector{Float64})
    return -0.5 * sum((theta .- prop_mean).^2) / (self.stepsize^2)
end

function mala_next!(self::MALASampler, theta::Vector{Float64})
    sig = 0
    theta_prime, proposal_mean, proposal_mean_prime = mala_proposal!(self, theta)
    logmh = 0.0

    for i in 1:self.data_size
        ll_old, ll_new = logistic_logtarget(self.X, self.y, self.T, theta, theta_prime, i)
        logmh += ll_new - ll_old
    end

    logmh += mala_logp_proposal(self, theta, proposal_mean_prime) -
             mala_logp_proposal(self, theta_prime, proposal_mean)

    acc_prob = stand_mh(logmh)
    if rand() < acc_prob
        theta .= theta_prime
        sig = 1
    end
    return theta, sig
end

function mala_train(self::MALASampler, nsamples::Int, burnin::Int, test_x::Array{Float64,2}, test_y::Vector{Int})
    theta = kaiming_unif_init(self.pca_dim)
    acc_init = test_accuracy(repeat(theta, 1, 2), test_x, test_y)

    samples = zeros(self.pca_dim, nsamples)
    succ = 0.0
    iters = nsamples + burnin
    interval = 100
    total_runtime = 0.0

    for i in 1:iters
        runtime = @elapsed begin
            theta, sig = mala_next!(self, theta)
            succ += sig
            if i > burnin
                samples[:, i - burnin] .= theta
            end
        end
        total_runtime += runtime
    end

    acc, acc_time, datause = finalize_trajectory(
        samples, test_x, test_y, interval, total_runtime, k -> k * self.data_size
    )
    acc = [acc_init; acc]
    acc_time = [0.0; acc_time]

    avg_accept_prob = succ / iters
    return Dict(
        "samples" => samples,
        "avg_accept_prob" => avg_accept_prob,
        "acc" => acc,
        "acc_time" => acc_time,
        "datause" => datause,
    )
end

function run_mala(args::Dict, X::Array{Float64,2}, y::Vector{Int}, test_x::Array{Float64,2}, test_y::Vector{Int})
    sampler = MALASampler(X, y, args["stepsize"], args["pca_dim"], args["T"])
    return mala_train(sampler, args["nsamples"], args["burnin"], test_x, test_y)
end

# =========================
# HMC
# =========================

struct HMCSampler
    X::Array{Float64,2}
    y::Vector{Int}
    leapfrog::Int
    stepsize::Float64
    data_size::Int
    pca_dim::Int
    T::Float64
    theta_prime::Vector{Float64}
end

function HMCSampler(X::Array{Float64,2}, y::Vector{Int}, leapfrog::Int, stepsize::Float64, pca_dim::Int, T::Float64)
    theta_prime = zeros(pca_dim)
    return HMCSampler(X, y, leapfrog, stepsize, size(X, 2), pca_dim, T, theta_prime)
end

function hmc_grad_U(self::HMCSampler, theta::Vector{Float64})
    grad = zeros(self.pca_dim)
    for i in 1:self.data_size
        Xi_dot_theta = 0.0
        for j in eachindex(theta)
            Xi_dot_theta += self.X[j, i] * theta[j]
        end
        predict = sigmoid(Xi_dot_theta)
        yi = self.y[i]
        grad .+= self.X[:, i] * (predict - yi) / self.T
    end
    return grad
end

function hmc_leapfrog(self::HMCSampler, theta::Vector{Float64}, p::Vector{Float64})
    theta_prime = copy(theta)
    p_prime = copy(p)

    grad_theta = hmc_grad_U(self, theta_prime)
    p_prime .-= 0.5 * self.stepsize * (2 .* grad_theta ./ norm(grad_theta))

    for i in 1:self.leapfrog
        theta_prime .+= self.stepsize .* p_prime
        if i != self.leapfrog
            grad_prime1 = hmc_grad_U(self, theta_prime)
            p_prime .-= self.stepsize * (2 .* grad_prime1 ./ norm(grad_prime1))
        end
    end

    grad_prime2 = hmc_grad_U(self, theta_prime)
    p_prime .-= 0.5 * self.stepsize * (2 .* grad_prime2 ./ norm(grad_prime2))
    p_prime = -p_prime
    return theta_prime, p_prime
end

function hmc_proposal!(self::HMCSampler, theta::Vector{Float64})
    p = randn(self.pca_dim)
    theta_prime, p_prime = hmc_leapfrog(self, theta, p)
    self.theta_prime .= theta_prime
    logp = -0.5 * dot(p, p)
    logp_prime = -0.5 * dot(p_prime, p_prime)
    return self.theta_prime, logp, logp_prime
end

function hmc_next!(self::HMCSampler, theta::Vector{Float64})
    sig = 0
    theta_prime, logp, logp_prime = hmc_proposal!(self, theta)
    logmh = 0.0

    for i in 1:self.data_size
        ll_old, ll_new = logistic_logtarget(self.X, self.y, self.T, theta, theta_prime, i)
        logmh += ll_new - ll_old
    end

    logmh += logp_prime - logp
    acc_prob = stand_mh(logmh)
    if rand() < acc_prob
        theta .= theta_prime
        sig = 1
    end
    return theta, sig
end

function hmc_train(self::HMCSampler, nsamples::Int, burnin::Int, test_x::Array{Float64,2}, test_y::Vector{Int})
    theta = kaiming_unif_init(self.pca_dim)
    acc_init = test_accuracy(repeat(theta, 1, 2), test_x, test_y)

    samples = zeros(self.pca_dim, nsamples)
    succ = 0.0
    iters = nsamples + burnin
    interval = 100
    total_runtime = 0.0

    for i in 1:iters
        runtime = @elapsed begin
            theta, sig = hmc_next!(self, theta)
            succ += sig
            if i > burnin
                samples[:, i - burnin] .= theta
            end
        end
        total_runtime += runtime
    end

    acc, acc_time, datause = finalize_trajectory(
        samples, test_x, test_y, interval, total_runtime, k -> k * self.data_size
    )
    acc = [acc_init; acc]
    acc_time = [0.0; acc_time]

    avg_accept_prob = succ / iters
    return Dict(
        "samples" => samples,
        "avg_accept_prob" => avg_accept_prob,
        "acc" => acc,
        "acc_time" => acc_time,
        "datause" => datause,
    )
end

function run_hmc(args::Dict, X::Array{Float64,2}, y::Vector{Int}, test_x::Array{Float64,2}, test_y::Vector{Int})
    sampler = HMCSampler(X, y, args["leapfrog"], args["stepsize"], args["pca_dim"], args["T"])
    return hmc_train(sampler, args["nsamples"], args["burnin"], test_x, test_y)
end

# =========================
# TunaMH
# =========================

struct TunaMHSampler
    X::Array{Float64,2}
    y::Vector{Int}
    stepsize::Float64
    data_size::Int
    pca_dim::Int
    c1::Float64
    psi::Vector{Float64}
    Psi::Float64
    gamma_A::AliasSampler
    lam::Float64
    T::Float64
    theta_prime::Vector{Float64}
end

function TunaMHSampler(X::Array{Float64,2}, y::Vector{Int}, stepsize::Float64, pca_dim::Int, lam::Float64, T::Float64)
    c1 = 1.0
    psi = c1 .* sqrt.(vec(sum(X.^2; dims=1))) ./ T
    Psi = sum(psi)
    gamma = Weights(psi ./ Psi)
    gamma_A = AliasSampler(gamma)
    theta_prime = zeros(pca_dim)
    return TunaMHSampler(X, y, stepsize, size(X, 2), pca_dim, c1, psi, Psi, gamma_A, lam, T, theta_prime)
end

function tuna_mh_proposal!(self::TunaMHSampler, theta::Vector{Float64})
    for i in eachindex(theta)
        self.theta_prime[i] = theta[i] + self.stepsize * randn()
    end
    return self.theta_prime
end

function tuna_phi_i(self::TunaMHSampler, idx::Int, Mi::Float64, theta::Vector{Float64}, theta_prime::Vector{Float64})
    logl, logl_prime = logistic_logtarget(self.X, self.y, self.T, theta, theta_prime, idx)
    return 0.5 * (logl - logl_prime + Mi), 0.5 * (logl_prime - logl + Mi)
end

function tuna_mh_next!(self::TunaMHSampler, theta::Vector{Float64})
    sig = 0
    theta_prime = tuna_mh_proposal!(self, theta)
    diff_norm = l2dist(theta_prime, theta)
    L = diff_norm * self.Psi
    lam = self.lam * L^2
    N = lam + L
    s = rand(Poisson(N))
    bs = 0
    logmh = 0.0

    for _ in 1:s
        idx = rand(self.gamma_A)
        M_i = diff_norm * self.psi[idx]
        phi_old, phi_new = tuna_phi_i(self, idx, M_i, theta, theta_prime)
        ps = (lam * M_i + L * phi_old) / (lam * M_i + L * M_i)
        if rand() <= ps
            bs += 1
            logmh += log(1 + L / (lam * M_i) * phi_new) -
                     log(1 + L / (lam * M_i) * phi_old)
        end
    end

    acc_prob = stand_mh(logmh)
    if rand() < acc_prob
        theta .= theta_prime
        sig = 1
    end
    return theta, sig, s
end

function tuna_mh_train(self::TunaMHSampler, nsamples::Int, burnin::Int, test_x::Array{Float64,2}, test_y::Vector{Int})
    theta = kaiming_unif_init(self.pca_dim)
    acc_init = test_accuracy(repeat(theta, 1, 2), test_x, test_y)

    samples = zeros(self.pca_dim, nsamples)
    succ = 0.0
    total_bs = 0.0
    iters = nsamples + burnin
    interval = 100
    K = Int(floor(nsamples / interval))
    acc = zeros(K)
    acc_time = zeros(K)
    datause = zeros(K)
    total_runtime = 0.0
    k = 1

    for i in 1:iters
        runtime = @elapsed begin
            theta, sig, bs = tuna_mh_next!(self, theta)
            total_bs += bs
            succ += sig
            if i > burnin
                samples[:, i - burnin] .= theta
            end
        end
        total_runtime += runtime

        if i > burnin && ((i - burnin) % interval == 0)
            acc[k] = test_accuracy(samples[:, 1:(i - burnin)], test_x, test_y)
            acc_time[k] = total_runtime
            datause[k] = total_bs
            k += 1
        end
    end

    avg_accept_prob = succ / iters
    acc = [acc_init; acc]
    acc_time = [0.0; acc_time]

    return Dict(
        "samples" => samples,
        "total_bs" => total_bs,
        "avg_accept_prob" => avg_accept_prob,
        "acc" => acc,
        "acc_time" => acc_time,
        "datause" => datause,
    )
end

function run_tuna_mh(args::Dict, X::Array{Float64,2}, y::Vector{Int}, test_x::Array{Float64,2}, test_y::Vector{Int})
    sampler = TunaMHSampler(X, y, args["stepsize"], args["pca_dim"], args["lam"], args["T"])
    return tuna_mh_train(sampler, args["nsamples"], args["burnin"], test_x, test_y)
end

# =========================
# Tuna-SGLD
# =========================

struct TunaSGLDSampler
    X::Array{Float64,2}
    y::Vector{Int}
    grad_size::Int
    stepsize::Float64
    data_size::Int
    pca_dim::Int
    c1::Float64
    psi::Vector{Float64}
    Psi::Float64
    gamma_A::AliasSampler
    lam::Float64
    T::Float64
    theta_prime::Vector{Float64}
end

function TunaSGLDSampler(X::Array{Float64,2}, y::Vector{Int}, grad_size::Int, stepsize::Float64, pca_dim::Int, lam::Float64, T::Float64)
    c1 = 1.0
    psi = c1 .* sqrt.(vec(sum(X.^2; dims=1))) ./ T
    Psi = sum(psi)
    gamma = Weights(psi ./ Psi)
    gamma_A = AliasSampler(gamma)
    theta_prime = zeros(pca_dim)
    return TunaSGLDSampler(X, y, grad_size, stepsize, size(X, 2), pca_dim, c1, psi, Psi, gamma_A, lam, T, theta_prime)
end

function tuna_sgld_logp_proposal(self::TunaSGLDSampler, theta::Vector{Float64}, prop_mean::Vector{Float64})
    return -0.5 * sum((theta .- prop_mean).^2) / (self.stepsize^2)
end

function tuna_sgld_proposal!(self::TunaSGLDSampler, theta::Vector{Float64})
    grad_idx = sample(1:self.data_size, self.grad_size, replace=false)
    X_sub = self.X[:, grad_idx]
    y_sub = self.y[grad_idx]

    sigmoid_term = sigmoid.(X_sub' * theta) .- y_sub
    grad = X_sub * sigmoid_term / self.T
    proposal_mean = theta - 0.5 * self.stepsize^2 * self.data_size / self.grad_size *
                    (2 .* grad ./ norm(grad))
    self.theta_prime .= proposal_mean .+ self.stepsize .* randn(self.pca_dim)

    sigmoid_term_prime = sigmoid.(X_sub' * self.theta_prime) .- y_sub
    grad_prime = X_sub * sigmoid_term_prime / self.T
    proposal_mean_prime = self.theta_prime -
                          0.5 * self.stepsize^2 * self.data_size / self.grad_size *
                          (2 .* grad_prime ./ norm(grad_prime))

    return self.theta_prime, proposal_mean, proposal_mean_prime
end

function tuna_sgld_phi_i(self::TunaSGLDSampler, idx::Int, Mi::Float64, theta::Vector{Float64}, theta_prime::Vector{Float64})
    logl, logl_prime = logistic_logtarget(self.X, self.y, self.T, theta, theta_prime, idx)
    return 0.5 * (logl - logl_prime + Mi), 0.5 * (logl_prime - logl + Mi)
end

function tuna_sgld_next!(self::TunaSGLDSampler, theta::Vector{Float64})
    sig = 0
    theta_prime, proposal_mean, proposal_mean_prime = tuna_sgld_proposal!(self, theta)
    diff_norm = l2dist(theta_prime, theta)
    L = diff_norm * self.Psi
    lam = self.lam * L^2
    N = lam + L
    s = rand(Poisson(N))
    bs = 0
    logmh = 0.0

    for _ in 1:s
        idx = rand(self.gamma_A)
        M_i = diff_norm * self.psi[idx]
        phi_old, phi_new = tuna_sgld_phi_i(self, idx, M_i, theta, theta_prime)
        ps = (lam * M_i + L * phi_old) / (lam * M_i + L * M_i)
        if rand() <= ps
            bs += 1
            logmh += log(1 + L / (lam * M_i) * phi_new) -
                     log(1 + L / (lam * M_i) * phi_old)
        end
    end

    logmh += tuna_sgld_logp_proposal(self, theta, proposal_mean_prime) -
             tuna_sgld_logp_proposal(self, theta_prime, proposal_mean)

    acc_prob = stand_mh(logmh)
    if rand() < acc_prob
        theta .= theta_prime
        sig = 1
    end
    return theta, sig, s
end

function tuna_sgld_train(self::TunaSGLDSampler, nsamples::Int, burnin::Int, test_x::Array{Float64,2}, test_y::Vector{Int})
    theta = kaiming_unif_init(self.pca_dim)
    acc_init = test_accuracy(repeat(theta, 1, 2), test_x, test_y)

    samples = zeros(self.pca_dim, nsamples)
    succ = 0.0
    total_bs = 0.0
    iters = nsamples + burnin
    interval = 100
    K = Int(floor(nsamples / interval))
    acc = zeros(K)
    acc_time = zeros(K)
    datause = zeros(K)
    total_runtime = 0.0
    k = 1

    for i in 1:iters
        runtime = @elapsed begin
            theta, sig, bs = tuna_sgld_next!(self, theta)
            total_bs += bs
            succ += sig
            if i > burnin
                samples[:, i - burnin] .= theta
            end
        end
        total_runtime += runtime

        if i > burnin && ((i - burnin) % interval == 0)
            acc[k] = test_accuracy(samples[:, 1:(i - burnin)], test_x, test_y)
            acc_time[k] = total_runtime
            datause[k] = total_bs
            k += 1
        end
    end

    avg_accept_prob = succ / iters
    acc = [acc_init; acc]
    acc_time = [0.0; acc_time]

    return Dict(
        "samples" => samples,
        "total_bs" => total_bs,
        "avg_accept_prob" => avg_accept_prob,
        "acc" => acc,
        "acc_time" => acc_time,
        "datause" => datause,
    )
end

function run_tuna_sgld(args::Dict, X::Array{Float64,2}, y::Vector{Int}, test_x::Array{Float64,2}, test_y::Vector{Int})
    sampler = TunaSGLDSampler(X, y, args["grad_size"], args["stepsize"], args["pca_dim"], args["lam"], args["T"])
    return tuna_sgld_train(sampler, args["nsamples"], args["burnin"], test_x, test_y)
end

# =========================
# Barker
# =========================

struct BarkerSampler
    X::Array{Float64,2}
    y::Vector{Int}
    stepsize::Float64
    data_size::Int
    pca_dim::Int
    T::Float64
    theta_prime::Vector{Float64}
end

function BarkerSampler(X::Array{Float64,2}, y::Vector{Int}, stepsize::Float64, pca_dim::Int, T::Float64)
    theta_prime = zeros(pca_dim)
    return BarkerSampler(X, y, stepsize, size(X, 2), pca_dim, T, theta_prime)
end

function barker_proposal!(self::BarkerSampler, theta::Vector{Float64})
    sigmoid_term = sigmoid.(self.X' * theta) .- self.y
    grad = self.X * sigmoid_term / self.T
    grad_clip = 2 .* grad ./ norm(grad)
    z = randn(self.pca_dim) .* self.stepsize
    p = 1.0 ./ (1.0 .+ exp.(-grad_clip .* z))

    for i in eachindex(theta)
        if rand() < p[i]
            self.theta_prime[i] = theta[i] + z[i]
        else
            self.theta_prime[i] = theta[i] - z[i]
        end
    end

    sigmoid_term_prime = sigmoid.(self.X' * self.theta_prime) .- self.y
    grad_prime = self.X * sigmoid_term_prime / self.T
    grad_clip_prime = 2 .* grad_prime ./ norm(grad_prime)

    return self.theta_prime, grad_clip, grad_clip_prime
end

function barker_next!(self::BarkerSampler, theta::Vector{Float64})
    sig = 0
    theta_prime, grad_clip, grad_clip_prime = barker_proposal!(self, theta)
    logmh = 0.0

    for i in 1:self.data_size
        ll_old, ll_new = logistic_logtarget(self.X, self.y, self.T, theta, theta_prime, i)
        logmh += ll_new - ll_old
    end

    diff = theta .- theta_prime
    logmh += sum(log.(1 .+ exp.(diff .* grad_clip)) .-
                 log.(1 .+ exp.(-diff .* grad_clip_prime)))

    acc_prob = stand_mh(logmh)
    if rand() < acc_prob
        theta .= theta_prime
        sig = 1
    end
    return theta, sig
end

function barker_train(self::BarkerSampler, nsamples::Int, burnin::Int, test_x::Array{Float64,2}, test_y::Vector{Int})
    theta = kaiming_unif_init(self.pca_dim)
    acc_init = test_accuracy(repeat(theta, 1, 2), test_x, test_y)

    samples = zeros(self.pca_dim, nsamples)
    succ = 0.0
    iters = nsamples + burnin
    interval = 100
    total_runtime = 0.0

    for i in 1:iters
        runtime = @elapsed begin
            theta, sig = barker_next!(self, theta)
            succ += sig
            if i > burnin
                samples[:, i - burnin] .= theta
            end
        end
        total_runtime += runtime
    end

    acc, acc_time, datause = finalize_trajectory(
        samples, test_x, test_y, interval, total_runtime, k -> k * self.data_size
    )
    acc = [acc_init; acc]
    acc_time = [0.0; acc_time]

    avg_accept_prob = succ / iters
    return Dict(
        "samples" => samples,
        "avg_accept_prob" => avg_accept_prob,
        "acc" => acc,
        "acc_time" => acc_time,
        "datause" => datause,
    )
end

function run_barker(args::Dict, X::Array{Float64,2}, y::Vector{Int}, test_x::Array{Float64,2}, test_y::Vector{Int})
    sampler = BarkerSampler(X, y, args["stepsize"], args["pca_dim"], args["T"])
    return barker_train(sampler, args["nsamples"], args["burnin"], test_x, test_y)
end