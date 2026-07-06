function poismh(self::Params, theta_init::Array{Float64, 1})
    y = self.y
    x = self.x
    M = self.M
    L = self.L
    lam = self.lam
    c = self.c
    df = self.df
    beta = self.beta
    theta_cur = zeros(self.dim)
    theta_cur .= theta_init
    samples = zeros(self.dim, self.nsamples)
    acc_count = 0
    time_step = zeros(Float64, self.nsamples)
    total_time = 0.0

    for step in 1:self.nsamples
        if (mod(step, 100) == 0)
            println("poismh step: ", step)
        end
        runtime = @elapsed begin
            prop_time = @elapsed begin
                theta_prime = rw_proposal(self, theta_cur)
            end
            if (norm(theta_prime) > c)
                samples[:,step] .= theta_cur
                total_time += prop_time
                time_step[step] = total_time
                continue
            end
            s = quick_poisson(self, theta_cur)
            I = findall(x -> x > 0, s)
            phi_cur = [get_phi(theta_cur, y[i], x[:, i], df, beta, M[i]) for i in I]
            phi_prime = [get_phi(theta_prime, y[i], x[:, i], df, beta, M[i]) for i in I]
            logp_ratio = sum(s[I] .* log.(1 .+ L ./ (lam * M[I]) .* phi_prime)) - sum(s[I] .* log.(1 .+ L ./ (lam * M[I]) .* phi_cur))
            if log(rand()) <= logp_ratio
                samples[:,step] .= theta_prime
                theta_cur .= theta_prime
                acc_count += 1
            else
                samples[:,step] .= theta_cur
            end
        end
        total_time += runtime
        time_step[step] = total_time
    end
    return samples, acc_count, time_step
end

function pois_barker(self::Params, theta_init::Array{Float64,1})
    y = self.y
    x = self.x
    M = self.M
    L = self.L
    lam = self.lam
    c = self.c
    df = self.df
    beta = self.beta
    theta_cur = zeros(self.dim)
    theta_cur .= theta_init
    samples = zeros(self.dim, self.nsamples)
    acc_count = 0
    time_step = zeros(Float64, self.nsamples)
    total_time = 0.0

    for step in 1:self.nsamples
        if (mod(step, 100) == 0)
            println("poismh barker step: ", step)
        end
        runtime = @elapsed begin
            prop_time = @elapsed begin
                s = quick_poisson(self, theta_cur)
                I = findall(x -> x > 0, s)
                phi_cur = [get_phi(theta_cur, y[i], x[:, i], df, beta, M[i]) for i in I]
                grad_cur = zeros(self.dim)
                idx_phi = 0
                for i in I
                    idx_phi += 1
                    grad_cur .+= grad_barker(theta_cur, y[i], x[:, i], s[i], df, phi_cur[idx_phi], lam, beta, M[i], L)
                end
                theta_prime = barker_proposal(self, theta_cur, grad_cur)
            end
            if (norm(theta_prime) > c)
                samples[:,step] .= theta_cur
                total_time += prop_time
                time_step[step] = total_time
                continue
            end
            phi_prime = [get_phi(theta_prime, y[i], x[:, i], df, beta, M[i]) for i in I]
            grad_prime = zeros(self.dim)
            idx_phi_prime = 0
            for i in I
                idx_phi_prime += 1
                grad_prime .+= grad_barker(theta_prime, y[i], x[:, i], s[i], df, phi_prime[idx_phi_prime], lam, beta, M[i], L)
            end
            diff = theta_prime .- theta_cur
            logp_ratio = sum(s[I] .* log.(1 .+ L ./ (lam * M[I]) .* phi_prime)) - sum(s[I] .* log.(1 .+ L ./ (lam * M[I]) .* phi_cur))
            log_prop_ratio = sum(log.(1.0 .+ exp.(-diff .* grad_cur)) .- log.(1.0 .+ exp.(diff .* grad_prime)))
            log_accept_ratio = logp_ratio + log_prop_ratio
            if log(rand()) <= log_accept_ratio
                samples[:,step] .= theta_prime
                theta_cur .= theta_prime
                acc_count += 1
            else
                samples[:,step] .= theta_cur
            end
        end
        total_time += runtime
        time_step[step] = total_time
    end
    return samples, acc_count, time_step
end

function pois_mala(self::Params, theta_init::Array{Float64,1})
    y = self.y
    x = self.x
    M = self.M
    L = self.L
    lam = self.lam
    c = self.c
    df = self.df
    beta = self.beta
    theta_cur = zeros(self.dim)
    theta_cur .= theta_init
    samples = zeros(self.dim, self.nsamples)
    acc_count = 0
    time_step = zeros(Float64, self.nsamples)
    total_time = 0.0

    for step in 1:self.nsamples
        if (mod(step, 100) == 0)
            println("poismh mala step: ", step)
        end
        runtime = @elapsed begin
            prop_time = @elapsed begin
                s = quick_poisson(self, theta_cur)
                I = findall(x -> x > 0, s)
                phi_cur = [get_phi(theta_cur, y[i], x[:, i], df, beta, M[i]) for i in I]
                grad_cur = zeros(self.dim)
                idx_phi = 0
                for i in I
                    idx_phi += 1
                    grad_cur .+= grad_barker(theta_cur, y[i], x[:, i], s[i], df, phi_cur[idx_phi], lam, beta, M[i], L)
                end
                theta_prime = mala_proposal(self, theta_cur, grad_cur)
            end
            if (norm(theta_prime) > c)
                samples[:,step] .= theta_cur
                total_time += prop_time
                time_step[step] = total_time
                continue
            end
            phi_prime = [get_phi(theta_prime, y[i], x[:, i], df, beta, M[i]) for i in I]
            grad_prime = zeros(self.dim)
            idx_phi_prime = 0
            for i in I
                idx_phi_prime += 1
                grad_prime .+= grad_barker(theta_prime, y[i], x[:, i], s[i], df, phi_prime[idx_phi_prime], lam, beta, M[i], L)
            end
            diff = theta_prime .- theta_cur
            logp_ratio = sum(s[I] .* log.(1 .+ L ./ (lam * M[I]) .* phi_prime)) - sum(s[I] .* log.(1 .+ L ./ (lam * M[I]) .* phi_cur))
            log_prop_ratio = -0.5*(norm(-diff .- self.stepsize^2*grad_prime)^2 - norm(diff .-self.stepsize^2*grad_cur)^2)/(2*self.stepsize^2)
            log_accept_ratio = logp_ratio + log_prop_ratio
            if log(rand()) <= log_accept_ratio
                samples[:,step] .= theta_prime
                theta_cur .= theta_prime
                acc_count += 1
            else
                samples[:,step] .= theta_cur
            end
        end
        total_time += runtime
        time_step[step] = total_time
    end
    return samples, acc_count, time_step
end

function mh(self::Params, theta_init::Array{Float64,1})
    y = self.y
    x = self.x
    c = self.c
    df = self.df
    M = self.M
    beta = self.beta
    theta_cur = zeros(self.dim)
    theta_cur .= theta_init
    samples = zeros(self.dim, self.nsamples)
    acc_count = 0
    time_step = zeros(Float64, self.nsamples)
    total_time = 0.0

    for step in 1:self.nsamples
        if (mod(step, 100) == 0)
            println("mh step: ", step)
        end
        runtime = @elapsed begin
            prop_time = @elapsed begin
                theta_prime = rw_proposal(self, theta_cur)
            end
            if (norm(theta_prime) > c)
                samples[:, step] .= theta_cur
                total_time += prop_time
                time_step[step] = total_time
                continue
            end
            phi = [get_phi(theta_cur, y[i], x[:, i], df, beta, M[i]) for i in 1:self.data_size]
            phi_prime = [get_phi(theta_prime, y[i], x[:, i], df, beta, M[i]) for i in 1:self.data_size]
            logp_ratio = sum(phi_prime .- phi)
            if log(rand()) <= logp_ratio
                samples[:,step] .= theta_prime
                theta_cur .= theta_prime
                acc_count += 1
            else
                samples[:,step] .= theta_cur
            end
        end
        total_time += runtime
        time_step[step] = total_time
    end
    return samples, acc_count, time_step
end

function mala(self::Params, theta_init::Array{Float64,1})
    y = self.y
    x = self.x
    c = self.c
    df = self.df
    M = self.M
    beta = self.beta
    theta_cur = zeros(self.dim)
    theta_cur .= theta_init
    samples = zeros(self.dim, self.nsamples)
    acc_count = 0
    time_step = zeros(Float64, self.nsamples)
    total_time = 0.0

    for step in 1:self.nsamples
        if (mod(step, 100) == 0)
            println("mala step: ", step)
        end
        runtime = @elapsed begin
            prop_time = @elapsed begin
                grad = zeros(self.dim)
                for i in 1:self.data_size
                    grad .+= grad_mala(theta_cur, y[i], x[:, i], df, beta)
                end
                theta_prime = mala_proposal(self, theta_cur, grad)
            end
            if (norm(theta_prime) > c)
                samples[:, step] .= theta_cur
                total_time += prop_time
                time_step[step] = total_time
                continue
            end
            phi = [get_phi(theta_cur, y[i], x[:, i], df, beta, M[i]) for i in 1:self.data_size]
            phi_prime = [get_phi(theta_prime, y[i], x[:, i], df, beta, M[i]) for i in 1:self.data_size]
            grad_prime = zeros(self.dim)
            for i in 1:self.data_size
                grad_prime .+= grad_mala(theta_prime, y[i], x[:, i], df, beta)
            end
            logp_ratio = sum(phi_prime .- phi)
            diff = theta_prime .- theta_cur
            log_prop_ratio = -0.5*(norm(-diff .- self.stepsize^2*grad_prime)^2 - norm(diff .- self.stepsize^2*grad)^2)/(2*self.stepsize^2)
            log_accept_ratio = logp_ratio + log_prop_ratio
            if log(rand()) <= log_accept_ratio
                samples[:,step] .= theta_prime
                theta_cur .= theta_prime
                acc_count += 1
            else
                samples[:,step] .= theta_cur
            end
        end
        total_time += runtime
        time_step[step] = total_time
    end
    return samples, acc_count, time_step
end

function hmc(self::Params, theta_init::Array{Float64,1}, lf_steps::Int64=5)
    y = self.y
    x = self.x
    c = self.c
    df = self.df
    M = self.M
    beta = self.beta
    theta_cur = zeros(self.dim)
    theta_cur .= theta_init
    samples = zeros(self.dim, self.nsamples)
    acc_count = 0
    time_step = zeros(Float64, self.nsamples)
    total_time = 0.0

    for step in 1:self.nsamples
        if (mod(step, 100) == 0)
            println("hmc step: ", step)
        end
        runtime = @elapsed begin
            prop_time = @elapsed begin
                current_p = randn(self.dim)
                theta_prime = zeros(self.dim)
                theta_prime .= theta_cur
                p_prime = zeros(self.dim)
                p_prime .= current_p
                for _ in 1:lf_steps
                    grad = zeros(self.dim)
                    for i in 1:self.data_size
                        grad .+= grad_mala(theta_prime, y[i], x[:, i], df, beta)
                    end
                    theta_prime, p_prime = leapfrog(self, theta_prime, p_prime, grad)
                end
                p_prime = -p_prime
            end
            if (norm(theta_prime) > c)
                samples[:, step] .= theta_cur
                total_time += prop_time
                time_step[step] = total_time
                continue
            end
            phi = [get_phi(theta_cur, y[i], x[:, i], df, beta, M[i]) for i in 1:self.data_size]
            phi_prime = [get_phi(theta_prime, y[i], x[:, i], df, beta, M[i]) for i in 1:self.data_size]
            logp_ratio = sum(phi_prime .- phi)
            log_prop_ratio = -0.5 * dot(p_prime, p_prime) + 0.5 * dot(current_p, current_p)
            if log(rand()) < logp_ratio + log_prop_ratio
                samples[:,step] .= theta_prime
                theta_cur .= theta_prime
                acc_count += 1
            else
                samples[:,step] .= theta_cur
            end
        end
        total_time += runtime
        time_step[step] = total_time
    end
    return samples, acc_count, time_step
end

function barker(self::Params, theta_init::Array{Float64,1})
    y = self.y
    x = self.x
    c = self.c
    df = self.df
    M = self.M
    beta = self.beta
    theta_cur = zeros(self.dim)
    theta_cur .= theta_init
    samples = zeros(self.dim, self.nsamples)
    acc_count = 0
    time_step = zeros(Float64, self.nsamples)
    total_time = 0.0

    for step in 1:self.nsamples
        if step % 100 == 0
            println("barker step: ", step)
        end
        runtime = @elapsed begin
            prop_time = @elapsed begin
                grad = zeros(self.dim)
                for i in 1:self.data_size
                    grad .+= grad_mala(theta_cur, y[i], x[:, i], df, beta)
                end
                theta_prime = barker_proposal(self, theta_cur, grad)
            end
            if (norm(theta_prime) > c)
                samples[:, step] .= theta_cur
                total_time += prop_time
                time_step[step] = total_time
                continue
            end
            phi = [get_phi(theta_cur, y[i], x[:, i], df, beta, M[i]) for i in 1:self.data_size]
            phi_prime = [get_phi(theta_prime, y[i], x[:, i], df, beta, M[i]) for i in 1:self.data_size]
            grad_prime = zeros(self.dim)
            for i in 1:self.data_size
                grad_prime .+= grad_mala(theta_prime, y[i], x[:, i], df, beta)
            end
            logp_ratio = sum(phi_prime .- phi)
            diff = theta_prime .- theta_cur
            log_prop_ratio = sum(log.(1.0 .+ exp.(-diff .* grad)) - log.(1.0 .+ exp.(diff .* grad_prime)))
            log_accept_ratio = logp_ratio + log_prop_ratio
            if log(rand()) <= log_accept_ratio
                samples[:,step] .= theta_prime
                theta_cur .= theta_prime
                acc_count += 1
            else
                samples[:,step] .= theta_cur
            end
        end
        total_time += runtime
        time_step[step] = total_time
    end
    return samples, acc_count, time_step
end

function sgld(self::Params, theta_init::Array{Float64,1}; batchsize::Int=512, decay::Bool=true, t0::Float64=5000.0, gamma::Float64=0.55)
    y = self.y
    x = self.x
    c = self.c
    df = self.df
    beta = self.beta
    theta_cur = zeros(self.dim)
    theta_cur .= theta_init
    samples = zeros(self.dim, self.nsamples)
    acc_count = 0
    time_step = zeros(Float64, self.nsamples)
    total_time = 0.0

    stepsize0 = self.stepsize

    for step in 1:self.nsamples
        if (mod(step, 100) == 0)
            println("sgld step: ", step)
        end
        runtime = @elapsed begin
            batch_idx = sample(1:self.data_size, batchsize; replace=false)
            grad = zeros(self.dim)
            for i in batch_idx
                grad .+= grad_mala(theta_cur, y[i], x[:, i], df, beta)
            end
            grad .*= self.data_size / length(batch_idx)

            if decay
                stepsize_t = stepsize0 / (1.0 + step / t0)^(gamma / 2)
            else
                stepsize_t = stepsize0
            end

            theta_prime = sgld_proposal(self, theta_cur, grad, stepsize_t)

            if norm(theta_prime) <= c
                theta_cur .= theta_prime
            end

            samples[:, step] .= theta_cur
        end

        total_time += runtime
        time_step[step] = total_time
    end

    return samples, acc_count, time_step
end

function run_sampler(theta_init::Array{Float64,1}, config::ExperimentConfig, steps::Array{Int,1})
    data_size = config.data_size
    dim = config.dim
    c = config.c
    df = config.df
    lam_const = config.lam_const
    beta = config.beta
    stepsize = config.stepsize_list

    y, x, theta_true = generate_data(dim, data_size)

    params_mh = Params(y, x, dim, c, df, beta, lam_const, stepsize[1], steps[1])
    params_mala = Params(y, x, dim, c, df, beta, lam_const, stepsize[2], steps[2])
    params_pois = Params(y, x, dim, c, df, beta, lam_const, stepsize[3], steps[3])
    params_pois_barker = Params(y, x, dim, c, df, beta, lam_const, stepsize[4], steps[4])
    params_pois_mala = Params(y, x, dim, c, df, beta, lam_const, stepsize[5], steps[5])
    params_hmc = Params(y, x, dim, c, df, beta, lam_const, stepsize[6], steps[6])
    params_barker = Params(y, x, dim, c, df, beta, lam_const, stepsize[7], steps[7])

    mh_samples, accept_mh, time_mh = mh(params_mh, theta_init)
    mala_samples, accept_mala, time_mala = mala(params_mala, theta_init)
    poismh_samples, accept_pois, time_pois = poismh(params_pois, theta_init)
    pois_barker_samples, accept_pois_barker, time_pois_barker = pois_barker(params_pois_barker, theta_init)
    pois_mala_samples, accept_pois_mala, time_pois_mala = pois_mala(params_pois_mala, theta_init)
    hmc_samples, accept_hmc, time_hmc = hmc(params_hmc, theta_init)
    barker_samples, accept_barker, time_barker = barker(params_barker, theta_init)

    result = Dict(
        "theta_init" => theta_init,
        "theta_true" => theta_true,
        "mh_samples" => mh_samples,
        "mala_samples" => mala_samples,
        "poismh_samples" => poismh_samples,
        "pois_barker_samples" => pois_barker_samples,
        "pois_mala_samples" => pois_mala_samples,
        "hmc_samples" => hmc_samples,
        "barker_samples" => barker_samples,
        "time_mh" => time_mh,
        "time_mala" => time_mala,
        "time_pois" => time_pois,
        "time_pois_barker" => time_pois_barker,
        "time_pois_mala" => time_pois_mala,
        "time_hmc" => time_hmc,
        "time_barker" => time_barker,
        "accept_mh" => accept_mh,
        "accept_mala" => accept_mala,
        "accept_pois" => accept_pois,
        "accept_pois_barker" => accept_pois_barker,
        "accept_pois_mala" => accept_pois_mala,
        "accept_hmc" => accept_hmc,
        "accept_barker" => accept_barker,
        "data_size" => data_size,
        "dim" => dim,
        "c" => c,
        "df" => df,
        "lam_const" => lam_const,
        "beta" => beta,
        "stepsize" => stepsize,
        "steps" => steps,
        "experiment" => config.experiment,
    )

    if config.include_sgld
        params_sgld = Params(y, x, dim, c, df, beta, lam_const, 0.01, steps[8])
        sgld_samples, accept_sgld, time_sgld = sgld(params_sgld, theta_init, batchsize=512, decay=false)
        result["sgld_samples"] = sgld_samples
        result["time_sgld"] = time_sgld
        result["accept_sgld"] = accept_sgld
    end

    return result
end
