using Random

function qcorridor(T, k, com, seed, alpha0, alphaT, eps0, epsT, N, q_init=0.0, bonus_beta=0.0, ucb_c=0.0)
    # Initialize algorithm paramters
    rng = Xoshiro(seed + 1000*k)
    pi = [0.5, 0.5]

    # Compute decreasing alpha, eps
    function c(t, c0, cT)
        if c0 == cT return c0 end
        cb = cT * T / (c0 - cT)
        ca = c0 * cb
        return ca / (cb + t)
    end

    # Initialize Q-table and start episode
    # ALGO-1: Optimistic initialization (q_init > 0)
    q = q_init > 0 ? fill(q_init, 2, 2) : 0.1 * randn(rng, 2, 2)

    # ALGO-2: Visit counter for count-based exploration bonus and UCB
    visit_count = zeros(Int, 2, 2)

    ts = N > 1 ? floor.(10 .^ range(0, log10(T), N)) :
        N == 0 ? range(1, T) : [T]
    qs = N != 1 ? zeros(size(ts, 1), 2, 2) : zeros(1, 2, 2)
    x = 0
    z = 0
    u = rand(rng) < pi[z + 1]
    j = 1

    for t in 1:T
        # Sample transition
        x += 2*u - 1
        z_ = x > 0
        r = x == 0 ? 0 : x == k + 1 ? k : -1

        # ALGO-2: Count-based intrinsic exploration bonus
        visit_count[z + 1, u + 1] += 1
        r_total = r
        if bonus_beta > 0
            r_total += bonus_beta / sqrt(visit_count[z + 1, u + 1])
        end

        # Update Q-values
        q[z + 1, u + 1] += c(t, alpha0, alphaT) * (r_total +
            (0 <= x <= k ? maximum(@view q[z_ + 1, :]) : 0) - q[z + 1, u + 1])

        # ALGO-4: UCB-based policy (when ucb_c > 0) or epsilon-greedy (default)
        if ucb_c > 0
            for zi in 1:2
                ucb_l = q[zi, 1] + ucb_c * sqrt(log(t + 1) / (visit_count[zi, 1] + 1))
                ucb_r = q[zi, 2] + ucb_c * sqrt(log(t + 1) / (visit_count[zi, 2] + 1))
                # Deterministic: prefer argmax of UCB scores
                pi[zi] = ucb_r > ucb_l ? 1.0 : 0.0
            end
        elseif eps0 > 0
            eps = c(t, eps0, epsT)
            pi[1] = q[1, 2] > q[1, 1] ? 1 - eps : eps
            pi[2] = q[2, 2] > q[2, 1] ? 1 - eps : eps
        end

        # Log Q-table (in logarithmic intervals)
        while j <= size(ts, 1) && t == ts[j]
            qs[j, :, :] = q
            j += 1
        end

        if !(0 <= x <= k)  # Start new episode
            x = 0
            z = 0
            u = rand(rng) < pi[z + 1]
        elseif !com || z_ != z  # Sample new action
            z = z_
            u = rand(rng) < pi[z_ + 1]
        end
    end
    return ts, qs
end
