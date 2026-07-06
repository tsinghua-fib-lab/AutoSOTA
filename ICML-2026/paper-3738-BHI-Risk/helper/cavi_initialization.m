function [initials] = cavi_initialization(seed_init, ...
    initial_method, E_hat, ...
    X_train, Y_train, true_params)

P = size(X_train, 2);
V = size(Y_train, 2);


rng(seed_init);

switch initial_method
    case  "true"
        H = true_params.H;
        mu = true_params.mu;
        gamma = true_params.gamma;
        alpha = true_params.alpha;

        E = size(H, 2);
        % true as initials
        initials = struct();
        initials.z_prob = [ones(E, 1); zeros(E_hat - E, 1)]; % initialize edge inclusion probabilities
        initials.m_prob = [H, zeros(V, E_hat - E)]; % initialize membership probabilities
        initials.gamma_prob = [gamma, zeros(P, E_hat - E)]; % initialize coefficient inclusion probabilities
        initials.mu_mean = [mu, zeros(P, E_hat - E)]; % initialize coefficient means to true values
        initials.mu_var = ones(P, E_hat); % initialize coefficient variances

        initials.alpha_mean = alpha; % initialize intercept means to true values
        initials.alpha_var = ones(1, V); % initialize intercept variances


    case "NNMF"
        % rng(seed_init);
        C = corr(Y_train, Rows="pairwise");
        C(C < 0) = 0;
        [W, ~] = nnmf(C, E_hat, 'replicates', 200, 'algorithm','als');
        m_prob = W ./ max(W(:));
        m_prob = 0.8 * (m_prob > quantile(m_prob(:), 0.6)) ...
            + 0.2 * rand(V, E_hat);
        initials.m_prob = m_prob;
        initials.z_prob = ones(E_hat, 1);
        gamma_prob = zeros(P, E_hat);
        for j = 1:P
            for e = 1:E_hat
                gamma_prob(j, e) = abs(corr(X_train(:, j), ...
                    mean(Y_train(:, m_prob(:, e) > 0.5), 2), "Rows", "pairwise"));
                if isnan(gamma_prob(j, e))
                    gamma_prob(j, e) = 0.5;
                end
            end
        end
        initials.gamma_prob = gamma_prob;
        initials.mu_mean = ones(P, E_hat);
        initials.mu_var = ones(P, E_hat);

        prev_train = mean(Y_train, 1, 'omitnan'); % (1, V)
        eps0 = 1 / (2 * size(Y_train, 1));
        initials.alpha_mean = log((prev_train + eps0) ./ (1 - prev_train + eps0));
        initials.alpha_var = ones(1, V);


    otherwise % random
        initials = [];
end


end