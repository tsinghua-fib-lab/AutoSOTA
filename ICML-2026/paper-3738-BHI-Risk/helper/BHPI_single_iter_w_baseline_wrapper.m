function [alpha_ast, alpha_varrho, mu_ast, sigma2_ast, nu_ast, rho_ast, r_ast, ...
    E_Omega, E_gamma, E_O, E_O_m_diff, E_s_b_given_z, E_mu_given_gamma, ...
    a_mu_ast, b_mu_ast, E_sigma2_inv, E_log_sigma2, E_eta_tilde, E_eta_tilde_sq, ...
    a_nu_ast, b_nu_ast, a_rho_ast, b_rho_ast, a_r_ast, b_r_ast, ...
    E_nu_logit, E_log_nu, E_log_1_minus_nu, ...
    E_rho_logit, E_log_rho, E_log_1_minus_rho, ...
    E_r_logit, E_log_r, E_log_1_minus_r, ELBO] = ...
    BHPI_single_iter_w_baseline_wrapper(iter, X, Kappa, E_edges, ...
    r_ast, rho_ast, nu_ast, mu_ast, sigma2_ast, alpha_ast, alpha_varrho, Xi_Xj, X_sq, ...
    E_mu_j_mu_k_given_z, E_Omega, E_gamma, E_O, E_O_m_diff, E_mu_sq_given_z, ...
    E_eta_tilde, E_eta_tilde_sq, ...
    E_s_b_given_z, E_mu_given_gamma, E_sigma2_inv, E_log_sigma2, ...
    E_nu_logit, E_log_nu, E_log_1_minus_nu, ...
    E_rho_logit, E_log_rho, E_log_1_minus_rho, E_r_logit, ...
    a_mu, b_mu, a_nu, b_nu, a_rho, b_rho, a_r, b_r, omega_repulsion, ...
    fix_z, z_constraint, update_m, fix_gamma, freeze_gamma, tau_temper, sigma2_alpha, ...
    batch_size, t0, weights)

N = size(X, 1);

if ~exist('batch_size', 'var') || isempty(batch_size)
    batch_size = false;
end
if batch_size > 0
    batch_size = min(batch_size, N);
    batch_scale = N / batch_size;
else
    batch_scale = 1;
end

if ~exist('t0', 'var') || isempty(t0)
    t0 = 10;
end

if ~exist("weights", "var") || isempty(weights)
    weights = 1;
end

dv_inv = 1 / sqrt(E_edges); % Fixed normalization

robbins_monro = @(it) (it + t0)^(-0.3); % Robbins-Monro step size
robbins_monro_update = @(x_old, x_new, it) (1 - robbins_monro(it)) * x_old + robbins_monro(it) * x_new;


if batch_size
    idx_batch = randsample(N, batch_size, false);
else
    idx_batch = 1:N;
end
mask_batch = Kappa(idx_batch, :) == 0;

if isscalar(E_Omega) && isnan(E_Omega)
    % E[eta | z]
    E_b_given_z = X(idx_batch, :) * (nu_ast .* mu_ast); % N x E (given z=1)
    E_s_b_given_z = dv_inv * E_b_given_z .* permute(rho_ast, [3, 2, 1]); % N x E x V (given z=1)
    E_s_b = permute(r_ast' .* E_s_b_given_z, [1, 3, 2]); % N x V x E
    E_eta = sum(E_s_b, 3); % N x V
    E_eta_tilde = E_eta + alpha_ast; % N x V
    E_b_sq_given_z = X_sq(idx_batch, :) * E_mu_sq_given_z + 2 * Xi_Xj(idx_batch, :) * E_mu_j_mu_k_given_z; % N x E (given z=1)
    E_s_sq_b_sq = permute(dv_inv^2 * E_b_sq_given_z .* r_ast' .* permute(rho_ast, [3, 2, 1]), [1, 3, 2]); % N x V x E
    % tmp1 = E_s_b .* permute(E_s_b, [1, 2, 4, 3]); % N x V x E x E
    % E_s_b_e1e2_sum = sum(tmp1(:, :, triu(true(E_edges), 1)), 3); % N x V x E2 -> N x V
    S = sum(E_s_b, 3);              % N x V
    S2 = sum(E_s_b.^2, 3);          % N x V
    E_s_b_e1e2_sum = 0.5 * (S.^2 - S2);
    E_eta_sq = sum(E_s_sq_b_sq, 3) + E_s_b_e1e2_sum; % N x V

    E_alpha_sq = alpha_ast.^2 + alpha_varrho; % N x V
    E_eta_tilde_sq = E_eta_sq + E_alpha_sq + 2 * alpha_ast .* E_eta; % N x V
    eta_ast = sqrt(E_eta_tilde_sq + eps); % N x V
    E_Omega = 1 ./ (2 * eta_ast) .* tanh(0.5 * eta_ast); % N x V
    E_Omega(mask_batch) = 0; % Apply mask to E_Omega to ignore missing entries
end

[alpha_ast_new, alpha_varrho_new, mu_ast_new, sigma2_ast_new, ...
    nu_ast_new, rho_ast_new, r_ast_new, ~, ...
    E_Omega_new, E_gamma_new, E_O_new, E_O_m_diff_new, ...
    E_s_b_given_z_new, E_mu_given_gamma_new, ...
    a_mu_ast, b_mu_ast, E_sigma2_inv_new, E_log_sigma2_new, ...
    E_eta_tilde_new, E_eta_tilde_sq_new, ...
    a_nu_ast, b_nu_ast, a_rho_ast, b_rho_ast, a_r_ast, b_r_ast, ...
    E_nu_logit_new, E_log_nu_new, E_log_1_minus_nu_new, ...
    E_rho_logit_new, E_log_rho_new, E_log_1_minus_rho_new, ...
    E_r_logit_new, E_log_r, E_log_1_minus_r] = ...
    BHPI_single_iter_w_baseline(Kappa(idx_batch,:), ...
    X(idx_batch,:), X_sq(idx_batch, :), Xi_Xj(idx_batch, :), ...
    r_ast, rho_ast, nu_ast, E_Omega, E_gamma, E_O, E_O_m_diff, ...
    E_s_b_given_z, E_mu_given_gamma, E_sigma2_inv, E_log_sigma2, ...
    E_nu_logit, E_log_nu, E_log_1_minus_nu, ...
    E_rho_logit, E_log_rho, E_log_1_minus_rho, E_r_logit, ...
    a_mu, b_mu, a_nu, b_nu, a_rho, b_rho, a_r, b_r, omega_repulsion, ...
    fix_z, z_constraint, update_m, fix_gamma, freeze_gamma, tau_temper, ...
    sigma2_alpha, batch_scale, weights);

alpha_ast = robbins_monro_update(alpha_ast, alpha_ast_new, iter);
alpha_varrho = robbins_monro_update(alpha_varrho, alpha_varrho_new, iter);
mu_ast = robbins_monro_update(mu_ast, mu_ast_new, iter);
sigma2_ast = robbins_monro_update(sigma2_ast, sigma2_ast_new, iter);
nu_ast = robbins_monro_update(nu_ast, nu_ast_new, iter);
rho_ast = robbins_monro_update(rho_ast, rho_ast_new, iter);
r_ast = robbins_monro_update(r_ast, r_ast_new, iter);
E_Omega = robbins_monro_update(E_Omega, E_Omega_new, iter);
E_gamma = robbins_monro_update(E_gamma, E_gamma_new, iter);
E_O = robbins_monro_update(E_O, E_O_new, iter);
E_O_m_diff = robbins_monro_update(E_O_m_diff, E_O_m_diff_new, iter);
E_s_b_given_z = robbins_monro_update(E_s_b_given_z, E_s_b_given_z_new, iter);
E_mu_given_gamma = robbins_monro_update(E_mu_given_gamma, E_mu_given_gamma_new, iter);
E_sigma2_inv = robbins_monro_update(E_sigma2_inv, E_sigma2_inv_new, iter);
E_log_sigma2 = robbins_monro_update(E_log_sigma2, E_log_sigma2_new, iter);
E_eta_tilde = robbins_monro_update(E_eta_tilde, E_eta_tilde_new, iter);
E_eta_tilde_sq = robbins_monro_update(E_eta_tilde_sq, E_eta_tilde_sq_new, iter);
E_nu_logit = robbins_monro_update(E_nu_logit, E_nu_logit_new, iter);
E_log_nu = robbins_monro_update(E_log_nu, E_log_nu_new, iter);
E_log_1_minus_nu = robbins_monro_update(E_log_1_minus_nu, E_log_1_minus_nu_new, iter);
E_rho_logit = robbins_monro_update(E_rho_logit, E_rho_logit_new, iter);
E_log_rho = robbins_monro_update(E_log_rho, E_log_rho_new, iter);
E_log_1_minus_rho = robbins_monro_update(E_log_1_minus_rho, E_log_1_minus_rho_new, iter);
E_r_logit = robbins_monro_update(E_r_logit, E_r_logit_new, iter);


%% ELBO

if nargout >= 35
    lik = sum((Kappa(idx_batch, :) .* E_eta_tilde - 0.5 * E_eta_tilde_sq .* E_Omega) .* weights, "all") * batch_scale; % N x V -> Scalar

    L_alpha = sum(log(alpha_varrho) - (alpha_ast.^2 + alpha_varrho) / sigma2_alpha) / 2; % Prior and entropy for alpha
    L_mu = sum(r_ast' .* nu_ast .* (log(sigma2_ast) - E_log_sigma2 - (mu_ast.^2 + sigma2_ast) * E_sigma2_inv + 1), "all") / 2; % % Prior and entropy for mu

    repulsion_term = 0;
    if omega_repulsion > 0
        for e1 = 1:E_edges
            for e2 = (e1+1):E_edges
                repulsion_term = repulsion_term + sum(E_O(e1, e2) * r_ast(e1) * r_ast(e2) * nu_ast(:, e1) .* nu_ast(:, e2));
            end
        end
        repulsion_term = - omega_repulsion * repulsion_term;
    end

    kl_term_z_nu = sum(nu_ast .* (E_log_nu - log(nu_ast)) + ...
        (1 - nu_ast) .* (E_log_1_minus_nu - log(1 - nu_ast)), 1); % P x E -> 1 x E
    L_gamma = kl_term_z_nu * r_ast + repulsion_term; % Prior and entropy for gamma

    kl_term_z_rho = sum(rho_ast .* (E_log_rho - log(rho_ast)) ...
        + (1 - rho_ast) .* (E_log_1_minus_rho - log(1 - rho_ast)), 1); % V x E -> 1 x E
    L_m = kl_term_z_rho * r_ast; % Prior and entropy for m

    if fix_z == true
        L_z = 0; % No contribution if z is fixed
    else
        L_z = r_ast' * (E_log_r - log(r_ast)) + (1 - r_ast)' * (E_log_1_minus_r - log(1 - r_ast)); % Prior and entropy for z
    end

    L_hyper = sum((a_nu - a_nu_ast) .* E_log_nu + (b_nu - b_nu_ast) .* E_log_1_minus_nu + log(beta(a_nu_ast, b_nu_ast)), 'all') + ...
        sum((a_rho - a_rho_ast) .* E_log_rho + (b_rho - b_rho_ast) .* E_log_1_minus_rho + log(beta(a_rho_ast, b_rho_ast)), 'all') + ...
        sum((a_r - a_r_ast)' * E_log_r + (b_r - b_r_ast)' * E_log_1_minus_r + log(beta(a_r_ast, b_r_ast)), 'all') + ...
        (a_mu_ast - a_mu) * E_log_sigma2 + (b_mu_ast - b_mu) * E_sigma2_inv - a_mu_ast * log(b_mu_ast) + gammaln(a_mu_ast); % Hyperprior terms

    ELBO = lik + L_mu + L_gamma + L_m + L_z + L_hyper + L_alpha;

end
end