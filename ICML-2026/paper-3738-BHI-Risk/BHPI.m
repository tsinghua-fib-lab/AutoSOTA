%%%%%%%%%%%%%%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Logistic CAVI update for hypergraph data with multistage updates
% add baseline intercept alpha_v for each disease v
% Masked version that only updates based on observed entries in Kappa (matrix MASK)
function [model, mu_mean_all, mu_var_all, gamma_prob_all, m_prob_all, z_prob_all] = ...
    BHPI(X, Y, E_edges, max_iter, ...
    seed, initials, omega_repulsion, staged, ...
    final_fix_z, final_z_constraint, sigma2_alpha, warmup_iters, ...
    batch_size, t0, weights, tol, verbose)
tic;
[N, P] = size(X);
[~, V] = size(Y);
dv_inv = 1 / sqrt(E_edges); % Fixed normalization
tau_temper = 1;

convergence = false;

% E2 = E_edges * (E_edges - 1) / 2; % Number of unique edge pairs
P2 = P * (P - 1) / 2; % Number of unique pairs

X_sq = X.^2; % Precompute X squared

MASK = isnan(Y); % Mask for observed entries,
% assume missing entries are NaN, N x V
% 1 indicates observed, 0 indicates missing
Kappa = Y - 0.5; % Centered labels for PG augmentation, N x V
Kappa(MASK) = 0;

ELBO = NaN(1, max_iter); % To store ELBO values
eps = 1e-16;

if ~exist('warmup_iters', 'var') || isempty(warmup_iters)
    warmup_iters = 50;
end

if ~exist('batch_size', 'var') || isempty(batch_size)
    batch_size = false;
end
if batch_size > 0
    batch_size = min(batch_size, N);
    % batch_scale = N / batch_size;
    % else
    % batch_scale = 1;
end

if ~exist('t0', 'var') || isempty(t0)
    t0 = 10;
end

if ~exist("weights", "var") || isempty(weights)
    weights = 1;
end


if ~exist('tol', 'var') || isempty(tol)
    tol = 1e-16;
end

if ~exist('verbose', 'var') || isempty(verbose)
    verbose = true;
end

% Initialize Variational Parameters
if exist('initials', 'var') && (~isempty(initials))
    r_ast = initials.z_prob; % Edge inclusion probabilities
    rho_ast = initials.m_prob; % Membership probabilities
    nu_ast = initials.gamma_prob; % Coefficient inclusion probabilities
    mu_ast = initials.mu_mean; % Coefficient means
    sigma2_ast = initials.mu_var; % Coefficient variances
    alpha_ast = initials.alpha_mean; % Baseline intercept means
    alpha_varrho = initials.alpha_var; % Baseline intercept variances
else
    rng(seed);
    r_ast = random("Uniform", 0.5, 0.8, [E_edges, 1]);  % Edge inclusion probabilities
    rho_ast = random("Uniform", 0.5, 0.8, [V, E_edges]); % Membership probabilities
    nu_ast = random("Uniform", 0.5, 0.8, [P, E_edges]); % Coefficient inclusion probabilities
    sigma2_ast = ones(P, E_edges) * 100; % Coefficient variances
    mu_ast = randn(P, E_edges) .* sqrt(sigma2_ast); % Coefficient means
    alpha_ast = zeros(1, V); % Baseline intercept means
    alpha_varrho = ones(1, V) * 100; % Baseline intercept variances
end

nu_ast = clip(nu_ast, eps, 1 - eps);
rho_ast = clip(rho_ast, eps, 1 - eps);
r_ast = clip(r_ast, eps, 1 - eps);

% Hyperparameters
a_mu = 1 / 2;
b_mu = 1 / 2;
a_r = 1 / 2;
b_r = 1 / 2;
a_rho = 1 / 2;
b_rho = 1 / 2;
a_nu = 1 / 2;
b_nu = 1 / 2;

if ~exist("omega_repulsion", "var") || isempty(omega_repulsion)
    omega_repulsion = 1.0; % Repulsion weight
end

if ~exist("staged", "var") || isempty(staged)
    staged = false;
end

if ~exist("final_fix_z", "var") || isempty(final_fix_z)
    final_fix_z = false;
end

if ~exist("final_z_constraint", "var") || isempty(final_z_constraint)
    final_z_constraint = 1;
end

if ~exist("sigma2_alpha", "var") || isempty(sigma2_alpha)
    sigma2_alpha = 10;
end

%% --- Precompute expectations ---
E_gamma_given_z = nu_ast; % (P x E) (given z=1)
E_z = r_ast; % (E x 1)
E_gamma = E_z' .* E_gamma_given_z; % (P x E)
E_mu_given_gamma = mu_ast; % P x E (given gamma=1)
E_mu_sq_given_gamma = mu_ast.^2 + sigma2_ast; % P x E (given gamma=1)
E_mu_sq_given_z = E_gamma_given_z .* E_mu_sq_given_gamma; % P x E (given z=1)

% E[eta^2]
E_mu_j_mu_k_given_z = zeros(P2, E_edges); % P2 x E (given z=1)
Xi_Xj = zeros(N, P2); % N x P2
idx = 1;
for j = 1:P
    for k = (j+1):P
        E_mu_j_mu_k_given_z(idx, :) = nu_ast(j, :) .* mu_ast(j, :) .* nu_ast(k, :) .* mu_ast(k, :);
        Xi_Xj(:, idx) = X(:, j) .* X(:, k);
        idx = idx + 1;
    end
end

[E_O, E_O_m_diff] = E_Overlap(rho_ast);

a_mu_ast = a_mu + 0.5 * sum(E_gamma, [1, 2]); % Updated shape parameter for sigma_mu
E_mu_sq = r_ast' .* E_mu_sq_given_z; % (P x E)
b_mu_ast = b_mu + 0.5 * sum(E_mu_sq, [1, 2]); % Updated rate parameter for sigma_mu
E_log_sigma2 = log(b_mu_ast) - psi(a_mu_ast); % Scalar expectation of log(sigma^2)
E_sigma2_inv = a_mu_ast / b_mu_ast; % Scalar expectation of 1/sigma^2

a_nu_ast = a_nu + E_gamma; % Updated shape parameter for nu, P x E
b_nu_ast = b_nu + 1 - E_gamma; % Updated rate parameter for nu, P x E
a_nu_digamma = psi(a_nu_ast);
b_nu_digamma = psi(b_nu_ast);
a_plus_b_nu_digamma = psi(a_nu_ast + b_nu_ast);
E_nu_logit = a_nu_digamma - b_nu_digamma; % P x E
E_log_nu = a_nu_digamma - a_plus_b_nu_digamma; % P x E
E_log_1_minus_nu = b_nu_digamma - a_plus_b_nu_digamma; % P x E

E_m = r_ast' .* rho_ast; % (V x E)
a_rho_ast = a_rho + E_m; % Updated shape parameter for rho, V x E
b_rho_ast = b_rho + 1 - E_m; % Updated rate parameter for rho, V x E
a_rho_digamma = psi(a_rho_ast);
b_rho_digamma = psi(b_rho_ast);
E_rho_logit = a_rho_digamma - b_rho_digamma; % V x E
a_plus_b_rho_digamma = psi(a_rho_ast + b_rho_ast);
E_log_rho = a_rho_digamma - a_plus_b_rho_digamma; % V x E
E_log_1_minus_rho = b_rho_digamma - a_plus_b_rho_digamma; % V x E

a_r_ast = a_r + E_z; % Updated shape parameter for r, E x 1
b_r_ast = b_r + 1 - E_z; % Updated rate parameter for r, E x 1
E_r_logit = psi(a_r_ast) - psi(b_r_ast); % E x 1

E_Omega = NaN; E_s_b_given_z = NaN; E_eta_tilde = NaN; E_eta_tilde_sq = NaN;

if staged

    %% Stage 0. Fix z=1, freeze m, fix gamma=1, update mu (Warm-up)
    fix_z = true;
    z_constraint = 0;
    update_m = false;
    fix_gamma = true;
    freeze_gamma = false;
    disp('Stage 0: Fix z=1, freeze m, fix gamma=1, update mu (Warm-up)');
    for iter = 1:warmup_iters

        [alpha_ast, alpha_varrho, mu_ast, sigma2_ast, nu_ast, rho_ast, r_ast, ...
            E_Omega, E_gamma, E_O, E_O_m_diff, E_s_b_given_z, E_mu_given_gamma, ...
            a_mu_ast, b_mu_ast, E_sigma2_inv, E_log_sigma2, E_eta_tilde, E_eta_tilde_sq, ...
            a_nu_ast, b_nu_ast, a_rho_ast, b_rho_ast, a_r_ast, b_r_ast, ...
            E_nu_logit, E_log_nu, E_log_1_minus_nu, ...
            E_rho_logit, E_log_rho, E_log_1_minus_rho, ...
            E_r_logit, ~, ~] = ...
            BHPI_single_iter_w_baseline_wrapper(iter, X, Kappa, E_edges, ...
            r_ast, rho_ast, nu_ast, mu_ast, sigma2_ast, alpha_ast, alpha_varrho, Xi_Xj, X_sq, ...
            E_mu_j_mu_k_given_z, E_Omega, E_gamma, E_O, E_O_m_diff, E_mu_sq_given_z, ...
            E_eta_tilde, E_eta_tilde_sq, ...
            E_s_b_given_z, E_mu_given_gamma, E_sigma2_inv, E_log_sigma2, ...
            E_nu_logit, E_log_nu, E_log_1_minus_nu, ...
            E_rho_logit, E_log_rho, E_log_1_minus_rho, E_r_logit, ...
            a_mu, b_mu, a_nu, b_nu, a_rho, b_rho, a_r, b_r, omega_repulsion, ...
            fix_z, z_constraint, update_m, fix_gamma, freeze_gamma, tau_temper, sigma2_alpha, ...
            batch_size, t0, weights);

    end

    fprintf('sum(r) = %.3f, mean(rho) = %.3f, mean(nu) = %.3f\n', ...
        sum(r_ast), mean(rho_ast(:)), mean(nu_ast(:)));
    toc;


    %% Stage 1. Fix z=1, freeze m, update gamma & mu
    fix_z = true;
    z_constraint = 0;
    update_m = false;
    fix_gamma = false;
    freeze_gamma = false;
    disp('Stage 1: Fix z=1, freeze m, update gamma & mu');

    % start gamma from initials
    nu_ast = clip(initials.gamma_prob, 0.5, 0.99);

    for iter = 1:warmup_iters

        [alpha_ast, alpha_varrho, mu_ast, sigma2_ast, nu_ast, rho_ast, r_ast, ...
            E_Omega, E_gamma, E_O, E_O_m_diff, E_s_b_given_z, E_mu_given_gamma, ...
            a_mu_ast, b_mu_ast, E_sigma2_inv, E_log_sigma2, E_eta_tilde, E_eta_tilde_sq, ...
            a_nu_ast, b_nu_ast, a_rho_ast, b_rho_ast, a_r_ast, b_r_ast, ...
            E_nu_logit, E_log_nu, E_log_1_minus_nu, ...
            E_rho_logit, E_log_rho, E_log_1_minus_rho, ...
            E_r_logit, ~, ~] = ...
            BHPI_single_iter_w_baseline_wrapper(iter, X, Kappa, E_edges, ...
            r_ast, rho_ast, nu_ast, mu_ast, sigma2_ast, alpha_ast, alpha_varrho, Xi_Xj, X_sq, ...
            E_mu_j_mu_k_given_z, E_Omega, E_gamma, E_O, E_O_m_diff, E_mu_sq_given_z, ...
            E_eta_tilde, E_eta_tilde_sq, ...
            E_s_b_given_z, E_mu_given_gamma, E_sigma2_inv, E_log_sigma2, ...
            E_nu_logit, E_log_nu, E_log_1_minus_nu, ...
            E_rho_logit, E_log_rho, E_log_1_minus_rho, E_r_logit, ...
            a_mu, b_mu, a_nu, b_nu, a_rho, b_rho, a_r, b_r, omega_repulsion, ...
            fix_z, z_constraint, update_m, fix_gamma, freeze_gamma, tau_temper, sigma2_alpha, ...
            batch_size, t0, weights);
    end

    fprintf('sum(r) = %.3f, mean(rho) = %.3f, mean(nu) = %.3f\n', ...
        sum(r_ast), mean(rho_ast(:)), mean(nu_ast(:)));

    toc;

    %% Stage 2b. Fix z=1, update m, gamma & mu
    fix_z = true;
    z_constraint = 0;
    update_m = true;
    fix_gamma = false;
    freeze_gamma = false;
    disp('Stage 2b: Fix z=1, update m, gamma & mu');

    for iter = 1:warmup_iters

        [alpha_ast, alpha_varrho, mu_ast, sigma2_ast, nu_ast, rho_ast, r_ast, ...
            E_Omega, E_gamma, E_O, E_O_m_diff, E_s_b_given_z, E_mu_given_gamma, ...
            a_mu_ast, b_mu_ast, E_sigma2_inv, E_log_sigma2, E_eta_tilde, E_eta_tilde_sq, ...
            a_nu_ast, b_nu_ast, a_rho_ast, b_rho_ast, a_r_ast, b_r_ast, ...
            E_nu_logit, E_log_nu, E_log_1_minus_nu, ...
            E_rho_logit, E_log_rho, E_log_1_minus_rho, ...
            E_r_logit, ~, ~] = ...
            BHPI_single_iter_w_baseline_wrapper(iter, X, Kappa, E_edges, ...
            r_ast, rho_ast, nu_ast, mu_ast, sigma2_ast, alpha_ast, alpha_varrho, Xi_Xj, X_sq, ...
            E_mu_j_mu_k_given_z, E_Omega, E_gamma, E_O, E_O_m_diff, E_mu_sq_given_z, ...
            E_eta_tilde, E_eta_tilde_sq, ...
            E_s_b_given_z, E_mu_given_gamma, E_sigma2_inv, E_log_sigma2, ...
            E_nu_logit, E_log_nu, E_log_1_minus_nu, ...
            E_rho_logit, E_log_rho, E_log_1_minus_rho, E_r_logit, ...
            a_mu, b_mu, a_nu, b_nu, a_rho, b_rho, a_r, b_r, omega_repulsion, ...
            fix_z, z_constraint, update_m, fix_gamma, freeze_gamma, tau_temper, sigma2_alpha, ...
            batch_size, t0, weights);

    end

    fprintf('sum(r) = %.3f, mean(rho) = %.3f, mean(nu) = %.3f\n', ...
        sum(r_ast), mean(rho_ast(:)), mean(nu_ast(:)));
    toc;

end % End of staged warm-up

E_m = r_ast' .* rho_ast; % (V x E)
beta_old = dv_inv * (nu_ast .* E_mu_given_gamma) * E_m'; % P x E * E x V -> P x V


%% Stage 3. Update z, m, gamma & mu
if staged
    disp('Stage 3: Update all with z_constraint');
end
if ~final_fix_z && (~exist("final_z_constraint", "var") || isempty(final_z_constraint))
    final_z_constraint = 1;
end

update_m = true;
fix_gamma = false;
freeze_gamma = false;
expected_log_lik = NaN(1, max_iter);
if nargout > 1
    mu_mean_all = NaN(P, E_edges, max_iter);
    mu_var_all = NaN(P, E_edges, max_iter);
    gamma_prob_all = NaN(P, E_edges, max_iter);
    m_prob_all = NaN(V, E_edges, max_iter);
    z_prob_all = NaN(E_edges, max_iter);
end
for iter = 1:max_iter

    [alpha_ast, alpha_varrho, mu_ast, sigma2_ast, nu_ast, rho_ast, r_ast, ...
        E_Omega, E_gamma, E_O, E_O_m_diff, E_s_b_given_z, E_mu_given_gamma, ...
        a_mu_ast, b_mu_ast, E_sigma2_inv, E_log_sigma2, E_eta_tilde, E_eta_tilde_sq, ...
        a_nu_ast, b_nu_ast, a_rho_ast, b_rho_ast, a_r_ast, b_r_ast, ...
        E_nu_logit, E_log_nu, E_log_1_minus_nu, ...
        E_rho_logit, E_log_rho, E_log_1_minus_rho, ...
        E_r_logit, ~, ~, ELBO(iter)] = ...
        BHPI_single_iter_w_baseline_wrapper(iter, X, Kappa, E_edges, ...
        r_ast, rho_ast, nu_ast, mu_ast, sigma2_ast, alpha_ast, alpha_varrho, Xi_Xj, X_sq, ...
        E_mu_j_mu_k_given_z, E_Omega, E_gamma, E_O, E_O_m_diff, E_mu_sq_given_z, ...
        E_eta_tilde, E_eta_tilde_sq, ...
        E_s_b_given_z, E_mu_given_gamma, E_sigma2_inv, E_log_sigma2, ...
        E_nu_logit, E_log_nu, E_log_1_minus_nu, ...
        E_rho_logit, E_log_rho, E_log_1_minus_rho, E_r_logit, ...
        a_mu, b_mu, a_nu, b_nu, a_rho, b_rho, a_r, b_r, omega_repulsion, ...
        final_fix_z, final_z_constraint, update_m, fix_gamma, freeze_gamma, tau_temper, sigma2_alpha, ...
        batch_size, t0, weights);

    yhat = 1 ./ (1 + exp(-E_eta_tilde));
    expected_log_lik(iter) = mean(Y .* log(yhat) + (1 - Y) .* log(1 - yhat), 'all', 'omitnan'); % (N, V) -> scalar


    %% --- Convergence Check ---
    % change of beta
    E_m = r_ast' .* rho_ast; % (V x E)
    beta_new = dv_inv * (nu_ast .* E_mu_given_gamma) * E_m'; % P x E * E x V -> P x V
    % E_beta_diff_norm = norm(beta_new - beta_old, 'fro') / (norm(beta_old, 'fro') + 1e-10);
    E_beta_diff = max(abs(beta_new - beta_old), [], 'all');

    if iter == 1
        loglik_diff = -1000;
    else
        loglik_diff = expected_log_lik(iter) - expected_log_lik(iter-1);
    end

    if verbose && mod(iter, 10) == 0
        fprintf('Iteration %d: ELBO=%.4g, Exp_lik=%.4g, diff of E[beta]: %.6g \n', ...
            iter, ELBO(iter), expected_log_lik(iter), E_beta_diff);
        toc;
    end

    if (E_beta_diff < tol) && (loglik_diff < tol)
        fprintf('Converged at iteration %d, ELBO: %.4g, Exp_lik=%.4g, diff of E[beta]: %.6g \n', ...
            iter, ELBO(iter), expected_log_lik(iter), E_beta_diff);
        convergence = true;
        if verbose
            fprintf('sum(r) = %.3f, mean(rho) = %.3f, mean(nu) = %.3f\n', ...
                sum(r_ast), mean(rho_ast(:)), mean(nu_ast(:)));
        end
        toc;
        break;
    else
        beta_old = beta_new;
    end


    %% record results for all iterations
    if nargout > 1
        mu_mean_all(:, :, iter) = mu_ast;
        mu_var_all(:, :, iter) = sigma2_ast;
        gamma_prob_all(:, :, iter) = nu_ast;
        m_prob_all(:, :, iter) = rho_ast;
        z_prob_all(:, iter) = r_ast;
    end
end




% Return model
model.ELBO = ELBO(1:iter);
model.expected_log_lik = expected_log_lik(1:iter);
model.mu_mean = mu_ast;
model.mu_var = sigma2_ast;
model.gamma_prob = nu_ast;
model.m_prob = rho_ast;
model.z_prob = r_ast;
model.beta = beta_new;
% model.eta = eta_ast;
% model.eta_mean = E_eta;
model.alpha_mean = alpha_ast;
model.alpha_var = alpha_varrho;

model.sigma2_a = a_mu_ast;
model.sigma2_b = b_mu_ast;
model.rho_a = a_rho_ast;
model.rho_b = b_rho_ast;
model.nu_a = a_nu_ast;
model.nu_b = b_nu_ast;
model.r_a = a_r_ast;
model.r_b = b_r_ast;

model.convergence = convergence;

end