%%%%%%%%%%%%%%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Logistic CAVI update for hypergraph data - single iteration
% add baseline intercept alpha_v for each disease v
function [alpha_ast, alpha_varrho, mu_ast, sigma2_ast, ...
    nu_ast, rho_ast, r_ast, eta_ast, ...
    E_Omega, E_gamma, E_O, E_O_m_diff, E_s_b_given_z, E_mu_given_gamma, ...
    a_mu_ast, b_mu_ast, E_sigma2_inv, E_log_sigma2, E_eta_tilde, E_eta_tilde_sq, ...
    a_nu_ast, b_nu_ast, a_rho_ast, b_rho_ast, a_r_ast, b_r_ast, ...
    E_nu_logit, E_log_nu, E_log_1_minus_nu, ...
    E_rho_logit, E_log_rho, E_log_1_minus_rho, ...
    E_r_logit, E_log_r, E_log_1_minus_r] = ...
    BHPI_single_iter_w_baseline(Kappa, X, X_sq, Xi_Xj, ...
    r_ast, rho_ast, nu_ast, E_Omega, E_gamma, E_O, E_O_m_diff, ...
    E_s_b_given_z, E_mu_given_gamma, E_sigma2_inv, E_log_sigma2, ...
    E_nu_logit, E_log_nu, E_log_1_minus_nu, ...
    E_rho_logit, E_log_rho, E_log_1_minus_rho, E_r_logit, ...
    a_mu, b_mu, a_nu, b_nu, a_rho, b_rho, a_r, b_r, omega_repulsion, ...
    fix_z, z_constraint, update_m, fix_gamma, freeze_gamma, ...
    tau_temper, sigma2_alpha, batch_scale, weights)

[~, V] = size(Kappa);
[~, P] = size(X);
[~, E_edges] = size(rho_ast);
dv_inv = 1 / sqrt(E_edges); % Fixed normalization

% E2 = E_edges * (E_edges - 1) / 2; % Number of unique edge pairs
P2 = P * (P - 1) / 2; % Number of unique pairs
eps = 1e-16;

if fix_gamma || ~exist("freeze_gamma", "var") || isempty(freeze_gamma)
    freeze_gamma = false;
end


if ~exist("tau_temper", "var") || isempty(tau_temper)
    tau_temper = 1;
end

if ~exist("sigma2_alpha", "var") || isempty(sigma2_alpha)
    sigma2_alpha = 10;
end

if ~exist("batch_scale", "var") || isempty(batch_scale)
    batch_scale = 1;
end

if ~exist("weights", "var") || isempty(weights)
    weights = 1;
end

batch_scale = batch_scale .* weights; % scalar or 1 x V

%% --- 1. Update q(alpha_v) ---
alpha_varrho = 1 ./ (batch_scale .* sum(E_Omega, 1) + 1 / sigma2_alpha); % 1 x V
E_s_b = permute(r_ast' .* E_s_b_given_z, [1, 3, 2]); % N x V x E
E_eta = sum(E_s_b, 3); % N x V
alpha_ast = alpha_varrho .* sum(Kappa - E_Omega .* E_eta, 1) .* batch_scale; % 1 x V

%% --- 2. Update q(Mu | gamma) ---
% Calculate precision for mu
E_s_sq_given_z = dv_inv^2 * rho_ast; % V x E (given z=1)
tau_ast = E_sigma2_inv + X_sq' * (batch_scale .* E_Omega) * E_s_sq_given_z; % P x E
sigma2_ast = 1 ./ tau_ast; % P x E

% mean for mu
E_s_given_z = dv_inv * rho_ast; % V x E
E_eta_given_z = E_eta - E_s_b + permute(E_s_b_given_z, [1, 3, 2]); % N x V x E
E_mu_given_z = nu_ast .* E_mu_given_gamma; % P x E
B_ast = zeros(P, E_edges); % P x E
for j = 1:P
    for e = 1:E_edges
        resid = Kappa - E_Omega .* (alpha_ast + E_eta_given_z(:, :, e) - ...
            X(:, j) .* E_mu_given_z(j, e) .* E_s_given_z(:, e)'); % N x V
        B_ast(j, e) = X(:, j)' * tau_temper * (resid .* batch_scale) * E_s_given_z(:, e); % Scalar
    end
end
% % %%%%% E_a_exclude_j_e, N x V x E x P %%%%%
% X_reshaped = permute(X, [1, 3, 4, 2]); % N x 1 x 1 x P
% E_x_s_mu_given_z = permute(X, [1, 3, 4, 2]) .* permute(nu_ast .* E_mu_given_gamma, ...
%     [3, 4, 2, 1]) .* permute(E_s_given_z, [3, 1, 2, 4]); % N x V x  E x P
% E_a_exclude_j_e = E_eta_given_z - E_x_s_mu_given_z; % N x V x E x P
% resid = Kappa - E_Omega .* (alpha_ast + E_a_exclude_j_e); % N x V x E x P
% B_ast = tau_temper * permute(sum(X_reshaped .* resid .* permute(E_s_given_z, ...
%     [3, 1, 2]), [1, 2]), [4, 3, 1, 2]); % N x V x E x P -> P x E

mu_ast = sigma2_ast .* B_ast; % P x E
% mu_ast = clip(mu_ast, -1e2, 1e2);

E_mu_given_gamma = mu_ast; % P x E (given gamma=1)
E_mu_sq_given_gamma = mu_ast.^2 + sigma2_ast; % P x E (given gamma=1)


%% --- 3. Update logit for q(gamma | z) ---
if fix_gamma % fix at 1
    nu_ast = ones(P, E_edges);
else
    if ~freeze_gamma
        % logit for gamma
        lik_gain = 0.5 * ((mu_ast.^2).* tau_ast - log(tau_ast) - E_log_sigma2); % P x E
        if omega_repulsion > 0
            repulsion_logit_gamma = omega_repulsion * permute(sum(...
                E_O .* permute(E_gamma, [3, 2, 1]), 2), [3, 1, 2]); % E1 x P -> P x E1
        else
            repulsion_logit_gamma = 0;
        end
        logit_gamma = tau_temper * lik_gain + E_nu_logit - repulsion_logit_gamma; % P x E
        nu_ast = 1 ./ (1 + exp(-logit_gamma)); % P x E
    end
end

nu_ast = clip(nu_ast, eps, 1 - eps);
E_gamma_given_z = nu_ast; % (P x E) (given z=1)


%% --- 4. Update Memberships (m) V x E ---
% E[zeta | z=1]
E_b_given_z = X * (E_gamma_given_z .* E_mu_given_gamma); % N x E (given z=1)
E_s_b_given_z = dv_inv * E_b_given_z .* permute(rho_ast, [3, 2, 1]); % N x E x V (given z=1)
E_s_b = permute(r_ast' .* E_s_b_given_z, [1, 3, 2]); % N x V x E
E_a_exclude_e = sum(E_s_b, 3) - E_s_b; % N x V x E
E_zeta_given_z = reshape(sum((Kappa - E_Omega .* (E_a_exclude_e + alpha_ast)) .* permute(E_b_given_z, [1, 3, 2]), 1), V, E_edges); % N x V x E -> V x E

% E[xi | z=1]
E_mu_sq_given_z = nu_ast .* E_mu_sq_given_gamma; % P x E (given z=1)
E_mu_given_z = nu_ast .* mu_ast; % P x E
E_mu_j_mu_k_given_z = zeros(P2, E_edges); % P2 x E (given z=1)
idx = 1;
for j = 1:P
    for k = (j+1):P
        E_mu_j_mu_k_given_z(idx, :) = E_mu_given_z(j, :) .* E_mu_given_z(k, :);
        idx = idx + 1;
    end
end
% tmp1 = E_mu_given_z' .* permute(E_mu_given_z, [2, 3, 1]); % E x P x P
% E_mu_j_mu_k_given_z = tmp1(:, triu(true(P), 1))'; % P2 x E (given z=1)
E_b_sq_given_z = X_sq * E_mu_sq_given_z + 2 * Xi_Xj * E_mu_j_mu_k_given_z; % N x E (given z=1)
E_xi_given_z = reshape(sum(E_Omega .* permute(E_b_sq_given_z, [1, 3, 2]), 1), V, E_edges); % N x V x E -> V x E

residual = dv_inv * E_zeta_given_z - 0.5 * dv_inv^2 * E_xi_given_z; % V x E

% repulsion term for m
E_gamma = r_ast' .* E_gamma_given_z; % (P x E)
if omega_repulsion > 0
    repulsion_logit_m_all_e = E_O_m_diff .* permute(E_gamma_given_z, [2, 3, 4, 1]) .* permute(E_gamma, [3, 2, 4, 1]); % E1 x E2 x V * E1 x 1 x 1 x P -> E1 x E2 x V x P, * 1 x E2 x 1 x P -> E1 x E2 x V x P
    repulsion_logit_m = omega_repulsion * permute(sum(repulsion_logit_m_all_e, [2, 4]), [3, 1, 2, 4]); % E1 x E2 x V x P -> V x E1
else
    repulsion_logit_m = 0;
end

if update_m
    logit_m = E_rho_logit + tau_temper * (residual .* batch_scale') - repulsion_logit_m; % V x E
    rho_ast = 1 ./ (1 + exp(-logit_m)); % V x E
end

rho_ast = clip(rho_ast, eps, 1 - eps);


%% --- 5. Update Z (r) ---
if update_m
    % repulsion term for z
    if omega_repulsion > 0
        [E_O, E_O_m_diff] = E_Overlap(rho_ast);
        repulsion_logit_z_all_e = E_O .* permute(E_gamma_given_z, ...
            [2, 3, 1]) .* permute(E_gamma, [3, 2, 1]); % E1 x E2 * E1 x 1 x P -> E1 x E2 x P, * 1 x E2 x P -> E1 x E2 x P
        repulsion_logit_z = omega_repulsion * sum(repulsion_logit_z_all_e, [2, 3]); % E1 x 1
    else
        repulsion_logit_z = 0;
    end
    
    % KL terms
    kl_term_z_rho = sum(rho_ast .* (E_log_rho - log(rho_ast)) ...
        + (1 - rho_ast) .* (E_log_1_minus_rho - log(1 - rho_ast)), 1); % V x E -> 1 x E
    kl_term_z_nu = sum(nu_ast .* (E_log_nu - log(nu_ast)) + ...
        (1 - nu_ast) .* (E_log_1_minus_nu - log(1 - nu_ast)), 1); % P x E -> 1 x E
    
    % residual
    % E[zeta | z=1]
    E_s_b_given_z = dv_inv * E_b_given_z .* permute(rho_ast, [3, 2, 1]); % N x E x V (given z=1)
    E_s_b = permute(r_ast' .* E_s_b_given_z, [1, 3, 2]); % N x V x E
    E_a_exclude_e = sum(E_s_b, 3) - E_s_b; % N x V x E
    E_zeta_given_z = reshape(sum((Kappa - E_Omega .* (E_a_exclude_e + alpha_ast)) .* permute(E_b_given_z, ...
        [1, 3, 2]), 1), V, E_edges); % N x V x E -> V x E
    % E[xi | z=1] does not change here
    residual = dv_inv * E_zeta_given_z - 0.5 * dv_inv^2 * E_xi_given_z; % V x E
end

if fix_z == 1 % fix at 1
    r_ast = ones(E_edges, 1);
else
    if fix_z == 0 % update
        logit_z = sum(rho_ast .* (residual .* batch_scale') * tau_temper, 1) ...
            + E_r_logit' + kl_term_z_rho + kl_term_z_nu ...
            - repulsion_logit_z'; % 1 x E
        r_ast = 1 ./ (1 + exp(-logit_z')); % E x 1
        
        if z_constraint > 0
            [~, idx] = sort(r_ast, 'descend');
            r_ast(idx(1:z_constraint)) = 1;
        end
    end
end

r_ast = clip(r_ast, eps, 1 - eps);
E_z = r_ast; % (E x 1)


%% --- 1. Update Polya-Gamma q(omega) ---
% E[eta]
E_s_b_given_z = dv_inv * E_b_given_z .* permute(rho_ast, [3, 2, 1]); % N x E x V (given z=1)
E_s_b = permute(r_ast' .* E_s_b_given_z, [1, 3, 2]); % N x V x E
E_eta = sum(E_s_b, 3); % N x V
E_eta_tilde = E_eta + alpha_ast; % N x V

% E[eta^2]
E_s_sq_b_sq = permute(dv_inv^2 * E_b_sq_given_z .* r_ast' .* permute(rho_ast, [3, 2, 1]), [1, 3, 2]); % N x V x E
% tmp1 = E_s_b .* permute(E_s_b, [1, 2, 4, 3]); % N x V x E x E
% E_s_b_e1e2_sum = sum(tmp1(:, :, triu(true(E_edges), 1)), 3); % N x V x E2 -> N x V
S = sum(E_s_b, 3);              % N x V
S2 = sum(E_s_b.^2, 3);          % N x V
E_s_b_e1e2_sum = 0.5 * (S.^2 - S2);

E_eta_sq = sum(E_s_sq_b_sq, 3) + E_s_b_e1e2_sum; % N x V

E_alpha_sq = alpha_ast.^2 + alpha_varrho; % 1 x V
E_eta_tilde_sq = max(E_eta_sq + E_alpha_sq + 2 * alpha_ast .* E_eta, eps); % N x V
eta_ast = sqrt(E_eta_tilde_sq); % N x V
E_Omega = 1 ./ (2 * eta_ast) .* tanh(0.5 * eta_ast); % N x V

%% --- 6. Update variationals for hyperparameters ---
% Update q(sigma^2)
E_gamma = E_z' .* E_gamma_given_z; % (P x E)
E_mu_sq = E_z' .* E_mu_sq_given_z; % (P x E)
a_mu_ast = a_mu + 0.5 * sum(E_gamma, [1, 2]); % Updated shape parameter for sigma_mu
b_mu_ast = b_mu + 0.5 * sum(E_mu_sq, [1, 2]); % Updated rate parameter for sigma_mu
E_sigma2_inv = a_mu_ast / b_mu_ast; % Scalar expectation of 1/sigma^2
E_log_sigma2 = log(b_mu_ast) - psi(a_mu_ast); % Scalar expectation of log(sigma^2)

% Update q(nu)
a_nu_ast = a_nu + E_gamma; % Updated shape parameter for nu, P x E
b_nu_ast = b_nu + 1 - E_gamma; % Updated rate parameter for nu, P x E
a_nu_digamma = psi(a_nu_ast);
b_nu_digamma = psi(b_nu_ast);
a_plus_b_nu_digamma = psi(a_nu_ast + b_nu_ast);
E_nu_logit = a_nu_digamma - b_nu_digamma; % P x E
E_log_nu = a_nu_digamma - a_plus_b_nu_digamma; % P x E
E_log_1_minus_nu = b_nu_digamma - a_plus_b_nu_digamma; % P x E

% Update q(rho)
E_m = E_z' .* rho_ast; % (V x E)
a_rho_ast = a_rho + E_m; % Updated shape parameter for rho, V x E
b_rho_ast = b_rho + 1 - E_m; % Updated rate parameter for rho, V x E
a_rho_digamma = psi(a_rho_ast);
b_rho_digamma = psi(b_rho_ast);
a_plus_b_rho_digamma = psi(a_rho_ast + b_rho_ast);
E_rho_logit = a_rho_digamma - b_rho_digamma; % V x E
E_log_rho = a_rho_digamma - a_plus_b_rho_digamma; % V x E
E_log_1_minus_rho = b_rho_digamma - a_plus_b_rho_digamma; % V x E

% Update q(r)
a_r_ast = a_r + E_z; % Updated shape parameter for r, E x 1
b_r_ast = b_r + 1 - E_z; % Updated rate parameter for r, E x 1
E_r_logit = psi(a_r_ast) - psi(b_r_ast); % E x 1
E_log_r = psi(a_r_ast) - psi(a_r_ast + b_r_ast) ; % E x 1
E_log_1_minus_r = psi(b_r_ast) - psi(a_r_ast + b_r_ast) ; % E x 1



end