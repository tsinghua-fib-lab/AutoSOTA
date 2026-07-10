clear all
clc
close all
rng(7)

%% ============================================================
% Grid for (m, sigma)
%% ============================================================
mu_vals    = linspace(-20, 25, 500);
sigma_vals = linspace(0.01, 40, 500);
[MU, SIGMA] = meshgrid(mu_vals, sigma_vals);

%% ============================================================
% Reference Gaussians
%% ============================================================
mu_all    = [-10, 0, 15];
sigma_all = [0.8, 6.01, 4.01];
ref_params = [mu_all', sigma_all'];
nRef = size(ref_params,1);

%% ============================================================
% Lambda values for KW
%% ============================================================
lambda_vals = [1e-2 1e-1 1e0 1e1 1e2];
nLambda = numel(lambda_vals);

% Colors and line styles for lambdas
lambda_colors = parula(nLambda);       % distinguish λ with color
lambda_styles = {'-', '--', '-.', ':', '-'}; % different line styles

%% ============================================================
% Figure setup
%% ============================================================
figure('Color','w');
hold on
grid on
xlabel('$m$','Interpreter','latex')
ylabel('$\sigma$','Interpreter','latex')
xlim([-15 22])
ylim([0 18])

%% ============================================================
% Loop over reference Gaussians
%% ============================================================
for i = 1:nRef
    mu0    = ref_params(i,1);
    sigma0 = ref_params(i,2);

    % Optional: light shaded region for λ = smallest value
    lambda0 = lambda_vals(1);
    KW0 = KW_eval(mu0, sigma0, MU, SIGMA, lambda0);
    KW_mask = KW0 <= 1;
    contourf(MU, SIGMA, double(KW_mask), [0.1 0.9], ...
        'FaceColor', 0.95*ones(1,3), 'EdgeColor','none');
    
    % Reference point
    plot(mu0, sigma0, 'ko', 'MarkerFaceColor','k', 'MarkerSize',5)

    % Plot KW = 1 contours for all lambdas
    for l = 1:nLambda
        lambda = lambda_vals(l);
        KW = KW_eval(mu0, sigma0, MU, SIGMA, lambda);
        contour(MU, SIGMA, KW, [1 1], ...
            'LineWidth', 2, ...
            'LineStyle', lambda_styles{l}, ...
            'Color', lambda_colors(l,:));
    end
end

%% ============================================================
% Legend
%% ============================================================
hLambda = gobjects(nLambda,1);
for l = 1:nLambda
    hLambda(l) = plot(NaN, NaN, ...
        'LineWidth',2, ...
        'LineStyle', lambda_styles{l}, ...
        'Color', lambda_colors(l,:));
end

legend_strings = arrayfun(@(x) ...
    sprintf('$\\lambda=10^{%d}$', round(log10(x))), ...
    lambda_vals, 'UniformOutput', false);

legend(hLambda, legend_strings, 'Interpreter','latex', ...
       'Location','northoutside','NumColumns', nLambda);

title('KW Divergence Balls for Different \lambda','Interpreter','latex','FontSize',14)

function D = KW_eval(mu0, sigma0, mu1, sigma1, lambda)
%KW_EVAL  Vectorized Kalman–Wasserstein divergence for univariate Gaussians
%
%   D = KW_eval(mu0, sigma0, mu1, sigma1, lambda)
%
%   Inputs may be scalars or arrays (e.g. meshgrid outputs).
%   sigma0, sigma1 are standard deviations (> 0).
%   lambda may be scalar or array (implicit expansion supported).
%
%   Output:
%   D has the broadcasted size of the inputs.

    % --- parameters ---
    kappa_g = 1;
    tol = 1e-6;

    % --- variances ---
    v0 = sigma0.^2;
    v1 = sigma1.^2;

    % --- regularized variances ---
    v0_l = kappa_g .* v0 + lambda;
    v1_l = kappa_g .* v1 + lambda;

    % --- mean difference ---
    dm = mu1 - mu0;

    % --- equal-variance mask ---
    eq = abs(v1 - v0) < tol;

    % ================================================================
    % Case 1: v1 == v0   (computed everywhere, then masked)
    % ================================================================
    D_eq = 0.5 .* (dm.^2) ./ v0_l;

    % ================================================================
    % Case 2: v1 ~= v0   (computed everywhere, then masked)
    % ================================================================

    % log terms
    log_v_ratio  = log(v1 ./ v0);
    log_vl_ratio = log(v0_l ./ v1_l);

    % Xi term
    Xi = v1 .* log_v_ratio ...
       + (1 ./ kappa_g) .* v1_l .* log_vl_ratio;

    % variance–log contribution
    % term_var = (1 ./ (4 .* lambda)) .* ...
    %     (v1_l .* log_vl_ratio + kappa_g .* v1 .* log_v_ratio);

    term_var = (1 ./ (4 .* lambda)) .* kappa_g*...
        Xi;

    % mean contribution
    denom = (sigma1 - sigma0).^2;   % = (sqrt(v1)-sqrt(v0)).^2
    term_mean = (1 ./ (4 .* lambda)) .* (dm.^2 ./ denom) .* Xi;

    D_neq = term_var + term_mean;

    % ================================================================
    % Blend the two cases (no logical indexing!)
    % ================================================================
    D = eq .* D_eq + (~eq) .* D_neq;

end



