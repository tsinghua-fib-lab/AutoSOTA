clear all
clc
close all
rng(7) % 7
%%
% Define grid for (mu, sigma)
mu_vals = linspace(-20, 25, 500);
sigma_vals = linspace(0.01, 40, 500);
[MU, SIGMA] = meshgrid(mu_vals, sigma_vals);

% Define list of reference distributions (mu0, sigma0)
mu_all = -10:10:10;
sigma_all = 10*rand(1,size(mu_all,2));

mu_all = [-10,0,15];
sigma_all = [0.8,6.01,4.01];
ref_params = [mu_all',sigma_all'];
% 
%%
colors = lines(size(ref_params, 1));  % Unique colors for each ref

% Plot KL Divergence 1-balls
figure()
hold on
% subplot(1,2,1); hold on;
% title('KL Divergence 1-Balls');
xlabel('$m$', 'Interpreter', 'latex'); ylabel('\sigma');
grid on; 
% ylim([0.1 3]);

for i = 1:size(ref_params,1)
    mu0 = ref_params(i,1);
    sigma0 = ref_params(i,2);

    % KL divergence formula
    % KL = log(SIGMA / sigma0) + (sigma0^2 + (MU - mu0).^2) ./ (2 * SIGMA.^2) - 0.5;
    KL = KL_eval(MU,SIGMA,mu0,sigma0);
    WKL = WKL_eval(mu0,sigma0,MU,SIGMA);
    KW = KW_eval(mu0,sigma0,MU,SIGMA,1e-2);

        % Plot grey shaded region for WKL <= 1
    WKL_mask = WKL <= 1;
    KL_mask = KL <=1;
    KW_mask = KW<=1;

    % Use a light grey colormap (for shading effect)
    gray_shade = 0.88 * ones(1,3); % light grey
    gray_shade_KL = 0.7 * ones(1,3); % light grey
    gray_shade_KW = 0.98 * ones(1,3); % light grey
    
    % Plot filled contour (shaded region)
    if i ==1
        contourf(MU, SIGMA, double(WKL_mask), [0.1 0.9], 'FaceColor', gray_shade_KW, 'EdgeColor', 'none');
        contourf(MU, SIGMA, double(WKL_mask), [0.1 0.9], 'FaceColor', gray_shade, 'EdgeColor', 'none');
        contourf(MU, SIGMA, double(KL_mask), [0.1 0.9], 'FaceColor', gray_shade_KL, 'EdgeColor', 'none');
    else        
        contourf(MU, SIGMA, double(KW_mask), [0.1 0.9], 'FaceColor', gray_shade_KW, 'EdgeColor', 'none');
        contourf(MU, SIGMA, double(KL_mask), [0.1 0.9], 'FaceColor', gray_shade_KL, 'EdgeColor', 'none');
        contourf(MU, SIGMA, double(WKL_mask), [0.1 0.9], 'FaceColor', gray_shade, 'EdgeColor', 'none');
    end
    
   
    
    % Plot KL divergence dashed line
    % contour(MU, SIGMA, KL, [1 1], 'k:', 'LineWidth', 1.5);

    plot(mu0, sigma0, 'ko', 'MarkerSize', 5, 'MarkerFaceColor', 'black', 'MarkerEdgeColor', 'k')
    % legend_entries{i} = sprintf('\\mu_0=%.1f, \\sigma_0=%.1f', mu0, sigma0);

    % Plot contour where KL divergence = 1
    contour(MU, SIGMA, KL, [1 1],'k:', 'LineWidth', 1.5);
    contour(MU, SIGMA, WKL, [1 1],'k','LineWidth', 1.5);
    contour(MU, SIGMA, KW, [1 1],'k-.','LineWidth', 2);

    % text(mu0 + 0, sigma0 + 0.5, ...
    % sprintf('$\\ (m_0,\\ \\sigma_0) $'), ...
    % 'Interpreter', 'latex', 'FontSize', 12);
end
% legend(legend_entries, 'Location', 'northeast');
% Dummy plots for legend
hKL = plot(NaN, NaN, 'k:', 'LineWidth', 1.5);  % KL (dashed)
hWKL = plot(NaN, NaN, 'k-', 'LineWidth', 1.5);   % WKL (solid)
hKW = plot(NaN, NaN, 'k-.', 'LineWidth', 2);   % KW (solid)
% % legend([h1 h2], {'D_{KL}=1$', 'D_{WKL}=1'}, 'Location', 'northeast');
% legend([h1 h2], {'$D_{\mathrm{KL}} = 1$', '$D_{\mathrm{WKL}} = 1$'}, ...
%        'Location', 'northeast', 'Interpreter', 'latex');

hKLpatch = patch(NaN, NaN, gray_shade_KL);  % KL region
hWKLpatch = patch(NaN, NaN, gray_shade);     % WKL region
hKWpatch = patch(NaN, NaN, gray_shade_KW); 

lgd = legend([hKL hKLpatch hWKL hWKLpatch hKW hKWpatch], ...
    {'$D^{\mathrm{KL}}(\mathcal{N}(m,\sigma)|\mathcal{N}(m_0,\sigma_0)) = 1$', ...
     '$D^{\mathrm{KL}}(\mathcal{N}(m,\sigma)|\mathcal{N}(m_0,\sigma_0)) \leq 1$', ...
     '$D^{\mathrm{WKL}}(\mathcal{N}(m_0,\sigma_0)|\mathcal{N}(m,\sigma)) = 1$', ...
     '$D^{\mathrm{WKL}}(\mathcal{N}(m_0,\sigma_0)|\mathcal{N}(m,\sigma)) \leq 1$', ...
     '$D^{\mathrm{KW}_{\lambda=1}}(\mathcal{N}(m_0,\sigma_0)|\mathcal{N}(m,\sigma)) = 1$', ...
     '$D^{\mathrm{KW}_{\lambda=1}}(\mathcal{N}(m_0,\sigma_0)|\mathcal{N}(m,\sigma)) \leq 1$'}, ...
    'Location','northoutside', ...
    'NumColumns',3, ...
    'Interpreter','latex');

lgd.FontSize = 12;  % Example: set font size to 14

ylim([0 18])
xlim([-15 22])

%% User-defined functions

function KL = KL_eval(mu0,sigma0,mu1,sigma1)
KL = log(sigma1 ./ sigma0) + (sigma0.^2 + (mu1 - mu0).^2) ./ (2 * sigma1.^2) - 0.5;
end

function WKL = WKL_eval(mu0,sigma0,mu1,sigma1)
WKL = (sigma0.^2-sigma1.^2 +sigma1.^2.*log(sigma1.^2 / sigma0.^2))./ (4*(sigma1-sigma0).^2).*((sigma1-sigma0).^2 + (mu1-mu0).^2);
end

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


