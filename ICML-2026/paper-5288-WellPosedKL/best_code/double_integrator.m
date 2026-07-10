% ===============================================================
% MATLAB script for KL vs WKL regularized LQR (double integrator)
% Overlays KW for multiple lambda values with green shades
% ===============================================================

clear all; close all; clc;
seed_id = 2;

% System matrices (double integrator)
A = [1 1; 0 1];
B = [0; 1];
Q = eye(2); % LQR tuning weight
Sigma0 = eye(2); % Initial condition covariance
gamma = 0.9; % Discount factor in the LQR cost

% Range of noise variances rho
rho_vals = logspace(-10, 3, 20);  % from very small to moderately large
n_rho = length(rho_vals);

% Noiminal covariance matrix
Sigma_w0 = eye(2);

% Control cost weights (KL and WKL do not depend on lambda)
R_KL_func  = @(Sigma_w) B' * (Sigma_w \ B);   % = B^T Sigma_w^{-1} B
R_WKL_func = @(Sigma_w) B' * B;               % = B^T B

% Lambda values requested for KW (we will overlay these as green shades)
lambda_vals = [1e-4, 1e-2, 1e-1, 1];

n_lambda = length(lambda_vals);

% Pre-generate a set of green shades (from light to dark)
% Create RGB triplets for green shades
% start (light green) -> end (dark green)
greens = zeros(n_lambda,3);
for k = 1:n_lambda
    t = (k-1)/(n_lambda-1); % 0..1
    % interpolate between a light green and a dark green
    light = [0.75, 1.0, 0.75];
    dark  = [0.0, 0.45, 0.0];
    greens(k,:) = (1-t)*light + t*dark;
end

% Compute KL and WKL once (lambda-independent)
[V_KL, F_KL, spec_rad_KL] = solve_OCP(A,B,Q,R_KL_func,Sigma0,Sigma_w0,gamma,rho_vals);
[V_WKL, F_WKL, spec_rad_WKL] = solve_OCP(A,B,Q,R_WKL_func,Sigma0,Sigma_w0,gamma,rho_vals);

% Preallocate storage for KW results across lambdas
V_KW_all      = zeros(n_rho, n_lambda);
F_KW_all      = zeros(n_rho, 2, n_lambda);
spec_rad_KW_all = zeros(n_rho, n_lambda);

% Compute KW for each lambda
for idx = 1:n_lambda
    lambda_reg = lambda_vals(idx);
    R_KW_func = @(Sigma_w) B' * ((Sigma_w + lambda_reg*eye(size(Sigma_w))) \ B);
    [Vtmp, Ftmp, spectmp] = solve_OCP(A,B,Q,R_KW_func,Sigma0,Sigma_w0,gamma,rho_vals);
    V_KW_all(:,idx) = Vtmp;
    F_KW_all(:,:,idx) = Ftmp;
    spec_rad_KW_all(:,idx) = spectmp;
end

% ---------------------------------------------------------------
% Figure 1: Feedback gains vs noise variance rho
% ---------------------------------------------------------------
figure;
% KL (two state components)
semilogx(rho_vals, F_KL(:,1), 'r-o', 'LineWidth', 2); hold on;
semilogx(rho_vals, F_KL(:,2), 'r--o', 'LineWidth', 2);  % keep different style for the two components
% WKL (two components)
semilogx(rho_vals, F_WKL(:,1), 'b-s', 'LineWidth', 2);
semilogx(rho_vals, F_WKL(:,2), 'b--s', 'LineWidth', 2);

% KW overlays for each lambda (two components each) using green shades
% Keep dashed style similar to your original 'g--s' but vary shade
h_handles = []; h_labels = {};
for idx = 1:n_lambda
    c = greens(idx,:);
    % component 1
    hh1 = semilogx(rho_vals, squeeze(F_KW_all(:,1,idx)), '-', 'LineWidth', 1.5, 'Color', c);
    % component 2
    hh2 = semilogx(rho_vals, squeeze(F_KW_all(:,2,idx)), '--', 'LineWidth', 1.5, 'Color', c);
    h_handles = [h_handles; hh1; hh2];
    h_labels{end+1} = sprintf('F_{KW}(1), \\lambda=%.0e', lambda_vals(idx));
    h_labels{end+1} = sprintf('F_{KW}(2), \\lambda=%.0e', lambda_vals(idx));
end

xlabel('\rho (noise variance)');
ylabel('Feedback gain F');
% Build legend: KL(1),KL(2),WKL(1),WKL(2), then KW entries
legend_entries = [{'F_{KL}(1)'}, {'F_{KL}(2)'}, {'F_{WKL}(1)'}, {'F_{WKL}(2)'} , h_labels];
legend(legend_entries, 'Location', 'best');
title('Feedback gains vs noise variance');
grid on;

% ---------------------------------------------------------------
% Figure 2: Cost vs noise variance rho (log-log scale)
% ---------------------------------------------------------------
figure;
loglog(rho_vals, V_KL, 'r-o', 'LineWidth', 2); hold on;
loglog(rho_vals, V_WKL, 'b-s', 'LineWidth', 2);

h_cost = [];
labels_cost = {};
for idx = 1:n_lambda
    c = greens(idx,:);
    h_temp = loglog(rho_vals, V_KW_all(:,idx), '-', 'LineWidth', 1.8, 'Color', c);
    h_cost = [h_cost; h_temp];
    labels_cost{end+1} = sprintf('V_{KW}, \\lambda=%.0e', lambda_vals(idx));
end

xlabel('\rho (noise variance)');
ylabel('Cost');
legend(['V_{KL}', 'V_{WKL}', labels_cost], 'Location','best');
title('V vs noise variance (semi-log scale)');
grid on;

% ---------------------------------------------------------------
% Figure 3: Closed-loop spectral radius vs noise variance rho
% ---------------------------------------------------------------
figure;
semilogx(rho_vals, spec_rad_KL, 'r-o', 'LineWidth', 2); hold on;
semilogx(rho_vals, spec_rad_WKL, 'b-s', 'LineWidth', 2);

h_spec = [];
labels_spec = {};
for idx = 1:n_lambda
    c = greens(idx,:);
    h_temp = semilogx(rho_vals, spec_rad_KW_all(:,idx), '-', 'LineWidth', 1.8, 'Color', c);
    h_spec = [h_spec; h_temp];
    labels_spec{end+1} = sprintf('\\rho_{KW}, \\lambda=%.0e', lambda_vals(idx));
end

xlabel('\rho (noise variance)');
ylabel('spectral radius of closed-loop');
legend(['spectral radius_{KL}', 'spectral radius_{WKL}', labels_spec], 'Location', 'best');
title('Spectral radius vs noise variance');
grid on;

%%
% ---------------------------------------------------------------
% Figure 4: Closed-loop trajectories for different rho (overlay KW lambdas)
% ---------------------------------------------------------------
rho_demo = [1e-4,1e-3, 1e-2, 1e-1];  % pick small and moderate noise for illustration
T = 300;                             % time horizon
rng(seed_id); x0 = 1*(-0.5+rand(2,1)); % initial condition (generate once)
figure;

% Will collect legend handles for top-line legend
legend_handles = []; legend_labels = {};

for j = 1:length(rho_demo)
    rho = rho_demo(length(rho_demo)-j+1);
    Sigma_w = rho * eye(2);

    % simulate KL and WKL (they are independent of lambda)
    x_KL = simulate_opt_closed_loop(A,B,Q,R_KL_func,x0,T,Sigma_w,gamma,seed_id);
    x_WKL = simulate_opt_closed_loop(A,B,Q,R_WKL_func,x0,T,Sigma_w,gamma,seed_id);

    % Subplot
    subplot(2,2,j)
    h1 = plot(0:T, x_KL(1,:), 'r-', 'LineWidth', 1.5); hold on;
    h2 = plot(0:T, x_WKL(1,:), 'b-', 'LineWidth', 1.5);

    % store legend handles first time through
    if j==1
        legend_handles = [legend_handles; h1; h2];
        legend_labels = [{'KL'}, {'WKL'}];
    end

    % now overlay KW trajectories for each lambda (green shades)
    for idx = 1:n_lambda
        lambda_reg = lambda_vals(idx);
        R_KW_func = @(Sigma_w) B' * ((Sigma_w + lambda_reg*eye(size(Sigma_w))) \ B);
        x_KW = simulate_opt_closed_loop(A,B,Q,R_KW_func,x0,T,Sigma_w,gamma,seed_id);
        h_kw = plot(0:T, x_KW(1,:), '-', 'LineWidth', 1.1, 'Color', greens(idx,:));
        if j==1
            legend_handles = [legend_handles; h_kw];
            legend_labels{end+1} = sprintf('KW, \\lambda=%.0e', lambda_reg);
        end
    end

    xlabel('Time step');
    ylabel('$q_t$', 'Interpreter','latex');
    title(sprintf('$\\rho = %.0e$', rho), 'Interpreter','latex');
    grid on;
end

% Add single legend on top of figure
lgd = legend(legend_handles, legend_labels, 'Orientation', 'horizontal');
lgd.Position = [0.15 0.97 0.7 0.05];  % Adjust position (x,y,width,height)

% ===========================
% User-defined functions
% ===========================
function [V,F,spec_rad] = solve_OCP(A,B,Q,R_func,Sigma0,Sigma_w0,gamma,rho_vals)
    n_rho = length(rho_vals);
    
    % Storage for feedback gains
    F  = zeros(n_rho,2);
    
    % Storage for Optimal cost 
    V  = zeros(n_rho,1);
    
    % Storage for spectral radius
    spec_rad  = zeros(n_rho,1);
    
    % Compute optimal gains, optimal cost and closed-loop poles for each rho
    for i = 1:n_rho
        rho = rho_vals(i);
        Sigma_w = rho * Sigma_w0;
        R = R_func(Sigma_w);
        
        % Solve Riccati equations
        A_gamma = sqrt(gamma)*A;
        R_gamma = (1/gamma)*R;
        
        P = idare(A_gamma, B, Q, R_gamma);
        
        % Compute the optimal cost   
        V(i,1) = trace(P*Sigma0) + gamma/(1-gamma) * trace(Sigma_w*P);
    
        % Feedback gains/ control laws
        F(i,:)  = - gamma * ((R + gamma*B' * P * B) \ (B' * P * A));
       
        % Spectral radius of the closed-loop
        spec_rad(i,1)  = max(abs(eig(A + B * F(i,:))));
    end
end

function x = simulate_opt_closed_loop(A,B,Q,R_func,x0,T,Sigma_w,gamma,seed_id)
    rng(seed_id)
    % Control cost weights
    R  = R_func(Sigma_w);
    
    % Solve Riccati equations
    A_gamma = sqrt(gamma)*A;
    R_gamma = (1/gamma)*R;
    P = idare(A_gamma, B, Q, R_gamma);
    
    % Feedback gains
    F_demo  = - gamma * ((R + gamma*B' * P * B) \ (B' * P * A));

    % Simulate trajectories
    x  = zeros(2, T+1);
    x(:,1)  = x0;
    
    for t = 1:T
        u = F_demo * x(:,t);
        w = mvnrnd([0;0], Sigma_w)'; % noise
        x(:,t+1) = A * x(:,t) + B * u + w;
    end
end
