clc;
clear;
close all;

rng(1);

% ===============================================================
% 1D double integrator and 2D extension via Kronecker products
% ===============================================================

A1 = [1 1;
      0 1];

B1 = [0;
      1];

I2 = eye(2);

% 2D double integrator: state = [q1, v1, q2, v2]
% input = [u1, u2]
A = kron(I2, A1);   % 4x4
B = kron(I2, B1);   % 4x2

n = size(A,1);   % 4
m = size(B,2);   % 2

Q = eye(n);
Sigma0 = eye(n);

gamma = 0.9;
lambda_reg = 1.0;

% ===============================================================
% Process noise:
% low in one spatial dimension, high in the other
% ===============================================================

Sigma_w0 = diag([1e-3, 1e-3, 1e3, 1e3]);

% Range of noise scaling
rho_vals = logspace(-2, 0, 10);
n_rho = length(rho_vals);

% ===============================================================
% Storage
% ===============================================================

F_KL   = zeros(n_rho, m, n);
F_WKL  = zeros(n_rho, m, n);
F_KWKL = zeros(n_rho, m, n);

V_KL   = zeros(n_rho,1);
V_WKL  = zeros(n_rho,1);
V_KWKL = zeros(n_rho,1);

spec_rad_KL   = zeros(n_rho,1);
spec_rad_WKL  = zeros(n_rho,1);
spec_rad_KWKL = zeros(n_rho,1);

% ===============================================================
% Compute optimal gains for each rho
% ===============================================================

for i = 1:n_rho

    rho = rho_vals(i);

    Sigma_w = rho * Sigma_w0;

    % Control cost weights
    R_KL   = B' * inv(Sigma_w) * B;
    R_WKL  = B' * B;
    R_KWKL = B' * inv(Sigma_w + lambda_reg * eye(n)) * B;

    % Solve discounted LQR
    [P_KL,   Ftmp] = discounted_lqr_gain(A, B, Q, R_KL, gamma);
    F_KL(i,:,:) = Ftmp;

    [P_WKL,  Ftmp] = discounted_lqr_gain(A, B, Q, R_WKL, gamma);
    F_WKL(i,:,:) = Ftmp;

    [P_KWKL, Ftmp] = discounted_lqr_gain(A, B, Q, R_KWKL, gamma);
    F_KWKL(i,:,:) = Ftmp;

    % Optimal costs
    V_KL(i) = trace(P_KL * Sigma0) + ...
              gamma/(1-gamma) * trace(Sigma_w * P_KL);

    V_WKL(i) = trace(P_WKL * Sigma0) + ...
               gamma/(1-gamma) * trace(Sigma_w * P_WKL);

    V_KWKL(i) = trace(P_KWKL * Sigma0) + ...
                gamma/(1-gamma) * trace(Sigma_w * P_KWKL);

    % Spectral radius of closed-loop system
    Acl_KL   = A + B * squeeze(F_KL(i,:,:));
    Acl_WKL  = A + B * squeeze(F_WKL(i,:,:));
    Acl_KWKL = A + B * squeeze(F_KWKL(i,:,:));

    spec_rad_KL(i)   = max(abs(eig(Acl_KL)));
    spec_rad_WKL(i)  = max(abs(eig(Acl_WKL)));
    spec_rad_KWKL(i) = max(abs(eig(Acl_KWKL)));

end

% ===============================================================
% Figure 1: Feedback gains vs noise variance
% ===============================================================

state_labels = {'q_1', 'v_1', 'q_2', 'v_2'};
input_labels = {'u_1', 'u_2'};

figure('Position',[100 100 1600 700]);

for ui = 1:m
    for xi = 1:n

        subplot(m,n,(ui-1)*n + xi);
        hold on;

        semilogx(rho_vals, squeeze(F_KL(:,ui,xi)), ...
            'r-o', 'LineWidth',1.2);

        semilogx(rho_vals, squeeze(F_WKL(:,ui,xi)), ...
            'b-s', 'LineWidth',1.2);

        semilogx(rho_vals, squeeze(F_KWKL(:,ui,xi)), ...
            'g-^', 'LineWidth',1.2);

        grid on;

        title([input_labels{ui}, ' vs ', state_labels{xi}]);

        if ui == m
            xlabel('\rho');
        end

        if xi == 1
            ylabel('gain');
        end

    end
end

legend('KL','WKL','KWKL');
sgtitle('Feedback gains vs noise variance');

% ===============================================================
% Figure 2: Cost vs noise variance
% ===============================================================

figure;

semilogx(rho_vals, V_KL, 'r-o', 'LineWidth',1.5);
hold on;

semilogx(rho_vals, V_WKL, 'b-s', 'LineWidth',1.5);

semilogx(rho_vals, V_KWKL, 'g-^', 'LineWidth',1.5);

xlabel('\rho');
ylabel('Cost');

grid on;

legend('V_{KL}','V_{WKL}','V_{KWKL}');
title('Cost vs noise variance');

% ===============================================================
% Figure 3: Closed-loop spectral radius vs noise variance
% ===============================================================

figure;

semilogx(rho_vals, spec_rad_KL, 'r-o', 'LineWidth',1.5);
hold on;

semilogx(rho_vals, spec_rad_WKL, 'b-s', 'LineWidth',1.5);

semilogx(rho_vals, spec_rad_KWKL, 'g-^', 'LineWidth',1.5);

xlabel('\rho');
ylabel('Spectral radius');

grid on;

legend('spec\_rad\_KL', ...
       'spec\_rad\_WKL', ...
       'spec\_rad\_KWKL');

title('Closed-loop spectral radius vs noise variance');

% ===============================================================
% Figure 4: Closed-loop trajectories
% ===============================================================

rho_demo = [1e-4, 1e-3, 1e-2, 1e-1];

T = 300;

x0 = -0.5 + rand(n,1);

figure('Position',[100 100 1200 800]);

for j = 1:length(rho_demo)

    rho = rho_demo(end-j+1);

    Sigma_w = rho * Sigma_w0;

    % Control cost weights
    R_KL   = B' * inv(Sigma_w) * B;
    R_WKL  = B' * B;
    R_KWKL = B' * inv(Sigma_w + lambda_reg * eye(n)) * B;

    % Gains
    [~, F_KL_demo]   = discounted_lqr_gain(A, B, Q, R_KL, gamma);
    [~, F_WKL_demo]  = discounted_lqr_gain(A, B, Q, R_WKL, gamma);
    [~, F_KWKL_demo] = discounted_lqr_gain(A, B, Q, R_KWKL, gamma);

    % Simulate
    x_KL   = zeros(n, T+1);
    x_WKL  = zeros(n, T+1);
    x_KWKL = zeros(n, T+1);

    x_KL(:,1)   = x0;
    x_WKL(:,1)  = x0;
    x_KWKL(:,1) = x0;

    for t = 1:T

        % Same disturbance for all policies
        w = mvnrnd(zeros(n,1), Sigma_w)';

        u_KL   = F_KL_demo * x_KL(:,t);
        u_WKL  = F_WKL_demo * x_WKL(:,t);
        u_KWKL = F_KWKL_demo * x_KWKL(:,t);

        x_KL(:,t+1)   = A*x_KL(:,t)   + B*u_KL   + w;
        x_WKL(:,t+1)  = A*x_WKL(:,t)  + B*u_WKL  + w;
        x_KWKL(:,t+1) = A*x_KWKL(:,t) + B*u_KWKL + w;

    end

    subplot(2,2,j);

    plot(x_KL(1,:), 'r-', 'LineWidth',1.5);
    hold on;

    plot(x_WKL(1,:), 'b-', 'LineWidth',1.5);

    plot(x_KWKL(1,:), 'g-', 'LineWidth',1.5);

    xlabel('Time step');
    ylabel('q_1');

    title(sprintf('\\rho = %.0e', rho));

    grid on;

end

legend('KL policy', 'WKL policy', 'KWKL policy');

sgtitle('Closed-loop trajectories');

% ===============================================================
% Local function
% ===============================================================

function [P, F] = discounted_lqr_gain(A, B, Q, R, gamma)

    % Convert discounted LQR into undiscounted DARE
    A_gamma = sqrt(gamma) * A;
    R_gamma = R / gamma;

    % Solve DARE
    P = dare(A_gamma, B, Q, R_gamma);

    % Feedback gain
    F = -gamma * ((R + gamma * (B' * P * B)) \ (B' * P * A));

end