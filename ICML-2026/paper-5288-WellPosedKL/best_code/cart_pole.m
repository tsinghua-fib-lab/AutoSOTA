% cartpole_plots_and_anim.m
% Complete MATLAB script that:
%  - computes LQR gains for three R choices
%  - simulates linear and nonlinear cart-pole closed-loop
%  - generates publication-style plots
%  - animates the nonlinear cart-pole trajectory (and can save GIF)
%
% Usage: run this script in MATLAB. The helper functions are included at
% the end of the file as local functions.

clear; clc; close all;

%% -------------------------
%  Publication-style settings
%  -------------------------
set(groot,'defaultAxesFontName','Times');
set(groot,'defaultAxesFontSize',10);
set(groot,'defaultLineLineWidth',1);
set(groot,'defaultFigureColor','w');

FIG_DIR = 'figures';
if ~exist(FIG_DIR,'dir'), mkdir(FIG_DIR); end

%% =========================
% Cart-Pole Parameters
% =========================
g = 9.81;
m_c = 1.0;
m_p = 0.1;
l = 0.5;
dt = 0.01;

%% =========================
% Linearized Continuous-Time Dynamics
% (small-angle linearization about the upright pole)
% =========================
A_c = [0 1 0 0;
       0 0 (m_p*g)/m_c 0;
       0 0 0 1;
       0 0 g*(m_c+m_p)/(l*m_c) 0];

B_c = [0;
       1/m_c;
       0;
       1/(l*m_c)];

C_c = eye(4);
D_c = zeros(4,1);

%% =========================
% Discretization
% =========================
sysc = ss(A_c, B_c, C_c, D_c);
sysd = c2d(sysc, dt, 'zoh');
A_d = sysd.A;
B_d = sysd.B;

%% =========================
% Noise Model (discrete-time additive noise)
% - If you want continuous-time noise instead, change the interpretation
%   and scale accordingly. Default chosen to avoid ill-conditioning.
% =========================
noise_scale = 1e-3;   % tune this (avoid extremely small values like 1e-10)
noise_cov = noise_scale * eye(4);

%% =========================
% LQR Weights
% =========================
Q = eye(4);
% Regularization to avoid ill-conditioning when inverting noise_cov
eps_reg = 1e-9;
R_KL   = B_d' * ((noise_cov) \ B_d);
R_WKL  = B_d' * B_d;
lambda_reg = 1.0;
R_KWKL = B_d' * ((noise_cov + lambda_reg*eye(4)) \ B_d);

gamma = 0.99;
K_KL   = lqr_gain(A_d, B_d, Q, R_KL,gamma);
K_WKL  = lqr_gain(A_d, B_d, Q, R_WKL,gamma);
K_KWKL = lqr_gain(A_d, B_d, Q, R_KWKL,gamma);

disp('LQR gains:')
disp(['K_KL   = ', mat2str(K_KL, 6)])
disp(['K_WKL  = ', mat2str(K_WKL, 6)])
disp(['K_KWKL = ', mat2str(K_KWKL, 6)])

%% =========================
% Simulation Params
% =========================
T = 100.0;            % shorter default to keep animation quick (s)
N = round(T/dt);

rng(0);               % reproducible initial condition and noise
x0 = -0.5 + rand(4,1);


%% Run sims (linear and nonlinear)
% Use identical initial condition for fair comparison. We'll use the same
% discrete-time noise covariance for both sims (added to the state update).

% Simulate linear closed-loop (discrete-time)
[x_lin_KL,  u_lin_KL]  = simulate_linear(A_d,B_d,K_KL, N,dt,x0,noise_cov);
[x_lin_WKL, u_lin_WKL] = simulate_linear(A_d,B_d,K_WKL,N,dt,x0,noise_cov);
[x_lin_KWKL,u_lin_KWKL]= simulate_linear(A_d,B_d,K_KWKL,N,dt,x0,noise_cov);

% Simulate nonlinear closed-loop (full continuous dynamics with Euler)
[x_nl_KL,  u_nl_KL]  = simulate_nonlinear(K_KL,  N,dt,noise_cov,x0);
[x_nl_WKL, u_nl_WKL] = simulate_nonlinear(K_WKL, N,dt,noise_cov,x0);
[x_nl_KWKL,u_nl_KWKL]= simulate_nonlinear(K_KWKL,N,dt,noise_cov,x0);

stacked_cartpole_animation( ...
    {x_nl_KL, x_nl_WKL, x_nl_KWKL}, ...
    {u_nl_KL, u_nl_WKL, u_nl_KWKL}, ...
    dt, [true true true], l, ...
    {'KL','WKL','KWKL'} );

% extract angles
theta_lin_KL = x_lin_KL(3,:);
theta_lin_WKL = x_lin_WKL(3,:);
theta_lin_KWKL = x_lin_KWKL(3,:);

theta_nl_KL = wrap_to_pi(x_nl_KL(3,:) - pi);
theta_nl_WKL = wrap_to_pi(x_nl_WKL(3,:) - pi);
theta_nl_KWKL = wrap_to_pi(x_nl_KWKL(3,:) - pi);

time = (0:N-1)*dt;
%% Figure 1: Linear closed-loop
fig1 = figure('Position',[100 100 600 650]);
tiledlayout(3,1, 'TileSpacing','compact');

nexttile; hold on; grid on;
plot(time, theta_lin_KL, 'DisplayName','R\_KL');
plot(time, theta_lin_WKL, '--', 'DisplayName','R\_WKL');
plot(time, theta_lin_KWKL, ':', 'DisplayName','R\_KWKL');
ylabel('Pole angle (rad)');
title('Linear closed-loop (small-angle model)');
legend();

nexttile; hold on; grid on;
plot(time, x_lin_KL(1,:), 'DisplayName','R\_KL');
plot(time, x_lin_WKL(1,:), '--', 'DisplayName','R\_WKL');
plot(time, x_lin_KWKL(1,:), ':', 'DisplayName','R\_KWKL');
ylabel('Cart position (m)');

nexttile; hold on; grid on;
plot(time(1:end-1), u_lin_KL, 'DisplayName','R\_KL');
plot(time(1:end-1), u_lin_WKL, '--', 'DisplayName','R\_WKL');
plot(time(1:end-1), u_lin_KWKL, ':', 'DisplayName','R\_KWKL');
ylabel('Control (N)');
xlabel('Time (s)');

% save_fig(fig1, 'linear_closed_loop_timeseries');

%% Figure 2: Nonlinear closed-loop
fig2 = figure('Position',[100 100 600 650]);
tiledlayout(3,1, 'TileSpacing','compact');

nexttile; hold on; grid on;
plot(time, theta_nl_KL, 'DisplayName','R\_KL');
plot(time, theta_nl_WKL, '--', 'DisplayName','R\_WKL');
plot(time, theta_nl_KWKL, ':', 'DisplayName','R\_KWKL');
ylabel('Pole angle error (rad)');
title('Nonlinear closed-loop (full dynamics, upright = 0)');
legend();

nexttile; hold on; grid on;
plot(time, x_nl_KL(1,:), 'DisplayName','R\_KL');
plot(time, x_nl_WKL(1,:), '--', 'DisplayName','R\_WKL');
plot(time, x_nl_KWKL(1,:), ':', 'DisplayName','R\_KWKL');
ylabel('Cart position (m)');

nexttile; hold on; grid on;
plot(time(1:end-1), u_nl_KL, 'DisplayName','R\_KL');
plot(time(1:end-1), u_nl_WKL, '--', 'DisplayName','R\_WKL');
plot(time(1:end-1), u_nl_KWKL, ':', 'DisplayName','R\_KWKL');
ylabel('Control (N)');
xlabel('Time (s)');

% save_fig(fig2, 'nonlinear_closed_loop_timeseries');

%% Animate the nonlinear KWKL run (upright = pi convention)
% Speed and GIF options
animate_speed = 1.0;      % 1 = real-time
save_gif = false;
gif_name = fullfile(FIG_DIR, 'cartpole_kwkl.gif');

% call animator
animate_cartpole(x_nl_KWKL, u_nl_KWKL, dt, true, l, 'saveGif', save_gif, 'gifName', gif_name, 'speed', animate_speed);


%% -------------------------
% Run experiments over several noise scales and plot results
% Paste this after your Simulation Params (T, N, x0, dt defined)
% -------------------------

% noise scales to test (4 values -> 2x2 subplot)
noise_scales = [1e-3, 1e-4, 1e-5, 1e-6];

% controllers storage
K_all = cell(numel(noise_scales), 3);    % {scale}{controller} each K row vector (1x4)
Xnl_all = cell(numel(noise_scales), 3);  % store nonlinear trajectories
Unl_all = cell(numel(noise_scales), 3);

% Use same rng for reproducibility across experiments
rng(0);

for i = 1:numel(noise_scales)
    noise_scale = noise_scales(i);
    noise_cov = noise_scale * eye(4);

    % Build the three R variants exactly as in your script
    R_KL   = B_d' * ((noise_cov) \ B_d);
    R_WKL  = B_d' * B_d;
    lambda_reg = 1.0;
    R_KWKL = B_d' * ((noise_cov + lambda_reg*eye(4)) \ B_d);

    % compute gains (using your lqr_gain function and same Q,gamma)
    K_KL   = lqr_gain(A_d, B_d, Q, R_KL, gamma);
    K_WKL  = lqr_gain(A_d, B_d, Q, R_WKL, gamma);
    K_KWKL = lqr_gain(A_d, B_d, Q, R_KWKL, gamma);

    K_all{i,1} = K_KL;
    K_all{i,2} = K_WKL;
    K_all{i,3} = K_KWKL;

    % simulate nonlinear closed-loop for each controller
    % NOTE: simulate_nonlinear expects x0 (angle part will be shifted internally)
    [X1, U1] = simulate_nonlinear(K_KL,   N, dt, noise_cov, x0);
    [X2, U2] = simulate_nonlinear(K_WKL,  N, dt, noise_cov, x0);
    [X3, U3] = simulate_nonlinear(K_KWKL, N, dt, noise_cov, x0);

    Xnl_all{i,1} = X1;
    Xnl_all{i,2} = X2;
    Xnl_all{i,3} = X3;

    Unl_all{i,1} = U1;
    Unl_all{i,2} = U2;
    Unl_all{i,3} = U3;
end

% Time vector
time = (0:N-1) * dt;

%% Figure: 2x2 subplot of nonlinear cart position (x) for each noise scale
fig_pos = figure('Name','Nonlinear cart position vs noise scale','Position',[200 -200 900 700]);
tiledlayout(2,2,'TileSpacing','compact','Padding','compact');

controller_names = {'R\_KL','R\_WKL','R\_KWKL'};
plot_styles = {'-','--',':'};
colors = lines(3);

for i = 1:numel(noise_scales)
    nexttile;
    hold on; grid on;
    for c = 1:3
        Xsim = Xnl_all{i,c};
        plot(time, Xsim(1,:), plot_styles{c}, 'DisplayName', controller_names{c}, 'LineWidth', 1);
    end
    title(sprintf('noise scale = %.1e', noise_scales(i)));
    ylabel('Cart position x (m)');
    xlabel('Time (s)');
    legend('Location','best');
    xlim([0 time(end)]);
end

% save the subplot figure
% save_fig(fig_pos, 'nonlinear_cart_position_noisegrid');

%% -------------------------
% Local functions
% -------------------------

function K = lqr_gain(A, B, Q, R,gamma)
    % Discrete LQR via DARE
    [P,~,~] = dare(sqrt(gamma)*A, B, Q, 1/gamma*R);
    K = (R + gamma*B'*P*B) \ (gamma*B'*P*A);
end

%% Linear Simulation
function [X, U] = simulate_linear(A,B,K,N,dt,x0,noise_cov)
    % simulate discrete linear dynamics x_{k+1} = A x_k + B u_k + w_k
    X = zeros(4,N);
    U = zeros(1,N-1);
    X(:,1) = x0;
    % precompute noise cholesky
    L = chol(noise_cov,'lower');
    for k = 1:N-1
        u = -K * X(:,k);
        u = max(min(u,10), -10);   % actuator saturation (same used in nonlinear sim)
        w = L * randn(4,1);
        X(:,k+1) = A*X(:,k) + B*u + w;
        U(k) = u;
    end
end

%% Nonlinear Dynamics (deterministic derivative)
function dx = cartpole_nonlinear(x,u)
    x_pos = x(1); x_dot = x(2);
    theta = x(3); theta_dot = x(4);

    g = 9.81; m_c = 1.0; m_p = 0.1; l = 0.5;

    s = sin(theta); c = cos(theta);
    D = m_c + m_p*s^2;

    x_ddot = (u + m_p*s*(l*theta_dot^2 + g*c)) / D;
    theta_ddot = (-u*c - m_p*l*theta_dot^2*c*s - (m_c+m_p)*g*s) / (l*D);

    dx = [x_dot; x_ddot; theta_dot; theta_ddot];
end

function a = wrap_to_pi(a)
    a = mod(a+pi, 2*pi) - pi;
end

%% Nonlinear Simulation (Euler integration + additive discrete noise)
function [X, U] = simulate_nonlinear(K,N,dt,noise_cov,x0)
    % The nonlinear state uses theta=pi as upright in this script's
    % convention. We place the initial state around upright:
    X = zeros(4,N);
    U = zeros(1,N-1);
    X(:,1) = [0;0;pi + x0(3);0]; % shift initial angle around upright
    L = chol(noise_cov,'lower');
    for k = 1:N-1
        theta = X(3,k);
        theta_err = wrap_to_pi(theta - pi);
        state_ctrl = [X(1,k); X(2,k); theta_err; X(4,k)];
        u = -K * state_ctrl;
        u = max(min(u,1e6),-1e6);
        dx_det = cartpole_nonlinear(X(:,k), u);
        w = L * randn(4,1);           % discrete additive process noise
        X(:,k+1) = X(:,k) + dt * dx_det + w;
        U(k) = u;
    end
end

%% Helper for saving as PDF + PNG
function save_fig(fig_handle, fname)
    FIG_DIR = 'figures';
    pdfpath = fullfile(FIG_DIR, [fname, '.pdf']);
    pngpath = fullfile(FIG_DIR, [fname, '.png']);
    exportgraphics(fig_handle, pdfpath, 'ContentType','vector');
    exportgraphics(fig_handle, pngpath, 'Resolution',300);
    disp(['Saved: ', pdfpath, ' and ', pngpath]);
end

%% Animator (in-file)
function animate_cartpole(X, U, dt, is_nonlinear, l, varargin)
% ANIMATE_CARTPOLE Animate cart-pole states over time.
% See top of file for usage example.

    p = inputParser;
    addRequired(p,'X',@(x) isnumeric(x) && size(x,1)==4);
    addRequired(p,'U',@(x) isnumeric(x));
    addRequired(p,'dt',@isnumeric);
    addRequired(p,'is_nonlinear',@islogical);
    addRequired(p,'l',@isnumeric);
    addParameter(p,'saveGif',false,@islogical);
    addParameter(p,'gifName','cartpole.gif',@ischar);
    addParameter(p,'speed',1,@(x) isnumeric(x) && x>0);
    addParameter(p,'cartW',0.3,@isnumeric);
    addParameter(p,'cartH',0.2,@isnumeric);
    parse(p,X,U,dt,is_nonlinear,l,varargin{:});
    opts = p.Results;

    N = size(X,2);
    if isempty(U)
        U = zeros(1,max(0,N-1));
    end

    x = X(1,:);
    if opts.is_nonlinear
        theta = wrap_to_pi(X(3,:) - pi);   % upright = pi in nonlinear sim
    else
        theta = wrap_to_pi(X(3,:));
    end

    padx = max(1.0, 0.5*opts.cartW);
    xmin = min(x) - padx; xmax = max(x) + padx;
    ymin = -0.5; ymax = opts.cartH + l + 0.5;

    fig = figure('Name','Cart-Pole Animation','Color','w');
    ax = axes(fig);
    hold(ax,'on');
    axis(ax,[xmin xmax ymin ymax]);
    axis(ax,'equal');
    xlabel(ax,'Cart position (m)');
    ylabel(ax,'Height (m)');
    set(ax,'YTick',[]);

    % ground
    plot(ax,[xmin-10 xmax+10],[0 0],'k-','LineWidth',1);

    % initial drawing objects
    cartW = opts.cartW; cartH = opts.cartH;
    xi = x(1);
    cartPos = [xi - cartW/2, 0, cartW, cartH];
    cartRect = rectangle(ax,'Position',cartPos,'Curvature',0.1,'FaceColor',[0.2 0.6 0.8],'EdgeColor','k');

    wheelR = 0.06;
    wheelLpatch = rectangle(ax,'Position',[xi - cartW/3 - wheelR, -wheelR, 2*wheelR, 2*wheelR],'Curvature',[1 1],'FaceColor',[0 0 0]);
    wheelRpatch = rectangle(ax,'Position',[xi + cartW/3 - wheelR, -wheelR, 2*wheelR, 2*wheelR],'Curvature',[1 1],'FaceColor',[0 0 0]);

    pivot = [xi, cartH];
    pend_x = pivot(1) + l * sin(theta(1));
    pend_y = pivot(2) + l * cos(theta(1));
    poleLine = line(ax,[pivot(1) pend_x],[pivot(2) pend_y],'LineWidth',3,'Color',[0.1 0.1 0.1]);

    timeText = text(ax, xmin + 0.02*(xmax-xmin), ymax - 0.08*(ymax-ymin), sprintf('t = %.2fs',0),'FontSize',10,'HorizontalAlignment','left');
    ctrlText = text(ax, xmin + 0.02*(xmax-xmin), ymax - 0.14*(ymax-ymin), sprintf('u = %.2f N',U(1)),'FontSize',10,'HorizontalAlignment','left');

    drawnow;

    if opts.saveGif
        gifFilename = opts.gifName;
        firstFrame = true;
    end

    for k = 1:N
        if ~isvalid(fig), break; end
        xi = x(k);
        the = theta(k);
        cartPos = [xi - cartW/2, 0, cartW, cartH];
        set(cartRect,'Position',cartPos);
        set(wheelLpatch,'Position',[xi - cartW/3 - wheelR, -wheelR, 2*wheelR, 2*wheelR]);
        set(wheelRpatch,'Position',[xi + cartW/3 - wheelR, -wheelR, 2*wheelR, 2*wheelR]);
        pivot = [xi, cartH];
        pend_x = pivot(1) + l * sin(the);
        pend_y = pivot(2) + l * cos(the);
        set(poleLine,'XData',[pivot(1) pend_x],'YData',[pivot(2) pend_y]);
        tnow = (k-1) * dt;
        set(timeText,'String',sprintf('t = %.2f s', tnow));
        if k <= numel(U)
            set(ctrlText,'String',sprintf('u = %.2f N', U(k)));
        else
            set(ctrlText,'String',sprintf('u = %.2f N', 0));
        end
        drawnow limitrate;

        if opts.saveGif
            frame = getframe(fig);
            im = frame2im(frame);
            [A,map] = rgb2ind(im,256);
            if firstFrame
                imwrite(A,map,gifFilename,'gif','LoopCount',Inf,'DelayTime',dt*opts.speed);
                firstFrame = false;
            else
                imwrite(A,map,gifFilename,'gif','WriteMode','append','DelayTime',dt*opts.speed);
            end
        end

        pause(dt * opts.speed);
    end
end

function stacked_cartpole_animation(X_list, U_list, dt, is_nonlinear_list, l, names)
% X_list: cell array {X_KL, X_WKL, X_KWKL}
% U_list: cell array {U_KL, U_WKL, U_KWKL}
% is_nonlinear_list: logical array [true true true]
% names: cell array {'KL','WKL','KWKL'}

Nsim = numel(X_list);

% figure layout
fig = figure('Name','Stacked Cart-Pole Animations','Color','w');
tiled = tiledlayout(Nsim,1,'TileSpacing','compact');

% Pre-create axes and graphics handles for each simulation
axs = gobjects(Nsim,1);
gfx = struct([]);

for i = 1:Nsim
    axs(i) = nexttile(tiled); hold(axs(i),'on'); axis(axs(i),'equal');
    X = X_list{i}; N = size(X,2);
    x = X(1,:);
    if is_nonlinear_list(i)
        theta = wrap_to_pi(X(3,:) - pi);
    else
        theta = wrap_to_pi(X(3,:));
    end
    pad = 1;
    xmin = min(x)-pad; xmax = max(x)+pad;
    ymin = -0.5; ymax = l+0.5;
    axis(axs(i),[xmin xmax ymin ymax]);
    title(axs(i),names{i});
    xlabel(axs(i),'x (m)'); set(axs(i),'YTick',[]);

    % Draw static ground
    plot(axs(i),[xmin-5 xmax+5],[0 0],'k-','LineWidth',1);

    % Initialize cart + pole graphics for later updating
    cartW = 0.3; cartH = 0.2;
    xi = x(1); the = theta(1);
    cartRect = rectangle(axs(i),'Position',[xi-cartW/2,0,cartW,cartH],...
        'FaceColor',[0.2 0.6 0.8],'EdgeColor','k');
    pivot = [xi cartH];
    pole = line(axs(i),[pivot(1) pivot(1)+l*sin(the)], [pivot(2) pivot(2)+l*cos(the)], ...
        'LineWidth',3,'Color',[0.1 0.1 0.1]);

    gfx(i).cartRect = cartRect;
    gfx(i).pole = pole;
end

% ------- Animation loop -------
kmax = min(cellfun(@(X) size(X,2), X_list));
for k = 1:kmax
    for i = 1:Nsim
        X = X_list{i};
        x = X(1,k);
        if is_nonlinear_list(i)
            theta = wrap_to_pi(X(3,k)-pi);
        else
            theta = wrap_to_pi(X(3,k));
        end
        cartW = 0.3; cartH = 0.2;

        % Update cart
        set(gfx(i).cartRect,'Position',[x-cartW/2,0,cartW,cartH]);

        % Update pole
        pivot = [x cartH];
        pend = [pivot(1)+l*sin(theta), pivot(2)+l*cos(theta)];
        set(gfx(i).pole,'XData',[pivot(1) pend(1)], 'YData',[pivot(2) pend(2)]);
    end
    drawnow limitrate;
    pause(dt);
end
end
