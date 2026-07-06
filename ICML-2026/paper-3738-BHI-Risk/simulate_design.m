
%% simulation settings
addpath("helper");

home = pwd;
seed = 42;
N = 2000; % test different sample size: 2k, 5k, 10k
mu_mean0 = 1.5; % test different level of signals: 2.5 or 1
mu_sd = 0.5;

% E_hat_set = [10, 15, 20];
E_hat_set = 10;
n_E_hat = length(E_hat_set);

V = 30;
E = 5;
P = 6;
nRarePerEdge   = 1;
nCommonPerEdge = 6;

%% model settings
initial_method = 'NNMF';
max_iter = 2000;
tol = 1e-4;
omega_repulsion_set = [0, 0.5, 1, 2, 5];
n_omega_repulsion = length(omega_repulsion_set);

verbose = false;
weights = 1;
sigma2_alpha = 10;
fix_z = false; z_constraint = 1; batch_size = 0;
t0 = 10; staged = 0; warmup_iters = 0;


%%%%%% filepath to save the results
output_dir = home + "/simu/";
if ~isfolder(output_dir)
    mkdir(output_dir);
end

resname = sprintf("BHPI_N=%d_mu=%.1f_sd=%.1f_seed=%d.mat", N, mu_mean0, mu_sd, ...
    seed);
resname = output_dir + resname;
disp(resname);

%% data generation
[X, Y, alpha, Beta, H, gamma, mu] = simu_data_gen(N, P, V, E, mu_mean0, ...
    mu_sd, seed, nRarePerEdge, nCommonPerEdge);
prev = mean(Y, 1);
rare_idx = prev < 0.05;
common_idx = prev >= 0.05;


% Cross validation
cv = cvpartition(N, "HoldOut", 0.2);
all_idx = 1:N;
idxTrainVal = training(cv);
idxTest = test(cv);

X_temp = X(idxTrainVal, :);
Y_temp = Y(idxTrainVal, :);
X_test = X(idxTest, :);
Y_test = Y(idxTest, :);
R_test = ~isnan(Y_test);

% Second split: 25% of the temp data goes to Validation
cv_val = cvpartition(size(X_temp, 1), 'HoldOut', 0.25);

X_train = X_temp(training(cv_val), :);
Y_train = Y_temp(training(cv_val), :);
% R_train = ~isnan(Y_train);

X_val = X_temp(test(cv_val), :);
Y_val = Y_temp(test(cv_val), :);
R_val = ~isnan(Y_val);

clear cv_val X_temp Y_temp cv idxTrainVal idxTest


%% Model fitting
y_test_score = NaN(size(X_test, 1), V, n_omega_repulsion, n_E_hat);
auroc_per_disease = NaN(V, n_omega_repulsion, n_E_hat);
beta_est = NaN(2, n_omega_repulsion, n_E_hat);
match_idx_all = NaN(E, n_omega_repulsion, n_E_hat);
H_auroc_all = NaN(1 + E, n_omega_repulsion, n_E_hat);
H_prob_overlap = NaN(n_omega_repulsion, n_E_hat);
gamma_entropy_per_predictor = NaN(P, n_omega_repulsion, n_E_hat);
gamma_auroc_all = NaN(1 + P, n_omega_repulsion, n_E_hat);
repulsion_redundancy_metrics = NaN(P, 2, n_omega_repulsion, n_E_hat);
z_prob_all_cell = cell(1, n_E_hat);
H_prob_all_cell = cell(1, n_E_hat);
gamma_prob_all_cell = cell(1, n_E_hat);
mu_hat_all_cell = cell(1, n_E_hat);
H_prob_entropy_per_edge_cell = cell(1, n_E_hat);

tic;
for idx_E_hat = 1:n_E_hat
    E_hat = E_hat_set(idx_E_hat);
    fprintf('Estimate using E_hat = %d \n', E_hat);

    %% check performance under different initializations
    disp('Choose best initialization seed');
    n_seed_init = 3;
    omega_repulsion = 0;
    AUROC_val_mean = NaN(1, n_seed_init);
    % seed_init = 1;
    for seed_init = 1:n_seed_init
        [initials] = cavi_initialization(seed_init, ...
            initial_method, E_hat, X_train, Y_train, []);

        model = BHPI(X_train, Y_train, E_hat, max_iter, ...
            seed_init, initials, omega_repulsion, staged, fix_z, z_constraint, sigma2_alpha, ...
            warmup_iters, batch_size, t0, weights, tol, verbose);

        % evaluate on validation set
        eta_val = X_val * model.beta + model.alpha_mean;
        y_fitted_prob_val = 1 ./ (1 + exp(-eta_val));
        % AUROC on test set
        AUROC_val = NaN(1, V);
        for v = 1:V
            [~,~,~,AUROC_val(v)] = perfcurve(Y_val(:, v), y_fitted_prob_val(:, v), 1);
        end
        AUROC_val_mean(seed_init) = mean(AUROC_val);
    end

    [AUROC_val_best, seed_init_best] = max(AUROC_val_mean);
    fprintf("Best initial seed=%d, validation marco AUROC=%.4f \n", seed_init_best, AUROC_val_best);


    %% finalize initialization & check different repulsion strengths
    [initials] = cavi_initialization(seed_init_best, ...
        initial_method, E_hat, X_train, Y_train, []);

    z_prob_all = NaN(E_hat, n_omega_repulsion);
    H_prob_all = NaN(V, E_hat, n_omega_repulsion);
    gamma_prob_all = NaN(P, E_hat, n_omega_repulsion);
    mu_hat_all = NaN(P, E_hat, n_omega_repulsion);
    H_prob_entropy_per_edge = NaN(E_hat, n_omega_repulsion);

    % for idx_repulsion = 1
    for idx_repulsion = 1:n_omega_repulsion
        omega_repulsion = omega_repulsion_set(idx_repulsion);
        fprintf("repulsion weight = %.1f\n", omega_repulsion);

        % model fitting
        model = BHPI(X_train, Y_train, E_hat, max_iter, ...
            seed_init_best, initials, omega_repulsion, staged, fix_z, z_constraint, sigma2_alpha, ...
            warmup_iters, batch_size, t0, weights, tol, verbose);

        m_prob = model.m_prob;
        z_prob = model.z_prob;
        z_prob_all(:, idx_repulsion) = z_prob;
        gamma_prob = model.gamma_prob;
        gamma_prob_joint = gamma_prob .* z_prob';
        gamma_prob_all(:, :, idx_repulsion) = gamma_prob_joint;
        H_prob = m_prob .* z_prob';
        H_prob_all(:, :, idx_repulsion) = H_prob;

        beta_hat = model.beta;
        mu_mean = model.mu_mean;
        mu_hat_all(:, :, idx_repulsion) = mu_mean;
        mu_var = model.mu_var;
        alpha_mean = model.alpha_mean;
        alpha_var = model.alpha_var;

        %% estimation & predictive results
        % mechanism recovery: disease-level coefficient
        mse_beta = mean((Beta - beta_hat).^2, "all");
        beta_corr = corr(Beta(:), beta_hat(:));
        beta_est(:, idx_repulsion, idx_E_hat) = [mse_beta, beta_corr];

        % prediction on test set
        eta_test = X_test * beta_hat;
        alpha_hat = model.alpha_mean;
        eta_test = eta_test + alpha_hat;
        y_fitted_prob_test = 1 ./ (1 + exp(-eta_test));
        y_test_score(:, :, idx_repulsion, idx_E_hat) = y_fitted_prob_test;

        % AUROC on test set
        AUROC_test = NaN(1, V);
        for v = 1:V
            [~,~,~,AUROC_test(v)] = perfcurve(Y_test(:, v), y_fitted_prob_test(:, v),1);
        end
        auroc_per_disease(:, idx_repulsion, idx_E_hat) = AUROC_test;

        %% latent hypergraph structure
        %%%%%%%%%% soft alignment using PIP of H
        H_prob_entropy = - (H_prob .* log(H_prob) + ...
            (1 - H_prob) .* log(1 - H_prob)); % V x E
        H_prob_entropy_per_edge(:, idx_repulsion) = sum(H_prob_entropy, 1); % 1 x E

        % (1) define similarity between true & estimated hyperedges
        H_sim = NaN(E, E_hat);
        for e1 = 1:E
            for e2 = 1:E_hat
                H_sim(e1, e2) = H(:,e1)' * H_prob(:, e2);
            end
        end

        % (2) Hungarian algorithm (soft matching)
        pairs = matchpairs(-H_sim, 0);
        match_idx = zeros(1, E);
        H_prob_aligned = zeros(V, E);
        for k = 1:size(pairs,1)
            i_true  = pairs(k,1);
            j_est = pairs(k,2);
            match_idx(i_true) = j_est;
            H_prob_aligned(:, i_true) = H_prob(:, j_est);
        end
        match_idx_all(:, idx_repulsion, idx_E_hat) = match_idx;

        %%%%%%%%%% AUROC of hyperedge inclusion using PIP
        [~, ~, ~, H_auroc_pool] = perfcurve(H(:), H_prob_aligned(:), 1);
        fprintf("Pooled AUROC of H = %.4f\n", H_auroc_pool);
        H_auroc_all(1, idx_repulsion, idx_E_hat) = H_auroc_pool;

        H_auroc = NaN(1, E);
        for e = 1:E
            [~,~,~,H_auroc(e)] = perfcurve(H(:,e), H_prob_aligned(:,e), 1);
        end
        H_auroc_all(2:end, idx_repulsion, idx_E_hat) = H_auroc;

        Ohat = E_Overlap(H_prob);
        Ohat = Ohat(triu(true(E_hat), 1));
        mean_overlap = mean(Ohat);
        H_prob_overlap(idx_repulsion, idx_E_hat) = mean_overlap;


        %% Active predictor recovery
        % entropy of gamma, comparison of with/without repulsion, lower is better
        gamma_entropy = - (gamma_prob_joint .* log(gamma_prob_joint) + ...
            (1 - gamma_prob_joint) .* log(1 - gamma_prob_joint)); % P x E
        gamma_entropy_per_predictor(:, idx_repulsion, idx_E_hat) = sum(gamma_entropy, 2); % P x 1

        % AUROC
        gamma_prob_aligned = gamma_prob_joint(:, match_idx);
        [~, ~, ~, gamma_auroc_pool] = perfcurve(gamma(:), gamma_prob_aligned(:), 1);
        gamma_auroc_all(1, idx_repulsion, idx_E_hat) = gamma_auroc_pool;
        fprintf("Pooled AUROC of gamma = %.4f\n", gamma_auroc_pool);

        gamma_auroc = NaN(P, 1);
        for p = 1:P
            [~, ~, ~, gamma_auroc(p)] = perfcurve(gamma(p,:), gamma_prob_aligned(p,:), 1);
        end
        gamma_auroc_all(2:end, idx_repulsion, idx_E_hat) = gamma_auroc;

        % redundancy metrics per predictor
        [effect_hyperedge_per_predictor, ~, average_hyperedge_overlap, ~] = ...
            repulsion_strength(gamma_prob, m_prob, z_prob); % P x 1
        repulsion_redundancy_metrics(:,:, idx_repulsion, idx_E_hat) = [...
            effect_hyperedge_per_predictor, average_hyperedge_overlap];


    end

    z_prob_all_cell{idx_E_hat} = z_prob_all;
    H_prob_all_cell{idx_E_hat} = H_prob_all;
    gamma_prob_all_cell{idx_E_hat} = gamma_prob_all;
    mu_hat_all_cell{idx_E_hat} = mu_hat_all;
    H_prob_entropy_per_edge_cell{idx_E_hat} = H_prob_entropy_per_edge;

    toc;
end

%% ind logistic regression
p_test_logistic = NaN(size(X_test, 1),V);
beta_logistic = NaN(P, V);
for v=1:V
    % idx = Rtr(:,v)==1;
    % if sum(idx)<20, continue; end
    mdl = fitglm(X_train, Y_train(:,v), ...
        'Distribution','binomial','Link','logit');
    p_test_logistic(:,v) = predict(mdl,X_test);
    beta_logistic(:, v) = mdl.Coefficients.Estimate(2:end);
end

mse_beta_logistic = mean((Beta - beta_logistic).^2, "all");
beta_corr_logistic = corr(Beta(:), beta_logistic(:));
beta_est_logistic = [mse_beta_logistic, beta_corr_logistic];

disp('est(beta) Comparison (mse & corr)')
disp([beta_est(:,:,1), beta_est_logistic'])

% AUROC on test set
AUROC_test_logistic = NaN(1, V);
for v = 1:V
    [~,~,~,AUROC_test_logistic(v)] = perfcurve(Y_test(:, v), ...
        p_test_logistic(:, v),1);
end

auroc_test = [auroc_per_disease(:,:,1), AUROC_test_logistic'];
disp('AUROC for rare diseases')
disp(auroc_test(1:8, :))
disp('Marco AUROC for rare diseases')
disp(mean(auroc_test(1:8, :), 1))

disp("#diseases that CAVI is better than logistic")
disp(sum(auroc_test(:,1:n_omega_repulsion) > auroc_test(:,end), 1))
disp('Marco AUROC Comparison')
disp(mean(auroc_test, 1));
