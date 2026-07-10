% CDAL Evaluation Script for Yale Dataset
% Reproduces paper metrics from Causal Disentangled Anchor Learning (ICML 2026)
%
% Paper targets (Table 2, Yale): ACC=60.36 NMI=64.07 Pur=63.30 Fsc=30.90 Bal=12.50 MNCE=66.50
% Rubric CI: ACC [55.76, 60.82], NMI [64.01, 64.08], Pur [58.79, 63.75], Fsc [29.66, 43.31]

pkg load statistics;
pkg load optim;
addpath(genpath('./'));

% Load Yale dataset
load('./datasets/yaleA_3view.mat');
Y = double(y);
g = double(g);
k = length(unique(Y));
n = length(Y);

fprintf('Dataset: Yale | Samples: %d | Views: %d | Classes: %d | Groups: %d\n', n, length(X), k, length(unique(g)));

% Hyperparameters (grid search from paper ranges)
% Paper ranges: alpha,beta,gamma in {1e-5, 1e-3, 1e-1, 10, 1000}, anchor in {k,2k,3k,4k,5k}
alpha = 1000;
beta = 1000;
gamma = 1000;
anchor = 2 * k;  % 30

fprintf('Parameters: anchor=%d alpha=%g beta=%g gamma=%g\n', anchor, alpha, beta, gamma);

% Run CDAL
tic;
[Z, iter, obj] = CDAL(X, g', anchor, alpha, beta, gamma);
cdal_time = toc;
fprintf('CDAL converged in %d iterations (%.1fs)\n', iter, cdal_time);

% Spectral clustering via SVD of Zu
[UU, S, V] = mySVD(Z', k);
F = UU ./ repmat(sqrt(sum(UU .^ 2, 2)), 1, size(UU, 2));

% K-means clustering (20 replicates as in paper)
MAXiter = 1000;
REPlic = 20;
n_reps = 20;

res = zeros(n_reps, 8);
balance_list = zeros(n_reps, 1);
mnce_list = zeros(n_reps, 1);

for rep = 1:n_reps
    pY = kmeans(F, k, 'maxiter', MAXiter, 'replicates', REPlic, 'emptyaction', 'singleton');
    res(rep, :) = Clustering8Measure(Y, pY);
    [bal, mnce] = eval_fair(pY, g);
    balance_list(rep) = bal;
    mnce_list(rep) = mnce;
end

% Average results
ACC = mean(res(:, 1)) * 100;
NMI = mean(res(:, 2)) * 100;
Pur = mean(res(:, 3)) * 100;
Fsc = mean(res(:, 4)) * 100;
Bal = mean(balance_list) * 100;
MNCE = mean(mnce_list) * 100;

fprintf('\n========== RESULTS ==========\n');
fprintf('ACC  = %.2f  (paper: 60.36, CI: [55.76, 60.82])\n', ACC);
fprintf('NMI  = %.2f  (paper: 64.07, CI: [64.01, 64.08])\n', NMI);
fprintf('Pur  = %.2f  (paper: 63.30, CI: [58.79, 63.75])\n', Pur);
fprintf('Fsc  = %.2f  (paper: 30.90, CI: [29.66, 43.31])\n', Fsc);
fprintf('Bal  = %.2f  (paper: 12.50, CI: [12.46, 12.90])\n', Bal);
fprintf('MNCE = %.2f  (paper: 66.50, CI: [66.38, 67.74])\n', MNCE);
fprintf('=============================\n');

% Check which metrics are within CI
fprintf('\nCI Check:\n');
if ACC >= 55.76 && ACC <= 60.82
    fprintf('  ACC:  %.2f in [55.76, 60.82] -> WITHIN CI\n', ACC);
else
    fprintf('  ACC:  %.2f in [55.76, 60.82] -> OUTSIDE CI\n', ACC);
end
if NMI >= 64.01 && NMI <= 64.08
    fprintf('  NMI:  %.2f in [64.01, 64.08] -> WITHIN CI\n', NMI);
else
    fprintf('  NMI:  %.2f in [64.01, 64.08] -> OUTSIDE CI\n', NMI);
end
if Pur >= 58.79 && Pur <= 63.75
    fprintf('  Pur:  %.2f in [58.79, 63.75] -> WITHIN CI\n', Pur);
else
    fprintf('  Pur:  %.2f in [58.79, 63.75] -> OUTSIDE CI\n', Pur);
end
if Fsc >= 29.66 && Fsc <= 43.31
    fprintf('  Fsc:  %.2f in [29.66, 43.31] -> WITHIN CI\n', Fsc);
else
    fprintf('  Fsc:  %.2f in [29.66, 43.31] -> OUTSIDE CI\n', Fsc);
end
if Bal >= 12.46 && Bal <= 12.90
    fprintf('  Bal:  %.2f in [12.46, 12.90] -> WITHIN CI\n', Bal);
else
    fprintf('  Bal:  %.2f in [12.46, 12.90] -> OUTSIDE CI\n', Bal);
end
if MNCE >= 66.38 && MNCE <= 67.74
    fprintf('  MNCE: %.2f in [66.38, 67.74] -> WITHIN CI\n', MNCE);
else
    fprintf('  MNCE: %.2f in [66.38, 67.74] -> OUTSIDE CI\n', MNCE);
end
