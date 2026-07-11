% Evaluation script for SalinasA noisy tensor completion
% Reproduces paper results for SR=10%, Gaussian noise c=0.05, DCT transform
% S3 setting: p=0.80, q=0.81, first 30 bands, 10 independent trials

pkg load signal;
pkg load image;

% Load data
load('SalinasA_corrected_83x86x204.mat');
addpath('./HelperFunctions/');

% Use first 30 bands (matching paper)
maxFrames = 30;
if (size(T, 3) > maxFrames)
    T = T(:, :, 1:maxFrames);
end

% Experiment settings (Table 1, Section 6)
obsRatio = 0.10;    % 10% sampling rate
NSR = 0.05;         % Noise coefficient c=0.05
n_trials = 10;

% S3 parameters
p = 0.85;
q = 0.86;

% Normalize
L = T / max(abs(T(:)));
sz = size(L);
sigma = NSR * norm(double(L(:))) / sqrt(prod(sz));

psnr_vals = zeros(n_trials, 1);
ssim_vals = zeros(n_trials, 1);
relErr_vals = zeros(n_trials, 1);
runtime_vals = zeros(n_trials, 1);

fprintf('=== SalinasA Noisy Tensor Completion ===\n');
fprintf('SR=%.0f%%, NSR=%.2f, p=%.2f, q=%.2f, bands=%d, trials=%d\n', ...
    obsRatio*100, NSR, p, q, size(T,3), n_trials);

for trial = 1:n_trials
    rand('seed', trial);
    randn('seed', trial);

    G = randn(sz) * sigma;
    B = rand(sz) < obsRatio;
    vIdx = find(B > 0);
    G = G .* B;
    Y = (L + G) .* B;
    y = Y(vIdx);

    obs.tsize = sz;
    obs.y = y;
    obs.idx = vIdx;

    opts.obs = obs;
    opts.para.lambda = 0.15;
    opts.para.rho = 1e-5;
    opts.para.nu = 1.1;
    opts.MAX_ITER_OUT = 250;
    opts.p = p;
    opts.q = q;
    opts.weightGap = 1;
    opts.recordGap = 1;
    opts.MIN_RHO = 1e-5;
    opts.MAX_RHO = 1e5;
    opts.MAX_EPS = 2e-5;
    opts.psnr_patience = 25;
    opts.psnr_tol = 0.01;
    opts.verbose = 0;

    memoLpSq = h_construct_memo(opts);
    memoLpSq.printerInterval = 20;
    memoLpSq.truth = L;

    tic;
    memoLpSq = f_ntc_LpSq_ADMM_dct(obs, opts, memoLpSq);
    runtime_vals(trial) = toc;

    psnr_vals(trial) = h_Psnr(L, memoLpSq.T_hat);
    ssim_vals(trial) = compute_ssim_3d(L, memoLpSq.T_hat);
    relErr_vals(trial) = norm(double(memoLpSq.T_hat(:) - L(:))) / (norm(double(L(:))) + eps);

    fprintf('Trial %2d: PSNR = %.4f dB, SSIM = %.4f, time = %.2f s\n', ...
        trial, psnr_vals(trial), ssim_vals(trial), runtime_vals(trial));
end

fprintf('\n=== Final Results ===\n');
fprintf('PSNR: mean=%.4f, std=%.4f\n', mean(psnr_vals), std(psnr_vals));
fprintf('SSIM: mean=%.4f, std=%.4f\n', mean(ssim_vals), std(ssim_vals));
fprintf('Runtime: mean=%.2f s, total=%.2f s\n', mean(runtime_vals), sum(runtime_vals));

% Output final metric for parsing
fprintf('\nEVAL_RESULT PSNR=%.4f SSIM=%.4f\n', mean(psnr_vals), mean(ssim_vals));

% Save results
save('-mat7-binary', '/repo/eval_results.mat', 'psnr_vals', 'ssim_vals', 'relErr_vals', 'runtime_vals', ...
    'obsRatio', 'NSR', 'p', 'q', 'maxFrames', 'n_trials');
