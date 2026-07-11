% Octave-compatible experiment script for Salinas A tensor completion
% Reproduces: PSNR and SSIM at SR=10%, Gaussian noise c=0.05, DCT transform
% Paper settings: p=0.80, q=0.81 (S3), 10 independent trials, first 30 bands
% Matching the released MATLAB code configuration

pkg load signal;

% Load the hyperspectral image data
load('SalinasA_corrected_83x86x204.mat');
addpath('./HelperFunctions/');

% Set maximum number of frames to process (matching original code)
maxFrames = 30;

% Limit the number of frames if exceeding maxFrames
if (size(T, 3) > maxFrames)
    T = T(:, :, 1:maxFrames);
end

fprintf('Tensor size: %dx%dx%d\n', size(T,1), size(T,2), size(T,3));

% Experiment settings (rubric / Table 1)
obsRatio = 0.10;    % 10% sampling rate
NSR = 0.05;         % Noise-to-signal ratio (c=0.05)
n_trials = 10;      % 10 independent trials

% Lp and Sq norm parameters (S3 setting from paper)
p = 0.80;
q = 0.81;

% Pre-allocate results
psnr_vals = zeros(n_trials, 1);
ssim_vals = zeros(n_trials, 1);
relErr_vals = zeros(n_trials, 1);
runtime_vals = zeros(n_trials, 1);

fprintf('=== SalinasA Noisy Tensor Completion ===\n');
fprintf('SR=%.0f%%, NSR=%.2f, p=%.2f, q=%.2f, bands=%d, trials=%d\n', ...
    obsRatio*100, NSR, p, q, size(T,3), n_trials);

% Normalize the tensor data
L = T / max(abs(T(:)));
sz = size(L);
sigma = NSR * norm(double(L(:))) / sqrt(prod(sz));

fprintf('Normalized tensor range: [%.4f, %.4f], sigma=%.6f\n', min(L(:)), max(L(:)), sigma);

for trial = 1:n_trials
    % Set random seed for reproducibility (Octave-compatible)
    rand('seed', trial);
    randn('seed', trial);

    % Generate Gaussian noise
    G = randn(sz) * sigma;

    % Create binary mask for observed entries
    B = rand(sz) < obsRatio;

    % Get indices of observed entries
    vIdx = find(B > 0);
    actual_ratio = length(vIdx) / prod(sz);
    if trial == 1
        fprintf('Actual sampling ratio: %.4f\n', actual_ratio);
    end

    % Apply mask to noise
    G = G .* B;

    % Create observed tensor with noise
    Y = (L + G) .* B;

    % Extract observed values
    y = Y(vIdx);

    % Set observation parameters
    obs.tsize = sz;
    obs.y = y;
    obs.idx = vIdx;

    % Configure optimization options
    opts.obs = obs;
    opts.para.lambda = 0.11;
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
    opts.verbose = 0;   % Quiet mode for multiple trials

    % Initialize memoization structure
    memoLpSq = h_construct_memo(opts);
    memoLpSq.printerInterval = 20;
    memoLpSq.truth = L;

    % Run Lp-Sq optimization using ADMM with DCT
    tic;
    memoLpSq = f_ntc_LpSq_ADMM_dct(obs, opts, memoLpSq);
    runtime_vals(trial) = toc;

    % Compute PSNR
    psnr_vals(trial) = h_Psnr(L, memoLpSq.T_hat);

    % Compute SSIM using custom 3D implementation
    ssim_vals(trial) = compute_ssim_3d(L, memoLpSq.T_hat);

    % Compute relative error
    relErr_vals(trial) = norm(double(memoLpSq.T_hat(:) - L(:))) / (norm(double(L(:))) + eps);

    fprintf('Trial %2d: PSNR = %.4f dB, SSIM = %.4f, relErr = %.4e, time = %.2f s\n', ...
        trial, psnr_vals(trial), ssim_vals(trial), relErr_vals(trial), runtime_vals(trial));
end

% Summary statistics
fprintf('\n=== Summary over %d trials ===\n', n_trials);
fprintf('PSNR: mean=%.4f, std=%.4f, min=%.4f, max=%.4f\n', ...
    mean(psnr_vals), std(psnr_vals), min(psnr_vals), max(psnr_vals));
fprintf('SSIM: mean=%.4f, std=%.4f, min=%.4f, max=%.4f\n', ...
    mean(ssim_vals), std(ssim_vals), min(ssim_vals), max(ssim_vals));
fprintf('Runtime: mean=%.2f s, total=%.2f s\n', mean(runtime_vals), sum(runtime_vals));

% Save results
save('-mat7-binary', '/repo/results_salinasA.mat', 'psnr_vals', 'ssim_vals', 'relErr_vals', 'runtime_vals', ...
    'obsRatio', 'NSR', 'p', 'q', 'maxFrames', 'n_trials');
fprintf('Results saved to /repo/results_salinasA.mat\n');

% Display final PSNR and SSIM in a format easy to parse
fprintf('\nFINAL_RESULT: PSNR=%.4f SSIM=%.4f\n', mean(psnr_vals), mean(ssim_vals));
