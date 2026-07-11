% Load the hyperspectral image data
load('SalinasA_corrected_83x86x204.mat');
addpath('./HelperFunctions/');
rng(10);
% Set maximum number of frames to process
maxFrames = 30;

% Limit the number of frames if exceeding maxFrames
if (size(T, 3)>maxFrames)
    T=T(:,:,1:maxFrames);
end

% Set observation ratio (percentage of observed pixels)
obsRatio=0.05;

% Set noise-to-signal ratio
NSR=0.05;

% Normalize the tensor data
L = T/max(abs(T(:)));

% Get size dimensions of the normalized tensor
sz=size(L);

% Calculate noise standard deviation based on tensor norm
sigma=NSR*h_tnorm(L)/sqrt(prod(sz));

% Generate Gaussian noise
G=randn(sz)*sigma;

% Create binary mask for observed entries
B = rand(sz)<obsRatio;

% Get indices of observed entries
vIdx=find(B>0);

% Apply mask to noise
G = G.*B;

% Create observed tensor with noise
Y=(L+G).*B;

% Extract observed values
y=Y(vIdx);

% Set Lp and Sq norm parameters
p = 0.80; q = 0.81;

% Set observation parameters
obs.tsize=sz;
obs.y=y;
obs.idx=vIdx;

% Configure optimization options
opts.obs=obs;
opts.para.lambda=0.11;     % Regularization parameter
opts.para.rho=1e-5;        % Initial penalty parameter
opts.para.nu=1.1;          % Parameter update rate
opts.MAX_ITER_OUT=250;     % Maximum outer iterations
opts.p=p;                  % Lp norm parameter
opts.q=q;                  % Sq norm parameter
opts.weightGap=1;          % Refresh weights at every iteration
opts.recordGap=1;          % Record progress at every iteration
opts.MIN_RHO=1e-5;         % Minimum penalty parameter
opts.MAX_RHO=1e5;          % Maximum penalty parameter
opts.MAX_EPS=2e-5;         % Maximum error tolerance
opts.verbose=1;            % Enable verbose output

% Initialize memoization structure for Lp-Sq optimization
memoLpSq=h_construct_memo(opts);
memoLpSq.printerInterval = 20;  % Set print interval for progress
memoLpSq.truth=L;               % Store ground truth for evaluation

% Run Lp-Sq optimization using ADMM with DCT
tic;
memoLpSq=f_ntc_LpSq_ADMM_dct(obs,opts,memoLpSq);
runtime=toc;

psnrVal = h_Psnr(L,memoLpSq.T_hat);
relErr = norm(double(memoLpSq.T_hat(:)-L(:)))/(norm(double(L(:)))+eps);
fprintf('Final: PSNR = %.4f dB, relative error = %.4e, time = %.2f seconds\n', ...
    psnrVal,relErr,runtime);
