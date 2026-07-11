function ssim_val = compute_ssim_3d(img1, img2)
% Compute SSIM for 3D tensor (average over all bands/slices)
% img1, img2: 3D tensors of same size
% Returns: mean SSIM across all 3rd-dimension slices
%
% Follows the standard SSIM formula from Wang et al. 2004
% with MATLAB default parameters: K=[0.01, 0.03], dynamic range = 1.0

sz = size(img1);
n_bands = sz(3);

% Convert to double
img1 = double(img1);
img2 = double(img2);

% SSIM parameters (matching MATLAB defaults for double data)
K1 = 0.01;
K2 = 0.03;
L_val = 1.0;  % MATLAB default for double class

C1 = (K1 * L_val)^2;
C2 = (K2 * L_val)^2;

% Create Gaussian window (11x11, sigma=1.5) using fspecial
pkg load image;
window = fspecial('gaussian', 11, 1.5);
window = window / sum(window(:));

ssim_slices = zeros(n_bands, 1);

for k = 1:n_bands
    slice1 = img1(:, :, k);
    slice2 = img2(:, :, k);

    % Compute local statistics using filter2 with 'valid'
    mu1 = filter2(window, slice1, 'valid');
    mu2 = filter2(window, slice2, 'valid');

    mu1_sq = mu1 .* mu1;
    mu2_sq = mu2 .* mu2;
    mu1_mu2 = mu1 .* mu2;

    sigma1_sq = filter2(window, slice1 .* slice1, 'valid') - mu1_sq;
    sigma2_sq = filter2(window, slice2 .* slice2, 'valid') - mu2_sq;
    sigma12  = filter2(window, slice1 .* slice2, 'valid') - mu1_mu2;

    % Clamp negative variances to zero (numerical stability)
    sigma1_sq(sigma1_sq < 0) = 0;
    sigma2_sq(sigma2_sq < 0) = 0;

    % SSIM map
    numerator = (2 * mu1_mu2 + C1) .* (2 * sigma12 + C2);
    denominator = (mu1_sq + mu2_sq + C1) .* (sigma1_sq + sigma2_sq + C2);

    ssim_map = numerator ./ denominator;
    ssim_slices(k) = mean(ssim_map(:));
end

ssim_val = mean(ssim_slices);
end
