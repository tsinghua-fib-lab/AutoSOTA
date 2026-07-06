function alpha = calibrate_intercept(eta, target_prev)
% eta: n x 1 vector (X * beta for disease v)
% target_prev: scalar in (0,1)

obj = @(a) mean(1 ./ (1 + exp(-(a + eta)))) - target_prev;

% good initialization
a0 = log(target_prev / (1 - target_prev)) - mean(eta);

alpha = fzero(obj, a0);
end
