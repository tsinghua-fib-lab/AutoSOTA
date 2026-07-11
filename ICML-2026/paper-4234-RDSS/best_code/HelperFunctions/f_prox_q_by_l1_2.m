function x = f_prox_q_by_l1_2(a, z, q, lambda)
% Proximal operator for Lq norm using L1/2 norm transformation
% This function implements the proximal operator for the Lq norm by transforming
% it to the L1/2 norm case using a closed-form solution
%
% Input parameters:
%   a: Input vector to be processed 
%   z: Reference vector for weight calculation
%   q: Parameter for Lq norm
%   lambda: Regularization parameter
%
% Output:
%   x: Processed vector after applying proximal operator

% Initialize output vector with same dimensions as input
x = zeros(size(a));

% Calculate weighted lambda values using reference vector z
% Add small epsilon to avoid division by zero
v_lambda = 2*q*(abs(z).^(1/2)+eps).^(2*q-1)*lambda;

% Process each element of input vector
for k = 1:length(a)
    % Calculate threshold parameter h(lambda)
    h_lambda = 54^(1/3)/4 * v_lambda(k)^(2/3);
    
    % Calculate phase angle phi(a) for current element
    phi_a = acos((lambda/8) * (abs(a(k))/3)^(-3/2));
    
    % Apply three-part thresholding rule:
    % 1. If value exceeds positive threshold
    % 2. If value is below negative threshold
    % 3. If value is within thresholds (set to zero)
    if a(k) > h_lambda
        x(k) = (2/3) * abs(a(k)) * (1 + cos((2*pi/3) - (2/3)*phi_a));
    elseif a(k) < -h_lambda
        x(k) = -(2/3) * abs(a(k)) * (1 + cos((2*pi/3) - (2/3)*phi_a));
    else
        x(k) = 0;
    end
end

% Ensure output is real-valued
x = real(x);
end