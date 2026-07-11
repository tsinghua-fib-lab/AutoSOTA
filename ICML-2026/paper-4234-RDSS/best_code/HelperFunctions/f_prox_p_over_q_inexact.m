function X = f_prox_p_over_q_inexact(M, lambda, oldX, p, q)
% Inexact proximal operator for p/q structured sparsity
% Input:
%   M: Input matrix, each column represents a group
%   lambda: Regularization parameter
%   oldX: Previous iteration result for weight calculation
%   p: Parameter for outer norm
%   q: Parameter for inner norm
% Output:
%   X: Matrix after group sparse proximal operation

% Select appropriate proximal operator based on q value
q_closedform = 1;
switch q
    case 1
        f_prox=@f_prox_l1;         % L1 norm proximal
    case 1/2
        f_prox=@f_prox_l1_2;       % L1/2 norm proximal
    case 2/3
        f_prox=@f_prox_l2_3;       % L2/3 norm proximal
    case 0
        f_prox=@f_prox_l0;         % L0 norm proximal
    otherwise
        q_closedform = 0;          % No closed-form solution available
end

% Initialize output matrix
X = zeros(size(M));

% Process each column (group) separately
for i = 1:size(M, 2)
    % Calculate Lq norm of each column from previous iteration as historical weight
    % Add small epsilon (1e-14) to avoid division by zero
    history = sum(abs(oldX(:,i)).^q+1e-14)^(p/(q+1e-14) - 1);
    
    % Apply proximal operator based on whether closed-form solution exists
    if q_closedform == 1
        if history > 0
            X(:,i) = f_prox(M(:,i), lambda*history);
        else
            X(:,i) = f_prox(M(:,i), lambda);
        end
    else
        % Use L1/2 approximation for non-standard q values
        if history > 0
            X(:,i) = f_prox_q_by_l1_2(M(:,i), oldX(:,i), q, lambda*history);
        else
            X(:,i) = f_prox_q_by_l1_2(M(:,i), oldX(:,i), q, lambda);
        end
    end
end
end