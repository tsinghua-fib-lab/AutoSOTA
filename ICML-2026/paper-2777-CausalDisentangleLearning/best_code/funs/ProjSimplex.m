function Z = ProjSimplex(Z)
    % Z: m * n matrix, project each column to simplex
    [m, n] = size(Z);
    Z_sorted = sort(Z, 1, 'descend');
    cumsum_Z = cumsum(Z_sorted, 1);
    rho = zeros(1, n);
    for j = 1:n
        idx = find(Z_sorted(:,j) + (1 - cumsum_Z(:,j)) ./ (1:m)' > 0, 1, 'last');
        if isempty(idx)
            rho(j) = 1;
        else
            rho(j) = idx;
        end
    end
    
    % Calculate theta
    idx_linear = sub2ind([m, n], rho, 1:n);
    theta = (1 - cumsum_Z(idx_linear)) ./ rho;
    
    % Project
    Z = max(Z + repmat(theta, m, 1), 0);
end