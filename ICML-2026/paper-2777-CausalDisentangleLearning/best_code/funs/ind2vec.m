function vec = ind2vec(ind)
    % ind2vec replacement for Octave
    % Convert indices to vectors
    % ind: 1×n row vector of class indices (positive integers)
    % vec: max(ind)×n matrix with 1 at position (ind(i), i)
    n = length(ind);
    u = unique(ind);
    m = max(u);
    vec = zeros(m, n);
    for i = 1:n
        if ind(i) > 0
            vec(ind(i), i) = 1;
        end
    end
end
