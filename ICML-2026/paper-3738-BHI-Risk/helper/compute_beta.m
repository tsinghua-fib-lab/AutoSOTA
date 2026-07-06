function Beta = compute_beta(H, gamma, mu)
[p, E] = size(gamma);
V = size(H,1);

Beta = zeros(p, V);

for j = 1:p
    for e = 1:E
        if gamma(j,e)
            Beta(j,:) = Beta(j,:) + mu(j,e) * H(:,e)';
        end
    end
end
end
