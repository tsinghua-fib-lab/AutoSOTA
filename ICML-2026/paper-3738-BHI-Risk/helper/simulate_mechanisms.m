function [gamma, mu] = simulate_mechanisms(P, H, mu_mean, mu_sd)

E = size(H, 2);

gamma = zeros(P, E);
% exactly one hyperedge per predictor
for j = 1:P
    e = randi(E);
    gamma(j,e) = 1;
    % mu(j,e) = mu_mean + mu_sd * randn;
end

% check those empty hyperedge effects
for e = 1:E
    while sum(gamma(:, e)) < 1
        % j = randi(P);
        for j = 1:P
            if sum(gamma(j, :)) >= 1
                idx = gamma(j, :) == 1;
                idx(e) = 1;
                overlap = E_Overlap(H(:, idx));
                if max(overlap(triu(true(sum(idx)), 1))) < 0.3
                    gamma(j,e) = 1;
                    break;
                end
            end
        end
    end

end

mu = zeros(P, E);
mu(gamma~=0) = mu_mean + mu_sd * randn(1, sum(gamma~=0, 'all'));
% mu = mu_mean + mu_sd * randn(P, E);
% mu = mu .* gamma;