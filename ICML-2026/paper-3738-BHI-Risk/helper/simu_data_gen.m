function [X, Y, alpha, Beta, H, gamma, mu] = simu_data_gen(N, P, V, E, ...
    mu_mean, mu_sd, seed, nRarePerEdge, nCommonPerEdge, seed_H)

if ~exist('seed_H', 'var') || isempty(seed_H)
    seed_H = 42;
end




% Hypergraph generation with forced mixing
rng(seed_H);
rare_idx   = 1:round(V/4);          % ~25% rare
common_idx = (round(V/4)+1):V;         % remaining common
H = simulate_mixed_hypergraph(V, E, rare_idx, common_idx, ...
    nRarePerEdge, nCommonPerEdge, seed_H);
[gamma, mu] = simulate_mechanisms(P, H, mu_mean, mu_sd);

Beta = compute_beta(H, gamma, mu);
dv_inv = 1 / sqrt(E);
Beta = Beta * dv_inv;

rng(seed);
X = randn(N, P);
xb = X * Beta;

% prevalence assignment
prev0 = zeros(V,1);
prev0(rare_idx)   = 0.02 + 0.03*rand(numel(rare_idx),1);   % 2–5%
prev0(common_idx) = 0.1  + 0.2*rand(numel(common_idx),1); % 10–30%

alpha = zeros(1,V);
for v = 1:V
    alpha(v) = calibrate_intercept(xb(:,v), prev0(v));
end

eta = X * Beta + alpha;
prob = 1 ./ (1 + exp(-eta));
Y = binornd(1, prob);


end