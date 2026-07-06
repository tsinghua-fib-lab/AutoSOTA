function H = simulate_mixed_hypergraph(V, E, rare_idx, common_idx, ...
    nRarePerEdge, nCommonPerEdge, seed)
%%%%% Hypergraph generation with forced mixing

rng(seed);
H = zeros(V, E);

for e = 1:E
    % sample rare diseases
    r_idx = rare_idx(randperm(numel(rare_idx), nRarePerEdge));

    % sample common diseases
    c_idx = common_idx(randperm(numel(common_idx), nCommonPerEdge));

    H([r_idx, c_idx], e) = 1;
end

for v = common_idx
    if sum(H(v, :)) == 0
        e = randi(E);
        H(v,e) = 1;
    end
end

% Ensure rare diseases appear multiple times
% after initial construction
for v = rare_idx
    if sum(H(v,:)) < 2
        e = randi(E);
        H(v,e) = 1;
    end
end


end
