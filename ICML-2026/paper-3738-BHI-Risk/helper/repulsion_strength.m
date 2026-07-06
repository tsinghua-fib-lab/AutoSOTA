function [effect_hyperedge_per_predictor, repulsion_term, ...
    average_hyperedge_overlap, redundancy_ratio] = ...
    repulsion_strength(gamma_prob, m_prob, z_prob)
[P, E_edges] = size(gamma_prob);

[E_O, ~] = E_Overlap(m_prob);

gamma_prob_joint = gamma_prob .* z_prob'; % P x E

% effective number of hyperedges
effect_hyperedge_per_predictor = sum(gamma_prob_joint, 2); % P x 1


repulsion_term = zeros(P, 1);
for e1 = 1:E_edges
    for e2 = (e1+1):E_edges
        repulsion_term = repulsion_term + E_O(e1, e2) * ...
            gamma_prob_joint(:, e1) .* gamma_prob_joint(:, e2);
    end
end

% average hyperedge overlap
average_hyperedge_overlap = repulsion_term ./ (effect_hyperedge_per_predictor.^2);
% redundancy ratio
redundancy_ratio = repulsion_term ./ effect_hyperedge_per_predictor;


end