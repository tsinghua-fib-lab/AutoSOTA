function [fair,mnce]= eval_fair(pre_y,g)
G = full(ind2vec(g'))';
if size(pre_y,1) == 1
    pre_y = pre_y';
end
Y = full(ind2vec(pre_y'))';
C = G'*Y;
fair = compute_fair(C);
mnce = MNCE(C);
end

