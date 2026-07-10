function [balance] = compute_fair(G)

maximum = max(G,[],1);
minimum = min(G,[],1);
balance_vector = minimum./maximum;
balance = min(balance_vector);
end

