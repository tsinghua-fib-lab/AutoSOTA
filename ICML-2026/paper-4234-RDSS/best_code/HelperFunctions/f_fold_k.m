function M=f_fold_k(M, sz,k)
K=length(sz);
R = k;
C = [k+1:K,1:k-1];
M = reshape(M,[sz(R), sz(C)]);
order_dst = [K-k+2:K, 1:K-k+1];    
M = permute(M,order_dst);
end