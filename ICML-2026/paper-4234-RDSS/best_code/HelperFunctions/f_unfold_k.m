function X=f_unfold_k(X, k)
sz=size(X); K=length(sz);
R = k;
C = [k+1:K, 1:k-1];
X = permute(X,[R C]);
X = reshape(X,[sz(R), prod(sz(C))]);
end