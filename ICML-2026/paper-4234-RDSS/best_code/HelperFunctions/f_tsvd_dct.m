function [U,S,V] = f_tsvd_dct(X)
sz =size(X);
X = f_dct_tube(X);
r = min(sz(1),sz(2));
U = zeros(sz(1),r,sz(3));
S = zeros(r,r,sz(3));
V = zeros(sz(2),r,sz(3));
for i=1:sz(3)
    [U(:,:,i),S(:,:,i),V(:,:,i)]=svd(X(:,:,i),'econ');
end