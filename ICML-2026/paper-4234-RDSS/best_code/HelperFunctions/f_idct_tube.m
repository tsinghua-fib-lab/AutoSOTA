function X = f_idct_tube(X)
sz = size(X);
X = f_unfold_k(X,3);
X = idct(X);
X = f_fold_k(X,sz,3);
end