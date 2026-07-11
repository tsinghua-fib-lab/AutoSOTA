
function X = f_dct_tube(X)
sz = size(X);
X = f_unfold_k(X,3);
X = dct(X);
X = f_fold_k(X,sz,3);
end