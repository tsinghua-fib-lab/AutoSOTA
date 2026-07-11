function [X,mS]=f_prox_t_LpSq_dct(Y,rho,oldS,p,q)
% Proximal step for Lp(Sq) regularization in the DCT domain.

[U,S,V]=f_tsvd_dct(Y);
mS=f_fdiag_to_matrix(S);
mS=f_prox_p_over_q_inexact(mS,rho,oldS,p,q);

[~,~,n3]=size(Y);
X=Y*0;
for i=1:n3
    X(:,:,i)=U(:,:,i)*diag(mS(:,i))*V(:,:,i)';
end
X=f_idct_tube(X);
end
