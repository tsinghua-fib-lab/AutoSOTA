function mS=f_fdiag_to_matrix(S)
mS = zeros(size(S,1),size(S,3));
for i = 1:size(S,3)
    mS(:,i)=diag(S(:,:,i));
end
end