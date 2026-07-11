function [ PSNR, MSE] =h_Psnr(T_true,T_hat)
dT=double(T_true-T_hat);
MSE = mean(dT(:).*dT(:));
peak = max(T_true(:));
PSNR = 10*log10(peak^2/MSE);
