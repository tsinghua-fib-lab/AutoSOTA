function memo=h_construct_memo(opts)
max_iter=opts.MAX_ITER_OUT;
memo.max_iter=max_iter;
memo.iter=0;
memo.opts=opts;
memo.printerInterval=10;
vZero = zeros(max_iter,1);
memo.fval=vZero;
% relative error between every two iterations
memo.eps=vZero;
% error: distance to the truth 
memo.err=vZero;
% RSE
memo.rse=vZero; 
% SE: squared error 
memo.se=vZero;
% PSNR
memo.psnr=vZero;
% SSIM
memo.ssim=vZero;
