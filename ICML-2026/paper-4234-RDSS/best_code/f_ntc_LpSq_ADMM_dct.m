function memo=f_ntc_LpSq_ADMM_dct(obs,opts,memo)
% ADMM solver for tensor completion with Lp(Sq) regularization in the DCT domain.
% Extended with PSNR-based plateau detection for early stopping (CODE-1).

lambda=opts.para.lambda;
rho=opts.para.rho;
nu=opts.para.nu;
p=opts.p;
q=opts.q;

if isfield(opts,'weightGap')
    weightGap=opts.weightGap;
else
    weightGap=1;
end
if isfield(opts,'recordGap')
    recordGap=opts.recordGap;
else
    recordGap=1;
end

% PSNR-based early stopping configuration (CODE-1)
if isfield(opts,'psnr_patience')
    psnr_patience=opts.psnr_patience;
else
    psnr_patience=0;
end
if isfield(opts,'psnr_tol')
    psnr_tol=opts.psnr_tol;
else
    psnr_tol=0.01;
end

normTruth=norm(double(memo.truth(:)));
B=zeros(obs.tsize); B(obs.idx)=1;
T=zeros(obs.tsize); T(obs.idx)=obs.y;

X=zeros(obs.tsize);
if isfield(opts,'initL')
    X=opts.initL;
end
Y=X;

if isfield(opts,'initL')
    [~,weightS,~]=f_tsvd_dct(X);
    weightS=f_fdiag_to_matrix(weightS);
else
    weightS=zeros(min(obs.tsize(1),obs.tsize(2)),obs.tsize(3));
end

% Initialize PSNR plateau tracking (CODE-1)
best_psnr=-inf;
best_X=X;
plateau_count=0;

fprintf('++++LpSq-ADMM-DCT: p=%g, q=%g, weightGap=%d',p,q,weightGap);
if psnr_patience>0
    fprintf(', psnr_patience=%d, psnr_tol=%g',psnr_patience,psnr_tol);
end
fprintf('++++\n');

for iter=1:opts.MAX_ITER_OUT
    oldX=X;

    Z=(Y+rho*X+B.*T)./(rho+B);
    X_tmp=Z-Y/rho;
    [X,newS]=f_prox_t_LpSq_dct(X_tmp,lambda/rho,weightS,p,q);

    memo.iter=iter;
    memo.rho(iter)=rho;
    memo.eps(iter)=norm(double(X(:)-oldX(:)))/(norm(double(oldX(:)))+eps);

    if mod(iter,recordGap)==0 || iter==1
        memo.err(iter)=norm(double(X(:)-memo.truth(:)))/normTruth;
        memo.pnsr(iter)=h_Psnr(memo.truth(:),X(:));

        % PSNR plateau detection (CODE-1)
        if psnr_patience>0
            if memo.pnsr(iter)>best_psnr+psnr_tol
                best_psnr=memo.pnsr(iter);
                best_X=X;
                plateau_count=0;
            else
                plateau_count=plateau_count+1;
            end
        end
    end

    if opts.verbose && mod(iter,memo.printerInterval)==0
        if memo.err(iter)==0
            tmpErr=norm(double(X(:)-memo.truth(:)))/normTruth;
            tmpPsnr=h_Psnr(memo.truth(:),X(:));
        else
            tmpErr=memo.err(iter);
            tmpPsnr=memo.pnsr(iter);
        end
        if psnr_patience>0
            fprintf('++%d: eps=%0.2e, err=%0.2e, rho=%0.2e, PSNR=%0.2f (best=%0.2f, plat=%d)\n', ...
                iter,memo.eps(iter),tmpErr,memo.rho(iter),tmpPsnr,best_psnr,plateau_count);
        else
            fprintf('++%d: eps=%0.2e, err=%0.2e, rho=%0.2e, PSNR=%0.2f\n', ...
                iter,memo.eps(iter),tmpErr,memo.rho(iter),tmpPsnr);
        end
    end

    % Original convergence check (retained as safety net)
    if (memo.eps(iter)<opts.MAX_EPS) && (iter>60) && memo.eps(iter)>1e-10
        memo.err(iter)=norm(double(X(:)-memo.truth(:)))/normTruth;
        memo.pnsr(iter)=h_Psnr(memo.truth(:),X(:));
        fprintf('Stopped:%d: eps=%0.2e, err=%0.2e, rho=%0.2e, PSNR=%0.2f\n', ...
            iter,memo.eps(iter),memo.err(iter),memo.rho(iter),memo.pnsr(iter));
        break;
    end

    % PSNR plateau-based early stopping (CODE-1)
    if psnr_patience>0 && iter>60 && plateau_count>=psnr_patience
        X=best_X;
        memo.err(iter)=norm(double(X(:)-memo.truth(:)))/normTruth;
        memo.pnsr(iter)=h_Psnr(memo.truth(:),X(:));
        fprintf('PSNR-Stopped:%d: PSNR=%0.2f (best=%0.2f), plateau=%d iters, eps=%0.2e\n', ...
            iter,memo.pnsr(iter),best_psnr,plateau_count,memo.eps(iter));
        break;
    end

    Y=Y+rho*(X-Z);
    if mod(iter,weightGap)==0
        weightS=newS;
    end
    rho=min(rho*nu,opts.MAX_RHO);
end

if memo.err(memo.iter)==0
    memo.err(memo.iter)=norm(double(X(:)-memo.truth(:)))/normTruth;
end
if memo.pnsr(memo.iter)==0
    memo.pnsr(memo.iter)=h_Psnr(memo.truth(:),X(:));
end
memo.T_hat=X;
end
