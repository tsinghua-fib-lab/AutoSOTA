function [Zu,iter,obj] = CDAL_split_grad_warm(X,f,numanchor,alpha,beta,gamma,seed,fairness_scale,Zu0)
% m      : the number of anchor. the size of Z is m*n.
% X      : n*di

%% initialize
maxIter = 50 ; % the number of iterations
IterMax = 50;

m = numanchor;
numview = length(X);
numsample = size(X{1},1);

numfair = length(unique(f));
G =  full(sparse(f,1:numsample,1,numfair,numsample));
G_head = G - sum(G, 2) / numsample;
norm_G_head = svds(G_head,1);

XX = [];
for iv = 1 : numview
    X{iv} = mapstd(X{iv}',0,1);
    XX = [XX;X{iv}];
end

%% initial Au,As,Zu,Zs
rand('twister',seed);
[IDX,ACu] = kmeans(XX',m, 'MaxIter',100,'Replicates',10);
ACu = ACu';
[~,ind,~] = graphgen_anchor(XX',m);
ACs = XX(:, ind);

Zs = computeIniGraph(XX,ACs);
if nargin < 9 || isempty(Zu0)
    Zu = zeros(m,numsample);
    for is = 1:numsample
        Zu(IDX(is),is) = 1;
    end
else
    Zu = Zu0;
end

count = 1;
for iv = 1:numview
    di = size(X{iv},1); 
    Au{iv} = ACu(count:count+di-1,:);
    As{iv} = ACs(count:count+di-1,:);
    count = count+di;
end

flag = 1;
iter = 0;
%%
while flag
    iter = iter + 1;
    
    %% optimize W
    [U,~,V] = svd(Zs*G','econ');
    W = U*V';
    
    %% optimize As
    for iv = 1:numview
        A_syl = beta * (Au{iv} * Au{iv}');
        B_syl = Zs * Zs';
        C_syl = (X{iv} - Au{iv} * Zu) * Zs';
        As{iv} = sylvester(A_syl, B_syl, C_syl);
    end
    
    %% optimize Zs
    Sum_AsAs = zeros(m,m);
    Sum_AsX = zeros(m,numsample);
    for iv = 1:numview
        Sum_AsAs = Sum_AsAs + As{iv}' * As{iv};
        Sum_AsX = Sum_AsX + As{iv}' * (X{iv} - Au{iv} * Zu);
    end
    Q_Zs = Sum_AsAs + gamma * eye(m);
    
    C_Zs = Sum_AsX + gamma * W * G;
    
    H_qp = 2 * Q_Zs;
    Aeq = ones(1, m);
    beq = 1;
    lb = zeros(m, 1);
    ub = ones(m, 1); 
    opts = optimset('Display','off','Algorithm','interior-point-convex');
    for i = 1:numsample
        f_qp = -2 * C_Zs(:, i);
        Zs(:, i) = quadprog(H_qp, f_qp, [], [], Aeq, beq, lb, ub, [], opts);
    end
    
    %% optimize Au
    for iv = 1:numview
        A_syl = beta * (As{iv} * As{iv}');
        B_syl = Zu * Zu';
        C_syl = (X{iv} - As{iv} * Zs) * Zu';
        Au{iv} = sylvester(A_syl, B_syl, C_syl);
    end
    
    %% optimize Zu
    Sum_AuAu = zeros(m,m);
    Sum_AuX = zeros(m,numsample);
    for iv = 1:numview
        Sum_AuAu = Sum_AuAu + Au{iv}' * Au{iv};
        Sum_AuX = Sum_AuX + Au{iv}' * (X{iv} - As{iv} * Zs);
    end
    L_zu_recon = 2 * norm(Sum_AuAu, 2);
    L_zu_fair = 2 * alpha * (norm_G_head^2);
    eta_recon = 1 / (L_zu_recon + eps);
    eta_fair = fairness_scale / (L_zu_fair + eps);

     for piter = 1:5
        Grad_Zu_recon = 2 * Sum_AuAu * Zu - 2 * Sum_AuX;
        Grad_Zu_fair = 2 * alpha * (Zu * G_head' ) * G_head;
        Zu = Zu - eta_recon * Grad_Zu_recon - eta_fair * Grad_Zu_fair;
        Zu = ProjSimplex(Zu);
    end
    
    %% objective value calculation
    term1 = 0;
    term2 = norm(Zu * G_head','fro')^2;
    term3 = 0;
    term4 = norm(G-W'*Zs,'fro')^2;
    for iv = 1:numview
        term1 = term1 + norm(X{iv}-Au{iv}*Zu-As{iv}*Zs,'fro')^2;
        term3 = term3 + norm(Au{iv}' * As{iv},'fro')^2;
    end
    
    obj(iter) = term1+alpha*term2+beta*term3+gamma*term4;
    
	if (iter>1) && (abs((obj(iter-1)-obj(iter))/(obj(iter-1)))<1e-3 || iter>maxIter || obj(iter) < 1e-10)
        flag = 0;
    end
end