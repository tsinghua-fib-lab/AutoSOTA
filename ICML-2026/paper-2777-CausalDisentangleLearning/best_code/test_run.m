pkg load statistics;
pkg load optim;
addpath(genpath("./"));
load("./datasets/yaleA_3view.mat");
Y=double(y);
g=double(g);
k=length(unique(Y));
n=length(Y);

anchor=[k];
alpha=[0.001];
beta=[0.001];
gamma=[0.001];

tic;
[Z,iter,obj] = CDAL(X,g',anchor(1),alpha(1),beta(1),gamma(1));
[UU,S,V]=mySVD(Z',k);
F = UU ./ repmat(sqrt(sum(UU .^ 2, 2)), 1, size(UU,2));
MAXiter = 1000;
REPlic = 20;
res = [];
balance_list = [];
mnce_list = [];
for rep = 1:5
    pY = kmeans(F, k, 'maxiter', MAXiter, 'replicates', REPlic, 'emptyaction', 'singleton');
    res(rep,:) = Clustering8Measure(Y, pY);
    [bal,mnce]= eval_fair(pY,g);
    balance_list(rep) = bal;
    mnce_list(rep) = mnce;
end
tempResBest = mean(res);
tempBalance = mean(balance_list);
tempMnce = mean(mnce_list);
t=toc;
fprintf('Anchor:%d alpha:%g beta:%g gamma:%g Res: ACC=%.4f NMI=%.4f Pur=%.4f Fsc=%.4f Bal=%.4f MNCE=%.4f Time:%.4f\n', anchor(1), alpha(1), beta(1), gamma(1), tempResBest(1), tempResBest(2), tempResBest(3), tempResBest(4), tempBalance, tempMnce, t);
