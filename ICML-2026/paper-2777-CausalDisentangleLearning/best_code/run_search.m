pkg load statistics;
pkg load optim;
addpath(genpath("./"));
load("./datasets/yaleA_3view.mat");
Y=double(y);
g=double(g);
k=length(unique(Y));
n=length(Y);

% Focus search on parameters that give non-zero Balance
% Based on ablation: Lorth (beta) gives Bal=8.33, ACC=60.82
% Full CDAL adds HSIC (alpha) + Lsens (gamma) for Bal=12.50, ACC=60.36
% So beta should be moderate-large, alpha and gamma moderate

alpha_vals = [0.001, 0.1, 10];
beta_vals = [0.1, 10, 1000];
gamma_vals = [0.001, 0.1, 10];
anchor_vals = [2*k, 3*k];  % k=15, so 30 and 45

fid = fopen('./res/yale_search_results.txt', 'w');
fprintf(fid, 'anchor,alpha,beta,gamma,ACC,NMI,Pur,Fsc,Bal,MNCE,Time,Iter\n');

total = length(anchor_vals) * length(alpha_vals) * length(beta_vals) * length(gamma_vals);
count = 0;

for ichor = 1:length(anchor_vals)
    for ia = 1:length(alpha_vals)
        for ib = 1:length(beta_vals)
            for ig = 1:length(gamma_vals)
                count = count + 1;
                tic;
                [Z,iter,obj] = CDAL(X,g',anchor_vals(ichor),alpha_vals(ia),beta_vals(ib),gamma_vals(ig));
                [UU,S,V]=mySVD(Z',k);
                F = UU ./ repmat(sqrt(sum(UU .^ 2, 2)), 1, size(UU,2));
                MAXiter = 1000;
                REPlic = 20;
                res = [];
                balance_list = [];
                mnce_list = [];
                for rep = 1:10
                    pY = kmeans(F, k, 'maxiter', MAXiter, 'replicates', REPlic, 'emptyaction', 'singleton');
                    res(rep,:) = Clustering8Measure(Y, pY);
                    [bal,mnce]= eval_fair(pY,g);
                    balance_list(rep) = bal;
                    mnce_list(rep) = mnce;
                end
                tempResBest = mean(res);
                tempBalance = mean(balance_list);
                tempMnce = mean(mnce_list);
                timer = toc;

                fprintf('[%d/%d] anchor=%d,a=%g,b=%g,g=%g | ACC=%.2f NMI=%.2f Pur=%.2f Fsc=%.2f Bal=%.2f MNCE=%.2f iter=%d t=%.0fs\n', ...
                    count, total, anchor_vals(ichor), alpha_vals(ia), beta_vals(ib), gamma_vals(ig), ...
                    tempResBest(1)*100, tempResBest(2)*100, tempResBest(3)*100, tempResBest(4)*100, ...
                    tempBalance*100, tempMnce*100, iter, timer);

                fprintf(fid, '%d,%g,%g,%g,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.1f,%d\n', ...
                    anchor_vals(ichor), alpha_vals(ia), beta_vals(ib), gamma_vals(ig), ...
                    tempResBest(1), tempResBest(2), tempResBest(3), tempResBest(4), ...
                    tempBalance, tempMnce, timer, iter);
            end
        end
    end
end

fclose(fid);
fprintf('\nSearch complete. Results saved to ./res/yale_search_results.txt\n');
