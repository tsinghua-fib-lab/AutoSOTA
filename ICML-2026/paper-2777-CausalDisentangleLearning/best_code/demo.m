clear;
clc;
warning off;
addpath(genpath('./'));

%% dataset
ds = {'yaleA_3view'};
dsPath = '.\datasets\';
resPath = './res/';

for dsi =1:1:length(ds)
    dataName = ds{dsi}; disp(dataName);
    load(strcat(dsPath,dataName));
    Y=double(y);
    g = double(g);
    k = length(unique(Y));
    n = length(Y);

    alpha=[0.00001,0.001,0.1,10,1000];
    beta=[0.00001,0.001,0.1,10,1000];
    gamma=[0.00001,0.001,0.1,10,1000];
    anchor=[k,2*k,3*k,4*k,5*k];
    
    txtpathmax = strcat(resPath,strcat(dataName,'.txt'));
    dlmwrite(txtpathmax, strcat('Dataset:',cellstr(dataName), '  Date:',datestr(now)),'-append','delimiter','','newline','pc');
        
    %%
    allresult = [];
    for ichor = 1:length(anchor)
        for ia = 1:length(alpha)
            for ib = 1:length(beta)
                for ig = 1:length(gamma)
                    tic;
                    [Z,iter,obj] = CDAL(X,g',anchor(ichor),alpha(ia),beta(ib),gamma(ig));
                    [UU,~,V]=mySVD(Z',k);
                    F = UU ./ repmat(sqrt(sum(UU .^ 2, 2)), 1, size(UU,2));
                    MAXiter = 1000; 
                    REPlic = 20; 
                    for rep = 1 : 20
                        pY = kmeans(F, k, 'maxiter', MAXiter, 'replicates', REPlic, 'emptyaction', 'singleton');
                        res(rep,:) = Clustering8Measure(Y, pY);
                        [balance,mnce]= eval_fair(pY,g);
                    end
                    tempResBest = mean(res);
                    tempBalance = mean(balance);
                    tempMnce = mean(mnce);
                    
                    timer  = toc;
                    fprintf('Anchor:%d \t alpha:%d\t beta:%d\t gamma:%d\t Res:%12.6f %12.6f %12.6f %12.6f %12.6f %12.6f \tTime:%12.6f \n',[anchor(ichor) alpha(ia) beta(ib) gamma(ig) tempResBest(1) tempResBest(2) tempResBest(3) tempResBest(4) tempBalance tempMnce timer]);
                    dlmwrite(txtpathmax, [anchor(ichor) alpha(ia) beta(ib) gamma(ig) tempResBest tempBalance tempMnce timer],'-append','delimiter','\t','newline','pc');
                    allresult = [allresult;tempResBest tempBalance tempMnce timer];
                end
            end
        end
    end
    [c,d] = max(allresult(:,1));
    maxresult = allresult(d,:);
    dlmwrite('./totalResults/AllDatasetResult.txt',char(dataName),'-append','delimiter','\t','newline','pc');
    dlmwrite('./totalResults/AllDatasetResult.txt',maxresult,'-append','delimiter','\t','newline','pc');
end


