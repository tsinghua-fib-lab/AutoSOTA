function [ anchor, ind2, score ] = graphgen_anchor(X, m )

[n,d] = size(X);
vm = min(X,[],1);
Xm = ones(n,1)*vm;
X = X-Xm;
% for i=1:d

score = sum(X, 2);
score(:,1) = score/max(score);
[~,ind(1)] = max(score);
for i=2:m
   score(:,i) = score(:,i-1).*(ones(n,1)-score(:,i-1));
   score(:,i) = score(:,i)/max(score(:,i));
   [~,ind(i)] = max(score(:,i));
end
 ind2 = sort(ind,'ascend');
anchor = X(ind2,:);

end
