function Z = computeIniGraph(X,A)

n = size(X,2);
m = size(A,2);
options = optimset('Algorithm','interior-point-convex','Display','off');
H=2*A'*A;
H=(H+H')/2;
B=2*X'*A;
for ji=1:n
    ff=-B(ji,:)';
    Z(:,ji)=quadprog(H,ff',[],[],ones(1,m),1,zeros(m,1),ones(m,1),[],options);
end