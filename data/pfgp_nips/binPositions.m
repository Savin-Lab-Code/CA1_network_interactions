function [xx,xg,i,j,kk] = binPositions(x,nk,n1,n2,n3)

% create the "computational grid"
x1 = linspace(0,1.0001,n1+1); 
x2 = linspace(0,1.0001,n2+1);
x3 = linspace(0,1.0001,n3+1);

% centers
xg = {(x1(1:n1)+x1(2:n1+1))'/2, (x2(1:n2)+x2(2:n2+1))'/2,(x3(1:n3)+x3(2:n3+1))'/2};

% indexes
i = ceil((x(:,1)-x1(1))/(x1(2)-x1(1)));
j = ceil((x(:,2)-x2(1))/(x2(2)-x2(1)));
%kk = ceil((nk(:)-x3(1))/(x3(2)-x3(1)));
edges = prctile(nk,linspace(0,100,n3+1));
kk=discretize(nk,edges);


xx = sub2ind([n1,n2,n3],i,j,kk);
