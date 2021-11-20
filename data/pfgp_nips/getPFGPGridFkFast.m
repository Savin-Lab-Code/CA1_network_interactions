function [pf,hyp2,counts,ll] = getPFGPGridFkFast(z,x,nk,n1,n2,n3,maxiter)
%inputs: spike counts as function of position
% n1,n2: grid dimensions
if nargin < 5, n2 = n1; end
if nargin < 6, n3 = n1; end
if nargin < 7, maxiter = 100; end

if min(x(:))== 0, x = x+1e-10;end

% create the "computational grid"
[xx,xg,i,j,kk] = binPositions(x,nk,n1,n2,n3);

counts = zeros(n1,n2,n3);
for k=1:n3
    kidx = kk==k;
    counts(:,:,k) = full(sparse(i(kidx),j(kidx),z(kidx),n1,n2))./full(sparse(i(kidx),j(kidx),1,n1,n2));
end

% setup the GP
% cv = {@covProd,{{@covMask,{1,@covSEiso}},{@covMask,{2,@covSEiso}},{@covMask,{3,@covSEiso}}}};
cvGrid = {@covGrid, {@covSEiso,  @covSEiso, @covSEiso}, xg};
hyp0.cov = log([.1  1 .1 1 0.05 1]);
hyp0.mean = .005;

X = covGrid('expand', cvGrid{3});
lik = {@likPoisson, 'exp'};

idxBin = binPositions(X(:,1:2),X(:,3),n1,n2,n3);
idxSeen = ~isnan(counts(idxBin));

tic
hyp2 = minimize(hyp0, @gp, -maxiter, @infGrid_Laplace, @meanConst, cvGrid, lik, xx, z);
toc
ll = gp(hyp2, @infGrid_Laplace, @meanConst, cvGrid, lik, xx, z)/length(z);
fprintf('Log-likelihood learned with Kronecker: %.04f\n',ll);

% posterior predictions
tic
[Ef2, Varf2, fmu2, fs2] = gp(hyp2, @infGrid_Laplace, @meanConst, cvGrid, lik, xx, z, X(idxSeen,:));
toc

ml = zeros(1,n1*n2*n3);
ml(idxSeen) = Ef2;
pf.ml =reshape(ml,n1,n2,n3); %mean and variance for f (log lambda)

varl = zeros(1,n1*n2*n3);
varl(idxSeen) = Varf2;
pf.varl = reshape(varl,n1,n2,n3);

mp = zeros(1,n1*n2*n3);
mp(idxSeen) = fmu2;
pf.mp = reshape(mp,n1,n2,n3); % mean and variance of observations

varp=zeros(1,n1*n2*n3);
varp(idxSeen) = fs2;
pf.varp = reshape(varp,n1,n2,n3);

pf.mtuning= reshape(exp(mp +varp/2),n1,n2,n3); % mean and variance of f (log normal)
pf.vartuning = reshape((exp(varp)-1).*exp(2*varp+mp),n1,n2,n3);
