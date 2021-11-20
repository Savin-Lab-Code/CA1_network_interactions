% Compute 3D tuning function (x,y,k) for each of the synthetic data cells
% this code is based on Cristina Savin's code released with the 2016 paper
% "Estimating nonlinear neural response functions using GP priors and Kronecker methods" Savin, Tkacik, NeurIPS 2016
% 
% IMPORTANT: within the folder 'pfgp_nips', unzip gpml.zip before continuing here
%
%%
clear,clc
addpath('pfgp_nips'); % load code from repository
run ./data/pfgp_nips/startup.m
%% parameters
N=50; % N cells
n=20; % n bins
%% load data generated from script "synthetic_data"
y=load(['data/y_Ising_Osc_',num2str(N),'_C']);
x=load(['data/x_Ising_Osc_',num2str(N)]);
%% split k in to 10 equally populated bins
npk=10; % bins to split into the pk
nk=sum(y,2);
edges = prctile(nk,linspace(0,100,npk+1));
N = size(y,2);
neurons= 1:N ;
%% save log tuning functions (3D place fields): both mean and std
pfs = zeros(N,2,n,n,npk);
%% add a tiny bit of noise on the position, else they are all aligned
x = x + 0.02*randn(size(x));
x(x>1)=0.99999;
x(x<0)=0.00001;

%% fit 3D GP field for each cell 
for i = 1:1
    neuron = neurons(i)
    maxiter = 100;
    [pf, hyp,counts,ll] = getPFGPGridFkFast(y(:,neuron),x,nk,n,n,npk,maxiter);
    if (max(pf.mtuning(:)) > 100 || isnan(ll) || ll > 100) && maxiter > 2
        % if it explodes, reduce iterations
        maxiter = round(maxiter / 2);
        [pf, hyp,counts,ll] = getPFGPGridFkFast(y(:,neuron),x,nk,n,n,npk,maxiter);
    end
    pfs(neuron,1,:,:,:) = pf.mp;
    pfs(neuron,2,:,:,:) = pf.varp;
end
save(['data/PFGP3D_N',num2str(N)],'pfs')
