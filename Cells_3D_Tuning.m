% Compute 3D tuning function (x,y,k) for each of the synthetic data cells
% this code is based on Cristina Savin's code released with the 2016 paper
% "Estimating nonlinear neural response functions using GP priors and Kronecker methods" Savin, Tkacik, NeurIPS 2016
% 
% IMPORTANT: within the folder 'pfgp_nips', unzip gpml.zip before continuing here
%

%% load libraries
clear,clc
addpath('pfgp_nips');
run ./data/pfgp_nips/startup.m

%% parameters
N=50; % N cells
n=20; % n (equally spaced) 2D spatial bins
npk=10; % number of (equally populated) bins to split cell synchrony into

%% load data generated from script "synthetic_data"
y=load(['data/y_Ising_Osc_',num2str(N),'_C']);
x=load(['data/x_Ising_Osc_',num2str(N)]);

%% compute nk (i.e. cells synchrony)
nk=sum(y,2); % nk = sum cells active at each time point

%% allocate space to save log tuning functions (3D place fields): both mean and std
pfs = zeros(N,2,n,n,npk);

%% fit 3D firing field for each cell by calling getPFGPGridFkFast.m - see file in folder pfgp_nips

% this will fit mean+std of log-firing of each cell as a function of spatial location and population synchrony
% under the assumptions that responses are Poisson, smoothness is ensured by assuming a GP prior

for i = 1:N
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

%% save results
save(['data/PFGP3D_N',num2str(N)],'pfs')
