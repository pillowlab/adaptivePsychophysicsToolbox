function [probEst,prsMean,infoCrit,covEnt,chainLmat,moreOutput] = ...
    fun_BASS_MCMC(xx,dat,dims,moreOpts)
% Run posterior inference and calculate BASS utility functions, with MCMC
% 
% INPUTS
%           xx [array] - each row is a stimulus at which to evaluate probs
%         dat [struct] - training data:
%                        .x = stimuli (each row is a stim),
%                        .y = output state (one from 0:(ydim-1))
%        dims [struct] - degrees of freedom of variables
%                        .y = of output
%                        .g = of the input feature vector g(x)
%    moreOpts [struct] - more options for inference
%                        MCMC-related fields: {prs0,steps,nsamples,nburn}
%                        for MCMC re-setting: {prsInit,nburnInit}
%                        prior-related: {prior,prsLB,prsUB}
%                        other options: {reportMoreValues}
%
% OUTPUTS
%             probEst - probability estimate based on the sampled posterior
%             prsMean - mean of all sampled parameters
%            infoCrit - mutual information calculated from posterior
%              covEnt - entropy of sampled posterior (fow now wrt weights)
%           chainLmat - the cholesky cov^(1/2) matrix from current chain
% moreOutput [struct] - fields {accept,psamps,cov}
% -----------------------------------------------------------------------

% Copyright 2018 Ji Hyun Bak

%% unpack input

% prior hyperparameters input
myPrior = moreOpts.prior; 

% get range constraints
lb = moreOpts.prsLB;
ub = moreOpts.prsUB;

% additional switches
reportMoreValues = moreOpts.reportMoreValues;
talkative = moreOpts.talkative; % display option
myshowopt = max(0,talkative-1); % set through talkative; one level down

cutPartialCov = false; % always match full covariance
mySampler = 'mh'; % for now only use Metropolis-Hastings (other samplers: TO-DO)

%% run MCMC sampling

% negative log posterior
neglogpdf = @(prs) negLogPost_MNLogLapse(prs,dat,dims,myPrior);

% extract sampler options
sampOpt = struct('prs0',moreOpts.prs0,'steps',moreOpts.steps,...
    'nsamples',moreOpts.nsamples,'nburn',moreOpts.nburn);

% run sampler
try
    [psamps,accept] = runMCMC_rangePrior(neglogpdf,mySampler,lb,ub,sampOpt); 
    
catch ME
    % ------------------------------------------------------
    % note: this try-catch is actually a legacy chunk.
    %       we should not be catching errors at this point.
    % ------------------------------------------------------
    disp([' - Error: ',ME.identifier]);
    disp(' - resetting initialization and re-running...')
    % reset and rerun
    sampOpt.prs0 = moreOpts.prsInit; 
    sampOpt.nburn = moreOpts.nburnInit; 
    [psamps,accept] = runMCMC_rangePrior(neglogpdf,mySampler,lb,ub,sampOpt); 
end

if(myshowopt>0)
    display([' --- accept ',num2str(accept,3)]);
end

% process sample covariance, to match next step sizes
chaincov = cov(psamps); % chain covariance
[chainLmat,Lwarn] = matchStepFromCov(chaincov,dims,cutPartialCov);
if(Lwarn>0)
    disp(['- fun_BASS_MCMC: cov match warning: ',num2str(Lwarn)]);
end

% covariance entropy
K0 = (dims.y)*(dims.g);
chaincovcut = chaincov(1:K0,1:K0);
covEnt = logdet(chaincovcut); % only wrt weights, for now

% pack more information
moreOutput = struct('accept',accept);
if(reportMoreValues) % optional
    moreOutput.psamps = psamps;
    moreOutput.cov = chaincov;
end

%% predictive distribution P(y|x,w)

%%% two ways of getting the distribution:

% -- evaluate prob at the mean of samples
prsMean = mean(psamps,1);
probAtMean = getProbAtPrs_MNLogLapse(prsMean,dims,xx);

% -- evaluate probabilities at individual samples, then take average
nx = size(xx,1);
ydim = dims.y;
nsamples = size(psamps,1);
Pyw = zeros(ydim+1,nx,nsamples);
for j = 1:nsamples
    myProb = getProbAtPrs_MNLogLapse(psamps(j,:),dims,xx);
    Pyw(:,:,j) = myProb;
end
Py = mean(Pyw,3); % marginal output distribution

% report both estimated probabilities 
probEst_struct = struct('fmean',probAtMean,'meanf',Py);
% - fmean: f(<s>), prob at mean parameters
% - meanf: <f(s)>, mean prob at all sampled parameters

probEst = probEst_struct.meanf; % always use <f(s)> in this project


%%% mutual information I(y;w) 
Hy = -sum(Py.*log(Py))'; % marginal entropy
Hyw = -mean(sum(Pyw.*log(Pyw),1),3)'; % conditional entropy
infoCrit = Hy - Hyw; % mutual information


end

% ------------------------------------------------------------------------

function [myLmat_full,warn] = matchStepFromCov(chaincov_full,dims,cutpartialcov)
% semi-adaptive MCMC based on adaptive Metropolis-Hastings [Haario2001]

% whether to take partial covariance wrt weights
if(cutpartialcov) 
    Kcut = (dims.y)*(dims.g);
else
    Kcut = size(chaincov_full,1); 
end

% fixed parameters for optimizing
sdratio = (2.38^2)/Kcut; % [Gelman1996, via Harrio2001]
addeps = 0.01; % just to prevent singular covariance matrix


%%% cholesky-Lmat based formulation, matching full cov

covCut = chaincov_full(1:Kcut,1:Kcut);
myCmat = sdratio*(covCut+(addeps^2)*eye(size(covCut)));

% -- choleskly decomposition
[myLmat,warn] = cholfix_nearestSPD(myCmat,'lower'); % for occasional non-SPD cases
if(warn>=2)
    % means that nearestSPD did not succeed in two attempts
    myLmat = diag(sqrt(diag(myCmat))); % just match the std
end

% for now, only match weight covariance, fixing lapse parts
myLmat_full = eye(size(chaincov_full)); % unit diagonal matrix
myLmat_full(1:Kcut,1:Kcut) = myLmat; % fill in the weight cov part

end
