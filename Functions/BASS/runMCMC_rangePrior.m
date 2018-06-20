function [psamps,chainacc] = ...
    runMCMC_rangePrior(neglogpdf,mySampler,lb,ub,sampOpt)
% Run MCMC sampling, given neglogpost function & range-constraint prior
%
% INPUTS
%       neglogpdf  - function handle for negative log posterior 
%       mySampler  - which sampler to use
%           lb,ub  - range constraints for the parameters
% sampOpt [struct] - options for the MCMC sampler,
%                    with fields {prs0,nsamples,nburn,steps}
%
% OUTPUTS
%           psamps - samples from the MCMC chain
%         chainacc - acceptance rate
% -----------------------------------------------------------------------

% Copyright 2018 Ji Hyun Bak

%% unpack input

% unpack sampling options
prs0 = sampOpt.prs0; % initial value for the parameter
nsamples = sampOpt.nsamples; % chain length
nburn = sampOpt.nburn; % burn-in

% step size is dynamically changing at each chain (semi-adaptive MCMC)
steps = sampOpt.steps; % either a std vector or a lower matrix
if(isequal(size(steps),numel(prs0)*[1 1]))
    if(istril(steps))
        Lmat = steps; % this is lower triangular L, such that L*L' = Cov
    else
        error('runMCMC: steps can either be a std vector or a lower-chol matrix of cov.');
    end
elseif(numel(steps)==numel(prs0))
    stepvec = reshape(steps,size(prs0));
    Lmat = diag(stepvec); % sqrt of diagonal of covariance matrix (naive ver)
end 


%% sample posterior by MCMC

% set up target distribution (with bounds)
flogpdf = @(prs) logPost_withBounds(neglogpdf,prs,lb,ub);

% proposal distribution, scaled using our semi-adaptive algorithm
myRdist = @(x,myLmat) x(:) + myLmat*randn(size(x(:))); % multivariate normal
proprnd = @(x) reshape(myRdist(x,Lmat),size(x)); % match to prs shape

% run sampler
switch mySampler
    case 'mh' % Metropolis-Hastings
        [psamps,accept] = mhsample(prs0,nburn+nsamples,'logpdf',flogpdf,...
            'proprnd',proprnd,'symmetric',1);
        psamps = psamps(nburn+1:end,:); % burn in manually
        chainacc = accept;
        
    otherwise
        % for now, this program only supports MH sampler.
        error('Unknown sampler');
end


end

function logLP = logPost_withBounds(neglogpdf,prs,lb,ub)
% Returns the log-posterior of multinomial Logistic model with lapses,
% with range constraints on the lapse parameters.

% apply bounds
if( any(prs>ub) || any(prs<lb) )
    logLP = -Inf;
    return;
end

% if within range, compute neglogpost
negLP = neglogpdf(prs); % single output argument (no derivatives)
logLP = -negLP;

end
