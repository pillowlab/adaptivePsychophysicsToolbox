function [probEst,prmMAP,infoCrit,covEnt,moreOutput] = ...
    fun_BASS_MAP(xx,dat,dims,moreOpts)
% Run MAP inference and calculate BASS utility functions
%
% INPUTS
%        xx [array]  - each row is a stimulus at which to evaluate probs
%       dat [struct] - training data:
%                      .x = stimuli (each row is a stim),
%                      .y = output state (one from 0:(ydim-1))
%      dims [struct] - degrees of freedom of variables
%                      .y = of output
%                      .g = of the input feature vector g(x)
%  moreOpts [struct] - more options for inference
%                      .prsInit : initial values for parameters
%                      .prsLB/prsUB : range constraints for parameters
%                      .prior : prior hyperparameters
%                      .talkative : display option
%
% OUTPUTS
%              probEst - estimated probability at posterior mode (MAP)
%               prmMAP - MAP estimate of the parameters
%             infoCrit - expected information gain from next stimulus
%               covEnt - entropy of inferred posterior
%  moreOutput [struct] - fields {cov,negHess}
% -----------------------------------------------------------------------

% Copyright 2018 Ji Hyun Bak

%% unpack input

prs_init = moreOpts.prsInit; % initial values for the parameters

% prior hyperparameters input
myPrior = moreOpts.prior; 

% get range constraints
lb = moreOpts.prsLB;
ub = moreOpts.prsUB;

% whether to take partial hessian wrt weights
cutPartialCovLaplace = true; % for now always cut

% accuracy level for Sparse Grids
acclev = 4; % (fixed in this project)

% additional switches
reportMoreValues = moreOpts.reportMoreValues;
talkative = moreOpts.talkative; % display option
myshowopt = max(0,talkative-1); % set through talkative; one level down

%% run inference with MAP

% negative log posterior
neglogpdf = @(prs) negLogPost_MNLogLapse(prs,dat,dims,myPrior);

% run numerical optimization for MAP
[prmMAP,negHess] = ...
    runMAP_rangePrior(neglogpdf,prs_init(:),lb,ub,myshowopt); 

% evaluate probability at posterior mode
probEst = getProbAtPrs_MNLogLapse(prmMAP,dims,xx); 

% prepare partial Hessian (for now always cut to weights)
K0 = (dims.y)*(dims.g);
if(cutPartialCovLaplace) 
    Kcut = K0;
else
    Kcut = size(negHess,1);
end
negHcut = negHess(1:Kcut,1:Kcut); 

% posterior entropy (with Laplace approx.)
covEnt = -logdet(negHcut); 

% pack more information
moreOutput = [];
if(reportMoreValues)
    moreOutput.cov = inv(negHcut);
    % moreOutput.negHess = negHcut; % no need
end

%% compute infomax utility function

% calculate expected information gain 
[infoCrit,critWarn] = ...
    infoPost_MNLogLapse(xx,dims,prmMAP(1:Kcut),negHcut,acclev); 
if(critWarn>0)
    N = numel(dat.y); % with N observations
    disp(['- fun_BASS_MAP: N=',num2str(N),', critWarn=',num2str(critWarn)]); 
    if(critWarn>1)
        error('at least one bad case in infoPost; should stop and check.');
    end
end

end
