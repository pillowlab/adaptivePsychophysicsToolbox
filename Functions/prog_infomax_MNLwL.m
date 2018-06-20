function [snapshotN,trackSeq,settings] ...
    = prog_infomax_MNLwL(fulldat,mynums,myopts)
% Master program for adaptivePsychophysicsToolbox, with various options.
% Supports psychometric function inference and adaptive stimulus selection
% with infomax (or alternatively uniform) algorithm, using multinomial
% logistic model with lapses.
%
% Some example uses:
% #1. Fit a dataset to infer the psychometric function 
%     (set myopts.allfit==1; also see demo1_inferPFs_1D)
% #2. Run a simulated experiment with adaptive stimulus selection
%     (set myopts.freshData==1 and provide atrue model; 
%      also see demo2_AdaptiveStimSelect_2D)
% #3. Obtain an optimal re-ordeirng of an existing dataset
%     (set myopts.freshData==0; also see demo3_reorderingExpt_2D)
% 
% INPUTS
% -------
% fulldat [struct]  .xx: stimulus space (required field)
%                   .yy: response space
%                   .x : list of stimuli in data (each row is a trial)
%                   .y : list of response categories (matching rows with .x)
%  mynums [struct]  .nsamples : number of MCMC samples (if 0, MAP estimate)
%                   .N : number of trials in sequential experiment
%                   .ninit : number of initial trials
%                   (and other optional fields)
%  myopts [struct]  .allfit: switch for all-data fitting mode
%                   .withLapse: switch for lapse-aware algorithm
%                   .doingAL: switch for adaptive stimulus selection
%                   .freshData: for simulating responses vs data re-ordering
%                   .truemodel: parameters of the true model, if applicable
%                   (and other optional fields)
% 
% OUTPUTS
% -------
% snapshotN [struct] - summarizes final estimate after the last trial
%                      fields: param, prob, probTrue, ...
%  trackSeq [struct] - a trial-by-trial history in a sequential experiment
%                      fields: x,y,prs,MSE,entropy,time,std, ...
%  settings [struct] - summarizes run settings (also copies input options)
%                      fields: runtag,trueParamVec,dims,...
% ------------------------------------------------------------------------

% Copyright (c) 2018 Ji Hyun Bak

%% Unpack input 

% --- required input ---

% inference algorithm
nsamples = mynums.nsamples; % number of MCMC samples
withLapse = myopts.withLapse; % whether algorithm is aware of lapse (0/1)

allfit = getFromStruct(myopts,'allfit',0); % 1 means do all fit only and return
if(~allfit)
    % if doing sequantial simulation
    N = mynums.N; % total #trials (observations) in sequence
    ninit = mynums.ninit; % #initial trials before algorithm starts working
    doingAL = myopts.doingAL; %  whether stimulus selection is adaptive (0/1)
    freshData = myopts.freshData; % 1:draw, 0: data re-sampling, -1:fixed data sequence
end

% --- optional inputs ---

truemodel = getFromStruct(myopts,'truemodel',[]); % true parameter struct
talkative = getFromStruct(myopts,'talkative',false); % default is to be silent
pickby = getFromStruct(myopts,'pickby',2); % 1: random, 2: regular (for initial stimuli)


%%% check for obvious option clashes

haveData = and(isfield(fulldat,'x'),isfield(fulldat,'y'));
haveModel = ~isempty(truemodel);

if(~haveData && ~haveModel)
    error('simfun: should provide either full data or model.')
end
if(~haveData && freshData<=0)
    error('simfun: data sampling requested, but data not provided.');
end
if(allfit>0)
    if(~haveData)
        error('simfun: allfit mode requested, but data not provided.');
    end
    haveModel = false; % if doing allfit, ignore the model
end


%%% algorithm tag

doingMAP = (nsamples==0);
doingMCMC = (nsamples>0);

algtagList = {'uniform','infomax'};
inftagList = {'Laplace',['MCMC',num2str(nsamples)]};
inftag = inftagList{doingMCMC+1}; % if not MCMC, then MAP
if(~allfit)
    algtag = algtagList{doingAL+1};
else
    algtag = 'AllDataFit';
end
runtag = [algtag,'-',inftag];

if(talkative)
    disp(runtag);
end



%% Unpack model, and/or do all-fit estimate on dataset


%%% Unpack input data

% fulldat can have fields {x,y,xx,yy}; xx required; x,y,yy optional

% stimulus space - each row is a (possibly multlidimensional) stimulus
xx = fulldat.xx; % xx is a required field

if(haveData)
    xall = fulldat.x; % note: xall usually contains many duplicates
    yall = fulldat.y;
    % trial number check
    if(size(fulldat.x,1)~=size(fulldat.y,1))
        error('simfun_infomax: stimulus-response trial number mismatch');
    end
    if(talkative)
        disp('- Data detected.')
    end
    % set up datapoint index
    [xuniq,~,iall0] = unique([xall; xx],'rows');
    iall = iall0(1:size(xall,1)); % indexing from xall to xx
    if(~isequal(xuniq,xx))
        error('simfun: stimulus space mismatch');
    end
end


%%% unpack model dimensions

if(haveModel) 
    % -- provided as true parameter struct
    [theta,dims,probTrue] = getTrueModel_byParamStruct(truemodel,xx);
elseif(haveData)
    % if truemodel is not supplied, specify dimensions from data (ad hoc),
    ydim0 = numel(unique(yall))-1; % possible issue if some responses are not sampled in data
    gdim0 = numel(gfun(xx(1,:))); % this is always correct
    dims = struct('y',ydim0,'g',gdim0);
    % in this case, all-fit best model will be obtained below.
end


%%% Set parameter range/initialization given the dims

% -- hard-coded numbers passed from upstream
hyperprs = mynums.hyperprs;
sampnums = struct('nsamples',nsamples);
sampnums.nburnInit = getFromStruct(mynums,'nburnInit',500);
sampnums.nburnAdd = getFromStruct(mynums,'nburnAdd',50);

[prsInit,myOptBase] = initializeParams_simple(dims,hyperprs,sampnums,myopts);
myOptBase.talkative = talkative; % also pass display options along

reportMoreValues = myOptBase.reportMoreValues;


%% Best possible probability estimates ("allfit") if necessary

optAllFit = myOptBase; % copy options for this chunk

if(haveData && ~haveModel) 
    % allfit==1 always comes down to this combination
    % but this block can be run even when allfit was not explicitly on
    
    %%% All-data fit
    
    trackSeqMore_allfit = []; % initialize empty struct

    if(doingMAP) 
        [probTrue,theta,~,~,moreOutput] = ...
            fun_BASS_MAP(xx,fulldat,dims,optAllFit);
        % store optional output
        if(reportMoreValues) 
            trackSeqMore_allfit.cov = moreOutput.cov;
        end
        
    elseif(doingMCMC)
        numFullRuns = 5; % run multiple times to adjust step sizes etc.
        for nfr = 1:numFullRuns
            disp(['  warming up ',num2str(nfr),'/',num2str(numFullRuns),' ...']);
            [probTrue,theta,~,~,chainLmat,moreOutput] = ...
                fun_BASS_MCMC(xx,fulldat,dims,optAllFit);
            
            % update sampling parameters
            optAllFit.prs0 = theta;
            optAllFit.steps = chainLmat;
            % store optional output
            if(reportMoreValues)
                trackSeqMore_allfit.cov = moreOutput.cov;
                trackSeqMore_allfit.psamps = moreOutput.psamps;
            end
        end
    end
end

%%% Pack all settings for output

settings = struct(...
    'doingMCMC',doingMCMC,'doingMAP',doingMAP,...
    'hyperprs',hyperprs,... 
    'withLapse',withLapse,...
    'truemodel',truemodel,'trueParamVec',theta(:)','dims',dims);
if(~allfit)
    settings.N = N;
    settings.ninit = ninit;
    settings.doingAL = doingAL;
    settings.freshData = freshData;
    settings.pickby = pickby;
end
if(doingMCMC>0)
    settings.nsamples = nsamples;
    settings.nburnInit = myOptBase.nburnInit;
    settings.nburnAdd = myOptBase.nburnAdd;
end
settings.runtag = runtag; % a human-readable tag for the running mode

% copy any missing input options
optfs = fieldnames(myopts);
for f = 1:numel(optfs)
    myfield = optfs{f}; % field name in myopts
    if(~isfield(settings,myfield))
        settings.(myfield) = myopts.(myfield); % copy field into settings if not already included
    end
end 


if(allfit) % return with all-fit data
    
    % note: even if we performed "allfit" above,
    % if allfit switch was not explicitly on, program does not return here.
    
    % pack output struct
    trackSeq = trackSeqMore_allfit; % already filled if "reportMoreValues" is on
    trackSeq.prs = theta(:)';

    % -- note: when freshdata==false (data resampling), 
    % -- probTrue may be a struct variable rather than a numeric array
    snapshotN = struct('xx',xx,'probTrue',probTrue,'prob',probTrue);
    snapshotN.param = paramVec2Struct(theta,dims);
    
    if(talkative)
        disp('- Returning all-data fit result.');
    end
    return;
    
end


%% Simulate sequential experiment

if(talkative)
    if(freshData==1)
        disp('Drawing from model.');
    elseif(freshData==0)
        disp('Sampling from data.');
    elseif(freshData==-1)
        disp('Fixed data sequence.');
    else
        error('simfun: unknown freshData value');
    end
end

%%% Set up small functions

% for drawing responses

% response space
ydim = dims.y;
ylist = getFromStruct(fulldat,'yy',(0:ydim)'); % list of possible responses

drawResp_yfixed = @(pTrue,inds) ...
    mnrnd(1,pTrue(:,inds)')*ylist(:); % draw fresh single y from pTrue

% for error estimates
getMSE = @(pTrue,pEst) sum((pTrue'-pEst').^2,2); % mean-square error



%%% initialize learning sequence trackers

trackPrs = zeros(N,numel(prsInit));
trackMSE = zeros(N,1);
trackEntropy = zeros(N,1); % posterior entropy
if(doingMCMC)
    trackStd = zeros(N,numel(prsInit));
end
trackTime = zeros(N,2); % computation times, by chunks [inference stimSelection]

if(reportMoreValues)
    trackProb_cell = cell(N,1);
    trackInfo_cell = cell(N,1);
    trackErrX_cell = cell(N,1);
    trackCov_cell = cell(N,1);
end


%%% pick initial points

if(talkative)
    disp(' ');
    disp(['Choosing ',num2str(ninit),' initial data...']);
end

ixx = getIinit(ninit,xx,pickby);
if(freshData==1)
    % draw fresh responses (simulation)
    iinit = ixx; 
    xinit = xx(iinit,:);
    yinit = drawResp_yfixed(probTrue,iinit);
else
    % sampling from using input (xall,yall) pairs
    if(freshData==0)
        % dataset re-ordering: first pick x value, then index i
        iinit = NaN(ninit,1);
        for inum = 1:ninit
            xtemp = xx(ixx(inum),:);
            alli_atx = find(all(bsxfun(@eq,xall,xtemp),2));
            if(~isempty(alli_atx))
                iinit(inum) = randsample(alli_atx,1); % randomly choose one
            else
                % this may happen when xall did not include all points
                % in the stimulus space xx. just sample a random pair..
                iinit(inum) = randsample(ninit,1);
            end
        end
    elseif(freshData==-1)
        % fixed sequence as provided in input data
        iinit = (1:ninit)';
    end
    xinit = xall(iinit,:);
    yinit = yall(iinit);
end

% initialize data sequence
seqdat = struct('x',xinit,'y',yinit,'i',iinit(:));


%%% add one stimulus at a time and track the error

optSeq = myOptBase; % copy options for this sequencial experiment chunk

for jj=(ninit+1):(N+1) % 1 extra step for final inference
    
    if(talkative)
        disp(['Trial #',num2str(jj-1)]);
    end
    
    t1start = tic; % computation time (inference part)
    
    if(doingMAP)
        
        % MAP estimate
        [probEst,prmEst,infoCrit,entldcov,moreOutput] = ...
            fun_BASS_MAP(xx,seqdat,dims,optSeq);
        
    elseif(doingMCMC)
        
        [probEst,prmEst,infoCrit,entldcov,chainLmat,moreOutput] = ...
            fun_BASS_MCMC(xx,seqdat,dims,optSeq);
        
        % --- MCMC-specific ---
        
        % Adjust next sampling parameters
        optSeq.prs0 = prmEst;
        optSeq.steps = chainLmat;
        optSeq.nburn = myOptBase.nburnAdd; % for non-initial trials
        
        % track sampler properties
        chainstd = diag(chainLmat)'; % store diagonal part (std) only
        trackStd(jj-1,:) = chainstd;
        
    end
    
    t1end = toc(t1start); % computation time (inference part)
    
    % error estimate
    postErr = getMSE(probTrue,probEst);
    trackMSE(jj-1,:) = mean(postErr,1);
    
    % track simulation quantities
    trackPrs(jj-1,:) = prmEst;
    trackEntropy(jj-1,:) = entldcov;
    trackTime(jj-1,1) = t1end;
    
    % track more quantities
    if(reportMoreValues)
        trackProb_cell{jj-1} = probEst;
        trackErrX_cell{jj-1} = postErr;
        trackCov_cell{jj-1} = moreOutput.cov;
        trackInfo_cell{jj-1} = infoCrit; % expected utility of stimuli
    end
    
    % ----------------------------
    if(numel(seqdat.i)>=N)
        break; % stop at N
    end
    
    % Select next stimulus using infomax & Add to dataset
    
    t2start = tic; % computation time (stimulus selection part)
    
    if(freshData==1)
        myscore = getUtility_infoMax(xx,infoCrit,doingAL); % expected utility
        idxnext = randsample(size(xx,1),1,true,myscore); % use weighted sampling
        xnext = xx(idxnext,:);
        ynext = drawResp_yfixed(probTrue,idxnext);
    else
        if(freshData==0)
            remains = find(~ismember(1:size(xall,1),seqdat.i));
            % - remains: indices for xall
            % - iall(remains): indices for xx
            myscore = getUtility_infoMax(xx,infoCrit,doingAL,unique(iall(remains)));
            idxR = randsample(numel(remains),1,true,myscore(iall(remains)));
            % - idxR is the index for remains (subset of xall indices)
            idxnext = remains(idxR);
        elseif(freshData==-1)
            idxnext = max(seqdat.i)+1; % fixed data sequence
        else
            error('simfun: unknown freshData (stimulus selection)')
        end
        xnext = xall(idxnext,:);
        ynext = yall(idxnext);
    end
    
    seqdat.x(end+1,:) = xnext;
    seqdat.y(end+1) = ynext;
    seqdat.i(end+1) = idxnext; 
    
    % track computation time (stimulus selection part)
    t2end = toc(t2start);
    trackTime(jj-1,2) = t2end;
    
end

%%% Store final estimates (figure ingredients)
snapshotN.param = paramVec2Struct(trackPrs(end,:),dims); % final parameter struct
snapshotN.prob = probEst;
snapshotN.probTrue = probTrue;
snapshotN.info = infoCrit;
snapshotN.err = postErr;
snapshotN.xx = xx;
if(doingMCMC && reportMoreValues)
    snapshotN.psamps = moreOutput.psamps;
end

%%% Store learning history
trackSeq.x = (seqdat.x);
trackSeq.y = (seqdat.y);
trackSeq.prs = trackPrs;
trackSeq.MSE = trackMSE;
trackSeq.entropy = trackEntropy;
trackSeq.time = sum(trackTime,2);
if(doingMCMC)
    trackSeq.std = trackStd;
end
if(reportMoreValues) 
    trackSeq.probEst = {trackProb_cell};
    trackSeq.infoCrit = {trackInfo_cell};
    trackSeq.errx = {trackErrX_cell};
    trackSeq.cov = {trackCov_cell};
end

if(talkative)
    disp(' ');
end


end

% ------------------------------------------------------------------------

function ilist = getIinit(npick,myxx,pickby)
% pick npick points from stimulus space xx, either randomly or regularly

if(pickby==1)
    % random choice
    ilist = randsample(1:size(myxx,1),npick,false); 
    
elseif(pickby==2)
    % regular intervals
    
    getIinit_reg1D = @(nArg,xArg) 1+floor((size(xArg,1)-1)/(nArg-1)*(0:nArg-1)'); % regular
    
    if(size(myxx,2)==1)
        % 1D
        ilist = getIinit_reg1D(npick,myxx); % 1D xx
    elseif(size(myxx,2)==2)
        % 2D grid 
        mm = median(unique(myxx(:)));
        iaxis = any(myxx==mm,2); % axes points
        xu1 = unique(myxx(iaxis,1));
        xu2 = unique(myxx(iaxis,2));
        xpick1 = xu1(getIinit_reg1D(floor(npick/2),xu1));
        xpick2 = xu2(getIinit_reg1D(floor(npick/2),xu2));
        ipick1 = find(ismember(myxx,[xpick1 mm*ones(numel(xpick1),1)],'rows'));
        ipick2 = find(ismember(myxx,[mm*ones(numel(xpick2),1) xpick2],'rows'));
        if(npick > 2*floor(npick/2)) % if odd npick
            ipick0 = find(ismember(myxx,mm*[1 1],'rows'));
        else
            ipick0 = []; % empty array
        end
        ilist = [ipick0; ipick1; ipick2];
    else
        error('getIinit: higher-dim xx not yet supported.');
    end
 
end

end

function [prsInit,myOptPlus] = initializeParams_simple(dims,hyperprs,sampnums,myopts)

% unpack dimensions
ydim = dims.y;
gdim = dims.g;
K0 = ydim*gdim;
udim = ydim+1; % lapse parameter length

% numbers for MCMC samplings
nsamples = sampnums.nsamples; % length of chain
nburnInit = sampnums.nburnInit; % # samples to burn in initially
nburnAdd = sampnums.nburnAdd; % in additional runs

% parameter initialization based on hyperprs
w0 = hyperprs.wgtmean; 
u0 = hyperprs.lpsInit; 
lpsLB = hyperprs.lpsLB;
lpsUB = hyperprs.lpsUB;
if(~and(u0>=lpsLB,u0<=lpsUB))
    % lapse range check
    warning('initializeParams: lapse range error. reset to mid-range');
    u0 = mean([lpsLB lpsUB]);
end

% unpack lapse option for algorithm & set prsInit,prsLB,prsUB
% (all downstream functions recognize lapse option based on #parameters)
withLapse = myopts.withLapse;
if(withLapse==1)
    prsInit = [w0*ones(K0,1); u0*ones(udim,1)]; % initial value for parameters
    prsLB = [-Inf*ones(K0,1); lpsLB*ones(udim,1)]; % lower bound
    prsUB = [Inf*ones(K0,1); lpsUB*ones(udim,1)]; % upper bound
else
    prsInit = w0*ones(K0,1);
    prsLB = -Inf*ones(K0,1);
    prsUB = Inf*ones(K0,1);
end

% initialization for MCMC sampling
stepsInit = ones(1,numel(prsInit)); 

% more options
reportMoreValues = getFromStruct(myopts,'reportMoreValues',false);

% wrap in a single struct (base option for all inference algorithms)
myOptPlus = struct('prs0',prsInit(:)',...
    'prsLB',prsLB(:)','prsUB',prsUB(:)',... % pass LB/UB as full vectors
    'steps',stepsInit,...
    'prior',hyperprs,... 
    'reportMoreValues',reportMoreValues,... 
    'nsamples',nsamples,'nburn',nburnInit,...
    'prsInit',prsInit(:)','nburnInit',nburnInit,'nburnAdd',nburnAdd);
% last three fields {prsInit, nburnInit, nburnAdd}: for resetting purposes

end

function myscore = getUtility_infoMax(xx,infoCrit,doingAL,varargin)
% get utility for the candidate stimuli

if(~doingAL)
    myscore = ones(size(xx,1),1); % uniform score (random sampling)
    return;
end

if(nargin>3)
    irem = varargin{1}; % for data-resampling, make sure remaining stimuli are selected
    info_original = infoCrit; % copy
    infoCrit = NaN(size(infoCrit)); % empty template
    infoCrit(irem) = info_original(irem); % only xx's with remaining datapoints survive
end

myscore = double(infoCrit==max(infoCrit)); % for "hard" InfoMax (deterministic)

if(all(or(isnan(myscore),myscore<eps))) % if all zero or NaN
    myscore = ones(size(xx,1),1); % should have at least one positive
end

end

