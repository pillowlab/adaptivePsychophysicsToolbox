% demo2_AdaptiveStimSelect_2D.m
%
% Demo script for running a simulated experiment with adaptive (infomax)
% stimulus sampling, with 2-dimensional stimulus + 4-alternative responses:
% a trial-by-trial example.

% Related to the example shown in Figs 5-6 in Bak & Pillow 2018.

% Copyright (c) 2018 Ji Hyun Bak

%% initialize

clear all; % clear workspace
clc; % clear command window
setpaths; % add toolbox to path


%% Generate true psychometric model

% === Set up stimulus/response spaces ===============

% Set up stimulus space (2D stimulus)
xgrid1D = (-1:0.2:1)';
xx1D = xgrid1D/std(xgrid1D); % scaled to have unit std
xx = combvec(xx1D',xx1D')'; % list of points on 2-dimensional regular grid

% Set up response space (4-alternative)
yvalslist = [0 1 2 3];


% === Set true model ================================

% True model parameters
truebias = [1 0 1]; % bias associated to responses 1,2,3
trueslope = [3 6 3; -3 0 3]; % slopes associated to responses 1,2,3
truelapse = 0; % total lapse rate, from 0 to 1 (lambda)

truemodel = struct('b',truebias,'a',trueslope,'lapse',truelapse); % pack into a struct

fprintf('\nGenerating true model:\n');
fprintf('--------------------------------------\n');
fprintf('        biases:      b =%4.1f %4.1f %4.1f\n', truemodel.b);
fprintf('weights for x1:   a[1] =%4.1f %4.1f %4.1f\n', truemodel.a(1,:));
fprintf('weights for x2:   a[2] =%4.1f %4.1f %4.1f\n', truemodel.a(2,:));
fprintf('    lapse rate:  lapse =%5.2f\n\n', truemodel.lapse);

% Get true psychometric function
[theta,dims,probTrue] = getTrueModel_byParamStruct(truemodel,xx);


% === Plot true PF regimes on stimulus space ===

% Response map color
mymap = [ 0 0.4470 0.7410 ;...
    0.8500 0.3250 0.0980 ;...
    0.9290 0.6940 0.1250 ;...
    0.4660 0.6740 0.1880];
mymap = (1-mymap)*0.5 + mymap;

% Determine which response dominates (has the highest probability) 
% given each stimulus value.
[~,mostLikelyResponse] = max(probTrue,[],1); % based on the true choice probability

clf; subplot(2,2,1)
imagesc(xx1D,xx1D,reshape(mostLikelyResponse,numel(xx1D)*[1 1])')
set(gca,'YDir','normal')
colormap(mymap)
hold on
plot(xx(:,1),xx(:,2),'k.') % stimulus space
hold off
axis square
xlabel('stim 1')
ylabel('stim 2')
title('true choice regimes')


% === Plot true PF surfaces ===

subplot(2,2,2)
surf(xx1D,xx1D,reshape(probTrue(1,:),numel(xx1D)*[1 1])','FaceColor',mymap(1,:))
hold on
surf(xx1D,xx1D,reshape(probTrue(2,:),numel(xx1D)*[1 1])','FaceColor',mymap(2,:))
surf(xx1D,xx1D,reshape(probTrue(3,:),numel(xx1D)*[1 1])','FaceColor',mymap(3,:))
surf(xx1D,xx1D,reshape(probTrue(4,:),numel(xx1D)*[1 1])','FaceColor',mymap(4,:))
hold off
axis tight
axis square
zlim([0 1])
xlabel('stim 1')
ylabel('stim 2') 
zlabel('P(choice)')
title('true PF')
set(get(gca,'xlabel'),'rotation',25);
set(get(gca,'ylabel'),'rotation',-30);
set(gca,'LineWidth',1.5)


%% Generate some initial stimulus-response observations

% function for drawing responses
drawResp_yfixed = @(choiceProb,inds) ...
    mnrnd(1,choiceProb(:,inds)')*yvalslist(:); % draw fresh single y from pTrue

% pick initial points
ninit = 10; % # initial trials (not adaptively sampled) 
iinit = randsample(1:size(xx,1),ninit,false); % choose random stimuli
xinit = xx(iinit,:);
yinit = drawResp_yfixed(probTrue,iinit); % draw responses

% initialize dataset
seqdat = struct('x',xinit,'y',yinit,'i',iinit(:));

% plot initial stimuli
subplot(2,2,1)
hold on
for np = 1:ninit
    pointcolor = mymap(yinit(np)+1,:); % color indicates observed response
    plot(xinit(np,1),xinit(np,2),'ko','markerfacecolor',pointcolor) % initial stimuli
end


%% Infer PF from initial data

% === Specify algorithms to use =====================

% Select inference method (either MAP or MCMC)
nsamples = 500; % MCMC chain length
doingMAP = (nsamples==0); % if 0, we make MAP estimate;
doingMCMC = (nsamples>0); % if >0, we run MCMC sampling.

% Set prior hyperpameters
hyperprs = ... 
    struct('wgtmean',0,'wgtsigma',3,... % gaussian prior for weights
    'lpsLB',log(0.001),'lpsUB',0, ... % range constraints for lapses
    'lpsInit',-5 ... % starting point for lapse parameters
    );

% Set whether to include lapses in the model being inferred
withLapse = 0; % 1: use lapse-aware model, 0: ignore lapse

% Unpack parameter dimensions
ydim = dims.y;
gdim = dims.g;
if(withLapse==1)
    udim = ydim+1; % lapse parameter length (equal to # choices)
else
    udim = 0; % no lapse parameter
end

% a human-readable string to remind what we do in this simulation
inftagList = {'Laplace',['MCMC',num2str(nsamples)]};
inftag = inftagList{doingMCMC+1}; % if not MCMC, then MAP
algtag = 'infomax';
withLapseTagList = {'lapse-unaware','lapse-aware'};
lpstag = withLapseTagList{withLapse+1};
runTag = [algtag,'-',inftag];



% === Pack options for sequential experiment =======

optSeq = [];

% parameter initialization, bounds and step sizes
K0 = ydim*gdim;
prsInit = [(hyperprs.wgtmean)*ones(K0,1); ...
    (hyperprs.lpsInit)*ones(udim,1)]; % initial value for parameters
optSeq.prs0 = prsInit(:)';
optSeq.prsInit = prsInit(:)'; % duplicate for re-initialization
optSeq.prsLB = [-Inf*ones(K0,1); (hyperprs.lpsLB)*ones(udim,1)]'; % lower bound
optSeq.prsUB = [Inf*ones(K0,1); (hyperprs.lpsUB)*ones(udim,1)]'; % upper bound
optSeq.steps = ones(1,numel(prsInit)); % initial step sizes

% numbers for MCMC samplings
optSeq.nsamples = nsamples; % length of chain
optSeq.nburn = 500; % # samples for "burn-in"
optSeq.nburnInit = 500; % duplicate for re-initialization
optSeq.nburnAdd = 50; % burn-in for additional runs

% more options
optSeq.prior = hyperprs;
optSeq.reportMoreValues = false;
optSeq.talkative = 1; % display level


% === Posterior inference with initial data =======

if(doingMAP)
    % MAP estimate
    [probEst,prmEst,infoCrit,covEntropy,~] = ...
        fun_BASS_MAP(xx,seqdat,dims,optSeq);
elseif(doingMCMC)
    % MCMC sampling
    [probEst,prmEst,infoCrit,covEntropy,chainLmat,~] = ...
        fun_BASS_MCMC(xx,seqdat,dims,optSeq);
    % adjust next sampling parameters
    optSeq.prs0 = prmEst;
    optSeq.steps = chainLmat;
    optSeq.nburn = optSeq.nburnAdd; % shorter burn-in for non-initial trials
    % track sampler properties
    chainstd = diag(chainLmat)'; % store diagonal part (std) only
end

% select next stimulus using infomax
[~,idxnext] = max(infoCrit); % find info-max stimulus
xnext = xx(idxnext,:);
ynext = drawResp_yfixed(probTrue,idxnext);

% detect choice regimes / decision boundaries
[~,mostLikelyResponse] = max(probEst,[],1); % from estimated PF


% plot estimated PF regimes
subplot(2,2,2)
imagesc(xx1D,xx1D,reshape(mostLikelyResponse,numel(xx1D)*[1 1])')
set(gca,'YDir','normal')
colormap(mymap)
hold on
plot(xx(:,1),xx(:,2),'k.') % stimulus space
hold off
axis square
xlabel('stim 1')
ylabel('stim 2')
title('estimated choice regimes')

% plot expected information gain
subplot(2,2,3)
imagesc(xx1D,xx1D,reshape(infoCrit,numel(xx1D)*[1 1])')
set(gca,'YDir','normal')
axis square
colormap(gca,'gray')
hold on
plot(xnext(1),xnext(2),'r*')
hold off
xlabel('stim 1')
ylabel('stim 2')
title('expected info gain')


%% Adaptively sample and add one stimulus at a time

N = 50; % total # trials in the experiment

% track performance measures
MSE = NaN(N,1); % mean-square error
postEnt = NaN(N,1); % (approximate) posterior covariance entropy

% fill in result with initial data
MSE(ninit) = mean(sum((probTrue'-probEst').^2,2),1); 
postEnt(ninit) = covEntropy; 


% sequential experiment 

for jj=(ninit+1):N
    
    disp(['Trial #',num2str(jj)]);
    
    % add to dataset
    seqdat.x(end+1,:) = xnext;
    seqdat.y(end+1) = ynext;
    seqdat.i(end+1) = idxnext;
    
    % posterior inference
    if(doingMAP)
        % MAP estimate
        [probEst,prmEst,infoCrit,covEntropy,~] = ...
            fun_BASS_MAP(xx,seqdat,dims,optSeq);
    elseif(doingMCMC)
        % MCMC sampling
        [probEst,prmEst,infoCrit,covEntropy,chainLmat,~] = ...
            fun_BASS_MCMC(xx,seqdat,dims,optSeq);
        % adjust next sampling parameters
        optSeq.prs0 = prmEst;
        optSeq.steps = chainLmat;
        optSeq.nburn = optSeq.nburnAdd; % shorter burn-in for non-initial trials
        % track sampler properties
        chainstd = diag(chainLmat)'; % store diagonal part (std) only
    end
    
    % detect choice regimes / decision boundaries
    [~,mostLikelyResponse] = max(probEst,[],1); % from estimated PF

    % performance measure
    MSE(jj) = mean(sum((probTrue'-probEst').^2,2),1); % mean-square error
    postEnt(jj) = covEntropy; % (approximate) covariance entropy
    
    % select next stimulus using infomax
    [~,idxnext] = max(infoCrit); % find info-max stimulus
    xnext = xx(idxnext,:);
    ynext = drawResp_yfixed(probTrue,idxnext);
    
    
    % === Plot ========================================
    
    % plot selected stimuli
    subplot(2,2,1)
    hold on
    plot(seqdat.x(1:jj-1,1),seqdat.x(1:jj-1,2),'ko','markerfacecolor','k') % previous trials
    pointcolor = mymap(seqdat.y(jj)+1,:); % color indicates observed response
    plot(seqdat.x(jj,1),seqdat.x(jj,2),'ko','markerfacecolor',pointcolor) % current trial
    
    % plot estimated PF regimes
    subplot(2,2,2)
    imagesc(xx1D,xx1D,reshape(mostLikelyResponse,numel(xx1D)*[1 1])')
    set(gca,'YDir','normal')
    colormap(mymap)
    hold on
    plot(xx(:,1),xx(:,2),'k.') % stimulus space
    hold off
    axis square
    xlabel('stim 1')
    ylabel('stim 2')
    title('estimated choice regimes')
    
    % plot expected information gain
    subplot(2,2,3)
    imagesc(xx1D,xx1D,reshape(infoCrit,numel(xx1D)*[1 1])')
    set(gca,'YDir','normal')
    axis square
    colormap(gca,'gray')
    hold on
    plot(xnext(1),xnext(2),'r*')
    hold off
    xlabel('stim 1')
    ylabel('stim 2')
    title('expected info gain')
    
    subplot(4,2,6)
    plot(ninit:jj,postEnt(ninit:jj),'k.-')
    xlim([ninit N])
    ylabel('post. ent.')
    
    subplot(4,2,8)
    plot(ninit:jj,MSE(ninit:jj),'k.-')
    xlim([ninit N])
    ylabel('error')
    xlabel('# trials')
    
    drawnow;
    
    % =================================================
    
end

% Final estimate after the last trial in sequence
paramEst = paramVec2Struct(prmEst,dims); % final parameter struct
fprintf('\nEstimated after %d trials:\n',N);
fprintf('--------------------------------------\n');
fprintf('        biases:      b =%4.1f %4.1f %4.1f\n', paramEst.b);
fprintf('weights for x1:   a[1] =%4.1f %4.1f %4.1f\n', paramEst.a(1,:));
fprintf('weights for x2:   a[2] =%4.1f %4.1f %4.1f\n', paramEst.a(2,:));
if(withLapse)
    lapseEst = sum(getLapseProb(paramEst.u)); % u is auxiliary lapse parameter
    fprintf('    lapse rate:  lapse =%5.2f\n\n', lapseEst);
end


