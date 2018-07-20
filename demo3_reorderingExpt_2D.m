% demo3_reorderingExpt_2D.m

% Demo script for running a dataset re-ordering analysis.
% Dataset is generated under a hidden model in this demo, 
% but can be easily replaced by a user dataset in matching format.

% Related to Fig 8 in Bak & Pillow 2018.

% Copyright (c) 2018 Ji Hyun Bak

%% initialize

clear all; % clear workspace
clc; % clear command window
setpaths; % add toolbox to path


%% Load a dataset to re-order

% === Generate with 2D stimulus & 4-alternatives responses

% Here we generate a dataset under a "hidden" true model;
% to test the program with another dataset, simply replace this chunk
% and provide your own {xdata, ydata, xx0} in matching formats.
% Golden rule: the i-th row in {xdata,ydata} corresponds to the i-th trial.

Ngen = 500; % We will generate 500 trials
[xdata,ydata,~,xx0,yy0] = auxFun_gendat_x2y4(Ngen); 
% xdata: the list of stimuli presented 
%        (each row is a stimulus vector presented in a trial)
% ydata: the list of response categories observed (each row is a trial)
%        xdata and ydata should match by the rows (trials).
% xx0:   the original stimulus space, defined by the experimenter.
%        again each row should be a (distinct) stimulus vector.
% yy0:   the response space (list of all available response categories).



% === Unpack the dataset ====================================

% Detect stimulus/response dimensions
ydim0 = numel(yy0)-1; % minus 1 for choice probability normalization
gdim0 = numel(gfun(xx0(1,:))); % assume that gfun is known
dims = struct('y',ydim0,'g',gdim0);

% Sort and index the stimuli in dataset
xdata_temp = [xdata;xx0]; % append full stimulus set xx0 to xdata for now,
                          % to take care of the case where some of the 
                          % stimuli in xx0 were not used in xdata
myuniqrows_flip = @(A) unique(fliplr(A),'rows'); % column orders are flipped
[xx_flip,~,idata_temp] = myuniqrows_flip(xdata_temp);
xx = fliplr(xx_flip); % so that first column increases first
idata = idata_temp(1:size(xdata,1)); % now cut away the appended rows
%    xx: the stimulus space;
%        should be equal to xx0 up to sorting.
% idata: indexing of xdata to the stimulus set xx,
%        such that xdata = xx(idata,:).

% Pack data in the input format
mydat = struct('x',xdata,'y',ydata,'i',idata,'xx',xx);



%% Set options for the algorithm

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

optBase = [];

% parameter initialization, bounds and step sizes
K0 = ydim*gdim;
prsInit = [(hyperprs.wgtmean)*ones(K0,1); ...
    (hyperprs.lpsInit)*ones(udim,1)]; % initial value for parameters
optBase.prs0 = prsInit(:)';
optBase.prsInit = prsInit(:)'; % duplicate for re-initialization
optBase.prsLB = [-Inf*ones(K0,1); (hyperprs.lpsLB)*ones(udim,1)]'; % lower bound
optBase.prsUB = [Inf*ones(K0,1); (hyperprs.lpsUB)*ones(udim,1)]'; % upper bound
optBase.steps = ones(1,numel(prsInit)); % initial step sizes

% numbers for MCMC samplings
optBase.nsamples = nsamples; % length of chain
optBase.nburn = 500; % # samples for "burn-in"
optBase.nburnInit = 500; % duplicate for re-initialization
optBase.nburnAdd = 50; % burn-in for additional runs

% more options
optBase.prior = hyperprs;
optBase.reportMoreValues = false;
optBase.talkative = 1; % display level



%% All-data fit, to get a "best" estimate

% Because we do not know the true PF, get a best estimate of the PF
% using the entire dataset (which is supposed to be large).

optAllFit = optBase; % reset options for all-data fit

fprintf('\nRunning inference with all %d trials...\n',Ngen);

if(doingMAP)
    [probTrue,theta,~,~,~] = fun_BASS_MAP(xx,mydat,dims,optAllFit);

elseif(doingMCMC)
    numFullRuns = 5; % iterate a few times to adjust step sizes
    for nfr = 1:numFullRuns
        [probTrue,theta,~,~,chainLmat,~] = ...
            fun_BASS_MCMC(xx,mydat,dims,optAllFit);
        % update sampling parameters
        optAllFit.prs0 = theta;
        optAllFit.steps = chainLmat;
    end
end


% "True" parameters (in this case, approximated by the best estimate 
% using all trials in dataset)
paramBest_AllDataFit = paramVec2Struct(theta,dims);
fprintf('\nBest estimate using all %d trials:\n',Ngen);
fprintf('--------------------------------------\n');
fprintf('        biases:      b =%4.1f %4.1f %4.1f\n', paramBest_AllDataFit.b);
fprintf('weights for x1:   a[1] =%4.1f %4.1f %4.1f\n', paramBest_AllDataFit.a(1,:));
fprintf('weights for x2:   a[2] =%4.1f %4.1f %4.1f\n', paramBest_AllDataFit.a(2,:));
if(withLapse)
    lapseBest = sum(getLapseProb(paramBest_AllDataFit.u)); % u is auxiliary lapse parameter
    fprintf('    lapse rate:  lapse =%5.2f\n\n', lapseBest);
end



% === Plot best PF regimes on stimulus space ===

% Response map color
mymap = [ 0 0.4470 0.7410 ;...
    0.8500 0.3250 0.0980 ;...
    0.9290 0.6940 0.1250 ;...
    0.4660 0.6740 0.1880];
mymap = (1-mymap)*0.5 + mymap;

% Determine which response dominates (has the highest probability) 
% given each stimulus value.
[~,mostLikelyResponse] = max(probTrue,[],1); % based on the true choice probability

xx1D = unique(xx(:,1)); % 1D grid, just for plotting (in this case stimulus grid is symmetric)

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
title('choice regimes, all-data estimate')


% === Plot best PF surfaces ===

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
title('PF, all-data estimate')
set(get(gca,'xlabel'),'rotation',25);
set(get(gca,'ylabel'),'rotation',-30);
set(gca,'LineWidth',1.5)


%% Choose some initial stimulus-response observations

% pick initial points
ninit = 10;
iinit = randsample(1:size(xdata,1),ninit,false); % choose random stimuli
xinit = xdata(iinit,:);
yinit = ydata(iinit);

% initialize dataset
seqdat = struct('x',xinit,'y',yinit,'i',iinit(:));

% plot initial stimuli
subplot(2,2,1)
hold on
for np = 1:ninit
    pointcolor = mymap(yinit(np)+1,:); % color indicates observed response
    plot(xinit(np,1),xinit(np,2),'ko','markerfacecolor',pointcolor) % initial stimuli
end


% === Posterior inference with initial data =======

optSeq = optBase; % reset options for sequencial experiment

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


% === Select next stimulus using infomax =========

% get indexing right: choose from remaining stimulus-response pairs
remains = find(~ismember(1:size(xdata,1),seqdat.i)); % index for xdata

% infoMax score (1 if max info gain; 0 otherwise)
myscore = double(infoCrit==max(infoCrit(idata(remains)))); % infoCrit matches xx
myscoreR = myscore(idata(remains)); % idata(remains): index for xx
idxR = randsample(numel(remains),1,true,myscoreR); % idxR: index for remains

% pull next trial
idxnext = remains(idxR);
xnext = xdata(idxnext,:);
ynext = ydata(idxnext);

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
    
    
    
    % === Select next stimulus using infomax =========
    
    % get indexing right: choose from remaining stimulus-response pairs
    remains = find(~ismember(1:size(xdata,1),seqdat.i)); % index for xdata
    
    % infoMax score (1 if max info gain; 0 otherwise)
    myscore = double(infoCrit==max(infoCrit(idata(remains)))); % infoCrit matches xx
    myscoreR = myscore(idata(remains)); % idata(remains): index for xx
    idxR = randsample(numel(remains),1,true,myscoreR); % idxR: index for remains
    
    % pull next trial
    idxnext = remains(idxR);
    xnext = xdata(idxnext,:);
    ynext = ydata(idxnext);
    
    
    
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
paramEst = paramVec2Struct(prmEst,dims); % get a param struct
fprintf('\nEstimated after %d trials:\n',N);
fprintf('--------------------------------------\n');
fprintf('        biases:      b =%4.1f %4.1f %4.1f\n', paramEst.b);
fprintf('weights for x1:   a[1] =%4.1f %4.1f %4.1f\n', paramEst.a(1,:));
fprintf('weights for x2:   a[2] =%4.1f %4.1f %4.1f\n', paramEst.a(2,:));
if(withLapse)
    lapseEst = sum(getLapseProb(paramEst.u)); % u is auxiliary lapse parameter
    fprintf('    lapse rate:  lapse =%5.2f\n\n', lapseEst);
end

