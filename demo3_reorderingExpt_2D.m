% demo3_reorderingExpt_2D.m

% Demo script for illustrating a dataset re-ordering analysis.
% In this case the dataset is generated under a hidden model
% with 2-dimensional stimulus + 4-alternatives responses,
% but it can be replaced with any other dataset of matching format.

% Reproduces a single run for Fig 8 in Bak & Pillow 2018.

% Copyright (c) 2018 Ji Hyun Bak

%% initialize

clear all; % clear workspace
clc; % clear command window
setpaths; % add toolbox to path


%% Load (or generate) a dataset

% Here we generate a dataset under a "hidden" true model;
% to test the program with another dataset, simply replace this chunk
% and provide your own {xdata, ydata, xx0} in matching formats.
% Golden rule: the i-th row in {xdata,ydata} corresponds to the i-th trial.

% === Generate with 2D stimulus & 4-alternatives responses

Ngen = 500; % We will generate 500 trials
[xdata,ydata,~,xx0,yy0] = auxFun_gendat_x2y4(Ngen); 
% xdata: the list of stimuli presented 
%        (each row is a stimulus vector presented in a trial)
% ydata: the list of response categories observed (each row is a trial)
%        xdata and ydata should match by the rows (trials).
% xx0:   the original stimulus space, defined by the experimenter.
%        again each row should be a (distinct) stimulus vector.
% yy0:   the response space (list of all available response categories).


%% Prepare for dataset re-ordering

% === Prepare the dataset ==========================

% Sort and index the stimuli presented in the dataset
xdata_temp = [xdata;xx0]; % append full stimulus set xx0 to xdata for now,
                          % to take care of the case where some of the 
                          % stimuli in xx0 were not used in xdata

[xx,~,idata_temp] = unique(xdata_temp,'rows');

if(~isequal(unique(xx0,'rows'),xx))
    error('Something is wrong: xdata has unknown stimuli.');
end

idata = idata_temp(1:size(xdata,1)); % now cut away the appended rows
%    xx: the stimulus space;
%        should be equal to xx0 up to sorting.
% idata: indexing of xdata to the stimulus set xx,
%        such that xdata = xx(idata,:).

if(~isequal(xdata,xx(idata,:)))
    error('Indexing error.');
end

% Pack data in the input format
mydat = struct('x',xdata,'y',ydata,'i',idata,'xx',xx);



% === Specify algorithms to use =====================

% Initialize input options for the program
mynums = [];
myopts = [];

% Select inference method (either MAP or MCMC)
mynums.nsamples = 500; % MCMC chain length (if 0, MAP; if >0, MCMC)

% Set prior hyperpameters
mynums.hyperprs = ... 
    struct('wgtmean',0,'wgtsigma',3,... % gaussian prior for weights
    'lpsLB',log(0.001),'lpsUB',0, ... % range constraints for lapses
    'lpsInit',-5 ... % starting point for lapse parameters
    );

% Set whether to include lapses in the model being inferred
myopts.withLapse = 1; % 1: use lapse-aware model, 0: ignore lapse

% Set switch for stimulus selection algorithm
myopts.doingAL = 1; % 1: adaptive stim sampling, 0: uniform sampling

% Settings for a simulated experiment
myopts.freshData = 0; % switch for the program mode
% -----------------------------------------------------------------------
% available values for option freshData:
%    1: draw response from truemodel on each trial (simulated experiment)
%    0: sample a stimulus-response pair from dataset (data re-ordering)
%   -1: use the next stimulus-response pair in dataset (in fixed order)
% -----------------------------------------------------------------------

% Additional option
myopts.talkative = 1; % 0:silent, 1:minimal messages, 2:more messages



%% Run a sequential "experiment" of dataset re-ordering

% === Run a full sequence experiment ===============

% Number of trials in the experiment
mynums.N = 50; % total # trials
mynums.ninit = 10; % # initial trials (not adaptively sampled) 

% Run program (to get the entire sequence at once)
[finalEstimate,trackSeq,settings] = prog_infomax_MNLwL(mydat,mynums,myopts);


% === Unpack output from simulation ================

% Settings for the "experiment" are stored and returned
N = settings.N; % total # trials
ninit = settings.ninit; % # "initial" trials, before adaptive stimulus selection starts
withLapse = settings.withLapse; % whether lapse was included in the model
runTag = settings.runtag; % a human-readable string to remind what we did in this simulation

% "True" parameters (in this case, approximated by the best estimate 
% using all trials in dataset)
paramBest_AllDataFit = paramVec2Struct(settings.trueParamVec,settings.dims);
lapseBest = sum(getLapseProb(paramBest_AllDataFit.u)); % u is auxiliary lapse parameter
fprintf('\nBest estimate using all %d trials:\n',Ngen);
fprintf('--------------------------------------\n');
fprintf('        biases:      b =%4.1f %4.1f %4.1f\n', paramBest_AllDataFit.b);
fprintf('weights for x1:   a[1] =%4.1f %4.1f %4.1f\n', paramBest_AllDataFit.a(1,:));
fprintf('weights for x2:   a[2] =%4.1f %4.1f %4.1f\n', paramBest_AllDataFit.a(2,:));
fprintf('    lapse rate:  lapse =%5.2f\n\n', lapseBest);

% Final estimate after the last trial in sequence
paramEst = finalEstimate.param; % param struct
lapseEst = sum(getLapseProb(paramEst.u)); % u is auxiliary lapse parameter
fprintf('\nEstimated after %d trials:\n',N);
fprintf('--------------------------------------\n');
fprintf('        biases:      b =%4.1f %4.1f %4.1f\n', paramEst.b);
fprintf('weights for x1:   a[1] =%4.1f %4.1f %4.1f\n', paramEst.a(1,:));
fprintf('weights for x2:   a[2] =%4.1f %4.1f %4.1f\n', paramEst.a(2,:));
fprintf('    lapse rate:  lapse =%5.2f\n\n', lapseEst);

probEst = finalEstimate.prob; % estimated choice probabilities
probBest_AllDataFit = finalEstimate.probTrue; % "best" estimate from all data

% % Program stores trial information from the sequence
xdata = trackSeq.x; % stimuli actually presented on each trial
postEnt = trackSeq.entropy; % entropy of the posterior after each trial
MSE = trackSeq.MSE; % mean-square error of the PF inferred after each trial
time_per_trial = trackSeq.time; % computation time for each trial



% === Plot =========================================

% response map color
mymap = [ 0 0.4470 0.7410 ;...
    0.8500 0.3250 0.0980 ;...
    0.9290 0.6940 0.1250 ;...
    0.4660 0.6740 0.1880];
mymap = (1-mymap)*0.5 + mymap;

% --- stimulus space ---

% Determine which response dominates (has the highest probability) 
% given each stimulus value.
[~,mostLikelyResponse] = max(probBest_AllDataFit,[],1); % based on the best (all-data) estimate

xx1D = unique(xx0(:,1)); % Stimulus values on one dimension (for plotting)

clf; subplot(1,2,1)
imagesc(xx1D,xx1D,reshape(mostLikelyResponse,numel(xx1D)*[1 1])')
set(gca,'YDir','normal')
colormap(mymap)
hold on
plot(xx(:,1),xx(:,2),'ko') % stimulus space
plot(xdata(ninit:N,1),xdata(ninit:N,2),...
    'ko','markerfacecolor','k') % selected stimuli
hold off
axis square
xlabel('stim 1')
ylabel('stim 2')
title(char({['Xdim=',num2str(size(xx,2)),', Ydim=',num2str(numel(yy0)),','],...
    'dataset re-ordering'}))

% --- trial sequence ---

subplot(3,2,2)
plot(ninit:N,postEnt(ninit:N),'k.-')
xlim([ninit N])
ylabel('post. entropy (rel.)')
withLapseTagList = {'ignoring lapse','considering lapse'};
title([runTag,', ',withLapseTagList{withLapse+1}])

subplot(3,2,4)
plot(ninit:N,MSE(ninit:N),'k.-')
xlim([ninit N])
ylabel('error (MSE)')

subplot(3,2,6)
plot(ninit:N,time_per_trial(ninit:N),'k.-')
xlim([ninit N])
ylabel('comp. time (s)')
xlabel('trial #')


set(findall(gcf,'-property','fontsize'),'fontsize',14)




