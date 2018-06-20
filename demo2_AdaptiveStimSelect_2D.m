% demo2_AdaptiveStimSelect_2D.m

% Demo script for running a simulated experiment with adaptive (infomax)
% stimulus sampling, with 2-dimensional stimulus + 4-alternative responses.

% Reproduces a single run for Figs 5-6 in Bak & Pillow 2018.

% Copyright (c) 2018 Ji Hyun Bak

%% initialize

clear all; % clear workspace
clc; % clear command window
setpaths; % add toolbox to path


%% Prepare for a simulated experiment

% === Set up stimulus/response spaces ===============

% Set up stimulus space (2D stimulus)
xgrid1D = (-1:0.2:1)';
xx1D = xgrid1D/std(xgrid1D); % scaled to have unit std
xx = combvec(xx1D',xx1D')'; % list of points on 2-dimensional regular grid

% Set up response space (4-alternative)
yvalslist = [0 1 2 3];

% Pack stimulus and response spaces
mybasis = struct('xx',xx,'yy',yvalslist);



% === Set true model ================================

% True model parameters
truebias = [1 0 1]; % bias associated to responses 1,2,3
trueslope = [3 6 3; -3 0 3]; % slopes associated to responses 1,2,3
truelapse = 0.20; % total lapse rate, from 0 to 1 (lambda)

truemodel = struct('b',truebias,'a',trueslope,'lapse',truelapse);

fprintf('\nGenerating true model:\n');
fprintf('--------------------------------------\n');
fprintf('        biases:      b =%4.1f %4.1f %4.1f\n', truemodel.b);
fprintf('weights for x1:   a[1] =%4.1f %4.1f %4.1f\n', truemodel.a(1,:));
fprintf('weights for x2:   a[2] =%4.1f %4.1f %4.1f\n', truemodel.a(2,:));
fprintf('    lapse rate:  lapse =%5.2f\n\n', truemodel.lapse);


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
myopts.truemodel = truemodel; % pass true model to the program
myopts.freshData = 1; % switch for the program mode
% -----------------------------------------------------------------------
% available values for option freshData:
%    1: draw response from truemodel on each trial (simulated experiment)
%    0: sample a stimulus-response pair from dataset (data re-ordering)
%   -1: use the next stimulus-response pair in dataset (in fixed order)
% -----------------------------------------------------------------------

% Additional option
myopts.talkative = 1; % 0:silent, 1:minimal messages, 2:more messages



%% Run simulated experiment


%%%% TODO: a trial-by-trial illustration of how the simulation works %%%%


% === Run a full sequence experiment ===============

% Number of trials in the experiment
mynums.N = 50; % total # trials
mynums.ninit = 10; % # initial trials (not adaptively sampled) 

% Run program (to get the entire sequence at once)
[finalEstimate,trackSeq,settings] = prog_infomax_MNLwL(mybasis,mynums,myopts);


% === Unpack output from simulation ================

% Settings for the "experiment" are stored and returned
N = settings.N; % total # trials
ninit = settings.ninit; % # "initial" trials, before adaptive stimulus selection starts
withLapse = settings.withLapse; % whether lapse was included in the model
runTag = settings.runtag; % a human-readable string to remind what we did in this simulation

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
probTrue = finalEstimate.probTrue; % true probability is also returned along with the final estimate

% Program stores trial information from the sequence
xdata = trackSeq.x; % stimuli actually presented on each trial
postEnt = trackSeq.entropy; % entropy of the posterior after each trial
MSE = trackSeq.MSE; % mean-square error of the PF inferred after each trial
time_per_trial = trackSeq.time; % computation time for each trial


% === Plot =========================================

% Response map color
mymap = [ 0 0.4470 0.7410 ;...
    0.8500 0.3250 0.0980 ;...
    0.9290 0.6940 0.1250 ;...
    0.4660 0.6740 0.1880];
mymap = (1-mymap)*0.5 + mymap;

% --- stimulus space ---

% Determine which response dominates (has the highest probability) 
% given each stimulus value.
[~,mostLikelyResponse] = max(probTrue,[],1); % based on the true choice probability

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
title(char({['Xdim=',num2str(size(xx,2)),...
    ', Ydim=',num2str(numel(yvalslist)),','],...
    ['total lapse ',num2str(truelapse)]}))

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




