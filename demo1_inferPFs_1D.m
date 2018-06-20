% demo1_inferPFs_1D.m

% Demo script illustrating inference for 1-dimensional psychometric
% functions from data.

% Reproduces Fig 2 from Bak & Pillow 2018.

% Examines data from three simulated models
% 1) 2-alternative model without omission or lapse 
% 2) 2-alternative model + omission (making 3 response alternatives).
% 3) 2-alternative model + lapse

% Copyright (c) 2018 Ji Hyun Bak

%% initialize

clear all; % clear workspace
clc; % clear command window
setpaths; % add toolbox to path


%% Model 1: standard 2-alternative model without omission or lapse

% === Generate simulated dataset ===============

% Set up true model 
truemodel1 = struct('b',0,'a',2);  % b: bias, a: slope
fprintf('\nGenerating model 1:\n');
fprintf('-------------------\n');
fprintf(' bias:     b=%.1f\n', truemodel1.b);
fprintf('slope:     a=%.1f\n\n', truemodel1.a);

% Set grid of stimulus values 
xx = 2.5*(-1:0.1:1)'; % 1D stimulus grid

% Compute true psychometric function at stimulus values
[~,~,probTrue] = getTrueModel_byParamStruct(truemodel1,xx);

% Batch of stimuli to be presented
Nrep = 20; % number of repeated observations per stimulus
xdata = repmat(xx,Nrep,1); % full set of stimuli presented

% Sample data from true model
yvalslist = [0 1]'; % two alternatives for the response (0 for "Left", 1 for "Right")
yrawdata = mnrnd(1,repmat(probTrue',Nrep,1));  % one-hot representation of response data
ydata = yrawdata*yvalslist; % categorical representation of response data

% Pack data into a struct object
dat1 = struct('x',xdata,'y',ydata,'xx',xx);

% Get mean prob(y==1) conditioned at each stimulus (for plotting later)
meanRespAtX = zeros(size(xx,1),1);
for ix = 1:size(xx,1)
    myy = ydata(xdata==xx(ix)); % all responses at select x
    meanRespAtX(ix) = mean(myy==1);
end


% === Set options for the program ================

% Initialize
mynums = [];
myopts = [];

% Choose inference method through nsamples
mynums.nsamples = 0; % either 0 or a positive integer
% If nsamples > 0, it specifies the # samples in each MCMC chain;
% if nsamples == 0, it means a MAP estimate.

% Set prior hyperpameters
mynums.hyperprs = ... 
    struct('wgtmean',0,'wgtsigma',3,... % gaussian prior for weights
    'lpsLB',log(0.001),'lpsUB',0, ... % range constraints for lapses
    'lpsInit',-5 ... % starting point for lapse parameters
    );

% Set switches for running modes
myopts.allfit = true; % if true, program simply does parameter estimate
myopts.withLapse = false; % if false: inference algorithm is unaware of lapses

% Additional options
myopts.reportMoreValues = true; % report all sampled parameters from MCMC
myopts.talkative = 1; % 0:silent, 1:minimal messages, 2:more messages

% === Run fitting code to infer the PF from data =========

[fitResult,~,~] = prog_infomax_MNLwL(dat1,mynums,myopts);
probEst = fitResult.prob; % estimated choice probability

% Display estimated parameter values
paramEst1 = fitResult.param;
fprintf('\nEstimated model 1:\n');
fprintf('-------------------\n');
fprintf(' bias:     b=%.1f\n', paramEst1.b);
fprintf('slope:     a=%.1f\n\n', paramEst1.a);

% === plot ==========================================

clf; set(gcf,'position',[100 100 900 400])

mygray = 0.8*[1 1 1]; % gray color
msz = 7; % marker size

subplot(1,3,1)
plot(xx,probTrue(yvalslist==1,:),'k--') % true P(y==1)
hold on;
plot(xx,probEst(yvalslist==1,:),'k-') % inferred P(y==1)
plot(xx,probTrue(yvalslist==0,:),'--','color',mygray) % true P(y==0)
plot(xx,meanRespAtX,'ko','markerfacecolor','k','markersize',msz) % empirical P(y==1|x) at each x
hold off
xlim([min(xx) max(xx)]); ylim([0 1]); axis square
xlabel('stimulus x'); ylabel('P(y)')
legend('true PF','inferred PF','location','southeast')
title({'standard model with';'binary response'})

% annotations
text(1.5,0.85,'y=R','fontweight','bold')
text(-1.5,0.85,'y=L','fontweight','bold','color',mygray,'horizontalalignment','right')

% line/font formatting
set(findall(gcf,'-property','fontsize'),'fontsize',14)
set(findall(gca,'-property','linewidth'),'linewidth',2)
set(gca,'box','off','linewidth',1)


%% Model 2: with omission (making 3 response alternatives)

% There are two "normal" response options 
% and an additional "omission" or "no response" response option.
% Omission trials are marked in category -1.

yvalslist_with_omission = [-1 0 1]'; % [Omission, Left, Right]

% === Generate new simulated dataset =============

% Generate data from model with omissions
truemodel2 = struct('b',[-1 -1],'a',[-3 3]); 
% b: biases for the "valid" responses 0 and 1
% a: weights (slopes) "valid" responses 0 and 1

fprintf('\nGenerating model 2:\n');
fprintf('-------------------\n');
fprintf(' bias:     b0=%4.1f\n', truemodel2.b(1));
fprintf('           b1=%4.1f\n', truemodel2.b(2));
fprintf('slope:     a0=%4.1f\n', truemodel2.a(1));
fprintf('           a1=%4.1f\n\n', truemodel2.a(2));

% Compute true psychometric function
[~,~,probTrue] = getTrueModel_byParamStruct(truemodel2,xx);

% Sample data from this true model (with omissions)
xdata = repmat(xx,Nrep,1); % full set of stimuli presented
yrawdata = mnrnd(1,repmat(probTrue',Nrep,1));  % one-hot representation of response data
ydata = yrawdata*yvalslist_with_omission; % categorical representation of response data

% But let's drop all omitted trials (keep only the non-omission/"normal" trials)
% and see what happens
xdata_withoutomission = xdata(ismember(ydata,[0 1]),:);
ydata_withoutomission = ydata(ismember(ydata,[0 1]));

% Pack data into a struct object
dat2 = struct('x',xdata_withoutomission,'y',ydata_withoutomission,'xx',xx);

% Get mean prob(y==1) conditioned at each stimulus (for plotting later)
meanRespAtX_alltrials = zeros(size(xx,1),1);
meanRespAtX_normal_only = zeros(size(xx,1),1);
for ix = 1:size(xx,1)
    myy0 = ydata(xdata==xx(ix));
    myy = ydata_withoutomission(xdata_withoutomission==xx(ix));
    meanRespAtX_alltrials(ix) = mean(myy0==1);
    meanRespAtX_normal_only(ix) = mean(myy==1);
end


% === Run program to infer the PF from data =========

% Run the program, using the same options as before (model 1).
[fitResult,~,~] = prog_infomax_MNLwL(dat2,mynums,myopts);

% Estimated choice probability (for the two non-omission choices only!)
probEst = fitResult.prob; 

% Display estimated parameter values
paramEst2 = fitResult.param;
fprintf('\nEstimated model 2:\n');
fprintf('(with non-omission trials only)\n');
fprintf('-------------------------------\n');
fprintf(' bias:     b1=%4.1f\n', paramEst2.b);
fprintf('slope:     a1=%4.1f\n', paramEst2.a);


% === Plot ==========================================

myred = [0.8 0 0]; % red color

subplot(1,3,2)
plot(xx,probTrue(yvalslist_with_omission==1,:),'k--') % true P(y==1)
hold on
plot(xx,probTrue(yvalslist_with_omission==0,:),'--','color',mygray) % true P(y==0)
plot(xx,probTrue(yvalslist_with_omission==-1,:),'--','color',myred) % true omission probability P(y==-1)
plot(xx,meanRespAtX_alltrials,'ko','markersize',msz) % empirical P(y==1|x), with all trials
plot(xx,meanRespAtX_normal_only,... % empirical P(y==1|x), with non-omission trials only
    'ko','markerfacecolor','k','markersize',msz) 
plot(xx,probEst(yvalslist==1,:),'k-') % inferred P(y==1)
hold off
xlim([min(xx) max(xx)]); ylim([0 1]); axis square
xlabel('stimulus x'); ylabel('P(y)')
title({'with omissions';''})

% annotations
text(1.5,0.85,'y=R','fontweight','bold')
text(-1.5,0.85,'y=L','fontweight','bold','color',mygray,'horizontalalignment','right')
text(0.9,0.25,'omission','fontweight','bold','color',myred)

% line/font formatting
set(findall(gcf,'-property','fontsize'),'fontsize',14)
set(findall(gca,'-property','linewidth'),'linewidth',2)
set(gca,'box','off','linewidth',1)


%% Model 3: 2-alternative model with lapses

% Simulate data with non-zero lapse rate;
% algorithm tries to fit with a lapse-free psychometric function.

% === Generate new simulated dataset =============

% Set up true model 
truemodel3 = struct('b',0,'a',2,'lapse',0.2); 
% b: bias, a:slope, % lapse: total lapse rate

fprintf('\nGenerating model 3:\n');
fprintf('-------------------\n');
fprintf(' bias:     b=%4.1f\n', truemodel3.b);
fprintf('slope:     a=%4.1f\n', truemodel3.a);
fprintf('lapse: lapse=%5.2f\n\n', truemodel3.lapse);

% Compute true psychometric function
[~,~,probTrue] = getTrueModel_byParamStruct(truemodel3,xx);

% Sample data from this true model (with lapses)
xdata = repmat(xx,Nrep,1); % full set of stimuli presented
yrawdata = mnrnd(1,repmat(probTrue',Nrep,1));  % one-hot representation of response data
ydata = yrawdata*yvalslist; % categorical representation of response data

% Pack data into a struct object
dat3 = struct('x',xdata,'y',ydata,'xx',xx);

% Get mean prob(y==1) conditioned at each stimulus (for plotting later)
meanRespAtX = zeros(size(xx,1),1);
for ix = 1:size(xx,1)
    myy0 = ydata(xdata==xx(ix));
    meanRespAtX(ix) = mean(myy0==1);
end


% === Run program to infer the PF from data =========

% Run the program, using the same options as before; 
% in particular, the algorithm is still not aware of lapses
% (myopts.withLapse is set to false)

[fitResult,~,~] = prog_infomax_MNLwL(dat3,mynums,myopts);
probEst = fitResult.prob; % estimated choice probability

% Display estimated parameter values
paramEst3 = fitResult.param;
fprintf('\nEstimated model 3:\n');
fprintf('(using lapse-UNAWARE model)\n');
fprintf('----------------------------\n');
fprintf(' bias:     b=%4.1f\n', paramEst3.b);
fprintf('slope:     a=%4.1f\n\n', paramEst3.a);


% === Plot ==========================================

subplot(1,3,3)
plot(xx,probTrue(yvalslist==1,:),'k--') % true P(y==1)
hold on
plot(xx,probEst(yvalslist==1,:),'k-') % inferred P(y==1)
plot(xx,probTrue(yvalslist==0,:),'--','color',mygray) % true P(y==0)
plot(xx,meanRespAtX,... % empirical P(y==1|x)
    'ko','markerfacecolor','k','markersize',msz) 
hold off
xlim([min(xx) max(xx)]); ylim([0 1]); axis square
xlabel('stimulus x'); ylabel('P(y)')
% legend('true PF','inferred PF','location','southeast')
title({'with lapses';''});

% annotations
text(1.5,0.75,'y=R','fontweight','bold')
text(-1.5,0.75,'y=L','fontweight','bold','color',mygray,'horizontalalignment','right')

% line/font formatting
set(findall(gcf,'-property','fontsize'),'fontsize',14)
set(findall(gca,'-property','linewidth'),'linewidth',2)
set(gca,'box','off','linewidth',1)


% === Now infer with the lapse-aware model ==========

myopts.withLapse = 1; % use the lapse-aware model

[fitResult,~,~] = prog_infomax_MNLwL(dat3,mynums,myopts);
probEst = fitResult.prob; % estimated choice probability

% Display estimated parameter values
paramEst3L = fitResult.param;
lapseEst = sum(getLapseProb(paramEst3L.u)); % u is auxiliary lapse parameter
fprintf('\nEstimated model 3:\n');
fprintf('(using lapse-AWARE model)\n');
fprintf('----------------------------\n');
fprintf(' bias:     b=%4.1f\n', paramEst3L.b);
fprintf('slope:     a=%4.1f\n', paramEst3L.a);
fprintf('lapse: lapse=%5.2f\n\n', lapseEst);

