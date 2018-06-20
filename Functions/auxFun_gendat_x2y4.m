function [xall,yall,probTrue,xx0,yy0] = auxFun_gendat_x2y4(Ngen)
% Generates Ngen stimulus-response pairs using a fixed underlying model,
% with 2D stimulus and 4-alternative responses.
% Intended as an auxiliary function for demo3_reorderingExpt_2D.m
% 
% INPUT:  Ngen:  number of stimulus-response pairs (trials) in dataset
%
% OUTPUTS:
%       xall   [array] : a list of stimulus presented, each row is a trial
%       yall  [vector] : a list of responses, in response categories (integers)
%   probTrue   [array] : the underlying choice probability used to generate responses
%        xx0   [array] : the stimulus space; each row is a stimulus vector
%        yy0  [vector] : the response space; a list of response categories
% ------------------------------------------------------------------------

% Copyright (c) 2018 Ji Hyun Bak

%% generate a fake dataset

%%% set up underlying model

% set up stimulus space (2D stimulus)
xgrid1D = (-1:0.2:1)';
xx1D = xgrid1D/std(xgrid1D); % scaled
getgrid2Dfull = @(grid1D) combvec(grid1D(:)',grid1D(:)')';
xx0 = getgrid2Dfull(xx1D);

% set up response space (4-alternative)
yy0 = (0:3);

% true model parameters
truemodel = struct('b',[1 0 1],'a',[3 6 3; -3 0 3],'lapse',0);

% true model (to draw responses from)
[~,~,probTrue] = getTrueModel_byParamStruct(truemodel,xx0);
drawResp_hidden = @(inds) ...
    mnrnd(1,probTrue(:,inds)')*yy0(:); % draw fresh single y from pTrue


%%% generate a fake dataset

% Ngen = 500; % size of "offline" dataset
igen = randsample(size(xx0,1),Ngen,true); % sample stimuli
xall = xx0(igen,:);
yall = drawResp_hidden(igen); % draw responses


end
