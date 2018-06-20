function myprob = getProbAtPrs_MNLogLapse(theta,dims,xx)
% Choice probability under multinomial logistic model with lapse
%
% INPUTS
%     theta [vector] - full parameter vector [weight; auxiliary lapse prs]
%      dims [struct] - degrees of freedom of variables
%                      .y = of output
%                      .g = of the input feature vector g(x)
%        xx [array]  - each row is a stimulus at which to evaluate probs
%
% OUTPUTS
%     myprob [array] - each row: for each choice index (ydim+1 rows)
%                    - each column: for each stimulus (matches xx #rows)
% -----------------------------------------------------------------------

% Copyright 2018 Ji Hyun Bak

%% unpack input

% -- input check
if(isempty(xx))
    myprob = []; % empty xx option added 5/16/2018 (moved from getTrueModel)
    return;
end

% unpack dimensions
ydim = dims.y;
gdim = dims.g;

% unpack theta
w = theta(1:ydim*gdim); % weight parameter
u = theta(ydim*gdim+1:end); % lapse parameter


%% get choice probability

% -------------------------
% parameter array & linear predictor vector
getWarray = @(argParam,argDims) reshape(argParam(:),argDims.g,argDims.y);
getV = @(argStim,argParam,argDims) vertcat(zeros(1,size(argStim,1)), ...
    getWarray(argParam,argDims)'*gfun(argStim)); % K rows, NX cols
% -------------------------

myprob0 = mysoftmax2(getV(xx,w,dims)); % lapse-free MNLogistic probability

if(isempty(u))
    myprob = myprob0; % keep lapse-free; [K,NX] matrix
else
    % add lapses
    lc = getLapseProb(u(:)); % lapse prob lambda*c; force [K,1] column vector
    lambda = sum(lc); % total lapse rate
    myprob = bsxfun(@plus,(1-lambda)*myprob0,lc); 
end

end
