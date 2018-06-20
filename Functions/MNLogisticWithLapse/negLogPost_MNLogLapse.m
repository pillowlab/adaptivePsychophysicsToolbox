function [negLP,dnegLP,ddnegLP] = negLogPost_MNLogLapse(prs,dat,dims,myPrior)
% Negative log-posterior for multinomial logistic model with lapses
% 
% INPUT
%      prs [vector]  - model parameters flattened, with [weight,lapse]
%                      {b1, a1, b2, a2, ... u0, u1, u2, ...}
%      dat [struct]  - data to compute likelihood for:
%                      .x = stimuli (each row is a stimulus),
%                      .y = output state (one from 0:(ydim-1))
%      dims [struct] - degrees of freedom of variables
%                      .y = of output
%                      .g = of the input feature vector g(x)
%      myPrior       - prior hyperparameters
%
% OUTPUT
%      negLP [1 x 1] - negative log-posterior
%     dnegLP [n x 1] - gradient
%    ddnegLP [n x n] - Hessian (2nd derivative matrix)
% -----------------------------------------------------------------------

% Copyright 2018 Ji Hyun Bak

%% preparation

% unpack dimensions
ydim = dims.y;
gdim = dims.g;
udim = (ydim+1); % dimension of lapse parameter (equal to # choices)
Vdim = (ydim+1); % dimension of linear predictor V (also equal to # choices)

K0 = ydim*gdim; % weight dimension
Kfull = K0+udim; % full parameter dimension, including lapse

% unpack parameter
wgt = prs(1:K0);
if(numel(prs)==Kfull)
    u = prs((K0+1):end);
    withLapse = 1;
elseif(numel(prs)==K0)
    u = getAuxLapse(zeros(1,udim)); % add zero lapse
    withLapse = 0;
else
    error('negLL: prs dimension mismatch'); 
end

% unpack data
x = dat.x;
y = dat.y; % vector of output state indices
M = bsxfun(@eq,(0:ydim),y); % boolean output
nx = size(x,1);


%% gaussian prior 

% prior hyperparameters (passed through input)
w0 = myPrior.wgtmean;
sigma0_w = myPrior.wgtsigma;

u0 = myPrior.lpsInit;
sigma0_u = Inf; % lapse prior is flat within the range constraint

% --- legacy chunk ---
% u0 = myPrior.lpsmean;
% sigma0_u = myPrior.lpssigma;

% construct gaussian prior
if(~withLapse)
    prs0 = w0*ones(K0,1);
    invC0 = diag((1/sigma0_w^2)*ones(K0,1));
else
    prs0 = [w0*ones(K0,1); u0*ones(udim,1)];
    invC0 = diag([(1/sigma0_w^2)*ones(K0,1);...
       (1/sigma0_u^2)*ones(udim,1)]);
end

% set up log prior
logprior = -(1/2)*(prs(:)-prs0(:))'*invC0*(prs(:)-prs0(:));

% also the derivatives of log prior
dlogprior = -invC0*(prs(:)-prs0(:));
ddlogprior = -invC0;


%% log-likelihood with lapse 

% get probabilities
Q = getProbAtPrs_MNLogLapse(wgt,dims,x); % probs without lapse
P = getProbAtPrs_MNLogLapse(prs,dims,x); % using full parameter with laspe

% auxiliary quantities
R = bsxfun(@rdivide,getLapseProb(u(:)),P); % force column auxLapse
T = M'.*(1-R);
S = M'.*R.*(1-R);

% evaluate log-likelihood L
logli = sum(sum(M'.*mylog(P,true))); % use mylog to avoid -Inf
if( isnan(logli) || isinf(logli) )
    warning('negLogPost_MNLogLapse: non-finite logli'); % just to check; should not happen
end

% report negative log-posterior
negLP = -logli -logprior; 

if(nargout==1)
    % do not calculate derivatives if not asked for
    dnegLP = [];
    ddnegLP = [];
    return; 
end


%% derivatives

%%% derivatives of log-likelihood, with respect to V and u

% first derivatives {dL/dV, dL/dg} at each x
dLdV = T - bsxfun(@times,Q,sum(T,1));
dLdu = R.*(M'-P);

% second derivatives {ddL/dVdV, ddL/dudu, ddL/dVdu} at each x
ddLdVdV = zeros(Vdim,Vdim,nx);
ddLdudu = zeros(udim,udim,nx);
ddLdVdu = zeros(Vdim,udim,nx);
for idx = 1:nx
    m = M(idx,:);
    q = Q(:,idx);
    p = P(:,idx);
    s = S(:,idx);
    t = T(:,idx);
    r = R(:,idx);
    % -- dVdV
    ddLdVdV(:,:,idx) = ...
        -sum(t)*(diag(q)-(q*q')) + (diag(s)-(q*s'+s*q')+sum(s)*(q*q'));
    % -- dudu
    auxvec_dudu = s - r.*p + sum(m./p)*(r.*p).^2;
    ddLdudu(:,:,idx) = diag(auxvec_dudu);
    % -- dVdu
    ddLdVdu(:,:,idx) = - diag(s) + (q.^2)*(s./q)'; % (V row, u col)
    
end



%%% derivatives of log-likelihood, with respect to w and u

dlogli_w = zeros(K0,1);
Hlogli_ww = zeros(K0,K0);
Hlogli_wu = zeros(K0,udim);
for idx = 1:nx
    myXmat = getXmat(x(idx,:),ydim); 
    mydw = myXmat'*dLdV(2:end,idx);
    mydwdw = myXmat'*ddLdVdV(2:end,2:end,idx)*myXmat;
    mydwdu = myXmat'*ddLdVdu(2:end,:,idx);
    % sum over x
    dlogli_w = dlogli_w + mydw;
    Hlogli_ww = Hlogli_ww + mydwdw;
    Hlogli_wu = Hlogli_wu + mydwdu;
end
dlogli_u = sum(dLdu,2);
Hlogli_uu = sum(ddLdudu,3);

if(withLapse==1)
    dlogli = [dlogli_w; dlogli_u];
    Hlogli = [Hlogli_ww Hlogli_wu; Hlogli_wu' Hlogli_uu];
else
    dlogli = dlogli_w;
    Hlogli = Hlogli_ww;
end


%%% derivatives of log-posterior

dnegLP = - dlogli -dlogprior;
ddnegLP = -Hlogli -ddlogprior;


end

function B = mylog(A,fixInf)
% optionally replace log(0) by a large negative number, rather than -Inf

B = log(A);

if(fixInf)
    isZero = (A==0);
    %B(isZero) = -realmax;
    B(isZero) = log(realmin); % this is more realistic
end

end
