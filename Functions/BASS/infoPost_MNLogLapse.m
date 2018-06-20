function [infoCrit,warn] = infoPost_MNLogLapse(xx,dims,wMode,negHcut,acclev)
% calculates the expected mutual information at each stimulus in xx
% from the posterior distribution with mean wMode and hessian -negHcut
%
% INPUTS
%        xx [array]  - each row is a stimulus at which to evaluate probs
%      dims [struct] - degrees of freedom of variables
%                      .y = of output
%                      .g = of the input feature vector g(x)
%             wMode  - parameter value at posterior mode
%           negHcut  - negative Hessian at posterior mode 
%                      possibly cut for weight parameters only
%             acclev - accuracy level for sparse grids
%
% OUTPUTS
%    infoCrit - mutual information, calculated under Laplace approx
%        warn - warning flags for singular or non-PSD matrices
% -----------------------------------------------------------------------

% Copyright 2018 Ji Hyun Bak

%% initialize

warn = 0; % means normal termination (positive values means problems)

% unpack input variables
ydim = dims.y;
gdim = dims.g;
qdim = ydim*gdim;

nx = size(xx,1);

%% obtain partial covariance (with respect to the weights)

% test if negHcut is ill-conditioned, and if so, return null output
condthresh = 1e8;
if(rcond(negHcut)<(1/condthresh))
    % note: nearestSPD doesn't work when not full-rank
    disp('infoPost: singular negHcut');
    infoCrit = NaN(nx,1);
    warn = 9;
    return;
end

% ensure positive definiteness of partial Hessian (5/26/2017)
[~,notSPD] = chol(negHcut);
if(notSPD>0)
    negHcut = nearestSPD_custom(negHcut); 
end


%% calculate mutual information (using Sparse Grids)

getGammaMat = @(argP) diag(argP)-argP*argP'; % used for posterior update

infoCrit = zeros(nx,1);

%%% Quadrature on Sparse Grids (codes from: www.sparse-grids.de)
% 'GQN': Gaussian quadrature with Gaussian weights
% Unode : (totnodes,ydim) matrix
% weight: (totnodes,1) vector
% -- using a slightly customized version of nwspgr.m --
[Unode,weight] = nwspgr_custom('GQN',ydim,acclev); % quadrature points and weights
totnodes = size(Unode,1);

%%% sum over y

ylist = 0:ydim;
IK = eye(numel(ylist)); % unit matrix

Imat = eye(qdim);

%%% iterate over xx

for np = 1:nx 
    
    myXmat = getXmat(xx(np,:),ydim); % input design matrix
    
    %%% reduce the parameter space
    
    % Diagonalization (using SVD)
    [~,~,svd_V] = svd(myXmat);
    Qmat = svd_V';
    
    % Marginalization
    Hmat = Qmat*negHcut*Qmat'; % H = inv(Q*Cov*Q')
    Hkk = Hmat(1:ydim,1:ydim);
    Hqq = Hmat(ydim+1:qdim,ydim+1:qdim);
    Hkq = Hmat(1:ydim,ydim+1:qdim);
    Hstar = Hkk - Hkq*(Hqq\Hkq');
    
    % -- choleskly decomposition
    [Rmat,cholwarn] = cholfix_nearestSPD(Hstar); % for occasional non-SPD cases
    if(cholwarn>=2)
        % means that nearestSPD did not succeed in two attempts
        infoCrit(np) = NaN; % for now just skip / return null result
        warn = max(warn,cholwarn);
        continue;
    end
    
    % Standardization
    myGmat = myXmat*Qmat'; 
    sqdim = min(size(myGmat)); 
    Gk = myGmat(1:sqdim,1:sqdim); % take the square part (a diagonal matrix)
    Lmat = Gk/Rmat;
    
    
    %%% evaluate the integrand
    
    XC = (myXmat/negHcut); % X*Cov
    
    % current V and P
    vMode = myXmat*wMode(:); % linear predictor V (Xw=XQ'Qw)
    myVeff_list = bsxfun(@plus,vMode,Lmat*Unode'); % indexed for [p u]
    myV_list = [zeros(1,totnodes); myVeff_list]; % idx: [pfull u]
    myP_list = mysoftmax2(myV_list);
    
    % next (candidate) V and P
    YminusP_all = bsxfun(@minus,IK,permute(myP_list,[1 3 2])); % idx: [p y u]
    myDelta_all = sum(bsxfun(@times,permute(myXmat,[2 3 4 1]),...
        permute(YminusP_all(2:end,:,:),[4 2 3 1])),4); % idx: [p ylist u]
    myVnexteff_all = myVeff_list + ...
        sum(bsxfun(@times,permute(XC,[1 3 4 2]),permute(myDelta_all,[4 3 2 1])),4); % [p u ylist]
    myPnext_all = mysoftmax2(...
        vertcat(zeros(1,numel(ylist),totnodes),permute(myVnexteff_all,[1 3 2]))); % [pfull ylist u]
    
    % loop over u and y 
    integList = zeros(numel(ylist),totnodes);
    for uind = 1:totnodes 
        for iy = 1:numel(ylist)
            myPeff = myPnext_all(2:end,iy,uind);
            % calculate C[t]\C[t+1] matrix
            myGamma = getGammaMat(myPeff);
            Mmat = myXmat'*myGamma*XC; 
            CCmat = Imat-(Imat+Mmat)\Mmat; % C[t]\C[t+1]
            integList(iy,uind) = -logdet(CCmat); % posterior entropy
        end
    end
    
    % get quadrature 
    finteg = sum(myP_list.*integList,1); % 1*totnodes row vector
    infoCrit(np) = finteg*weight; 
    
end

end
