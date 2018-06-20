function [prsMode,negHess] = ...
    runMAP_rangePrior(neglogpdf,prs_init,lb,ub,showopt)
% Compute MAP estimate, given neglogpost function & range-constraint prior
%
% INPUTS
%   neglogpdf     - function handle for negative log posterior 
%   prs_init      - initial values for the parameters
%   lb,ub         - range constraints for the parameters
%   showopt       - option for display
%
% OUTPUTS
%   prsMode     - parameter at posterior mode
%   negHess     - hessian of the loss function at the mode
%   probAtMode  - choice probability at posterior mode
% -----------------------------------------------------------------------

% Copyright 2018 Ji Hyun Bak

%% unpack input

% display switch for optimization
if(showopt>0)
    mydisplayswitch = 'iter';
else
    mydisplayswitch = 'off';
end

%% run MAP estimate

% loss function
lossfun = @(prs) neglogpdf(prs); % passed directly from input

% run optimization (fmincon)
opts = optimoptions('fmincon','display',mydisplayswitch,...
    'Algorithm','trust-region-reflective',...
    'SpecifyObjectiveGradient',true,'HessianFcn','objective');
[prsMode,~,flag,~,~,~,negHess] = ...
    fmincon(lossfun,prs_init,[],[],[],[],lb,ub,[],opts);
if(flag<0)
    warning('runMAP: something wrong with fmincon.');
end

end
