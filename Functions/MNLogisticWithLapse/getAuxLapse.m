function u = getAuxLapse(lapseProb) 
% From lapse probabilities p, get auxiliary lapse parameters u
% Reverse function of getLapseProb
%
% INPUTS
%  lapseProb [vector] - single vector, #elements = #choices
%                     - corresponds to lambda*c in Bak & Pillow (2018)
%
% OUTPUTS
%          u [vector] - auxiliary lapse parameter u, matching dimensions
% -----------------------------------------------------------------------

% Copyright 2018 Ji Hyun Bak

% laspe parametrization: p = exp(u)./(1+sum(exp(u))))
u = log(lapseProb./(1-sum(lapseProb))); 

end
