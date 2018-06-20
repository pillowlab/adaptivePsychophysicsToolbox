function lapseProb = getLapseProb(u) 
% From auxiliary lapse parameters u, get lapse probabilities p
% Reverse function of getAuxLapse
%
% INPUTS
%          u [vector] - auxiliary lapse parameter u (#elements=#choices)
%
% OUTPUTS
%  lapseProb [vector] - single vector, #elements = #choices
%                     - corresponds to lambda*c in Bak & Pillow (2018)
% -----------------------------------------------------------------------

% Copyright 2018 Ji Hyun Bak

% laspe parametrization: p = exp(u)./(1+sum(exp(u))))
lapseProb = mysigmoid(u); 

end
