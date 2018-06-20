function p = mysigmoid(u)
% Sigmoid function p = exp(u)./(1+sum(exp(u))), protected from Inf blowup
%
% INPUT:    u is a numeric vector (either row or column)
% OUTPUT:   p is a vector of matching dimensions
% -----------------------------------------------------------------------

% Copyright 2018 Ji Hyun Bak

% size check
if(min(size(u))>1)
    % for now only support vectors (but can be extend to multidim arrays)
    error('mysigmoid: argument should be a vector');
end

v = u;
v0 = 0; % fixed such that exp(v0)=1

vmax = max(v(:)); % for now, single number
if(vmax==Inf)
    % this should never happen
    error('mysigmoid: bad argument (Inf)')
elseif(vmax==-Inf)
    % use original arguments (important to keep this -Inf filter!)
else
    % vmax is finite
    % shift to ensure that each component is small (big v gives blowup)
    v = v - vmax;
    v0 = v0 - vmax;
end

Z = (exp(v0)+sum(exp(v)));
p = exp(v)./Z;

end
