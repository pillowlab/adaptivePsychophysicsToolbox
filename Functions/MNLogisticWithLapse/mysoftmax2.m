function p = mysoftmax2(v)
% Softmax function p = exp(v)./sum(exp(v)), protected from blow-up;
% can also deal with arrays.
%
% INPUT:    v is a numeric array
%             - 1st dim: runs over choice index (marked as kdim)
%             - the other dims: for different stimulus points, etc.
% OUTPUT:   p is an array of matching dimensions
% -----------------------------------------------------------------------

% Copyright 2018 Ji Hyun Bak


kdim = 1; % dimension along which "choices" are listed

vmax = max(v,[],kdim);
if(any(vmax==Inf))
    error('mysoftmax: bad argument (Inf)')
end
v = bsxfun(@minus,v,vmax); % shifted

Z = sum(exp(v),kdim);
p = bsxfun(@rdivide,exp(v),Z); 

end
