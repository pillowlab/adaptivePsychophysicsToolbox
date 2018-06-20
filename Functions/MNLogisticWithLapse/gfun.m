function g = gfun(x) 
% Construct the input feature [carrier] vectors, given a list of stimuli
% 
% INPUT:    x (each *row* is one stimulus vector)
% OUTPUT:   g (each *column* is one feature vector)
% ------------------------------------------------------------------------

% Copyright 2018 Ji Hyun Bak

g = [ones(size(x,1),1) x]'; % added row of 1's accounts for the bias

end
