function myXmat = getXmat(x,ydim) 
% Constructs the input design matrix, given a list of stimuli
%
% INPUTS
%       x [array]  - each row is a stimulus at which to evaluate probs
%    ydim [number] - degrees of freedom of response (#choice-1)
%
% OUTPUTS
%   myXmat [array] - a block diagonal array of g(x)' 
%                  - in this version, g(x) is identical for all responses
% -----------------------------------------------------------------------

% Copyright 2018 Ji Hyun Bak

g = gfun(x); % each column is one feature vector g
myXmat = kron(eye(ydim),g'); 

end
