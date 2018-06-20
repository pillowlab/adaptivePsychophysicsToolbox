function [prmVec,dims,probTrue] = getTrueModel_byParamStruct(prmStruct,xx)
% Construct "true" PF on the stimulus space, given model parameter struct
%
% INPUTS: 
%   prmStruct - a struct with required fields {b,a}; optional {u,lapse}
%           .b: bias weights -- a row vector with size [1,ydim]
%           .a: sensitivies  -- array with size [xdim,ydim]
%           .u: lapse params -- vector with length (ydim+1), or #choices
%           .lapse: total lapse rate (to be converted to u here)
%   xx [Nx * Dx array] - the stimulus set,
%           where each row is one stimulus vector.
%           Nx (number of rows) = number of stimuli in grid
%           Dx (number of columns) = dimensionality of the stimulus space
% 
% OUTPUTS:
%    prmVec [vector] - a flattened parameter vector 
%                      {b1, a1, b2, a2, ... u0, u1, u2, ...}
%      dims [struct] - degrees of freedom of variables
%                      .y = of output (choice/response)
%                      .g = of the input feature vector g(x)
%  probTrue [Ny * Nx array] - the "true" choice probability map,
%           where each column is a (normalized) choice probability vector
%           at a given stimulus.
%           Ny (number of rows) = number of choice alternatives
%           Nx (number of columns) = number of stimuli in grid
% -----------------------------------------------------------------------

% Copyright 2018 Ji Hyun Bak

[prmVec,dims] = paramStruct2Vec(prmStruct);
probTrue = getProbAtPrs_MNLogLapse(prmVec,dims,xx);

end
