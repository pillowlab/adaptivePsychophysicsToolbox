function [prmVec,dims] = paramStruct2Vec(prmStructInput)
% Unpack a parameter struct to return a flattened single parameter vector
%
% INPUT: prmStruct, a struct with required fields {b,a}; optional {u,lapse}
%           .b: bias weights -- a row vector with size [1,ydim]
%           .a: sensitivies  -- array with size [xdim,ydim]
%           .u: lapse params -- vector with length (ydim+1), or #choices
%           .lapse: total lapse rate (to be converted to u here)
%
% OUTPUT:   prmVec, a flattened parameter vector 
%           {b1, a1, b2, a2, ... u0, u1, u2, ...}
% -----------------------------------------------------------------------

% Copyright 2018 Ji Hyun Bak

% check for required fields
prmStruct = check_prmStruct_bau(prmStructInput);

% unpack prmStruct
b = prmStruct.b;
a = prmStruct.a;
u = prmStruct.u;

% reshape into a single vector
Warray = [b;a];
Wvec = Warray(:); % the parameter vector (indexed by choice first)
prmVec = [Wvec; u(:)]; % full parameter vector

% set dimensions
ydim = numel(b); % effective number of outputs (e.g. ydim=2 for {0,1,2})
gdim = size(Warray,1); % number of parameters for each choice (input feature dimension)
dims = struct('y',ydim,'g',gdim);

end

% ----------------------------------------------------------------------

function prmStructOutput = check_prmStruct_bau(prmStructInput)
% set up field u (auxiliary lapse parameter) if not passed through input

if(~and(isfield(prmStructInput,'b'),isfield(prmStructInput,'a')))
    error('paramStruct2Vec: incomplete model input (struct)');
end
b = prmStructInput.b;
a = prmStructInput.a;

ydim = numel(b);

% get auxiliary lapse parameter
if(isfield(prmStructInput,'u'))
    u = prmStructInput.u;
elseif(isfield(prmStructInput,'lapse')) % only the lapse rate
    lapseProb = (prmStructInput.lapse)*ones(1,ydim+1)/(ydim+1); % uniform by default
    u = getAuxLapse(lapseProb);
else
    u = getAuxLapse(zeros(1,ydim+1)); % zero lapse if unspecified
end

% pack into a struct variable
prmStructOutput = struct('a',a,'b',b,'u',u);

end
