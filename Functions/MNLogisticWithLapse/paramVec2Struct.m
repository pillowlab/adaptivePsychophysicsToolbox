function paramStruct = paramVec2Struct(paramVecInput,dims)
% Re-pack the parameter vector into a struct variable
%
% INPUT:   prmVec, a flattened parameter vector 
%           {b1, a1, b2, a2, ... u0, u1, u2, ...}
%
% OUTPUT: prmStruct, a struct with fields {b,a}; & if applicable {u}
%           .b: bias weights -- a row vector with size [1,ydim]
%           .a: sensitivies  -- array with size [xdim,ydim]
%           .u: lapse params -- vector with length (ydim+1), or #choices
% -----------------------------------------------------------------------

% Copyright 2018 Ji Hyun Bak

ydim = dims.y;
gdim = dims.g;

paramVec = paramVecInput(:); % force vector

% unpack parameter vector
wvec = paramVec(1:gdim*ydim); % weight parameters
u = paramVec(gdim*ydim+1:end); % lapse parameters

% weights
prmat = reshape(wvec,gdim,ydim);
b = prmat(1,:);
a = prmat(2:gdim,:);

% pack into a struct; for now weights only
paramStruct = struct('b',b,'a',a);

% add lapse parameters if applicable
if(~isempty(u))
    if(numel(u)==(ydim+1))
        paramStruct.u = u(:)';
        % paramStruct.lapse = sum(getLapseProb(u)); % total lapse rate
    else
        error('paramVec2Struct: dimension mismatch');
    end
end

end
