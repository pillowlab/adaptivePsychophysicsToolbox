function [myLmat,warn] = cholfix_nearestSPD(myCmat,varargin)
% Make two attempts at approximate Cholesky decomposition 
% by using nearestSPD (originally by John D'Errico, customized by JHB)
%
% INPUTS:   myCmat: a square matrix, expected to be close to SPD
%           cholmode (optional 2nd input): either 'upper' or 'lower'
% OUTPUTS:  myLmat: a lower-triangular matrix that comes out as a result of
%                   cholesky decomposition
%           warn: flag for error cases (0 means normal)
% ------------------------------------------------------------------------

% Copyright 2018 Ji Hyun Bak

% second input argumeent for chol
if(nargin>1)
    cholmode = varargin{1};
else
    cholmode = 'upper'; % default in chol
end

% try twice
warn = 0; % 0 means normal 
[myLmat,notSPD] = chol(myCmat,cholmode);
if(notSPD)
    warn = 1;
    myCmat1 = nearestSPD_custom(myCmat); % tweak until nearest SPD is found
    [myLmat,stillNotSPD] = chol(myCmat1,cholmode); % try again
    if(stillNotSPD)
        warn = 2;
        myLmat = []; % didn't work
    end
end

end
